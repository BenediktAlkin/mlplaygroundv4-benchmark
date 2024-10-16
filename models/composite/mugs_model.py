from models.ssl.mugs_head import MugsHead
from collections import defaultdict
from copy import deepcopy
from functools import partial

import torch
from kappamodules.layers.drop_path import DropPath
from kappaschedules import object_to_schedule
from torch import nn

from initializers import initializer_from_kwargs
from models import model_from_kwargs, prepare_momentum_kwargs
from models.base.composite_model_base import CompositeModelBase
from models.poolings.base.handle_extractor_pooling import handle_extractor_pooling
from utils.factory import create, create_collection
from utils.model_utils import update_ema, copy_params
from utils.schedule_utils import get_value_or_default


class MugsModel(CompositeModelBase):
    def __init__(
            self,
            encoder,
            heads,
            copy_ema_on_start=False,
            target_factor=None,
            target_factor_schedule=None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder = create(
            encoder,
            model_from_kwargs,
            input_shape=self.input_shape,
            update_counter=self.update_counter,
            path_provider=self.path_provider,
            dynamic_ctx=self.dynamic_ctx,
            static_ctx=self.static_ctx,
        )
        assert self.encoder.output_shape is not None

        self.copy_ema_on_start = copy_ema_on_start
        self.target_factor = target_factor
        # currently only increasing schedules from target_factor -> 1 are supported as this is the common case
        # one could also support decreasing schedules by not passing the target_factor and require the full
        # schedule (start_value/end_value + max_value) to be specified
        self.target_factor_schedule = object_to_schedule(
            target_factor_schedule,
            batch_size=self.update_counter.effective_batch_size if self.update_counter is not None else None,
            updates_per_epoch=self.update_counter.updates_per_epoch if self.update_counter is not None else None,
            start_value=target_factor,
        )
        # propagate only when not None -> allow projector only ema
        # propagate raw objects (i.e. dict) because otherwise ctor_kwargs can contain a schedule object
        propagate_target_factor = {}
        if target_factor is not None:
            propagate_target_factor["target_factor"] = target_factor
        if target_factor_schedule is not None:
            propagate_target_factor["target_factor_schedule"] = target_factor_schedule
        self.heads = nn.ModuleDict(
            create_collection(
                heads,
                model_from_kwargs,
                input_shape=self.encoder.output_shape,
                update_counter=self.update_counter,
                path_provider=self.path_provider,
                dynamic_ctx=self.dynamic_ctx,
                static_ctx=self.static_ctx,
                data_container=self.data_container,
                **propagate_target_factor,
            ),
        )

        # initialize encoder EMA
        if self.target_factor is not None:
            assert isinstance(encoder, (dict, partial))
            momentum_encoder = prepare_momentum_kwargs(encoder)
            if isinstance(encoder, dict) and len(momentum_encoder) == 0:
                # initialize momentum_encoder via checkpoint_kwargs of encoder
                assert "initializers" in encoder and encoder["initializers"][0].get("use_checkpoint_kwargs", False)
                initializer_kwargs = deepcopy(encoder["initializers"][0])
                initializer_kwargs.pop("use_checkpoint_kwargs")
                initializer = initializer_from_kwargs(**initializer_kwargs, path_provider=self.path_provider)
                momentum_encoder = initializer.get_model_kwargs()
            self.momentum_encoder = create(
                momentum_encoder,
                model_from_kwargs,
                input_shape=self.input_shape,
                update_counter=self.update_counter,
                path_provider=self.path_provider,
                dynamic_ctx=self.dynamic_ctx,
                static_ctx=self.static_ctx,
                is_frozen=True,
                allow_frozen_train_mode=True,
            )
            # disable drop_path in momentum_encoder: momentum_encoder is kept in train mode to
            # track batchnorm stats (following MoCoV3) -> drop_path would be applied in forward pass
            assert self.momentum_encoder.is_frozen and self.momentum_encoder.training
            self.logger.info(f"disabling DropPath for momentum_encoder")
            for m in self.momentum_encoder.modules():
                if isinstance(m, DropPath):
                    m.drop_prob = 0.
        else:
            self.momentum_encoder = None

        # register pooling hooks (required for ExtractorPooling)
        for head in self.heads.values():
            head.pooling.register_hooks(self.encoder)
            if self.momentum_encoder:
                head.momentum_pooling.register_hooks(self.momentum_encoder)

    @property
    def submodels(self):
        submodels = dict(encoder=self.encoder, **{f"heads.{key}": value for key, value in self.heads.items()})
        if self.momentum_encoder is not None:
            submodels["momentum_encoder"] = self.momentum_encoder
        return submodels

    # noinspection PyMethodOverriding
    def forward(self, x, batch_size, is_weak_aug):
        if self.training:
            # weak/strong augmentation split not implemented yet
            if len(x[0]) == batch_size * 4:
                # first half of global views are teacher views
                teacher_x, student_global_x = x[0].chunk(2)
                student_x = [student_global_x] + x[1:]
            elif len(x[0] == batch_size * 2):
                # normal byolaug
                assert is_weak_aug is None
                is_weak_aug = True
                teacher_x = x[0]
                student_x = x
            else:
                raise NotImplementedError
        else:
            # only single view forward pass supported
            assert len(x) == 1 and len(x[0]) == batch_size
            teacher_x = x[0]
            student_x = x

        # forward student encoder
        poolings = [head.pooling for head in self.heads.values()]
        with handle_extractor_pooling(poolings):
            encoder_outputs = defaultdict(list)
            for i, xx in enumerate(student_x):
                # encoder forward
                encoder_output = self.encoder(xx)["main"]
                # pool
                for head in self.heads.values():
                    # only add if it wasn't already added (multiple heads can have the same pooling)
                    if len(encoder_outputs[head.pooling]) == i:
                        encoder_outputs[head.pooling].append(head.pooling(encoder_output))
        # concat outputs
        encoder_outputs = {pooling: torch.concat(outputs) for pooling, outputs in encoder_outputs.items()}

        # forward student heads
        head_outputs = {}
        for name, head in self.heads.items():
            if isinstance(head, MugsHead):
                head_outputs[name] = head(
                    encoder_outputs[head.pooling],
                    batch_size=batch_size,
                    is_weak_aug=is_weak_aug,
                    apply_pooling=False,
                )
            else:
                head_outputs[name] = head(encoder_outputs[head.pooling], apply_pooling=False)
        # unpack predicted (head_outputs is projected/predicted tuple)
        predicted = {name: head_outputs[1] for name, head_outputs in head_outputs.items()}

        if self.momentum_encoder is not None:
            # momentum encoder forward (only propagate global views)
            momentum_poolings = [head.momentum_pooling for head in self.heads.values()]
            with handle_extractor_pooling(momentum_poolings):
                with torch.no_grad():
                    momentum_encoder_output = self.momentum_encoder(teacher_x)["main"]
                pooled = {
                    head.pooling: head.momentum_pooling(momentum_encoder_output)
                    for head in self.heads.values()
                }
            # momentum head forward
            projected = {}
            for name, head in self.heads.items():
                if isinstance(head, MugsHead):
                    projected[name] = head.forward_momentum(
                        encoder_outputs[head.pooling],
                        batch_size=batch_size,
                        is_weak_aug=True,
                        apply_pooling=False,
                    )
                else:
                    projected[name] = head.forward_momentum(pooled[head.momentum_pooling], apply_pooling=False)
        else:
            # no momentum encoder
            projected = {
                name: head_outputs[0].detach()[:len(teacher_x)]
                for name, head_outputs in head_outputs.items()
            }
        return projected, predicted

    def _after_initializers(self):
        if self.momentum_encoder is not None:
            if self.copy_ema_on_start:
                self.logger.info(f"initializing momentum_encoder with parameters from encoder")
                copy_params(self.encoder, self.momentum_encoder)
            else:
                self.logger.info(f"initializing momentum_encoder randomly")

    def after_update_step(self):
        if self.momentum_encoder is not None:
            target_factor = get_value_or_default(
                default=self.target_factor,
                schedule=self.target_factor_schedule,
                update_counter=self.update_counter,
            )
            # MoCoV3 tracks batchnorm stats from the ema model instead of copying it from the source model
            update_ema(self.encoder, self.momentum_encoder, target_factor, copy_buffers=False)

