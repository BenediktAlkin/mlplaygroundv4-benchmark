from copy import deepcopy
from functools import partial

import torch
from torch import nn
from kappaschedules import object_to_schedule
from models.mugs.mugs_group_head import MugsGroupHead
from models.mugs.mugs_group_head_og import MugsGroupHeadOg
from models.mugs.mugs_instance_head_og import MugsInstanceHeadOg
from models.mugs.mugs_localgroup_head import MugsLocalgroupHead
from models.mugs.mugs_localgroup_head_og import MugsLocalgroupHeadOg
from kappamodules.layers.drop_path import DropPath

from initializers import initializer_from_kwargs
from models import model_from_kwargs, prepare_momentum_kwargs
from models.base.composite_model_base import CompositeModelBase
from models.extractors.vit_block_extractor import VitBlockExtractor
from models.poolings.class_token import ClassToken
from utils.factory import create, create_collection
from utils.model_utils import update_ema, copy_params
from utils.schedule_utils import get_value_or_default


class MugsOgModel(CompositeModelBase):
    def __init__(
            self,
            encoder,
            heads,
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
        self.class_token = ClassToken(static_ctx=self.static_ctx)

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
                input_shape=(self.encoder.output_shape[-1],),
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

        # local-group head uses pre-norm features
        self.prelastnorm_extractor = VitBlockExtractor(block_indices=[-1], use_next_norm=False)
        self.prelastnorm_extractor.register_hooks(self.encoder)
        if self.momentum_encoder is not None:
            self.momentum_prelastnorm_extractor = VitBlockExtractor(block_indices=[-1], use_next_norm=False)
            self.momentum_prelastnorm_extractor.register_hooks(self.momentum_encoder)

    @property
    def submodels(self):
        submodels = dict(encoder=self.encoder, **{f"heads.{key}": value for key, value in self.heads.items()})
        if self.momentum_encoder is not None:
            submodels["momentum_encoder"] = self.momentum_encoder
        return submodels

    # noinspection PyMethodOverriding
    def forward(self, x, batch_size, is_weak_aug=None):
        if self.training:
            # weak/strong augmentation split not implemented yet
            assert len(x[0]) == batch_size * 4
            # first half of global views are teacher views
            teacher_x, student_global_x = x[0].chunk(2)
            student_x = [student_global_x] + x[1:]
        else:
            # only single view forward pass supported
            assert len(x) == 1 and len(x[0]) == batch_size
            teacher_x = x[0]
            student_x = x

        # forward student encoder
        encoder_outputs = []
        prelastnorm_outputs = []
        class_tokens = []
        for i, xx in enumerate(student_x):
            encoder_output = self.encoder(xx)["main"]
            encoder_outputs.append(encoder_output)
            prelastnorm_outputs.append(self.prelastnorm_extractor.extract())
            class_tokens.append(self.class_token(encoder_output))
        class_tokens = torch.concat(class_tokens)

        # forward student heads
        head_outputs = {}
        predicted = {}
        for name, head in self.heads.items():
            if isinstance(head, MugsInstanceHeadOg):
                head_output = head(class_tokens)
                head_outputs[name] = head_output
                predicted[name] = head_output[1]
            elif isinstance(head, (MugsLocalgroupHead, MugsLocalgroupHeadOg)):
                relation_outputs = torch.concat([
                    head.forward_relation(
                        prelastnorm_output,
                        batch_size=batch_size,
                        is_weak_aug=is_weak_aug if i == 0 else None,
                    )
                    for i, prelastnorm_output in enumerate(prelastnorm_outputs)
                ])
                head_output = head(relation_outputs)
                head_outputs[name] = head_output
                predicted[name] = head_output[1]
            elif isinstance(head, (MugsGroupHead, MugsGroupHeadOg)):
                head_output = head(class_tokens)
                head_outputs[name] = head_output
                predicted[name] = head_output
            else:
                raise NotImplementedError

        if self.momentum_encoder is not None:
            # momentum encoder forward (only propagate global views)
            with torch.no_grad():
                momentum_encoder_output = self.momentum_encoder(teacher_x)["main"]
                momentum_prelastnorm_output = self.momentum_prelastnorm_extractor.extract()
                momentum_class_token = self.class_token(momentum_encoder_output)
                projected = {}
                for name, head in self.heads.items():
                    if isinstance(head, MugsInstanceHeadOg):
                        projected[name] = head.forward_momentum(momentum_class_token)
                    elif isinstance(head, (MugsLocalgroupHead, MugsLocalgroupHeadOg)):
                        momentum_relation_output = head.forward_momentum_relation(
                            momentum_prelastnorm_output,
                            batch_size=batch_size,
                            is_weak_aug=True,
                        )
                        projected[name] = head.forward_momentum(momentum_relation_output)
                    elif isinstance(head, (MugsGroupHead, MugsGroupHeadOg)):
                        projected[name] = head.forward_momentum(momentum_class_token)
                    else:
                        raise NotImplementedError
        else:
            # no momentum encoder
            projected = {}
            for name, head in self.heads.items():
                if isinstance(head, MugsInstanceHeadOg):
                    projected[name] = head_outputs[name][0].detach()[:len(teacher_x)]
                elif isinstance(head, (MugsLocalgroupHead, MugsLocalgroupHeadOg)):
                    projected[name] = head_outputs[name][0].detach()[:len(teacher_x)]
                elif isinstance(head, (MugsGroupHead, MugsGroupHead)):
                    projected[name] = head_outputs[name].detach()[:len(teacher_x)]
                else:
                    raise NotImplementedError
        return projected, predicted

    def model_specific_initialization(self):
        if self.momentum_encoder is not None:
            self.logger.info(f"initializing momentum_encoder with parameters from encoder")
            copy_params(self.encoder, self.momentum_encoder)
        super().model_specific_initialization()

    def after_update_step(self):
        if self.momentum_encoder is not None:
            target_factor = get_value_or_default(
                default=self.target_factor,
                schedule=self.target_factor_schedule,
                update_counter=self.update_counter,
            )
            # MoCoV3 tracks batchnorm stats from the ema model instead of copying it from the source model
            update_ema(self.encoder, self.momentum_encoder, target_factor, copy_buffers=False)

