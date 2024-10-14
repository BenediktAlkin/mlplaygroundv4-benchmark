from copy import deepcopy
from functools import partial

import torch
from kappamodules.layers.drop_path import DropPath
from kappaschedules import object_to_schedule

from initializers import initializer_from_kwargs
from models import model_from_kwargs, prepare_momentum_kwargs
from models.base.composite_model_base import CompositeModelBase
from models.poolings import pooling_from_kwargs
from models.poolings.identity import Identity
from models.ssl.nnclr_head import NnclrHead
from utils.factory import create
from utils.model_utils import update_ema, copy_params
from utils.schedule_utils import get_value_or_default


class MugsHead(CompositeModelBase):
    def __init__(
            self,
            relation_model,
            head,
            pooling,
            copy_ema_on_start=False,
            target_factor=None,
            target_factor_schedule=None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        # relation model
        # TODO not sure why pooling is here...should be in relation model
        self.pooling = create(pooling, pooling_from_kwargs, static_ctx=self.static_ctx)
        self.relation_model = create(
            relation_model,
            model_from_kwargs,
            pooling=self.pooling,
            input_shape=self.input_shape,
            update_counter=self.update_counter,
            path_provider=self.path_provider,
            dynamic_ctx=self.dynamic_ctx,
            static_ctx=self.static_ctx,
        )
        assert self.relation_model.output_shape is not None
        # head
        self.head = create(
            head,
            model_from_kwargs,
            pooling=Identity,
            copy_ema_on_start=copy_ema_on_start,
            target_factor=target_factor,
            target_factor_schedule=target_factor_schedule,
            input_shape=self.relation_model.output_shape,
            update_counter=self.update_counter,
            path_provider=self.path_provider,
            dynamic_ctx=self.dynamic_ctx,
            static_ctx=self.static_ctx,
        )

        # momentum relation model
        self.copy_ema_on_start = copy_ema_on_start
        self.target_factor = target_factor
        self.target_factor_schedule = object_to_schedule(
            target_factor_schedule,
            batch_size=self.update_counter.effective_batch_size if self.update_counter is not None else None,
            updates_per_epoch=self.update_counter.updates_per_epoch if self.update_counter is not None else None,
            start_value=target_factor,
        )
        if self.target_factor is not None:
            assert isinstance(relation_model, (dict, partial))
            momentum_relation_model = prepare_momentum_kwargs(relation_model)
            if isinstance(relation_model, dict) and len(momentum_relation_model) == 0:
                # initialize momentum_relation_model via checkpoint_kwargs of relation_model
                assert "initializers" in relation_model and relation_model["initializers"][0].get(
                    "use_checkpoint_kwargs", False)
                initializer_kwargs = deepcopy(relation_model["initializers"][0])
                initializer_kwargs.pop("use_checkpoint_kwargs")
                initializer = initializer_from_kwargs(**initializer_kwargs, path_provider=self.path_provider)
                momentum_relation_model = initializer.get_model_kwargs()
            self.momentum_pooling = create(pooling, pooling_from_kwargs, static_ctx=self.static_ctx)
            self.momentum_relation_model = create(
                momentum_relation_model,
                model_from_kwargs,
                pooling=self.momentum_pooling,
                input_shape=self.input_shape,
                update_counter=self.update_counter,
                path_provider=self.path_provider,
                dynamic_ctx=self.dynamic_ctx,
                static_ctx=self.static_ctx,
                is_frozen=True,
                allow_frozen_train_mode=True,
            )
            assert self.momentum_relation_model.is_frozen and self.momentum_relation_model.training
            self.logger.info(f"disabling DropPath for momentum_relation_model")
            for m in self.momentum_relation_model.modules():
                if isinstance(m, DropPath):
                    m.drop_prob = 0.
        else:
            self.momentum_relation_model = None

        # make sure to not overwrite EMA update
        assert type(self).after_update_step == MugsHead.after_update_step

    @property
    def submodels(self):
        submodels = dict(relation_model=self.relation_model, head=self.head)
        if self.momentum_relation_model is not None:
            submodels["momentum_relation_model"] = self.momentum_relation_model
        return submodels

    @property
    def queue(self):
        assert isinstance(self.head, NnclrHead)
        return self.head.queue

    def forward(
            self,
            x,
            momentum_x=None,
            idx=None,
            cls=None,
            confidence=None,
            batch_size=None,
            num_teacher_views=None,
            apply_pooling=True,
            is_weak_aug=None,
    ):
        # pool
        if apply_pooling:
            assert x.ndim == 3
            x = self.pooling(x)
            if momentum_x is not None:
                momentum_x = self.momentum_pooling(momentum_x)
        else:
            assert x.ndim == 2
            if momentum_x is not None:
                assert momentum_x.ndim == 2 and momentum_x.grad_fn is None
        # forward relation model
        x = self.relation_model(x, apply_pooling=apply_pooling, batch_size=batch_size, is_weak_aug=is_weak_aug)
        # forward ema relation model
        if self.momentum_relation_model is not None:
            if momentum_x is None:
                # no encoder ema -> use normal x
                momentum_x = x.detach()
            with torch.no_grad():
                momentum_x = self.momentum_relation_model(
                    momentum_x,
                    apply_pooling=apply_pooling,
                    batch_size=batch_size,
                    is_weak_aug=is_weak_aug,
                )
        else:
            assert momentum_x is None
        # forward head
        return self.head(
            x,
            momentum_x=momentum_x,
            idx=idx,
            cls=cls,
            confidence=confidence,
            batch_size=batch_size,
            num_teacher_views=num_teacher_views,
            apply_pooling=False,
        )

    def _after_initializers(self):
        if self.momentum_relation_model is not None:
            if self.copy_ema_on_start:
                self.logger.info(f"initializing momentum_relation_model with parameters from relation_model")
                copy_params(self.relation_model, self.momentum_relation_model)
            else:
                self.logger.info(f"initializing momentum_relation_model randomly")

    def after_update_step(self):
        if self.momentum_relation_model is not None:
            target_factor = get_value_or_default(
                default=self.target_factor,
                schedule=self.target_factor_schedule,
                update_counter=self.update_counter,
            )
            # MoCoV3 tracks batchnorm stats from the ema model instead of copying it from the source model
            update_ema(self.relation_model, self.momentum_relation_model, target_factor, copy_buffers=False)
