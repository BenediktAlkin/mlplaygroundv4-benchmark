import numpy as np
import torch
import torch.nn.functional as F
from kappamodules.init import init_norm_as_noaffine, init_xavier_uniform_zero_bias, init_truncnormal_zero_bias
from kappamodules.utils.mode_to_ctor import mode_to_norm_ctor
from kappaschedules import object_to_schedule
from torch import nn

from models.base.single_model_base import SingleModelBase
from models.poolings import pooling_from_kwargs
from modules.ssl.nnclr_queue import NnclrQueue
from utils.factory import create
from utils.model_utils import update_ema, copy_params
from utils.schedule_utils import get_value_or_default
from modules.ssl.massl_queue import MemoryBank

class MasslHead(SingleModelBase):
    def __init__(
            self,
            output_dim,
            queue_size,
            pooling,
            norm_mode="batchnorm",
            proj_hidden_dim=None,
            proj_hidden_layers=1,
            target_factor=None,
            target_factor_schedule=None,
            copy_ema_on_start=False,
            init_weights="xavier_uniform",
            **kwargs,
    ):
        kwargs.pop("output_shape", None)
        super().__init__(output_shape=(output_dim,), **kwargs)
        self.pooling = create(pooling, pooling_from_kwargs, static_ctx=self.static_ctx)
        input_shape = self.input_shape if self.pooling is None else self.pooling.get_output_shape(self.input_shape)
        self.input_dim = np.prod(input_shape)
        self.proj_hidden_dim = proj_hidden_dim
        self.proj_hidden_layers = proj_hidden_layers
        self.output_dim = output_dim
        self.norm_ctor, self.requires_bias = mode_to_norm_ctor(norm_mode)
        self.init_weights = init_weights
        self.act_ctor = nn.GELU
        self.projector = self.create_projector()

        # queue
        self.queue = MemoryBank(
            K=queue_size,
            out_dim=self.output_dim,
        )

        # EMA
        self.copy_ema_on_start = copy_ema_on_start
        self.target_factor = target_factor
        self.target_factor_schedule = object_to_schedule(
            target_factor_schedule,
            batch_size=self.update_counter.effective_batch_size if self.update_counter is not None else None,
            updates_per_epoch=self.update_counter.updates_per_epoch if self.update_counter is not None else None,
            max_value=target_factor,
        )
        if self.target_factor is not None:
            self.momentum_projector = self.create_projector()
            for param in self.momentum_projector.parameters():
                param.requires_grad = False
            # create second set of poolings (required for ExtractorPooling)
            self.momentum_pooling = create(pooling, pooling_from_kwargs, static_ctx=self.static_ctx)
        else:
            self.momentum_projector = None
            self.momentum_pooling = None

        # make sure to not overwrite EMA update
        assert type(self).after_update_step == MasslHead.after_update_step

    @property
    def is_batch_size_dependent(self):
        return True

    def create_projector(self):
        # first layer
        first_layer = nn.Sequential(
            nn.Linear(self.input_dim, self.proj_hidden_dim, bias=self.requires_bias),
            self.norm_ctor(self.proj_hidden_dim),
            self.act_ctor(),
        )
        # hidden layers
        hidden_layers = [
            nn.Sequential(
                nn.Linear(self.proj_hidden_dim, self.proj_hidden_dim, bias=self.requires_bias),
                self.norm_ctor(self.proj_hidden_dim),
                self.act_ctor(),
            )
            for _ in range(self.proj_hidden_layers)
        ]
        # last layer
        last_layer = nn.Sequential(
            nn.Linear(self.proj_hidden_dim, self.output_dim, bias=self.requires_bias),
            self.norm_ctor(self.output_dim),
        )
        return nn.Sequential(first_layer, *hidden_layers, last_layer)

    def load_state_dict(self, state_dict, strict: bool = True):
        if self.momentum_projector is None:
            return super().load_state_dict(state_dict=state_dict, strict=strict)
        # initialize momentum_projector with weights from projector (if no momentum_projector weights are found)
        momentum_projector_keys = [key for key in state_dict.keys() if key.startswith("momentum_projector.")]
        if len(momentum_projector_keys) == 0:
            self.logger.info(f"no momentum_projector found -> initialize with projector from state_dict")
            projector_keys = [key for key in list(state_dict.keys()) if key.startswith("projector.")]
            for projector_keys in projector_keys:
                momentum_projector_key = f"momentum_projector.{projector_keys[len('projector.'):]}"
                if self.copy_ema_on_start:
                    src_key = projector_keys
                else:
                    raise NotImplementedError
                state_dict[momentum_projector_key] = state_dict[src_key].clone()
        return super().load_state_dict(state_dict=state_dict, strict=strict)

    def model_specific_initialization(self):
        self.apply(init_norm_as_noaffine)
        if self.init_weights == "xavier_uniform":
            self.apply(init_xavier_uniform_zero_bias)
        elif self.init_weights == "truncnormal":
            self.apply(init_truncnormal_zero_bias)
        else:
            raise NotImplementedError

    def _after_initializers(self):
        if self.momentum_projector is not None:
            if self.copy_ema_on_start:
                self.logger.info(f"initializing {type(self).__name__}.target_projector with parameters from projector")
                copy_params(self.projector, self.momentum_projector)
            else:
                self.logger.info(f"initializing {type(self).__name__}.target_projector randomly")

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
        # forward student
        projected_student = self.projector(x)
        projected_student = F.normalize(projected_student, dim=-1, eps=1e-6)
        # forward teacher
        if self.momentum_projector is not None:
            if momentum_x is None:
                # no encoder ema -> use normal x
                momentum_x = x.detach()[:batch_size * num_teacher_views]
            with torch.no_grad():
                projected_teacher = self.momentum_projector(momentum_x)
                projected_teacher = F.normalize(projected_teacher, dim=-1, eps=1e-6)
        else:
            # without momentum_projector the teacher views of the projector are used
            if batch_size is not None and num_teacher_views is not None:
                projected_teacher = projected_student.detach()[:batch_size * num_teacher_views]
            else:
                raise NotImplementedError
        # forward queue
        projected_student = self.queue(projected_student)
        projected_teacher = self.queue(projected_teacher, update=True)
        return dict(
            student_output=projected_student,
            teacher_output=projected_teacher,
        )

    def after_update_step(self):
        if self.momentum_projector is None:
            return
        target_factor = get_value_or_default(
            default=self.target_factor,
            schedule=self.target_factor_schedule,
            update_counter=self.update_counter,
        )
        # MoCoV3 tracks batchnorm stats from the ema model instead of copying it from the source model
        update_ema(self.projector, self.momentum_projector, target_factor, copy_buffers=False)
