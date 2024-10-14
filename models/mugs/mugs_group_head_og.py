import numpy as np
import torch
from kappamodules.init import init_norm_as_noaffine, init_truncnormal_zero_bias
from kappamodules.layers import Normalize, WeightNormLinear
from kappaschedules import object_to_schedule
from torch import nn

from models.base.single_model_base import SingleModelBase
from utils.model_utils import update_ema
from utils.schedule_utils import get_value_or_default


class MugsGroupHeadOg(SingleModelBase):
    def __init__(
            self,
            proj_hidden_dim,
            bottleneck_dim,
            output_dim,
            fixed_g,
            target_factor=None,
            target_factor_schedule=None,
            **kwargs,
    ):
        kwargs.pop("output_shape", None)
        super().__init__(output_shape=(output_dim,), **kwargs)
        self.input_dim = np.prod(self.input_shape)
        self.proj_hidden_dim = proj_hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.output_dim = output_dim
        self.fixed_g = fixed_g

        self.act_ctor = nn.GELU
        self.projector = self.create_projector()

        # EMA
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
        else:
            self.momentum_projector = None

        self.register_buffer("center", torch.zeros(output_dim))

        # make sure to not overwrite EMA update
        assert type(self).after_update_step == MugsGroupHeadOg.after_update_step

    @property
    def is_batch_size_dependent(self):
        return True

    def create_projector(self):
        # first layer
        first_layer = nn.Sequential(
            nn.Linear(self.input_dim, self.proj_hidden_dim),
            self.act_ctor(),
            nn.Identity(),
        )
        # hidden layers
        hidden_layers = [
            nn.Sequential(
                nn.Linear(self.proj_hidden_dim, self.proj_hidden_dim),
                self.act_ctor(),
            )
        ]
        # bottleneck
        bottleneck = nn.Linear(self.proj_hidden_dim, self.bottleneck_dim)
        # last layer
        last_layer = nn.Sequential(
            Normalize(),
            WeightNormLinear(self.bottleneck_dim, self.output_dim, fixed_g=self.fixed_g),
        )
        return nn.Sequential(first_layer, *hidden_layers, bottleneck, last_layer)

    def model_specific_initialization(self):
        self.apply(init_norm_as_noaffine)
        self.apply(init_truncnormal_zero_bias)

    def forward(self, x):
        projected = self.projector(x)
        return projected

    def forward_momentum(self, x):
        assert self.momentum_projector is not None
        x = self.momentum_projector(x)
        return x

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
