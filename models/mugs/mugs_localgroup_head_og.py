from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from kappamodules.init import init_norm_as_noaffine, init_xavier_uniform_zero_bias
from kappaschedules import object_to_schedule
from modules.ssl.mugs_queue import MugsQueue
from kappamodules.vit import VitBlock
from torch import nn

from models.base.single_model_base import SingleModelBase
from models.poolings.class_token import ClassToken
from models.poolings.mean_patch import MeanPatch
from utils.model_utils import update_ema, copy_params
from utils.schedule_utils import get_value_or_default


class MugsLocalgroupHeadOg(SingleModelBase):
    def __init__(
            self,
            relation_depth,
            relation_num_attn_heads,
            relation_dim,
            queue_size,
            queue_topk,
            output_dim,
            proj_hidden_dim,
            pred_hidden_dim,
            loss_queue_size,
            target_factor=None,
            target_factor_schedule=None,
            eps=1e-6,
            **kwargs,
    ):
        kwargs.pop("output_shape", None)
        super().__init__(output_shape=(output_dim,), **kwargs)
        self.input_dim = np.prod(self.input_shape)
        self.proj_hidden_dim = proj_hidden_dim
        self.output_dim = output_dim
        self.pred_hidden_dim = pred_hidden_dim

        norm_layer = partial(nn.LayerNorm, eps=eps)
        self.relation_blocks = nn.ModuleList([
            VitBlock(
                dim=relation_dim,
                num_heads=relation_num_attn_heads,
                norm_ctor=norm_layer,
            )
            for _ in range(relation_depth)
        ])
        self.relation_norm = norm_layer(relation_dim)
        self.queue = MugsQueue(size=queue_size, dim=relation_dim, topk=queue_topk)

        self.act_ctor = nn.GELU
        self.projector = self.create_projector()
        self.predictor = self.create_predictor()

        # EMA
        self.target_factor = target_factor
        self.target_factor_schedule = object_to_schedule(
            target_factor_schedule,
            batch_size=self.update_counter.effective_batch_size if self.update_counter is not None else None,
            updates_per_epoch=self.update_counter.updates_per_epoch if self.update_counter is not None else None,
            max_value=target_factor,
        )
        if self.target_factor is not None:
            # momentum queue
            self.momentum_queue = MugsQueue(size=queue_size, dim=relation_dim, topk=queue_topk)
            # momentum relation blocks
            self.momentum_relation_blocks = nn.ModuleList([
                VitBlock(
                    dim=relation_dim,
                    num_heads=relation_num_attn_heads,
                    norm_ctor=norm_layer,
                )
                for _ in range(relation_depth)
            ])
            for param in self.momentum_relation_blocks.parameters():
                param.requires_grad = False
            # momentum relation norm
            self.momentum_relation_norm = norm_layer(relation_dim)
            for param in self.momentum_relation_norm.parameters():
                param.requires_grad = False
            # momentum projector blocks
            self.momentum_projector = self.create_projector()
            for param in self.momentum_projector.parameters():
                param.requires_grad = False
        else:
            self.momentum_queue = None
            self.momentum_relation_blocks = None
            self.momentum_relation_norm = None
            self.momentum_projector = None

        # init poolings
        self.class_token = ClassToken(static_ctx=self.static_ctx)
        self.mean_patch = MeanPatch(static_ctx=self.static_ctx)

        # register queue for loss
        self.register_buffer("loss_queue", torch.empty(loss_queue_size, output_dim))
        self.register_buffer("loss_queue_ptr", torch.zeros(1, dtype=torch.long))

        # make sure to not overwrite EMA update
        assert type(self).after_update_step == MugsLocalgroupHeadOg.after_update_step

    @property
    def is_batch_size_dependent(self):
        return True

    def create_projector(self):
        # first layer
        first_layer = nn.Sequential(
            nn.Linear(self.input_dim, self.proj_hidden_dim, bias=False),
            self.act_ctor(),
            nn.Identity(),
        )
        # hidden layers
        hidden_layers = [
            nn.Sequential(
                nn.Linear(self.proj_hidden_dim, self.proj_hidden_dim, bias=False),
                self.act_ctor(),
            )
        ]
        # last layer
        last_layer = nn.Linear(self.proj_hidden_dim, self.output_dim, bias=False)
        return nn.Sequential(first_layer, *hidden_layers, last_layer)

    def create_predictor(self):
        # first layer
        first_layer = nn.Sequential(
            nn.Linear(self.output_dim, self.pred_hidden_dim, bias=False),
            self.act_ctor(),
        )
        # hidden layers
        hidden_layers = [
            nn.Sequential(
                nn.Linear(self.pred_hidden_dim, self.pred_hidden_dim, bias=False),
                self.act_ctor(),
            )
        ]
        # last layer
        last_layer = nn.Linear(self.pred_hidden_dim, self.output_dim, bias=False)
        return nn.Sequential(first_layer, *hidden_layers, last_layer)

    def model_specific_initialization(self):
        self.apply(init_norm_as_noaffine)
        self.apply(init_xavier_uniform_zero_bias)
        self.loss_queue.copy_(F.normalize(torch.randn_like(self.loss_queue), dim=1))
        self.queue.reset_parameters()

    def forward_relation(self, x, batch_size=None, is_weak_aug=None):
        avg = self.mean_patch(x)
        neighbors = self.queue(avg, batch_size=batch_size, is_weak_aug=is_weak_aug)
        x = torch.concat([avg.unsqueeze(1), neighbors], dim=1)
        for block in self.relation_blocks:
            x = block(x)
        return self.relation_norm(x[:, 0])

    def forward(self, x):
        projected = self.projector(x)
        predicted = self.predictor(projected)
        return projected, predicted

    def forward_momentum_relation(self, x, batch_size=None, is_weak_aug=None):
        avg = self.mean_patch(x)
        neighbors = self.momentum_queue(avg, batch_size=batch_size, is_weak_aug=is_weak_aug)
        x = torch.concat([avg.unsqueeze(1), neighbors], dim=1)
        for block in self.momentum_relation_blocks:
            x = block(x)
        return self.momentum_relation_norm(x[:, 0])

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
        update_ema(self.relation_blocks, self.relation_blocks, target_factor, copy_buffers=False)
        update_ema(self.relation_norm, self.relation_norm, target_factor, copy_buffers=False)
        update_ema(self.projector, self.momentum_projector, target_factor, copy_buffers=False)
