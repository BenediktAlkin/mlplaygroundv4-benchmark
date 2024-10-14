from models.base.single_model_base import SingleModelBase
from utils.factory import create
from models.poolings import pooling_from_kwargs
from kappamodules.vit import VitBlock
from kappamodules.init import init_norm_as_noaffine
from modules.ssl.mugs_queue import MugsQueue
from torch import nn
import torch

class MugsRelationModel(SingleModelBase):
    def __init__(
            self,
            pooling,
            queue_size,
            topk,
            dim,
            num_heads,
            depth,
            eps=1e-6,
            init_weights="xavier_uniform",
            **kwargs,
    ):
        super().__init__(**kwargs)
        # pooling
        self.pooling = create(pooling, pooling_from_kwargs, static_ctx=self.static_ctx)
        self.output_shape = self.pooling.get_output_shape(self.input_shape)

        # properties
        self.dim = dim
        self.num_heads = num_heads
        self.depth = depth

        # layers
        self.queue = MugsQueue(size=queue_size, dim=dim, topk=topk)
        self.blocks = nn.Sequential(*[
            VitBlock(
                dim=dim,
                num_heads=num_heads,
                eps=eps,
                init_weights=init_weights,
            )
            for _ in range(depth)
        ])
        # original code uses the norm of the encoder -> makes no sense
        self.norm = nn.LayerNorm(dim, eps=eps)

    def model_specific_initialization(self):
        init_norm_as_noaffine(self.norm)

    def forward(self, x, apply_pooling=True, batch_size=None, is_weak_aug=None):
        if apply_pooling:
            x = self.pooling(x)
        neighbors = self.queue(x, batch_size=batch_size, is_weak_aug=is_weak_aug)
        x = torch.concat([x.unsqueeze(1), neighbors], dim=1)
        x = self.blocks(x)
        x = x[:, 0]
        x = self.norm(x)
        return x
