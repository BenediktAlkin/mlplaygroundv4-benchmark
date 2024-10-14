import torch
from torch import nn

from .base.initializer_base import InitializerBase
from kappamodules.init import init_norms_as_identity


class VitBlocksToIdentityInitializer(InitializerBase):
    def __init__(self, start_index, **kwargs):
        super().__init__(**kwargs)
        self.start_index = start_index

    def init_weights(self, model):
        start_index = self.start_index
        if self.start_index < 0:
            start_index += len(model.blocks)
        # replace attention projection layer and last MLP layer to 0s
        for i in range(start_index, len(model.blocks)):
            block = model.blocks[i]
            nn.init.zeros_(block.attn.proj.weight)
            nn.init.zeros_(block.attn.proj.bias)
            nn.init.zeros_(block.mlp.fc2.weight)
            nn.init.zeros_(block.mlp.fc2.bias)
