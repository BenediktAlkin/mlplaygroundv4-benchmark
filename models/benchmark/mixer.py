from einops.layers.torch import Rearrange
from kappamodules.init import init_xavier_uniform_zero_bias
from kappamodules.vit import VitMlp
from torch import nn

from models.base.single_model_base import SingleModelBase


class Mixer(SingleModelBase):
    def __init__(self, dim, depth, **kwargs):
        super().__init__(**kwargs)
        assert len(self.input_shape) == 2
        seqlen, input_dim = self.input_shape
        self.proj = nn.Linear(input_dim, dim)

        self.blocks = nn.Sequential(
            *[
                nn.Sequential(
                    VitMlp(dim, hidden_dim=dim * 4),
                    Rearrange("batch_size seqlen dim -> batch_size dim seqlen"),
                    VitMlp(seqlen, hidden_dim=seqlen * 4),
                    Rearrange("batch_size dim seqlen -> batch_size seqlen dim"),
                ) for _ in range(depth)
            ]
        )
        assert len(self.output_shape) == 1
        output_dim = self.output_shape[0]
        self.pred = nn.Linear(dim, output_dim)

    def model_specific_initialization(self):
        init_xavier_uniform_zero_bias(self.proj)
        init_xavier_uniform_zero_bias(self.pred)

    def forward(self, x):
        x = self.proj(x)
        x = self.blocks(x)
        x = x.mean(dim=1)
        x = self.pred(x)
        return dict(main=x)
