from kappamodules.init import init_xavier_uniform_zero_bias
from torch import nn

from models.base.single_model_base import SingleModelBase


class Maxcomm(SingleModelBase):
    def __init__(self, dim, depth, **kwargs):
        super().__init__(**kwargs)
        assert len(self.input_shape) == 1
        input_dim = self.input_shape[0]
        self.proj = nn.Linear(input_dim, dim)

        self.blocks = nn.Sequential(
            *[
                nn.BatchNorm1d(dim, affine=False)
                for _ in range(depth)
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
        x = self.pred(x)
        return dict(main=x)
