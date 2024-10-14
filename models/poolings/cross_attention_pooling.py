from functools import partial

import torch
from kappamodules.pooling import CrossAttentionPooling as KMCrossAttentionPooling
from torch import nn

from .base.pooling_base import PoolingBase


class CrossAttentionPooling(PoolingBase):
    def __init__(self, num_query_tokens, num_heads=None, dim=None, concat_query=False, **kwargs):
        super().__init__(**kwargs)
        self.concat_query = concat_query
        if dim is None:
            dim = self.static_ctx["dim"]
        if num_heads is None:
            num_heads = self.static_ctx["num_heads"]
        self.pooling = KMCrossAttentionPooling(
            dim=dim,
            num_query_tokens=num_query_tokens,
            num_heads=num_heads,
            norm_ctor=partial(nn.LayerNorm, eps=1e-6),
        )

    def get_output_shape(self, input_shape):
        _, dim = input_shape
        if self.concat_query:
            dim *= 2
        return dim,

    def forward(self, all_tokens, *_, **__):
        if self.pooling.num_query_tokens == 0:
            assert self.static_ctx["num_aux_tokens"] == 1
            x = self.pooling(all_tokens[:, 1:], query_tokens=all_tokens[:, :1])
            if self.concat_query:
                x = torch.concat([all_tokens[:, :1], x], dim=2)
        else:
            assert not self.concat_query
            x = self.pooling(all_tokens)
        assert x.size(1) == 1
        return x[:, 0]

    def __str__(self):
        return type(self).__name__
