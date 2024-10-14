from kappamodules.layers import WeightNormLinear
from torch import nn

from .base.freezer_base import FreezerBase


class PoolingFreezer(FreezerBase):
    def __str__(self):
        return type(self).__name__

    def _update_state(self, model, requires_grad):
        for p in model.pooling.parameters():
            p.requires_grad = requires_grad

    def _set_to_none(self, model):
        for p in model.pooling.parameters():
            p.grad = None