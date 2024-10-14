from kappamodules.layers import WeightNormLinear
from torch import nn

from .base.freezer_base import FreezerBase


class WeightNormLinearFreezer(FreezerBase):
    def __init__(self, schedule=None, set_to_none=True, **kwargs):
        if schedule is None:
            set_to_none = False
        else:
            assert set_to_none
        super().__init__(schedule=schedule, set_to_none=set_to_none, **kwargs)

    def __str__(self):
        return type(self).__name__

    def _update_state(self, model, requires_grad):
        layer = model.projector[-1]
        if isinstance(layer, nn.Sequential):
            layer = layer[-1]
        assert isinstance(layer, WeightNormLinear)
        for p in layer.parameters():
            p.requires_grad = requires_grad

    def _set_to_none(self, model):
        layer = model.projector[-1]
        if isinstance(layer, nn.Sequential):
            layer = layer[-1]
        assert isinstance(layer, WeightNormLinear)
        for p in layer.parameters():
            p.grad = None