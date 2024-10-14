import torch
from torch import nn

from losses import basic_loss_fn_from_kwargs
from utils.factory import create
from utils.loss_utils import apply_reduction
from kappamodules.functional.patchify import patchify_as_1d
from .basic.mse_loss import MSELoss


class PercmaeLoss(nn.Module):
    def __init__(self, loss_fn=None):
        super().__init__()
        self.loss_fn = create(loss_fn, basic_loss_fn_from_kwargs) or MSELoss()

    def forward(self, prediction, target, reduction="mean"):
        # unreduced loss
        loss = self.loss_fn(prediction, target, reduction="none")
        # mean loss per sample
        loss = loss.flatten(start_dim=1).mean(dim=1)
        # apply reduction
        loss = apply_reduction(loss, reduction=reduction)
        return loss
