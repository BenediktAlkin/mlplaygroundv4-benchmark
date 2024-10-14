import torch
from torch import nn

from losses import loss_fn_from_kwargs
from utils.factory import create


class PerceptualLoss(nn.Module):
    def __init__(self, loss_fn=None):
        super().__init__()
        self.loss_fn = create(loss_fn, loss_fn_from_kwargs) or nn.L1Loss()

    def forward(self, pred, target, reduction="mean"):
        if torch.is_tensor(pred):
            pred = [pred]
        if torch.is_tensor(target):
            target = [target]
        assert len(pred) == len(target)

        # perceptual loss (individual features should be similar)
        perc_losses = {}
        for i in range(len(pred)):
            loss = self.loss_fn(pred[i], target[i], reduction=reduction)
            if reduction == "none":
                loss = loss.flatten(start_dim=1).mean(dim=1)
            perc_losses[f"perceptual{i}"] = loss

        if reduction == "mean":
            # average over all losses
            perc_loss = torch.stack(list(perc_losses.values())).mean()
        elif reduction == "none":
            # average per sample
            perc_loss = torch.stack(list(perc_losses.values())).mean(dim=0)
        else:
            raise NotImplementedError
        return perc_loss, perc_losses
