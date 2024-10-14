import torch
from torch import nn

from losses import loss_fn_from_kwargs
from utils.factory import create
from utils.functional import gram_matrix


class PerceptualStyleLoss(nn.Module):
    def __init__(self, loss_fn=None):
        super().__init__()
        self.loss_fn = create(loss_fn, loss_fn_from_kwargs) or nn.L1Loss()

    def forward(self, pred, target, reduction="mean"):
        if torch.is_tensor(pred):
            pred = [pred]
        if torch.is_tensor(target):
            target = [target]
        assert len(pred) == len(target)

        # style loss (features statistics should be similar)
        style_losses = {}
        for i in range(len(pred)):
            loss = self.loss_fn(gram_matrix(pred[i]), gram_matrix(target[i]), reduction=reduction)
            if reduction == "none":
                loss = loss.flatten(start_dim=1).mean(dim=1)
            style_losses[f"style{i}"] = loss

        if reduction == "mean":
            # average over all losses
            style_loss = torch.stack(list(style_losses.values())).mean()
        elif reduction == "none":
            # average per sample
            style_loss = torch.stack(list(style_losses.values())).mean(dim=0)
        else:
            raise NotImplementedError

        return style_loss, style_losses
