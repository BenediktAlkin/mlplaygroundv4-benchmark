import einops
import torch
import torch.nn.functional as F
from torch import nn

from losses import basic_loss_fn_from_kwargs
from utils.factory import create
from .basic.mse_loss import MSELoss


class MsmimLoss(nn.Module):
    def __init__(self, loss_fn=None, eps=1e-6):
        super().__init__()
        self.loss_fn = create(loss_fn, basic_loss_fn_from_kwargs) or MSELoss()
        self.eps = eps

    @staticmethod
    def _gram_matrix(x):
        _, _, seqlen, dim = x.shape
        xt = x
        x = einops.rearrange(x, "batch_size num_scales seqlen dim -> batch_size num_scales dim seqlen")
        # (batch_size num_scales seqlen dim) @ (batch_size num_scales dim seqlen) -> (batch_size num_scales dim dim)
        gram = (x @ xt) / (seqlen * dim)
        return gram

    def _style_loss(self, prediction, target):
        prediction_gram = self._gram_matrix(prediction)
        target_gram = self._gram_matrix(target)
        # style_loss.shape == (batch_size num_scales dim dim)
        style_loss = self.loss_fn(prediction_gram, target_gram, reduction="none")
        # (batch_size num_scales dim dim) -> (batch_size num_scales)
        style_loss = style_loss.mean(dim=[2, 3])
        return style_loss

    def _msmim_loss(self, prediction, target, mask):
        # unreduced loss (batch_size, num_layers, seqlen, dim)
        loss = self.loss_fn(prediction, target, reduction="none")
        # mean loss per token (batch_size, num_layers, seqlen, dim) -> (num_layers, batch_size, seqlen)
        loss = loss.mean(dim=-1)
        # average over tokens (num_layers, batch_size, seqlen) -> (batch_size, num_layers)
        if mask is None:
            # consider all tokens
            loss = loss.mean(dim=-1)
        else:
            # only consider masked tokens
            loss = (loss * mask.unsqueeze(1)).sum(dim=-1)
            loss = loss / mask.sum(dim=-1, keepdim=True)
        return loss

    def forward(
            self,
            prediction,
            target,
            mask,
            calculate_style_loss=False,
            calculate_aux_loss=False,
            reduction="mean",
    ):
        # convert to tensors (batch_size, num_layers, seqlen, dim)
        if not torch.is_tensor(prediction):
            prediction = torch.stack(prediction, dim=1)
        if not torch.is_tensor(target):
            target = torch.stack(target, dim=1)
        assert prediction.shape == target.shape, f"{tuple(prediction.shape)} != {tuple(target.shape)}"

        # apply non-affine layernorm
        prediction = F.layer_norm(prediction, normalized_shape=(prediction.size(-1),), eps=self.eps)
        target = F.layer_norm(target, normalized_shape=(target.size(-1),), eps=self.eps)

        # split into aux_tokens and patch_tokens
        num_aux_tokens = prediction.size(2) - mask.size(1)
        prediction_patches = prediction[:, :, num_aux_tokens:]
        target_patches = target[:, :, num_aux_tokens:]
        prediction_aux = prediction[:, :, :num_aux_tokens]
        target_aux = target[:, :, :num_aux_tokens]

        # style losses
        if calculate_style_loss:
            if calculate_aux_loss:
                aux_style_loss = self._style_loss(prediction=prediction_aux, target=target_aux)
            else:
                aux_style_loss = None
            patches_style_loss = self._style_loss(prediction=prediction_patches, target=target_patches)
        else:
            aux_style_loss = None
            patches_style_loss = None

        # feature regression losses
        patches_msmim_loss = self._msmim_loss(prediction=prediction_patches, target=target_patches, mask=mask)
        aux_msmim_loss = self._msmim_loss(prediction=prediction_aux, target=target_aux, mask=None)


        # apply reduction
        if reduction == "mean":
            patches_msmim_loss = patches_msmim_loss.mean(dim=0)
            if calculate_aux_loss:
                aux_msmim_loss = aux_msmim_loss.mean(dim=0)
            if calculate_style_loss:
                if calculate_aux_loss:
                    aux_style_loss = aux_style_loss.mean(dim=0)
                patches_style_loss = patches_style_loss.mean(dim=0)
        elif reduction is None or reduction == "none":
            pass
        else:
            raise NotImplementedError
        return dict(
            aux_style_loss=aux_style_loss,
            patches_style_loss=patches_style_loss,
            patches_msmim_loss=patches_msmim_loss,
            aux_msmim_loss=aux_msmim_loss,
        )
