import einops
import torch
import torch.nn.functional as F
from torch import nn

from utils.loss_utils import apply_reduction


class MocoLoss(nn.Module):
    def __init__(self, temperature=0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, projected, predicted, queue, queue_ptr, reduction="mean"):
        assert torch.is_tensor(projected) and projected.ndim == 3
        assert torch.is_tensor(predicted) and predicted.ndim == 3
        assert projected.grad_fn is None
        total_loss = 0.

        # preprocess
        projected = F.normalize(projected, dim=-1)
        predicted = F.normalize(predicted, dim=-1)

        # create labels
        batch_size = len(predicted)
        labels = torch.zeros(batch_size, device=predicted.device, dtype=torch.long)
        # queue has to be detached because enqueueing is in-place
        queue_features = queue.clone()

        # global loss
        global_loss = 0.
        global_alignment = 0.
        num_global_views = projected.size(1)
        assert predicted.size(1) >= num_global_views
        for i in range(num_global_views):
            for ii in range(num_global_views):
                if i == ii:
                    continue
                loss, alignment = self._loss(
                    projected=projected[:, i],
                    predicted=predicted[:, ii],
                    queue=queue_features,
                    labels=labels,
                    reduction=reduction,
                )
                total_loss = total_loss + loss
                global_loss = global_loss + loss
                global_alignment = global_alignment + alignment
        num_global_terms = num_global_views * (num_global_views - 1)
        if num_global_terms > 0:
            global_loss = global_loss / num_global_terms
            global_alignment = global_alignment / num_global_terms

        # local loss
        local_loss = 0.
        local_alignment = 0.
        num_local_views = predicted.size(1) - num_global_views
        for i in range(num_global_views):
            for ii in range(num_local_views):
                loss, alignment = self._loss(
                    projected=projected[:, i],
                    predicted=predicted[:, ii],
                    queue=queue_features,
                    labels=labels,
                    reduction=reduction,
                )
                total_loss = total_loss + loss
                local_loss = local_loss + loss
                local_alignment = local_alignment + alignment

        # update queue (this could also be done in a "after_update" method to avoid cloning it every iteration)
        if self.training:
            to_enqueue = einops.rearrange(projected, "batch_size num_views dim -> (batch_size num_views) dim")
            ptr_from = int(queue_ptr)
            ptr_to = ptr_from + len(to_enqueue)
            queue[ptr_from:ptr_to] = to_enqueue
            queue_ptr[0] = ptr_to % len(queue)

        num_local_terms = num_local_views * num_global_views
        if num_local_terms > 0:
            local_loss = local_loss / num_local_terms
            local_alignment = local_alignment / num_local_terms

        # normalize by loss terms (num_terms == 0 if called with single view for evaluation)
        num_terms = num_global_terms + num_local_terms
        if num_terms > 0:
            total_loss = total_loss / num_terms

        # compose infos
        losses = dict(total=total_loss, global_loss=global_loss)
        infos = dict(global_alignment=global_alignment)
        if num_local_terms > 0:
            losses["local_loss"] = local_loss
            infos["local_alignment"] = local_alignment
        return losses, infos

    def _loss(self, projected, predicted, queue, labels, reduction):
        logits_pos = torch.einsum("nc,nc->n", predicted, projected)
        logits_neg = torch.einsum("nc,qc->nq", predicted, queue)
        logits = torch.concat([logits_pos.unsqueeze(1), logits_neg], dim=1) / self.temperature
        loss = F.cross_entropy(logits, labels, reduction=reduction)
        # alignment
        with torch.no_grad():
            alignment = torch.gather(logits.softmax(dim=-1), index=labels.unsqueeze(1), dim=1)
        # apply reduction
        return loss, apply_reduction(alignment, reduction=reduction)
