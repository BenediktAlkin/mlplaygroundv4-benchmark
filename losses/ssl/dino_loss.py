import einops
import torch
import torch.nn.functional as F
from kappaschedules import object_to_schedule
from torch import nn

from distributed.gather import all_reduce_mean_grad
from utils.loss_utils import apply_reduction
from utils.schedule_utils import get_value_or_default


class DinoLoss(nn.Module):
    def __init__(
            self,
            teacher_temperature=0.07,
            teacher_temperature_schedule=None,
            student_temperature=0.1,
            update_counter=None,
    ):
        super().__init__()
        self.teacher_temperature = teacher_temperature
        if teacher_temperature_schedule is not None:
            assert update_counter is not None
            self.teacher_temperature_schedule = object_to_schedule(
                teacher_temperature_schedule,
                batch_size=update_counter.effective_batch_size,
                updates_per_epoch=update_counter.updates_per_epoch,
                max_value=teacher_temperature,
            )
        else:
            self.teacher_temperature_schedule = None
        self.student_temperature = student_temperature
        self.update_counter = update_counter

    def forward(self, projected, predicted, center, reduction="mean"):
        assert torch.is_tensor(projected) and projected.ndim == 3
        assert torch.is_tensor(predicted) and predicted.ndim == 3
        assert projected.grad_fn is None
        total_loss = 0.

        # preprocess
        teacher_temperature = get_value_or_default(
            default=self.teacher_temperature,
            schedule=self.teacher_temperature_schedule,
            update_counter=self.update_counter,
            training=self.training,
        )
        projected = F.softmax((projected - center[None, None, :]) / teacher_temperature, dim=-1)
        predicted = F.log_softmax(predicted / self.student_temperature, dim=-1)

        # global loss
        global_loss = 0.
        num_global_views = projected.size(1)
        assert predicted.size(1) >= num_global_views
        for i in range(num_global_views):
            for ii in range(num_global_views):
                if i == ii:
                    continue
                loss = self._loss(
                    projected=projected[:, i],
                    predicted=predicted[:, ii],
                    reduction=reduction,
                )
                total_loss = total_loss + loss
                global_loss = global_loss + loss
        num_global_terms = num_global_views * (num_global_views - 1)
        if num_global_terms > 0:
            global_loss = global_loss / num_global_terms

        # local loss
        local_loss = 0.
        num_local_views = predicted.size(1) - num_global_views
        for i in range(num_global_views):
            for ii in range(num_local_views):
                loss = self._loss(
                    projected=projected[:, i],
                    predicted=predicted[:, ii],
                    reduction=reduction,
                )
                total_loss = total_loss + loss
                local_loss = local_loss + loss
        num_local_terms = num_local_views * num_global_views
        if num_local_terms > 0:
            local_loss = local_loss / num_local_terms

        # normalize by loss terms (num_terms == 0 if called with single view for evaluation)
        num_terms = num_global_terms + num_local_terms
        if num_terms > 0:
            total_loss = total_loss / num_terms

        # compose infos
        losses = dict(total=total_loss, global_loss=global_loss)
        if num_local_terms > 0:
            losses["local_loss"] = local_loss
        return losses, {}

    @staticmethod
    def _loss(projected, predicted, reduction):
        loss = torch.sum(-projected * predicted, dim=-1)
        loss = apply_reduction(loss, reduction=reduction)
        return loss
