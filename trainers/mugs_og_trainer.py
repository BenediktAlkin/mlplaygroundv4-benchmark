import einops
import torch
from torch import nn
from kappadata.utils.multi_crop_utils import concat_same_shape_inputs
from kappadata.wrappers import ModeWrapper
from losses.ssl.moco_loss import MocoLoss
from models.mugs.mugs_group_head import MugsGroupHead
from models.mugs.mugs_group_head_og import MugsGroupHeadOg

from callbacks.online_callbacks.update_output_callback import UpdateOutputCallback
from losses import loss_fn_from_kwargs
from utils.factory import create_collection
from .base.sgd_trainer import SgdTrainer


class MugsOgTrainer(SgdTrainer):
    def __init__(self, loss_functions, **kwargs):
        super().__init__(**kwargs)
        self.loss_functions = nn.ModuleDict(
            create_collection(
                loss_functions,
                loss_fn_from_kwargs,
                update_counter=self.update_counter,
            )
        )

    def get_trainer_callbacks(self, model=None):
        return [
            UpdateOutputCallback(
                patterns=["target_factor/"],
                reduce="last",
                verbose=False,
                every_n_updates=self.track_every_n_updates,
                every_n_samples=self.track_every_n_samples,
                **self.get_default_callback_kwargs(),
            ),
        ]

    @property
    def output_shape(self):
        return None

    @property
    def dataset_mode(self):
        return "index x"

    def get_trainer_model(self, model):
        return self.Model(model=model, trainer=self)

    class Model(nn.Module):
        def __init__(self, model, trainer):
            super().__init__()
            self.model = model
            self.trainer = trainer

        def forward(self, batch, reduction="mean"):
            # prepare data
            batch, ctx = batch
            x = ModeWrapper.get_item(mode=self.trainer.dataset_mode, item="x", batch=batch)
            if isinstance(x, list):
                x = [xx.to(self.model.device, non_blocking=True) for xx in x]
            else:
                x = [x.to(self.model.device, non_blocking=True)]
            x, batch_size = concat_same_shape_inputs(x)
            if "is_weak_global_aug" in ctx:
                is_weak_aug = ctx["is_weak_global_aug"].to(self.model.device, non_blocking=True)
            else:
                is_weak_aug = torch.ones(batch_size, dtype=torch.bool, device=x[0].device)
            # mugs uses different augs for teacher and student -> first half of the global views are teacher views
            # rest are student views -> num_global_views % 2 == 0
            if self.training:
                assert len(x[0]) // batch_size % 2 == 0

            # model forward pass
            all_projected, all_predicted = self.model(x, batch_size=batch_size, is_weak_aug=is_weak_aug)

            # iterate over heads
            total_loss = 0
            losses = {}
            infos = {}
            for head_name in self.trainer.loss_functions.keys():
                head = self.model.heads[head_name]
                projected = einops.rearrange(
                    all_projected[head_name],
                    "(num_global_views bs) ... -> bs num_global_views ...",
                    bs=batch_size,
                )
                predicted = einops.rearrange(
                    all_predicted[head_name],
                    "(num_views bs) ... -> bs num_views ...",
                    bs=batch_size,
                )

                # calculate losses
                loss_kwargs = {}
                if isinstance(head, (MugsGroupHead, MugsGroupHeadOg)):
                    loss_kwargs["center"] = head.center
                loss_fn = self.trainer.loss_functions[head_name]
                if isinstance(loss_fn, MocoLoss):
                    loss_kwargs["queue"] = head.loss_queue
                    loss_kwargs["queue_ptr"] = head.loss_queue_ptr
                head_losses, head_infos = loss_fn(
                    projected=projected,
                    predicted=predicted,
                    reduction=reduction,
                    **loss_kwargs,
                )
                total_loss = total_loss + head_losses["total"]
                losses.update({f"heads.{head_name}.{key}": value for key, value in head_losses.items()})
                infos.update({f"infos/{key}/{head_name}": value for key, value in head_infos.items()})

            yield dict(total=total_loss, **losses), infos
