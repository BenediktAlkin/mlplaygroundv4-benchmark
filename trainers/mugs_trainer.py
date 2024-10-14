import einops
import torch
from kappadata.utils.multi_crop_utils import concat_same_shape_inputs
from kappadata.wrappers import ModeWrapper
from torch import nn

from callbacks.online_callbacks.update_output_callback import UpdateOutputCallback
from losses import loss_fn_from_kwargs
from losses.ssl.moco_loss import MocoLoss
from losses.ssl.nnclr_loss import NnclrLoss
from losses.ssl.dino_loss import DinoLoss
from utils.factory import create_collection
from .base.sgd_trainer import SgdTrainer


class MugsTrainer(SgdTrainer):
    def __init__(self, loss_functions, filter_strong_aug_in_queue=True, **kwargs):
        super().__init__(**kwargs)
        self.loss_functions = nn.ModuleDict(
            create_collection(
                loss_functions,
                loss_fn_from_kwargs,
                update_counter=self.update_counter,
            )
        )
        self.filter_strong_aug_in_queue = filter_strong_aug_in_queue

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
        return "index x class"

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
            cls = ModeWrapper.get_item(mode=self.trainer.dataset_mode, item="class", batch=batch)
            idx = ModeWrapper.get_item(mode=self.trainer.dataset_mode, item="index", batch=batch)
            if isinstance(x, list):
                x = [xx.to(self.model.device, non_blocking=True) for xx in x]
            else:
                x = [x.to(self.model.device, non_blocking=True)]
            cls = cls.to(self.model.device, non_blocking=True)
            idx = idx.to(self.model.device, non_blocking=True)
            x, batch_size = concat_same_shape_inputs(x)
            if "is_weak_global_aug" in ctx:
                if self.trainer.filter_strong_aug_in_queue:
                    is_weak_aug = True
                else:
                    is_weak_aug = ctx["is_weak_global_aug"].to(self.model.device, non_blocking=True)
            else:
                is_weak_aug = None
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
                loss_fn = self.trainer.loss_functions[head_name]
                projected = all_projected[head_name]

                # queue
                if isinstance(loss_fn, NnclrLoss):
                    projected_postswap, metrics = head.queue(x=projected, cls=cls, idx=idx)
                    projected_postswap = einops.rearrange(
                        projected_postswap,
                        "(num_global_views bs) dim -> bs num_global_views dim",
                        bs=batch_size,
                    )
                else:
                    projected_postswap = None
                    metrics = None


                # convert to (bs, num_views, dim) for loss
                projected = einops.rearrange(
                    projected,
                    "(num_global_views bs) dim -> bs num_global_views dim",
                    bs=batch_size,
                )
                predicted = einops.rearrange(
                    all_predicted[head_name],
                    "(num_views bs) dim -> bs num_views dim",
                    bs=batch_size,
                )

                # calculate losses
                loss_kwargs = {}
                if isinstance(loss_fn, NnclrLoss):
                    head_losses, head_infos = loss_fn(
                        projected_preswap=projected,
                        projected_postswap=projected_postswap,
                        predicted=predicted,
                        reduction=reduction,
                        **loss_kwargs,
                    )
                elif isinstance(loss_fn, DinoLoss):
                    head_losses, head_infos = loss_fn(
                        projected=projected,
                        predicted=predicted,
                        center=head.center,
                        reduction=reduction,
                        **loss_kwargs,
                    )
                elif isinstance(loss_fn, MocoLoss):
                    head_losses, head_infos = loss_fn(
                        projected=projected,
                        predicted=predicted,
                        queue=head.loss_queue,
                        queue_ptr=head.loss_queue_ptr,
                        reduction=reduction,
                        **loss_kwargs,
                    )
                else:
                    raise NotImplementedError

                # sum losses
                total_loss = total_loss + head_losses["total"]
                losses.update({f"heads.{head_name}.{key}": value for key, value in head_losses.items()})
                infos.update({f"infos/{key}/{head_name}": value for key, value in head_infos.items()})

                # metrics
                if metrics is not None:
                    infos.update({f"nnclr_queue/{metric_name}/{head_name}": v for metric_name, v in metrics.items()})

            yield dict(total=total_loss, **losses), infos
