import einops
from torch import nn
from kappadata.utils.multi_crop_utils import concat_same_shape_inputs
from kappadata.wrappers import ModeWrapper

from callbacks.online_callbacks.update_output_callback import UpdateOutputCallback
from losses import loss_fn_from_kwargs
from utils.factory import create_collection, create
from .base.sgd_trainer import SgdTrainer


class DinoTrainer(SgdTrainer):
    def __init__(self, loss_functions, **kwargs):
        super().__init__(**kwargs)
        self.loss_functions = create_collection(
            loss_functions,
            loss_fn_from_kwargs,
            update_counter=self.update_counter,
        )

    def get_trainer_callbacks(self, model=None):
        return [
            UpdateOutputCallback(
                patterns=["nnclr_queue/", "infos/"],
                verbose=False,
                every_n_updates=self.track_every_n_updates,
                every_n_samples=self.track_every_n_samples,
                **self.get_default_callback_kwargs(),
            ),
            UpdateOutputCallback(
                patterns=["nnclr_queue/", "infos/"],
                verbose=True,
                **self.get_default_callback_intervals(),
                **self.get_default_callback_kwargs(),
            ),
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
            if isinstance(x, list):
                x = [xx.to(self.model.device, non_blocking=True) for xx in x]
            else:
                assert not self.training
                x = [x.to(self.model.device, non_blocking=True)]
            x, batch_size = concat_same_shape_inputs(x)

            # model forward pass
            all_projected, all_predicted = self.model(x)

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
                head_losses, head_infos = self.trainer.loss_functions[head_name](
                    projected=projected,
                    predicted=predicted,
                    center=head.center,
                    reduction=reduction,
                )
                total_loss = total_loss + head_losses["total"]
                losses.update({f"heads.{head_name}.{key}": value for key, value in head_losses.items()})
                infos.update({f"infos/{key}/{head_name}": value for key, value in head_infos.items()})
            yield dict(total=total_loss, **losses), infos
