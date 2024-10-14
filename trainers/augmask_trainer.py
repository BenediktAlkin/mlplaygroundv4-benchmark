import torch.nn.functional as F
from torch import nn

from callbacks.online_callbacks.online_accuracy_callback import OnlineAccuracyCallback
from models.vit.mask_generators import mask_generator_from_kwargs
from utils.factory import create
from .base.sgd_trainer import SgdTrainer


class AugmaskTrainer(SgdTrainer):
    def __init__(self, mask_generator, **kwargs):
        super().__init__(**kwargs)
        self.mask_generator = create(
            mask_generator,
            mask_generator_from_kwargs,
            update_counter=self.update_counter,
        )

    def get_trainer_callbacks(self, model=None):
        # select suited callback_ctor for dataset type (binary/multiclass/multilabel)
        ds = self.data_container.get_dataset("train")
        assert ds.getdim("class") > 2
        callback_ctor = OnlineAccuracyCallback

        # create callbacks
        return [
            callback_ctor(
                verbose=False,
                every_n_updates=self.track_every_n_updates,
                every_n_samples=self.track_every_n_samples,
                **self.get_default_callback_kwargs(),
            ),
            callback_ctor(
                **self.get_default_callback_intervals(),
                **self.get_default_callback_kwargs(),
            ),
        ]

    @property
    def output_shape(self):
        return self.data_container.get_dataset("train").getdim_class(),

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
            (idx, x, target), ctx = batch
            x = x.to(self.model.device, non_blocking=True)
            idx = idx.to(self.model.device, non_blocking=True)
            target = target.to(self.model.device, non_blocking=True)

            # forward
            preds = self.model(x)["main"]
            augmask_preds = self.model(x, idx=idx, mask_generator=self.trainer.mask_generator)["main"]
            # calculate loss
            main_loss = F.cross_entropy(preds, target, reduction=reduction)
            augmask_loss = F.cross_entropy(
                augmask_preds,
                F.softmax(preds.detach(), dim=-1),
                reduction=reduction,
            )
            losses = {
                "main": main_loss,
                "augmask": augmask_loss,
                "total": (main_loss + augmask_loss) / 2,
            }
            # compose outputs (for callbacks to use)
            outputs = {
                "idx": idx,
                "preds": dict(main=preds, augmask=augmask_preds),
                "class": target,
            }
            yield losses, outputs
