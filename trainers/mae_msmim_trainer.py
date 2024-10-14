import torch
from distributed.config import is_distributed
import einops
import torch.nn as nn
from kappadata.wrappers import ModeWrapper
from kappaschedules import object_to_schedule
from utils.schedule_utils import get_value_or_default

from callbacks.online_callbacks.update_output_callback import UpdateOutputCallback
from losses import loss_fn_from_kwargs
from models.vit.mask_generators import mask_generator_from_kwargs
from utils.factory import create
from .base.sgd_trainer import SgdTrainer


class MaeMsmimTrainer(SgdTrainer):
    def __init__(
            self,
            mae_loss_fn,
            msmim_loss_fn,
            mask_generator,
            num_masks,
            msmim_loss_weight_distribution,
            style_loss_weight,
            aux_loss_weight,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.mae_loss_fn = create(mae_loss_fn, loss_fn_from_kwargs, update_counter=self.update_counter)
        self.msmim_loss_fn = create(msmim_loss_fn, loss_fn_from_kwargs, update_counter=self.update_counter)
        self.num_masks = num_masks
        # mask generator
        self.mask_generator = create(
            mask_generator,
            mask_generator_from_kwargs,
            update_counter=self.update_counter,
        )
        # loss weights
        self.msmim_loss_weight_distribution = msmim_loss_weight_distribution
        self.style_loss_weight = style_loss_weight
        self.aux_loss_weight = aux_loss_weight

    @property
    def lr_scale_factor(self):
        return self.effective_batch_size * self.num_masks

    def get_trainer_callbacks(self, model=None):
        return [
            UpdateOutputCallback(
                patterns=["variance/"],
                verbose=False,
                every_n_updates=self.track_every_n_updates,
                every_n_samples=self.track_every_n_samples,
                **self.get_default_callback_kwargs(),
            ),
            UpdateOutputCallback(
                patterns=["variance/"],
                verbose=True,
                **self.get_default_callback_intervals(),
                **self.get_default_callback_kwargs(),
            ),
        ]

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

        def forward(self, batch, mask_generator=None, num_masks=None, reduction="mean"):
            outputs = {}
            # prepare data
            batch, ctx = batch
            idx = ModeWrapper.get_item(mode=self.trainer.dataset_mode, item="index", batch=batch)
            x = ModeWrapper.get_item(mode=self.trainer.dataset_mode, item="x", batch=batch)
            idx = idx.to(self.model.device, non_blocking=True)
            x = x.to(self.model.device, non_blocking=True)

            # model forward pass
            # for evaluation generators can be provided (if not provided -> no generator)
            if self.model.training:
                assert mask_generator is None and num_masks is None
                mask_generator = self.trainer.mask_generator
                num_masks = self.trainer.num_masks
            # repeat x such that multiple masks are generated for the same samples
            if num_masks is not None and num_masks > 1:
                x = einops.repeat(x, "batch_size ... -> (num_masks batch_size) ...", num_masks=num_masks)
            model_outputs = self.model(x, idx=idx, num_masks=num_masks, mask_generator=mask_generator)

            # check unmasked eval step
            if "x_hat" not in model_outputs:
                yield {}, {}
                return

            # calculate losses
            losses = {}
            # reconstruction loss
            mae_loss = self.trainer.mae_loss_fn(
                prediction=model_outputs["x_hat"],
                target=x,
                mask=model_outputs["mask"],
                patch_size=model_outputs["patch_size"],
                reduction=reduction,
            )
            losses["reconstruction"] = mae_loss
            # msmim loss
            all_losses = self.trainer.msmim_loss_fn(
                prediction=model_outputs["masked_z_hat"],
                target=model_outputs["z"],
                mask=model_outputs["mask"],
                reduction=reduction,
                calculate_style_loss=self.trainer.style_loss_weight > 0,
                calculate_aux_loss=self.trainer.aux_loss_weight > 0,
            )
            # track patch losses
            patches_msmim_loss = all_losses["patches_msmim_loss"]
            num_scales = len(patches_msmim_loss)
            for i in range(num_scales):
                losses[f"patches_msmim{i:02d}"] = patches_msmim_loss[i]
            if self.trainer.style_loss_weight > 0:
                patches_style_loss = all_losses["patches_style_loss"]
                for i in range(len(patches_style_loss)):
                    losses[f"patches_msmim_style{i:02d}"] = patches_style_loss[i]
            else:
                patches_style_loss = None
            # track aux losses
            if self.trainer.aux_loss_weight > 0:
                aux_msmim_loss = all_losses["aux_msmim_loss"]
                for i in range(len(aux_msmim_loss)):
                    losses[f"aux_msmim{i:02d}"] = aux_msmim_loss[i]
                if self.trainer.style_loss_weight > 0:
                    aux_style_loss = all_losses["aux_style_loss"]
                    for i in range(len(aux_style_loss)):
                        losses[f"aux_msmim_style{i:02d}"] = aux_style_loss[i]
                else:
                    aux_style_loss = None
            else:
                aux_msmim_loss = None
                aux_style_loss = None

            # metrics
            # assert not is_distributed(), "variance calculation not supported for multi-gpu"
            for i in range(num_scales):
                with torch.no_grad():
                    outputs[f"variance/target/{i:02d}"] = model_outputs["z"][i].detach().var()

            # weight msmim losses
            msmim_loss_weight_distribution_kind = self.trainer.msmim_loss_weight_distribution["kind"]
            if msmim_loss_weight_distribution_kind == "uniform":
                patches_msmim_loss = patches_msmim_loss.mean()
                if self.trainer.aux_loss_weight > 0:
                    aux_msmim_loss = aux_msmim_loss.mean()
                if self.trainer.style_loss_weight > 0:
                    patches_style_loss = patches_style_loss.mean()
                    if self.trainer.aux_loss_weight > 0:
                        aux_style_loss = aux_style_loss.mean()
            elif msmim_loss_weight_distribution_kind in [
                "linear_decay",
                "linear_increase",
                "exponential_decay",
                "fixed",
            ]:
                if msmim_loss_weight_distribution_kind in ["linear_decay", "linear_increase"]:
                    if msmim_loss_weight_distribution_kind == "linear_decay":
                        # linear decay
                        # e.g. 8 scales: [0.8889, 0.7778, 0.6667, 0.5556, 0.4444, 0.3333, 0.2222, 0.1111]
                        weights = torch.arange(num_scales, 0, -1, device=patches_msmim_loss.device) / (num_scales + 1)
                    elif msmim_loss_weight_distribution_kind == "linear_increase":
                        # linear increase
                        # e.g. 8 scales: [0.1111, 0.2222, 0.3333, 0.4444, 0.5556, 0.6667, 0.7778, 0.8889]
                        weights = torch.arange(1, num_scales + 1, device=patches_msmim_loss.device) / (num_scales + 1)
                    else:
                        raise NotImplementedError
                    min_value = self.trainer.msmim_loss_weight_distribution.get("min_value", 0)
                    max_value = self.trainer.msmim_loss_weight_distribution.get("max_value", 1)
                    weights = (max_value - min_value) * weights + min_value
                elif msmim_loss_weight_distribution_kind == "exponential_decay":
                    # exponential decay
                    # 8 scales + 0.85: [0.2725, 0.3206, 0.3771, 0.4437, 0.522, 0.6141, 0.7225, 0.85]
                    factor = self.trainer.msmim_loss_weight_distribution.get("factor", 0)
                    weights = (factor ** torch.arange(1, num_scales + 1, device=patches_msmim_loss.device))
                elif msmim_loss_weight_distribution_kind == "fixed":
                    weights = torch.tensor(
                        self.trainer.msmim_loss_weight_distribution["weights"],
                        device=patches_msmim_loss.device,
                    )
                    assert len(weights) == num_scales
                else:
                    raise NotImplementedError
                patches_msmim_loss = (patches_msmim_loss * weights).mean()
                if self.trainer.aux_loss_weight > 0:
                    aux_msmim_loss = (aux_msmim_loss * weights).mean()
                if self.trainer.style_loss_weight > 0:
                    patches_style_loss = (patches_style_loss * weights).mean()
                    if self.trainer.aux_loss_weight > 0:
                        aux_style_loss = (aux_style_loss * weights).mean()
            else:
                raise NotImplementedError
            losses["total"] = mae_loss + patches_msmim_loss
            if self.trainer.style_loss_weight > 0:
                losses["total"] = losses["total"] + self.trainer.style_loss_weight * patches_style_loss
            if self.trainer.aux_loss_weight > 0:
                losses["total"] = losses["total"] + self.trainer.aux_loss_weight * aux_msmim_loss
                if self.trainer.style_loss_weight > 0:
                    weight = self.trainer.aux_loss_weight * self.trainer.style_loss_weight
                    losses["total"] = losses["total"] + weight * aux_style_loss

            yield losses, outputs
