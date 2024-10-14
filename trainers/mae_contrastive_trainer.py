import einops
from kappadata.utils.multi_crop_utils import concat_same_shape_inputs
from kappadata.wrappers import ModeWrapper
from torch import nn

from callbacks.online_callbacks.update_output_callback import UpdateOutputCallback
from losses import loss_fn_from_kwargs
from models.vit.mask_generators import mask_generator_from_kwargs
from utils.factory import create_collection, create
from .base.sgd_trainer import SgdTrainer


class MaeContrastiveTrainer(SgdTrainer):
    def __init__(self, mask_generator, mae_loss_function, contrastive_loss_functions, **kwargs):
        super().__init__(find_unused_params=True, static_graph=True, **kwargs)
        # MAE
        self.mask_generator = create(
            mask_generator,
            mask_generator_from_kwargs,
            update_counter=self.update_counter,
        )
        self.mae_loss_function = create(
            mae_loss_function,
            loss_fn_from_kwargs,
            update_counter=self.update_counter,
        )
        # contrastive
        self.contrastive_loss_functions = create_collection(
            contrastive_loss_functions,
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

        def mae_step(self, x, idx, mask_generator, reduction):
            # MAE step
            forward_kwargs = {}
            if self.model.training:
                assert mask_generator is None
                forward_kwargs["mask_generator"] = self.trainer.mask_generator
            else:
                forward_kwargs["mask_generator"] = mask_generator
            mae_outputs = self.model.forward_mae(x, idx=idx, **forward_kwargs)
            # calculate loss
            mae_loss = self.trainer.mae_loss_function(
                prediction=mae_outputs["prediction"],
                target=x,
                mask=mae_outputs["mask"],
                patch_size=mae_outputs["patch_size"],
                reduction=reduction,
            )
            return dict(total=mae_loss, reconstruction=mae_loss), {}

        def contrastive_step(self, x, idx, cls, batch_size, reduction):
            # contrastive step
            # model forward pass
            all_projected, all_predicted = self.model.forward_contrastive(x, idx=idx)

            # iterate over heads
            total_loss = 0
            losses = {}
            infos = {}
            for head_name in self.trainer.contrastive_loss_functions.keys():
                head = self.model.heads[head_name]

                # queue
                projected = all_projected[head_name]
                projected_postswap, metrics = head.queue(x=projected, cls=cls, idx=idx)

                # convert to (bs, num_views, dim) for loss
                projected = einops.rearrange(
                    projected,
                    "(num_global_views bs) dim -> bs num_global_views dim",
                    bs=batch_size,
                )
                projected_postswap = einops.rearrange(
                    projected_postswap,
                    "(num_global_views bs) dim -> bs num_global_views dim",
                    bs=batch_size,
                )
                predicted = einops.rearrange(
                    all_predicted[head_name],
                    "(num_views bs) dim -> bs num_views dim",
                    bs=batch_size,
                )

                # calculate losses
                head_losses, head_infos = self.trainer.contrastive_loss_functions[head_name](
                    projected_preswap=projected,
                    projected_postswap=projected_postswap,
                    predicted=predicted,
                    reduction=reduction,
                )
                total_loss = total_loss + head_losses["total"]
                losses.update({f"heads.{head_name}.{key}": value for key, value in head_losses.items()})
                infos.update({f"infos/{key}/{head_name}": value for key, value in head_infos.items()})

                # metrics
                infos.update({f"nnclr_queue/{metric_name}/{head_name}": v for metric_name, v in metrics.items()})
            return dict(total=total_loss, **losses), infos

        def forward(self, batch, mask_generator=None, reduction="mean"):
            # prepare data
            batch, ctx = batch
            x = ModeWrapper.get_item(mode=self.trainer.dataset_mode, item="x", batch=batch)
            cls = ModeWrapper.get_item(mode=self.trainer.dataset_mode, item="class", batch=batch)
            idx = ModeWrapper.get_item(mode=self.trainer.dataset_mode, item="index", batch=batch)
            if isinstance(x, list):
                x = [xx.to(self.model.device, non_blocking=True) for xx in x]
            else:
                assert not self.training
                assert mask_generator is None
                x = x.to(self.model.device, non_blocking=True)
                _ = self.model(x)
                yield {}, {}
                return
            cls = cls.to(self.model.device, non_blocking=True)
            idx = idx.to(self.model.device, non_blocking=True)
            x, batch_size = concat_same_shape_inputs(x)
            assert len(x) == 1, "local views not supported"
            mae_x, contrastive_x = x[0].chunk(2)

            # MAE step
            mae_losses, mae_outputs = self.mae_step(
                x=mae_x,
                idx=idx,
                mask_generator=mask_generator,
                reduction=reduction,
            )
            yield mae_losses, mae_outputs

            # contrastive step
            contrastive_losses, contrastive_outputs = self.contrastive_step(
                x=contrastive_x,
                idx=idx,
                cls=cls,
                batch_size=batch_size,
                reduction=reduction,
            )
            yield contrastive_losses, contrastive_outputs
