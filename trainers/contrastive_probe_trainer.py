import einops
import torch.nn.functional as F
from kappadata.utils.multi_crop_utils import concat_same_shape_inputs
from kappadata.wrappers import ModeWrapper
from torch import nn

from callbacks.online_callbacks.update_output_callback import UpdateOutputCallback
from losses import loss_fn_from_kwargs
from utils.factory import create_collection
from .base.sgd_trainer import SgdTrainer


class ContrastiveProbeTrainer(SgdTrainer):
    def __init__(self, loss_functions, train_probes_mode, **kwargs):
        super().__init__(**kwargs)
        self.train_probes_mode = train_probes_mode
        self.loss_functions = create_collection(
            loss_functions,
            loss_fn_from_kwargs,
            update_counter=self.update_counter,
        )

    def get_trainer_callbacks(self, model=None):
        return [
            UpdateOutputCallback(
                patterns=["nn-", "global_alignment/", "local_alignment/", "groundtruth_accuracy/", "pseudo_accuracy/"],
                verbose=False,
                every_n_updates=self.track_every_n_updates,
                every_n_samples=self.track_every_n_samples,
                **self.get_default_callback_kwargs(),
            ),
            UpdateOutputCallback(
                patterns=["nn-", "global_alignment/", "local_alignment/", "groundtruth_accuracy/", "pseudo_accuracy/"],
                verbose=True,
                **self.get_default_callback_intervals(),
                **self.get_default_callback_kwargs(),
            ),
        ]

    @property
    def output_shape(self):
        return self.data_container.get_dataset("train").getdim_class(),

    @property
    def dataset_mode(self):
        return "index x class class_all"

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
                assert not self.training
                x = [x.to(self.model.device, non_blocking=True)]
            cls = cls.to(self.model.device, non_blocking=True)
            idx = idx.to(self.model.device, non_blocking=True)
            x, batch_size = concat_same_shape_inputs(x)
            clsall = ModeWrapper.get_item(mode=self.trainer.dataset_mode, item="class_all", batch=batch)
            clsall = clsall.to(self.model.device, non_blocking=True)

            # model forward pass
            model_outputs = self.model(x, idx=idx, cls=cls)

            total_loss = 0
            losses = {}
            infos = {}
            # probe losses
            is_gt_label = cls != -1
            gt_cls = cls[is_gt_label]
            pseudo_cls = clsall[~is_gt_label]
            num_global_views = len(x[0]) // batch_size
            is_semi = (~is_gt_label).sum() > 0
            if self.trainer.train_probes_mode == "global":
                is_gt_label = is_gt_label.repeat(num_global_views)
                gt_cls = gt_cls.repeat(num_global_views)
                pseudo_cls = pseudo_cls.repeat(num_global_views)
                num_pred_views = num_global_views
            else:
                raise NotImplementedError
            assert reduction == "mean"
            for key in self.model.probes.keys():
                preds = model_outputs[f"probes.{key}"][:num_pred_views * batch_size]
                gt_preds = preds[is_gt_label]
                loss = F.cross_entropy(gt_preds, gt_cls, reduction=reduction)
                losses[f"probes.{key}"] = loss
                total_loss = total_loss + loss
                # ground truth accuracy
                gt_acc = gt_preds.argmax(dim=1) == gt_cls
                gt_acc = gt_acc.float().mean()
                infos[f"groundtruth_accuracy/{key}"] = gt_acc
                # pseudo accuracy
                if is_semi:
                    pseudo_preds = preds[~is_gt_label]
                    pseudo_acc = pseudo_preds.argmax(dim=1) == pseudo_cls
                    pseudo_acc = pseudo_acc.float().mean()
                    infos[f"pseudo_accuracy/{key}"] = pseudo_acc

            # iterate over heads
            for head_name in self.trainer.loss_functions.keys():
                # postprocess outputs (heads can return metrics, e.g. NN-accuracy for NNCLR)
                head_outputs = model_outputs[f"heads.{head_name}"]
                if "metrics" in head_outputs:
                    infos.update({
                        f"{key}/{head_name}": value
                        for key, value in head_outputs.pop("metrics").items()
                    })

                # convert projected/predicted to (bs, num_views, dim) for loss
                for key in head_outputs.keys():
                    # heads can return multiple "projected" values (e.g. NNCLR returns pre-swap/post-swap projected)
                    if "projected" in key:
                        head_outputs[key] = einops.rearrange(
                            head_outputs[key],
                            "(num_global_views bs) dim -> bs num_global_views dim",
                            bs=batch_size,
                        )
                    if key == "predicted":
                        head_outputs[key] = einops.rearrange(
                            head_outputs[key],
                            "(num_views bs) dim -> bs num_views dim",
                            bs=batch_size,
                        )

                # calculate losses
                head_losses, loss_infos = self.trainer.loss_functions[head_name](
                    **head_outputs,
                    reduction=reduction,
                )
                total_loss = total_loss + head_losses["total"]
                losses.update({f"heads.{head_name}.{key}": value for key, value in head_losses.items()})
                infos.update({f"{key}/{head_name}": value for key, value in loss_infos.items()})
            yield dict(total=total_loss, **losses), infos
