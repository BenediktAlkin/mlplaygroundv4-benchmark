from functools import partial

import torch

from callbacks.base.periodic_callback import PeriodicCallback


class OfflineLogitsCallback(PeriodicCallback):
    def __init__(self, dataset_key, mode="hard", **kwargs):
        super().__init__(**kwargs)
        self.dataset_key = dataset_key
        self.__config_id = None
        self.mode = mode
        self.out = self.path_provider.stage_output_path / "logits"

    def _before_training(self, model, trainer, **kwargs):
        assert len(model.output_shape) == 1
        self.out.mkdir(exist_ok=True)

    def _register_sampler_configs(self, trainer):
        self.__config_id = self._register_sampler_config_from_key(key=self.dataset_key, mode="x class class_all")

    @staticmethod
    def _forward(batch, model, trainer):
        (x, cls, cls_all), _ = batch
        x = x.to(model.device, non_blocking=True)
        with trainer.autocast_context:
            preds = model.classify(x)
        preds = {name: pred.cpu() for name, pred in preds.items()}
        return preds, cls.clone(), cls_all.clone()

    # noinspection PyMethodOverriding
    def _periodic_callback(self, model, trainer, batch_size, data_iter, **_):
        predictions, cls, cls_all = self.iterate_over_dataset(
            forward_fn=partial(self._forward, model=model, trainer=trainer),
            config_id=self.__config_id,
            batch_size=batch_size,
            data_iter=data_iter,
        )

        for name, preds in predictions.items():
            cls_hat = preds.argmax(dim=1)

            # calculate accuracy
            is_unlabeled_idx = (cls == -1).nonzero().squeeze(1)
            unlabeled_acc = (cls_hat[is_unlabeled_idx] == cls_all[is_unlabeled_idx]).sum() / len(is_unlabeled_idx)
            is_labeled_idx = (cls != -1).nonzero().squeeze(1)
            labeled_acc = (cls_hat[is_labeled_idx] == cls[is_labeled_idx]).sum() / len(is_labeled_idx)
            full_acc = (cls_hat == cls_all).sum() / len(preds)
            # log
            key = f"accuracy1/{self.dataset_key}/{name}"
            self.logger.info(f"{key}/unlabeled: {unlabeled_acc:.4f}")
            self.writer.add_scalar(f"{key}/unlabeled", unlabeled_acc)
            self.logger.info(f"{key}/labeled: {labeled_acc:.4f}")
            self.writer.add_scalar(f"{key}/labeled", labeled_acc)
            self.logger.info(f"{key}/full: {full_acc:.4f}")
            self.writer.add_scalar(f"{key}/full", full_acc)
            # save
            cur_out = self.out / f"{self.dataset_key}_{name}_{self.update_counter.cur_checkpoint}.th"
            self.logger.info(f"writing predicted logits to {cur_out}")
            if self.mode == "hard":
                torch.save(dict(label=cls_hat, confidence=preds.softmax(dim=1).max(dim=1).values), cur_out)
            elif self.mode == "soft":
                torch.save(preds, cur_out)
            else:
                raise NotImplementedError
