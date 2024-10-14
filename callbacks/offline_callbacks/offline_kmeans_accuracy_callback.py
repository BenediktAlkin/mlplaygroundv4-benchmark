from functools import partial

from kappadata.wrappers import ModeWrapper

from callbacks.base.periodic_callback import PeriodicCallback
from metrics.kmeans_accuracy import kmeans_accuracy
from models.extractors import extractor_from_kwargs
from models.extractors.base.forward_hook import StopForwardException
from utils.factory import create_collection
from utils.formatting_util import dict_to_string
from utils.object_from_kwargs import objects_from_kwargs


class OfflineKmeansAccuracyCallback(PeriodicCallback):
    def __init__(self, dataset_key, extractors, **kwargs):
        super().__init__(**kwargs)
        self.dataset_key = dataset_key
        self.extractors = extractors
        self.__config_id = None
        self.__dataset_mode = None

    def _register_sampler_configs(self, trainer):
        dataset_mode = ModeWrapper.add_item(mode=trainer.dataset_mode, item="class")
        self.__dataset_mode = dataset_mode
        self.__config_id = self._register_sampler_config_from_key(key=self.dataset_key, mode=dataset_mode)

    def _before_training(self, model, **kwargs):
        # extractors
        self.extractors = create_collection(self.extractors, extractor_from_kwargs, static_ctx=model.static_ctx)
        for extractor in self.extractors:
            extractor.register_hooks(model)
            extractor.disable_hooks()

    def _forward(self, batch, trainer_model, trainer):
        features = {}
        generator = trainer_model(batch=batch)
        with trainer.autocast_context:
            try:
                next(generator)
            except StopForwardException:
                pass
        for extractor in self.extractors:
            features[str(extractor)] = extractor.extract().cpu()
        batch, _ = batch  # remove ctx
        target = ModeWrapper.get_item(mode=self.__dataset_mode, item="class", batch=batch)
        return features, target.clone()

    # noinspection PyMethodOverriding
    def _periodic_callback(self, model, trainer_model, trainer, batch_size, data_iter, **_):
        # setup
        for extractor in self.extractors:
            extractor.enable_hooks()

        # forward
        features, y = self.iterate_over_dataset(
            forward_fn=partial(self._forward, trainer_model=trainer_model, trainer=trainer),
            config_id=self.__config_id,
            batch_size=batch_size,
            data_iter=data_iter,
        )

        # calculate/log metrics
        for key in features.keys():
            value = kmeans_accuracy(x=features[key], y=y)
            self.writer.add_scalar(
                f"accuracy1/{key}/{self.dataset_key}",
                value,
                logger=self.logger,
                format_str=".4f",
            )

        # cleanup
        for extractor in self.extractors:
            extractor.disable_hooks()
