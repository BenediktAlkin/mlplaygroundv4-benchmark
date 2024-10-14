from functools import partial

from kappadata.wrappers import ModeWrapper

from callbacks.base.periodic_callback import PeriodicCallback
from metrics.silhouette_score import silhouette_score, max_distance_to_centroid
from models.extractors import extractor_from_kwargs
from models.extractors.base.forward_hook import StopForwardException
from utils.factory import create_collection
from utils.formatting_util import dict_to_string
from utils.object_from_kwargs import objects_from_kwargs


class OfflineSilhouetteScoreCallback(PeriodicCallback):
    def __init__(self, dataset_key, extractors, forward_kwargs=None, **kwargs):
        super().__init__(**kwargs)
        self.dataset_key = dataset_key
        self.extractors = extractors
        self.forward_kwargs = objects_from_kwargs(forward_kwargs)
        self.__config_id = None
        self.__dataset_mode = None
        self._num_classes = None

    def _register_sampler_configs(self, trainer):
        dataset_mode = ModeWrapper.add_item(mode=trainer.dataset_mode, item="class")
        self.__dataset_mode = dataset_mode
        self.__config_id = self._register_sampler_config_from_key(key=self.dataset_key, mode=dataset_mode)

    def _before_training(self, model, **kwargs):
        # num_classes
        class_shape = self.data_container.get_dataset(self.dataset_key).getshape_class()
        assert len(class_shape) == 1
        self._num_classes = class_shape[0]

        # extractors
        self.extractors = create_collection(self.extractors, extractor_from_kwargs, static_ctx=model.static_ctx)
        for extractor in self.extractors:
            extractor.register_hooks(model)
            extractor.disable_hooks()

    def _forward(self, batch, trainer_model, trainer):
        features = {}
        generator = trainer_model(batch=batch, **self.forward_kwargs)
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
        features, _ = self.iterate_over_dataset(
            forward_fn=partial(self._forward, trainer_model=trainer_model, trainer=trainer),
            config_id=self.__config_id,
            batch_size=batch_size,
            data_iter=data_iter,
        )

        # calculate/log metrics
        # y = y.to(model.device)
        for feature_key in features.keys():
            x = features[feature_key].to(model.device)
            forward_kwargs_str = f"/{dict_to_string(self.forward_kwargs)}" if len(self.forward_kwargs) > 0 else ""
            # silhouette_score
            for distance in ["euclidean"]:
                score, inter_cluster_dist, intra_cluster_dist = silhouette_score(
                    x=x,
                    # y=y,
                    num_classes=self._num_classes,
                    distance=distance,
                )
                for metric_name, value in [
                    (f"silhouette/{distance}", score),
                    (f"cluster_inter_dist/{distance}", inter_cluster_dist),
                    (f"cluster_intra_dist/{distance}", intra_cluster_dist),
                ]:
                    self.writer.add_scalar(
                        f"{metric_name}/{feature_key}/{self.dataset_key}{forward_kwargs_str}",
                        value,
                        logger=self.logger,
                        format_str=".4f",
                    )
                # max distance to centroid
                # max_distances = max_distance_to_centroid(
                #     x=x,
                #     # y=y,
                #     num_classes=self._num_classes,
                #     distance=distance,
                # )
                # self.writer.add_scalar(
                #     f"max_distance/max/{distance}/{feature_key}/{self.dataset_key}{forward_kwargs_str}",
                #     max_distances.max(),
                #     logger=self.logger,
                #     format_str=".4f",
                # )
                # self.writer.add_scalar(
                #     f"max_distance/mean/{distance}/{feature_key}/{self.dataset_key}{forward_kwargs_str}",
                #     max_distances.mean(),
                #     logger=self.logger,
                #     format_str=".4f",
                # )

        # cleanup
        for extractor in self.extractors:
            extractor.disable_hooks()
