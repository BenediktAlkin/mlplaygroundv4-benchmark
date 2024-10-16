from functools import partial

from kappadata.wrappers import ModeWrapper

from callbacks.base.periodic_callback import PeriodicCallback
from metrics.knn import multiclass_knn
from models.extractors import extractor_from_kwargs
from models.extractors.base.forward_hook import StopForwardException
from utils.factory import create_collection
from utils.formatting_util import dict_to_string
from utils.object_from_kwargs import objects_from_kwargs


class OfflineKnnAgreementCallback(PeriodicCallback):
    def __init__(
            self,
            train_dataset_key,
            test_dataset_key,
            extractors,
            knns=None,
            taus=None,
            forward_kwargs=None,
            inplace=False,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.train_dataset_key = train_dataset_key
        self.test_dataset_key = test_dataset_key
        self.extractors = extractors
        self.forward_kwargs = objects_from_kwargs(forward_kwargs)
        self.knns = knns or [10]
        self.taus = taus or [0.07]
        self.inplace = inplace
        self.__train_config_id = None
        self.__test_config_id = None
        self.__dataset_mode = None
        self._num_classes = None

    def _register_sampler_configs(self, trainer):
        dataset_mode = trainer.dataset_mode
        dataset_mode = ModeWrapper.add_item(mode=dataset_mode, item="class")
        self.__dataset_mode = dataset_mode
        self.__train_config_id = self._register_sampler_config_from_key(key=self.train_dataset_key, mode=dataset_mode)
        self.__test_config_id = self._register_sampler_config_from_key(key=self.test_dataset_key, mode=dataset_mode)

    def _before_training(self, model, **kwargs):
        # num_classes
        class_shape = self.data_container.get_dataset(self.test_dataset_key).getshape_class()
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

        # train_dataset foward
        train_features, train_y = self.iterate_over_dataset(
            forward_fn=partial(self._forward, trainer_model=trainer_model, trainer=trainer),
            config_id=self.__train_config_id,
            batch_size=batch_size,
            data_iter=data_iter,
        )
        # test_dataset forward
        test_features, test_y = self.iterate_over_dataset(
            forward_fn=partial(self._forward, trainer_model=trainer_model, trainer=trainer),
            config_id=self.__test_config_id,
            batch_size=batch_size,
            data_iter=data_iter,
        )

        # calculate/log metrics
        train_y = train_y.to(model.device)
        test_y = test_y.to(model.device)
        # check that len(train_features) == len(train_y) -> raises error when 2 views are propagated
        assert all(len(v) == len(train_y) for v in train_features.values())
        all_preds = {}
        for feature_key in train_features.keys():
            train_x = train_features[feature_key].to(model.device)
            test_x = test_features[feature_key].to(model.device)

            knn_kwargs = dict(
                train_x=train_x,
                test_x=test_x,
                train_y=train_y,
                test_y=test_y,
                k=self.knns,
                tau=self.taus,
                batch_size=min(1024, batch_size),
                inplace=self.inplace,
                mode="predict",
            )

            if train_y.ndim == 1:
                # multiclass
                preds = multiclass_knn(**knn_kwargs)
                assert len(preds) == 1
                all_preds[feature_key] = preds[(10, 0.07)]
            else:
                raise NotImplementedError

        # agreement
        keys = list(all_preds.keys())
        for i in range(len(keys)):
            key_i = keys[i]
            pred_i = all_preds[key_i]
            for j in range(i, len(keys)):
                key_j = keys[j]
                pred_j = all_preds[key_j]
                # agreement -> how often do predictions match (i.e. the same classes are predicted)
                agreement = (pred_i == pred_j).sum() / len(pred_i)
                key = (
                    f"knn_agreement/{key_i}-{key_j}/"
                    f"{self.train_dataset_key}-{self.test_dataset_key}"
                )
                self.writer.add_scalar(key, agreement, logger=self.logger, format_str=".6f")
                # positive agreement -> how often correct/incorrect matches
                pos_agreement = ((pred_i == test_y) == (pred_j == test_y)).sum() / len(pred_i)
                key = (
                    f"knn_positive_agreement/{key_i}-{key_j}/"
                    f"{self.train_dataset_key}-{self.test_dataset_key}"
                )
                self.writer.add_scalar(key, pos_agreement, logger=self.logger, format_str=".6f")

        # cleanup
        for extractor in self.extractors:
            extractor.disable_hooks()
