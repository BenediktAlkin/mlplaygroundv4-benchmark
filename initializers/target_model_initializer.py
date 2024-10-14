from utils.model_utils import copy_params
from .base.initializer_base import InitializerBase


class TargetModelInitializer(InitializerBase):
    def init_weights(self, model):
        self.logger.info(f"TargetModelInitializer: initializing target_model with parameters from online_model")
        copy_params(model.model, model.teacher_model)
