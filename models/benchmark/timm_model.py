import numpy as np
from torch import nn

from kappamodules.init import init_xavier_uniform_zero_bias, init_norm_as_noaffine

from models.base.single_model_base import SingleModelBase
from utils.factory import instantiate
from functools import partial



class TimmModel(SingleModelBase):
    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        if self.output_shape is None:
            num_classes = 1
        else:
            num_classes = np.prod(self.output_shape)

        module_name, model_name = name.split(".")
        self.model = instantiate(
            module_names=[f"timm.models.{module_name}"],
            type_names=[model_name],
            in_chans=self.input_shape[0],
            num_classes=num_classes,
        )

    def model_specific_initialization(self):
        self.apply(init_xavier_uniform_zero_bias)
        self.apply(init_norm_as_noaffine)

    def forward(self, x):
        return dict(main=self.model(x))
