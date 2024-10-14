import numpy as np
from torch import nn

from kappamodules.init import init_xavier_uniform_zero_bias, init_norm_as_noaffine

from models.base.single_model_base import SingleModelBase
from utils.factory import instantiate
from functools import partial
from kappamodules.layers import AsyncBatchNorm


class Resnet(SingleModelBase):
    def __init__(self, version, batchnorm="sync", **kwargs):
        super().__init__(**kwargs)
        # resnet is hardcoded to have a linear classifier as last layer
        if self.output_shape is None:
            num_classes = 1
        else:
            num_classes = np.prod(self.output_shape)
        self.model = instantiate(
            module_names=[f"torchvision.models.resnet"],
            type_names=[version],
            num_classes=num_classes,
        )
        if self.output_shape is None:
            self.output_shape = (self.model.fc.in_features,)
            self.model.fc = nn.Identity()
        assert len(self.input_shape) == 3
        c, h, w = self.input_shape
        if h == w == 32 or h == w == 28:
            self.model.conv1 = nn.Conv2d(
                in_channels=self.input_shape[0],
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding="same",
                bias=False,
            )
            self.model.maxpool = nn.Identity()
        elif h != 224 or w != 224:
            raise NotImplementedError
        # resnet is hardcoded to 3 input channels
        if self.input_shape[0] != self.model.conv1.in_channels:
            self.model.conv1 = nn.Conv2d(
                in_channels=self.input_shape[0],
                out_channels=self.model.conv1.out_channels,
                kernel_size=self.model.conv1.kernel_size,
                stride=self.model.conv1.stride,
                dilation=self.model.conv1.dilation,
                groups=self.model.conv1.groups,
                padding=self.model.conv1.padding,
                bias=self.model.conv1.bias,
                padding_mode=self.model.conv1.padding_mode,
            )
        if batchnorm == "sync":
            pass
        elif batchnorm == "async":
            self.model = AsyncBatchNorm.convert_async_batchnorm(self.model)
        else:
            raise NotImplementedError

    def model_specific_initialization(self):
        self.apply(partial(init_xavier_uniform_zero_bias, gain=2 ** 0.5))
        self.apply(init_norm_as_noaffine)

    def forward(self, x):
        return dict(main=self.model(x))

    def classify(self, x):
        return self(x)
