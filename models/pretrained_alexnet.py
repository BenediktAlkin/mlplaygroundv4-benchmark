from torchvision.models import alexnet, AlexNet_Weights

from models.base.single_model_base import SingleModelBase


class PretrainedAlexnet(SingleModelBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
        AlexNet_Weights.IMAGENET1K_V1.transforms

    def forward(self, x):
        return dict(main=self.model(x))

    def classify(self, x):
        return self(x)
