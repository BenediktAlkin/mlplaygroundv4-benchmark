import torch

from models.base.single_model_base import SingleModelBase
from .dinov2.dinov2 import vit_small


class VitDinov2(SingleModelBase):
    def __init__(self, model, sd_path, load_student=False, **kwargs):
        super().__init__(**kwargs)
        if model == "vit_small":
            self.model = vit_small(
                init_values=1e-5,
                ffn_layer="mlp",
                block_chunks=4,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
            )
        else:
            raise NotImplementedError
        sd = torch.load(sd_path, map_location="cpu")["model"]
        startswith = "student." if load_student else "teacher."
        sd = {
            key.replace(startswith, ""): value
            for key, value in sd.items() if key.startswith(startswith)
        }
        sd = {key.replace("backbone.", ""): value for key, value in sd.items() if key.startswith("backbone.")}
        self.model.load_state_dict(sd)

    @property
    def blocks(self):
        return self.model.blocks

    def forward(self, x):
        return dict(main=self.model(x))

    def classify(self, x):
        return self(x)
