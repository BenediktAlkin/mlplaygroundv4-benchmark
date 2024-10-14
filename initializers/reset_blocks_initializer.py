import torch
from torch import nn

from .base.initializer_base import InitializerBase
from kappamodules.init import init_norms_as_identity


class ResetBlocksInitializer(InitializerBase):
    """
    if mode=="identity":
    replaces blocks (e.g. of ViT) with the identity function by
    - re-initialize all blocks after the specified index
    - re-initializing norm layers as identity
    - (biases are assumed to be initialized to 0 by the model)
    if mode=="random":
    replaces block with random initialization
    """

    def __init__(self, start_idx, mode, **kwargs):
        super().__init__(**kwargs)
        self.start_idx = start_idx
        self.mode = mode

    def init_weights(self, model):
        # sd requires clone becuase otherwise the inplace operation to initialize norms as identity would
        # overwrite the value in the sd which would make the whole network an identity function
        sd = {k: v.clone() for k, v in model.state_dict().items()}
        # remove blocks from weights to replace with identity
        expected_missing_keys = []
        start_idx = self.start_idx
        if self.start_idx < 0:
            max_block_idx = -1
            for key in sd.keys():
                if key.startswith("blocks."):
                    max_block_idx = max(max_block_idx, int(key.split(".")[1]))
            assert 0 <= max_block_idx
            start_idx += max_block_idx + 1
            assert 0 <= start_idx
        # handle last norm
        if self.mode == "random":
            # remove norm (norm is reinitialized via model_specific_initialization)
            for key in list(sd.keys()):
                if key.startswith("norm."):
                    sd.pop(key)
                    expected_missing_keys.append(key)
        elif self.mode in ["identity_random", "identity_preserve"]:
            # copy norm parameters from first deleted block into last norm
            norm_pattern = f"blocks.{start_idx}.norm1."
            for key in list(sd.keys()):
                if key.startswith(norm_pattern):
                    sd[key.replace(norm_pattern, "norm.")] = sd[key]
        elif self.mode == "delete":
            for i in range(start_idx, len(model.blocks)):
                model.blocks[i] = nn.Identity()
            if hasattr(model, "use_last_norm") and model.use_last_norm:
                model.norm = nn.Identity()
                model.use_last_norm = False
            if hasattr(model, "last_norm"):
                model.last_norm = nn.Identity()
                for key in list(sd.keys()):
                    if key.startswith("last_norm"):
                        sd.pop(key)
            if hasattr(model, "last_proj"):
                model.pred = nn.Identity()
                for key in list(sd.keys()):
                    if key.startswith("last_proj"):
                        sd.pop(key)
        else:
            raise NotImplementedError

        for key in list(sd.keys()):
            # remove head (if it is present)
            if key.startswith("head.") or key.startswith("pooling."):
                sd.pop(key)
                expected_missing_keys.append(key)
            if not key.startswith("blocks."):
                continue
            block_idx = int(key.split(".")[1])
            if start_idx <= block_idx:
                sd.pop(key)
                expected_missing_keys.append(key)

        if self.mode == "identity_random":
            model.model_specific_initialization()
            # init block norms as identity (last norm is overwritten via state_dict)
            model.apply(init_norms_as_identity)
        elif self.mode == "identity_preserve":
            # init block norms as identity (last norm is overwritten via state_dict)
            model.apply(init_norms_as_identity)
            # zero v bias + MLP biases to preserve identity
            for block in model.blocks:
                v_bias = block.attn.qkv.bias
                torch.nn.init.zeros_(v_bias[-len(v_bias) // 3:])
                torch.nn.init.zeros_(block.attn.proj.bias)
                torch.nn.init.zeros_(block.mlp.fc1.bias)
                torch.nn.init.zeros_(block.mlp.fc2.bias)
        elif self.mode == "random":
            model.model_specific_initialization()
        elif self.mode == "delete":
            expected_missing_keys = [
                k for k in expected_missing_keys
                if int(k.split(".")[1]) < start_idx
            ]
        else:
            raise NotImplementedError

        missing_keys, unexpected_keys = model.load_state_dict(sd, strict=False)
        assert set(expected_missing_keys) == set(missing_keys), f"{expected_missing_keys} != {missing_keys}"
        assert len(unexpected_keys) == 0, f"len({unexpected_keys}) != 0"
