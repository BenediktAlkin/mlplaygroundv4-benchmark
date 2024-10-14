import torch
from torch import nn

from .base.initializer_base import InitializerBase
from kappamodules.init import init_norms_as_identity


class CopyBlocksInitializer(InitializerBase):
    def __init__(self, mappings, **kwargs):
        super().__init__(**kwargs)
        self.mappings = mappings

    def init_weights(self, model):
        # sd requires clone becuase otherwise the inplace operation to initialize norms as identity would
        # overwrite the value in the sd which would make the whole network an identity function
        src_sd = {k: v.clone() for k, v in model.state_dict().items()}
        dst_sd = {k: v.clone() for k, v in model.state_dict().items()}

        for mapping in self.mappings:
            src_idx = mapping["src_idx"]
            dst_idx = mapping["dst_idx"]
            is_identity = mapping.get("is_identity")

            src_keys = [key for key in src_sd.keys() if key.startswith(f"blocks.{src_idx}")]
            dst_keys = [key for key in dst_sd.keys() if key.startswith(f"blocks.{dst_idx}")]
            assert len(src_keys) == len(dst_keys)
            for i in range(len(src_keys)):
                assert all(
                    dst_split == src_split
                    for j, (dst_split, src_split) in enumerate(zip(dst_keys[i].split("."), src_keys[i].split(".")))
                    if j != 1
                )
                dst_sd[dst_keys[i]] = src_sd[src_keys[i]].clone()
            if is_identity:
                # differentiate between pre/postnorm block by order of parameters
                # prenorm has norm -> proj
                # postnorm has proj -> norm
                if "norm" in dst_keys[0]:
                    zero_keys = [key for key in dst_keys if "proj" in key or "fc2" in key]
                    one_keys = []
                    assert len(zero_keys) == 4
                elif "attn" in dst_keys[0]:
                    zero_keys = [
                        key
                        for key in dst_keys
                        if "proj" in key or "fc2" in key or ("norm" in key and "bias" in key)
                    ]
                    one_keys = [key for key in dst_keys if ("norm" in key and "weight" in key)]
                    assert len(zero_keys) == 6
                    assert len(one_keys) == 2
                    raise NotImplementedError("would need to change the norm weights of the previous block as well")
                else:
                    raise NotImplementedError

                # set projection layers to zero
                # 4 keys expected (weights and biases for attn and mlp)
                self.logger.info(f"set keys to zero: {zero_keys}")
                for zero_key in zero_keys:
                    dst_sd[zero_key] = torch.zeros_like(dst_sd[zero_key])
                self.logger.info(f"set keys to one: {one_keys}")
                for one_key in one_keys:
                    dst_sd[one_key] = torch.zeros_like(dst_sd[one_key])

        model.load_state_dict(dst_sd)
