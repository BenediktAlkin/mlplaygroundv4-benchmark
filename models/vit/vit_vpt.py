import einops
import torch
from kappamodules.layers import LinearProjection
from torch import nn

from optimizers.param_group_modifiers.exclude_from_wd_by_name_modifier import ExcludeFromWdByNameModifier
from .vit import Vit


class VitVpt(Vit):
    def __init__(self, num_prompt_tokens, prompt_token_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_prompt_tokens = num_prompt_tokens
        self.prompt_token_dim = prompt_token_dim
        self.prompt_tokens = nn.Parameter(torch.empty(1, self.depth, num_prompt_tokens, prompt_token_dim))
        if prompt_token_dim != self.dim:
            self.prompt_proj = LinearProjection(prompt_token_dim, self.dim, init_weights="truncnormal")
        else:
            self.prompt_proj = nn.Identity()

    def model_specific_initialization(self):
        super().model_specific_initialization()
        nn.init.trunc_normal_(self.prompt_tokens, std=0.02)

    def get_model_specific_param_group_modifiers(self):
        return super().get_model_specific_param_group_modifiers() + [ExcludeFromWdByNameModifier(name="prompt_tokens")]

    def load_state_dict(self, state_dict, strict=True):
        missing_keys, unexpected_keys = super().load_state_dict(state_dict=state_dict, strict=False)
        assert len(unexpected_keys) == 0, f"VitVpt encountered unexpected keys: {unexpected_keys}"
        if len(missing_keys) > 0:
            assert "prompt_tokens" in missing_keys, f"prompt_tokens not in missing keys {missing_keys}"
            if len(missing_keys) == 1:
                pass
            elif len(missing_keys) == 3:
                assert "prompt_proj.proj.weight" in missing_keys, missing_keys
                assert "prompt_proj.proj.bias" in missing_keys, missing_keys
            else:
                raise NotImplementedError
        return [], []

    # noinspection PyMethodOverriding
    def forward(self, x):
        outputs = {}

        # embed patches
        x = self.patch_embed(x)
        # add pos_embed
        x = self.pos_embed(x)

        # no mask -> flatten to 1d
        x = einops.rearrange(x, "b ... d -> b (...) d")

        # add cls token
        x = self.cls_tokens(x)

        # project prompts
        prompt_tokens = self.prompt_proj(self.prompt_tokens.expand(len(x), -1, -1, -1))

        # apply blocks
        for i, blk in enumerate(self.blocks):
            # add prompt tokens
            x = torch.concat([prompt_tokens[:, i], x], dim=1)
            # block
            x = blk(x)
            # remove prompt tokens
            x = x[:, self.num_prompt_tokens:]

        if self.pooling is not None:
            x = self.pooling(x)
        x = self.norm(x)
        if self.head is not None:
            x = self.head(x)

        outputs["main"] = x
        return outputs
