from functools import partial

import einops
import numpy as np
import torch
from kappamodules.functional.pos_embed import get_sincos_pos_embed_from_seqlens
from kappamodules.init import init_norm_as_noaffine, init_xavier_uniform_zero_bias, init_xavier_uniform_merged_linear
from kappamodules.vit import VitBlock
from torch import nn

from models.base.single_model_base import SingleModelBase
from optimizers.param_group_modifiers.exclude_from_wd_by_name_modifier import ExcludeFromWdByNameModifier


class MsmimMsgDecoder(SingleModelBase):
    def __init__(self, dim, depth, num_attn_heads, add_one_block, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = self.static_ctx["patch_size"]
        encoder_input_shape = self.static_ctx["input_shape"]
        assert len(self.patch_size) == len(encoder_input_shape) - 1
        self.dim = dim
        self.depth = depth
        self.num_attn_heads = num_attn_heads
        self.add_one_block = add_one_block
        self.eps = eps

        # ctors
        norm_ctor = partial(nn.LayerNorm, eps=eps)

        # decoder doesn't produce original image shape but flattened patches
        self.patch_numel = np.prod(self.patch_size) * encoder_input_shape[0]
        num_tokens, encoder_dim = self.input_shape
        self.output_shape = (num_tokens, self.patch_numel)
        self.num_aux_tokens = self.static_ctx["num_aux_tokens"]

        # encoder dim to decoder dim
        self.embed = nn.ModuleList([
            nn.Linear(encoder_dim if i == 0 else encoder_dim + dim, dim)
            for i in range(depth)
        ])
        # mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))

        # fixed pos embedding
        self.seqlens = self.static_ctx["sequence_lengths"]
        pos_embed = get_sincos_pos_embed_from_seqlens(seqlens=self.seqlens, dim=self.dim)
        self.register_buffer("pos_embed", einops.rearrange(pos_embed, "... dim -> 1 (...) dim"))

        # blocks
        self.blocks = nn.ModuleList([
            VitBlock(
                dim=self.dim,
                num_heads=self.num_attn_heads,
                norm_ctor=norm_ctor,
            )
            for _ in range(self.depth)
        ])
        # predictors (decoder latent to encoder latent)
        self.zhat_preds = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim, eps=eps),
                nn.Linear(dim, encoder_dim),
            )
            for _ in range(self.depth)
        ])
        # last block
        if add_one_block:
            self.last_block = VitBlock(
                dim=self.dim,
                num_heads=self.num_attn_heads,
                norm_ctor=norm_ctor,
            )
        else:
            self.last_block = nn.Identity()
        # decoder to patch
        self.xhat_pred = nn.Sequential(
            nn.LayerNorm(dim, eps=eps),
            nn.Linear(dim, self.patch_numel),
        )

    def model_specific_initialization(self):
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(init_norm_as_noaffine)
        self.apply(init_xavier_uniform_zero_bias)
        for block in self.blocks:
            block.reset_parameters()

    @staticmethod
    def get_model_specific_param_group_modifiers():
        # ExcludeFromWdByNameModifier(name="pos_embed") -> not used because pos_embed is never learnable
        return [ExcludeFromWdByNameModifier(name="mask_token")]

    def forward(self, x, ids_restore):
        outputs = {}

        # intermediate features are in reverse order (first encoder block to last encoder block) -> reverse
        inputs = list(reversed(x))

        # extract shapes
        x = inputs[0]
        bs, num_input_tokens, _ = x.shape
        _, total_num_patches = ids_restore.shape
        num_hidden_patches = total_num_patches - (num_input_tokens - self.num_aux_tokens)
        num_visible_patches = total_num_patches - num_hidden_patches
        # prepare shapes
        mask_tokens = self.mask_token.repeat(bs, num_hidden_patches, 1)
        ids_shuffle = ids_restore.argsort(dim=1)
        ids_shuffle_visible = ids_shuffle[:, :num_visible_patches].unsqueeze(-1).expand(-1, -1, self.dim)
        ids_shuffle_masked = ids_shuffle[:, num_visible_patches:].unsqueeze(-1).expand(-1, -1, self.dim)
        ids_restore = ids_restore.unsqueeze(-1).expand(-1, -1, self.dim)

        # apply Transformer blocks
        z_hat = []
        for i in range(self.depth):
            # embed encoder dim to decoder dim
            if i == 0:
                # pad with mask tokens
                x = self.embed[i](x)
                aux_tokens = x[:, :self.num_aux_tokens, :]
                x = x[:, self.num_aux_tokens:, :]
                x = torch.cat([x, mask_tokens], dim=1)
                x = torch.gather(x, dim=1, index=ids_restore)
                x = x + self.pos_embed
                x = torch.cat([aux_tokens, x], dim=1)
            else:
                # compress encoder dim + decoder dim to decoder dim by only processing unmasked patches
                aux_tokens = x[:, :self.num_aux_tokens]
                x = x[:, self.num_aux_tokens:]
                visible_patches = torch.gather(x, dim=1, index=ids_shuffle_visible)
                masked_patches = torch.gather(x, dim=1, index=ids_shuffle_masked)
                visible_x = torch.concat([aux_tokens, visible_patches], dim=1)
                x = torch.concat([inputs[i], visible_x], dim=-1)
                x = self.embed[i](x)
                aux_tokens = x[:, :self.num_aux_tokens]
                x = x[:, self.num_aux_tokens:]
                x = torch.concat([x, masked_patches], dim=1)
                x = torch.gather(x, dim=1, index=ids_restore)
                x = torch.concat([aux_tokens, x], dim=1)
            x = self.blocks[i](x)
            pred = self.zhat_preds[i](x)
            z_hat.append(pred)
        # predict in reversed order (first decoder layer predicts last encoder latent)
        outputs["z_hat"] = list(reversed(z_hat))

        # optional additional block without multi-scale output
        x = self.last_block(x)

        # remove aux token
        x = x[:, self.num_aux_tokens:, :]

        # last layer
        x = self.xhat_pred(x)

        outputs["x_hat"] = x
        return outputs
