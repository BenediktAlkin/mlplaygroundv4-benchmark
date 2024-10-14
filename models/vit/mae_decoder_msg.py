from functools import partial

import einops
import numpy as np
import torch
from kappamodules.functional.pos_embed import get_sincos_pos_embed_from_seqlens
from kappamodules.init import init_xavier_uniform_zero_bias, init_norm_as_noaffine
from kappamodules.vit import VitBlock, VitSeperateNorm
from torch import nn

from models.base.single_model_base import SingleModelBase
from optimizers.param_group_modifiers.exclude_from_wd_by_name_modifier import ExcludeFromWdByNameModifier


class MaeDecoderMsg(SingleModelBase):
    def __init__(
            self,
            dim,
            depth,
            num_attn_heads,
            use_seperate_norm=False,
            eps=1e-6,
            init_weights="xavier_uniform",
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = self.static_ctx["patch_size"]
        encoder_input_shape = self.static_ctx["input_shape"]
        assert len(self.patch_size) == len(encoder_input_shape) - 1
        self.dim = dim
        self.depth = depth
        self.num_attn_heads = num_attn_heads
        self.eps = eps

        # decoder doesn't produce original image shape but flattened patches
        num_channels = self.static_ctx["input_shape"][0]
        self.patch_numel = int(np.prod(self.patch_size)) * num_channels
        num_tokens, encoder_dim = self.input_shape
        self.output_shape = (num_tokens, self.patch_numel)
        self.num_aux_tokens = self.static_ctx["num_aux_tokens"]

        # embed
        self.embed = nn.ModuleList([
            nn.Linear(encoder_dim if i == 0 else encoder_dim + dim, dim)
            for i in range(depth)
        ])
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))

        # fixed pos embedding
        self.seqlens = self.static_ctx["sequence_lengths"]
        pos_embed = get_sincos_pos_embed_from_seqlens(seqlens=self.seqlens, dim=dim)
        self.register_buffer("pos_embed", einops.rearrange(pos_embed, "... dim -> 1 (...) dim"))

        # norm ctor
        if use_seperate_norm:
            norm_ctor = partial(VitSeperateNorm, num_aux_tokens=self.num_aux_tokens)
        else:
            norm_ctor = nn.LayerNorm

        # blocks
        self.blocks = nn.ModuleList([
            VitBlock(
                dim=dim,
                num_heads=num_attn_heads,
                norm_ctor=norm_ctor,
                eps=eps,
                init_weights=init_weights,
            )
            for _ in range(depth)
        ])

        # decoder to patch
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.pred = nn.Linear(dim, self.patch_numel)

    def model_specific_initialization(self):
        # mask token
        torch.nn.init.normal_(self.mask_token, std=.02)
        # layers
        init_xavier_uniform_zero_bias(self.embed)
        init_xavier_uniform_zero_bias(self.pred)
        # norms
        init_norm_as_noaffine(self.norm)

    @staticmethod
    def get_model_specific_param_group_modifiers():
        return [ExcludeFromWdByNameModifier(name="mask_token")]

    # noinspection PyMethodOverriding
    def forward(self, x, ids_restore):
        assert len(x) == self.depth
        outputs = {}
        inputs = list(reversed(x))
        x = inputs[0]

        # extract shapes
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

        # apply blocks
        for i in range(self.depth):
            if i == 0:
                # default decoder embedding
                x = self.embed[i](x)
                all_patches = torch.cat([x[:, self.num_aux_tokens:, :], mask_tokens], dim=1)
                all_patches = torch.gather(all_patches, dim=1, index=ids_restore)
                all_patches = all_patches + self.pos_embed
                x = torch.cat([x[:, :self.num_aux_tokens, :], all_patches], dim=1)
            else:
                # embedding with skip connection from encoder tokens
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

        # to rgb
        x = x[:, self.num_aux_tokens:]
        x = self.norm(x)
        x = self.pred(x)

        outputs["x_hat"] = x
        return outputs
