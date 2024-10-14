import torch.nn.functional as F
from copy import deepcopy
from functools import partial

import einops
import torch
from kappaschedules import object_to_schedule
from models.extractors.base.without_extractor_hooks import without_extractor_hooks

from initializers import initializer_from_kwargs
from models import model_from_kwargs, prepare_momentum_kwargs
from models.base.composite_model_base import CompositeModelBase
from models.extractors.base.forward_hook import StopForwardException
from models.extractors.vit_block_extractor import VitBlockExtractor
from kappamodules.layers import DropPath
from utils.factory import create
from utils.model_utils import update_ema
from utils.schedule_utils import get_value_or_default
from models.extractors.finalizers.stack_finalizer import StackFinalizer


class MaeMsmimMsgdec(CompositeModelBase):
    def __init__(
            self,
            encoder,
            decoder,
            target_factor,
            average_target_blocks=False,
            target_factor_schedule=None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.average_target_blocks = average_target_blocks
        self.encoder = create(
            encoder,
            model_from_kwargs,
            input_shape=self.input_shape,
            update_counter=self.update_counter,
            path_provider=self.path_provider,
            dynamic_ctx=self.dynamic_ctx,
            static_ctx=self.static_ctx,
            data_container=self.data_container,
            use_last_norm=False,
        )
        assert self.encoder.output_shape is not None
        self.decoder = create(
            decoder,
            model_from_kwargs,
            input_shape=self.encoder.output_shape,
            update_counter=self.update_counter,
            path_provider=self.path_provider,
            dynamic_ctx=self.dynamic_ctx,
            static_ctx=self.static_ctx,
            data_container=self.data_container,
        )
        # EMA
        self.target_factor = target_factor
        self.target_factor_schedule = object_to_schedule(
            target_factor_schedule,
            batch_size=self.update_counter.effective_batch_size if self.update_counter is not None else None,
            updates_per_epoch=self.update_counter.updates_per_epoch if self.update_counter is not None else None,
            start_value=target_factor,
        )
        assert isinstance(encoder, (dict, partial))
        momentum_encoder = prepare_momentum_kwargs(encoder)
        if isinstance(encoder, dict) and len(momentum_encoder) == 0:
            # initialize momentum_encoder via checkpoint_kwargs of encoder
            assert "initializers" in encoder and encoder["initializers"][0].get("use_checkpoint_kwargs", False)
            initializer_kwargs = deepcopy(encoder["initializers"][0])
            initializer_kwargs.pop("use_checkpoint_kwargs")
            initializer = initializer_from_kwargs(**initializer_kwargs, path_provider=self.path_provider)
            momentum_encoder = initializer.get_model_kwargs()
        self.momentum_encoder = create(
            momentum_encoder,
            model_from_kwargs,
            input_shape=self.input_shape,
            update_counter=self.update_counter,
            path_provider=self.path_provider,
            dynamic_ctx=self.dynamic_ctx,
            static_ctx=self.static_ctx,
            is_frozen=True,
        )
        # disable drop_path in momentum_encoder: momentum_encoder is kept in train mode to
        # track batchnorm stats (following MoCoV3) -> drop_path would be applied in forward pass
        self.logger.info(f"disabling DropPath for momentum_encoder")
        for m in self.momentum_encoder.modules():
            if isinstance(m, DropPath):
                m.drop_prob = 0.

        # extractors
        if self.encoder.depth % self.decoder.depth == 0:
            # uniform spacing (e.g. with 24 blocks and depth8 [2, 5, 8, 11, 14, 17, 20, 23])
            interval = self.encoder.depth // self.decoder.depth
            block_indices = list(range(interval - 1, self.encoder.depth, interval))
        elif self.encoder.depth == 12 and self.decoder.depth == 8:
            # semi-uniform spacing by skipping every 3rd block
            block_indices = [1, 2, 4, 5, 7, 8, 10, 11]
        else:
            raise NotImplementedError
        self.logger.info(f"attaching decoder layers to encoder layers {block_indices}")
        assert len(block_indices) == self.decoder.depth
        # encoder extractor
        self.encoder_extractor = VitBlockExtractor(
            block_indices=block_indices,
            use_next_norm=False,
            finalizer=None,
        )
        self.encoder_extractor.register_hooks(self.encoder)
        # momentum_encoder extractor
        if average_target_blocks:
            # target representation is the average of the block outputs since the last target
            # e.g. 24 encoder blocks, 8 decoder blocks -> target[0] = block_outputs[0:3].mean()
            assert self.encoder.depth % self.decoder.depth == 0
            momentum_block_indices = None
        else:
            momentum_block_indices = block_indices
        self.momentum_encoder_extractor = VitBlockExtractor(
            block_indices=momentum_block_indices,
            use_next_norm=False,
            finalizer=partial(StackFinalizer, dim=1),
            raise_exception=True,
        )
        self.momentum_encoder_extractor.register_hooks(self.momentum_encoder)

    @property
    def submodels(self):
        return dict(
            encoder=self.encoder,
            decoder=self.decoder,
            momentum_encoder=self.momentum_encoder,
        )

    # noinspection PyMethodOverriding
    def forward(self, x, mask_generator=None, num_masks=None, idx=None):
        if mask_generator is None:
            # no mask generator -> unmasked encoder forward pass
            assert not self.training
            assert num_masks is None
            with without_extractor_hooks([self.encoder_extractor]):
                _ = self.encoder(x)
            with without_extractor_hooks([self.momentum_encoder_extractor]):
                _ = self.momentum_encoder(x)
            return {}
        assert num_masks is not None and len(x) % num_masks == 0
        outputs = {}

        # EMA forward
        batch_size = len(x) // num_masks
        with torch.no_grad():
            try:
                # x can be larger than batch_size if num_views > 1
                _ = self.momentum_encoder(x[:batch_size])
                raise RuntimeError("expected StopForwardException")
            except StopForwardException:
                z = self.momentum_encoder_extractor.extract()
            if self.average_target_blocks:
                # instance norm
                z = einops.rearrange(z, "bs enc_depth seqlen dim -> bs enc_depth dim seqlen")
                z = F.instance_norm(z)
                z = einops.rearrange(z, "bs enc_depth dim seqlen -> bs enc_depth seqlen dim")
                # average
                z = einops.rearrange(
                    z,
                    "bs (dec_depth num_blocks_to_avg) seqlen dim -> bs dec_depth num_blocks_to_avg seqlen dim",
                    dec_depth=self.decoder.depth,
                ).mean(dim=2)
            if num_masks > 1:
                z = einops.repeat(
                    z,
                    "batch_size ... -> (num_masks batch_size) ...",
                    num_masks=num_masks,
                )
            outputs["z"] = z

        # encoder forward
        outputs["patch_size"] = self.encoder.patch_size
        encoder_output = self.encoder(
            x,
            idx=idx,
            mask_generator=mask_generator,
        )
        ids_restore = encoder_output["ids_restore"]
        unmasked_z_hat = self.encoder_extractor.extract()
        # pad to have same seqlen as full image
        num_masked_patches = ids_restore.size(1) - unmasked_z_hat[0].size(1) + self.encoder.num_aux_tokens
        padding = torch.zeros(len(unmasked_z_hat[0]), num_masked_patches, self.encoder.dim, device=x.device)
        padded_unmasked_z_hat = []
        for i in range(len(unmasked_z_hat)):
            cur = unmasked_z_hat[i]
            aux_tokens = cur[:, :self.encoder.num_aux_tokens, :]
            # pad sequence
            cur = torch.cat([cur, padding], dim=1)
            # unshuffle
            cur = torch.gather(cur, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, self.encoder.dim))
            # append aux tokens
            cur = torch.cat([aux_tokens, cur], dim=1)
            # add to new list
            padded_unmasked_z_hat.append(cur)
        outputs["unmasked_z_hat"] = padded_unmasked_z_hat
        outputs["mask"] = encoder_output["mask"]

        # decoder forward
        decoder_output = self.decoder(
            unmasked_z_hat,
            ids_restore=ids_restore,
        )
        outputs["x_hat"] = decoder_output["x_hat"]
        outputs["masked_z_hat"] = decoder_output["z_hat"]

        return outputs

    # this is typically not done
    # def model_specific_initialization(self):
    #     self.logger.info("initializing teacher_model with parameters from model")
    #     copy_params(self.encoder, self.momentum_encoder)
    #     super().model_specific_initialization()

    def after_update_step(self):
        target_factor = get_value_or_default(
            default=self.target_factor,
            schedule=self.target_factor_schedule,
            update_counter=self.update_counter,
        )
        # MoCoV3 tracks batchnorm stats from the ema model instead of copying it from the source model
        update_ema(self.encoder, self.momentum_encoder, target_factor, copy_buffers=False)
