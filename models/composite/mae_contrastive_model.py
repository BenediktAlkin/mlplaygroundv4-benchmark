from collections import defaultdict

import torch
from torch import nn

from models import model_from_kwargs
from models.base.composite_model_base import CompositeModelBase
from models.poolings.base.handle_extractor_pooling import handle_extractor_pooling
from models.ssl.mugs_head import MugsHead
from utils.factory import create, create_collection


class MaeContrastiveModel(CompositeModelBase):
    def __init__(self, encoder, decoder, heads, **kwargs):
        super().__init__(**kwargs)
        self.encoder = create(
            encoder,
            model_from_kwargs,
            input_shape=self.input_shape,
            update_counter=self.update_counter,
            path_provider=self.path_provider,
            dynamic_ctx=self.dynamic_ctx,
            static_ctx=self.static_ctx,
            data_container=self.data_container,
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

        # contrastive heads
        self.heads = nn.ModuleDict(
            create_collection(
                heads,
                model_from_kwargs,
                input_shape=self.encoder.output_shape,
                update_counter=self.update_counter,
                path_provider=self.path_provider,
                dynamic_ctx=self.dynamic_ctx,
                static_ctx=self.static_ctx,
                data_container=self.data_container,
            ),
        )
        # register pooling hooks (required for ExtractorPooling)
        for head in self.heads.values():
            head.pooling.register_hooks(self.encoder)

    @property
    def submodels(self):
        return dict(
            encoder=self.encoder,
            decoder=self.decoder,
            **{f"heads.{key}": value for key, value in self.heads.items()},
        )

    def forward(self, x):
        assert not self.training
        return self.encoder(x)

    def forward_mae(self, x, mask_generator=None, idx=None):
        outputs = {}
        if mask_generator is None:
            # no mask generator -> unmasked encoder forward pass
            assert not self.training
            _ = self.encoder(x)

        # encoder forward
        encoder_output = self.encoder(
            x,
            idx=idx,
            mask_generator=mask_generator,
        )
        outputs["mask"] = encoder_output["mask"]

        # decoder forward
        decoder_output = self.decoder(
            encoder_output["main"],
            ids_restore=encoder_output["ids_restore"],
        )
        outputs["prediction"] = decoder_output["main"]
        outputs["patch_size"] = self.encoder.patch_size

        return outputs

    def forward_contrastive(self, x, idx=None):
        assert torch.is_tensor(x)

        # forward encoder
        poolings = [head.pooling for head in self.heads.values()]
        with handle_extractor_pooling(poolings):
            encoder_outputs = defaultdict(list)
            # encoder forward
            encoder_output = self.encoder(x)["main"]
            # pool
            for head in self.heads.values():
                # only add if it wasn't already added (multiple heads can have the same pooling)
                if len(encoder_outputs[head.pooling]) == 0:
                    encoder_outputs[head.pooling].append(head.pooling(encoder_output))
        # concat outputs
        encoder_outputs = {pooling: torch.concat(outputs) for pooling, outputs in encoder_outputs.items()}

        # forward student heads
        head_outputs = {}
        for name, head in self.heads.items():
            if isinstance(head, MugsHead):
                assert idx is not None
                head_outputs[name] = head(
                    encoder_outputs[head.pooling],
                    batch_size=len(idx),
                    is_weak_aug=True,
                    apply_pooling=False,
                )
            else:
                head_outputs[name] = head(encoder_outputs[head.pooling], apply_pooling=False)
        # unpack predicted (head_outputs is projected/predicted tuple)
        predicted = {name: head_outputs[1] for name, head_outputs in head_outputs.items()}
        # unpack projected
        projected = {
            name: head_outputs[0].detach()
            for name, head_outputs in head_outputs.items()
        }
        return projected, predicted
