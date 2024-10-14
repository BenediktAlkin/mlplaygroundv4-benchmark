from models import model_from_kwargs
from models.base.composite_model_base import CompositeModelBase
from models.extractors.vit_block_extractor import VitBlockExtractor
from utils.factory import create


class MaeMsg(CompositeModelBase):
    def __init__(self, encoder, decoder, msg_spacing="uniform", **kwargs):
        super().__init__(**kwargs)
        # encoder
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
        # decoder
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
        # create and register extractors
        if msg_spacing == "last":
            block_indices = list(range(self.encoder.depth - self.decoder.depth, self.encoder.depth))
        elif msg_spacing == "uniform":
            if self.encoder.depth % self.decoder.depth == 0:
                # uniform spacing (e.g. with 24 blocks and depth8 [2, 5, 8, 11, 14, 17, 20, 23])
                interval = self.encoder.depth // self.decoder.depth
                block_indices = list(range(interval - 1, self.encoder.depth, interval))
            elif self.encoder.depth == 12 and self.decoder.depth == 8:
                # semi-uniform spacing by skipping every 3rd block
                block_indices = [1, 2, 4, 5, 7, 8, 10, 11]
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        assert len(block_indices) == self.decoder.depth
        self.logger.info(f"attaching decoder layers to encoder layers {block_indices}")
        self.encoder_extractor = VitBlockExtractor(
            block_indices=block_indices,
            use_next_norm=False,
            finalizer=None,
        )
        self.encoder_extractor.register_hooks(self.encoder)

    @property
    def submodels(self):
        return dict(encoder=self.encoder, decoder=self.decoder)

    def forward(self, x, mask_generator=None, idx=None):
        outputs = {}
        if mask_generator is None:
            # no mask generator -> unmasked encoder forward pass
            assert not self.training
            _ = self.encoder(x)
            return outputs

        # encoder forward
        encoder_output = self.encoder(
            x,
            idx=idx,
            mask_generator=mask_generator,
        )
        outputs["mask"] = encoder_output["mask"]
        intermediate_outputs = self.encoder_extractor.extract()

        # decoder forward
        decoder_outputs = self.decoder(
            intermediate_outputs,
            ids_restore=encoder_output["ids_restore"],
        )
        outputs["prediction"] = decoder_outputs["x_hat"]
        outputs["patch_size"] = self.encoder.patch_size

        return outputs
