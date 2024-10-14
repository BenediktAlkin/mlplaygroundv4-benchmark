from collections import defaultdict

import einops
import torch
from torch import nn

from models import model_from_kwargs
from models.base.composite_model_base import CompositeModelBase
from models.poolings.base.handle_extractor_pooling import handle_extractor_pooling
from utils.factory import create, create_collection
from models.pseudo_sampler import pseudo_sampler_from_kwargs

class ContrastiveProbeModel(CompositeModelBase):
    def __init__(
            self,
            encoder,
            probes,
            heads,
            pseudo_sampler,
            **kwargs,
    ):
        super().__init__(**kwargs)
        #
        self.encoder = create(
            encoder,
            model_from_kwargs,
            input_shape=self.input_shape,
            update_counter=self.update_counter,
            path_provider=self.path_provider,
            dynamic_ctx=self.dynamic_ctx,
            static_ctx=self.static_ctx,
        )
        assert self.encoder.output_shape is not None
        self.probes = nn.ModuleDict(
            create_collection(
                probes,
                model_from_kwargs,
                input_shape=self.encoder.output_shape,
                output_shape=self.output_shape,
                update_counter=self.update_counter,
                path_provider=self.path_provider,
                dynamic_ctx=self.dynamic_ctx,
                static_ctx=self.static_ctx,
                data_container=self.data_container,
            ),
        )
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
        # sampler
        self.pseudo_sampler = create(pseudo_sampler, pseudo_sampler_from_kwargs)

        # register pooling hooks (required for ExtractorPooling)
        for probe in self.probes.values():
            probe.pooling.register_hooks(self.encoder)
        for head in self.heads.values():
            head.pooling.register_hooks(self.encoder)

    @property
    def submodels(self):
        return dict(
            encoder=self.encoder,
            **{f"probes.{key}": value for key, value in self.probes.items()},
            **{f"heads.{key}": value for key, value in self.heads.items()},
        )

    # noinspection PyMethodOverriding
    def forward(self, x, idx, cls, mask_generator=None):
        forward_kwargs = {}
        if mask_generator is not None:
            assert len(x) == 1, "multi-crop with masking not supported (should maybe exclude masking in local views)"
            forward_kwargs["mask_generator"] = mask_generator
            if idx is not None:
                forward_kwargs["idx"] = idx

        # split into student/teacher
        student_x = x
        teacher_x = x[0]
        batch_size = len(idx)
        assert len(teacher_x) % len(idx) == 0
        num_teacher_views = len(teacher_x) // batch_size

        # forward student encoder
        poolings = [head.pooling for head in self.heads.values()]
        with handle_extractor_pooling(poolings):
            encoder_outputs = defaultdict(list)
            for i, xx in enumerate(student_x):
                # encoder forward
                encoder_output = self.encoder(xx, **forward_kwargs)["main"]
                # pool
                for head in self.heads.values():
                    # only add if it wasn't already added (multiple heads can have the same pooling)
                    if len(encoder_outputs[head.pooling]) == i:
                        encoder_outputs[head.pooling].append(head.pooling(encoder_output))
                for labeler in self.probes.values():
                    # only add if it wasn't already added (multiple labeler can have the same pooling)
                    if len(encoder_outputs[labeler.pooling]) == i:
                        encoder_outputs[labeler.pooling].append(labeler.pooling(encoder_output))
        # concat outputs
        encoder_outputs = {pooling: torch.concat(outputs) for pooling, outputs in encoder_outputs.items()}
        
        # forward probes
        outputs = {}
        for name, probe in self.probes.items():
            outputs[f"probes.{name}"] = probe(
                x=encoder_outputs[probe.pooling],
                apply_pooling=False,
            )
        # predict labels
        headname_to_cls = {}
        if len(self.probes) == 1:
            # one prediction for all heads
            preds = list(outputs.values())[0][:batch_size * num_teacher_views].softmax(dim=-1)
            preds = einops.rearrange(
                preds,
                "(num_teacher_views batch_size) num_classes -> num_teacher_views batch_size num_classes",
                num_teacher_views=num_teacher_views,
            ).mean(dim=0)
            cls = self.pseudo_sampler(preds)
            for name in self.heads.keys():
                headname_to_cls[name] = cls
        else:
            # predict per head
            for probename in self.probes.keys():
                preds = outputs[f"probes.{probename}"][:batch_size * num_teacher_views].softmax(dim=-1)
                preds = einops.rearrange(
                    preds,
                    "(num_teacher_views batch_size) num_classes -> num_teacher_views batch_size num_classes",
                    num_teacher_views=num_teacher_views,
                ).mean(dim=0)
                cls = self.pseudo_sampler(preds)
                headname_to_cls[probename] = cls
        # forward heads
        for name, head in self.heads.items():
            outputs[f"heads.{name}"] = head(
                x=encoder_outputs[head.pooling],
                idx=idx,
                cls=headname_to_cls[name],
                batch_size=len(idx),
                num_teacher_views=num_teacher_views,
                apply_pooling=False,
            )
        return outputs
