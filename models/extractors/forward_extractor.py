from .base.extractor_base import ExtractorBase
from .base.forward_hook import ForwardHook


class ForwardExtractor(ExtractorBase):
    def to_string(self):
        return "ForwardExtractor"

    def _get_own_outputs(self):
        return {"forward": self.outputs[f"forward"]["main"]}

    def _register_hooks(self, model):
        hook = ForwardHook(
            outputs=self.outputs,
            output_name="forward",
            raise_exception=self.raise_exception,
            **self.hook_kwargs,
        )
        model.register_forward_hook(hook)
        self.hooks.append(hook)
