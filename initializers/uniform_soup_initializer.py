from initializers import initializer_from_kwargs
from utils.factory import create_collection
from .base.checkpoint_initializer import CheckpointInitializer
from .base.initializer_base import InitializerBase


class UniformSoupInitializer(InitializerBase):
    def __init__(self, initializers, path_provider, **kwargs):
        super().__init__(path_provider=path_provider, **kwargs)
        self.initializers = create_collection(
            initializers,
            initializer_from_kwargs,
            path_provider=path_provider,
        )
        assert all(isinstance(init, CheckpointInitializer) and not init.load_optim for init in self.initializers)

    def _get_model_kwargs(self):
        return self.initializers[0].get_model_kwargs()

    def init_weights(self, model):
        # load state_dicts
        state_dicts, model_names, ckpt_uris = zip(*[init.get_model_state_dict(model) for init in self.initializers])
        # create uniform soup
        soup_sd = {}
        for state_dict in state_dicts:
            for k, v in state_dict.items():
                if k not in soup_sd:
                    soup_sd[k] = v / len(state_dicts)
                else:
                    soup_sd[k] += v / len(state_dicts)
        # load uniform soup
        model.load_state_dict(soup_sd)
        self.logger.info(f"loaded model_soup of {len(state_dicts)} models:")
        for model_name, ckpt_uri in zip(model_names, ckpt_uris):
            self.logger.info(f"- {model_name}: {ckpt_uri}")
