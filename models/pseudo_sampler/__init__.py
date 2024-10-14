from copy import deepcopy

from utils.factory import instantiate


def pseudo_sampler_from_kwargs(kind, **kwargs):
    kwargs = deepcopy(kwargs)
    return instantiate(
        module_names=[f"models.pseudo_sampler.{kind}"],
        type_names=[kind],
        **kwargs,
    )
