from kappadata.wrappers import KDPseudoLabelWrapper as KDPLW


class KDPseudoLabelWrapper(KDPLW):
    def __init__(self, dataset, uri, **kwargs):
        uri = dataset.path_provider.output_path / uri
        super().__init__(dataset=dataset, uri=uri, **kwargs)
