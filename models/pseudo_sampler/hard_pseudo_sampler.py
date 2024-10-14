from torch import nn

class HardPseudoSampler(nn.Module):
    @staticmethod
    def forward(preds):
        assert preds.ndim == 2
        return preds.argmax(dim=-1)