import torch
from torch import nn

class SoftPseudoSampler(nn.Module):
    @staticmethod
    def forward(preds):
        assert preds.ndim == 2
        return torch.multinomial(preds, num_samples=1).squeeze(1)