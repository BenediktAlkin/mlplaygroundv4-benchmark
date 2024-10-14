import torch
from torch import nn

class TopkPseudoSampler(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, preds):
        assert preds.ndim == 2
        _, topk_idx = preds.topk(k=self.k, largest=True)
        choice = torch.randint(self.k, size=(len(topk_idx),), device=preds.device)
        cls = torch.gather(topk_idx, dim=1, index=choice.unsqueeze(1)).squeeze(1)
        return cls