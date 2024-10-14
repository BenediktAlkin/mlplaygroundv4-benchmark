import torch
import torch.nn.functional as F
from torch import nn

from distributed.gather import all_gather_nograd


class MugsQueue(nn.Module):
    def __init__(self, size, dim, topk):
        super().__init__()
        self.size = size
        self.dim = dim
        self.topk = topk
        # queue properties
        self.register_buffer("x", torch.randn(size, dim))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))

    def reset_parameters(self):
        nn.init.normal_(self.x)
        nn.init.zeros_(self.ptr)

    @torch.no_grad()
    def forward(self, x, batch_size=None, is_weak_aug=None):
        assert x.ndim == 2
        x = x.detach()

        # lookup
        normed_x = F.normalize(x, dim=-1)
        normed_queue_x = F.normalize(self.x, dim=-1)
        sim = normed_x @ normed_queue_x.T
        _, topk_sim_idx = sim.topk(self.topk, dim=1)
        neighbors = self.x[topk_sim_idx.flatten()].view(len(x), self.topk, self.dim)

        # shift
        if self.training:
            assert is_weak_aug is not None
            assert batch_size is not None
            x_view0 = x[:batch_size]
            x_view0 = all_gather_nograd(x_view0)
            if torch.is_tensor(is_weak_aug):
                # student views can be strongly augmented
                is_weak_aug = all_gather_nograd(is_weak_aug)
                x_view0 = x_view0[is_weak_aug]
            else:
                # teacher view is always weak_aug (is_weak_aug is bool)
                assert isinstance(is_weak_aug, bool) and is_weak_aug
            ptr_from = int(self.ptr)
            ptr_to = ptr_from + len(x_view0)
            overflow = ptr_to - self.size
            if overflow > 0:
                # replace end-of-queue
                self.x[ptr_from:] = x_view0[:-overflow]
                # replace start-of-queue
                self.x[:overflow] = x_view0[-overflow:]
                # update pointer
                self.ptr[0] = overflow
            else:
                self.x[ptr_from:ptr_to] = x_view0
                self.ptr[0] = ptr_to % self.size

        return neighbors
