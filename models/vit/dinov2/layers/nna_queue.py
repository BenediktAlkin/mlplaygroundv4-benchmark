import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist

class NnaQueue(nn.Module):
    def __init__(self, size, dim, is_student, topk=1):
        super().__init__()
        self.size = size
        self.dim = dim
        self.is_student = is_student
        self.topk = topk
        # queue properties
        self.register_buffer("x", F.normalize(torch.randn(size, dim), dim=-1))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def forward(self, x):
        if self.is_student:
            return x

        assert x.ndim == 2
        x = x.detach()

        # NOTE x is already normalized
        # x = F.normalize(x, dim=-1)
        # since x is normaled -> store also normalized values in queue
        # normed_queue_x = F.normalize(self.x, dim=-1)

        # calculate similarities
        queue_x = self.x.to(x.dtype)
        sim = x @ queue_x.T
        # retrieve neighbor(s)
        _, topk_qidx = sim.topk(self.topk, dim=1, largest=True)
        if self.topk == 1:
            nn_qidx = topk_qidx.squeeze(1)
        else:
            choice = torch.randint(self.topk, size=(len(topk_qidx), 1), device=topk_qidx.device)
            nn_qidx = torch.gather(topk_qidx, dim=1, index=choice).squeeze(1)
        nn_x = queue_x[nn_qidx].to(x.dtype)

        # shift fifo queue
        # NOTE: hardcoded to 2 views
        batch_size = len(x) // 2
        x_view0 = x[:batch_size]
        if dist.is_initialized() and dist.get_world_size() > 1:
            # TODO
            # x_view0 = all_gather_nograd(x_view0)
            raise NotImplementedError
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

        return nn_x
