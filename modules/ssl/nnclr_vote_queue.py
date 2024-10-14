import einops
import torch
import torch.nn.functional as F
from torch import nn

from distributed.gather import all_gather_nograd
from utils.loss_utils import apply_reduction
import kappamodules.utils.tensor_cache as tc

class NnclrVoteQueue(nn.Module):
    def __init__(self, size, dim, num_queues, swap_topk=1, voting_topk=None, candidates_mode="view0"):
        super().__init__()
        # checks
        if voting_topk is not None:
            assert voting_topk >= swap_topk
        # assign
        self.size = size
        self.dim = dim
        self.num_queues = num_queues
        self.swap_topk = swap_topk
        self.voting_topk = voting_topk
        self.candidates_mode = candidates_mode
        # queue properties
        self.register_buffer("x", torch.randn(size, num_queues, dim))
        self.register_buffer("idx", -torch.ones(size, dtype=torch.long))
        self.register_buffer("cls", -torch.ones(size, dtype=torch.long))
        self.register_buffer("age", torch.zeros(size, dtype=torch.long))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))

    def reset_parameters(self):
        torch.nn.init.normal_(self.x)
        torch.nn.init.constant_(self.idx, -1)
        torch.nn.init.constant_(self.cls, -1)
        torch.nn.init.zeros_(self.age)
        torch.nn.init.zeros_(self.ptr)

    @torch.no_grad()
    def forward(self, x, idx, cls=None, reduction="mean"):
        assert x.grad_fn is None
        assert x.ndim == 2
        batch_size = len(idx)
        assert len(x) % batch_size == 0
        num_views = len(x) // batch_size
        if self.training:
            assert num_views == self.num_queues

        # preprocess
        normed_x = F.normalize(x, dim=-1)
        normed_queue_x = F.normalize(self.x, dim=-1)

        # vote for best samples in queue
        if self.voting_topk is not None and self.voting_topk < self.size:
            rankings = []
            topk_qidxs = []
            is_same_sample = idx[:, None] == self.idx[None, :]
            for i in range(num_views):
                normed_x_chunk = normed_x[i * batch_size:(i + 1) * batch_size]
                for j in range(self.num_queues):
                    sim = normed_x_chunk @ normed_queue_x[:, j].T
                    # exclude self (the same sample from a previous iteration) -> important for small datasets
                    if idx is not None and self.training:
                        sim[is_same_sample] = -1.
                    # create ranking
                    sorted_sim, sorted_qidx = sim.topk(k=self.voting_topk, dim=1, largest=True, sorted=True)
                    arange = tc.arange(start=0, end=self.voting_topk, device=x.device, dtype=sim.dtype)
                    ranking = torch.zeros_like(sim).scatter_(
                        dim=1,
                        index=sorted_qidx,
                        src=arange.unsqueeze(0).expand(len(x), -1),
                    )
                    rankings.append(ranking)
                    # track for consistency metrics
                    if i == j:
                        topk_qidxs.append(sorted_qidx[:, :self.swap_topk])
            # accumulate votings
            rankings = torch.stack(rankings).sum(dim=0)
            rankings[rankings == 0] = float("inf")
            # select the best candiates
            topk_rankings, topk_qidx = rankings.topk(k=self.voting_topk, largest=False, sorted=True)
            if self.candidates_mode == "view0":
                # candidates are the best candidates from the view0-queue
                candidates = torch.gather(
                    self.x[:, 0].expand(batch_size, -1, -1),
                    dim=1,
                    index=topk_qidx.unsqueeze(-1).expand(-1, -1, self.dim),
                )
                candidates_idx = torch.gather(self.idx.expand(batch_size, -1), dim=1, index=topk_qidx)
                candidates_age = torch.gather(self.age.expand(batch_size, -1), dim=1, index=topk_qidx)
                candidates_cls = torch.gather(self.cls.expand(batch_size, -1), dim=1, index=topk_qidx)
            elif self.candidates_mode == "all":
                raise NotImplementedError
                # # candiates are the best candidates from all queues
                # candidates = torch.gather(
                #     self.x.unsqueeze(0).expand(batch_size, -1, -1, -1),
                #     dim=1,
                #     index=topk_qidx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.num_queues, self.dim),
                # )
                # candidates = einops.rearrange(
                #     candidates,
                #     "batch_size voting_topk num_queues dim -> batch_size (voting_topk num_queues) dim",
                # )
                # candidates_idx =
                # candidates_age =
                # candidates_cls =
            else:
                raise NotImplementedError
        else:
            # equivalent to normal NNCLR queue
            assert self.candidates_mode == "view0"
            candidates = self.x[:, 0].unsqueeze(0).expand(batch_size, -1, -1)
            topk_qidxs = None
            topk_rankings = None
            candidates_idx = self.idx
            candidates_age = self.age
            candidates_cls = self.cls

        # retrieve nearest neighbor from candidates
        candidates = einops.repeat(
            candidates,
            "batch_size voting_topk dim -> (num_views batch_size) voting_topk dim",
            num_views=num_views,
        )
        normed_candidates = F.normalize(candidates, dim=-1)
        sim = torch.einsum("bd,bkd->bk", normed_x, normed_candidates)
        topk_sim, topk_qidx = sim.topk(self.swap_topk, dim=1, largest=True)
        if self.swap_topk == 1:
            nn_qidx = topk_qidx.squeeze(1)
        else:
            choice = torch.randint(self.swap_topk, size=(len(topk_qidx), 1), device=topk_qidx.device)
            nn_qidx = torch.gather(topk_qidx, dim=1, index=choice).squeeze(1)
        nn_x = torch.gather(
            candidates,
            dim=1,
            index=nn_qidx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.dim),
        ).squeeze(1)

        # metrics
        metrics = {
            # similarity with nn
            "nn-similarity": F.cosine_similarity(x, nn_x, dim=-1),
            # age of nn (how many samples were processed since the nn was added to the queue)
            "nn-age": torch.gather(candidates_age.repeat(num_views, 1), dim=1, index=nn_qidx.unsqueeze(1)).squeeze(1),
        }
        # ranking statistics
        if topk_rankings is not None:
            ranking_min = topk_rankings.min(dim=1).values
            ranking_max = topk_rankings.max(dim=1).values
            metrics["nn-vote-ranking/topk/mean"] = topk_rankings.float().mean(dim=1)
            metrics["nn-vote-ranking/topk/min"] = ranking_min
            metrics["nn-vote-ranking/topk/max"] = ranking_max
            metrics["nn-vote-ranking/topk/delta"] = ranking_max - ranking_min
            metrics["nn-vote-ranking/top1"] = topk_rankings[:, 0]
            metrics[f"nn-vote-ranking/top{self.swap_topk}/"] = topk_rankings[:, self.swap_topk - 1]
            # consistency of overall ranking with in-view ranking
            # i.e. are the samples retrieved by voting the same ones as the ones retrieved without voting
            # requires mapping back index from candidate index [0, voting_topk) to sample indices [-1, len(ds))
            backmapped_idx = torch.gather(
                candidates_idx.repeat(num_views, 1),
                dim=1,
                index=topk_qidx,
            ).chunk(num_views)
            for i in range(num_views):
                cmp = backmapped_idx[i].unsqueeze(1) == topk_qidxs[i].unsqueeze(2)
                metrics[f"nn-vote-consistency/view{i}/idx"] = cmp.sum(dim=[1, 2]) / self.swap_topk
            # consistency in highest frequency class of NNs
            # one could also measure the real consistency over all NNs but this would be pretty expensive
            if cls is not None:
                # retrieve cls of the topk candidates
                voted_cls = torch.gather(
                    candidates_cls.repeat(num_views, 1),
                    dim=1,
                    index=topk_qidx,
                )
                # select highest frequency sample
                voted_cls = voted_cls.mode(dim=1).values.chunk(num_views)
                for i in range(num_views):
                    unvoted_cls = torch.gather(
                        self.cls.expand(batch_size, -1),
                        dim=1,
                        index=topk_qidxs[i],
                    ).mode(dim=1).values
                    metrics[f"nn-vote-consistency/view{i}/cls"] = voted_cls[i] == unvoted_cls
        # nn-accuracy
        if cls is not None:
            metrics["nn-accuracy"] = torch.gather(
                candidates_cls.repeat(num_views, 1),
                dim=1,
                index=nn_qidx.unsqueeze(1),
            ) == cls
        metrics = {k: apply_reduction(v, reduction=reduction) for k, v in metrics.items()}

        # shift
        if self.training:
            x = einops.rearrange(x, "(num_views batch_size) dim -> batch_size num_views dim", num_views=num_views)
            x = all_gather_nograd(x)

            if idx is not None:
                idx = all_gather_nograd(idx)
            if cls is not None:
                cls = all_gather_nograd(cls)
            ptr_from = int(self.ptr)
            ptr_to = ptr_from + len(x)
            overflow = ptr_to - self.size
            if overflow > 0:
                # replace end-of-queue
                self.x[ptr_from:] = x[:-overflow]
                self.age[ptr_from:] = 0
                if idx is not None:
                    self.idx[ptr_from:] = idx[:-overflow]
                if cls is not None:
                    self.cls[ptr_from:] = cls[:-overflow]
                # replace start-of-queue
                self.x[:, :overflow] = x[-overflow:]
                self.age[:overflow] = 0
                if idx is not None:
                    self.idx[:overflow] = idx[-overflow:]
                if cls is not None:
                    self.cls[:overflow] = cls[-overflow:]
                # update pointer
                self.ptr[0] = overflow
            else:
                self.x[ptr_from:ptr_to] = x
                self.age[ptr_from:ptr_to] = 0
                if idx is not None:
                    self.idx[ptr_from:ptr_to] = idx
                if cls is not None:
                    self.cls[ptr_from:ptr_to] = cls
                self.ptr[0] = ptr_to % self.size
            self.age += len(x)

        return nn_x, metrics
