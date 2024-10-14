import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans


@torch.no_grad()
def silhouette_score(
        x,
        num_classes,
        y=None,
        cls_batch_size=1,
        batch_size=32,
        distance="cosine",
        batch_normalize=True,
        num_kmeans_runs=10,
        eps=1e-6,
):
    # apply batch normalization
    if batch_normalize:
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True) + eps
        x = (x - mean) / std

    # kmeans clustering
    if y is None:
        xcpu = x.cpu().numpy()
        best_kmeans = None
        num_classes = min(len(x), num_classes)
        for i in range(num_kmeans_runs):
            kmeans = MiniBatchKMeans(n_clusters=num_classes, max_iter=1000, n_init=10)
            kmeans.fit(xcpu)
            if best_kmeans is None or kmeans.inertia_ < best_kmeans.inertia_:
                best_kmeans = kmeans
        y = best_kmeans.predict(xcpu)
        y = torch.from_numpy(y).long().to(x.device)

    # count class labels
    class_counts = torch.zeros(num_classes, dtype=torch.long, device=x.device)
    one_hot = F.one_hot(y, num_classes=num_classes)
    class_counts += one_hot.sum(dim=0)

    # handle missing class
    if torch.any(class_counts == 0):
        return torch.tensor(-1.), torch.tensor(-1.), torch.tensor(-1.)

    if distance == "cosine":
        # normalize for cosine distance
        x = F.normalize(x, dim=-1)
        # cosine distance
        distances = 1 - x @ x.T
    elif distance == "euclidean":
        # calculate in chunks to avoid OOM
        num_chunks = (len(x) + batch_size - 1) // batch_size
        distances = torch.concat([
            torch.sqrt(((chunk.unsqueeze(1) - x.unsqueeze(0)) ** 2).sum(dim=-1))
            for chunk in x.chunk(num_chunks)
        ])
    else:
        raise NotImplementedError
    # sklearn explicitly sets diagonal to 0 (due to floating point errors they are typically small values)
    distances.masked_fill_(torch.eye(len(distances), dtype=torch.bool, device=x.device), 0)
    # calculate distances but in a memory efficient way
    # memory inefficient: clust_dists = (one_hot.unsqueeze(1) * distances.unsqueeze(-1)).sum(dim=0)
    clust_dists = torch.zeros(len(x), num_classes, device=x.device)
    num_chunks = (num_classes + cls_batch_size - 1) // cls_batch_size
    for i in range(num_chunks):
        start = i * cls_batch_size
        end = (i + 1) * cls_batch_size
        clust_dists[:, start:end] = (one_hot[:, start:end].unsqueeze(1) * distances.unsqueeze(-1)).sum(dim=0)

    # calculate intra cluster distances
    intra_clust_dists = torch.gather(clust_dists, dim=1, index=y.unsqueeze(-1))
    intra_clust_dists = intra_clust_dists.squeeze(1) / (class_counts - 1).take(y)
    # calculate inter cluster distances
    inter_clust_dists = clust_dists.masked_fill_(mask=one_hot.bool(), value=float("inf")) / class_counts
    inter_clust_dists = inter_clust_dists.min(dim=1).values
    # calculate score
    scores = (inter_clust_dists - intra_clust_dists) / torch.maximum(intra_clust_dists, inter_clust_dists)

    return scores.mean(), inter_clust_dists.mean(), intra_clust_dists.mean()


@torch.no_grad()
def max_distance_to_centroid(x, y, num_classes, distance="cosine"):
    # check args
    assert x.ndim == 2
    assert y.ndim == 1
    assert len(x) == len(y)
    assert 0 < num_classes

    # calculate max distances
    max_dists = torch.zeros(num_classes)
    for i in range(num_classes):
        is_cur_class = y == i
        if is_cur_class.sum() == 0:
            continue
        class_features = x[is_cur_class]
        centroid = class_features.mean(dim=0, keepdim=True)
        class_features = class_features
        if distance == "cosine":
            max_dists[i] = (1 - F.cosine_similarity(class_features, centroid)).max()
        elif distance == "euclidean":
            max_dists[i] = (1 - F.pairwise_distance(class_features, centroid)).max()
        else:
            raise NotImplementedError

    return max_dists
