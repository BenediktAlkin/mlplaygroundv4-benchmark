import numpy as np
import scipy
import torch
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm


@torch.no_grad()
def kmeans_accuracy(
        x,
        y,
        batch_normalize=True,
        num_kmeans_runs=10,
        kmeans_max_iter=1000,
        kmeans_n_init=10,
        eps=1e-6,
):
    # apply batch normalization
    if batch_normalize:
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True) + eps
        x = (x - mean) / std

    # create a good kmeans clustering
    xcpu = x.cpu().numpy()
    best_kmeans = None
    num_classes = y.max().item() + 1
    for _ in tqdm(range(num_kmeans_runs)):
        kmeans = MiniBatchKMeans(n_clusters=num_classes, max_iter=kmeans_max_iter, n_init=kmeans_n_init)
        kmeans.fit(xcpu)
        if best_kmeans is None or kmeans.inertia_ < best_kmeans.inertia_:
            best_kmeans = kmeans
    # predict labels
    y_hat = best_kmeans.predict(xcpu)
    assert y.shape == y_hat.shape
    # match to actual labels
    match_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for i in range(y.shape[0]):
        match_matrix[int(y[i]), int(y_hat[i])] -= 1
    indices = scipy.optimize.linear_sum_assignment(match_matrix)
    return float(-np.sum(match_matrix[indices]) / y_hat.size)
