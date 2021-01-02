import numpy as np
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from functools import partial
from sklearn.metrics import average_precision_score

def reconstruction_worker(adj, model, objects, progress=False):
    ranksum = nranks = ap_scores = iters = 0
    labels = np.empty(model.lt.weight.size(0))
    for object in tqdm(objects) if progress else objects:
        labels.fill(0)
        neighbors = np.array(list(adj[object]), dtype=np.int32)
        dists = model.energy(model.lt.weight[None, object], model.lt.weight)
        dists[object] = 1e12
        sorted_dists, sorted_idx = dists.sort()
        ranks, = np.where(np.in1d(sorted_idx.detach().cpu().numpy(), neighbors))
        # The above gives us the position of the neighbors in sorted order.  We
        # want to count the number of non-neighbors that occur before each neighbor
        ranks += 1
        N = ranks.shape[0]

        # To account for other positive nearer neighbors, we subtract (N*(N+1)/2)
        # As an example, assume the ranks of the neighbors are:
        # 0, 1, 4, 5, 6, 8
        # For each neighbor, we'd like to return the number of non-neighbors
        # that ranked higher than it.  In this case, we'd return 0+0+2+2+2+3=14
        # Another way of thinking about it is to return
        # 0 + 1 + 4 + 5 + 6 + 8 - (0 + 1 + 2 + 3 + 4 + 5)
        # (0 + 1 + 2 + ... + N) == (N * (N + 1) / 2)
        # Note that we include `N` to account for the source embedding itself
        # always being the nearest neighbor
        ranksum += ranks.sum() - (N * (N - 1) / 2)
        nranks += ranks.shape[0]
        labels[neighbors] = 1
        ap_scores += average_precision_score(labels, -dists.detach().cpu().numpy())
        iters += 1
    return float(ranksum), nranks, ap_scores, iters


def eval_reconstruction(adj, model, workers=1, progress=False):
    '''
    Reconstruction evaluation.  For each object, rank its neighbors by distance

    Args:
        adj (dict[int, set[int]]): Adjacency list mapping objects to its neighbors
        lt (torch.Tensor[N, dim]): Embedding table with `N` embeddings and `dim`
            dimensionality
        distfn ((torch.Tensor, torch.Tensor) -> torch.Tensor): distance function.
        workers (int): number of workers to use
    '''
    objects = np.array(list(adj.keys()))
    if workers > 1:
        with ThreadPool(workers) as pool:
            f = partial(reconstruction_worker, adj, model)
            results = pool.map(f, np.array_split(objects, workers))
            results = np.array(results).sum(axis=0).astype(float)
    else:
        results = reconstruction_worker(adj, model, objects, progress)
    return float(results[0]) / results[1], float(results[2]) / results[3]
