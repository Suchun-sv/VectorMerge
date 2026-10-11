"""Intrinsic dimension estimation for LA2M reference clusters.

doi:10.1145/3769803 Section 6.1 sizes each local reference cluster X by the intrinsic
dimension of the reference vectors, estimated with the Levina-Bickel MLE:

    d_hat = [ 1 / (N (K-1)) * sum_i sum_{j=1}^{K-1} log( T_K(x_i) / T_j(x_i) ) ]^-1

where T_j(x_i) is the distance from x_i to its j-th nearest neighbour. The paper uses
N = 10 sampled points and K = 20 neighbours.
"""

import numpy as np
from loguru import logger

DEFAULT_N = 10
DEFAULT_K = 20


def estimate_intrinsic_dimension(embeddings: np.ndarray, n: int = DEFAULT_N,
                                 k: int = DEFAULT_K, seed: int = 42) -> float:
    """Estimate the intrinsic dimension of ``embeddings``.

    Args:
        embeddings: Vectors to analyse, shape (num_points, dim).
        n: Number of points to sample (N in the paper).
        k: Number of nearest neighbours per point (K in the paper).
        seed: Seed for the point sample, so the estimate is reproducible.

    Returns:
        The MLE estimate, or ``nan`` when there are too few usable neighbours.
    """
    if len(embeddings) < 2:
        return float("nan")

    rng = np.random.default_rng(seed)
    sample = rng.choice(len(embeddings), size=min(n, len(embeddings)), replace=False)

    log_ratio_sum, term_count = 0.0, 0
    for i in sample:
        distances = np.linalg.norm(embeddings - embeddings[i], axis=1)
        neighbours = np.sort(distances)[1:k + 1]
        neighbours = neighbours[neighbours > 0]
        if len(neighbours) < 2:
            continue
        t_k = neighbours[-1]
        log_ratio_sum += float(np.sum(np.log(t_k / neighbours[:-1])))
        term_count += len(neighbours) - 1

    if term_count == 0 or log_ratio_sum <= 0:
        logger.warning("Intrinsic dimension estimate is undefined; falling back to d_prime")
        return float("nan")

    return term_count / log_ratio_sum
