"""Intrinsic dimension estimation for LA2M reference clusters."""

import numpy as np
from loguru import logger

DEFAULT_N = 10
DEFAULT_K = 20


def estimate_intrinsic_dimension(embeddings: np.ndarray, n: int = DEFAULT_N,
                                 k: int = DEFAULT_K, seed: int = 42) -> float:
    """Estimate the intrinsic dimension of ``embeddings``."""
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
