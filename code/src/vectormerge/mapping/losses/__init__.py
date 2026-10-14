"""Loss functions for VectorMerge mapping strategies."""

from .common import (
    cosine_similarity_loss,
    triplet_loss,
    ranking_loss,
    contrastive_loss,
    mse_loss,
    compute_loss
)

__all__ = [
    "cosine_similarity_loss",
    "triplet_loss",
    "ranking_loss",
    "contrastive_loss",
    "mse_loss",
    "compute_loss",
]