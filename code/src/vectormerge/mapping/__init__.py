"""VectorMerge Mapping Package."""

from .base import MappingStrategy, MappingConfig, MappingResult
from .manager import VectorSpaceMapper
from .wrapper import LA2MMapper, ProcrustesMapper

SUPPORTED_MAPPING_METHODS = [
    "procrustes",
    "nonlinear",
    "la2m",
    "cca",
    "gromov_wasserstein",
]

from .strategies import (
    ProcrustesMappingStrategy,
    NonLinearMappingStrategy,
    LA2MStrategy,
    CCAMappingStrategy,
    GromovWassersteinMappingStrategy
)

from .losses import (
    cosine_similarity_loss,
    triplet_loss,
    ranking_loss,
    contrastive_loss,
    mse_loss,
    compute_loss
)

from .utils import (
    load_embeddings,
    save_embeddings,
    create_reference_indices,
    validate_embeddings,
    compute_alignment_metrics
)

__version__ = "1.0.0"

__all__ = [
    "VectorSpaceMapper",
    "MappingStrategy",
    "MappingConfig",
    "MappingResult",

    "ProcrustesMappingStrategy",
    "NonLinearMappingStrategy",
    "LA2MStrategy",
    "CCAMappingStrategy",
    "GromovWassersteinMappingStrategy",
    "LA2MMapper",
    "ProcrustesMapper",
    "cosine_similarity_loss",
    "triplet_loss",
    "ranking_loss",
    "contrastive_loss",
    "mse_loss",
    "compute_loss",

    "load_embeddings",
    "save_embeddings",
    "create_reference_indices",
    "validate_embeddings",
    "compute_alignment_metrics",
]

__author__ = "VectorMerge Team"
__email__ = "contact@vectormerge.ai"
__description__ = "Vector space mapping and alignment for embedding models"
__url__ = "https://github.com/vectormerge/vectormerge"