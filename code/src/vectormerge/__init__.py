"""VectorMerge."""

from typing import List, Dict, Any
from .embeddings import SUPPORTED_MODELS
from .dataset import SUPPORTED_DATASETS

__version__ = "1.0.0"
__author__ = "Beining Yang"
__email__ = ""
__description__ = "A modern library for embedding evaluation and vector space mapping"

from .dataset import (
    load_dataset,
    list_datasets,
    validate_dataset,
)

from .embeddings import (
    EmbeddingGenerator,
    get_embedding_generator,
    generate_embeddings,
)

from .clustering import (
    ClusterManager,
    ClusteringConfig,
    ClusteringResult,
)

from .mapping import (
    VectorSpaceMapper,
    ProcrustesMappingStrategy,
    NonLinearMappingStrategy,
    LA2MStrategy,
    CCAMappingStrategy,
    GromovWassersteinMappingStrategy,
)

__all__ = [
    "load_dataset",
    "list_datasets",
    "validate_dataset",

    "EmbeddingGenerator",
    "get_embedding_generator",
    "generate_embeddings",

    "ClusterManager",
    "ClusteringConfig",
    "ClusteringResult",

    "VectorSpaceMapper",
    "ProcrustesMappingStrategy",
    "NonLinearMappingStrategy",
    "LA2MStrategy",
    "CCAMappingStrategy",
    "GromovWassersteinMappingStrategy",


    "SUPPORTED_MODELS",
    "SUPPORTED_DATASETS",

    "__version__",
    "__author__",
    "__email__",
    "__description__",
]