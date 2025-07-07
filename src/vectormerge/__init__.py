"""
VectorMerge: A modern library for embedding evaluation and vector space mapping.

This library provides tools for:
- Embedding generation from various models (BERT, RoBERTa, BGE, OpenAI, etc.)
- Vector space mapping techniques (Procrustes, non-linear mapping)
- Embedding evaluation on BEIR datasets
- Clustering-based reference generation
- Comprehensive evaluation metrics

Author: Beining Yang
License: MIT
"""

from typing import List, Dict, Any
from .embeddings import SUPPORTED_MODELS
from .dataset import SUPPORTED_DATASETS

# Version management
__version__ = "1.0.0"
__author__ = "Beining Yang"
__email__ = "suchunsv@outlook.com"
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
)

# Public API
__all__ = [
    # Dataset
    "load_dataset",
    "list_datasets",
    "validate_dataset",
    
    # Embeddings
    "EmbeddingGenerator",
    "get_embedding_generator",
    "generate_embeddings",

    # Clustering
    "ClusterManager",
    "ClusteringConfig",
    "ClusteringResult",

    # Vector space mapping
    "VectorSpaceMapper",
    "ProcrustesMappingStrategy",
    "NonLinearMappingStrategy",
    "LA2MStrategy",
    

    # Constants
    "SUPPORTED_MODELS",
    "SUPPORTED_DATASETS",
    
    # Package metadata
    "__version__",
    "__author__",
    "__email__",
    "__description__",
] 