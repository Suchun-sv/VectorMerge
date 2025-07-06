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

from .mapping import (
    VectorSpaceMapper,
    ProcrustesMappingStrategy,
    NonLinearMappingStrategy,
)

from .evaluation import (
    MetricsCalculator,
    RecallCalculator,
    NDCGCalculator,
)

# Package-level constants
SUPPORTED_MODELS = [
    "nv-embed",
    "openai",
    "mistral",
    "fast-text",
    "glove",
    "gte"
]

SUPPORTED_DATASETS = [
    "scifact",
    "nfcorpus", 
    "arguana",
    "scidocs",
    "fiqa",
]

# Public API
__all__ = [
    # Dataset
    "load_dataset",
    "list_datasets",
    "validate_dataset",
    
    # Vector space mapping
    "VectorSpaceMapper",
    "ProcrustesMappingStrategy",
    "NonLinearMappingStrategy",
    
    # Evaluation
    "MetricsCalculator",
    "RecallCalculator", 
    "NDCGCalculator",
    
    # Constants
    "SUPPORTED_MODELS",
    "SUPPORTED_DATASETS",
    
    # Package metadata
    "__version__",
    "__author__",
    "__email__",
    "__description__",
] 