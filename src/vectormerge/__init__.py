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

# Core imports
from .core import (
    EmbeddingEvaluator,
    EvaluationMetrics,
    EmbeddingDataset,
)

from .embeddings import (
    EmbeddingGenerator,
    get_embedding_generator,
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

from .config import (
    VectorMergeConfig,
    ModelConfig,
    EvaluationConfig,
)

# Package-level constants
SUPPORTED_MODELS = [
    "bert-base-uncased",
    "roberta-base", 
    "bge",
    "nv-embed",
    "openai",
    "mistral",
    "fast-text",
    "word2vec",
    "glove",
]

SUPPORTED_DATASETS = [
    "scifact",
    "nfcorpus", 
    "nq",
    "cqadupstack",
    "arguana",
    "scidocs",
    "fiqa",
]

# Public API
__all__ = [
    # Core classes
    "EmbeddingEvaluator",
    "EvaluationMetrics", 
    "EmbeddingDataset",
    
    # Embedding generation
    "EmbeddingGenerator",
    "get_embedding_generator",
    
    # Vector space mapping
    "VectorSpaceMapper",
    "ProcrustesMappingStrategy",
    "NonLinearMappingStrategy",
    
    # Evaluation
    "MetricsCalculator",
    "RecallCalculator", 
    "NDCGCalculator",
    
    # Configuration
    "VectorMergeConfig",
    "ModelConfig",
    "EvaluationConfig",
    
    # Constants
    "SUPPORTED_MODELS",
    "SUPPORTED_DATASETS",
    
    # Package metadata
    "__version__",
    "__author__",
    "__email__",
    "__description__",
] 