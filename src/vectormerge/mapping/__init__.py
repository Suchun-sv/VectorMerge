"""
VectorMerge Mapping Package

This package provides comprehensive vector space mapping functionality for VectorMerge.
It includes various mapping strategies, clustering algorithms, loss functions, and utilities
for transforming embeddings between different vector spaces.

Main Classes:
- VectorSpaceMapper: Main interface for all mapping strategies
- MappingConfig: Configuration class for mapping parameters
- MappingResult: Result container for mapping operations

Mapping Strategies:
- ProcrustesMappingStrategy: Procrustes analysis-based mapping
- NonLinearMappingStrategy: Non-linear neural network mapping
- LA2MStrategy: Clustering-based mapping from the LA2M paper

Quick Usage:
```python
from vectormerge.mapping import VectorSpaceMapper, MappingConfig

# Create mapper with default configuration
mapper = VectorSpaceMapper(strategy="procrustes")

# Or with custom configuration
config = MappingConfig(num_clusters=100, learning_rate=0.001)
mapper = VectorSpaceMapper(strategy="la2m", config=config)

# Fit and transform
transformed = mapper.fit_transform(
    source_embeddings, target_embeddings, reference_indices
)
```
"""

# Core classes
from .base import MappingStrategy, MappingConfig, MappingResult
from .manager import VectorSpaceMapper

# Mapping strategies
from .strategies import (
    ProcrustesMappingStrategy,
    NonLinearMappingStrategy,
    LA2MStrategy
)

# Loss functions
from .losses import (
    cosine_similarity_loss,
    triplet_loss,
    ranking_loss,
    contrastive_loss,
    mse_loss,
    compute_loss
)

# Convenience functions
from .utils import (
    load_embeddings,
    save_embeddings,
    create_reference_indices,
    validate_embeddings,
    compute_alignment_metrics
)

# Version information
__version__ = "1.0.0"

# Main exports
__all__ = [
    # Core classes
    "VectorSpaceMapper",
    "MappingStrategy", 
    "MappingConfig",
    "MappingResult",
    
    # Mapping strategies
    "ProcrustesMappingStrategy",
    "NonLinearMappingStrategy", 
    "LA2MStrategy",
    
    # Loss functions
    "cosine_similarity_loss",
    "triplet_loss",
    "ranking_loss", 
    "contrastive_loss",
    "mse_loss",
    "compute_loss",
    
    # Utility functions
    "load_embeddings",
    "save_embeddings",
    "create_reference_indices",
    "validate_embeddings",
    "compute_alignment_metrics",
]

# Package metadata
__author__ = "VectorMerge Team"
__email__ = "contact@vectormerge.ai"
__description__ = "Vector space mapping and alignment for embedding models"
__url__ = "https://github.com/vectormerge/vectormerge" 