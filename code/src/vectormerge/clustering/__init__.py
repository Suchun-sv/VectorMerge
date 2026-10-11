"""
VectorMerge Clustering Package

This package provides comprehensive clustering functionality for VectorMerge.
It includes various clustering strategies for embedding analysis, reference point clustering,
and cluster-based operations.

Main Classes:
- ClusteringStrategy: Abstract base class for clustering strategies
- ClusterData: Data structure for cluster information
- KMeansClusteringStrategy: K-means clustering implementation
- HierarchicalClusteringStrategy: Hierarchical clustering implementation
- ClusterManager: Main interface for clustering operations

Quick Usage:
```python
from vectormerge.clustering import ClusterManager, ClusteringConfig

# Create cluster manager
config = ClusteringConfig(num_clusters=50, method="kmeans")
manager = ClusterManager(config)

# Cluster embeddings
cluster_results = manager.cluster_embeddings(
    embeddings, reference_indices
)
```
"""

from .base import (
    ClusteringStrategy, 
    ClusteringConfig, 
    ClusteringResult, 
    ClusterData,
)

from .manage import ClusterManager
from .intrinsic_dimension import estimate_intrinsic_dimension
from .strategies import SUPPORTED_CLUSTERING_METHODS

from .strategies import (
    KMeansClusteringStrategy,
    LA2MClusteringStrategy,
)

from .utils import (
    load_cluster_data,
    save_cluster_data,
    compute_cluster_metrics,
    visualize_clusters
)


__version__ = "1.0.0"

__all__ = [
    "estimate_intrinsic_dimension",
    "ClusteringStrategy",
    "ClusteringConfig",
    "ClusteringResult",
    "ClusterManager",
    "ClusterData",
    "SUPPORTED_CLUSTERING_METHODS",
    
    "KMeansClusteringStrategy",
    "LA2MClusteringStrategy",
    
    "load_cluster_data",
    "save_cluster_data",
    "compute_cluster_metrics",
    "visualize_clusters",
] 