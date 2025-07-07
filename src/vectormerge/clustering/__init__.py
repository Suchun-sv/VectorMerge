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

# Core classes
from .base import ClusteringStrategy, ClusteringConfig, ClusteringResult
from .manager import ClusterManager, load_cluster_result, load_cluster_config
from .manager import SUPPORTED_CLUSTERING_METHODS

# Clustering strategies
from .strategies import (
    KMeansClusteringStrategy,
    LA2MClusteringStrategy
)

# Utility functions
from .utils import (
    load_cluster_data,
    save_cluster_data,
    compute_cluster_metrics,
    visualize_clusters
)

# Version information
__version__ = "1.0.0"

# Export main classes
__all__ = [
    # Core classes
    "ClusteringStrategy",
    "ClusteringConfig",
    "ClusteringResult",
    "ClusterManager",
    "SUPPORTED_CLUSTERING_METHODS",
    "load_cluster_result",
    "load_cluster_config",
    
    # Clustering strategies
    "KMeansClusteringStrategy",
    "LA2MClusteringStrategy",
    
    # Utility functions
    "load_cluster_data",
    "save_cluster_data",
    "compute_cluster_metrics",
    "visualize_clusters",
] 