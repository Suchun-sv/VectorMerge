"""VectorMerge Clustering Package."""

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