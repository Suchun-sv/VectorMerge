"""
Base classes and interfaces for clustering in VectorMerge.

This module provides abstract base classes and data structures for different
clustering strategies and configurations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import torch
from loguru import logger


@dataclass
class ClusterData:
    """Data structure to store cluster information."""
    
    ref_index: List[int]      # Indices of reference points in this cluster
    bound_index: List[int]    # Indices of target points assigned to this cluster
    diameter: float = 0.0     # Cluster diameter (optional)
    center: Optional[np.ndarray] = None  # Cluster center (optional)
    
    def __len__(self) -> int:
        """Return total number of points in cluster."""
        return len(self.ref_index) + len(self.bound_index)
    
    @property
    def ref_size(self) -> int:
        """Return number of reference points."""
        return len(self.ref_index)
    
    @property
    def bound_size(self) -> int:
        """Return number of bound points."""
        return len(self.bound_index)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert cluster data to dictionary for serialization."""
        return {
            'ref_index': self.ref_index,
            'bound_index': self.bound_index,
            'diameter': self.diameter,
            'center': self.center.tolist() if self.center is not None else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClusterData':
        """Create cluster data from dictionary."""
        center = np.array(data['center']) if data['center'] is not None else None
        return cls(
            ref_index=data['ref_index'],
            bound_index=data['bound_index'],
            diameter=data.get('diameter', 0.0),
            center=center
        )


@dataclass
class ClusteringConfig:
    """Configuration for clustering operations."""
    
    # Basic clustering parameters
    num_clusters: int = 50
    method: str = "kmeans"  # "kmeans", "la2m-cluster"
    min_cluster_size: int = 5
    max_cluster_size: int = 1000
    
    # K-means specific parameters
    random_state: int = 42
    max_iter: int = 300
    tol: float = 1e-4
    
    # LA2M clustering parameters
    d_prime: int = 10  # Number of top neighbors for LA2M neighborhood construction
    
    # General parameters
    device: str = "auto"
    verbose: bool = False
    
    # Quality metrics
    compute_metrics: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.num_clusters <= 0:
            raise ValueError("num_clusters must be positive")
        if self.min_cluster_size <= 0:
            raise ValueError("min_cluster_size must be positive")
        if self.max_cluster_size <= self.min_cluster_size:
            raise ValueError("max_cluster_size must be greater than min_cluster_size")
        if self.method not in ["kmeans", "la2m-cluster"]:
            raise ValueError(f"Unsupported clustering method: {self.method}")
        if self.d_prime <= 0:
            raise ValueError("d_prime must be positive")
        
    def to_string(self) -> str:
        """Convert configuration to string."""
        return str(self.__dict__)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'num_clusters': self.num_clusters,
            'method': self.method,
            'min_cluster_size': self.min_cluster_size,
            'max_cluster_size': self.max_cluster_size,
            'random_state': self.random_state,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'd_prime': self.d_prime,
            'device': self.device,
            'verbose': self.verbose,
            'compute_metrics': self.compute_metrics
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClusteringConfig':
        """Create configuration from dictionary."""
        # Get all field names from the dataclass
        import dataclasses
        field_names = {field.name for field in dataclasses.fields(cls)}
        
        # Filter the dictionary to only include fields that exist in the dataclass
        filtered_data = {k: v for k, v in data.items() if k in field_names}
        
        return cls(**filtered_data)


@dataclass
class ClusteringResult:
    """Result container for clustering operations."""
    
    cluster_data_list: List[ClusterData]
    cluster_labels: np.ndarray
    cluster_centers: Optional[np.ndarray] = None
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def num_clusters(self) -> int:
        """Return number of clusters."""
        return len(self.cluster_data_list)
    
    @property
    def cluster_sizes(self) -> List[int]:
        """Return sizes of all clusters."""
        return [len(cluster.ref_index) for cluster in self.cluster_data_list]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            'cluster_data_list': [cluster.to_dict() for cluster in self.cluster_data_list],
            'cluster_labels': self.cluster_labels.tolist(),
            'cluster_centers': self.cluster_centers.tolist() if self.cluster_centers is not None else None,
            'quality_metrics': self.quality_metrics,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClusteringResult':
        """Create result from dictionary."""
        cluster_data_list = [ClusterData.from_dict(cluster_data) for cluster_data in data['cluster_data_list']]
        cluster_labels = np.array(data['cluster_labels'])
        cluster_centers = np.array(data['cluster_centers']) if data['cluster_centers'] is not None else None
        
        return cls(
            cluster_data_list=cluster_data_list,
            cluster_labels=cluster_labels,
            cluster_centers=cluster_centers,
            quality_metrics=data.get('quality_metrics', {}),
            metadata=data.get('metadata', {})
        )


class ClusteringStrategy(ABC):
    """Abstract base class for clustering strategies."""
    
    def __init__(self, config: ClusteringConfig):
        """Initialize clustering strategy.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.device = torch.device(config.device if config.device != "auto" 
                                 else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.is_fitted = False
        
        logger.info(f"Initialized {self.__class__.__name__} with {config.num_clusters} clusters")
    
    @abstractmethod
    def fit(self, embeddings: np.ndarray, reference_indices: np.ndarray) -> ClusteringResult:
        """Fit the clustering strategy to the data.
        
        Args:
            embeddings: All embeddings (N x D)
            reference_indices: Indices of reference points to cluster
            
        Returns:
            ClusteringResult object with clustering information
        """
        pass
    
    @abstractmethod
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict cluster assignments for new embeddings.
        
        Args:
            embeddings: New embeddings to assign to clusters
            
        Returns:
            Cluster assignments for each embedding
        """
        pass
    
    @abstractmethod
    def _fit(self, embeddings: np.ndarray, reference_indices: np.ndarray) -> ClusteringResult:
        """Fit the clustering strategy to the data.
        
        Args:
            embeddings: All embeddings (N x D)
            reference_indices: Indices of reference points to cluster
            
        Returns:
        """
        pass
    
    @abstractmethod
    def _predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict cluster assignments for new embeddings.
        
        Args:
            embeddings: New embeddings to assign to clusters
            
        Returns:
            Cluster assignments for each embedding
        """
        pass

    
    
    def assign_to_clusters(self, embeddings: np.ndarray, target_indices: np.ndarray,
                          clustering_result: ClusteringResult) -> ClusteringResult:
        """Assign target points to existing clusters.
        
        Args:
            embeddings: All embeddings
            target_indices: Indices of target points to assign
            clustering_result: Existing clustering result
            
        Returns:
            Updated clustering result with assigned target points
        """
        if clustering_result.cluster_centers is None:
            raise ValueError("Cannot assign points without cluster centers")
        
        logger.info(f"Assigning {len(target_indices)} target points to clusters")
        
        # Extract target embeddings
        target_embeddings = embeddings[target_indices]
        
        # Predict cluster assignments
        cluster_assignments = self.predict(target_embeddings)
        
        # Update cluster data with assigned target points
        updated_clusters = []
        for cluster_id, cluster_data in enumerate(clustering_result.cluster_data_list):
            # Find target points assigned to this cluster
            cluster_mask = (cluster_assignments == cluster_id)
            cluster_target_indices = target_indices[cluster_mask]
            
            # Create updated cluster data
            updated_cluster = ClusterData(
                ref_index=cluster_data.ref_index.copy(),
                bound_index=cluster_target_indices.tolist(),
                diameter=cluster_data.diameter,
                center=cluster_data.center.copy() if cluster_data.center is not None else None
            )
            updated_clusters.append(updated_cluster)
        
        # Create updated result
        updated_result = ClusteringResult(
            cluster_data_list=updated_clusters,
            cluster_labels=clustering_result.cluster_labels,
            cluster_centers=clustering_result.cluster_centers,
            quality_metrics=clustering_result.quality_metrics,
            metadata=clustering_result.metadata
        )
        
        # Log assignment statistics
        assignment_counts = [len(cluster.bound_index) for cluster in updated_clusters]
        logger.info(f"Target point assignment completed. Points per cluster: {assignment_counts}")
        
        return updated_result
    
    def compute_quality_metrics(self, embeddings: np.ndarray, clustering_result: ClusteringResult) -> Dict[str, float]:
        """Compute quality metrics for the clustering.
        
        Args:
            embeddings: All embeddings
            clustering_result: Clustering result
            
        Returns:
            Dictionary with quality metrics
        """
        if not clustering_result.cluster_data_list:
            return {}
        
        # Compute within-cluster sum of squares (WCSS)
        total_wcss = 0.0
        cluster_wcss = []
        
        for cluster_data in clustering_result.cluster_data_list:
            if len(cluster_data.ref_index) > 0:
                cluster_embeddings = embeddings[cluster_data.ref_index]
                center = cluster_data.center
                
                if center is not None:
                    # Compute sum of squared distances to center
                    distances_sq = np.sum((cluster_embeddings - center) ** 2, axis=1)
                    wcss = np.sum(distances_sq)
                    cluster_wcss.append(wcss)
                    total_wcss += wcss
                else:
                    cluster_wcss.append(0.0)
            else:
                cluster_wcss.append(0.0)
        
        # Compute average cluster size
        cluster_sizes = [len(cluster.ref_index) for cluster in clustering_result.cluster_data_list]
        avg_cluster_size = float(np.mean(cluster_sizes)) if cluster_sizes else 0.0
        
        # Compute cluster size variance
        cluster_size_variance = float(np.var(cluster_sizes)) if cluster_sizes else 0.0
        
        return {
            'total_wcss': total_wcss,
            'avg_wcss': total_wcss / len(clustering_result.cluster_data_list) if clustering_result.cluster_data_list else 0,
            'avg_cluster_size': avg_cluster_size,
            'cluster_size_variance': cluster_size_variance,
            'num_clusters': len(clustering_result.cluster_data_list),
            'min_cluster_size': min(cluster_sizes) if cluster_sizes else 0,
            'max_cluster_size': max(cluster_sizes) if cluster_sizes else 0
        }
    
    def save(self, path: Path, clustering_result: ClusteringResult) -> None:
        """Save the clustering result and strategy.
        
        Args:
            path: Path to save the clustering result
            clustering_result: Clustering result to save
        """
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save clustering result
        import json
        with open(save_path / "clustering_result.json", "w") as f:
            json.dump(clustering_result.to_dict(), f, indent=2)
        
        # Save configuration
        with open(save_path / "clustering_config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        # Save cluster centers if available
        if clustering_result.cluster_centers is not None:
            np.save(save_path / "cluster_centers.npy", clustering_result.cluster_centers)
        
        logger.info(f"Saved clustering result to {save_path}")
    
    @classmethod
    def load(cls, path: Path) -> Tuple['ClusteringStrategy', ClusteringResult]:
        """Load clustering strategy and result.
        
        Args:
            path: Path to load from
            
        Returns:
            Tuple of (clustering_strategy, clustering_result)
        """
        load_path = Path(path)
        
        # Load configuration
        import json
        with open(load_path / "clustering_config.json", "r") as f:
            config_data = json.load(f)
        
        config = ClusteringConfig.from_dict(config_data)
        
        # Load clustering result
        with open(load_path / "clustering_result.json", "r") as f:
            result_data = json.load(f)
        
        clustering_result = ClusteringResult.from_dict(result_data)
        
        # Load cluster centers if available
        centers_path = load_path / "cluster_centers.npy"
        if centers_path.exists():
            clustering_result.cluster_centers = np.load(centers_path)
        
        # Create strategy instance
        strategy = cls(config)
        strategy.is_fitted = True
        
        return strategy, clustering_result 