"""
Cluster Manager for VectorMerge clustering operations.

This module provides the main interface for clustering operations,
including embedding clustering, reference point clustering, and cluster analysis.
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Union
import json
from pathlib import Path
from loguru import logger
import hashlib
import joblib

from .base import ClusteringStrategy, ClusteringConfig, ClusteringResult, ClusterData
from .strategies import KMeansClusteringStrategy, LA2MClusteringStrategy

from ..embeddings import get_embedding
from ..reference import get_reference

method_map = {
    "kmeans": KMeansClusteringStrategy,
    "la2m-cluster": LA2MClusteringStrategy
}

SUPPORTED_CLUSTERING_METHODS = list(method_map.keys())

def config_hash_path(dataset_name: str, model: str, reference_key: str, strategy_name: str, strategy_config: ClusteringConfig):
    hash_target_str = strategy_config.to_string() + reference_key
    config_hash = hashlib.md5(hash_target_str.encode()).hexdigest()
    return f"{strategy_name}_{dataset_name}_{model}_{config_hash}"

def load_cluster_result(cluster_path: Path) -> Optional[ClusteringResult]:
    """Load clustering result from disk."""
    if not (cluster_path / "cluster.pkl").exists():
        return None
    return joblib.load(cluster_path / "cluster.pkl")

def load_cluster_config(cluster_path: Path) -> Optional[Dict[str, Any]]:
    """Load clustering config from disk."""
    if not (cluster_path / "config.json").exists():
        return None
    with open(cluster_path / "config.json", "r") as f:
        return json.load(f)

class ClusterManager:
    """Manager for clustering operations in VectorMerge."""
    
    # def __init__(self, config: Optional[ClusteringConfig] = None,
    #              dataset_name: Optional[str] = None,
    #              source_embeddings: Optional[np.ndarray] = None,
    #              reference_indices: Optional[np.ndarray] = None,
    def __init__(self, dataset_name: str, model: str, reference_key: str, reference_path: str, cluster_path: str, embedding_path: str, strategy_name: str, force: bool = False, strategy_config: Optional[ClusteringConfig] = None, auto_save_results: bool = False, verbose: bool = False) -> None:
        """Initialize the cluster manager.
        
        Args:
            dataset_name: Name of the dataset
            model: Name of the embedding model
            reference_key: Key for reference data
            reference_path: Path to reference files
            cluster_path: Path to save/load clustering results
            embedding_path: Path to embedding files
            strategy_name: Name of the clustering strategy
            force: Whether to force re-clustering even if results exist
            strategy_config: Configuration for clustering strategy
            auto_save_results: Whether to automatically save clustering results
            verbose: Whether to enable verbose logging
        """
        self.dataset_name = dataset_name
        self.model = model
        self.reference_key = reference_key
        self.reference_path = reference_path
        self.embedding_path = embedding_path
        self.strategy_name = strategy_name
        self.strategy_config = strategy_config if strategy_config else ClusteringConfig()
        self.cluster_path = Path(cluster_path) if cluster_path else None
        self.cluster_key = config_hash_path(dataset_name, model, reference_key, strategy_name, self.strategy_config)
        self.final_save_path = Path(cluster_path) / self.cluster_key if cluster_path else None
        self.force = force
        self.auto_save_results = auto_save_results
        self.verbose = verbose
        
        # Initialize strategy
        self.strategy = self._create_strategy(self.strategy_config)
        self.last_result: Optional[ClusteringResult] = None
    
    def _to_dict(self) -> Dict[str, Any]:
        return {
            'dataset_name': self.dataset_name,
            'model': self.model,
            'reference_key': self.reference_key,
            'strategy_name': self.strategy_name,
            'cluster_key': str(self.cluster_key),
            'final_save_path': str(self.final_save_path),
        }
    
    def _create_strategy(self, config: ClusteringConfig) -> ClusteringStrategy:
        """Create a clustering strategy based on configuration."""
        if self.strategy_name not in SUPPORTED_CLUSTERING_METHODS:
            raise ValueError(f"Unsupported clustering method: {self.strategy_name}")
        return method_map[self.strategy_name](self.strategy_config)
    
    
    def _check_fit(self) -> bool:
        """Check if clustering results already exist."""
        if not self.final_save_path:
            return False
        return self.final_save_path.exists()
    
    def fit(self):
        """Fit the clustering strategy to the data."""
        if self.strategy is None:
            raise ValueError("Strategy not initialized")
        
        # Check if we should use existing results
        if not self.force and self._check_fit():
            logger.info(f"Found existing cluster result, loading from {self.final_save_path}")
            result = self.load()
            
            # If result exists but lacks embeddings, assemble and return
            if not result.has_embeddings:
                logger.info(f"Loading embeddings to assemble cluster result")
                source_embeddings = get_embedding(self.model, self.dataset_name, self.embedding_path, type_="corpus")
                if source_embeddings is None:
                    raise ValueError(f"Could not load embeddings for {self.dataset_name}")
                return self.assembles_cluster_result(source_embeddings, result)
            
            return result
        
        # Load embeddings and reference indices for new clustering
        logger.info(f"Loading embeddings for {self.dataset_name}")
        source_embeddings = get_embedding(self.model, self.dataset_name, self.embedding_path, type_="corpus")
        if source_embeddings is None:
            raise ValueError(f"Could not load embeddings for {self.dataset_name}")
        
        logger.info(f"Loading reference indices for {self.dataset_name}")
        reference_data = get_reference(self.reference_path, self.reference_key)
        reference_indices = reference_data['d0_index']
        if reference_indices is None:
            raise ValueError(f"Could not load reference indices for {self.dataset_name} with reference key {self.reference_key}")
        
        # Perform clustering
        logger.info(f"Starting clustering with {len(reference_indices)} reference points")
        cluster_result = self.strategy.fit(source_embeddings, reference_indices)
        self.last_result = cluster_result
        
        # Auto-save if enabled
        if self.auto_save_results:
            self.save_cluster_to_disk(cluster_result)
        
        return cluster_result
    
    def load(self) -> ClusteringResult:
        """Load existing clustering results."""
        if self.final_save_path is None:
            raise ValueError("No save path specified. Clustering result not saved.")
        if not self.final_save_path.exists():
            raise FileNotFoundError(f"Clustering result not found: {self.final_save_path}")
        
        logger.info(f"Loading cluster from {self.final_save_path}")
        clustering_result = joblib.load(self.final_save_path)
        self.last_result = clustering_result
        return clustering_result
    
    def assembles_cluster_result(self, embeddings: np.ndarray, clustering_result: ClusteringResult) -> ClusteringResult:
        """Assemble cluster result from embeddings and clustering result."""
        for cluster in clustering_result.cluster_data_list:
            cluster.reference_embeddings = embeddings[cluster.reference_indices]
            cluster.linked_target_embeddings = embeddings[cluster.linked_target_indices]
            cluster.center_embedding = np.mean(cluster.reference_embeddings, axis=0)
        return clustering_result
    
    def save_cluster_to_disk(self, clustering_result: ClusteringResult) -> None:
        """Save clustering result to disk."""
        if self.final_save_path is None:
            logger.warning("No save path specified. Clustering result not saved.")
            return
        
        self.final_save_path.mkdir(parents=True, exist_ok=True)
        # save cluster_reulsts to self.final_save_path/cluster.pkl
        # save all config to self.final_save_path/config.json
        joblib.dump(clustering_result, self.final_save_path / "cluster.pkl")
        config_dict = self._to_dict()
        config_dict.update(self.strategy_config.to_dict())
        with open(self.final_save_path / "config.json", "w") as f:
            json.dump(config_dict, f)
        
        logger.info(f"Clustering result saved to {self.final_save_path}")
    
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict cluster assignments for new embeddings."""
        if self.strategy is None:
            raise ValueError("Strategy not initialized")
        
        if not self.strategy.is_fitted:
            raise ValueError("Strategy not fitted. Run fit() first.")
        
        return self.strategy.predict(embeddings)
    
    def cluster_embeddings(self, embeddings: np.ndarray, 
                          reference_indices: np.ndarray) -> ClusteringResult:
        """Cluster embeddings using the configured strategy.
        
        This method is kept for backward compatibility.
        
        Args:
            embeddings: All embeddings (N x D)
            reference_indices: Indices of reference points to cluster
            
        Returns:
            ClusteringResult object with clustering information
        """
        # Set the attributes and call fit
        self.source_emebddings = embeddings
        self.reference_indices = reference_indices
        
        return self.fit()
    
    def assign_to_clusters(self, embeddings: np.ndarray, 
                          target_indices: np.ndarray,
                          clustering_result: Optional[ClusteringResult] = None) -> ClusteringResult:
        """Assign target points to existing clusters.
        
        Args:
            embeddings: All embeddings
            target_indices: Indices of target points to assign
            clustering_result: Existing clustering result (uses last result if None)
            
        Returns:
            Updated clustering result with assigned target points
        """
        if self.strategy is None:
            raise ValueError("Strategy not initialized")
        
        # Use last result if none provided
        if clustering_result is None:
            clustering_result = self.last_result
            if clustering_result is None:
                raise ValueError("No clustering result available. Run cluster_embeddings first.")
        
        logger.info(f"Assigning {len(target_indices)} target points to clusters")
        
        # Assign points to clusters
        updated_result = self.strategy.assign_to_clusters(embeddings, target_indices, clustering_result)
        
        logger.info(f"Assignment completed. Points per cluster: "
                   f"{[len(cluster.bound_index) for cluster in updated_result.cluster_data_list]}")
        
        return updated_result
    
    def get_cluster_statistics(self, clustering_result: Optional[ClusteringResult] = None) -> Dict[str, Any]:
        """Get detailed statistics about the clustering.
        
        Args:
            clustering_result: Clustering result to analyze (uses last result if None)
            
        Returns:
            Dictionary with cluster statistics
        """
        if clustering_result is None:
            clustering_result = self.last_result
            if clustering_result is None:
                raise ValueError("No clustering result available")
        
        stats = {
            'num_clusters': clustering_result.num_clusters,
            'cluster_sizes': clustering_result.cluster_sizes,
            'total_points': sum(len(cluster.reference_indices) + len(cluster.linked_target_indices) 
                              for cluster in clustering_result.cluster_data_list),
            'total_reference_points': sum(len(cluster.reference_indices) 
                                        for cluster in clustering_result.cluster_data_list),
            'total_linked_target_points': sum(len(cluster.linked_target_indices) 
                                    for cluster in clustering_result.cluster_data_list),
            'avg_cluster_size': np.mean(clustering_result.cluster_sizes),
            'cluster_size_std': np.std(clustering_result.cluster_sizes),
            'min_cluster_size': min(clustering_result.cluster_sizes) if clustering_result.cluster_sizes else 0,
            'max_cluster_size': max(clustering_result.cluster_sizes) if clustering_result.cluster_sizes else 0,
            'quality_metrics': clustering_result.quality_metrics,
            'metadata': clustering_result.metadata
        }
        
        return stats
    
    def save_clustering_result(self, path: Union[str, Path], 
                             clustering_result: Optional[ClusteringResult] = None) -> None:
        """Save clustering result to disk.
        
        Args:
            path: Path to save the clustering result
            clustering_result: Result to save (uses last result if None)
        """
        if self.strategy is None:
            raise ValueError("Strategy not initialized")
        
        if clustering_result is None:
            clustering_result = self.last_result
            if clustering_result is None:
                raise ValueError("No clustering result available")
        
        save_path = Path(path)
        self.strategy.save(save_path, clustering_result)
        
        logger.info(f"Clustering result saved to {save_path}")
    
    def get_cluster_centers(self, clustering_result: Optional[ClusteringResult] = None) -> Optional[np.ndarray]:
        """Get cluster centers from clustering result.
        
        Args:
            clustering_result: Clustering result (uses last result if None)
            
        Returns:
            Cluster centers or None if not available
        """
        if clustering_result is None:
            clustering_result = self.last_result
            if clustering_result is None:
                raise ValueError("No clustering result available")
        
        return clustering_result.cluster_centers
    
    def get_cluster_data(self, cluster_id: int, 
                        clustering_result: Optional[ClusteringResult] = None) -> ClusterData:
        """Get data for a specific cluster.
        
        Args:
            cluster_id: ID of the cluster
            clustering_result: Clustering result (uses last result if None)
            
        Returns:
            Cluster data
        """
        if clustering_result is None:
            clustering_result = self.last_result
            if clustering_result is None:
                raise ValueError("No clustering result available")
        
        if cluster_id < 0 or cluster_id >= len(clustering_result.cluster_data_list):
            raise ValueError(f"Invalid cluster ID: {cluster_id}")
        
        return clustering_result.cluster_data_list[cluster_id]
    
    def analyze_cluster_quality(self, embeddings: np.ndarray,
                               clustering_result: Optional[ClusteringResult] = None) -> Dict[str, Any]:
        """Analyze the quality of clustering.
        
        Args:
            embeddings: Original embeddings
            clustering_result: Clustering result (uses last result if None)
            
        Returns:
            Dictionary with quality analysis
        """
        if self.strategy is None:
            raise ValueError("Strategy not initialized")
        
        if clustering_result is None:
            clustering_result = self.last_result
            if clustering_result is None:
                raise ValueError("No clustering result available")
        
        # Compute quality metrics
        quality_metrics = self.strategy.compute_quality_metrics(embeddings, clustering_result)
        
        # Get cluster statistics
        cluster_stats = self.get_cluster_statistics(clustering_result)
        
        # Combine analysis
        analysis = {
            'quality_metrics': quality_metrics,
            'cluster_statistics': cluster_stats,
            'cluster_balance': {
                'size_variance': cluster_stats['cluster_size_std'],
                'size_coefficient_of_variation': cluster_stats['cluster_size_std'] / cluster_stats['avg_cluster_size'] if cluster_stats['avg_cluster_size'] > 0 else 0,
                'is_balanced': cluster_stats['cluster_size_std'] / cluster_stats['avg_cluster_size'] < 0.5 if cluster_stats['avg_cluster_size'] > 0 else False
            }
        }
        
        return analysis
    
    def filter_clusters_by_size(self, clustering_result: Optional[ClusteringResult] = None,
                               min_size: Optional[int] = None,
                               max_size: Optional[int] = None) -> List[int]:
        """Filter clusters by size constraints.
        
        Args:
            clustering_result: Clustering result (uses last result if None)
            min_size: Minimum cluster size
            max_size: Maximum cluster size
            
        Returns:
            List of cluster IDs that meet the size constraints
        """
        if clustering_result is None:
            clustering_result = self.last_result
            if clustering_result is None:
                raise ValueError("No clustering result available")
        
        min_size = min_size or self.strategy_config.min_cluster_size
        max_size = max_size or self.strategy_config.max_cluster_size
        
        valid_clusters = []
        for i, cluster_data in enumerate(clustering_result.cluster_data_list):
            cluster_size = len(cluster_data.reference_indices) + len(cluster_data.linked_target_indices)
            if min_size <= cluster_size <= max_size:
                valid_clusters.append(i)
        
        logger.info(f"Found {len(valid_clusters)} clusters meeting size constraints "
                   f"({min_size}-{max_size})")
        
        return valid_clusters
    
    def merge_small_clusters(self, embeddings: np.ndarray,
                           clustering_result: Optional[ClusteringResult] = None,
                           min_size: Optional[int] = None) -> ClusteringResult:
        """Merge clusters that are smaller than the minimum size.
        
        Args:
            embeddings: Original embeddings
            clustering_result: Clustering result (uses last result if None)
            min_size: Minimum cluster size
            
        Returns:
            Updated clustering result with merged clusters
        """
        if clustering_result is None:
            clustering_result = self.last_result
            if clustering_result is None:
                raise ValueError("No clustering result available")
        
        min_size = min_size or self.strategy_config.min_cluster_size
        
        # Find small clusters
        small_clusters = []
        valid_clusters = []
        
        for i, cluster_data in enumerate(clustering_result.cluster_data_list):
            cluster_size = len(cluster_data.reference_indices) + len(cluster_data.linked_target_indices)
            if cluster_size < min_size:
                small_clusters.append(i)
            else:
                valid_clusters.append(i)
        
        if not small_clusters:
            logger.info("No small clusters found")
            return clustering_result
        
        logger.info(f"Merging {len(small_clusters)} small clusters with valid clusters")
        
        # For each small cluster, find the nearest valid cluster
        for small_cluster_id in small_clusters:
            small_cluster = clustering_result.cluster_data_list[small_cluster_id]
            
            if small_cluster.center_embedding is None or clustering_result.cluster_centers is None:
                # If no centers available, merge with first valid cluster
                if valid_clusters:
                    target_cluster_id = valid_clusters[0]
                else:
                    continue
            else:
                # Find nearest valid cluster by center distance
                min_distance = float('inf')
                target_cluster_id = None
                
                for valid_cluster_id in valid_clusters:
                    distance = np.linalg.norm(
                        small_cluster.center_embedding - clustering_result.cluster_centers[valid_cluster_id]
                    )
                    if distance < min_distance:
                        min_distance = distance
                        target_cluster_id = valid_cluster_id
                
                if target_cluster_id is None:
                    continue
            
            # Merge small cluster into target cluster
            target_cluster = clustering_result.cluster_data_list[target_cluster_id]
            target_cluster.reference_indices.extend(small_cluster.reference_indices)
            target_cluster.linked_target_indices.extend(small_cluster.linked_target_indices)
            
            # Recompute center and update embeddings
            all_indices = target_cluster.reference_indices + target_cluster.linked_target_indices
            if all_indices:
                cluster_embeddings = embeddings[all_indices]
                target_cluster.center_embedding = np.mean(cluster_embeddings, axis=0)
                
                # Update reference embeddings if needed
                if target_cluster.reference_embeddings is not None:
                    ref_embeddings = embeddings[target_cluster.reference_indices]
                    target_cluster.reference_embeddings = ref_embeddings
                
                # Update target embeddings if needed
                if len(target_cluster.linked_target_indices) > 0:
                    target_embeddings = embeddings[target_cluster.linked_target_indices]
                    target_cluster.linked_target_embeddings = target_embeddings
        
        # Remove small clusters
        updated_clusters = [cluster for i, cluster in enumerate(clustering_result.cluster_data_list)
                           if i not in small_clusters]
        
        # Create updated result
        updated_result = ClusteringResult(
            cluster_data_list=updated_clusters,
            cluster_labels=clustering_result.cluster_labels,  # Labels may be outdated
            cluster_centers=clustering_result.cluster_centers,  # Centers may be outdated
            quality_metrics=clustering_result.quality_metrics,
            metadata={**clustering_result.metadata, 'merged_small_clusters': len(small_clusters)}
        )
        
        logger.info(f"Cluster merging completed. {len(updated_clusters)} clusters remaining")
        
        return updated_result 