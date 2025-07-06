"""
LA2M mapping strategy for VectorMerge.

This module implements the clustering-based mapping strategy from the LA2M paper,
which clusters reference points and learns local mappings for each cluster.
"""

import numpy as np
import torch
from typing import List, Optional, Dict, Any
from pathlib import Path
from loguru import logger

from ..base import MappingStrategy, MappingConfig, ClusterData
from ..clustering import KMeansClusteringStrategy, HierarchicalClusteringStrategy
from .procrustes import procrustes_mapping_torch


class LA2MStrategy(MappingStrategy):
    """
    Clustering-based mapping strategy (LA2M method from the paper).
    
    This strategy:
    1. Clusters reference points using K-means or hierarchical clustering
    2. Assigns target points to clusters
    3. Learns local Procrustes mappings for each cluster
    4. Applies appropriate local mapping based on cluster assignment
    """
    
    def __init__(self, config: MappingConfig):
        """Initialize the LA2M mapping strategy.
        
        Args:
            config: Configuration object
        """
        super().__init__(config)
        
        # Initialize clustering strategy
        if config.cluster_method == "hierarchical":
            self.clustering_strategy = HierarchicalClusteringStrategy(config)
        else:
            self.clustering_strategy = KMeansClusteringStrategy(config)
        
        # Storage for cluster data and local mappings
        self.cluster_data_list: List[ClusterData] = []
        self.local_mappings: Dict[int, Dict[str, np.ndarray]] = {}
        self.global_fallback_mapping: Optional[Dict[str, np.ndarray]] = None
        
        logger.info(f"LA2M mapping strategy initialized with {config.cluster_method} clustering")
    
    def fit(self, source_embeddings: np.ndarray, target_embeddings: np.ndarray,
            reference_indices: np.ndarray, **kwargs) -> None:
        """Fit the clustering-based mapping strategy.
        
        Args:
            source_embeddings: Source embedding space
            target_embeddings: Target embedding space
            reference_indices: Reference point indices for training
            **kwargs: Additional arguments
        """
        logger.info(f"Fitting LA2M mapping strategy with {len(reference_indices)} reference points")
        
        # Step 1: Cluster reference points
        logger.info("Step 1: Clustering reference points...")
        cluster_data_list, cluster_labels = self.clustering_strategy.cluster_reference(
            source_embeddings, reference_indices
        )
        
        # Step 2: Learn local mappings for each cluster
        logger.info("Step 2: Learning local mappings for each cluster...")
        self.local_mappings = {}
        successful_clusters = 0
        
        for cluster_id, cluster_data in enumerate(cluster_data_list):
            if len(cluster_data.ref_index) >= 2:  # Need at least 2 points for mapping
                try:
                    # Extract cluster reference embeddings
                    cluster_ref_indices = np.array(cluster_data.ref_index)
                    cluster_source_ref = source_embeddings[cluster_ref_indices]
                    cluster_target_ref = target_embeddings[cluster_ref_indices]
                    
                    # Learn local Procrustes mapping for this cluster
                    _, local_mapping = procrustes_mapping_torch(
                        source_embeddings=cluster_source_ref,
                        target_embeddings=cluster_target_ref,
                        overlap_ids=np.arange(len(cluster_ref_indices)),
                        source_bound=cluster_source_ref,
                        target_bound=cluster_target_ref,
                        approximate=self.config.approximate,
                        q=self.config.q,
                        with_rotation=self.config.with_rotation
                    )
                    
                    self.local_mappings[cluster_id] = local_mapping
                    successful_clusters += 1
                    
                    logger.debug(f"Cluster {cluster_id}: learned mapping with {len(cluster_ref_indices)} points")
                
                except Exception as e:
                    logger.warning(f"Failed to learn mapping for cluster {cluster_id}: {e}")
                    # Skip this cluster - will use global fallback
            else:
                logger.warning(f"Cluster {cluster_id} has insufficient points ({len(cluster_data.ref_index)})")
        
        # Step 3: Learn global fallback mapping using all reference points
        logger.info("Step 3: Learning global fallback mapping...")
        try:
            source_ref = source_embeddings[reference_indices]
            target_ref = target_embeddings[reference_indices]
            
            _, global_mapping = procrustes_mapping_torch(
                source_embeddings=source_ref,
                target_embeddings=target_ref,
                overlap_ids=np.arange(len(reference_indices)),
                source_bound=source_ref,
                target_bound=target_ref,
                approximate=self.config.approximate,
                q=self.config.q,
                with_rotation=self.config.with_rotation
            )
            
            self.global_fallback_mapping = global_mapping
            logger.info("Global fallback mapping learned successfully")
        
        except Exception as e:
            logger.error(f"Failed to learn global fallback mapping: {e}")
            raise
        
        # Step 4: Store cluster data for later use
        self.cluster_data_list = cluster_data_list
        
        self.is_fitted = True
        self.metadata = {
            'reference_size': len(reference_indices),
            'num_clusters': len(cluster_data_list),
            'successful_local_mappings': successful_clusters,
            'cluster_method': self.config.cluster_method,
            'cluster_sizes': [len(cluster.ref_index) for cluster in cluster_data_list],
            'clustering_info': self.clustering_strategy.get_cluster_info()
        }
        
        logger.info(f"LA2M mapping strategy fitted successfully. "
                   f"Local mappings: {successful_clusters}/{len(cluster_data_list)}")
    
    def transform(self, embeddings: np.ndarray, 
                 target_indices: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """Transform embeddings using the clustering-based mapping.
        
        Args:
            embeddings: Embeddings to transform
            target_indices: Optional indices of target points (for cluster assignment)
            **kwargs: Additional arguments
            
        Returns:
            Transformed embeddings
        """
        if not self.is_fitted:
            raise ValueError("Mapping must be fitted before transformation")
        
        logger.info(f"Transforming {len(embeddings)} embeddings using LA2M strategy")
        
        # Step 1: Assign points to clusters
        if target_indices is not None:
            # Use the clustering strategy to assign target points
            updated_clusters = self.clustering_strategy.assign_to_clusters(
                embeddings, target_indices, self.cluster_data_list, 
                embeddings[np.array([idx for cluster in self.cluster_data_list 
                                   for idx in cluster.ref_index])]
            )
            
            # Create mapping from global index to cluster assignment
            point_to_cluster = {}
            for cluster_id, cluster_data in enumerate(updated_clusters):
                for idx in cluster_data.bound_index:
                    point_to_cluster[idx] = cluster_id
            
            # Transform points based on cluster assignment
            transformed = np.zeros_like(embeddings)
            points_transformed = set()
            
            for cluster_id, cluster_data in enumerate(updated_clusters):
                if cluster_id in self.local_mappings and len(cluster_data.bound_index) > 0:
                    # Use local mapping for this cluster
                    cluster_indices = np.array(cluster_data.bound_index)
                    cluster_embeddings = embeddings[cluster_indices]
                    
                    transformed_cluster = self._apply_local_mapping(
                        cluster_embeddings, self.local_mappings[cluster_id]
                    )
                    
                    transformed[cluster_indices] = transformed_cluster
                    points_transformed.update(cluster_indices)
                    
                    logger.debug(f"Cluster {cluster_id}: transformed {len(cluster_indices)} points")
            
            # Use global fallback for remaining points
            remaining_indices = [i for i in range(len(embeddings)) if i not in points_transformed]
            if remaining_indices and self.global_fallback_mapping is not None:
                remaining_embeddings = embeddings[remaining_indices]
                transformed_remaining = self._apply_local_mapping(
                    remaining_embeddings, self.global_fallback_mapping
                )
                transformed[remaining_indices] = transformed_remaining
                
                logger.debug(f"Global fallback: transformed {len(remaining_indices)} points")
        
        else:
            # No target indices provided - assign all points to clusters
            if self.clustering_strategy.cluster_centers is None:
                logger.warning("No cluster centers available, using global fallback for all points")
                transformed = self._apply_local_mapping(embeddings, self.global_fallback_mapping)
            else:
                # Assign all points to nearest cluster centers
                from sklearn.metrics import pairwise_distances
                distances = pairwise_distances(embeddings, self.clustering_strategy.cluster_centers)
                cluster_assignments = np.argmin(distances, axis=1)
                
                transformed = np.zeros_like(embeddings)
                
                for cluster_id in range(len(self.cluster_data_list)):
                    cluster_mask = (cluster_assignments == cluster_id)
                    cluster_indices = np.where(cluster_mask)[0]
                    
                    if len(cluster_indices) > 0:
                        cluster_embeddings = embeddings[cluster_indices]
                        
                        if cluster_id in self.local_mappings:
                            # Use local mapping
                            transformed_cluster = self._apply_local_mapping(
                                cluster_embeddings, self.local_mappings[cluster_id]
                            )
                        else:
                            # Use global fallback
                            transformed_cluster = self._apply_local_mapping(
                                cluster_embeddings, self.global_fallback_mapping
                            )
                        
                        transformed[cluster_indices] = transformed_cluster
        
        return transformed
    
    def _apply_local_mapping(self, embeddings: np.ndarray, 
                           mapping_params: Dict[str, np.ndarray]) -> np.ndarray:
        """Apply local Procrustes mapping to embeddings.
        
        Args:
            embeddings: Embeddings to transform
            mapping_params: Local mapping parameters
            
        Returns:
            Transformed embeddings
        """
        if mapping_params is None:
            return embeddings
        
        rotation_matrix = mapping_params['rotation_matrix']
        source_mean = mapping_params['source_mean']
        target_mean = mapping_params['target_mean']
        
        # Apply transformation: center, rotate, translate
        embeddings_centered = embeddings - source_mean
        transformed = np.dot(embeddings_centered, rotation_matrix.T) + target_mean
        
        return transformed
    
    def get_cluster_assignments(self, embeddings: np.ndarray) -> np.ndarray:
        """Get cluster assignments for embeddings.
        
        Args:
            embeddings: Embeddings to assign to clusters
            
        Returns:
            Cluster assignments
        """
        if not self.is_fitted:
            raise ValueError("Mapping must be fitted before getting cluster assignments")
        
        return self.clustering_strategy.predict_cluster(embeddings)
    
    def get_cluster_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics about the clustering and mappings.
        
        Returns:
            Dictionary with cluster statistics
        """
        if not self.is_fitted:
            return {}
        
        stats = {
            'num_clusters': len(self.cluster_data_list),
            'successful_local_mappings': len(self.local_mappings),
            'cluster_sizes': [len(cluster.ref_index) for cluster in self.cluster_data_list],
            'cluster_diameters': [cluster.diameter for cluster in self.cluster_data_list],
            'has_global_fallback': self.global_fallback_mapping is not None,
            'clustering_method': self.config.cluster_method
        }
        
        # Add clustering quality metrics if available
        if hasattr(self.clustering_strategy, 'compute_cluster_quality_metrics'):
            dummy_embeddings = np.random.randn(100, 10)  # Dummy for interface
            quality_metrics = self.clustering_strategy.compute_cluster_quality_metrics(
                dummy_embeddings, self.cluster_data_list
            )
            stats['quality_metrics'] = quality_metrics
        
        return stats
    
    def save(self, path) -> None:
        """Save the fitted LA2M mapping strategy."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted mapping")
        
        super().save(path)
        
        save_path = Path(path)
        
        # Save cluster data
        cluster_data_dict = {}
        for i, cluster_data in enumerate(self.cluster_data_list):
            cluster_data_dict[i] = {
                'ref_index': cluster_data.ref_index,
                'bound_index': cluster_data.bound_index,
                'diameter': cluster_data.diameter,
                'center': cluster_data.center.tolist() if cluster_data.center is not None else None
            }
        
        import json
        with open(save_path / "cluster_data.json", "w") as f:
            json.dump(cluster_data_dict, f, indent=2)
        
        # Save local mappings
        for cluster_id, mapping in self.local_mappings.items():
            cluster_path = save_path / f"cluster_{cluster_id}"
            cluster_path.mkdir(exist_ok=True)
            
            np.save(cluster_path / "rotation_matrix.npy", mapping['rotation_matrix'])
            np.save(cluster_path / "source_mean.npy", mapping['source_mean'])
            np.save(cluster_path / "target_mean.npy", mapping['target_mean'])
        
        # Save global fallback mapping
        if self.global_fallback_mapping is not None:
            global_path = save_path / "global_fallback"
            global_path.mkdir(exist_ok=True)
            
            np.save(global_path / "rotation_matrix.npy", self.global_fallback_mapping['rotation_matrix'])
            np.save(global_path / "source_mean.npy", self.global_fallback_mapping['source_mean'])
            np.save(global_path / "target_mean.npy", self.global_fallback_mapping['target_mean'])
        
        logger.info(f"Saved LA2M mapping strategy to {save_path}")
    
    @classmethod
    def load(cls, path) -> 'LA2MStrategy':
        """Load a fitted LA2M mapping strategy."""
        instance = super().load(path)
        
        load_path = Path(path)
        
        # Load cluster data
        import json
        with open(load_path / "cluster_data.json", "r") as f:
            cluster_data_dict = json.load(f)
        
        instance.cluster_data_list = []
        for i in range(len(cluster_data_dict)):
            cluster_info = cluster_data_dict[str(i)]
            cluster_data = ClusterData(
                ref_index=cluster_info['ref_index'],
                bound_index=cluster_info['bound_index'],
                diameter=cluster_info['diameter'],
                center=np.array(cluster_info['center']) if cluster_info['center'] else None
            )
            instance.cluster_data_list.append(cluster_data)
        
        # Load local mappings
        instance.local_mappings = {}
        for cluster_id in range(len(instance.cluster_data_list)):
            cluster_path = load_path / f"cluster_{cluster_id}"
            if cluster_path.exists():
                rotation_matrix = np.load(cluster_path / "rotation_matrix.npy")
                source_mean = np.load(cluster_path / "source_mean.npy")
                target_mean = np.load(cluster_path / "target_mean.npy")
                
                instance.local_mappings[cluster_id] = {
                    'rotation_matrix': rotation_matrix,
                    'source_mean': source_mean,
                    'target_mean': target_mean
                }
        
        # Load global fallback mapping
        global_path = load_path / "global_fallback"
        if global_path.exists():
            rotation_matrix = np.load(global_path / "rotation_matrix.npy")
            source_mean = np.load(global_path / "source_mean.npy")
            target_mean = np.load(global_path / "target_mean.npy")
            
            instance.global_fallback_mapping = {
                'rotation_matrix': rotation_matrix,
                'source_mean': source_mean,
                'target_mean': target_mean
            }
        
        # Reinitialize clustering strategy
        config = MappingConfig.from_dict(instance.config.to_dict())
        if config.cluster_method == "hierarchical":
            instance.clustering_strategy = HierarchicalClusteringStrategy(config)
        else:
            instance.clustering_strategy = KMeansClusteringStrategy(config)
        
        logger.info(f"Loaded LA2M mapping strategy from {load_path}")
        return instance 