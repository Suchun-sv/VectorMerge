"""
LA2M mapping strategy for VectorMerge.

This module implements the clustering-based mapping strategy from the LA2M paper,
which clusters reference points and learns local mappings for each cluster.
"""

import numpy as np
import torch
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from loguru import logger

from ..base import MappingStrategy, MappingConfig, ClusterData
from ...clustering import KMeansClusteringStrategy, LA2MClusteringStrategy
from ...clustering.base import ClusteringConfig
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
        
        # Create clustering configuration from mapping config
        cluster_method = getattr(config, 'clustering_method', getattr(config, 'cluster_method', 'kmeans'))
        clustering_config = ClusteringConfig(
            num_clusters=getattr(config, 'num_clusters', 50),
            method=cluster_method,
            min_cluster_size=getattr(config, 'min_cluster_size', 5),
            random_state=42,
            linkage=getattr(config, 'linkage', 'ward'),
            distance_threshold=getattr(config, 'distance_threshold', None),
            device=getattr(config, 'device', 'auto'),
            verbose=getattr(config, 'verbose', False),
            compute_metrics=False  # We'll compute metrics separately if needed
        )
        
        # Initialize clustering strategy
        if cluster_method == "la2m-cluster":
            self.clustering_strategy = LA2MClusteringStrategy(clustering_config)
        else:
            self.clustering_strategy = KMeansClusteringStrategy(clustering_config)
        
        # Storage for cluster data and local mappings
        self.cluster_data_list: List[ClusterData] = []
        self.local_mappings: Dict[int, Dict[str, np.ndarray]] = {}
        self.global_fallback_mapping: Optional[Dict[str, np.ndarray]] = None
        
        logger.info(f"LA2M mapping strategy initialized with {cluster_method} clustering")
    
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
        
        min_cluster_size = getattr(self.config, 'min_cluster_size', 5)
        
        for cluster_id, cluster_data in enumerate(cluster_data_list):
            if len(cluster_data.ref_index) >= min_cluster_size:  # Need minimum points for stable mapping
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
                        approximate=getattr(self.config, 'approximate', False),
                        q=getattr(self.config, 'q', 1500),
                        with_rotation=getattr(self.config, 'with_rotation', True)
                    )
                    
                    if local_mapping is not None and isinstance(local_mapping, dict):
                        self.local_mappings[cluster_id] = local_mapping
                        successful_clusters += 1
                        
                        logger.debug(f"Cluster {cluster_id}: learned mapping with {len(cluster_ref_indices)} points")
                    else:
                        logger.warning(f"Failed to get valid mapping for cluster {cluster_id}")
                
                except Exception as e:
                    logger.warning(f"Failed to learn mapping for cluster {cluster_id}: {e}")
                    # Skip this cluster - will use global fallback
            else:
                logger.warning(f"Cluster {cluster_id} has insufficient points ({len(cluster_data.ref_index)} < {min_cluster_size})")
        
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
                approximate=getattr(self.config, 'approximate', False),
                q=getattr(self.config, 'q', 1500),
                with_rotation=getattr(self.config, 'with_rotation', True)
            )
            
            if global_mapping is not None and isinstance(global_mapping, dict):
                self.global_fallback_mapping = global_mapping
                logger.info("Global fallback mapping learned successfully")
            else:
                logger.error("Failed to get valid global fallback mapping")
                raise ValueError("Could not create global fallback mapping")
        
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
            'cluster_method': getattr(self.config, 'clustering_method', getattr(self.config, 'cluster_method', 'kmeans')),
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
        
        # If no specific cluster assignment strategy, use distance to cluster centers
        if hasattr(self.clustering_strategy, 'cluster_centers') and self.clustering_strategy.cluster_centers is not None:
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
                        if self.global_fallback_mapping is not None:
                            transformed_cluster = self._apply_local_mapping(
                                cluster_embeddings, self.global_fallback_mapping
                            )
                        else:
                            transformed_cluster = cluster_embeddings  # No transformation available
                    
                    transformed[cluster_indices] = transformed_cluster
        else:
            # No cluster centers available - use global fallback for all points
            logger.warning("No cluster centers available, using global fallback for all points")
            if self.global_fallback_mapping is not None:
                transformed = self._apply_local_mapping(embeddings, self.global_fallback_mapping)
            else:
                logger.error("No global fallback mapping available")
                transformed = embeddings  # Return original embeddings as fallback
        
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
        if mapping_params is None or not isinstance(mapping_params, dict):
            return embeddings
        
        try:
            rotation_matrix = mapping_params['rotation_matrix']
            source_mean = mapping_params['source_mean']
            target_mean = mapping_params['target_mean']
            
            # Apply transformation: center, rotate, translate
            embeddings_centered = embeddings - source_mean
            transformed = np.dot(embeddings_centered, rotation_matrix.T) + target_mean
            
            return transformed
        except KeyError as e:
            logger.error(f"Missing key in mapping parameters: {e}")
            return embeddings
        except Exception as e:
            logger.error(f"Error applying local mapping: {e}")
            return embeddings
    
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
            'clustering_method': getattr(self.config, 'clustering_method', getattr(self.config, 'cluster_method', 'kmeans'))
        }
        
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
            
            for key, value in mapping.items():
                np.save(cluster_path / f"{key}.npy", value)
        
        # Save global fallback mapping
        if self.global_fallback_mapping is not None:
            global_path = save_path / "global_fallback"
            global_path.mkdir(exist_ok=True)
            
            for key, value in self.global_fallback_mapping.items():
                np.save(global_path / f"{key}.npy", value)
        
        logger.info(f"Saved LA2M mapping strategy to {save_path}")
    
    @classmethod
    def load(cls, path) -> 'LA2MStrategy':
        """Load a fitted LA2M mapping strategy."""
        load_path = Path(path)
        
        # Load basic mapping info
        import json
        with open(load_path / "mapping_info.json", "r") as f:
            mapping_info = json.load(f)
        
        config = MappingConfig.from_dict(mapping_info['config'])
        instance = cls(config)
        
        # Load cluster data
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
                mapping = {}
                for npy_file in cluster_path.glob("*.npy"):
                    key = npy_file.stem
                    mapping[key] = np.load(npy_file)
                
                if mapping:
                    instance.local_mappings[cluster_id] = mapping
        
        # Load global fallback mapping
        global_path = load_path / "global_fallback"
        if global_path.exists():
            global_mapping = {}
            for npy_file in global_path.glob("*.npy"):
                key = npy_file.stem
                global_mapping[key] = np.load(npy_file)
            
            if global_mapping:
                instance.global_fallback_mapping = global_mapping
        
        instance.is_fitted = True
        instance.metadata = mapping_info.get('metadata', {})
        
        logger.info(f"Loaded LA2M mapping strategy from {load_path}")
        return instance 