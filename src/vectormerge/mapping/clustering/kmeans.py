"""
K-means clustering strategy for VectorMerge mapping.

This module implements K-means clustering used in the LA2M mapping strategy
for clustering reference points and assigning target points to clusters.
"""

import numpy as np
from typing import List, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from loguru import logger

from ..base import ClusteringStrategy, ClusterData, MappingConfig


class KMeansClusteringStrategy(ClusteringStrategy):
    """K-means clustering strategy for reference point clustering."""
    
    def __init__(self, config: MappingConfig):
        """Initialize K-means clustering strategy.
        
        Args:
            config: Configuration object containing clustering parameters
        """
        super().__init__(config)
        self.kmeans_model: Optional[KMeans] = None
        self.cluster_centers: Optional[np.ndarray] = None
        
        # K-means specific parameters
        self.n_clusters = config.num_clusters
        self.random_state = 42
        self.max_iter = 300
        self.tol = 1e-4
        
        logger.info(f"K-means clustering initialized with {self.n_clusters} clusters")
    
    def cluster_reference(self, embeddings: np.ndarray, 
                         reference_indices: np.ndarray) -> Tuple[List[ClusterData], np.ndarray]:
        """Cluster reference embeddings using K-means.
        
        Args:
            embeddings: All embeddings (N x D)
            reference_indices: Indices of reference points
            
        Returns:
            Tuple of (cluster_data_list, cluster_labels)
        """
        logger.info(f"Clustering {len(reference_indices)} reference points into {self.n_clusters} clusters")
        
        # Extract reference embeddings
        reference_embeddings = embeddings[reference_indices]
        
        # Adjust number of clusters if we have fewer reference points
        n_clusters = min(self.n_clusters, len(reference_indices))
        if n_clusters < self.n_clusters:
            logger.warning(f"Reducing number of clusters from {self.n_clusters} to {n_clusters} "
                          f"due to insufficient reference points")
        
        # Perform K-means clustering
        self.kmeans_model = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            max_iter=self.max_iter,
            tol=self.tol,
            n_init=10
        )
        
        cluster_labels = self.kmeans_model.fit_predict(reference_embeddings)
        self.cluster_centers = self.kmeans_model.cluster_centers_
        
        # Create cluster data structures
        cluster_data_list = []
        for cluster_id in range(n_clusters):
            # Find reference points in this cluster
            cluster_mask = (cluster_labels == cluster_id)
            cluster_ref_indices = reference_indices[cluster_mask]
            
            # Compute cluster diameter (maximum distance between points in cluster)
            if len(cluster_ref_indices) > 1:
                cluster_embeddings = embeddings[cluster_ref_indices]
                distances = pairwise_distances(cluster_embeddings)
                diameter = np.max(distances)
            else:
                diameter = 0.0
            
            # Create cluster data
            cluster_data = ClusterData(
                ref_index=cluster_ref_indices.tolist(),
                bound_index=[],  # Will be populated later
                diameter=diameter,
                center=self.cluster_centers[cluster_id].copy()
            )
            cluster_data_list.append(cluster_data)
        
        logger.info(f"K-means clustering completed. "
                   f"Cluster sizes: {[len(cluster.ref_index) for cluster in cluster_data_list]}")
        
        return cluster_data_list, cluster_labels
    
    def assign_to_clusters(self, embeddings: np.ndarray, target_indices: np.ndarray,
                          cluster_data_list: List[ClusterData], 
                          reference_embeddings: np.ndarray) -> List[ClusterData]:
        """Assign target points to clusters based on distance to cluster centers.
        
        Args:
            embeddings: All embeddings
            target_indices: Indices of target points to assign
            cluster_data_list: Existing cluster data with reference points
            reference_embeddings: Reference embeddings (not used in K-means)
            
        Returns:
            Updated cluster data list with assigned target points
        """
        if self.cluster_centers is None:
            raise ValueError("Must call cluster_reference first")
        
        logger.info(f"Assigning {len(target_indices)} target points to clusters")
        
        # Extract target embeddings
        target_embeddings = embeddings[target_indices]
        
        # Compute distances from target points to cluster centers
        distances = pairwise_distances(target_embeddings, self.cluster_centers)
        
        # Assign each target point to the nearest cluster
        cluster_assignments = np.argmin(distances, axis=1)
        
        # Update cluster data with assigned target points
        updated_clusters = []
        for cluster_id, cluster_data in enumerate(cluster_data_list):
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
        
        # Log cluster assignment statistics
        assignment_counts = [len(cluster.bound_index) for cluster in updated_clusters]
        logger.info(f"Target point assignment completed. "
                   f"Points per cluster: {assignment_counts}")
        
        return updated_clusters
    
    def predict_cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict cluster assignments for new embeddings.
        
        Args:
            embeddings: New embeddings to assign to clusters
            
        Returns:
            Cluster assignments for each embedding
        """
        if self.kmeans_model is None:
            raise ValueError("Must call cluster_reference first")
        
        return self.kmeans_model.predict(embeddings)
    
    def get_cluster_info(self) -> dict:
        """Get information about the fitted K-means model.
        
        Returns:
            Dictionary with cluster information
        """
        if self.kmeans_model is None:
            return {}
        
        return {
            'n_clusters': self.kmeans_model.n_clusters,
            'inertia': self.kmeans_model.inertia_,
            'n_iter': self.kmeans_model.n_iter_,
            'cluster_centers_shape': self.cluster_centers.shape if self.cluster_centers is not None else None
        }
    
    def compute_cluster_quality_metrics(self, embeddings: np.ndarray, 
                                      cluster_data_list: List[ClusterData]) -> dict:
        """Compute quality metrics for the clustering.
        
        Args:
            embeddings: All embeddings
            cluster_data_list: List of cluster data
            
        Returns:
            Dictionary with quality metrics
        """
        if not cluster_data_list:
            return {}
        
        # Compute within-cluster sum of squares
        total_wcss = 0.0
        cluster_wcss = []
        
        for cluster_data in cluster_data_list:
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
        
        # Compute between-cluster sum of squares
        if self.cluster_centers is not None and len(self.cluster_centers) > 1:
            center_distances = pairwise_distances(self.cluster_centers)
            # Average distance between cluster centers
            avg_center_distance = np.mean(center_distances[np.triu_indices_from(center_distances, k=1)])
        else:
            avg_center_distance = 0.0
        
        return {
            'total_wcss': total_wcss,
            'cluster_wcss': cluster_wcss,
            'avg_center_distance': avg_center_distance,
            'inertia': self.kmeans_model.inertia_ if self.kmeans_model else None,
            'silhouette_score': self._compute_silhouette_score(embeddings, cluster_data_list)
        }
    
    def _compute_silhouette_score(self, embeddings: np.ndarray, 
                                 cluster_data_list: List[ClusterData]) -> Optional[float]:
        """Compute silhouette score for the clustering.
        
        Args:
            embeddings: All embeddings
            cluster_data_list: List of cluster data
            
        Returns:
            Silhouette score or None if cannot compute
        """
        try:
            from sklearn.metrics import silhouette_score
            
            # Collect all reference points and their cluster labels
            all_ref_indices = []
            all_labels = []
            
            for cluster_id, cluster_data in enumerate(cluster_data_list):
                all_ref_indices.extend(cluster_data.ref_index)
                all_labels.extend([cluster_id] * len(cluster_data.ref_index))
            
            if len(all_ref_indices) > 1 and len(set(all_labels)) > 1:
                ref_embeddings = embeddings[all_ref_indices]
                return silhouette_score(ref_embeddings, all_labels)
            else:
                return None
        except ImportError:
            logger.warning("scikit-learn not available for silhouette score computation")
            return None
        except Exception as e:
            logger.warning(f"Error computing silhouette score: {e}")
            return None 