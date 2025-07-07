"""
LA2M clustering strategy implementation.

This module implements the clustering strategy used in LA2M (Local Anchor to Multiple)
where each point has a neighborhood of closest reference points, and these neighborhoods
are optimized and merged to form final clusters.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Any
from tqdm import tqdm, trange
from sklearn.cluster import KMeans
from loguru import logger

from ..base import ClusteringStrategy, ClusteringConfig, ClusteringResult, ClusterData


class LA2MClusteringStrategy(ClusteringStrategy):
    """LA2M clustering strategy using local neighborhood optimization.
    
    This strategy implements the clustering method used in LA2M approach:
    1. Creating a neighborhood for each point with its top-k closest reference points
    2. Merging neighborhoods that have identical reference point sets
    3. Optimizing neighborhoods by merging those with overlapping reference points
    4. Final merging to reach target number of clusters
    """
    
    def __init__(self, config: ClusteringConfig):
        super().__init__(config)
        self.d_prime = getattr(config, 'd_prime', 10)  # Number of top neighbors
        self.r2r_distances = None
        self.r_index2range = None
    
    def fit(self, embeddings: np.ndarray, reference_indices: np.ndarray) -> ClusteringResult:
        """Fit LA2M clustering to reference points."""
        logger.info(f"Starting LA2M clustering with {len(reference_indices)} reference points")
        
        # Convert to torch tensors
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ref_embeddings = torch.tensor(embeddings[reference_indices], device=device)
        
        # Initialize mappings
        self.r_index2range = {int(reference_indices[i]): i for i in range(len(reference_indices))}
        
        # Step 1: Pre-cluster if needed for large datasets
        if len(reference_indices) > 1000:
            cluster_assignments, _ = self._pre_cluster(ref_embeddings, self.config.num_clusters)
            neighborhoods, diameters = self._compute_neighborhoods_from_clusters(
                ref_embeddings, cluster_assignments, reference_indices
            )
        else:
            neighborhoods, diameters = self._compute_neighborhoods_individual(
                ref_embeddings, reference_indices
            )
        
        # Step 2: Merge and sort neighborhoods
        logger.info(f"Initial neighborhoods: {len(neighborhoods)}")
        sorted_neighborhoods = self._merge_and_sort_neighborhoods(neighborhoods)
        logger.info(f"After merging: {len(sorted_neighborhoods)}")
        
        # Step 3: Construct hypergraph and optimize
        edges = self._construct_hypergraph(sorted_neighborhoods)
        logger.info(f"Hypergraph edges: {len(edges)}")
        
        optimized_clusters = self._optimize_neighborhood_sets(
            sorted_neighborhoods, edges, diameters
        )
        logger.info(f"Final optimized clusters: {len(optimized_clusters)}")
        
        # Step 4: Merge by diameter if we have too many clusters
        if len(optimized_clusters) > self.config.num_clusters:
            optimized_clusters = self._merge_by_diameter(
                optimized_clusters, self.config.num_clusters
            )
        
        # Convert to ClusteringResult
        result = self._create_clustering_result(optimized_clusters, embeddings, reference_indices)
        self.is_fitted = True
        
        return result
    
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict cluster assignments for new embeddings."""
        if not self.is_fitted:
            raise ValueError("Strategy must be fitted before prediction")
        
        # Implementation would assign new points to nearest cluster centers
        # This is a simplified version - full implementation would be more complex
        if hasattr(self, 'cluster_centers_'):
            from sklearn.metrics.pairwise import euclidean_distances
            distances = euclidean_distances(embeddings, self.cluster_centers_)
            return np.argmin(distances, axis=1)
        else:
            raise NotImplementedError("Prediction not implemented for this strategy")
    
    def fit_predict(self, embeddings: np.ndarray, reference_indices: np.ndarray) -> Tuple[ClusteringResult, np.ndarray]:
        """Fit and predict cluster assignments."""
        clustering_result = self.fit(embeddings, reference_indices)
        
        # Predict assignments for all embeddings
        all_assignments = np.full(len(embeddings), -1, dtype=int)
        
        # Assign reference points to their clusters
        for cluster_id, cluster_data in enumerate(clustering_result.cluster_data_list):
            for ref_idx in cluster_data.ref_index:
                all_assignments[ref_idx] = cluster_id
        
        return clustering_result, all_assignments
    
    def _pre_cluster(self, embeddings: torch.Tensor, num_clusters: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pre-cluster embeddings using K-means."""
        kmeans = KMeans(n_clusters=num_clusters, n_init='auto', random_state=self.config.random_state)
        cluster_assignments = kmeans.fit_predict(embeddings.cpu().numpy())
        centers = torch.tensor(kmeans.cluster_centers_, device=embeddings.device)
        cluster_assignments = torch.tensor(cluster_assignments, device=embeddings.device)
        return cluster_assignments, centers
    
    def _compute_neighborhoods_from_clusters(self, embeddings: torch.Tensor, 
                                           cluster_assignments: torch.Tensor,
                                           reference_indices: np.ndarray) -> Tuple[List[ClusterData], List[float]]:
        """Compute neighborhoods from pre-clustered data."""
        neighborhoods = []
        diameters = []
        
        # Compute reference-to-reference distances
        self.r2r_distances = torch.cdist(embeddings, embeddings)
        
        unique_clusters = torch.unique(cluster_assignments)
        for cluster_id in unique_clusters:
            cluster_mask = cluster_assignments == cluster_id
            cluster_indices = torch.where(cluster_mask)[0]
            
            # Create cluster data
            cluster_data = ClusterData(
                ref_index=[int(reference_indices[i]) for i in cluster_indices],
                bound_index=[],
                diameter=0.0
            )
            
            # Compute diameter
            if len(cluster_indices) > 1:
                cluster_distances = self.r2r_distances[cluster_indices][:, cluster_indices]
                diameter = torch.max(cluster_distances).item()
            else:
                diameter = 0.0
            
            cluster_data.diameter = diameter
            neighborhoods.append(cluster_data)
            diameters.append(diameter)
        
        return neighborhoods, diameters
    
    def _compute_neighborhoods_individual(self, embeddings: torch.Tensor,
                                        reference_indices: np.ndarray) -> Tuple[List[ClusterData], List[float]]:
        """Compute neighborhoods for individual points."""
        neighborhoods = []
        diameters = []
        
        # Compute reference-to-reference distances
        self.r2r_distances = torch.cdist(embeddings, embeddings)
        
        for i in range(len(embeddings)):
            distances = self.r2r_distances[i]
            
            # Get top-d neighbors
            top_indices = torch.argsort(distances)[:self.d_prime]
            
            # Create neighborhood
            cluster_data = ClusterData(
                ref_index=[int(reference_indices[j]) for j in top_indices],
                bound_index=[],
                diameter=0.0
            )
            
            # Compute diameter
            if len(top_indices) > 1:
                neighborhood_distances = self.r2r_distances[top_indices][:, top_indices]
                diameter = torch.max(neighborhood_distances).item()
            else:
                diameter = 0.0
            
            cluster_data.diameter = diameter
            neighborhoods.append(cluster_data)
            diameters.append(diameter)
        
        return neighborhoods, diameters
    
    def _merge_and_sort_neighborhoods(self, neighborhoods: List[ClusterData]) -> List[ClusterData]:
        """Merge neighborhoods with same reference indices and sort by diameter."""
        merged_neighborhoods = []
        ref_index_map = {}
        
        for neighborhood in neighborhoods:
            ref_index_set = frozenset(neighborhood.ref_index)
            if ref_index_set not in ref_index_map:
                merged_neighborhoods.append(neighborhood)
                ref_index_map[ref_index_set] = neighborhood
            else:
                # Merge bound indices
                existing = ref_index_map[ref_index_set]
                existing.bound_index.extend(neighborhood.bound_index)
                # Keep the maximum diameter
                existing.diameter = max(existing.diameter, neighborhood.diameter)
        
        # Sort by diameter in descending order
        sorted_neighborhoods = sorted(merged_neighborhoods, key=lambda x: x.diameter, reverse=True)
        return sorted_neighborhoods
    
    def _construct_hypergraph(self, neighborhoods: List[ClusterData]) -> List[Tuple[int, int]]:
        """Construct hypergraph edges between neighborhoods with overlapping references."""
        edges = []
        
        for i in tqdm(range(len(neighborhoods)), desc="Constructing hypergraph"):
            Si_set = set(neighborhoods[i].ref_index)
            for j in range(i + 1, len(neighborhoods)):
                Sj_set = set(neighborhoods[j].ref_index)
                if Si_set & Sj_set:  # If there's an intersection
                    edges.append((i, j))
                    edges.append((j, i))
        
        return edges
    
    def _optimize_neighborhood_sets(self, neighborhoods: List[ClusterData], 
                                   edges: List[Tuple[int, int]], 
                                   diameters: List[float]) -> List[ClusterData]:
        """Optimize neighborhood sets using hypergraph structure."""
        if self.r2r_distances is None or self.r_index2range is None:
            logger.warning("Distance matrix or index mapping not initialized")
            return neighborhoods
            
        # Create adjacency list
        adj_list = {}
        for i, j in edges:
            if i not in adj_list:
                adj_list[i] = []
            adj_list[i].append(j)
        
        # Track which neighborhoods are still active
        active = [True] * len(neighborhoods)
        
        # Process neighborhoods in order of decreasing diameter
        for i in tqdm(range(len(neighborhoods)), desc="Optimizing neighborhoods"):
            if not active[i]:
                continue
            
            # Try to merge with neighbors
            if i in adj_list:
                for j in adj_list[i]:
                    if not active[j]:
                        continue
                    
                    # Test merge
                    combined_refs = list(set(neighborhoods[i].ref_index + neighborhoods[j].ref_index))
                    combined_ref_indices = [self.r_index2range[ref] for ref in combined_refs if ref in self.r_index2range]
                    
                    if len(combined_ref_indices) > 1:
                        combined_distances = self.r2r_distances[combined_ref_indices][:, combined_ref_indices]
                        new_diameter = torch.max(combined_distances).item()
                    else:
                        new_diameter = 0.0
                    
                    # Merge if diameter doesn't increase too much
                    if new_diameter <= diameters[i] * 1.1:  # Allow 10% increase
                        neighborhoods[i].ref_index = combined_refs
                        neighborhoods[i].bound_index.extend(neighborhoods[j].bound_index)
                        neighborhoods[i].diameter = new_diameter
                        active[j] = False
                        diameters[i] = new_diameter
        
        return [neighborhoods[i] for i in range(len(neighborhoods)) if active[i]]
    
    def _merge_by_diameter(self, neighborhoods: List[ClusterData], 
                          target_clusters: int) -> List[ClusterData]:
        """Merge neighborhoods to reach target number of clusters."""
        if self.r2r_distances is None or self.r_index2range is None:
            logger.warning("Distance matrix or index mapping not initialized")
            return neighborhoods[:target_clusters]  # Return first N clusters as fallback
            
        clusters = neighborhoods.copy()
        
        while len(clusters) > target_clusters:
            # Find the two clusters with minimum average distance
            min_dist = float('inf')
            merge_i, merge_j = -1, -1
            
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    # Calculate average distance between clusters
                    refs_i = [self.r_index2range[ref] for ref in clusters[i].ref_index if ref in self.r_index2range]
                    refs_j = [self.r_index2range[ref] for ref in clusters[j].ref_index if ref in self.r_index2range]
                    
                    if refs_i and refs_j:
                        distances = self.r2r_distances[refs_i][:, refs_j]
                        avg_dist = torch.mean(distances).item()
                        
                        if avg_dist < min_dist:
                            min_dist = avg_dist
                            merge_i, merge_j = i, j
            
            if merge_i == -1:
                break
            
            # Merge clusters
            clusters[merge_i].ref_index.extend(clusters[merge_j].ref_index)
            clusters[merge_i].bound_index.extend(clusters[merge_j].bound_index)
            clusters[merge_i].diameter = max(clusters[merge_i].diameter, clusters[merge_j].diameter)
            clusters.pop(merge_j)
        
        return clusters
    
    def _create_clustering_result(self, clusters: List[ClusterData], 
                                 embeddings: np.ndarray,
                                 reference_indices: np.ndarray) -> ClusteringResult:
        """Create ClusteringResult from optimized clusters."""
        # Create cluster labels
        cluster_labels = np.full(len(reference_indices), -1, dtype=int)
        
        for cluster_id, cluster_data in enumerate(clusters):
            for ref_idx in cluster_data.ref_index:
                # Find position in reference_indices
                pos = np.where(reference_indices == ref_idx)[0]
                if len(pos) > 0:
                    cluster_labels[pos[0]] = cluster_id
        
        # Compute cluster centers
        cluster_centers = []
        for cluster_data in clusters:
            if cluster_data.ref_index:
                cluster_embeddings = embeddings[cluster_data.ref_index]
                center = np.mean(cluster_embeddings, axis=0)
                cluster_centers.append(center)
                cluster_data.center = center
        
        cluster_centers = np.array(cluster_centers) if cluster_centers else None
        self.cluster_centers_ = cluster_centers
        
        return ClusteringResult(
            cluster_data_list=clusters,
            cluster_labels=cluster_labels,
            cluster_centers=cluster_centers,
            metadata={
                'method': 'la2m-cluster',
                'd_prime': self.d_prime,
                'num_reference_points': len(reference_indices)
            }
        ) 