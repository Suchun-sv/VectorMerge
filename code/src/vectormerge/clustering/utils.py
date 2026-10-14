"""Utility functions for clustering operations in VectorMerge."""

import json
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
from loguru import logger

from .base import ClusteringResult, ClusterData, ClusteringConfig


def load_cluster_data(path: Union[str, Path]) -> ClusteringResult:
    """Load cluster data from disk."""
    load_path = Path(path)

    with open(load_path / "clustering_result.json", "r") as f:
        result_data = json.load(f)

    clustering_result = ClusteringResult.from_dict(result_data)

    centers_path = load_path / "cluster_centers.npy"
    if centers_path.exists():
        clustering_result.cluster_centers = np.load(centers_path)

    logger.info(f"Loaded clustering result from {load_path}")
    return clustering_result


def save_cluster_data(clustering_result: ClusteringResult,
                     path: Union[str, Path]) -> None:
    """Save cluster data to disk."""
    save_path = Path(path)
    save_path.mkdir(parents=True, exist_ok=True)

    with open(save_path / "clustering_result.json", "w") as f:
        json.dump(clustering_result.to_dict(), f, indent=2)

    if clustering_result.cluster_centers is not None:
        np.save(save_path / "cluster_centers.npy", clustering_result.cluster_centers)

    logger.info(f"Saved clustering result to {save_path}")


def compute_cluster_metrics(embeddings: np.ndarray,
                          clustering_result: ClusteringResult) -> Dict[str, float]:
    """Compute comprehensive metrics for clustering quality."""
    metrics = {}

    metrics['num_clusters'] = clustering_result.num_clusters
    metrics['total_points'] = sum(len(cluster.ref_index) + len(cluster.bound_index)
                                 for cluster in clustering_result.cluster_data_list)

    cluster_sizes = clustering_result.cluster_sizes
    if cluster_sizes:
        metrics['avg_cluster_size'] = np.mean(cluster_sizes)
        metrics['cluster_size_std'] = np.std(cluster_sizes)
        metrics['min_cluster_size'] = min(cluster_sizes)
        metrics['max_cluster_size'] = max(cluster_sizes)
        metrics['cluster_size_cv'] = metrics['cluster_size_std'] / metrics['avg_cluster_size'] if metrics['avg_cluster_size'] > 0 else 0

    total_wcss = 0.0
    for cluster_data in clustering_result.cluster_data_list:
        if len(cluster_data.ref_index) > 0 and cluster_data.center is not None:
            cluster_embeddings = embeddings[cluster_data.ref_index]
            distances_sq = np.sum((cluster_embeddings - cluster_data.center) ** 2, axis=1)
            total_wcss += np.sum(distances_sq)

    metrics['total_wcss'] = total_wcss
    metrics['avg_wcss'] = total_wcss / clustering_result.num_clusters if clustering_result.num_clusters > 0 else 0

    try:
        silhouette_score = _compute_silhouette_score(embeddings, clustering_result)
        if silhouette_score is not None:
            metrics['silhouette_score'] = silhouette_score
    except Exception as e:
        logger.warning(f"Failed to compute silhouette score: {e}")

    try:
        ch_index = _compute_calinski_harabasz_index(embeddings, clustering_result)
        if ch_index is not None:
            metrics['calinski_harabasz_index'] = ch_index
    except Exception as e:
        logger.warning(f"Failed to compute Calinski-Harabasz index: {e}")

    try:
        db_index = _compute_davies_bouldin_index(embeddings, clustering_result)
        if db_index is not None:
            metrics['davies_bouldin_index'] = db_index
    except Exception as e:
        logger.warning(f"Failed to compute Davies-Bouldin index: {e}")

    return metrics


def _compute_silhouette_score(embeddings: np.ndarray,
                             clustering_result: ClusteringResult) -> Optional[float]:
    """Compute silhouette score for clustering."""
    try:
        from sklearn.metrics import silhouette_score

        all_ref_indices = []
        all_labels = []

        for cluster_id, cluster_data in enumerate(clustering_result.cluster_data_list):
            for ref_idx in cluster_data.ref_index:
                all_ref_indices.append(ref_idx)
                all_labels.append(cluster_id)

        if len(all_ref_indices) < 2 or len(set(all_labels)) < 2:
            return None

        ref_embeddings = embeddings[all_ref_indices]
        score = silhouette_score(ref_embeddings, all_labels)

        return float(score)

    except Exception as e:
        logger.warning(f"Failed to compute silhouette score: {e}")
        return None


def _compute_calinski_harabasz_index(embeddings: np.ndarray,
                                   clustering_result: ClusteringResult) -> Optional[float]:
    """Compute Calinski-Harabasz index for clustering."""
    try:
        from sklearn.metrics import calinski_harabasz_score

        all_ref_indices = []
        all_labels = []

        for cluster_id, cluster_data in enumerate(clustering_result.cluster_data_list):
            for ref_idx in cluster_data.ref_index:
                all_ref_indices.append(ref_idx)
                all_labels.append(cluster_id)

        if len(all_ref_indices) < 2 or len(set(all_labels)) < 2:
            return None

        ref_embeddings = embeddings[all_ref_indices]
        score = calinski_harabasz_score(ref_embeddings, all_labels)

        return float(score)

    except Exception as e:
        logger.warning(f"Failed to compute Calinski-Harabasz index: {e}")
        return None


def _compute_davies_bouldin_index(embeddings: np.ndarray,
                                 clustering_result: ClusteringResult) -> Optional[float]:
    """Compute Davies-Bouldin index for clustering."""
    try:
        from sklearn.metrics import davies_bouldin_score

        all_ref_indices = []
        all_labels = []

        for cluster_id, cluster_data in enumerate(clustering_result.cluster_data_list):
            for ref_idx in cluster_data.ref_index:
                all_ref_indices.append(ref_idx)
                all_labels.append(cluster_id)

        if len(all_ref_indices) < 2 or len(set(all_labels)) < 2:
            return None

        ref_embeddings = embeddings[all_ref_indices]
        score = davies_bouldin_score(ref_embeddings, all_labels)

        return float(score)

    except Exception as e:
        logger.warning(f"Failed to compute Davies-Bouldin index: {e}")
        return None


def visualize_clusters(embeddings: np.ndarray,
                      clustering_result: ClusteringResult,
                      method: str = "tsne",
                      save_path: Optional[Union[str, Path]] = None) -> None:
    """Visualize clustering results using dimensionality reduction."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        all_ref_indices = []
        all_labels = []

        for cluster_id, cluster_data in enumerate(clustering_result.cluster_data_list):
            for ref_idx in cluster_data.ref_index:
                all_ref_indices.append(ref_idx)
                all_labels.append(cluster_id)

        if len(all_ref_indices) < 2:
            logger.warning("Not enough points for visualization")
            return

        ref_embeddings = embeddings[all_ref_indices]

        if method == "tsne":
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42)
        elif method == "pca":
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
        elif method == "umap":
            try:
                import umap.umap_ as umap
                reducer = umap.UMAP(n_components=2, random_state=42)
            except ImportError:
                logger.warning("UMAP not available, falling back to t-SNE")
                from sklearn.manifold import TSNE
                reducer = TSNE(n_components=2, random_state=42)
                method = "tsne"
        else:
            raise ValueError(f"Unsupported dimensionality reduction method: {method}")

        reduced_embeddings = reducer.fit_transform(ref_embeddings)

        plt.figure(figsize=(12, 8))

        unique_labels = np.unique(all_labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        for label, color in zip(unique_labels, colors):
            mask = np.array(all_labels) == label
            plt.scatter(reduced_embeddings[mask, 0], reduced_embeddings[mask, 1],
                       c=[color], label=f'Cluster {label}', alpha=0.7, s=50)

        if clustering_result.cluster_centers is not None and method in ["pca", "umap"]:
            center_reduced = reducer.transform(clustering_result.cluster_centers)
            plt.scatter(center_reduced[:, 0], center_reduced[:, 1],
                       c='red', marker='x', s=200, linewidths=3, label='Centers')

        plt.title(f'Clustering Visualization ({method.upper()})\n'
                 f'{clustering_result.num_clusters} clusters, {len(all_ref_indices)} points')
        plt.xlabel(f'{method.upper()} 1')
        plt.ylabel(f'{method.upper()} 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Clustering visualization saved to {save_path}")
        else:
            plt.show()

    except Exception as e:
        logger.error(f"Failed to create visualization: {e}")


def analyze_cluster_overlap(clustering_result1: ClusteringResult,
                           clustering_result2: ClusteringResult) -> Dict[str, Any]:
    """Analyze overlap between two clustering results."""
    points1 = {}
    points2 = {}

    for cluster_id, cluster_data in enumerate(clustering_result1.cluster_data_list):
        for point_id in cluster_data.ref_index:
            points1[point_id] = cluster_id

    for cluster_id, cluster_data in enumerate(clustering_result2.cluster_data_list):
        for point_id in cluster_data.ref_index:
            points2[point_id] = cluster_id

    common_points = set(points1.keys()) & set(points2.keys())

    if not common_points:
        return {
            'common_points': 0,
            'total_points_1': len(points1),
            'total_points_2': len(points2),
            'overlap_ratio': 0.0,
            'adjusted_rand_index': 0.0,
            'normalized_mutual_info': 0.0
        }

    labels1 = [points1[point_id] for point_id in common_points]
    labels2 = [points2[point_id] for point_id in common_points]

    try:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

        ari = adjusted_rand_score(labels1, labels2)
        nmi = normalized_mutual_info_score(labels1, labels2)

    except Exception as e:
        logger.warning(f"Failed to compute overlap metrics: {e}")
        ari = 0.0
        nmi = 0.0

    analysis = {
        'common_points': len(common_points),
        'total_points_1': len(points1),
        'total_points_2': len(points2),
        'overlap_ratio': len(common_points) / max(len(points1), len(points2)),
        'adjusted_rand_index': ari,
        'normalized_mutual_info': nmi
    }

    return analysis


def create_cluster_summary(clustering_result: ClusteringResult,
                          embeddings: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Create a comprehensive summary of clustering results."""
    summary = {
        'basic_info': {
            'num_clusters': clustering_result.num_clusters,
            'total_points': sum(len(cluster.ref_index) + len(cluster.bound_index)
                               for cluster in clustering_result.cluster_data_list),
            'total_reference_points': sum(len(cluster.ref_index)
                                        for cluster in clustering_result.cluster_data_list),
            'total_bound_points': sum(len(cluster.bound_index)
                                    for cluster in clustering_result.cluster_data_list)
        },
        'cluster_sizes': clustering_result.cluster_sizes,
        'cluster_statistics': {
            'avg_cluster_size': np.mean(clustering_result.cluster_sizes) if clustering_result.cluster_sizes else 0,
            'cluster_size_std': np.std(clustering_result.cluster_sizes) if clustering_result.cluster_sizes else 0,
            'min_cluster_size': min(clustering_result.cluster_sizes) if clustering_result.cluster_sizes else 0,
            'max_cluster_size': max(clustering_result.cluster_sizes) if clustering_result.cluster_sizes else 0
        },
        'quality_metrics': clustering_result.quality_metrics,
        'metadata': clustering_result.metadata
    }

    if embeddings is not None:
        try:
            quality_metrics = compute_cluster_metrics(embeddings, clustering_result)
            summary['comprehensive_metrics'] = quality_metrics
        except Exception as e:
            logger.warning(f"Failed to compute comprehensive metrics: {e}")

    return summary