"""LA2M mapping strategy for VectorMerge."""

import numpy as np
import torch
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from loguru import logger
import time
from tqdm import tqdm
from sklearn.decomposition import PCA

from vectormerge.clustering import (ClusterData, ClusteringConfig, ClusterManager,
                                    estimate_intrinsic_dimension)
from ..base import MappingStrategy, MappingConfig
from .procrustes import procrustes_mapping_torch

class LA2MStrategy(MappingStrategy):
    """Clustering-based mapping strategy."""

    def __init__(self, config: MappingConfig, clustering_manager: ClusterManager):
        """Initialize the LA2M mapping strategy."""
        super().__init__(config)

        cluster_method = config.la2m_config.cluster_method

        self.cluster_manager = clustering_manager
        self.cluster_method = cluster_method

        self.cluster_data_list: List[ClusterData] = []
        self.local_mappings: Dict[int, Dict[str, np.ndarray]] = {}
        self.global_fallback_mapping: Optional[Dict[str, np.ndarray]] = None
        self.effective_pca_dim: int = config.la2m_config.pca_dim

        logger.info(f"LA2M mapping strategy initialized with {cluster_method} clustering")

    def _init_pca_mapping(self, source_embeddings: np.ndarray, target_embeddings: np.ndarray):
        """Initialize PCA mapping."""
        self.src_pca = PCA(n_components=self.effective_pca_dim)
        self.target_pca = PCA(n_components=self.effective_pca_dim)
        self.src_pca.fit(source_embeddings)
        self.target_pca.fit(target_embeddings)
        return self.src_pca.transform(source_embeddings), self.target_pca.transform(target_embeddings)

    def _fit(self, source_embeddings: np.ndarray, target_embeddings: np.ndarray,
            reference_indices: np.ndarray, **kwargs) -> None:
        """Fit the clustering-based mapping strategy."""
        logger.info(f"Fitting LA2M mapping strategy with {len(reference_indices)} reference points")

        ori_source_dimension = source_embeddings.shape[1]
        ori_target_dimension = target_embeddings.shape[1]

        assert ori_source_dimension == ori_target_dimension, f"Source and target embeddings have different dimensions: {ori_source_dimension} and {ori_target_dimension}"

        self.effective_pca_dim = self.config.la2m_config.pca_dim
        if self.config.la2m_config.use_intrinsic_dimension:
            d_hat = estimate_intrinsic_dimension(source_embeddings[reference_indices])
            if np.isfinite(d_hat):
                d = int(round(d_hat))
                d = max(2, min(d, len(reference_indices), ori_source_dimension))
                logger.info(f"Intrinsic dimension of references: {d_hat:.2f} -> |X| = pca_dim = {d}")
                self.effective_pca_dim = d
                self.cluster_manager.strategy.d_prime = d
                self.metadata['intrinsic_dimension'] = float(d_hat)
        self.metadata['effective_pca_dim'] = self.effective_pca_dim

        if self._global_pca_enabled():
            logger.info(f"Step 0: global PCA mapping to {self.effective_pca_dim} dimensions")
            source_embeddings, target_embeddings = self._init_pca_mapping(source_embeddings, target_embeddings)
            self.reduced_source_embeddings = source_embeddings
            self.reduced_target_embeddings = target_embeddings

        logger.info("Step 1: Clustering reference points...")
        clustering_results = self.cluster_manager.fit(embeddings=source_embeddings, save_path=self.cluster_manager.final_save_path, reference_indices=reference_indices)

        logger.info("Step 2: Learning local mappings for each cluster...")
        self.local_mappings = {}
        successful_clusters = 0

        min_cluster_size = getattr(self.config, 'min_cluster_size', 5)

        training_time_start = time.time()
        batched = None
        if self._local_pca_enabled():
            batched = self._fit_local_mappings_batched(
                source_embeddings, target_embeddings, clustering_results.cluster_data_list)
        if batched is not None:
            successful_clusters = batched
            logger.info(f"Fitted {batched} local subspaces and isometries in one batched pass")
        else:
            for cluster_id, cluster_data in tqdm(enumerate(clustering_results.cluster_data_list), total=len(clustering_results.cluster_data_list)):
                if len(cluster_data.reference_indices) >= min_cluster_size:
                    try:
                        cluster_ref_indices = np.array(cluster_data.reference_indices)
                        cluster_source_ref = source_embeddings[cluster_ref_indices]
                        cluster_target_ref = target_embeddings[cluster_ref_indices]

                        cluster_pca = None
                        if self._local_pca_enabled():
                            cluster_pca = self._fit_cluster_pca(cluster_source_ref, cluster_target_ref)
                            if cluster_pca is not None:
                                cluster_source_ref = cluster_pca["src"].transform(cluster_source_ref)
                                cluster_target_ref = cluster_pca["tgt"].transform(cluster_target_ref)

                        _, local_mapping = procrustes_mapping_torch(
                            source_embeddings=cluster_source_ref,
                            target_embeddings=cluster_target_ref,
                            overlap_ids=np.arange(len(cluster_ref_indices)),
                            source_bound=cluster_source_ref,
                            target_bound=cluster_target_ref,
                            approximate=getattr(self.config.la2m_config, 'approximate', False),
                            q=getattr(self.config.la2m_config, 'q', 1500),
                            with_rotation=getattr(self.config.la2m_config, 'with_rotation', True),
                            with_scaling=getattr(self.config.la2m_config, 'with_scaling', True)
                        )

                        if local_mapping is not None and isinstance(local_mapping, dict):
                            if getattr(self.config.la2m_config, "transpose_local", False):
                                local_mapping = dict(local_mapping)
                                local_mapping["rotation_matrix"] = local_mapping["rotation_matrix"].T
                            if cluster_pca is not None:
                                local_mapping = dict(local_mapping)
                                local_mapping["cluster_pca"] = cluster_pca
                            self.local_mappings[cluster_id] = local_mapping
                            successful_clusters += 1

                            elapsed_time = time.time() - training_time_start
                            if successful_clusters > 0:
                                avg_time_per_cluster = elapsed_time / successful_clusters
                                remaining_clusters = len(clustering_results.cluster_data_list) - cluster_id - 1
                                estimated_remaining_time = avg_time_per_cluster * remaining_clusters

                                logger.debug(f"Cluster {cluster_id}: learned mapping with {len(cluster_ref_indices)} points, "
                                          f"remaining {remaining_clusters} clusters, "
                                          f"estimated time remaining: {time.strftime('%H:%M:%S', time.gmtime(estimated_remaining_time))}")
                            else:
                                logger.debug(f"Cluster {cluster_id}: learned mapping with {len(cluster_ref_indices)} points")
                        else:
                            logger.warning(f"Failed to get valid mapping for cluster {cluster_id}")

                    except Exception as e:
                        logger.warning(f"Failed to learn mapping for cluster {cluster_id}: {e}")
                else:
                    logger.warning(f"Cluster {cluster_id} has insufficient points ({len(cluster_data.reference_indices)} < {min_cluster_size})")

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
                approximate=getattr(self.config.la2m_config, 'approximate', False),
                q=getattr(self.config.la2m_config, 'q', 1500),
                with_rotation=getattr(self.config.la2m_config, 'with_rotation', True),
                with_scaling=getattr(self.config.la2m_config, 'with_scaling', True)
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

        self.cluster_data_list = clustering_results.cluster_data_list
        self.training_time = time.time() - training_time_start
        self.formated_training_time = time.strftime("%H:%M:%S", time.gmtime(self.training_time))

        self.is_fitted = True
        self.metadata.update({
            'reference_size': len(reference_indices),
            'num_clusters': len(self.cluster_data_list),
            'successful_local_mappings': successful_clusters,
            'cluster_method': self.cluster_method,
            'cluster_sizes': [len(cluster.reference_indices) for cluster in self.cluster_data_list],
            'clustering_metadata': clustering_results.metadata,
            'training_time': self.training_time,
            'formated_training_time': self.formated_training_time
        })

        logger.info(f"LA2M mapping strategy fitted successfully. "
                   f"Local mappings: {successful_clusters}/{len(self.cluster_data_list)}")

    def _transform(self, embeddings: np.ndarray,
                 target_indices: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """Transform embeddings using the clustering-based mapping."""
        if target_indices is None:
            target_indices = np.arange(embeddings.shape[0])

        if not self.is_fitted or self.cluster_manager is None:
            raise ValueError("Mapping must be fitted before transformation")

        if self._global_pca_enabled():
            embeddings = self.src_pca.transform(embeddings)

        logger.info(f"Transforming {len(embeddings)} embeddings using LA2M strategy")

        cluster_result = self.cluster_manager.last_result
        if cluster_result is None:
            raise ValueError("Cluster result is not found, please fit the clustering first or load the clustering result from disk")
        cluster_assignments = self.cluster_manager.predict(cluster_result, embeddings, target_indices)

        target_dimension = self.metadata['target_dimension']
        if hasattr(self, 'reduced_target_embeddings'):
            transformed = np.zeros((embeddings.shape[0], self.reduced_target_embeddings.shape[1]))
        else:
            transformed = np.zeros((embeddings.shape[0], target_dimension))


        transformed_cluster_indices = []
        for cluster_id in range(len(self.cluster_data_list)):
            cluster_mask = (cluster_assignments == cluster_id)
            cluster_indices = np.where(cluster_mask)[0]

            if len(cluster_indices) > 0:
                cluster_embeddings = embeddings[cluster_indices]

                if cluster_id in self.local_mappings:
                    transformed_cluster = self._apply_local_mapping(
                        cluster_embeddings, self.local_mappings[cluster_id]
                    )
                else:
                    if self.global_fallback_mapping is not None:
                        transformed_cluster = self._apply_local_mapping(
                            cluster_embeddings, self.global_fallback_mapping
                        )
                    else:
                        transformed_cluster = cluster_embeddings

                if (cluster_id in self.local_mappings and self._complement_enabled()
                        and self.local_mappings[cluster_id].get('cluster_pca') is not None):
                    pca = self.local_mappings[cluster_id]['cluster_pca']
                    V = pca['tgt_components']
                    rest = self._apply_local_mapping(cluster_embeddings, self.global_fallback_mapping) - pca['tgt_mean']
                    transformed_cluster = transformed_cluster + rest - (rest @ V.T) @ V

                transformed[cluster_indices] = transformed_cluster
                transformed_cluster_indices.extend(cluster_indices)

        if self._global_pca_enabled():
            transformed = self.target_pca.inverse_transform(transformed)

        self._check_indices(transformed_cluster_indices, target_indices)

        return transformed

    def _check_indices(self, transformed_cluster_indices: List[int], target_indices: np.ndarray):
        """Check if the transformed cluster indices are the same as the target indices."""
        if not set(transformed_cluster_indices) == set(target_indices):
            raise ValueError("The transformed cluster indices are not the same as the target indices")

    def _global_pca_enabled(self) -> bool:
        cfg = self.config.la2m_config
        return cfg.pca_mapping and getattr(cfg, "pca_mode", "global") == "global"

    def _local_pca_enabled(self) -> bool:
        cfg = self.config.la2m_config
        return cfg.pca_mapping and getattr(cfg, "pca_mode", "global") == "local"

    def _complement_enabled(self) -> bool:
        return self._local_pca_enabled() and getattr(self.config.la2m_config, "complement", False)

    def _cluster_pca_components(self, points: np.ndarray, k: int):
        """Principal axes of one reference cluster, as (mean, components)."""
        mean = points.mean(axis=0)
        _, _, vh = np.linalg.svd(points - mean, full_matrices=False)
        return mean, vh[:k]

    def _fit_cluster_pca(self, source_ref: np.ndarray, target_ref: np.ndarray):
        """Fit a per-cluster subspace."""
        k = min(self.effective_pca_dim, len(source_ref) - 1, source_ref.shape[1])
        if k < 2:
            return None
        try:
            src_mean, src_components = self._cluster_pca_components(source_ref, k)
            tgt_mean, tgt_components = self._cluster_pca_components(target_ref, k)
        except Exception as exc:
            logger.warning(f"Per-cluster PCA failed, using the full space: {exc}")
            return None
        return {"src_mean": src_mean, "src_components": src_components,
                "tgt_mean": tgt_mean, "tgt_components": tgt_components}

    def _fit_local_mappings_batched(self, source_embeddings: np.ndarray,
                                    target_embeddings: np.ndarray, clusters) -> Optional[int]:
        """Fit every local subspace and isometry in one batched pass."""
        sizes = {len(c.reference_indices) for c in clusters}
        if len(sizes) != 1:
            return None
        m = sizes.pop()
        k = min(self.effective_pca_dim, m - 1, source_embeddings.shape[1])
        if k < 2 or m < getattr(self.config, "min_cluster_size", 5):
            return None

        idx = np.asarray([c.reference_indices for c in clusters])
        xs = torch.as_tensor(source_embeddings[idx], dtype=torch.float32)
        xt = torch.as_tensor(target_embeddings[idx], dtype=torch.float32)

        src_mean = xs.mean(dim=1, keepdim=True)
        tgt_mean = xt.mean(dim=1, keepdim=True)
        cs, ct = xs - src_mean, xt - tgt_mean

        src_basis = torch.linalg.svd(cs, full_matrices=False)[2][:, :k, :]
        tgt_basis = torch.linalg.svd(ct, full_matrices=False)[2][:, :k, :]

        zs = cs @ src_basis.transpose(1, 2)
        zt = ct @ tgt_basis.transpose(1, 2)

        with_scaling = getattr(self.config.la2m_config, "with_scaling", True)
        if with_scaling:
            zs_norm = zs.flatten(1).norm(dim=1).clamp_min(1e-12)
            zt_norm = zt.flatten(1).norm(dim=1).clamp_min(1e-12)
            zs = zs / zs_norm[:, None, None]
            zt = zt / zt_norm[:, None, None]
        else:
            zs_norm = torch.ones(len(idx))
            zt_norm = torch.ones(len(idx))

        h = zt.transpose(1, 2) @ zs
        u, _, vt = torch.linalg.svd(h, full_matrices=False)
        if getattr(self.config.la2m_config, "with_rotation", True):
            rot = u @ vt
            if getattr(self.config.la2m_config, "transpose_local", False):
                rot = rot.transpose(1, 2)
        else:
            rot = torch.eye(k).expand(len(idx), k, k)

        zero = np.zeros(k, dtype=np.float32)
        for i in range(len(idx)):
            self.local_mappings[i] = {
                "rotation_matrix": rot[i].numpy(),
                "source_mean": zero, "target_mean": zero,
                "with_scaling": with_scaling,
                "source_norm": float(zs_norm[i]), "target_norm": float(zt_norm[i]),
                "cluster_pca": {
                    "src_mean": src_mean[i, 0].numpy(), "src_components": src_basis[i].numpy(),
                    "tgt_mean": tgt_mean[i, 0].numpy(), "tgt_components": tgt_basis[i].numpy(),
                },
            }
        return len(idx)

    def _apply_local_mapping(self, embeddings: np.ndarray,
                           mapping_params: Dict[str, np.ndarray]) -> np.ndarray:
        """Apply local Procrustes mapping to embeddings."""
        if mapping_params is None or not isinstance(mapping_params, dict):
            return embeddings

        cluster_pca = mapping_params.get('cluster_pca') if isinstance(mapping_params, dict) else None
        if cluster_pca is not None:
            embeddings = (embeddings - cluster_pca['src_mean']) @ cluster_pca['src_components'].T

        try:
            rotation_matrix = mapping_params['rotation_matrix']
            source_mean = mapping_params['source_mean']
            target_mean = mapping_params['target_mean']
            with_scaling = mapping_params.get('with_scaling', False)

            embeddings_centered = embeddings - source_mean

            if with_scaling and 'source_norm' in mapping_params and 'target_norm' in mapping_params:
                source_norm = mapping_params['source_norm']
                target_norm = mapping_params['target_norm']
                embeddings_centered = embeddings_centered / source_norm
                transformed = np.dot(embeddings_centered, rotation_matrix.T) * target_norm + target_mean
            else:
                transformed = np.dot(embeddings_centered, rotation_matrix.T) + target_mean

            if cluster_pca is not None:
                transformed = transformed @ cluster_pca['tgt_components'] + cluster_pca['tgt_mean']
            return transformed
        except KeyError as e:
            logger.error(f"Missing key in mapping parameters: {e}")
            return embeddings
        except Exception as e:
            logger.error(f"Error applying local mapping: {e}")
            return embeddings

    def get_cluster_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics about the clustering and mappings."""
        if not self.is_fitted:
            return {}

        stats = {
            'num_clusters': len(self.cluster_data_list),
            'successful_local_mappings': len(self.local_mappings),
            'cluster_sizes': [len(cluster.reference_indices) for cluster in self.cluster_data_list],
            'has_global_fallback': self.global_fallback_mapping is not None,
            'clustering_method': self.cluster_method
        }


        return stats

    def save(self, path) -> None:
        """Save the fitted LA2M mapping strategy."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted mapping")

        super().save(path)

        save_path = Path(path)
        self.cluster_manager.save(save_path=save_path/ "cluster_manager", save_embeddings=False)

        np.save(save_path / "reduced_source_embeddings.npy", self.reduced_source_embeddings)
        np.save(save_path / "reduced_target_embeddings.npy", self.reduced_target_embeddings)

        cluster_data_dict = {}
        for i, cluster_data in enumerate(self.cluster_data_list):
            cluster_data_dict[i] = {
                'reference_indices': cluster_data.reference_indices,
                'linked_target_indices': cluster_data.linked_target_indices,
                'center_embedding': cluster_data.center_embedding.tolist() if cluster_data.center_embedding is not None else None
            }

        import json
        with open(save_path / "cluster_data.json", "w") as f:
            json.dump(cluster_data_dict, f, indent=2)

        try:
            import joblib
            local_mappings_data = {
                'mappings': self.local_mappings,
                'num_clusters': len(self.local_mappings)
            }
            joblib.dump(local_mappings_data, save_path / "local_mappings.joblib")
            logger.info(f"Saved {len(self.local_mappings)} local mappings using joblib")
        except ImportError:
            logger.warning("joblib not available, falling back to numpy for local mappings")
            for cluster_id, mapping in self.local_mappings.items():
                cluster_path = save_path / f"cluster_{cluster_id}"
                cluster_path.mkdir(exist_ok=True)

                for key, value in mapping.items():
                    np.save(cluster_path / f"{key}.npy", value)

        if self.global_fallback_mapping is not None:
            try:
                import joblib
                joblib.dump(self.global_fallback_mapping, save_path / "global_fallback_mapping.joblib")
                logger.info("Saved global fallback mapping using joblib")
            except ImportError:
                global_path = save_path / "global_fallback"
                global_path.mkdir(exist_ok=True)

                for key, value in self.global_fallback_mapping.items():
                    np.save(global_path / f"{key}.npy", value)

        if self.config.la2m_config.pca_mapping:
            self.save_pca_instance(self.src_pca, save_path, "src_pca")
            self.save_pca_instance(self.target_pca, save_path, "target_pca")

        logger.info(f"Saved LA2M mapping strategy to {save_path}")

    def save_pca_instance(self, pca_instance: PCA, save_path: Path, prefix: str):
        """Save PCA instance."""
        (save_path / prefix).mkdir(exist_ok=True)
        np.save(save_path / f"{prefix}/components.npy", pca_instance.components_)
        np.save(save_path / f"{prefix}/mean.npy", pca_instance.mean_)
        np.save(save_path / f"{prefix}/explained_variance.npy", pca_instance.explained_variance_)
        np.save(save_path / f"{prefix}/explained_variance_ratio.npy", pca_instance.explained_variance_ratio_)

    def load_pca_instance(self, load_path: Path, prefix: str):
        """Load PCA instance."""
        pca_instance = PCA(n_components=self.config.la2m_config.pca_dim)
        pca_instance.components_ = np.load(load_path / f"{prefix}/components.npy")
        pca_instance.mean_ = np.load(load_path / f"{prefix}/mean.npy")
        pca_instance.explained_variance_ = np.load(load_path / f"{prefix}/explained_variance.npy")
        pca_instance.explained_variance_ratio_ = np.load(load_path / f"{prefix}/explained_variance_ratio.npy")
        return pca_instance

    @classmethod
    def load(cls, path, clustering_manager: Optional[ClusterManager] = None) -> 'LA2MStrategy':
        """Load a fitted LA2M mapping strategy."""
        load_path = Path(path)

        import json
        with open(load_path / "mapping_info.json", "r") as f:
            mapping_info = json.load(f)

        config = MappingConfig.from_dict(mapping_info['config'])
        if clustering_manager is None:
            clustering_manager = ClusterManager.from_saved(load_path / "cluster_manager")
        instance = cls(config, clustering_manager)

        instance.reduced_source_embeddings = np.load(load_path / "reduced_source_embeddings.npy")
        instance.reduced_target_embeddings = np.load(load_path / "reduced_target_embeddings.npy")

        with open(load_path / "cluster_data.json", "r") as f:
            cluster_data_dict = json.load(f)

        instance.cluster_data_list = []
        for i in range(len(cluster_data_dict)):
            cluster_info = cluster_data_dict[str(i)]
            cluster_data = ClusterData(
                reference_indices=cluster_info['reference_indices'],
                linked_target_indices=cluster_info['linked_target_indices'],
                center_embedding=np.array(cluster_info['center_embedding']) if cluster_info['center_embedding'] else None
            )
            instance.cluster_data_list.append(cluster_data)

        try:
            import joblib
            local_mappings_path = load_path / "local_mappings.joblib"
            if local_mappings_path.exists():
                local_mappings_data = joblib.load(local_mappings_path)
                instance.local_mappings = local_mappings_data['mappings']
                logger.info(f"Loaded {len(instance.local_mappings)} local mappings using joblib")
            else:
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
        except ImportError:
            logger.warning("joblib not available, falling back to numpy for local mappings")
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

        try:
            import joblib
            global_mapping_path = load_path / "global_fallback_mapping.joblib"
            if global_mapping_path.exists():
                instance.global_fallback_mapping = joblib.load(global_mapping_path)
                logger.info("Loaded global fallback mapping using joblib")
            else:
                global_path = load_path / "global_fallback"
                if global_path.exists():
                    global_mapping = {}
                    for npy_file in global_path.glob("*.npy"):
                        key = npy_file.stem
                        global_mapping[key] = np.load(npy_file)

                    if global_mapping:
                        instance.global_fallback_mapping = global_mapping
        except ImportError:
            global_path = load_path / "global_fallback"
            if global_path.exists():
                global_mapping = {}
                for npy_file in global_path.glob("*.npy"):
                    key = npy_file.stem
                    global_mapping[key] = np.load(npy_file)

                if global_mapping:
                    instance.global_fallback_mapping = global_mapping

        if config.la2m_config.pca_mapping:
            instance.src_pca = instance.load_pca_instance(load_path, "src_pca")
            instance.target_pca = instance.load_pca_instance(load_path, "target_pca")

        instance.is_fitted = True
        instance.metadata = mapping_info.get('metadata', {})

        logger.info(f"Loaded LA2M mapping strategy from {load_path}")
        return instance