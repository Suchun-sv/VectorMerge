from .strategies.la2m import LA2MStrategy
from .base import MappingConfig, LA2MConfig, ProcrustesConfig
from vectormerge.mapping.strategies import ProcrustesMappingStrategy
from vectormerge.clustering import ClusterManager
from vectormerge.clustering.base import ClusteringConfig, LA2MClusteringConfig, KMeansConfig

class LA2MMapper(LA2MStrategy):
    def __init__(self,
            cluster_method: str="la2m-cluster", d_prime: int=10, pca_mapping: bool=True, pca_dim: int=14, use_norm: bool=False, device: str="auto", batch_size: int=32, verbose: bool=False, save_param: bool=True, save_embedding: bool=False, dataset_name: str="", model: str="", reference_key: str="", reference_path: str="./data/processed/references", cluster_path:str ="./data/processed/clusters", embedding_path: str="./data/processed/embeddings", force: bool=True, auto_save_results: bool=False, n_clusters: int=10, max_iter: int=300, tol: float=1e-4, random_state: int=42):
        """Args."""

        config = MappingConfig(
            device=device,
            batch_size=batch_size,
            verbose=verbose,
            save_param=save_param,
            save_embedding=save_embedding,
            la2m_config=LA2MConfig(
                cluster_method=cluster_method,
                d_prime=d_prime,
                pca_mapping=pca_mapping,
                pca_dim=pca_dim,
                use_norm=use_norm,
            )
        )


        clustering_manager = ClusterManager(
            dataset_name=dataset_name,
            model=model,
            reference_key=reference_key,
            reference_path=reference_path,
            cluster_path=cluster_path,
            embedding_path=embedding_path,
            strategy_name=cluster_method,
            strategy_config=ClusteringConfig(
                clustering_method=cluster_method,
                la2m_config=LA2MClusteringConfig(d_prime=d_prime),
                kmeans_config=KMeansConfig(n_clusters=n_clusters, max_iter=max_iter, tol=tol, random_state=random_state),
            ),
            force=force,
            auto_save_results=auto_save_results,
            verbose=verbose,
        )
        super().__init__(config=config, clustering_manager=clustering_manager)


class ProcrustesMapper(ProcrustesMappingStrategy):
    def __init__(self,
            approximate: bool = False,
            q: int = 1500,
            with_rotation: bool = True,
            with_scaling: bool = True,
            use_pca: bool = False,
            reduced_dim: int = 0,
            procrustes_pca_type: str = "none",
            use_norm: bool = True,
            device: str = "auto",
            batch_size: int = 32,
            verbose: bool = False,
            save_param: bool = False,
            save_embedding: bool = False):
        """Procrustes mapping with configurable parameters."""

        config = MappingConfig(
            device=device,
            batch_size=batch_size,
            verbose=verbose,
            save_param=save_param,
            save_embedding=save_embedding,
            procrustes_config=ProcrustesConfig(
                approximate=approximate,
                q=q,
                with_rotation=with_rotation,
                with_scaling=with_scaling,
                use_pca=use_pca,
                reduced_dim=reduced_dim,
                procrustes_pca_type=procrustes_pca_type,
                use_norm=use_norm,
            )
        )

        super().__init__(config=config)