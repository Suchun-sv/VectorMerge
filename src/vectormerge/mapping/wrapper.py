from .strategies.la2m import LA2MStrategy
from .base import MappingConfig, LA2MConfig
from vectormerge.clustering import ClusterManager
from vectormerge.clustering.base import ClusteringConfig, LA2MClusteringConfig, KMeansConfig

class LA2MMapper(LA2MStrategy):
    def __init__(self, 
            cluster_method: str="la2m-cluster", d_prime: int=10, pca_mapping: bool=True, pca_dim: int=14, use_norm: bool=False, device: str="auto", batch_size: int=32, verbose: bool=False, save_param: bool=True, save_embedding: bool=False, dataset_name: str="", model: str="", reference_key: str="", reference_path: str="./data/processed/references", cluster_path:str ="./data/processed/clusters", embedding_path: str="./data/processed/embeddings", force: bool=True, auto_save_results: bool=False, n_clusters: int=10, max_iter: int=300, tol: float=1e-4, random_state: int=42):
        """
        Args:
            cluster_method: str,
            d_prime: int,
            pca_mapping: bool,
            pca_dim: int,
            use_norm: bool,
            device: str,
            batch_size: int,
            verbose: bool,
            save_param: bool,
            save_embedding: bool,
            dataset_name: str,
            model: str,
            reference_key: str,
            reference_path: str,
            cluster_path: str,
            embedding_path: str,
            force: bool,
            auto_save_results: bool,
            n_clusters: int=10,
            max_iter: int=300,
            tol: float=1e-4,
            random_state: int=42,
        """

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


