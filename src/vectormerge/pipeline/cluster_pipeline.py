"""Cluster pipeline entrypoint."""

from typing import Any, Dict

from vectormerge.clustering import ClusterManager


def cluster_pipeline(
    config: Any,
    dataset: str,
    model: str,
    reference_key: str,
    cluster_method: str,
    force: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Create cluster results through a pipeline boundary."""

    manager = ClusterManager(
        dataset_name=dataset,
        model=model,
        reference_key=reference_key,
        reference_path=config.reference_path,
        cluster_path=config.cluster_path,
        embedding_path=config.embedding_path,
        strategy_name=cluster_method,
        strategy_config=config.clustering_config,
        force=force,
        verbose=verbose,
        auto_save_results=True,
    )
    result = manager.fit()
    return {
        "pipeline": "cluster",
        "cluster_path": config.cluster_path,
        "method": cluster_method,
        "result_type": type(result).__name__,
    }
