"""Mapping pipeline entrypoint."""

from typing import Any, Dict

from vectormerge.embeddings import get_embedding
from vectormerge.mapping import VectorSpaceMapper
from vectormerge.reference import get_reference


def mapping_pipeline(
    config: Any,
    dataset: str,
    source_model: str,
    target_model: str,
    mapping_method: str,
    reference_key: str,
    force: bool = False,
) -> Dict[str, Any]:
    """Fit mapping model through a pipeline boundary."""

    mapper = VectorSpaceMapper(
        strategy_name=mapping_method,
        config=config.mapping_config,
        dataset_name=dataset,
        source_model=source_model,
        target_model=target_model,
        reference_key=reference_key,
        reference_path=config.reference_path,
        cluster_path=config.cluster_path,
        embedding_path=config.embedding_path,
        mapping_param_path=config.mapping_param_path,
        mapping_embedding_path=config.mapping_embedding_path,
        force=force,
        save_param=config.mapping_config.save_param,
        save_embedding=config.mapping_config.save_embedding,
        clustering_config=config.clustering_config,
    )
    source_embeddings, target_embeddings = get_embedding(
        dataset,
        source_model,
        config.embedding_path,
        target_model,
        type_="corpus",
        align=True,
    )
    reference_indices = get_reference(config.reference_path, reference_key)
    mapper.fit(source_embeddings, target_embeddings, reference_indices["d0_index"])
    return {
        "pipeline": "mapping",
        "dataset": dataset,
        "source_model": source_model,
        "target_model": target_model,
        "mapping_method": mapping_method,
    }
