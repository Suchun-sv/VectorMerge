"""Evaluation pipeline entrypoint."""

from typing import Any, Dict, Optional

from vectormerge.dataset import load_dataset
from vectormerge.embedding_dataset import EmbeddingDataset
from vectormerge.embeddings import get_embedding
from vectormerge.evaluation import Evaluator
from vectormerge.mapping import VectorSpaceMapper
from vectormerge.reference import get_reference


def evaluation_pipeline(
    config: Any,
    dataset_name: str,
    source_model: str,
    target_model: str,
    mapping_method: str,
    reference_key: Optional[str] = None,
    reference_method: str = "la2m",
    force: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run evaluation through a pipeline boundary."""

    dataset = load_dataset(
        dataset_name=dataset_name,
        data_path=config.data_path,
        force=force,
        interactive=False,
        verbose=verbose,
    )

    vectormerge = VectorSpaceMapper(
        strategy_name=mapping_method,
        config=config.mapping_config,
        dataset_name=dataset_name,
        source_model=source_model,
        target_model=target_model,
        reference_key=reference_key,
        reference_path=config.reference_path,
        cluster_path=config.cluster_path,
        embedding_path=config.embedding_path,
        mapping_param_path=config.mapping_param_path,
        mapping_embedding_path=config.mapping_embedding_path,
        force=force,
        save_param=True,
        save_embedding=True,
        clustering_config=config.clustering_config,
    )

    corpus_emb_1, corpus_emb_2 = get_embedding(
        dataset_name=dataset_name,
        model_name=source_model,
        embedding_path=config.embedding_path,
        target_model_name=target_model,
        type_="corpus",
        align=True,
    )
    query_emb_1, query_emb_2 = get_embedding(
        dataset_name=dataset_name,
        model_name=source_model,
        embedding_path=config.embedding_path,
        target_model_name=target_model,
        type_="query",
        align=True,
    )

    if reference_key is not None:
        reference_indices = get_reference(
            reference_key=reference_key, reference_path=config.reference_path
        )
        d0_index, d1_index, d2_index = (
            reference_indices["d0_index"],
            reference_indices["d1_index"],
            reference_indices["d2_index"],
        )
    else:
        embedding_dataset = EmbeddingDataset(
            text_dataset_name=dataset_name,
            source_embedding_model_name=source_model,
            target_embedding_model_name=target_model,
            align_dimension=True,
            dataset_path=config.data_path,
            embedding_path=config.embedding_path,
            reference_method=reference_method,
            reference_config={},
        )
        d0_index, d1_index, d2_index = (
            embedding_dataset.d0_index,
            embedding_dataset.d1_index,
            embedding_dataset.d2_index,
        )

    transformed_embeddings = vectormerge.fit_and_transform(
        corpus_emb_1, corpus_emb_2, d0_index
    )

    evaluator = Evaluator(
        corpus_emb_1=corpus_emb_1,
        corpus_emb_2=corpus_emb_2,
        query_emb_1=query_emb_1,
        query_emb_2=query_emb_2,
        query_index2answer_index=dataset.query_index2answer_index,
        d0=d0_index,
        d1=d1_index,
        d2=d2_index,
        corpus_emb_1_transformed=transformed_embeddings,
        k_list=[10, 50, 100, 500, 1000],
    )
    results = evaluator.evaluate()
    return {
        "pipeline": "evaluation",
        "dataset": dataset_name,
        "source_model": source_model,
        "target_model": target_model,
        "results": results,
    }
