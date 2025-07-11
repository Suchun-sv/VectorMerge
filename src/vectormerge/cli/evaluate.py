import typer
from typing import Any
from click import Context
import wandb

from vectormerge.cli.base import SUPPORTED_DATASETS, SUPPORTED_MODELS, SUPPORTED_MAPPING_METHODS, set_wandb
from vectormerge.cli.utils import handle_extra_args

from vectormerge.mapping import VectorSpaceMapper
from vectormerge.dataset import load_dataset
from vectormerge.embeddings import get_embedding
from vectormerge.reference import get_reference
from vectormerge.evaluation import Evaluator

evaluate_app = typer.Typer(
    name="evaluate",
    help="Evaluate the performance of a mapping model, you can pass extra arguments to the command to override the default values. \n E.g. --mapping_config.la2m.num_clusters=50"
)

@evaluate_app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def single_run(
    ctx: Context,
    dataset_name: str = typer.Option(..., "--dataset", "-d", help="Dataset name, support: " + ", ".join(SUPPORTED_DATASETS)),
    source_model: str = typer.Option(None, "--source-model", "-s", help="Source model name, support: " + ", ".join(SUPPORTED_MODELS)),
    target_model: str = typer.Option(None, "--target-model", "-t", help="Target model name, support: " + ", ".join(SUPPORTED_MODELS)),
    src_tar_model: str = typer.Option(None, "--src-tar-model", "-stm", help="Combination of source and target model, use `_` to separate the source and target model, e.g. `mistral_openai` support: " + ", ".join(SUPPORTED_MODELS)),
    reference_key: str = typer.Option(..., "--reference-key", "-rk", help="Reference key"),
    mapping_method: str = typer.Option(..., "--mapping-method", "-mm", help="Mapping method, support: " + ", ".join(SUPPORTED_MAPPING_METHODS)),
    force: bool = typer.Option(False, "--force", "-f", help="Force evaluation"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive mode"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    wandb_entity: str = typer.Option(None, "--wandb-entity", "-we", help="Weights and Biases"),
    wandb_project: str = typer.Option("vector-merge", "--wandb-project", "-wp", help="Weights and Biases project"),
):
    """
    Evaluate the performance of a mapping model, you can pass extra arguments to the command to override the default values.
    E.g. --mapping_config.la2m.num_clusters=50
    """
    if src_tar_model is not None:
        source_model = src_tar_model.split("_")[0]
        target_model = src_tar_model.split("_")[1]
    
    if source_model not in SUPPORTED_MODELS:
        raise ValueError(f"Source model is not supported, support: " + ", ".join(SUPPORTED_MODELS))
    if target_model not in SUPPORTED_MODELS:
        raise ValueError(f"Target model is not supported, support: " + ", ".join(SUPPORTED_MODELS))
    
    if source_model is None or target_model is None:
        raise ValueError("Source and target model are required")

    config = handle_extra_args(ctx)

    set_wandb(wandb_entity=wandb_entity, wandb_project=wandb_project, config_dict=config.to_dict())

    dataset = load_dataset(dataset_name=dataset_name, data_path=config.data_path, force=force, interactive=False, verbose=verbose)

    vectormerge = VectorSpaceMapper(strategy=mapping_method, 
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
    clustering_config=config.clustering_config)

    corpus_emb_1, corpus_emb_2 = get_embedding(dataset_name=dataset_name, model_name=source_model, embedding_path=config.embedding_path, target_model_name=target_model, type_="corpus", align=True)
    query_emb_1, query_emb_2 = get_embedding(dataset_name=dataset_name, model_name=source_model, embedding_path=config.embedding_path, target_model_name=target_model, type_="query", align=True)

    assert corpus_emb_2 is not None
    assert query_emb_2 is not None

    reference_indices = get_reference(reference_key=reference_key, reference_path=config.reference_path)

    transformed_embeddings = vectormerge.fit_and_transform(corpus_emb_1, corpus_emb_2, reference_indices['d0_index'])

    evaluator = Evaluator(
        corpus_emb_1=corpus_emb_1,
        corpus_emb_2=corpus_emb_2,
        query_emb_1=query_emb_1,
        query_emb_2=query_emb_2,
        query_index2answer_index=dataset.query_index2answer_index,
        d0=reference_indices['d0_index'],
        d1=reference_indices['d1_index'],
        d2=reference_indices['d2_index'],
        corpus_emb_1_transformed=transformed_embeddings,
        k_list=[10, 50, 100, 500, 1000]
    )

    print("transformed_embeddings.shape", transformed_embeddings.shape)

    recalls = evaluator.evaluate()
    wandb.log(recalls)
    print(recalls)
    



