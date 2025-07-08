import typer
from typing import Any
from click import Context
import wandb

from vectormerge.cli import config_loader
from vectormerge.cli.config_loader import ConfigLoader
from .base import SUPPORTED_DATASETS, SUPPORTED_MODELS, SUPPORTED_MAPPING_METHODS, set_wandb
from . import parse_dynamic_config
from ..mapping.manager import VectorSpaceMapper
from ..dataset import load_dataset
from ..embeddings import get_embedding
from ..reference import get_reference
from ..evaluation import Evaluator, get_retrieval_list

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
    source_model: str = typer.Option(..., "--source-model", "-s", help="Source model name, support: " + ", ".join(SUPPORTED_MODELS)),
    target_model: str = typer.Option(..., "--target-model", "-t", help="Target model name, support: " + ", ".join(SUPPORTED_MODELS)),
    reference_key: str = typer.Option(..., "--reference-key", "-rk", help="Reference key"),
    mapping_method: str = typer.Option(..., "--mapping-method", "-mm", help="Mapping method, support: " + ", ".join(SUPPORTED_MAPPING_METHODS)),
    force: bool = typer.Option(False, "--force", "-f", help="Force evaluation"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive mode"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    wandb_entity: str = typer.Option(None, "--wandb-entity", "-we", help="Weights and Biases"),
):
    """
    Evaluate the performance of a mapping model, you can pass extra arguments to the command to override the default values.
    E.g. --mapping_config.la2m.num_clusters=50
    """
    extra_dict = parse_dynamic_config(ctx)
    config_loader = ConfigLoader().load_config()
    if extra_dict:
        config_loader.update_config(extra_dict)
    
    config = config_loader.config

    set_wandb(wandb_entity=wandb_entity, wandb_project="vector-merge", config_dict=config.to_dict())

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
    



