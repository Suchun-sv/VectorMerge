import typer
from typing import Any
from click import Context

from vectormerge.cli import config_loader
from vectormerge.cli.config_loader import ConfigLoader
from .base import SUPPORTED_DATASETS, SUPPORTED_MODELS, SUPPORTED_MAPPING_METHODS
from . import parse_dynamic_config
from ..mapping.manager import VectorSpaceMapper
from ..dataset import load_dataset
from ..embeddings import get_embedding
from ..reference import get_reference

evaluate_app = typer.Typer(
    name="evaluate",
    help="Evaluate the performance of a mapping model, you can pass extra arguments to the command to override the default values. \n E.g. --mapping_config.la2m.num_clusters=50"
)

@evaluate_app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def single_run(
    ctx: Context,
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset name, support: " + ", ".join(SUPPORTED_DATASETS)),
    source_model: str = typer.Option(..., "--source-model", "-s", help="Source model name, support: " + ", ".join(SUPPORTED_MODELS)),
    target_model: str = typer.Option(..., "--target-model", "-t", help="Target model name, support: " + ", ".join(SUPPORTED_MODELS)),
    reference_key: str = typer.Option(..., "--reference-key", "-rk", help="Reference key"),
    mapping_method: str = typer.Option(..., "--mapping-method", "-mm", help="Mapping method, support: " + ", ".join(SUPPORTED_MAPPING_METHODS)),
    force: bool = typer.Option(False, "--force", "-f", help="Force evaluation"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive mode"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Evaluate the performance of a mapping model, you can pass extra arguments to the command to override the default values.
    E.g. --mapping_config.la2m.num_clusters=50
    """
    extra_dict = parse_dynamic_config(ctx)
    config_loader = ConfigLoader()
    if extra_dict:
        config_loader.update_config(extra_dict)
    
    config = config_loader.config

    vectormerge = VectorSpaceMapper(strategy=mapping_method, 
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
    save_param=True,
    save_embedding=True,
    clustering_config=config.clustering_config)

    source_embeddings = get_embedding(model_name=source_model, dataset_name=dataset, embedding_path=config.embedding_path)
    target_embeddings = get_embedding(model_name=target_model, dataset_name=dataset, embedding_path=config.embedding_path)

    reference_indices = get_reference(reference_key=reference_key, reference_path=config.reference_path)

    transformed_embeddings = vectormerge.fit_and_transform(source_embeddings, target_embeddings, reference_indices['d0_index'])
    
    print("transformed_embeddings.shape", transformed_embeddings.shape)



