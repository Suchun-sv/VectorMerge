"""
Embedding mapping commands for VectorMerge CLI.

This module contains commands for creating and managing embedding mappings.
"""

import typer
from click import Context
from dataclasses import replace

from vectormerge.embeddings import SUPPORTED_MODELS, get_embedding
from vectormerge.dataset import SUPPORTED_DATASETS
from vectormerge.reference import get_reference
from vectormerge.mapping import VectorSpaceMapper, SUPPORTED_MAPPING_METHODS

from .base import cli_defaults
from .utils import (
    display_error_and_exit, handle_extra_args
)
from vectormerge.config import VectorMergeConfig

# Initialize mapping command group
map_app = typer.Typer(help="Create and manage embedding mappings")

# Load configuration
# config_loader = VectorMergeConfig()


@map_app.command("map", help="Map embeddings", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def map_command(
    ctx: Context,
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset to use for mapping, support: " + ", ".join(SUPPORTED_DATASETS)),
    source_model: str = typer.Option(None, "--source-model", "-s", help="Source model to use for mapping, support: " + ", ".join(SUPPORTED_MODELS)),
    target_model: str = typer.Option(None, "--target-model", "-t", help="Target model to use for mapping, support: " + ", ".join(SUPPORTED_MODELS)),
    src_tar_model: str = typer.Option(None, "--src-tar-model", "-stm", help="Combination of source and target model, use `_` to separate the source and target model, e.g. `mistral_openai` support: " + ", ".join(SUPPORTED_MODELS)),
    mapping_method: str = typer.Option(..., "--mapping-method", "-mm", help="Mapping method, support: " + ", ".join(SUPPORTED_MAPPING_METHODS)),
    reference_key: str = typer.Option(..., "--reference-key", "-rk", help="Reference key to use for mapping"),
    reference_path: str = typer.Option(cli_defaults.reference_path, "--reference-path", help="Path to the reference file, default: " + cli_defaults.reference_path),
    cluster_path: str = typer.Option(cli_defaults.cluster_path, "--cluster-path", help="Path to the cluster file, default: " + cli_defaults.cluster_path),
    embedding_path: str = typer.Option(cli_defaults.embedding_path, "--embedding-path", help="Path to the embedding file, default: " + cli_defaults.embedding_path),
    mapping_param_path: str = typer.Option(cli_defaults.mapping_param_path, "--mapping-param-path", help="Path to the mapping parameter file, default: " + cli_defaults.mapping_param_path),
    mapping_embedding_path: str = typer.Option(cli_defaults.mapping_embedding_path, "--mapping-embedding-path", help="Path to the mapping embedding file, default: " + cli_defaults.mapping_embedding_path),
    force: bool = typer.Option(False, "--force", "-f", help="Force the mapping to be re-run"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive mode"),
    verbose: bool = typer.Option(cli_defaults.verbose, "--verbose", "-v", help="Verbose output"),
    wandb_entity: str = typer.Option(None, "--wandb-entity", "-we", help="Weights and Biases"),
    wandb_project: str = typer.Option("vector-merge", "--wandb-project", "-wp", help="Weights and Biases project"),
):
    if (source_model is not None and target_model is not None) and (src_tar_model is not None):
        display_error_and_exit(f"Source model {source_model} and target model {target_model} cannot be specified at the same time, please use --src-tar-model instead")

    if src_tar_model is not None:
        source_model = src_tar_model.split("_")[0]
        target_model = src_tar_model.split("_")[1]

    if source_model not in SUPPORTED_MODELS:
        display_error_and_exit(f"Source model {source_model} not supported, supported models: {', '.join(SUPPORTED_MODELS)}")

    if target_model not in SUPPORTED_MODELS:
        display_error_and_exit(f"Target model {target_model} not supported, supported models: {', '.join(SUPPORTED_MODELS)}")

    if source_model is None or target_model is None:
        display_error_and_exit(f"Source model {source_model} and target model {target_model} not supported, supported models: {', '.join(SUPPORTED_MODELS)}")

    if reference_key is None:
        display_error_and_exit(f"Reference key {reference_key} not supported, supported models: {', '.join(SUPPORTED_MODELS)}")

    if dataset not in SUPPORTED_DATASETS:
        display_error_and_exit(f"Dataset {dataset} not supported, supported datasets: {', '.join(SUPPORTED_DATASETS)}")

    config = handle_extra_args(ctx)

    config = replace(config,
        verbose=verbose,
        reference_path=reference_path,
        cluster_path=cluster_path,
        embedding_path=embedding_path,
        mapping_param_path=mapping_param_path,
        mapping_embedding_path=mapping_embedding_path,

    )

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

    source_embeddings, target_embeddings = get_embedding(dataset, source_model, config.embedding_path, target_model, type_="corpus", align=True)
    reference_indices = get_reference(config.reference_path, reference_key)

    mapper.fit(source_embeddings, target_embeddings, reference_indices['d0_index'])


