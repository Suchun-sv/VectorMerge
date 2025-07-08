"""
Embedding mapping commands for VectorMerge CLI.

This module contains commands for creating and managing embedding mappings.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import typer
import numpy as np
from rich import print as rprint
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.table import Table
import json
from ..embeddings import SUPPORTED_MODELS, get_embedding
from ..dataset import SUPPORTED_DATASETS
from ..reference import get_reference
from ..mapping import VectorSpaceMapper
from ..mapping.base import MappingConfig

from .base import cli_defaults, set_seed, console
from .utils import (
    select_model_interactively, select_dataset_interactively,
    validate_model_and_dataset, display_error_and_exit
)
from .config_loader import VectorMergeConfig

# Initialize mapping command group
mapping_app = typer.Typer(help="Create and manage embedding mappings")

# Load configuration
config_loader = VectorMergeConfig()

# Get mapper defaults
nonlinear_defaults = config_loader.mapping_config.nonlinear_config
procrustes_defaults = config_loader.mapping_config.procrustes_config
la2m_defaults = config_loader.mapping_config.la2m_config


@mapping_app.command("procrustes", help="Create mapping using Procrustes analysis")
def procrustes_mapping(
    source_model: Optional[str] = typer.Option(None, "--source", "-s", help="Source model, supported models: " + ", ".join(SUPPORTED_MODELS)),
    target_model: Optional[str] = typer.Option(None, "--target", "-t", help="Target model, supported models: " + ", ".join(SUPPORTED_MODELS)),
    source_and_target_model: Optional[str] = typer.Option(None, "--source-and-target", "-st", help="Source and target model, use `_` to separate"),
    dataset: Optional[str] = typer.Option(None, "--dataset", "-d", help="Dataset name, supported datasets:" + ", ".join(SUPPORTED_DATASETS)),
    reference_key: Optional[str] = typer.Option(None, "--reference-key", "-rk", help="Reference key, use `vectormerge create-reference --check` to inspect"),
    embedding_path: Path = typer.Option(cli_defaults['embedding_path'], "--embedding-path", help="Path to embeddings"),
    reference_path: Path = typer.Option(cli_defaults['reference_path'], "--reference-path", help="Path to reference files"),
    mapping_param_path: Path = typer.Option(cli_defaults['mapping_param_path'], "--param-save-path", help="Path to save mapping parameters"),
    mapping_embedding_path: Path = typer.Option(cli_defaults['mapping_embedding_path'], "--embedding-save-path", help="Path to save mapping embeddings"),
    use_gpu: bool = typer.Option(False, "--gpu", help="Use GPU acceleration"),
    with_rotation: bool = typer.Option(True, "--with-rotation", help="Use rotation"),
    approximate: bool = typer.Option(False, "--approximate", help="Use approximate SVD"),
    q: int = typer.Option(1500, "--q", help="Approximation parameter"),
    save_param: bool = typer.Option(procrustes_defaults.save_param, "--save-param", help="Save mapping parameters"),
    save_embedding: bool = typer.Option(procrustes_defaults.save_embedding, "--save-embedding", help="Save mapping embeddings"),
    force: bool = typer.Option(False, "--force", help="Force regeneration"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive mode"),
    verbose: bool = typer.Option(cli_defaults['verbose'], "--verbose", "-v", help="Verbose output"),
):
    """Create mapping using Procrustes analysis (orthogonal transformation)."""
    
    # Set random seed
    set_seed()
    
    # Interactive mode
    if interactive:
        if not source_model:
            rprint("[cyan]Select source model:[/cyan]")
            source_model = select_model_interactively()
        if not target_model:
            rprint("[cyan]Select target model:[/cyan]")
            target_model = select_model_interactively()
        if not dataset:
            dataset = select_dataset_interactively()
    
    # Validate inputs
    if not source_model or not target_model or not dataset:
        display_error_and_exit("Please specify source model, target model, and dataset (or use --interactive)")
    
    # Type assertions after validation
    assert source_model is not None and target_model is not None and dataset is not None
    
    # Validate models and dataset
    for model in [source_model, target_model]:
        is_valid, error_msg = validate_model_and_dataset(model, dataset)
        if not is_valid:
            display_error_and_exit(error_msg)
    
    # Create configuration
    from ..mapping.base import MappingConfig, ProcrustesConfig
    config = MappingConfig(
        procrustes_config=ProcrustesConfig(
            with_rotation=with_rotation,
            approximate=approximate,
            q=q,
            use_pca=use_pca,
            reduced_dim=reduced_dim,
            procrustes_pca_type=procrustes_pca_type,
            use_norm=use_norm,
            save_param=save_param,
            save_embedding=save_embedding,
        ),
        verbose=verbose
    )

    # Run mapping workflow
    _common_mapping_workflow(
        strategy="procrustes",
        source_model=source_model,
        target_model=target_model,
        dataset=dataset,
        reference_key=str(reference_key),
        embedding_path=str(embedding_path),
        reference_path=str(reference_path),
        mapping_param_path=str(mapping_param_path),
        mapping_embedding_path=str(mapping_embedding_path),
        strategy_config=config,
        force=force,
        save_param=save_param,
        save_embedding=save_embedding,
        verbose=verbose,
    )


@mapping_app.command("nonlinear", help="Create mapping using nonlinear neural network")
def nonlinear_mapping(
    source_model: Optional[str] = typer.Option(None, "--source", "-s", help="Source model, supported models: " + ", ".join(SUPPORTED_MODELS)),
    target_model: Optional[str] = typer.Option(None, "--target", "-t", help="Target model, supported models: " + ", ".join(SUPPORTED_MODELS)),
    source_and_target_model: Optional[str] = typer.Option(None, "--source-and-target", "-st", help="Source and target model, use `_` to separate"),
    dataset: Optional[str] = typer.Option(None, "--dataset", "-d", help="Dataset name, supported datasets:" + ", ".join(SUPPORTED_DATASETS)),
    reference_key: Optional[str] = typer.Option(None, "--reference-key", "-rk", help="Reference key, use `vectormerge create-reference --check` to inspect"),
    embedding_path: Path = typer.Option(cli_defaults['embedding_path'], "--embedding-path", help="Path to embeddings"),
    reference_path: Path = typer.Option(cli_defaults['reference_path'], "--reference-path", help="Path to reference files"),
    mapping_param_path: Path = typer.Option(cli_defaults['mapping_param_path'], "--param-save-path", help="Path to save mapping parameters"),
    mapping_embedding_path: Path = typer.Option(cli_defaults['mapping_embedding_path'], "--embedding-save-path", help="Path to save mapping embeddings"),
    hidden_size: int = typer.Option(nonlinear_defaults.hidden_dim, "--hidden-size", help="Hidden layer size"),
    num_layers: int = typer.Option(nonlinear_defaults.num_layers, "--num-layers", help="Number of layers"),
    learning_rate: float = typer.Option(nonlinear_defaults.learning_rate, "--lr", help="Learning rate"),
    batch_size: int = typer.Option(nonlinear_defaults.batch_size, "--batch-size", help="Batch size"),
    epochs: int = typer.Option(nonlinear_defaults.num_epochs, "--epochs", help="Number of epochs"),
    dropout: float = typer.Option(nonlinear_defaults.dropout_rate, "--dropout", help="Dropout rate"),
    loss_function: str = typer.Option(nonlinear_defaults.loss_type, "--loss", help="Loss function (mse, cosine, ranking)"),
    force: bool = typer.Option(False, "--force", help="Force regeneration"),
    save_param: bool = typer.Option(nonlinear_defaults.save_param, "--save-param", help="Save mapping parameters"),
    save_embedding: bool = typer.Option(nonlinear_defaults.save_embedding, "--save-embedding", help="Save mapping embeddings"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive mode"),
    verbose: bool = typer.Option(cli_defaults['verbose'], "--verbose", "-v", help="Verbose output"),
):
    """Create mapping using nonlinear neural network."""
    
    # Set random seed
    set_seed()

    # Handle source_and_target_model parameter
    if source_and_target_model and "_" in source_and_target_model:
        parts = source_and_target_model.split("_", 1)
        if not source_model:
            source_model = parts[0]
        if not target_model:
            target_model = parts[1]
    
    # Interactive mode
    if interactive:
        if not source_model:
            rprint("[cyan]Select source model:[/cyan]")
            source_model = select_model_interactively()
        if not target_model:
            rprint("[cyan]Select target model:[/cyan]")
            target_model = select_model_interactively()
        if not dataset:
            dataset = select_dataset_interactively()
    
    # Validate inputs
    if not source_model or not target_model or not dataset:
        display_error_and_exit("Please specify source model, target model, and dataset (or use --interactive)")
    
    # Type assertions after validation
    assert source_model is not None and target_model is not None and dataset is not None and reference_key is not None
    
    # Validate models and dataset
    for model in [source_model, target_model]:
        is_valid, error_msg = validate_model_and_dataset(model, dataset)
        if not is_valid:
            display_error_and_exit(error_msg)
    
    # Create configuration
    from ..mapping.base import MappingConfig
    config = MappingConfig(
        hidden_dim=hidden_size,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_epochs=epochs,
        loss_type=loss_function,
        verbose=verbose
    )
    
    # Display loaded configuration
    rprint(f"[blue]🗺️ Using configuration from config.yaml:[/blue]")
    rprint(f"[cyan]Hidden size:[/cyan] {hidden_size}")
    rprint(f"[cyan]Number of layers:[/cyan] {num_layers}")
    rprint(f"[cyan]Learning rate:[/cyan] {learning_rate}")
    rprint(f"[cyan]Batch size:[/cyan] {batch_size}")
    rprint(f"[cyan]Epochs:[/cyan] {epochs}")
    rprint(f"[cyan]Dropout:[/cyan] {dropout}")
    rprint(f"[cyan]Loss function:[/cyan] {loss_function}")
    rprint(f"[cyan]Param save path:[/cyan] {mapping_param_path}")
    rprint(f"[cyan]Embedding save path:[/cyan] {mapping_embedding_path}")
    
    # Run mapping workflow with new paths
    _common_mapping_workflow(
        strategy="nonlinear",
        source_model=source_model,
        target_model=target_model,
        dataset=dataset,
        reference_key=reference_key,
        embedding_path=str(embedding_path),
        reference_path=str(reference_path),
        mapping_param_path=str(mapping_param_path),
        mapping_embedding_path=str(mapping_embedding_path),
        strategy_config=config,
        force=force,
        save_param=save_param,
        save_embedding=save_embedding,
        verbose=verbose,
    )


@mapping_app.command("la2m", help="Create mapping using LA2M strategy (clustering-based)")
def la2m_mapping(
    source_model: Optional[str] = typer.Option(None, "--source", "-s", help="Source model, supported models: " + ", ".join(SUPPORTED_MODELS)),
    target_model: Optional[str] = typer.Option(None, "--target", "-t", help="Target model, supported models: " + ", ".join(SUPPORTED_MODELS)),
    dataset: Optional[str] = typer.Option(None, "--dataset", "-d", help="Dataset name, supported datasets:" + ", ".join(SUPPORTED_DATASETS)),
    reference_key: Optional[str] = typer.Option(None, "--reference-key", "-rk", help="Reference key, use `vectormerge create-reference --check` to inspect"),
    embedding_path: Path = typer.Option(cli_defaults['embedding_path'], "--embedding-path", help="Path to embeddings"),
    reference_path: Path = typer.Option(cli_defaults['reference_path'], "--reference-path", help="Path to reference files"),
    mapping_param_path: Path = typer.Option(cli_defaults['mapping_param_path'], "--param-save-path", help="Path to save mapping parameters"),
    mapping_embedding_path: Path = typer.Option(cli_defaults['mapping_embedding_path'], "--embedding-save-path", help="Path to save mapping embeddings"),
    num_clusters: int = typer.Option(10, "--num-clusters", help="Number of clusters"),
    cluster_method: str = typer.Option("la2m-cluster", "--cluster-method", help="Clustering method (kmeans, hierarchical)"),
    local_strategy: str = typer.Option("procrustes", "--local-strategy", help="Local mapping strategy"),
    min_cluster_size: int = typer.Option(10, "--min-cluster-size", help="Minimum cluster size"),
    save_param: bool = typer.Option(False, "--save-param", help="Save mapping parameters"),
    save_embedding: bool = typer.Option(False, "--save-embedding", help="Save mapping embeddings"),
    with_scaling: bool = typer.Option(True, "--with-scaling", help="Use scaling"),
    with_rotation: bool = typer.Option(True, "--with-rotation", help="Use rotation"),
    approximate: bool = typer.Option(False, "--approximate", help="Use approximate SVD"),
    q: int = typer.Option(1500, "--q", help="Approximation parameter"),
    force: bool = typer.Option(False, "--force", help="Force regeneration"),
    save_transformed: bool = typer.Option(False, "--save-transformed", help="Save transformed embeddings"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive mode"),
    verbose: bool = typer.Option(cli_defaults['verbose'], "--verbose", "-v", help="Verbose output"),
):
    """Create mapping using LA2M strategy (clustering-based local mappings)."""

    # Set random seed
    set_seed()
    
    # Interactive mode
    if interactive:
        if not source_model:
            rprint("[cyan]Select source model:[/cyan]")
            source_model = select_model_interactively()
        if not target_model:
            rprint("[cyan]Select target model:[/cyan]")
            target_model = select_model_interactively()
        if not dataset:
            dataset = select_dataset_interactively()
    
    # Validate inputs
    if not source_model or not target_model or not dataset:
        display_error_and_exit("Please specify source model, target model, and dataset (or use --interactive)")
    
    # Type assertions after validation
    assert source_model is not None and target_model is not None and dataset is not None and reference_key is not None
    
    # Create configuration
    from ..mapping.base import MappingConfig
    config = MappingConfig(
        num_clusters=num_clusters,
        cluster_method=cluster_method,
        min_cluster_size=min_cluster_size,
        local_strategy=local_strategy,
        with_scaling=with_scaling,
        with_rotation=with_rotation,
        approximate=approximate,
        q=q,
        verbose=verbose
    )
    
    # Run mapping workflow
    _common_mapping_workflow(
        strategy="la2m",
        source_model=source_model,
        target_model=target_model,
        dataset=dataset,
        reference_key=reference_key,
        embedding_path=str(embedding_path),
        reference_path=str(reference_path),
        mapping_param_path=str(mapping_param_path),
        mapping_embedding_path=str(mapping_embedding_path),
        strategy_config=config,
        force=force,
        save_param=save_param,
        save_embedding=save_embedding,
        verbose=verbose,
    )


def _common_mapping_workflow(
    strategy: str, strategy_config: MappingConfig, source_model: str, target_model: str, dataset: str, reference_key: str, reference_path: str, embedding_path: str, mapping_param_path: str, mapping_embedding_path: str,  force: bool, save_param: bool, save_embedding: bool, verbose: bool):

    # Load embeddings
    source_embeddings = get_embedding(source_model, dataset, embedding_path, type_="corpus")
    target_embeddings = get_embedding(target_model, dataset, embedding_path, type_="corpus")

    # Load reference indices
    reference_data = get_reference(reference_path, reference_key)
    d0_index = reference_data['d0_index']
    
    # Create and fit mapper
    mapper = VectorSpaceMapper(strategy=strategy, 
                             config=strategy_config, 
                             dataset_name=dataset,
                             source_model=source_model,
                             target_model=target_model,
                             reference_key=reference_key,
                             mapping_param_path=mapping_param_path, 
                             mapping_embedding_path=mapping_embedding_path, 
                             save_param=save_param,
                             save_embedding=save_embedding,
                             force=force)
    
    # Transform embeddings
    mapper.fit_and_transform(source_embeddings, target_embeddings, d0_index)