"""
Embedding mapping commands for VectorMerge CLI.

This module contains commands for creating and managing embedding mappings.
"""

from pathlib import Path
from typing import Optional
import typer
from rich import print as rprint

from .base import cli_defaults, set_seed, _common_mapping_workflow
from .utils import (
    select_model_interactively, select_dataset_interactively,
    validate_model_and_dataset, display_error_and_exit
)

# Initialize mapping command group
mapping_app = typer.Typer(help="Create and manage embedding mappings")


@mapping_app.command("procrustes", help="Create mapping using Procrustes analysis")
def procrustes_mapping(
    source_model: Optional[str] = typer.Option(None, "--source", "-s", help="Source model"),
    target_model: Optional[str] = typer.Option(None, "--target", "-t", help="Target model"),
    dataset: Optional[str] = typer.Option(None, "--dataset", "-d", help="Dataset name"),
    embedding_path: Path = typer.Option(cli_defaults['embedding_path'], "--embedding-path", help="Path to embeddings"),
    reference_path: Path = typer.Option(cli_defaults['reference_path'], "--reference-path", help="Path to reference files"),
    output_path: Path = typer.Option(Path("./data/processed/mappings/"), "--output-path", help="Path to save mappings"),
    use_gpu: bool = typer.Option(False, "--gpu", help="Use GPU acceleration"),
    approximate: bool = typer.Option(False, "--approximate", help="Use approximate SVD"),
    force: bool = typer.Option(False, "--force", help="Force regeneration"),
    save_transformed: bool = typer.Option(False, "--save-transformed", help="Save transformed embeddings"),
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
    
    # Validate models and dataset
    for model in [source_model, target_model]:
        is_valid, error_msg = validate_model_and_dataset(model, dataset)
        if not is_valid:
            display_error_and_exit(error_msg)
    
    # Create configuration
    from ..mapping.base import MappingConfig
    config = MappingConfig(
        strategy="procrustes",
        use_gpu=use_gpu,
        approximate=approximate,
        verbose=verbose
    )
    
    # Run mapping workflow
    _common_mapping_workflow(
        source_model=source_model,
        target_model=target_model,
        dataset=dataset,
        strategy="procrustes",
        embedding_path=embedding_path,
        reference_path=reference_path,
        output_path=output_path,
        config=config,
        force=force,
        save_transformed=save_transformed
    )


# @mapping_app.command("linear", help="Create mapping using linear neural network")
# def linear_mapping(
#     source_model: Optional[str] = typer.Option(None, "--source", "-s", help="Source model"),
#     target_model: Optional[str] = typer.Option(None, "--target", "-t", help="Target model"),
#     dataset: Optional[str] = typer.Option(None, "--dataset", "-d", help="Dataset name"),
#     embedding_path: Path = typer.Option(cli_defaults['embedding_path'], "--embedding-path", help="Path to embeddings"),
#     reference_path: Path = typer.Option(cli_defaults['reference_path'], "--reference-path", help="Path to reference files"),
#     output_path: Path = typer.Option(Path("./data/processed/mappings/"), "--output-path", help="Path to save mappings"),
#     hidden_size: int = typer.Option(512, "--hidden-size", help="Hidden layer size"),
#     num_layers: int = typer.Option(2, "--num-layers", help="Number of layers"),
#     learning_rate: float = typer.Option(0.001, "--lr", help="Learning rate"),
#     batch_size: int = typer.Option(256, "--batch-size", help="Batch size"),
#     epochs: int = typer.Option(100, "--epochs", help="Number of epochs"),
#     loss_function: str = typer.Option("mse", "--loss", help="Loss function (mse, cosine, ranking)"),
#     force: bool = typer.Option(False, "--force", help="Force regeneration"),
#     save_transformed: bool = typer.Option(False, "--save-transformed", help="Save transformed embeddings"),
#     interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive mode"),
#     verbose: bool = typer.Option(cli_defaults['verbose'], "--verbose", "-v", help="Verbose output"),
# ):
#     """Create mapping using linear neural network."""
    
#     # Set random seed
#     set_seed()
    
#     # Interactive mode
#     if interactive:
#         if not source_model:
#             rprint("[cyan]Select source model:[/cyan]")
#             source_model = select_model_interactively()
#         if not target_model:
#             rprint("[cyan]Select target model:[/cyan]")
#             target_model = select_model_interactively()
#         if not dataset:
#             dataset = select_dataset_interactively()
    
#     # Validate inputs
#     if not source_model or not target_model or not dataset:
#         display_error_and_exit("Please specify source model, target model, and dataset (or use --interactive)")
    
#     # Validate models and dataset
#     for model in [source_model, target_model]:
#         is_valid, error_msg = validate_model_and_dataset(model, dataset)
#         if not is_valid:
#             display_error_and_exit(error_msg)
    
#     # Create configuration
#     from ..mapping.base import MappingConfig
#     config = MappingConfig(
#         strategy="linear",
#         hidden_size=hidden_size,
#         num_layers=num_layers,
#         learning_rate=learning_rate,
#         batch_size=batch_size,
#         epochs=epochs,
#         loss_function=loss_function,
#         verbose=verbose
#     )
    
#     # Run mapping workflow
#     _common_mapping_workflow(
#         source_model=source_model,
#         target_model=target_model,
#         dataset=dataset,
#         strategy="linear",
#         embedding_path=embedding_path,
#         reference_path=reference_path,
#         output_path=output_path,
#         config=config,
#         force=force,
#         save_transformed=save_transformed
#     )


@mapping_app.command("nonlinear", help="Create mapping using nonlinear neural network")
def nonlinear_mapping(
    source_model: Optional[str] = typer.Option(None, "--source", "-s", help="Source model"),
    target_model: Optional[str] = typer.Option(None, "--target", "-t", help="Target model"),
    dataset: Optional[str] = typer.Option(None, "--dataset", "-d", help="Dataset name"),
    embedding_path: Path = typer.Option(cli_defaults['embedding_path'], "--embedding-path", help="Path to embeddings"),
    reference_path: Path = typer.Option(cli_defaults['reference_path'], "--reference-path", help="Path to reference files"),
    output_path: Path = typer.Option(Path("./data/processed/mappings/"), "--output-path", help="Path to save mappings"),
    hidden_size: int = typer.Option(512, "--hidden-size", help="Hidden layer size"),
    num_layers: int = typer.Option(3, "--num-layers", help="Number of layers"),
    learning_rate: float = typer.Option(0.001, "--lr", help="Learning rate"),
    batch_size: int = typer.Option(256, "--batch-size", help="Batch size"),
    epochs: int = typer.Option(200, "--epochs", help="Number of epochs"),
    dropout: float = typer.Option(0.1, "--dropout", help="Dropout rate"),
    loss_function: str = typer.Option("mse", "--loss", help="Loss function (mse, cosine, ranking)"),
    force: bool = typer.Option(False, "--force", help="Force regeneration"),
    save_transformed: bool = typer.Option(False, "--save-transformed", help="Save transformed embeddings"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive mode"),
    verbose: bool = typer.Option(cli_defaults['verbose'], "--verbose", "-v", help="Verbose output"),
):
    """Create mapping using nonlinear neural network."""
    
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
    
    # Validate models and dataset
    for model in [source_model, target_model]:
        is_valid, error_msg = validate_model_and_dataset(model, dataset)
        if not is_valid:
            display_error_and_exit(error_msg)
    
    # Create configuration
    from ..mapping.base import MappingConfig
    config = MappingConfig(
        strategy="nonlinear",
        hidden_size=hidden_size,
        num_layers=num_layers,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        dropout=dropout,
        loss_function=loss_function,
        verbose=verbose
    )
    
    # Run mapping workflow
    _common_mapping_workflow(
        source_model=source_model,
        target_model=target_model,
        dataset=dataset,
        strategy="nonlinear",
        embedding_path=embedding_path,
        reference_path=reference_path,
        output_path=output_path,
        config=config,
        force=force,
        save_transformed=save_transformed
    )


@mapping_app.command("la2m", help="Create mapping using LA2M strategy (clustering-based)")
def la2m_mapping(
    source_model: Optional[str] = typer.Option(None, "--source", "-s", help="Source model"),
    target_model: Optional[str] = typer.Option(None, "--target", "-t", help="Target model"),
    dataset: Optional[str] = typer.Option(None, "--dataset", "-d", help="Dataset name"),
    embedding_path: Path = typer.Option(cli_defaults['embedding_path'], "--embedding-path", help="Path to embeddings"),
    reference_path: Path = typer.Option(cli_defaults['reference_path'], "--reference-path", help="Path to reference files"),
    output_path: Path = typer.Option(Path("./data/processed/mappings/"), "--output-path", help="Path to save mappings"),
    num_clusters: int = typer.Option(10, "--num-clusters", help="Number of clusters"),
    clustering_method: str = typer.Option("kmeans", "--clustering", help="Clustering method (kmeans, hierarchical)"),
    local_strategy: str = typer.Option("procrustes", "--local-strategy", help="Local mapping strategy"),
    min_cluster_size: int = typer.Option(10, "--min-cluster-size", help="Minimum cluster size"),
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
    
    # Validate models and dataset
    for model in [source_model, target_model]:
        is_valid, error_msg = validate_model_and_dataset(model, dataset)
        if not is_valid:
            display_error_and_exit(error_msg)
    
    # Create configuration
    from ..mapping.base import MappingConfig
    config = MappingConfig(
        strategy="la2m",
        num_clusters=num_clusters,
        clustering_method=clustering_method,
        local_strategy=local_strategy,
        min_cluster_size=min_cluster_size,
        verbose=verbose
    )
    
    # Run mapping workflow
    _common_mapping_workflow(
        source_model=source_model,
        target_model=target_model,
        dataset=dataset,
        strategy="la2m",
        embedding_path=embedding_path,
        reference_path=reference_path,
        output_path=output_path,
        config=config,
        force=force,
        save_transformed=save_transformed
    )


# Add aliases for backward compatibility
# mapping_app.command("ours", help="Alias for la2m")(la2m_mapping)
# mapping_app.command("LA2M", help="Uppercase alias for la2m")(la2m_mapping) 