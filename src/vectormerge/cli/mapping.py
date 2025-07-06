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
from ..embeddings import SUPPORTED_MODELS
from ..dataset import SUPPORTED_DATASETS

from .base import cli_defaults, set_seed, _common_mapping_workflow, console
from .utils import (
    select_model_interactively, select_dataset_interactively,
    validate_model_and_dataset, display_error_and_exit
)
from .config_loader import get_config

# Initialize mapping command group
mapping_app = typer.Typer(help="Create and manage embedding mappings")

# Load configuration
config_loader = get_config()

# Get mapper defaults
nonlinear_defaults = config_loader.get_mapper_defaults('nonlinear')
procrustes_defaults = config_loader.get_mapper_defaults('procrustes')
la2m_defaults = config_loader.get_mapper_defaults('la2m')


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
    
    # Type assertions after validation
    assert source_model is not None and target_model is not None and dataset is not None
    
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
    hidden_size: int = typer.Option(nonlinear_defaults['hidden_size'], "--hidden-size", help="Hidden layer size"),
    num_layers: int = typer.Option(nonlinear_defaults['num_layers'], "--num-layers", help="Number of layers"),
    learning_rate: float = typer.Option(nonlinear_defaults['learning_rate'], "--lr", help="Learning rate"),
    batch_size: int = typer.Option(nonlinear_defaults['batch_size'], "--batch-size", help="Batch size"),
    epochs: int = typer.Option(nonlinear_defaults['epochs'], "--epochs", help="Number of epochs"),
    dropout: float = typer.Option(nonlinear_defaults['dropout'], "--dropout", help="Dropout rate"),
    loss_function: str = typer.Option(nonlinear_defaults['loss_function'], "--loss", help="Loss function (mse, cosine, ranking)"),
    force: bool = typer.Option(False, "--force", help="Force regeneration"),
    save_param: bool = typer.Option(nonlinear_defaults['save_param'], "--save-param", help="Save mapping parameters"),
    save_embedding: bool = typer.Option(nonlinear_defaults['save_embedding'], "--save-embedding", help="Save mapping embeddings"),
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

def _enhanced_mapping_workflow(
    source_model: str, target_model: str, dataset: str, strategy: str,
    embedding_path: Path, reference_path: Path, param_save_path: Path, 
    embedding_save_path: Path, config, force: bool, save_transformed: bool,
    reference_key: Optional[str] = None
) -> None:
    """Enhanced mapping workflow with separate parameter and embedding save paths."""
    
    # Generate mapping name
    mapping_name = f"{source_model}_to_{target_model}_{dataset}_{strategy}"
    param_dir = param_save_path / mapping_name
    embedding_dir = embedding_save_path / mapping_name
    
    if param_dir.exists() and not force:
        rprint(f"[yellow]Mapping parameters already exist at {param_dir}[/yellow]")
        rprint("[yellow]Use --force to regenerate[/yellow]")
        return
    
    rprint(f"[blue]🗺️ Creating embedding mapping...[/blue]")
    rprint(f"[blue]Source model:[/blue] {source_model}")
    rprint(f"[blue]Target model:[/blue] {target_model}")
    rprint(f"[blue]Dataset:[/blue] {dataset}")
    rprint(f"[blue]Strategy:[/blue] {strategy}")
    rprint(f"[blue]Parameters will be saved to:[/blue] {param_dir}")
    if save_transformed:
        rprint(f"[blue]Embeddings will be saved to:[/blue] {embedding_dir}")
    
    try:
        from ..mapping import VectorSpaceMapper
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            
            # Load embeddings
            task = progress.add_task("Loading embeddings...", total=None)
            
            source_emb_file = embedding_path / f"{source_model}_{dataset}_corpus.npy"
            target_emb_file = embedding_path / f"{target_model}_{dataset}_corpus.npy"
            
            if not source_emb_file.exists():
                rprint(f"[red]Error:[/red] Source embeddings not found: {source_emb_file}")
                rprint("[yellow]Generate embeddings first with:[/yellow] vectormerge generate-embedding")
                raise typer.Exit(code=1)
            
            if not target_emb_file.exists():
                rprint(f"[red]Error:[/red] Target embeddings not found: {target_emb_file}")
                rprint("[yellow]Generate embeddings first with:[/yellow] vectormerge generate-embedding")
                raise typer.Exit(code=1)
            
            source_embeddings = np.load(source_emb_file)
            target_embeddings = np.load(target_emb_file)
            
            rprint(f"[green]✓[/green] Loaded embeddings: {source_embeddings.shape} -> {target_embeddings.shape}")
            
            # Load reference indices
            progress.update(task, description="Loading reference indices...")
            
            if reference_key:
                reference_file = reference_path / f"{dataset}_{reference_key}.npz"
            else:
                reference_file = reference_path / f"{dataset}_reference.npz"
                
            if not reference_file.exists():
                rprint(f"[red]Error:[/red] Reference file not found: {reference_file}")
                rprint("[yellow]Create reference first with:[/yellow] vectormerge create-reference")
                raise typer.Exit(code=1)
            
            reference_data = np.load(reference_file)
            d0_index = reference_data['d0_index']  # Reference indices
            
            rprint(f"[green]✓[/green] Loaded reference indices: {len(d0_index)} points")
            
            # Create and fit mapper
            progress.update(task, description=f"Initializing {strategy} mapper...")
            mapper = VectorSpaceMapper(strategy=strategy, config=config)
            
            progress.update(task, description=f"Training {strategy} mapping...")
            mapper.fit(source_embeddings, target_embeddings, d0_index)
            
            # Transform embeddings
            progress.update(task, description="Transforming embeddings...")
            transformed_embeddings = mapper.transform(source_embeddings)
            
            # Evaluate mapping on test data (D1 ∪ D2)
            if 'd1_index' in reference_data and 'd2_index' in reference_data:
                progress.update(task, description="Evaluating mapping...")
                d1_index = reference_data['d1_index']
                d2_index = reference_data['d2_index']
                test_indices = np.concatenate([d1_index, d2_index])
                
                metrics = mapper.evaluate_mapping(source_embeddings, target_embeddings, test_indices)
            else:
                # Use a random subset for evaluation if no split available
                test_size = min(1000, len(source_embeddings) // 4)
                test_indices = np.random.choice(
                    np.setdiff1d(np.arange(len(source_embeddings)), d0_index),
                    size=test_size, replace=False
                )
                metrics = mapper.evaluate_mapping(source_embeddings, target_embeddings, test_indices)
            
            # Save mapping parameters
            progress.update(task, description="Saving mapping parameters...")
            param_dir.mkdir(parents=True, exist_ok=True)
            mapper.save(param_dir / "mapper")
            
            # Save transformed embeddings if requested
            if save_transformed:
                progress.update(task, description="Saving transformed embeddings...")
                embedding_dir.mkdir(parents=True, exist_ok=True)
                transformed_file = embedding_dir / "transformed_embeddings.npy"
                np.save(transformed_file, transformed_embeddings)
                rprint(f"[green]✓[/green] Saved transformed embeddings: {transformed_file}")
            
            # Save mapping metadata to both locations
            metadata = {
                'source_model': source_model,
                'target_model': target_model,
                'dataset': dataset,
                'strategy': strategy,
                'config': config.to_dict(),
                'metrics': metrics,
                'reference_size': len(d0_index),
                'source_shape': source_embeddings.shape,
                'target_shape': target_embeddings.shape,
                'transformed_shape': transformed_embeddings.shape,
                'reference_key': reference_key
            }
            
            # Save metadata to parameter directory
            with open(param_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            # Save metadata to embedding directory if saving embeddings
            if save_transformed:
                with open(embedding_dir / "metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2)
            
            progress.update(task, description="Mapping completed!")
        
        # Display results
        from rich.panel import Panel
        from rich.table import Table
        
        # Create results table
        results_table = Table(show_header=True, header_style="bold magenta")
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="green")
        
        for metric, value in metrics.items():
            if isinstance(value, float):
                results_table.add_row(metric.upper(), f"{value:.6f}")
            else:
                results_table.add_row(metric.upper(), str(value))
        
        console.print(Panel(results_table, title="🎯 Mapping Results", border_style="green"))
        
        # Success message
        success_panel = Panel(
            f"[green]✅ Mapping completed successfully![/green]\n\n"
            f"📁 [cyan]Parameters saved to:[/cyan] {param_dir}\n"
            + (f"📁 [cyan]Embeddings saved to:[/cyan] {embedding_dir}\n" if save_transformed else "") +
            f"🔗 [cyan]Reference used:[/cyan] {reference_file.name}\n"
            f"📊 [cyan]Reference points:[/cyan] {len(d0_index)}\n"
            f"🎯 [cyan]Test MSE:[/cyan] {metrics.get('mse', 'N/A'):.6f}",
            title="🎉 Success",
            border_style="green"
        )
        console.print(success_panel)
        
    except Exception as e:
        rprint(f"[red]Error during mapping:[/red] {str(e)}")
        raise typer.Exit(code=1)


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