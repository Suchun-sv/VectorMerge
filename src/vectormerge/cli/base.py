"""
Base constants and utilities for VectorMerge CLI.

This module contains shared constants, configuration, and utility functions
used across all CLI commands.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import print as rprint
import typer
from ..clustering import SUPPORTED_CLUSTERING_METHODS
from ..dataset import SUPPORTED_DATASETS
from ..embeddings import SUPPORTED_MODELS, get_embedding
from ..reference import get_reference
from ..mapping import VectorSpaceMapper, MappingConfig
from loguru import logger

# Initialize rich console
console = Console()

# Load CLI defaults
def load_cli_defaults() -> Dict[str, Any]:
    """Load CLI default configuration."""
    from .config_loader import VectorMergeConfig
    config = VectorMergeConfig()
    
    # return config.to_dict()
    return {
        "data_path": config.data_path,
        "embedding_path": config.embedding_path,
        "reference_path": config.reference_path,
        "mapping_path": config.mapping_path,
        "mapping_param_path": config.mapping_param_path,
        "mapping_embedding_path": config.mapping_embedding_path,
        "cluster_path": config.cluster_path,
        "dataset": "scifact",
        "type": "mistral",
        "verbose": False,
        "mapping_config": config.mapping_config.to_dict(),
        "clustering_config": config.clustering_config.to_dict(),
    }

cli_defaults = load_cli_defaults()


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    
    random.seed(seed)
    np.random.seed(seed)
    
    if not globals().get('_seed_logged', False):
        rprint(f"[dim]🎲 Random seed set to {seed} for reproducibility[/dim]")
        globals()['_seed_logged'] = True


# def _common_mapping_workflow(
#     source_model: str, target_model: str, dataset: str, strategy: str,
#     embedding_path: str, reference_path: Path, output_path: Path,
#     config, force: bool, save_transformed: bool
# ) -> None:
#     """Common workflow for all mapping strategies."""
#     # Generate mapping name
#     mapping_name = f"{source_model}_to_{target_model}_{dataset}_{strategy}"
#     mapping_dir = output_path / mapping_name
    
#     if mapping_dir.exists() and not force:
#         rprint(f"[yellow]Mapping already exists at {mapping_dir}[/yellow]")
#         rprint("[yellow]Use --force to regenerate[/yellow]")
#         return
    
#     rprint(f"[blue]🗺️ Creating embedding mapping...[/blue]")
#     rprint(f"[blue]Source model:[/blue] {source_model}")
#     rprint(f"[blue]Target model:[/blue] {target_model}")
#     rprint(f"[blue]Dataset:[/blue] {dataset}")
#     rprint(f"[blue]Strategy:[/blue] {strategy}")
#     rprint(f"[blue]Output path:[/blue] {mapping_dir}")
    
#     try:
#         from ..mapping import VectorSpaceMapper
        
#         with Progress(
#             SpinnerColumn(),
#             TextColumn("[progress.description]{task.description}"),
#             console=console,
#         ) as progress:
            
#             # Load embeddings
#             task = progress.add_task("Loading embeddings...", total=None)
            
#             source_embeddings = get_embedding(source_model, dataset, embedding_path, type_="corpus")
#             target_embeddings = get_embedding(target_model, dataset, embedding_path, type_="corpus")
            
#             if not source_emb_file.exists():
#                 rprint(f"[red]Error:[/red] Source embeddings not found: {source_emb_file}")
#                 rprint("[yellow]Generate embeddings first with:[/yellow] vectormerge generate-embedding")
#                 raise typer.Exit(code=1)
            
#             if not target_emb_file.exists():
#                 rprint(f"[red]Error:[/red] Target embeddings not found: {target_emb_file}")
#                 rprint("[yellow]Generate embeddings first with:[/yellow] vectormerge generate-embedding")
#                 raise typer.Exit(code=1)
            
#             source_embeddings = np.load(source_emb_file)
#             target_embeddings = np.load(target_emb_file)
            
#             rprint(f"[green]✓[/green] Loaded embeddings: {source_embeddings.shape} -> {target_embeddings.shape}")
            
#             # Load reference indices
#             progress.update(task, description="Loading reference indices...")
            
#             reference_file = reference_path / f"{dataset}_reference.npz"
#             if not reference_file.exists():
#                 rprint(f"[red]Error:[/red] Reference file not found: {reference_file}")
#                 rprint("[yellow]Create reference first with:[/yellow] vectormerge create-reference")
#                 raise typer.Exit(code=1)
            
#             reference_data = np.load(reference_file)
#             d0_index = reference_data['d0_index']  # Reference indices
            
#             rprint(f"[green]✓[/green] Loaded reference indices: {len(d0_index)} points")
            
#             # Create and fit mapper
#             progress.update(task, description=f"Initializing {strategy} mapper...")
#             mapper = VectorSpaceMapper(strategy=strategy, config=config)
            
#             progress.update(task, description=f"Training {strategy} mapping...")
#             mapper.fit(source_embeddings, target_embeddings, d0_index)
            
#             # Transform embeddings
#             progress.update(task, description="Transforming embeddings...")
#             transformed_embeddings = mapper.transform(source_embeddings)
            
#             # Evaluate mapping on test data (D1 ∪ D2)
#             if 'd1_index' in reference_data and 'd2_index' in reference_data:
#                 progress.update(task, description="Evaluating mapping...")
#                 d1_index = reference_data['d1_index']
#                 d2_index = reference_data['d2_index']
#                 test_indices = np.concatenate([d1_index, d2_index])
                
#                 metrics = mapper.evaluate_mapping(source_embeddings, target_embeddings, test_indices)
#             else:
#                 # Use a random subset for evaluation if no split available
#                 test_size = min(1000, len(source_embeddings) // 4)
#                 test_indices = np.random.choice(
#                     np.setdiff1d(np.arange(len(source_embeddings)), d0_index),
#                     size=test_size, replace=False
#                 )
#                 metrics = mapper.evaluate_mapping(source_embeddings, target_embeddings, test_indices)
            
#             # Save mapping
#             progress.update(task, description="Saving mapping...")
#             mapping_dir.mkdir(parents=True, exist_ok=True)
#             mapper.save(mapping_dir / "mapper")
            
#             # Save transformed embeddings if requested
#             if save_transformed:
#                 transformed_file = mapping_dir / "transformed_embeddings.npy"
#                 np.save(transformed_file, transformed_embeddings)
#                 rprint(f"[green]✓[/green] Saved transformed embeddings: {transformed_file}")
            
#             # Save mapping metadata
#             metadata = {
#                 'source_model': source_model,
#                 'target_model': target_model,
#                 'dataset': dataset,
#                 'strategy': strategy,
#                 'config': config.to_dict(),
#                 'metrics': metrics,
#                 'reference_size': len(d0_index),
#                 'source_shape': source_embeddings.shape,
#                 'target_shape': target_embeddings.shape,
#                 'transformed_shape': transformed_embeddings.shape
#             }
            
#             with open(mapping_dir / "metadata.json", "w") as f:
#                 json.dump(metadata, f, indent=2)
            
#             progress.update(task, description="Mapping completed!")
        
#         # Display results
#         rprint(f"\n[green]✅ Mapping created successfully![/green]")
        
#         # Create results table
#         results_table = Table(title="Mapping Results")
#         results_table.add_column("Metric", style="cyan", no_wrap=True)
#         results_table.add_column("Value", style="green")
        
#         results_table.add_row("MSE", f"{metrics['mse']:.6f}")
#         results_table.add_row("RMSE", f"{metrics['rmse']:.6f}")
#         results_table.add_row("MAE", f"{metrics['mae']:.6f}")
#         results_table.add_row("Cosine Similarity", f"{metrics['cosine_similarity']:.6f}")
#         results_table.add_row("Correlation", f"{metrics['correlation']:.6f}")
        
#         console.print(results_table)
        
#         # Get strategy info if available
#         strategy_info = mapper.get_strategy_info()
#         if 'training_time' in strategy_info.get('training_history', {}):
#             training_time = strategy_info['training_history']['training_time']
#             rprint(f"[blue]⏱️ Training time:[/blue] {training_time:.2f} seconds")
        
#         # Show strategy-specific information
#         if strategy == "la2m" and 'cluster_statistics' in strategy_info:
#             cluster_stats = strategy_info['cluster_statistics']
#             rprint(f"[blue]🔍 Clusters:[/blue] {cluster_stats['num_clusters']}")
#             rprint(f"[blue]🎯 Local mappings:[/blue] {cluster_stats['successful_local_mappings']}")
        
#         rprint(f"\n[blue]📁 Mapping saved to:[/blue] {mapping_dir}")
        
#         # Show helpful next steps
#         next_steps = Text()
#         next_steps.append("💡 What's next?\n\n", style="bold yellow")
#         next_steps.append("📈 Run evaluation: ", style="dim")
#         next_steps.append("vectormerge evaluate", style="bold cyan")
#         next_steps.append("\n🔍 View results: ", style="dim")
#         next_steps.append(f"ls {mapping_dir}", style="bold cyan")
#         next_steps.append("\n🗺️ Try different strategy: ", style="dim")
#         next_steps.append("vectormerge map-embedding la2m", style="bold cyan")
        
#         panel = Panel(
#             next_steps,
#             title="[bold green]🎉 Success![/bold green]",
#             title_align="left",
#             border_style="green",
#             padding=(1, 2),
#             expand=False
#         )
        
#         console.print(panel)
        
#     except Exception as e:
#         rprint(f"[red]Error creating mapping:[/red] {e}")
#         import traceback
#         rprint(f"[red]Traceback:[/red] {traceback.format_exc()}")
#         raise typer.Exit(code=1) 

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
    # mapper.fit(source_embeddings, target_embeddings, d0_index)
    # transformed_embeddings = mapper.transform(source_embeddings)
    mapper.fit_and_transform(source_embeddings, target_embeddings, d0_index)
