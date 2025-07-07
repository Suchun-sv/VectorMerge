"""
Clustering commands for VectorMerge CLI.

This module contains commands for clustering operations including
reference point clustering and cluster analysis.
"""

from pathlib import Path
from typing import Optional, List
import typer
import numpy as np
from rich import print as rprint
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text

from .base import cli_defaults, console, set_seed
from .utils import (
    select_dataset_interactively, validate_model_and_dataset,
    display_error_and_exit, display_success,
    select_model_interactively
)
from ..clustering import ClusterManager, ClusteringConfig
from ..embeddings import get_embedding
from ..reference import get_reference
from ..clustering import load_cluster_result, load_cluster_config
from .base import SUPPORTED_DATASETS, SUPPORTED_CLUSTERING_METHODS, SUPPORTED_MODELS

# Initialize create-cluster command group
create_cluster_app = typer.Typer(help="Create and manage clustering operations")

def _handle_interactive_clustering_input(dataset: str, model: str, cluster_method: str, num_clusters: int):
    """Handle interactive input for clustering parameters."""
    
    # Interactive dataset selection
    if not dataset:
        dataset = select_dataset_interactively()
    
    # Interactive model selection
    if not model:
        model = select_model_interactively()
    
    # Interactive clustering method selection
    if not cluster_method:
        method_options = ["kmeans", "la2m-cluster"]
        rprint("[cyan]Available clustering methods:[/cyan]")
        for i, method in enumerate(method_options, 1):
            rprint(f"  {i}. {method}")
        
        while True:
            try:
                choice = typer.prompt("Select clustering method (index or name)", default="kmeans")
                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(method_options):
                        cluster_method = method_options[idx]
                        break
                elif choice.lower() in method_options:
                    cluster_method = choice.lower()
                    break
                else:
                    rprint(f"[red]Invalid choice: {choice}[/red]")
            except typer.Abort:
                raise typer.Exit(code=1)
    
    # Interactive number of clusters input
    if num_clusters == 50:  # default value
        num_clusters = typer.prompt("Number of clusters", default=50, type=int)
    
    return dataset, model, cluster_method, num_clusters

@create_cluster_app.callback(invoke_without_command=True)
def create_cluster(
    ctx: typer.Context,
    dataset: str = typer.Option(None, "--dataset", "-d", help="Dataset name, the supported datasets are: " + ", ".join(SUPPORTED_DATASETS)),
    model: str = typer.Option(None, "--model", "-m", help="Embedding model to use, the supported models are: " + ", ".join(SUPPORTED_MODELS)),
    reference_key: str = typer.Option(None, "--reference-key", "-rk", help="Reference key (use `vectormerge create-reference --check` to inspect)"),
    reference_path: str = typer.Option(cli_defaults['reference_path'], "--reference-path", help="Path to reference files"),
    cluster_method: str = typer.Option("kmeans", "--cluster-method", help="Clustering method, the supported methods are: " + ", ".join(SUPPORTED_CLUSTERING_METHODS)),
    cluster_path: str = typer.Option(cli_defaults['cluster_path'], "--cluster-path", help="Path to save cluster results"),
    embedding_path: str = typer.Option(cli_defaults['embedding_path'], "--embedding-path", help="Path to embeddings"),
    num_clusters: int = typer.Option(50, "--num-clusters", help="Number of clusters"),
    min_cluster_size: int = typer.Option(5, "--min-cluster-size", help="Minimum cluster size"),
    compute_metrics: bool = typer.Option(True, "--compute-metrics", help="Compute quality metrics"),
    save_visualization: bool = typer.Option(False, "--save-visualization", help="Save cluster visualization"),
    visualization_method: str = typer.Option("tsne", "--visualization-method", help="Visualization method (tsne, pca, umap)"),
    force: bool = typer.Option(False, "--force", help="Force re-clustering"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive mode"),
    verbose: bool = typer.Option(cli_defaults['verbose'], "--verbose", "-v", help="Verbose output"),
):
    """Create clusters from reference embeddings.
    
    Use sub-commands for additional operations:
    - analyze: Analyze existing clustering results
    - distribution: Show cluster size distribution
    """
    
    # If a sub-command is being invoked, don't run the main clustering logic
    if ctx.invoked_subcommand is not None:
        return
    
    # Set random seed
    set_seed()
    
    # Interactive mode
    if interactive:
        dataset, model, cluster_method, num_clusters = _handle_interactive_clustering_input(dataset, model, cluster_method, num_clusters)
        
    # Validate inputs
    if not dataset:
        display_error_and_exit("Please specify dataset name (or use --interactive)")
    
    if dataset not in SUPPORTED_DATASETS:
        display_error_and_exit(f"Dataset '{dataset}' not supported. Use 'vectormerge list-datasets' to see available datasets.")

    # Create cluster manager
    strategy_config = ClusteringConfig(
        num_clusters=num_clusters,
        min_cluster_size=min_cluster_size,
        compute_metrics=compute_metrics,
    )

    cluster_manager = ClusterManager(
        dataset_name=dataset,
        model=model,
        reference_key=reference_key,
        reference_path=reference_path,
        cluster_path=cluster_path,
        embedding_path=embedding_path,
        strategy_name=cluster_method,
        strategy_config=strategy_config,
        force=force,
        verbose=verbose,
        auto_save_results=True
    )
    
    rprint(f"[blue]🧮 Creating clusters for dataset: {dataset}[/blue]")
    rprint(f"[blue]Method:[/blue] {cluster_method}")
    rprint(f"[blue]Number of clusters:[/blue] {num_clusters}")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Clustering...", total=None)
        
        try:
            # Fit cluster manager
            progress.update(task, description="Fitting cluster manager...")
            clustering_result = cluster_manager.fit()
            
            # Save visualization if requested
            if save_visualization:
                progress.update(task, description="Creating visualization...")
                try:
                    from ..clustering.utils import visualize_clusters
                    if cluster_manager.final_save_path is None:
                        raise ValueError("No save path specified. Clustering result not saved.")
                    
                    if not cluster_manager.final_save_path.parent.exists():
                        cluster_manager.final_save_path.parent.mkdir(parents=True, exist_ok=True)
                        
                    viz_path = cluster_manager.final_save_path.with_suffix(".png")
                    
                    embeddings = get_embedding(model, dataset, str(embedding_path), type_="corpus")
                    if embeddings is None:
                        raise ValueError(f"Could not load embeddings for visualization")

                    visualize_clusters(embeddings, clustering_result, 
                                     method=visualization_method, save_path=viz_path)
                    rprint(f"[green]✓[/green] Saved visualization: {viz_path}")
                except Exception as e:
                    rprint(f"[yellow]Warning: Could not create visualization: {e}[/yellow]")
            
            progress.update(task, description="Clustering completed!")
            
        except Exception as e:
            rprint(f"[red]Error creating clusters:[/red] {e}")
            import traceback
            if verbose:
                rprint(f"[red]Traceback:[/red] {traceback.format_exc()}")
            raise typer.Exit(code=1)
    
    # Display results
    if cluster_manager.final_save_path is not None:
        _show_clustering_results(dataset, cluster_method, num_clusters, cluster_manager.final_save_path, clustering_result, cluster_manager)
    
    if verbose:
        # Load embeddings for analysis
        embeddings = get_embedding(model, dataset, str(embedding_path), type_="corpus")
        if embeddings is not None:
            analysis = cluster_manager.analyze_cluster_quality(embeddings, clustering_result)
            _show_cluster_analysis(clustering_result, analysis, cluster_manager)
    
    display_success("Clustering completed successfully!")


def _show_clustering_results(dataset: str, method: str, num_clusters: int, output_path: Path, 
                           clustering_result, cluster_manager) -> None:
    """Show clustering results."""
    
    # Get cluster statistics
    stats = cluster_manager.get_cluster_statistics(clustering_result)
    
    # Create results table
    results_table = Table(title=f"Clustering Results: {dataset}")
    results_table.add_column("Metric", style="cyan", no_wrap=True)
    results_table.add_column("Value", style="green", justify="right")
    
    results_table.add_row("Method", method)
    results_table.add_row("Number of Clusters", str(stats['num_clusters']))
    results_table.add_row("Total Points", str(stats['total_points']))
    results_table.add_row("Reference Points", str(stats['total_reference_points']))
    results_table.add_row("Average Cluster Size", f"{stats['avg_cluster_size']:.1f}")
    results_table.add_row("Cluster Size Std", f"{stats['cluster_size_std']:.1f}")
    results_table.add_row("Min Cluster Size", str(stats['min_cluster_size']))
    results_table.add_row("Max Cluster Size", str(stats['max_cluster_size']))
    
    console.print(results_table)
    
    # Show quality metrics if available
    if stats['quality_metrics']:
        rprint(f"\n[blue]📊 Quality Metrics:[/blue]")
        metrics_table = Table(show_header=True, header_style="bold magenta")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="yellow", justify="right")
        
        for metric, value in stats['quality_metrics'].items():
            if isinstance(value, float):
                metrics_table.add_row(metric.replace('_', ' ').title(), f"{value:.4f}")
            else:
                metrics_table.add_row(metric.replace('_', ' ').title(), str(value))
        
        console.print(metrics_table)
    
    # Show cluster size distribution
    rprint(f"\n[blue]📋 Cluster Size Distribution:[/blue]")
    size_table = Table(show_header=True, header_style="bold green")
    size_table.add_column("Cluster ID", style="dim", width=10)
    size_table.add_column("Size", style="green", justify="right")
    size_table.add_column("Percentage", style="yellow", justify="right")
    
    total_points = stats['total_points']
    for i, size in enumerate(stats['cluster_sizes'][:10]):  # Show top 10
        percentage = (size / total_points * 100) if total_points > 0 else 0
        size_table.add_row(str(i), str(size), f"{percentage:.1f}%")
    
    if len(stats['cluster_sizes']) > 10:
        size_table.add_row("...", "...", "...")
    
    console.print(size_table)
    
    # Success message
    success_text = Text()
    success_text.append("🎉 Clustering completed successfully!\n\n", style="bold green")
    success_text.append(f"📁 Results saved to: {output_path}\n", style="")
    success_text.append(f"🧮 Method: {method}\n", style="")
    success_text.append(f"📊 Clusters: {stats['num_clusters']}\n", style="")
    success_text.append(f"📈 Points: {stats['total_points']}\n\n", style="")
    success_text.append("💡 What's next?\n", style="bold yellow")
    success_text.append("  🗺️ Create mappings: ", style="dim")
    success_text.append("vectormerge map-embedding la2m\n", style="cyan")
    success_text.append("  📈 Analyze results: ", style="dim")
    success_text.append("vectormerge cluster-analyze\n", style="cyan")
    success_text.append("  🎨 Visualize clusters: ", style="dim")
    success_text.append("vectormerge cluster-visualize", style="cyan")
    
    panel = Panel(
        success_text,
        title="[bold green]✅ Success![/bold green]",
        title_align="left",
        border_style="green",
        padding=(1, 2),
        expand=False
    )
    
    console.print(panel)


def _show_cluster_analysis(clustering_result, analysis, cluster_manager) -> None:
    """Show cluster analysis results."""
    
    stats = cluster_manager.get_cluster_statistics(clustering_result)
    
    rprint(f"[blue]📊 Cluster Analysis Results[/blue]")
    
    # Basic statistics
    basic_table = Table(title="Basic Statistics")
    basic_table.add_column("Metric", style="cyan")
    basic_table.add_column("Value", style="green", justify="right")
    
    basic_table.add_row("Total Clusters", str(stats['num_clusters']))
    basic_table.add_row("Total Points", str(stats['total_points']))
    basic_table.add_row("Average Cluster Size", f"{stats['avg_cluster_size']:.2f}")
    basic_table.add_row("Size Standard Deviation", f"{stats['cluster_size_std']:.2f}")
    basic_table.add_row("Size Range", f"{stats['min_cluster_size']} - {stats['max_cluster_size']}")
    
    console.print(basic_table)
    
    # Cluster size distribution histogram
    _show_cluster_size_distribution(stats['cluster_sizes'])
    
    # Quality metrics if available
    if analysis and 'quality_metrics' in analysis:
        quality_metrics = analysis['quality_metrics']
        
        rprint(f"\n[blue]🎯 Quality Metrics[/blue]")
        quality_table = Table(show_header=True, header_style="bold magenta")
        quality_table.add_column("Metric", style="cyan")
        quality_table.add_column("Value", style="yellow", justify="right")
        quality_table.add_column("Interpretation", style="green")
        
        # Add metrics with interpretations
        if 'silhouette_score' in quality_metrics:
            score = quality_metrics['silhouette_score']
            interpretation = "Excellent" if score > 0.7 else "Good" if score > 0.5 else "Fair" if score > 0.25 else "Poor"
            quality_table.add_row("Silhouette Score", f"{score:.4f}", interpretation)
        
        if 'calinski_harabasz_index' in quality_metrics:
            score = quality_metrics['calinski_harabasz_index']
            interpretation = "Higher is better"
            quality_table.add_row("Calinski-Harabasz Index", f"{score:.2f}", interpretation)
        
        if 'davies_bouldin_index' in quality_metrics:
            score = quality_metrics['davies_bouldin_index']
            interpretation = "Lower is better"
            quality_table.add_row("Davies-Bouldin Index", f"{score:.4f}", interpretation)
        
        console.print(quality_table)
    
    # Cluster balance analysis
    if analysis and 'cluster_balance' in analysis:
        balance = analysis['cluster_balance']
        
        rprint(f"\n[blue]⚖️ Cluster Balance Analysis[/blue]")
        balance_table = Table()
        balance_table.add_column("Metric", style="cyan")
        balance_table.add_column("Value", style="yellow", justify="right")
        balance_table.add_column("Status", style="green")
        
        cv = balance['size_coefficient_of_variation']
        cv_status = "Well Balanced" if cv < 0.3 else "Moderately Balanced" if cv < 0.7 else "Imbalanced"
        balance_table.add_row("Coefficient of Variation", f"{cv:.4f}", cv_status)
        
        is_balanced = balance['is_balanced']
        balance_status = "✅ Balanced" if is_balanced else "⚠️ Imbalanced"
        balance_table.add_row("Overall Balance", "Yes" if is_balanced else "No", balance_status)
        
        console.print(balance_table)


def _show_cluster_size_distribution(cluster_sizes: List[int]) -> None:
    """Show ASCII histogram of cluster size distribution."""
    from collections import Counter
    import math
    
    if not cluster_sizes:
        return
    
    rprint(f"\n[blue]📈 Cluster Size Distribution[/blue]")
    
    # Calculate frequency distribution
    size_counts = Counter(cluster_sizes)
    
    # Sort by size
    sorted_sizes = sorted(size_counts.items())
    
    if not sorted_sizes:
        rprint("[yellow]No data to display[/yellow]")
        return
    
    # Determine bin ranges for better visualization if we have many different sizes
    min_size = min(cluster_sizes)
    max_size = max(cluster_sizes)
    
    if len(sorted_sizes) > 15:  # Too many individual sizes, use bins
        num_bins = min(15, max_size - min_size + 1)
        bin_width = max(1, (max_size - min_size) // num_bins)
        
        # Create bins
        bins = {}
        for size in cluster_sizes:
            bin_start = (size // bin_width) * bin_width
            bin_key = f"{bin_start}-{bin_start + bin_width - 1}"
            bins[bin_key] = bins.get(bin_key, 0) + 1
        
        data_to_plot = sorted(bins.items(), key=lambda x: int(x[0].split('-')[0]))
    else:
        data_to_plot = sorted_sizes
    
    # Find max count for scaling
    max_count = max(count for _, count in data_to_plot)
    
    # Scale bars to fit in terminal (max 50 characters)
    max_bar_length = 50
    scale_factor = max_bar_length / max_count if max_count > 0 else 1
    
    # Create the histogram
    histogram_table = Table(show_header=True, header_style="bold green")
    histogram_table.add_column("Size Range", style="cyan", width=12)
    histogram_table.add_column("Count", style="yellow", justify="right", width=8)
    histogram_table.add_column("Frequency", style="green", justify="right", width=8)
    histogram_table.add_column("Bar", style="blue", width=52)
    
    total_clusters = len(cluster_sizes)
    
    for size_range, count in data_to_plot:
        # Calculate bar length
        bar_length = int(count * scale_factor)
        bar = "█" * bar_length
        
        # Calculate frequency percentage
        frequency = (count / total_clusters) * 100
        
        # Format size range
        size_str = str(size_range) if isinstance(size_range, int) else size_range
        
        histogram_table.add_row(
            size_str,
            str(count),
            f"{frequency:.1f}%",
            bar
        )
    
    console.print(histogram_table)
    
    # Summary statistics
    rprint(f"\n[dim]Distribution Summary:[/dim]")
    rprint(f"[dim]• Total clusters: {total_clusters}[/dim]")
    rprint(f"[dim]• Size range: {min_size} - {max_size}[/dim]")
    
    # Find most common size from original data
    most_common_size, most_common_count = size_counts.most_common(1)[0]
    rprint(f"[dim]• Most common size: {most_common_size} (appears {most_common_count} times)[/dim]")

@create_cluster_app.command("check", help="Check cluster results")
def cluster_check(
    cluster_path: str = typer.Option(cli_defaults['cluster_path'], "--cluster-path", help="Path to cluster results"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Check how many clusters (files with .pkl extension) are in the cluster path, and display their configurations
    """
    if isinstance(cluster_path, str):
        cluster_path_obj = Path(cluster_path)
    
    if not cluster_path_obj.exists():
        display_error_and_exit(f"Cluster path does not exist: {cluster_path_obj}")
    
    # Get all cluster keys (subdirectories)
    cluster_keys = [path.name for path in cluster_path_obj.glob("*") if path.is_dir()]
    
    if not cluster_keys:
        rprint(f"[yellow]No cluster keys found in {cluster_path}[/yellow]")
        return
    
    # First, show overview of all cluster keys
    overview_table = Table(title="Available Cluster Keys")
    overview_table.add_column("Index", style="dim", width=6)
    overview_table.add_column("Cluster Key", style="cyan")
    overview_table.add_column("Status", style="green")
    
    valid_cluster_keys = []
    
    for i, cluster_key in enumerate(cluster_keys, 1):
        key_path = cluster_path_obj / cluster_key
        
        try:
            config_dict = load_cluster_config(key_path)
            if config_dict is None:
                status = "❌ No config"
            else:
                status = "✅ Config available"
                valid_cluster_keys.append((cluster_key, config_dict))
        except Exception as e:
            status = f"⚠️ Error: {str(e)[:20]}..."
        
        overview_table.add_row(str(i), cluster_key, status)
    
    console.print(overview_table)
    
    if not valid_cluster_keys:
        rprint(f"\n[yellow]No valid cluster configurations found[/yellow]")
        return
    
    # Then, show detailed configuration for each valid cluster key
    rprint(f"\n[blue]📋 Detailed Configurations:[/blue]")
    
    PRIORITY_CONFIG_KEYS = ['dataset_name', 'model', 'reference_key', 'strategy_name']
    
    for cluster_key_index, (cluster_key, config_dict) in enumerate(valid_cluster_keys):
        # Create individual table for each cluster key
        detail_table = Table(title=f"Configuration: [{cluster_key_index + 1}/{len(valid_cluster_keys)}]. {cluster_key}")
        detail_table.add_column("Key", style="cyan", min_width=20)
        detail_table.add_column("Value", style="yellow")
        detail_table.add_column("Type", style="dim", width=12)
        
        # Add priority keys first
        rprint(f"\n[bold]{cluster_key}[/bold]")
        
        # Show priority configurations
        priority_table = Table(show_header=False, box=None, padding=(0, 1))
        priority_table.add_column("Key", style="cyan")
        priority_table.add_column("Value", style="yellow")
        
        for key in PRIORITY_CONFIG_KEYS:
            if key in config_dict:
                priority_table.add_row(f"{key}:", str(config_dict[key]))
        
        console.print(priority_table)
        
        if verbose:
            # Show all other configurations
            other_configs = {k: v for k, v in config_dict.items() if k not in PRIORITY_CONFIG_KEYS}
            
            if other_configs:
                rprint(f"[dim]Additional configurations:[/dim]")
                other_table = Table(show_header=False, box=None, padding=(0, 1))
                other_table.add_column("Key", style="dim")
                other_table.add_column("Value", style="dim")
                
                for key, value in other_configs.items():
                    other_table.add_row(f"{key}:", str(value))
                
                console.print(other_table)
    
    # Add hint about verbose mode if not in verbose mode
    if not verbose and valid_cluster_keys:
        rprint(f"\n[dim]💡 Use --verbose to see additional configuration details[/dim]")
    
    # Summary
    rprint(f"\n[blue]📊 Summary:[/blue]")
    rprint(f"Total cluster keys: {len(cluster_keys)}")
    total_files = sum(len(list((cluster_path_obj / key).glob("**/*.pkl"))) for key in cluster_keys)
    rprint(f"Total cluster files: {total_files}")
    
    # Show cluster path
    rprint(f"\n[dim]📁 Cluster path: {cluster_path}[/dim]")

def select_cluster_key_interactively(cluster_path: str) -> str:
    """Select cluster key interactively."""
    cluster_path_obj = Path(cluster_path)
    if not cluster_path_obj.exists():
        display_error_and_exit(f"Cluster path does not exist: {cluster_path}")
    
    cluster_keys = [path.name for path in cluster_path_obj.glob("*") if path.is_dir()]
    
    if not cluster_keys:
        display_error_and_exit(f"No cluster keys found in {cluster_path}")
    
    rprint("[cyan]Available cluster keys:[/cyan]")
    for i, key in enumerate(cluster_keys, 1):
        rprint(f"  {i}. {key}")
    
    while True:
        try:
            choice = typer.prompt("Select cluster key (index or name)", default=cluster_keys[0])
            
            # Check if it's a valid index
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(cluster_keys):
                    return cluster_keys[idx]
                else:
                    rprint(f"[red]Invalid index: {choice}[/red]")
                    continue
            
            # Check if it's a valid name
            if choice in cluster_keys:
                return choice
            
            rprint(f"[red]Invalid choice: {choice}[/red]")
            rprint("[yellow]Please enter a valid index number or cluster key name[/yellow]")
            
        except typer.Abort:
            raise typer.Exit(code=1)


def cluster_visualize(
    cluster_path: Path = typer.Option(cli_defaults['cluster_path'], "--cluster-path", help="Path to cluster results"),
    cluster_key: str = typer.Option(None, "--cluster-key", help="Cluster key, use `vectormerge create-cluster check` to inspect"),
    dataset: str = typer.Option(None, "--dataset", "-d", help="Dataset name"),
    cluster_method: str = typer.Option("kmeans", "--cluster-method", help="Clustering method (kmeans, la2m-cluster)"),
    num_clusters: int = typer.Option(50, "--num-clusters", help="Number of clusters"),
    embedding_path: Path = typer.Option(cli_defaults['embedding_path'], "--embedding-path", help="Path to embeddings"),
    model: str = typer.Option("mistral", "--model", help="Embedding model,"),
    reference_key: str = typer.Option(None, "--reference-key", "-rk", help="Reference key (use `vectormerge create-reference --check` to inspect)"),
    reference_path: str = typer.Option(cli_defaults['reference_path'], "--reference-path", help="Path to reference files"),
    visualization_method: str = typer.Option("tsne", "--method", help="Visualization method (tsne, pca, umap)"),
    output_path: Path = typer.Option(None, "--output", help="Output path for visualization"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive mode"),
    verbose: bool = typer.Option(cli_defaults['verbose'], "--verbose", "-v", help="Verbose output"),
    force: bool = typer.Option(False, "--force", help="Force re-clustering"),
):
    """Visualize clustering results."""
    
    # Interactive mode
    if interactive and not dataset:
        dataset = select_dataset_interactively()
    
    if not dataset:
        display_error_and_exit("Please specify dataset name (or use --interactive)")
    
    # Load clustering result
    cluster_result_path = cluster_path / dataset / cluster_method / f"clusters_{num_clusters}"
    if not cluster_result_path.exists():
        display_error_and_exit(f"Clustering result not found: {cluster_result_path}")
    
    rprint(f"[blue]🎨 Creating visualization for clustering: {dataset}[/blue]")
    
    try:
        # Load clustering result
        cluster_manager = ClusterManager(
            dataset_name=dataset,
            model=model,
            reference_key=reference_key,
            reference_path=reference_path,
            cluster_path=str(cluster_path),
            embedding_path=str(embedding_path),
            strategy_name=cluster_method,
            force=force
        )
        clustering_result = cluster_manager.load()
        
        # Load embeddings
        embeddings = get_embedding(model, dataset, str(embedding_path), type_="corpus")
        if embeddings is None:
            display_error_and_exit(f"Embeddings not found for model '{model}' and dataset '{dataset}'")
        
        # Set output path
        if output_path is None:
            output_path = cluster_result_path / f"visualization_{visualization_method}.png"
        
        # Create visualization
        from ..clustering.utils import visualize_clusters
        visualize_clusters(embeddings, clustering_result, 
                         method=visualization_method, save_path=output_path)
        
        rprint(f"[green]✅ Visualization saved to: {output_path}[/green]")
        
    except Exception as e:
        rprint(f"[red]Error creating visualization:[/red] {e}")
        if verbose:
            import traceback
            rprint(f"[red]Traceback:[/red] {traceback.format_exc()}")
        raise typer.Exit(code=1)


@create_cluster_app.command("distribution", help="Show cluster size distribution")
def cluster_distribution(
    cluster_path: str = typer.Option(cli_defaults['cluster_path'], "--cluster-path", help="Path to cluster results"),
    dataset: str = typer.Option(None, "--dataset", "-d", help="Dataset name"),
    cluster_method: str = typer.Option("kmeans", "--cluster-method", help="Clustering method (kmeans, la2m-cluster)"),
    model: str = typer.Option("mistral", "--model", help="Embedding model"),
    reference_key: str = typer.Option(None, "--reference-key", "-rk", help="Reference key"),
    reference_path: str = typer.Option(cli_defaults['reference_path'], "--reference-path", help="Path to reference files"),
    embedding_path: str = typer.Option(cli_defaults['embedding_path'], "--embedding-path", help="Path to embeddings"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive mode"),
    verbose: bool = typer.Option(cli_defaults['verbose'], "--verbose", "-v", help="Verbose output"),
):
    """Show detailed cluster size distribution histogram."""
    
    # Interactive mode
    if interactive and not dataset:
        dataset = select_dataset_interactively()
    
    if not dataset:
        display_error_and_exit("Please specify dataset name (or use --interactive)")
    
    rprint(f"[blue]📊 Cluster Size Distribution for: {dataset}[/blue]")
    
    try:
        # Load clustering result
        cluster_manager = ClusterManager(
            dataset_name=dataset,
            model=model,
            reference_key=reference_key,
            reference_path=reference_path,
            cluster_path=cluster_path,
            embedding_path=embedding_path,
            strategy_name=cluster_method,
            force=False
        )
        clustering_result = cluster_manager.load()
        
        # Get cluster statistics
        stats = cluster_manager.get_cluster_statistics(clustering_result)
        
        # Show detailed distribution
        _show_cluster_size_distribution(stats['cluster_sizes'])
        
        # Additional statistics
        rprint(f"\n[blue]📋 Additional Statistics[/blue]")
        additional_table = Table()
        additional_table.add_column("Metric", style="cyan")
        additional_table.add_column("Value", style="green", justify="right")
        
        additional_table.add_row("Mean Size", f"{stats['avg_cluster_size']:.2f}")
        additional_table.add_row("Std Deviation", f"{stats['cluster_size_std']:.2f}")
        additional_table.add_row("Median Size", f"{np.median(stats['cluster_sizes']):.1f}")
        additional_table.add_row("Size Variance", f"{stats['cluster_size_std']**2:.2f}")
        
        # Calculate quartiles
        q1 = np.percentile(stats['cluster_sizes'], 25)
        q3 = np.percentile(stats['cluster_sizes'], 75)
        additional_table.add_row("Q1 (25th percentile)", f"{q1:.1f}")
        additional_table.add_row("Q3 (75th percentile)", f"{q3:.1f}")
        additional_table.add_row("IQR", f"{q3-q1:.1f}")
        
        console.print(additional_table)
        
    except Exception as e:
        rprint(f"[red]Error loading clustering results:[/red] {e}")
        if verbose:
            import traceback
            rprint(f"[red]Traceback:[/red] {traceback.format_exc()}")
        raise typer.Exit(code=1)

# Add analyze sub-command
@create_cluster_app.command("analyze", help="Analyze clustering results")
def cluster_analyze(
    cluster_key: str = typer.Option(None, "--cluster-key", "-ck", help="Cluster key, use `vectormerge create-cluster check` to inspect"),
    cluster_path: str = typer.Option(cli_defaults['cluster_path'], "--cluster-path", help="Path to cluster results"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive mode"),
    verbose: bool = typer.Option(cli_defaults['verbose'], "--verbose", "-v", help="Verbose output"),
    force: bool = typer.Option(False, "--force", help="Force re-clustering"),
):
    """Analyze clustering results by loading embeddings and cluster data directly."""
    
    # Set random seed
    set_seed()
    
    # Interactive mode
    if interactive:
        cluster_key = select_cluster_key_interactively(cluster_path)
    
    if not cluster_key:
        display_error_and_exit("Please specify cluster key (or use --interactive)")
    
    cluster_path_obj = Path(cluster_path)
    cluster_key_path = cluster_path_obj / cluster_key
    
    if not cluster_key_path.exists():
        display_error_and_exit(f"Cluster key does not exist: {cluster_key_path}")
    
    try:
        # Load cluster configuration
        config_dict = load_cluster_config(cluster_key_path)
        if config_dict is None:
            display_error_and_exit(f"No configuration found for cluster key: {cluster_key}")
        
        # Type assertion: config_dict is guaranteed to be not None after the check above
        assert config_dict is not None
        
        # Extract configuration
        dataset = config_dict['dataset_name']
        model = config_dict['model']
        reference_key = config_dict['reference_key']
        reference_path = config_dict.get('reference_path', cli_defaults['reference_path'])
        embedding_path = config_dict.get('embedding_path', cli_defaults['embedding_path'])
        cluster_method = config_dict['strategy_name']
        
        rprint(f"[blue]📊 Analyzing clustering results for: {cluster_key}[/blue]")
        rprint(f"[dim]Dataset: {dataset}, Model: {model}, Method: {cluster_method}[/dim]")
        
        # Load embeddings
        rprint(f"[blue]🔄 Loading embeddings...[/blue]")
        embeddings = get_embedding(model, dataset, str(embedding_path), type_="corpus")
        if embeddings is None:
            display_error_and_exit(f"Could not load embeddings for model '{model}' and dataset '{dataset}'")
        
        # Type assertion: embeddings is guaranteed to be not None after the check above
        assert embeddings is not None
        
        rprint(f"[green]✓[/green] Loaded embeddings: {embeddings.shape}")
        
        # Load clustering results
        rprint(f"[blue]🔄 Loading clustering results...[/blue]")
        clustering_result = load_cluster_result(cluster_key_path)
        if clustering_result is None:
            display_error_and_exit(f"Could not load clustering results from {cluster_key_path}")
        
        # Type assertion: clustering_result is guaranteed to be not None after the check above
        assert clustering_result is not None
        
        rprint(f"[green]✓[/green] Loaded clustering results: {clustering_result.num_clusters} clusters")
        
        # Create a minimal cluster manager for analysis utilities
        strategy_config = ClusteringConfig.from_dict(config_dict)
        cluster_manager = ClusterManager(
            dataset_name=dataset,
            model=model,
            reference_key=reference_key,
            reference_path=reference_path,
            cluster_path=str(cluster_path_obj),
            embedding_path=embedding_path,
            strategy_name=cluster_method,
            strategy_config=strategy_config,
            force=False,
            verbose=verbose,
            auto_save_results=False
        )
        
        # Perform comprehensive analysis
        rprint(f"[blue]🔍 Performing cluster analysis...[/blue]")
        
        # 1. Basic cluster statistics
        stats = cluster_manager.get_cluster_statistics(clustering_result)
        
        # 2. Quality analysis
        analysis = cluster_manager.analyze_cluster_quality(embeddings, clustering_result)
        
        # Display analysis results
        _show_cluster_analysis(clustering_result, analysis, cluster_manager)
        
        # Additional detailed analysis if verbose
        if verbose:
            rprint(f"\n[blue]🔬 Detailed Analysis:[/blue]")
            
            # Show cluster size details
            rprint(f"\n[yellow]Cluster Size Details:[/yellow]")
            for i, cluster_data in enumerate(clustering_result.cluster_data_list):
                rprint(f"  Cluster {i:2d}: {len(cluster_data.ref_index):4d} reference points, "
                      f"{len(cluster_data.bound_index):4d} bound points, "
                      f"diameter: {cluster_data.diameter:.4f}")
            
            # Show embedding statistics
            rprint(f"\n[yellow]Embedding Statistics:[/yellow]")
            rprint(f"  Embedding dimensions: {embeddings.shape[1]}")
            rprint(f"  Total data points: {embeddings.shape[0]}")
            rprint(f"  Memory usage: {embeddings.nbytes / 1024 / 1024:.2f} MB")
            
            # Show quality metrics breakdown
            if analysis and 'quality_metrics' in analysis:
                metrics = analysis['quality_metrics']
                rprint(f"\n[yellow]Quality Metrics Breakdown:[/yellow]")
                for metric_name, value in metrics.items():
                    if isinstance(value, float):
                        rprint(f"  {metric_name}: {value:.6f}")
                    else:
                        rprint(f"  {metric_name}: {value}")
        
        display_success("Cluster analysis completed successfully!")
        
    except KeyError as e:
        display_error_and_exit(f"Missing configuration key: {e}")
    except Exception as e:
        rprint(f"[red]Error analyzing clusters:[/red] {e}")
        import traceback
        if verbose:
            rprint(f"[red]Traceback:[/red] {traceback.format_exc()}")
        raise typer.Exit(code=1) 