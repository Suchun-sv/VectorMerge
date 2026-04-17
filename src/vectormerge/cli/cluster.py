"""
Clustering commands for VectorMerge CLI.

This module contains commands for clustering operations including
reference point clustering and cluster analysis.
"""

from pathlib import Path
from typing import Optional, List
import typer
import numpy as np
from click import Context

from rich import print as rprint
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text

from vectormerge.cli.base import cli_defaults, console, set_seed
from vectormerge.cli.utils import (
    select_dataset_interactively,
    display_error_and_exit,
    display_success,
    select_model_interactively,
    handle_extra_args,
)
from vectormerge.clustering import ClusterManager, ClusteringConfig
from vectormerge.embeddings import get_embedding
from vectormerge.pipeline import cluster_pipeline

from .base import SUPPORTED_DATASETS, SUPPORTED_CLUSTERING_METHODS, SUPPORTED_MODELS
from dataclasses import replace

# Initialize create-cluster command group
create_cluster_app = typer.Typer(help="Create and manage clustering operations")


def _handle_interactive_clustering_input(dataset: str, model: str, cluster_method: str):
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
                choice = typer.prompt(
                    "Select clustering method (index or name)", default="kmeans"
                )
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

    return dataset, model, cluster_method


@create_cluster_app.callback(
    invoke_without_command=True,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def create_cluster(
    ctx: Context,
    dataset: str = typer.Option(
        None,
        "--dataset",
        "-d",
        help="Dataset name, the supported datasets are: "
        + ", ".join(SUPPORTED_DATASETS),
    ),
    model: str = typer.Option(
        None,
        "--model",
        "-m",
        help="Embedding model to use, the supported models are: "
        + ", ".join(SUPPORTED_MODELS),
    ),
    reference_key: str = typer.Option(
        None,
        "--reference-key",
        "-rk",
        help="Reference key (use `vectormerge create-reference --check` to inspect)",
    ),
    reference_path: str = typer.Option(
        cli_defaults.reference_path, "--reference-path", help="Path to reference files"
    ),
    cluster_method: str = typer.Option(
        "kmeans",
        "--cluster-method",
        help="Clustering method, the supported methods are: "
        + ", ".join(SUPPORTED_CLUSTERING_METHODS),
    ),
    cluster_path: str = typer.Option(
        cli_defaults.cluster_path, "--cluster-path", help="Path to save cluster results"
    ),
    embedding_path: str = typer.Option(
        cli_defaults.embedding_path, "--embedding-path", help="Path to embeddings"
    ),
    compute_metrics: bool = typer.Option(
        True, "--compute-metrics", help="Compute quality metrics"
    ),
    force: bool = typer.Option(False, "--force", help="Force re-clustering"),
    interactive: bool = typer.Option(
        False, "--interactive", "-i", help="Interactive mode"
    ),
    verbose: bool = typer.Option(
        cli_defaults.verbose, "--verbose", "-v", help="Verbose output"
    ),
):
    """Create clusters from reference embeddings.

    Use sub-commands for additional operations:
    - analyze: Analyze existing clustering results
    - distribution: Show cluster size distribution
    """

    # Set random seed
    set_seed()

    # Interactive mode
    if interactive:
        dataset, model, cluster_method = _handle_interactive_clustering_input(
            dataset, model, cluster_method
        )

    config = handle_extra_args(ctx)

    config = replace(
        config,
        verbose=verbose,
        cluster_path=cluster_path,
        embedding_path=embedding_path,
        reference_path=reference_path,
        clustering_config=replace(
            config.clustering_config,
            clustering_method=cluster_method,
            compute_metrics=compute_metrics,
        ),
    )

    # Validate inputs
    if not dataset:
        display_error_and_exit("Please specify dataset name (or use --interactive)")

    if dataset not in SUPPORTED_DATASETS:
        display_error_and_exit(
            f"Dataset '{dataset}' not supported. Use 'vectormerge list-datasets' to see available datasets."
        )

    rprint(f"[blue]🧮 Creating clusters for dataset: {dataset}[/blue]")
    rprint(f"[blue]Method:[/blue] {cluster_method}")

    cluster_pipeline(
        config=config,
        dataset=dataset,
        model=model,
        reference_key=reference_key,
        cluster_method=cluster_method,
        force=force,
        verbose=verbose,
    )

    rprint(f"[green]🧮 Clustering result saved to {cluster_path}[/green]")
