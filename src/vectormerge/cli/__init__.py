"""
VectorMerge CLI - Main entry point.

This module assembles all CLI commands and provides the main application entry point.
"""

import typer
from rich import print as rprint
from pathlib import Path

from .base import set_seed
from .utils import version_callback
from .embedding import embedding_app
from .mapping import mapping_app
from .reference import create_reference
from .dataset import dataset_app
from .config import config_app, parse_dynamic_config
from .evaluate import evaluate_app
from .cluster import create_cluster_app


# Initialize main CLI application
app = typer.Typer(
    name="vectormerge",
    help="VectorMerge CLI - Advanced embedding generation and mapping tools",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)

# Add version callback
app.callback()(lambda version: None)
app.add_typer(embedding_app, name="generate-embedding", help="Generate and manage embeddings")
app.add_typer(mapping_app, name="map-embedding", help="Create and manage embedding mappings")
app.command("create-reference", help="Create and manage reference datasets")(create_reference)
app.add_typer(create_cluster_app, name="create-cluster", help="Create and manage clustering operations")
app.add_typer(dataset_app, name="dataset")
app.add_typer(config_app, name="config", help="Manage VectorMerge configuration")
app.add_typer(evaluate_app, name="evaluate", help="Evaluate the performance of a mapping model.")
# Add global options
@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", callback=version_callback, is_eager=True,
        help="Show version information"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
):
    """VectorMerge CLI - Advanced embedding generation and mapping tools."""
    # Set random seed for reproducibility
    set_seed()
    
    # Configure logging if verbose
    if verbose:
        import logging
        logging.basicConfig(level=logging.INFO)


# Add top-level commands for backward compatibility
@app.command("download-dataset", help="Download datasets")
def download_dataset_compat(
    dataset: str = typer.Argument(..., help="Dataset name"),
    data_path: Path = typer.Option(Path("./data/raw/"), "--data-path", help="Path to save datasets"),
    force: bool = typer.Option(False, "--force", help="Force re-download"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Download datasets (backward compatibility)."""
    # from ..dataset import download_dataset as download_dataset_func
    # download_dataset_func(dataset=dataset, data_path=data_path, force=force, interactive=False, verbose=verbose)
    from ..dataset import load_dataset
    load_dataset(dataset_name=dataset, data_path=data_path, force=force, interactive=False, verbose=verbose)


@app.command("list-datasets", help="List available datasets")
def list_datasets_compat(
    show_stats: bool = typer.Option(False, "--stats", help="Show dataset statistics"),
    data_path: Path = typer.Option(Path("./data/raw/"), "--data-path", help="Path to data directory"),
):
    """List available datasets (backward compatibility)."""
    from .dataset import list_datasets
    list_datasets(show_stats=show_stats, data_path=data_path)


@app.command("list-models", help="List available models")
def list_models():
    """List available models."""
    from .base import SUPPORTED_MODELS
    from rich.table import Table
    from .base import console
    
    rprint("[blue]📋 Available Models:[/blue]")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Index", style="dim", width=6)
    table.add_column("Model", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Description", style="yellow")
    
    for i, model in enumerate(SUPPORTED_MODELS, 1):
        if model.startswith("text-embedding"):
            model_type = "OpenAI API"
            desc = "OpenAI text embedding model"
        elif model.startswith("mistral"):
            model_type = "Mistral API"
            desc = "Mistral embedding model"
        elif model.startswith("sentence-transformers"):
            model_type = "Local"
            desc = "Sentence Transformers model"
        elif model.startswith("BAAI"):
            model_type = "Local"
            desc = "BGE embedding model"
        elif model.startswith("glove"):
            model_type = "Local"
            desc = "GloVe word vectors"
        elif model.startswith("word2vec"):
            model_type = "Local"
            desc = "Word2Vec vectors"
        else:
            model_type = "Unknown"
            desc = "Unknown model type"
        
        table.add_row(str(i), model, model_type, desc)
    
    console.print(table)

main_app = app

# For direct module usage
if __name__ == "__main__":
    app() 