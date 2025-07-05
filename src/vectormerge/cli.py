"""
Command-line interface for VectorMerge.

This module provides a beautiful, modern CLI interface using Typer and Rich.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import yaml

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt, Confirm
from rich import print as rprint
from rich.syntax import Syntax

from .config import VectorMergeConfig, ModelConfig, DatasetConfig, get_default_config
from . import __version__, SUPPORTED_MODELS, SUPPORTED_DATASETS

# Initialize console and app
console = Console()
app = typer.Typer(
    name="vectormerge",
    help="🚀 VectorMerge: A modern library for embedding evaluation and vector space mapping",
    add_completion=False,
    rich_markup_mode="rich",
)

# Global options
def version_callback(value: bool) -> None:
    """Print version information."""
    if value:
        rprint(f"[bold blue]VectorMerge[/bold blue] version [bold green]{__version__}[/bold green]")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", callback=version_callback, help="Show version and exit"
    ),
) -> None:
    """
    🚀 VectorMerge: A modern library for embedding evaluation and vector space mapping
    
    Use --help with any command for more information.
    """
    pass


@app.command()
def evaluate(
    model_1: str = typer.Option(
        ..., "--model-1", "-m1", help="First embedding model to evaluate"
    ),
    model_2: str = typer.Option(
        ..., "--model-2", "-m2", help="Second embedding model to evaluate"
    ),
    dataset: str = typer.Option(
        ..., "--dataset", "-d", help="Dataset to evaluate on"
    ),
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Configuration file path (YAML)"
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output directory for results"
    ),
    batch_size: int = typer.Option(
        32, "--batch-size", "-b", help="Batch size for embedding generation"
    ),
    k_list: List[int] = typer.Option(
        [10, 100, 1000], "--k", help="List of k values for evaluation"
    ),
    mapping_method: str = typer.Option(
        "procrustes", "--mapping-method", help="Vector space mapping method"
    ),
    num_clusters: int = typer.Option(
        10, "--num-clusters", help="Number of clusters for mapping"
    ),
    seed: int = typer.Option(
        42, "--seed", help="Random seed for reproducibility"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
    use_wandb: bool = typer.Option(
        False, "--wandb", help="Enable Weights & Biases logging"
    ),
    wandb_project: str = typer.Option(
        "vectormerge", "--wandb-project", help="W&B project name"
    ),
) -> None:
    """
    🎯 Evaluate embeddings from two models on a specified dataset.
    
    This command performs comprehensive evaluation including:
    - Standard metrics (Recall, NDCG, MAP)
    - Vector space mapping
    - Extended evaluation metrics
    """
    
    # Validate inputs
    if model_1 not in SUPPORTED_MODELS:
        rprint(f"[red]Error:[/red] Model '{model_1}' not supported")
        rprint(f"[yellow]Supported models:[/yellow] {', '.join(SUPPORTED_MODELS)}")
        raise typer.Exit(code=1)
    
    if model_2 not in SUPPORTED_MODELS:
        rprint(f"[red]Error:[/red] Model '{model_2}' not supported")
        rprint(f"[yellow]Supported models:[/yellow] {', '.join(SUPPORTED_MODELS)}")
        raise typer.Exit(code=1)
    
    if dataset not in SUPPORTED_DATASETS:
        rprint(f"[red]Error:[/red] Dataset '{dataset}' not supported")
        rprint(f"[yellow]Supported datasets:[/yellow] {', '.join(SUPPORTED_DATASETS)}")
        raise typer.Exit(code=1)
    
    # Load or create configuration
    if config_file:
        if not config_file.exists():
            rprint(f"[red]Error:[/red] Configuration file not found: {config_file}")
            raise typer.Exit(code=1)
        
        try:
            config = VectorMergeConfig.from_yaml(config_file)
        except Exception as e:
            rprint(f"[red]Error loading configuration:[/red] {e}")
            raise typer.Exit(code=1)
    else:
        # Create configuration from CLI arguments
        config = VectorMergeConfig(
            model_1=ModelConfig(name=model_1, batch_size=batch_size),
            model_2=ModelConfig(name=model_2, batch_size=batch_size),
            dataset=DatasetConfig(name=dataset),
            seed=seed,
        )
        
        # Update from CLI arguments
        config.mapping.method = mapping_method
        config.mapping.num_clusters = num_clusters
        config.evaluation.k_list = k_list
        config.logging.use_wandb = use_wandb
        config.logging.wandb_project = wandb_project
        
        if verbose:
            config.logging.level = "DEBUG"
    
    # Display configuration
    rprint("\n[bold blue]📋 Configuration[/bold blue]")
    config_panel = Panel(
        f"[bold]Model 1:[/bold] {config.model_1.name}\n"
        f"[bold]Model 2:[/bold] {config.model_2.name}\n"
        f"[bold]Dataset:[/bold] {config.dataset.name}\n"
        f"[bold]Mapping Method:[/bold] {config.mapping.method}\n"
        f"[bold]K values:[/bold] {config.evaluation.k_list}\n"
        f"[bold]Seed:[/bold] {config.seed}",
        title="Evaluation Setup",
        border_style="blue",
    )
    console.print(config_panel)
    
    # Run evaluation with progress tracking
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            
            # Initialize evaluator
            task = progress.add_task("Initializing evaluator...", total=100)
            rprint("[yellow]Note:[/yellow] Full evaluation functionality will be implemented in future versions")
            progress.update(task, advance=100)
            
            # Placeholder for actual evaluation
            rprint("[green]✓[/green] Evaluation setup completed successfully!")
            
    except Exception as e:
        rprint(f"[red]Error during evaluation:[/red] {e}")
        raise typer.Exit(code=1)


@app.command()
def list_models() -> None:
    """
    📋 List all supported embedding models.
    """
    rprint("\n[bold blue]📋 Supported Models[/bold blue]")
    
    table = Table(title="Embedding Models")
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Type", style="magenta")
    table.add_column("Description", style="green")
    
    model_info = {
        "bert-base-uncased": ("Transformer", "BERT base model"),
        "roberta-base": ("Transformer", "RoBERTa base model"),
        "bge": ("Transformer", "BGE embedding model"),
        "nv-embed": ("Transformer", "NVIDIA embedding model"),
        "openai": ("API", "OpenAI embedding models"),
        "mistral": ("API", "Mistral embedding model"),
        "fast-text": ("Traditional", "FastText embeddings"),
        "word2vec": ("Traditional", "Word2Vec embeddings"),
        "glove": ("Traditional", "GloVe embeddings"),
    }
    
    for model in SUPPORTED_MODELS:
        model_type, description = model_info.get(model, ("Unknown", ""))
        table.add_row(model, model_type, description)
    
    console.print(table)


@app.command()
def list_datasets() -> None:
    """
    📋 List all supported datasets.
    """
    rprint("\n[bold blue]📋 Supported Datasets[/bold blue]")
    
    table = Table(title="BEIR Datasets")
    table.add_column("Dataset", style="cyan", no_wrap=True)
    table.add_column("Domain", style="magenta")
    table.add_column("Description", style="green")
    
    dataset_info = {
        "scifact": ("Scientific", "Scientific claim verification"),
        "nfcorpus": ("Medical", "Nutrition facts corpus"),
        "nq": ("General", "Natural Questions"),
        "cqadupstack": ("Technical", "Community question answering"),
        "arguana": ("Argumentative", "Argument mining"),
        "scidocs": ("Scientific", "Scientific document classification"),
        "fiqa": ("Financial", "Financial question answering"),
    }
    
    for dataset in SUPPORTED_DATASETS:
        domain, description = dataset_info.get(dataset, ("Unknown", ""))
        table.add_row(dataset, domain, description)
    
    console.print(table)


@app.command()
def generate_config(
    output_path: Path = typer.Option(
        "config.yaml", "--output", "-o", help="Output path for configuration file"
    ),
    model_1: str = typer.Option(
        "bert-base-uncased", "--model-1", help="First model"
    ),
    model_2: str = typer.Option(
        "roberta-base", "--model-2", help="Second model"
    ),
    dataset: str = typer.Option(
        "scifact", "--dataset", help="Dataset name"
    ),
    interactive: bool = typer.Option(
        False, "--interactive", "-i", help="Interactive configuration creation"
    ),
) -> None:
    """
    ⚙️ Generate a configuration file for VectorMerge.
    """
    
    if interactive:
        rprint("[bold blue]🎯 Interactive Configuration Generator[/bold blue]")
        
        # Interactive prompts
        model_1 = Prompt.ask(
            "First model", 
            default=model_1, 
            choices=SUPPORTED_MODELS
        )
        model_2 = Prompt.ask(
            "Second model", 
            default=model_2, 
            choices=SUPPORTED_MODELS
        )
        dataset = Prompt.ask(
            "Dataset", 
            default=dataset, 
            choices=SUPPORTED_DATASETS
        )
    
    # Create configuration
    config = VectorMergeConfig(
        model_1=ModelConfig(name=model_1),
        model_2=ModelConfig(name=model_2),
        dataset=DatasetConfig(name=dataset),
    )
    
    # Save configuration
    try:
        config.to_yaml(output_path)
        rprint(f"[green]✓[/green] Configuration saved to [bold]{output_path}[/bold]")
    except Exception as e:
        rprint(f"[red]Error saving configuration:[/red] {e}")
        raise typer.Exit(code=1)


@app.command()
def validate_config(
    config_path: Path = typer.Argument(..., help="Path to configuration file"),
) -> None:
    """
    ✅ Validate a configuration file.
    """
    
    if not config_path.exists():
        rprint(f"[red]Error:[/red] Configuration file not found: {config_path}")
        raise typer.Exit(code=1)
    
    try:
        config = VectorMergeConfig.from_yaml(config_path)
        rprint(f"[green]✓[/green] Configuration file [bold]{config_path}[/bold] is valid")
        
        # Display configuration summary
        rprint("\n[bold blue]📋 Configuration Summary[/bold blue]")
        summary_panel = Panel(
            f"[bold]Model 1:[/bold] {config.model_1.name}\n"
            f"[bold]Model 2:[/bold] {config.model_2.name}\n"
            f"[bold]Dataset:[/bold] {config.dataset.name}\n"
            f"[bold]Mapping Method:[/bold] {config.mapping.method}\n"
            f"[bold]K values:[/bold] {config.evaluation.k_list}",
            title="Configuration Details",
            border_style="green",
        )
        console.print(summary_panel)
        
    except Exception as e:
        rprint(f"[red]Error validating configuration:[/red] {e}")
        raise typer.Exit(code=1)


@app.command()
def doctor() -> None:
    """
    🔧 Run system diagnostics and check dependencies.
    """
    rprint("[bold blue]🔧 VectorMerge System Diagnostics[/bold blue]")
    
    # Check Python version
    python_version = sys.version_info
    rprint(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check dependencies
    dependencies = [
        "torch", "transformers", "faiss", "beir", "numpy", 
        "tqdm", "wandb", "typer", "rich", "yaml"
    ]
    
    table = Table(title="Dependencies Status")
    table.add_column("Package", style="cyan")
    table.add_column("Status", style="magenta")
    table.add_column("Version", style="green")
    
    for dep in dependencies:
        try:
            if dep == "yaml":
                import yaml
                version = getattr(yaml, "__version__", "unknown")
            else:
                module = __import__(dep)
                version = getattr(module, "__version__", "unknown")
            
            table.add_row(dep, "✓ Installed", version)
        except ImportError:
            table.add_row(dep, "✗ Missing", "N/A")
    
    console.print(table)


if __name__ == "__main__":
    app() 