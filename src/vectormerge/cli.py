"""
VectorMerge CLI module.

This module provides command-line interface for VectorMerge operations.
"""

import sys
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import yaml
import shutil
import torch

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt, Confirm
from rich import print as rprint
from rich.syntax import Syntax

from . import __version__, SUPPORTED_MODELS, SUPPORTED_DATASETS
from .embeddings import generate_embeddings
from .cli_util import select_dataset_interactively, select_model_interactively
from .config_loader import get_config

# Initialize console and get config
console = Console()
config = get_config()
cli_defaults = config.get_cli_defaults()

# Create the main app
app = typer.Typer(
    name="vectormerge",
    help="🚀 VectorMerge: A modern library for embedding evaluation and vector space mapping",
    add_completion=False,
    rich_markup_mode="rich"
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
    verbose: bool = typer.Option(cli_defaults['verbose'], "--verbose", "-v", help="Enable verbose output"),
) -> None:
    """
    🚀 VectorMerge: A modern library for embedding evaluation and vector space mapping
    
    Use --help with any command for more information.
    """
    if verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)


@app.command("download-dataset")
def download_dataset(
    dataset: Optional[str] = typer.Option(None, "-d", "--data", help="Dataset name to download"),
    data_path: Path = typer.Option(
        cli_defaults['data_path'], "--data-path", help="Path to store downloaded datasets"
    ),
    force: bool = typer.Option(cli_defaults['force_download'], "--force", help="Force re-download if exists"),
) -> None:
    """
    📥 Download and prepare datasets for evaluation.
    
    Downloads BEIR datasets and prepares them for embedding generation.
    """
    
    # Interactive dataset selection if not provided
    if not dataset:
        dataset = select_dataset_interactively()
    
    if dataset not in SUPPORTED_DATASETS:
        rprint(f"[red]Error:[/red] Dataset '{dataset}' not supported")
        rprint(f"[yellow]Supported datasets:[/yellow] {', '.join(SUPPORTED_DATASETS)}")
        raise typer.Exit(code=1)
    
    data_path.mkdir(parents=True, exist_ok=True)
    
    rprint(f"[blue]📥 Downloading dataset:[/blue] {dataset}")
    rprint(f"[blue]📂 Target path:[/blue] {data_path}")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            
            task = progress.add_task(f"Downloading {dataset}...", total=100)
            
            # Import here to avoid circular imports
            from beir import util
            from beir.datasets.data_loader import GenericDataLoader
            
            dataset_path = data_path / dataset
            
            if dataset_path.exists() and not force:
                rprint(f"[yellow]Dataset already exists at {dataset_path}[/yellow]")
                rprint(f"[yellow]Use --force to re-download[/yellow]")
                return
            
            progress.update(task, advance=20, description=f"Checking {dataset}...")
            
            # Download dataset
            url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
            data_path_str = str(data_path)
            util.download_and_unzip(url, data_path_str)
            
            progress.update(task, advance=60, description=f"Loading {dataset}...")
            
            # Verify download
            corpus, queries, qrels = GenericDataLoader(data_folder=str(dataset_path)).load(split="test")
            
            progress.update(task, advance=20, description="Verification complete")
            
            rprint(f"[green]✓[/green] Dataset downloaded successfully!")
            rprint(f"[green]📊 Stats:[/green]")
            rprint(f"  - Corpus: {len(corpus)} documents")
            rprint(f"  - Queries: {len(queries)} queries")
            rprint(f"  - Qrels: {len(qrels)} relevance judgments")
            
    except Exception as e:
        rprint(f"[red]Error downloading dataset:[/red] {e}")
        raise typer.Exit(code=1)


def show_success_tips() -> None:
    """Show helpful tips after successful embedding generation."""
    next_steps = Text()
    next_steps.append("💡 What's next?\n\n", style="bold yellow")
    next_steps.append("🔍 Verify completeness: ", style="dim")
    next_steps.append("vectormerge generate-embedding --check", style="bold cyan")
    next_steps.append("\n🗑️  Clean up cache: ", style="dim")
    next_steps.append("vectormerge generate-embedding --rm-cache", style="bold cyan")
    next_steps.append("\n📊 View configuration: ", style="dim")
    next_steps.append("vectormerge show-config", style="bold cyan")
    
    panel = Panel(
        next_steps,
        title="[bold green]🎉 Success![/bold green]",
        title_align="left",
        border_style="green",
        padding=(1, 2),
        expand=False
    )
    
    console.print(panel)


@app.command("generate-embedding")
def generate_embedding(
    model: Optional[str] = typer.Option(None, "-m", "--model", help="Model name for embedding generation"),
    dataset: Optional[str] = typer.Option(None, "-d", "--data", help="Dataset name"),
    type_: str = typer.Argument(cli_defaults['type'], help="Type of data to process (corpus/query/all)"),
    data_path: Path = typer.Option(
        cli_defaults['data_path'], "--data-path", help="Path to dataset files"
    ),
    embedding_path: Path = typer.Option(
        cli_defaults['embedding_path'], "--embedding-path", help="Path to store embeddings"
    ),
    cache_dir: Path = typer.Option(
        "./data/processed/embeddings/.cache/", "--cache-dir", help="Path to store embeddings cache"
    ),
    model_config: Optional[Path] = typer.Option(
        None, "--model-config", help="Path to model configuration file"
    ),
    batch_size: int = typer.Option(-1, "--batch-size", help="Batch size for embedding generation. If -1, use model default."),
    device: str = typer.Option("", "--device", help="Device to use (auto/cpu/cuda). If auto, use model default."),
    force: bool = typer.Option(False, "--force", help="Force regeneration of embeddings"),
    check: bool = typer.Option(False, "--check", help="Check embedding files completeness and consistency"),
    rm_cache: bool = typer.Option(False, "--rm-cache", help="Remove cache directory and exit"),
) -> None:
    """
    🔮 Generate embeddings for corpus and queries using specified model.
    
    Generates and caches embeddings for both corpus and query texts.
    Use --check to verify all model-dataset combinations exist.
    Use --rm-cache to clean up cache directory.
    """
    
    # If rm-cache mode, clean cache and exit
    if rm_cache:
        clean_cache(cache_dir)
        return
    
    # If check mode, run the check and exit
    if check:
        check_embedding_completeness(embedding_path)
        return
    
    # Use config defaults if not provided
    if not model:
        model = cli_defaults['model']
    
    if not dataset:
        dataset = cli_defaults['dataset']
    
    if model not in SUPPORTED_MODELS:
        rprint(f"[red]Error:[/red] Model '{model}' not supported")
        rprint(f"[yellow]Supported models:[/yellow] {', '.join(SUPPORTED_MODELS)}")
        raise typer.Exit(code=1)
    
    if dataset not in SUPPORTED_DATASETS:
        rprint(f"[red]Error:[/red] Dataset '{dataset}' not supported")
        rprint(f"[yellow]Supported datasets:[/yellow] {', '.join(SUPPORTED_DATASETS)}")
        raise typer.Exit(code=1)
    
    if type_ not in ["corpus", "query", "all"]:
        rprint(f"[red]Error:[/red] Invalid type: {type_}")
        rprint(f"[yellow]Valid types:[/yellow] corpus, query, all")
        raise typer.Exit(code=1)
    
    # Load model configuration from config loader first
    model_settings = config.get_model_settings(model)
    
    # Load additional config if provided
    if model_config and model_config.exists():
        with open(model_config, 'r') as f:
            config_data = yaml.safe_load(f)
            additional_settings = config_data.get(model, {})
            model_settings.update(additional_settings)
    
    # Override with CLI arguments if provided
    if batch_size != -1:
        model_settings["batch_size"] = batch_size
    elif "batch_size" not in model_settings:
        model_settings["batch_size"] = cli_defaults['batch_size']
        
    if device != "":
        model_settings["device"] = device
    elif "device" not in model_settings:
        model_settings["device"] = cli_defaults['device']
    
    rprint(f"[blue]Model settings:[/blue] {model_settings}")
    
    embedding_path.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Import here to avoid circular imports
        from .embeddings.generate import generate_embeddings
        
        if type_ == "all":
            rprint(f"[blue]Generating embeddings for corpus and queries...[/blue]")
            generate_embeddings(model, dataset, str(data_path), str(cache_dir), model_settings, type_="corpus", force=force, embedding_path=str(embedding_path))
            rprint(f"[blue]Generating embeddings for queries...[/blue]")
            generate_embeddings(model, dataset, str(data_path), str(cache_dir), model_settings, type_="query", force=force, embedding_path=str(embedding_path))
            rprint(f"[green]✓[/green] Embeddings generated successfully!")
            
            # Show helpful next steps
            show_success_tips()
        else:
            rprint(f"[blue]Generating embeddings for {type_}...[/blue]")
            generate_embeddings(model, dataset, str(data_path), str(cache_dir), model_settings, type_=type_, force=force, embedding_path=str(embedding_path))
            rprint(f"[green]✓[/green] Embeddings generated successfully!")
            
            # Show helpful next steps
            show_success_tips()
    except Exception as e:
        rprint(f"[red]Error generating embeddings:[/red] {e}")
        raise typer.Exit(code=1)
    

@app.command("list-models")
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
        "gte": ("Transformer", "GTE embedding model"),
    }
    
    for model in SUPPORTED_MODELS:
        model_type, description = model_info.get(model, ("Unknown", ""))
        table.add_row(model, model_type, description)
    
    console.print(table)


@app.command("list-datasets")
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


@app.command("init")
def init_command(
    scope: Optional[str] = typer.Option(
        None, "--scope", "-s", help="Configuration scope (global/local)"
    ),
) -> None:
    """
    🚀 Initialize VectorMerge configuration.
    
    Creates .vectormerge directory with default configuration file.
    """
    
    # Determine scope: global or local
    config_scope = scope
    
    if not config_scope:
        # Interactive prompt
        rprint("[bold blue]🎯 VectorMerge Configuration Setup[/bold blue]")
        rprint("\nWhere would you like to create the .vectormerge directory?")
        rprint("  [bold cyan]1. Global[/bold cyan] - ~/.vectormerge/ (affects all projects)")
        rprint("  [bold cyan]2. Local[/bold cyan] - ./.vectormerge/ (current project only)")
        
        choice = Prompt.ask(
            "\nChoose configuration scope",
            choices=["1", "2", "global", "local"],
            default="1"
        )
        
        if choice in ["1", "global"]:
            config_scope = "global"
        elif choice in ["2", "local"]:
            config_scope = "local"
    
    # Normalize scope names
    if config_scope in ["project", "local"]:
        config_scope = "local"
    elif config_scope in ["global", "user"]:
        config_scope = "global"
    
    # Determine target directory
    if config_scope == "global":
        target_dir = Path.home() / ".vectormerge"
    elif config_scope == "local":
        target_dir = Path.cwd() / ".vectormerge"
    else:
        rprint(f"[red]Error:[/red] Invalid scope: {config_scope}")
        rprint(f"[yellow]Valid scopes:[/yellow] global, local")
        raise typer.Exit(code=1)
    
    # Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy config file from configs/config.yaml
    source_config = Path("configs/config.yaml")
    target_config = target_dir / "config.yaml"
    
    if source_config.exists():
        shutil.copy2(source_config, target_config)
        rprint(f"[green]✓[/green] Configuration initialized at [bold]{target_config}[/bold]")
    else:
        # Create a basic config file if source doesn't exist
        with open(target_config, 'w') as f:
            f.write("""# VectorMerge Configuration File
# This file contains default settings for VectorMerge

# Default paths
default_data_path: "./data/raw/beir/"
default_embedding_path: "./data/processed/embeddings/"
default_output_path: "./results/"

# Model settings
default_model_1: "bert-base-uncased"
default_model_2: "roberta-base"
default_dataset: "scifact"
default_batch_size: 32
default_device: "auto"

# Advanced settings
verbose: false
use_cache: true
""")
        rprint(f"[green]✓[/green] Basic configuration created at [bold]{target_config}[/bold]")
    
    rprint(f"[blue]Scope:[/blue] {config_scope.title()}")
    rprint(f"[blue]You can edit this file to customize your settings.[/blue]")


@app.command("doctor")
def doctor() -> None:
    """
    🔧 Run system diagnostics and check dependencies.
    """
    rprint("[bold blue]🔧 VectorMerge System Diagnostics[/bold blue]")
    
    # Check Python version
    python_version = sys.version_info
    rprint(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check configuration directories
    global_config_dir = Path.home() / ".vectormerge"
    local_config_dir = Path.cwd() / ".vectormerge"
    
    rprint(f"\n[bold blue]⚙️ Configuration Status[/bold blue]")
    rprint(f"Global config dir: {global_config_dir}")
    rprint(f"Global exists: {'✓' if global_config_dir.exists() else '✗'}")
    rprint(f"Local config dir: {local_config_dir}")
    rprint(f"Local exists: {'✓' if local_config_dir.exists() else '✗'}")
    
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
    
    # Check directories
    rprint(f"\n[bold blue]📂 Directory Structure[/bold blue]")
    dirs_to_check = [
        "./data/raw/beir/",
        "./data/processed/embeddings/",
        "./data/processed/references/",
        "./data/processed/mappings/",
        "./results/",
        "./configs/"
    ]
    
    for dir_path in dirs_to_check:
        path = Path(dir_path)
        status = "✓ Exists" if path.exists() else "✗ Missing"
        rprint(f"  {dir_path}: {status}")


@app.command("show-config")
def show_config() -> None:
    """
    📋 Show current configuration and default values.
    
    Displays configuration loaded from all sources and current defaults.
    """
    rprint("\n[bold blue]📋 VectorMerge Configuration[/bold blue]")
    
    # Show configuration sources
    rprint(f"\n[bold green]Configuration Sources (in order of precedence):[/bold green]")
    rprint("1. CLI arguments (highest priority)")
    rprint("2. Project config (./.vectormerge/config.yaml)")
    rprint("3. Global config (~/.vectormerge/config.yaml)")
    rprint("4. Package config (configs/config.yaml)")
    rprint("5. Built-in defaults (lowest priority)")
    
    # Show current defaults
    rprint(f"\n[bold green]Current CLI Defaults:[/bold green]")
    defaults_table = Table(title="CLI Default Values")
    defaults_table.add_column("Setting", style="cyan", no_wrap=True)
    defaults_table.add_column("Value", style="green")
    defaults_table.add_column("Type", style="magenta")
    
    for key, value in cli_defaults.items():
        value_str = str(value)
        type_str = type(value).__name__
        defaults_table.add_row(key, value_str, type_str)
    
    console.print(defaults_table)
    
    # Show model configurations
    rprint(f"\n[bold green]Model Configurations:[/bold green]")
    if config.config.embedding_models:
        for model_name, model_config in config.config.embedding_models.items():
            model_table = Table(title=f"Model: {model_name}")
            model_table.add_column("Parameter", style="cyan")
            model_table.add_column("Value", style="green")
            
            for param, value in model_config.items():
                model_table.add_row(param, str(value))
            
            console.print(model_table)
    else:
        rprint("[yellow]No model configurations found[/yellow]")
    
    # Show configuration file locations
    rprint(f"\n[bold green]Configuration File Locations:[/bold green]")
    locations = [
        ("Package config", Path(__file__).parent.parent.parent / "configs" / "config.yaml"),
        ("Global config", Path.home() / ".vectormerge" / "config.yaml"),
        ("Project config", Path.cwd() / ".vectormerge" / "config.yaml"),
    ]
    
    for name, path in locations:
        exists = path.exists()
        status = "[green]✓ Found[/green]" if exists else "[red]✗ Not found[/red]"
        rprint(f"  {name}: {status} ({path})")


@app.command("create-config")
def create_config(
    scope: str = typer.Option("global", "--scope", "-s", help="Config scope (global/local)"),
    force: bool = typer.Option(False, "--force", help="Overwrite existing config"),
) -> None:
    """
    📝 Create a user configuration file.
    
    Creates a configuration file template that you can customize.
    """
    if scope not in ["global", "local"]:
        rprint(f"[red]Error:[/red] Invalid scope '{scope}'. Use 'global' or 'local'.")
        raise typer.Exit(code=1)
    
    # Determine target path
    if scope == "global":
        config_dir = Path.home() / ".vectormerge"
        config_path = config_dir / "config.yaml"
        rprint(f"[blue]Creating global config at:[/blue] {config_path}")
    else:
        config_dir = Path.cwd() / ".vectormerge"
        config_path = config_dir / "config.yaml"
        rprint(f"[blue]Creating local config at:[/blue] {config_path}")
    
    # Check if file exists
    if config_path.exists() and not force:
        rprint(f"[yellow]Config file already exists:[/yellow] {config_path}")
        rprint("[yellow]Use --force to overwrite[/yellow]")
        return
    
    # Create directory
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy example config
    try:
        example_config_path = Path(__file__).parent.parent.parent / "configs" / "example_user_config.yaml"
        if example_config_path.exists():
            import shutil
            shutil.copy2(example_config_path, config_path)
            rprint(f"[green]✓[/green] Configuration file created successfully!")
            rprint(f"[blue]Edit the file to customize your defaults:[/blue] {config_path}")
        else:
            rprint(f"[red]Error:[/red] Example config template not found")
            raise typer.Exit(code=1)
    except Exception as e:
        rprint(f"[red]Error creating config:[/red] {e}")
        raise typer.Exit(code=1)


def check_embedding_completeness(embedding_path: Path) -> None:
    """
    Check completeness and consistency of embedding files.
    
    1. Verify all SUPPORTED_MODEL × SUPPORTED_DATASET combinations exist
    2. Identify files not in supported combinations (but don't delete)
    """
    rprint("\n[bold blue]🔍 Checking Embedding Files Completeness[/bold blue]")
    
    if not embedding_path.exists():
        rprint(f"[red]Error:[/red] Embedding directory does not exist: {embedding_path}")
        raise typer.Exit(code=1)
    
    # Find all .npy files
    embedding_files = list(embedding_path.glob("*.npy"))
    
    rprint(f"[blue]📂 Scanning directory:[/blue] {embedding_path}")
    rprint(f"[blue]📊 Found {len(embedding_files)} .npy files[/blue]")
    
    # Parse existing files
    existing_combinations = set()
    unsupported_files = []
    
    for file_path in embedding_files:
        parsed = parse_embedding_filename(file_path.name)
        if parsed:
            type_, model, dataset = parsed
            if model in SUPPORTED_MODELS and dataset in SUPPORTED_DATASETS:
                existing_combinations.add((type_, model, dataset))
            else:
                unsupported_files.append((file_path.name, model, dataset))
        else:
            unsupported_files.append((file_path.name, "unknown", "unknown"))
    
    # Generate all expected combinations
    expected_combinations = set()
    for model in SUPPORTED_MODELS:
        for dataset in SUPPORTED_DATASETS:
            expected_combinations.add(("corpus", model, dataset))
            expected_combinations.add(("query", model, dataset))
    
    # Find missing combinations
    missing_combinations = expected_combinations - existing_combinations
    
    # Display results
    rprint(f"\n[bold green]✅ Coverage Analysis[/bold green]")
    rprint(f"Expected combinations: {len(expected_combinations)}")
    rprint(f"Found combinations: {len(existing_combinations)}")
    rprint(f"Missing combinations: {len(missing_combinations)}")
    rprint(f"Coverage: {len(existing_combinations)/len(expected_combinations)*100:.1f}%")
    
    # Show missing combinations
    if missing_combinations:
        rprint(f"\n[bold yellow]⚠️  Missing Embeddings ({len(missing_combinations)})[/bold yellow]")
        
        # Group by type for better display
        missing_corpus = [(m, d) for t, m, d in missing_combinations if t == "corpus"]
        missing_query = [(m, d) for t, m, d in missing_combinations if t == "query"]
        
        if missing_corpus:
            rprint(f"\n[yellow]Missing Corpus Embeddings:[/yellow]")
            table = Table(title="Missing Corpus Embeddings")
            table.add_column("Model", style="cyan")
            table.add_column("Dataset", style="green")
            table.add_column("Expected Filename", style="dim")
            
            for model, dataset in sorted(missing_corpus):
                filename = f"corpus_embeddings_{model}_{dataset}.npy"
                table.add_row(model, dataset, filename)
            
            console.print(table)
        
        if missing_query:
            rprint(f"\n[yellow]Missing Query Embeddings:[/yellow]")
            table = Table(title="Missing Query Embeddings")
            table.add_column("Model", style="cyan")
            table.add_column("Dataset", style="green")
            table.add_column("Expected Filename", style="dim")
            
            for model, dataset in sorted(missing_query):
                filename = f"query_embeddings_{model}_{dataset}.npy"
                table.add_row(model, dataset, filename)
            
            console.print(table)
    else:
        rprint(f"[green]🎉 All expected combinations are present![/green]")
    
    # Show unsupported files
    if unsupported_files:
        rprint(f"\n[bold blue]ℹ️  Files Outside Supported Combinations ({len(unsupported_files)})[/bold blue]")
        rprint("[dim]These files exist but are not part of the standard model-dataset matrix:[/dim]")
        
        table = Table(title="Additional Files")
        table.add_column("Filename", style="cyan")
        table.add_column("Parsed Model", style="yellow")
        table.add_column("Parsed Dataset", style="yellow")
        table.add_column("Status", style="magenta")
        
        for filename, model, dataset in sorted(unsupported_files):
            if model == "unknown":
                status = "Unrecognized format"
            elif model not in SUPPORTED_MODELS:
                status = f"Unsupported model: {model}"
            elif dataset not in SUPPORTED_DATASETS:
                status = f"Unsupported dataset: {dataset}"
            else:
                status = "Other"
            
            table.add_row(filename, model, dataset, status)
        
        console.print(table)
        rprint(f"[dim]💡 These files are kept as-is. Remove manually if not needed.[/dim]")
    
    # Summary
    rprint(f"\n[bold green]📋 Summary[/bold green]")
    rprint(f"✅ Found: {len(existing_combinations)} supported combinations")
    rprint(f"❌ Missing: {len(missing_combinations)} combinations")
    rprint(f"ℹ️  Additional: {len(unsupported_files)} files outside matrix")
    
    if missing_combinations:
        rprint(f"\n[yellow]💡 To generate missing embeddings, run:[/yellow]")
        rprint(f"[dim]vectormerge generate-embedding <model> <dataset> <type>[/dim]")


def parse_embedding_filename(filename: str) -> Optional[tuple]:
    """
    Parse embedding filename to extract type, model, and dataset.
    
    Expected format: {type}_embeddings_{model}_{dataset}.npy
    
    Returns:
        Tuple of (type, model, dataset) or None if parsing fails
    """
    if not filename.endswith('.npy'):
        return None
    
    # Remove .npy extension
    name_without_ext = filename[:-4]
    
    # Split by underscores
    parts = name_without_ext.split('_')
    
    if len(parts) < 4:
        return None
    
    # Expected format: type_embeddings_model_dataset
    if parts[1] != 'embeddings':
        return None
    
    type_ = parts[0]
    model = parts[2] 
    dataset = '_'.join(parts[3:])  # Handle datasets with underscores
    
    # Normalize model names
    model_mapping = {
        'gpt3': 'openai',
        'nv-embed': 'nv-embed',
        'fast-text': 'fast-text',
        # Add other mappings as needed
    }
    
    model = model_mapping.get(model.lower(), model.lower())
    dataset = dataset.lower()
    
    return type_, model, dataset


def clean_cache(cache_dir: Path) -> None:
    """
    Clean up cache directory.
    
    Args:
        cache_dir: Path to the cache directory to clean
    """
    import shutil
    from rich.prompt import Confirm
    
    rprint(f"\n[bold yellow]🗑️  Cache Cleanup[/bold yellow]")
    rprint(f"[blue]Cache directory:[/blue] {cache_dir}")
    
    if not cache_dir.exists():
        rprint(f"[yellow]⚠️  Cache directory does not exist: {cache_dir}[/yellow]")
        return
    
    # Calculate cache size
    cache_size = get_directory_size(cache_dir)
    cache_size_str = format_size(cache_size)
    
    # Count files
    file_count = sum(1 for _ in cache_dir.rglob("*") if _.is_file())
    
    # Show cache statistics in a beautiful panel
    stats_text = Text()
    stats_text.append("📊 Cache Statistics\n\n", style="bold blue")
    stats_text.append("📁 Directory: ", style="dim")
    stats_text.append(f"{cache_dir}\n", style="cyan")
    stats_text.append("📏 Total size: ", style="dim")
    stats_text.append(f"{cache_size_str}\n", style="bold yellow")
    stats_text.append("📄 Total files: ", style="dim")
    stats_text.append(f"{file_count:,}", style="bold yellow")
    
    stats_panel = Panel(
        stats_text,
        title="[bold blue]Cache Information[/bold blue]",
        title_align="left",
        border_style="blue",
        padding=(1, 2),
        expand=False
    )
    
    console.print(stats_panel)
    
    if file_count == 0:
        rprint(f"\n[green]✅ Cache directory is already empty.[/green]")
        return
    
    # Ask for confirmation
    rprint(f"\n[bold red]⚠️  Warning: This action cannot be undone![/bold red]")
    if not Confirm.ask(f"[bold yellow]Are you sure you want to delete the cache directory?[/bold yellow]"):
        rprint(f"\n[yellow]❌ Cache cleanup cancelled.[/yellow]")
        return
    
    try:
        # Remove cache directory
        shutil.rmtree(cache_dir)
        
        # Show beautiful success message
        success_text = Text()
        success_text.append("🗑️ Cache directory removed successfully!\n", style="bold green")
        success_text.append(f"💾 Freed up {cache_size_str} of disk space\n\n", style="dim green")
        success_text.append("💡 Tips:\n", style="bold yellow")
        success_text.append("• Cache will be rebuilt automatically when needed\n", style="dim")
        success_text.append("• Use ", style="dim")
        success_text.append("--cache-dir", style="bold cyan")
        success_text.append(" to specify custom cache location", style="dim")
        
        panel = Panel(
            success_text,
            title="[bold green]🎉 Cleanup Complete![/bold green]",
            title_align="left",
            border_style="green",
            padding=(1, 2),
            expand=False
        )
        
        console.print(panel)
        
    except Exception as e:
        rprint(f"[red]❌ Failed to remove cache directory: {e}[/red]")
        raise typer.Exit(code=1)


def get_directory_size(path: Path) -> int:
    """
    Calculate the total size of a directory.
    
    Args:
        path: Path to the directory
        
    Returns:
        Total size in bytes
    """
    total = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total += os.path.getsize(filepath)
                except (OSError, FileNotFoundError):
                    # Skip files that can't be accessed
                    continue
    except (OSError, FileNotFoundError):
        pass
    return total


def format_size(size_bytes: int) -> str:
    """
    Format size in bytes to human readable string.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"


if __name__ == "__main__":
    app(standalone_mode=False) 