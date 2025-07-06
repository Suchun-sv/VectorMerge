"""
Embedding generation commands for VectorMerge CLI.

This module contains commands for generating, checking, and managing embeddings.
"""

import os
import shutil
from pathlib import Path
from typing import Optional, List, Tuple
import typer
from rich import print as rprint
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text

from .base import cli_defaults, console, set_seed, SUPPORTED_MODELS, SUPPORTED_DATASETS
from .utils import (
    select_model_interactively, select_dataset_interactively,
    validate_model_and_dataset, parse_embedding_filename,
    get_directory_size, format_size, confirm_action,
    display_success, display_error_and_exit
)

# Initialize embedding command group
embedding_app = typer.Typer(help="Generate and manage embeddings")


@embedding_app.command("generate", help="Generate embeddings for specified models and datasets")
def generate_embeddings(
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model name or 'all' for all models"),
    dataset: Optional[str] = typer.Option(None, "--dataset", "-d", help="Dataset name or 'all' for all datasets"),
    data_path: Path = typer.Option(cli_defaults['data_path'], "--data-path", help="Path to raw data directory"),
    embedding_path: Path = typer.Option(cli_defaults['embedding_path'], "--embedding-path", help="Path to save embeddings"),
    cache_dir: Path = typer.Option(Path("./cache/embeddings/"), "--cache-dir", help="Path to cache directory"),
    type_: str = typer.Option(cli_defaults['type'], "--type", help="Type of embeddings to generate"),
    batch_size: int = typer.Option(32, "--batch-size", help="Batch size for processing"),
    force: bool = typer.Option(False, "--force", help="Force regeneration of existing embeddings"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive mode"),
    rm_cache: bool = typer.Option(False, "--rm-cache", help="Remove cache directory after generation"),
    check: bool = typer.Option(False, "--check", help="Check embedding completeness"),
    verbose: bool = typer.Option(cli_defaults['verbose'], "--verbose", "-v", help="Verbose output"),
):
    """Generate embeddings for models and datasets."""
    
    # Set random seed
    set_seed()
    
    # Set up logging
    if verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
        
        # Only set VectorMerge loggers to DEBUG
        for logger_name in ['vectormerge', 'sentence_transformers']:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.DEBUG)
        
        # Set third-party loggers to INFO/WARNING
        for logger_name in ['httpx', 'mistralai', 'openai', 'urllib3']:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.INFO)
    
    # Handle check option
    if check:
        _check_embedding_completeness(embedding_path)
        return
    
    # Handle cache removal
    if rm_cache:
        _clean_cache(cache_dir)
        return
    
    # Interactive mode
    if interactive:
        if not model:
            model = select_model_interactively()
        if not dataset:
            dataset = select_dataset_interactively()
    
    # Validate inputs
    if not model or not dataset:
        display_error_and_exit("Please specify model and dataset (or use --interactive)")
    
    # Handle 'all' options
    models = SUPPORTED_MODELS if model == 'all' else [model]
    datasets = SUPPORTED_DATASETS if dataset == 'all' else [dataset]
    
    # Validate each model-dataset combination
    for m in models:
        for d in datasets:
            is_valid, error_msg = validate_model_and_dataset(m, d)
            if not is_valid:
                display_error_and_exit(error_msg)
    
    # Generate embeddings
    total_combinations = len(models) * len(datasets)
    rprint(f"[blue]📊 Generating embeddings for {total_combinations} combinations...[/blue]")
    
    from ..embeddings.generate import generate_embeddings as generate_embeddings_fn
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Generating embeddings...", total=total_combinations)
        
        for m in models:
            for d in datasets:
                progress.update(task, description=f"Processing {m} on {d}...")
                
                try:
                    generate_embeddings_fn(
                        model=m,
                        dataset=d,
                        data_path=data_path,
                        embedding_path=embedding_path,
                        cache_dir=cache_dir,
                        type_=type_,
                        batch_size=batch_size,
                        force=force
                    )
                    rprint(f"[green]✓[/green] Generated: {m} on {d}")
                except Exception as e:
                    rprint(f"[red]✗[/red] Failed: {m} on {d} - {e}")
                
                progress.advance(task)
    
    # Success message
    _show_generation_success(embedding_path, total_combinations)


def _check_embedding_completeness(embedding_path: Path) -> None:
    """Check completeness of embeddings."""
    if not embedding_path.exists():
        display_error_and_exit(f"Embedding path does not exist: {embedding_path}")
    
    rprint(f"[blue]🔍 Checking embedding completeness in {embedding_path}...[/blue]")
    
    # Parse existing files
    existing_files = []
    other_files = []
    
    for file_path in embedding_path.glob("*.npy"):
        parsed = parse_embedding_filename(file_path.name)
        if parsed:
            model, dataset, type_ = parsed
            existing_files.append((model, dataset, type_, file_path))
        else:
            other_files.append(file_path)
    
    # Create expected combinations
    expected_combinations = []
    for model in SUPPORTED_MODELS:
        for dataset in SUPPORTED_DATASETS:
            for type_ in ['corpus', 'query']:
                expected_combinations.append((model, dataset, type_))
    
    # Find missing combinations
    existing_combinations = {(m, d, t) for m, d, t, _ in existing_files}
    missing_combinations = []
    
    for combo in expected_combinations:
        if combo not in existing_combinations:
            missing_combinations.append(combo)
    
    # Display results
    total_expected = len(expected_combinations)
    total_existing = len(existing_combinations)
    total_missing = len(missing_combinations)
    
    rprint(f"\n[blue]📊 Embedding Coverage Report[/blue]")
    rprint(f"[green]✓ Found:[/green] {total_existing} embeddings")
    rprint(f"[red]✗ Missing:[/red] {total_missing} embeddings")
    rprint(f"[blue]📈 Coverage:[/blue] {total_existing/total_expected*100:.1f}%")
    
    if other_files:
        rprint(f"[yellow]⚠️  Other files:[/yellow] {len(other_files)} files")
    
    # Show missing combinations in organized table
    if missing_combinations:
        rprint("\n[red]❌ Missing Embeddings:[/red]")
        
        # Group by model
        by_model = {}
        for model, dataset, type_ in missing_combinations:
            if model not in by_model:
                by_model[model] = []
            by_model[model].append((dataset, type_))
        
        for model, items in by_model.items():
            table = Table(title=f"Missing: {model}")
            table.add_column("Dataset", style="cyan")
            table.add_column("Type", style="green")
            
            for dataset, type_ in items:
                table.add_row(dataset, type_)
            
            console.print(table)
    
    # Show helpful commands
    if missing_combinations:
        rprint("\n[blue]💡 To generate missing embeddings:[/blue]")
        rprint("[cyan]vectormerge generate-embedding --model all --dataset all[/cyan]")
        rprint("[cyan]vectormerge generate-embedding --model <model> --dataset <dataset>[/cyan]")


def _clean_cache(cache_dir: Path) -> None:
    """Clean cache directory."""
    if not cache_dir.exists():
        rprint(f"[yellow]Cache directory does not exist: {cache_dir}[/yellow]")
        return
    
    # Get cache statistics
    cache_size = get_directory_size(cache_dir)
    file_count = sum(1 for _ in cache_dir.rglob('*') if _.is_file())
    
    if cache_size == 0:
        rprint(f"[yellow]Cache directory is empty: {cache_dir}[/yellow]")
        return
    
    # Display cache info
    rprint(f"[blue]📁 Cache directory:[/blue] {cache_dir}")
    rprint(f"[blue]📊 Cache size:[/blue] {format_size(cache_size)}")
    rprint(f"[blue]📄 Files:[/blue] {file_count}")
    
    # Confirm deletion
    if confirm_action(f"Delete cache directory ({format_size(cache_size)})?"):
        try:
            shutil.rmtree(cache_dir)
            
            # Success message
            panel = Panel(
                Text.assemble(
                    ("🗑️ Cache cleaned successfully!\n\n", "bold green"),
                    ("💾 Space freed: ", "dim"),
                    (format_size(cache_size), "bold cyan"),
                    ("\n🔄 Cache will be recreated automatically when needed", "dim")
                ),
                title="[bold green]Cache Cleaned[/bold green]",
                border_style="green",
                padding=(1, 2)
            )
            console.print(panel)
            
        except Exception as e:
            display_error_and_exit(f"Failed to clean cache: {e}")
    else:
        rprint("[yellow]Cache cleaning cancelled[/yellow]")


def _show_generation_success(embedding_path: Path, total_combinations: int) -> None:
    """Show success message after embedding generation."""
    # Count actual files
    actual_files = len(list(embedding_path.glob("*.npy")))
    
    # Create success panel
    success_text = Text()
    success_text.append("🎉 Embedding generation completed!\n\n", style="bold green")
    success_text.append(f"📊 Generated: {total_combinations} combinations\n", style="")
    success_text.append(f"📁 Location: {embedding_path}\n", style="")
    success_text.append(f"📄 Total files: {actual_files}\n\n", style="")
    success_text.append("💡 What's next?\n", style="bold yellow")
    success_text.append("  📋 Check completeness: ", style="dim")
    success_text.append("vectormerge generate-embedding --check\n", style="cyan")
    success_text.append("  🗺️ Create mappings: ", style="dim")
    success_text.append("vectormerge map-embedding\n", style="cyan")
    success_text.append("  🔍 Create references: ", style="dim")
    success_text.append("vectormerge create-reference", style="cyan")
    
    panel = Panel(
        success_text,
        title="[bold green]✅ Success![/bold green]",
        title_align="left",
        border_style="green",
        padding=(1, 2),
        expand=False
    )
    
    console.print(panel)


# Add aliases for backward compatibility
# embedding_app.command("gen", help="Alias for generate")(generate_embeddings)
# embedding_app.command("g", help="Short alias for generate")(generate_embeddings) 