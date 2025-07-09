"""
Utility functions for VectorMerge CLI.

This module contains interactive selection functions, file validation,
and other helper utilities used across CLI commands.
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Tuple
from rich import print as rprint
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import typer

from .base import SUPPORTED_MODELS, SUPPORTED_DATASETS, SUPPORTED_REFERENCE_STRATEGIES, console


def select_model_interactively() -> str:
    """Interactive model selection."""
    rprint("[cyan]Available models:[/cyan]")
    
    # Create table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Index", style="dim", width=6)
    table.add_column("Model", style="cyan")
    table.add_column("Type", style="green")
    
    for i, model in enumerate(SUPPORTED_MODELS, 1):
        model_type = "API" if model.startswith(("text-embedding", "mistral")) else "Local"
        table.add_row(str(i), model, model_type)
    
    console.print(table)
    
    while True:
        try:
            choice = typer.prompt("Select model (index or name)")
            
            # Try by index
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(SUPPORTED_MODELS):
                    return SUPPORTED_MODELS[idx]
            
            # Try by name (partial match)
            for model in SUPPORTED_MODELS:
                if choice.lower() in model.lower():
                    return model
            
            rprint(f"[red]Invalid choice: {choice}[/red]")
        except typer.Abort:
            raise typer.Exit(code=1)

def select_reference_strategy_interactively() -> str:
    """Interactive reference strategy selection."""
    rprint("[cyan]Available reference strategies:[/cyan]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Index", style="dim", width=6)
    table.add_column("Strategy", style="cyan")
    for i, strategy in enumerate(SUPPORTED_REFERENCE_STRATEGIES, 1):
        table.add_row(str(i), strategy)
    console.print(table)
    while True:
        try:
            choice = typer.prompt("Select reference strategy (index or name)")
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(SUPPORTED_REFERENCE_STRATEGIES):
                    return SUPPORTED_REFERENCE_STRATEGIES[idx]
            for strategy in SUPPORTED_REFERENCE_STRATEGIES:
                if choice.lower() in strategy.lower():
                    return strategy
            rprint(f"[red]Invalid choice: {choice}[/red]")
        except typer.Abort:
            raise typer.Exit(code=1)


def select_dataset_interactively() -> str:
    """Interactive dataset selection."""
    rprint("[cyan]Available datasets:[/cyan]")
    
    # Create table with multiple columns
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Index", style="dim", width=6)
    table.add_column("Dataset", style="cyan")
    table.add_column("Index", style="dim", width=6)
    table.add_column("Dataset", style="cyan")
    
    # Split datasets into two columns
    half = len(SUPPORTED_DATASETS) // 2
    for i in range(half):
        left_idx = i + 1
        left_dataset = SUPPORTED_DATASETS[i]
        
        right_idx = i + half + 1 if i + half < len(SUPPORTED_DATASETS) else ""
        right_dataset = SUPPORTED_DATASETS[i + half] if i + half < len(SUPPORTED_DATASETS) else ""
        
        table.add_row(str(left_idx), left_dataset, str(right_idx), right_dataset)
    
    console.print(table)
    
    while True:
        try:
            choice = typer.prompt("Select dataset (index or name)")
            
            # Try by index
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(SUPPORTED_DATASETS):
                    return SUPPORTED_DATASETS[idx]
            
            # Try by name (partial match)
            for dataset in SUPPORTED_DATASETS:
                if choice.lower() in dataset.lower():
                    return dataset
            
            rprint(f"[red]Invalid choice: {choice}[/red]")
        except typer.Abort:
            raise typer.Exit(code=1)


def validate_model_and_dataset(model: str, dataset: str) -> Tuple[bool, str]:
    """Validate model and dataset combination."""
    if model not in SUPPORTED_MODELS:
        return False, f"Model '{model}' not supported. Use 'vectormerge list-models' to see available models."
    
    if dataset not in SUPPORTED_DATASETS:
        return False, f"Dataset '{dataset}' not supported. Use 'vectormerge list-datasets' to see available datasets."
    
    return True, ""


def check_file_exists(file_path: Path, description: str, required: bool = True) -> bool:
    """Check if a file exists and provide user feedback."""
    if file_path.exists():
        return True
    
    if required:
        rprint(f"[red]Error:[/red] {description} not found: {file_path}")
        return False
    else:
        rprint(f"[yellow]Warning:[/yellow] {description} not found: {file_path}")
        return False


def parse_embedding_filename(filename: str) -> Optional[Tuple[str, str, str]]:
    """
    Parse embedding filename to extract model, dataset, and type.
    
    Args:
        filename: Embedding filename (e.g., "model_dataset_corpus.npy")
        
    Returns:
        Tuple of (model, dataset, type) or None if parsing fails
    """
    if not filename.endswith('.npy'):
        return None
    
    # Remove .npy extension
    base_name = filename[:-4]
    
    # Split by underscore
    parts = base_name.split('_')
    
    if len(parts) < 3:
        return None
    
    # Last part is type (corpus/query)
    type_ = parts[-1]
    if type_ not in ['corpus', 'query']:
        return None
    
    # Second to last is dataset
    dataset = parts[-2]
    
    # Everything before that is model (may contain underscores)
    model = '_'.join(parts[:-2])
    
    return model, dataset, type_


def get_directory_size(path: Path) -> int:
    """Get total size of directory in bytes."""
    total_size = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file(follow_symlinks=False):
                total_size += entry.stat().st_size
            elif entry.is_dir(follow_symlinks=False):
                total_size += get_directory_size(Path(entry.path))
    except (OSError, PermissionError):
        pass
    return total_size


def format_size(size_bytes: int) -> str:
    """Format size in bytes to human readable format."""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"


def show_success_tips() -> None:
    """Show helpful tips after successful operations."""
    tips = [
        "💡 Use 'vectormerge doctor' to check system health",
        "📚 Check 'vectormerge --help' for all commands",
        "🔧 Configure defaults with 'vectormerge init'",
        "📊 View configs with 'vectormerge show-config'"
    ]
    
    import random
    tip = random.choice(tips)
    
    panel = Panel(
        tip,
        title="[bold blue]💡 Tip[/bold blue]",
        border_style="blue",
        padding=(0, 1)
    )
    console.print(panel)


def version_callback(value: bool) -> None:
    """Callback for version option."""
    if value:
        rprint("VectorMerge CLI version 1.0.0")
        raise typer.Exit()


def confirm_action(message: str, default: bool = False) -> bool:
    """Ask user for confirmation."""
    suffix = " [Y/n]" if default else " [y/N]"
    try:
        response = typer.prompt(message + suffix, default="" if not default else "y")
        if not response:
            return default
        return response.lower().startswith('y')
    except typer.Abort:
        return False


def display_error_and_exit(message: str, code: int = 1) -> None:
    """Display error message and exit."""
    rprint(f"[red]Error:[/red] {message}")
    raise typer.Exit(code=code)


def display_warning(message: str) -> None:
    """Display warning message."""
    rprint(f"[yellow]Warning:[/yellow] {message}")


def display_info(message: str) -> None:
    """Display info message."""
    rprint(f"[blue]Info:[/blue] {message}")


def display_success(message: str) -> None:
    """Display success message."""
    rprint(f"[green]Success:[/green] {message}") 