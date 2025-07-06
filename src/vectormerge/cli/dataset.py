"""
Dataset management commands for VectorMerge CLI.

This module contains commands for downloading, listing, and managing datasets.
"""

from pathlib import Path
from typing import Optional, List
import typer
from rich import print as rprint
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text

from .base import cli_defaults, console, SUPPORTED_DATASETS
from .utils import (
    select_dataset_interactively, format_size, display_error_and_exit,
    display_success, display_info
)

# Initialize dataset command group
dataset_app = typer.Typer(help="Download and manage datasets")


@dataset_app.command("download", help="Download datasets")
def download_dataset(
    dataset: Optional[str] = typer.Option(None, "--dataset", "-d", help="Dataset name or 'all' for all datasets"),
    data_path: Path = typer.Option(cli_defaults['data_path'], "--data-path", help="Path to save datasets"),
    force: bool = typer.Option(False, "--force", help="Force re-download"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive mode"),
    verbose: bool = typer.Option(cli_defaults['verbose'], "--verbose", "-v", help="Verbose output"),
):
    """Download BEIR datasets."""
    
    # Interactive mode
    if interactive:
        if not dataset:
            dataset = select_dataset_interactively()
    
    # Validate inputs
    if not dataset:
        display_error_and_exit("Please specify dataset name (or use --interactive)")
    
    # Handle 'all' option
    datasets = SUPPORTED_DATASETS if dataset == 'all' else [dataset]
    
    # Validate each dataset
    for d in datasets:
        if d not in SUPPORTED_DATASETS:
            display_error_and_exit(f"Dataset '{d}' not supported. Use 'vectormerge list-datasets' to see available datasets.")
    
    # Create data path
    data_path.mkdir(parents=True, exist_ok=True)
    
    # Download datasets
    total_datasets = len(datasets)
    rprint(f"[blue]📥 Downloading {total_datasets} dataset(s)...[/blue]")
    
    from ..dataset import load_dataset
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Downloading datasets...", total=total_datasets)
        
        success_count = 0
        failed_datasets = []
        
        for d in datasets:
            progress.update(task, description=f"Downloading {d}...")
            
            try:
                # Check if dataset already exists
                dataset_path = data_path / d
                if dataset_path.exists() and not force:
                    rprint(f"[yellow]⚠️  Dataset already exists: {d}[/yellow]")
                    if not force:
                        rprint("[yellow]Use --force to re-download[/yellow]")
                        continue
                
                # Load dataset (this will download if not exists)
                dataset_obj = load_dataset(d, data_path=data_path, force_download=force)
                
                # Get dataset statistics
                stats = dataset_obj.get_stats()
                
                rprint(f"[green]✓[/green] Downloaded: {d}")
                rprint(f"  📄 Documents: {stats['num_documents']:,}")
                rprint(f"  ❓ Queries: {stats['num_queries']:,}")
                rprint(f"  🔗 Qrels: {stats['num_qrels']:,}")
                
                success_count += 1
                
            except Exception as e:
                rprint(f"[red]✗[/red] Failed: {d} - {e}")
                failed_datasets.append(d)
                if verbose:
                    import traceback
                    rprint(f"[red]Traceback:[/red] {traceback.format_exc()}")
            
            progress.advance(task)
    
    # Show results
    _show_download_results(success_count, failed_datasets, data_path)


@dataset_app.command("list", help="List available datasets")
def list_datasets(
    show_stats: bool = typer.Option(False, "--stats", help="Show dataset statistics"),
    data_path: Path = typer.Option(cli_defaults['data_path'], "--data-path", help="Path to data directory"),
):
    """List available datasets."""
    
    rprint("[blue]📋 Available BEIR Datasets:[/blue]")
    
    # Create table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Index", style="dim", width=6)
    table.add_column("Dataset", style="cyan")
    table.add_column("Status", style="green")
    
    if show_stats:
        table.add_column("Documents", style="yellow", justify="right")
        table.add_column("Queries", style="yellow", justify="right")
        table.add_column("Size", style="blue", justify="right")
    
    # Add datasets to table
    for i, dataset in enumerate(SUPPORTED_DATASETS, 1):
        # Check if dataset is downloaded
        dataset_path = data_path / dataset
        status = "✓ Downloaded" if dataset_path.exists() else "⚪ Not downloaded"
        
        if show_stats and dataset_path.exists():
            try:
                from ..dataset import load_dataset
                dataset_obj = load_dataset(dataset, data_path=data_path)
                stats = dataset_obj.get_stats()
                
                # Calculate size
                size = format_size(sum(
                    f.stat().st_size for f in dataset_path.rglob('*') if f.is_file()
                ))
                
                table.add_row(
                    str(i), 
                    dataset, 
                    status,
                    f"{stats['num_documents']:,}",
                    f"{stats['num_queries']:,}",
                    size
                )
            except Exception:
                table.add_row(str(i), dataset, status, "N/A", "N/A", "N/A")
        else:
            if show_stats:
                table.add_row(str(i), dataset, status, "-", "-", "-")
            else:
                table.add_row(str(i), dataset, status)
    
    console.print(table)
    
    # Show helpful commands
    rprint("\n[blue]💡 Helpful commands:[/blue]")
    rprint("[cyan]vectormerge download-dataset --dataset <name>[/cyan] - Download specific dataset")
    rprint("[cyan]vectormerge download-dataset --dataset all[/cyan] - Download all datasets")
    rprint("[cyan]vectormerge list-datasets --stats[/cyan] - Show dataset statistics")


@dataset_app.command("info", help="Show dataset information")
def dataset_info(
    dataset: Optional[str] = typer.Option(None, "--dataset", "-d", help="Dataset name"),
    data_path: Path = typer.Option(cli_defaults['data_path'], "--data-path", help="Path to data directory"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive mode"),
):
    """Show detailed information about a dataset."""
    
    # Interactive mode
    if interactive:
        if not dataset:
            dataset = select_dataset_interactively()
    
    # Validate inputs
    if not dataset:
        display_error_and_exit("Please specify dataset name (or use --interactive)")
    
    if dataset not in SUPPORTED_DATASETS:
        display_error_and_exit(f"Dataset '{dataset}' not supported. Use 'vectormerge list-datasets' to see available datasets.")
    
    # Check if dataset exists
    dataset_path = data_path / dataset
    if not dataset_path.exists():
        display_error_and_exit(f"Dataset '{dataset}' not found. Download it first with 'vectormerge download-dataset --dataset {dataset}'")
    
    # Load dataset and show info
    try:
        from ..dataset import load_dataset
        dataset_obj = load_dataset(dataset, data_path=data_path)
        
        # Get statistics
        stats = dataset_obj.get_stats()
        
        # Create info table
        info_table = Table(title=f"Dataset Information: {dataset}")
        info_table.add_column("Property", style="cyan", no_wrap=True)
        info_table.add_column("Value", style="green")
        
        info_table.add_row("Name", dataset)
        info_table.add_row("Path", str(dataset_path))
        info_table.add_row("Documents", f"{stats['num_documents']:,}")
        info_table.add_row("Queries", f"{stats['num_queries']:,}")
        info_table.add_row("Qrels", f"{stats['num_qrels']:,}")
        
        # Calculate size
        size = format_size(sum(
            f.stat().st_size for f in dataset_path.rglob('*') if f.is_file()
        ))
        info_table.add_row("Size", size)
        
        console.print(info_table)
        
        # Show sample data
        corpus = dataset_obj.get_corpus()
        queries = dataset_obj.get_queries()
        
        if corpus and queries:
            rprint(f"\n[blue]📄 Sample Document:[/blue]")
            first_doc_id = next(iter(corpus))
            first_doc = corpus[first_doc_id]
            rprint(f"[dim]ID:[/dim] {first_doc_id}")
            rprint(f"[dim]Title:[/dim] {first_doc.get('title', 'N/A')}")
            rprint(f"[dim]Text:[/dim] {first_doc.get('text', 'N/A')[:200]}...")
            
            rprint(f"\n[blue]❓ Sample Query:[/blue]")
            first_query_id = next(iter(queries))
            first_query = queries[first_query_id]
            rprint(f"[dim]ID:[/dim] {first_query_id}")
            rprint(f"[dim]Query:[/dim] {first_query}")
        
    except Exception as e:
        display_error_and_exit(f"Error loading dataset: {e}")


def _show_download_results(success_count: int, failed_datasets: List[str], data_path: Path) -> None:
    """Show download results."""
    
    if success_count > 0:
        rprint(f"\n[green]✅ Successfully downloaded {success_count} dataset(s)![/green]")
    
    if failed_datasets:
        rprint(f"\n[red]❌ Failed to download {len(failed_datasets)} dataset(s):[/red]")
        for dataset in failed_datasets:
            rprint(f"  - {dataset}")
    
    # Show helpful next steps
    success_text = Text()
    success_text.append("💡 What's next?\n\n", style="bold yellow")
    success_text.append("📊 Check datasets: ", style="dim")
    success_text.append("vectormerge list-datasets --stats\n", style="cyan")
    success_text.append("🔍 Create references: ", style="dim")
    success_text.append("vectormerge create-reference\n", style="cyan")
    success_text.append("🎯 Generate embeddings: ", style="dim")
    success_text.append("vectormerge generate-embedding", style="cyan")
    
    panel = Panel(
        success_text,
        title="[bold green]📥 Download Complete![/bold green]",
        title_align="left",
        border_style="green",
        padding=(1, 2),
        expand=False
    )
    
    console.print(panel)


# Add aliases for backward compatibility
dataset_app.command("dl", help="Alias for download")(download_dataset)
dataset_app.command("ls", help="Alias for list")(list_datasets) 