"""
Reference creation commands for VectorMerge CLI.

This module contains commands for creating and managing reference datasets.
"""

from pathlib import Path
from typing import Optional
import typer
from rich import print as rprint
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text

from .base import cli_defaults, console, set_seed, SUPPORTED_DATASETS
from .utils import (
    select_dataset_interactively, validate_model_and_dataset,
    display_error_and_exit, display_success
)
from ..reference import RandomSplit, LA2MSplit

# Initialize reference command group
reference_app = typer.Typer(help="Create and manage reference datasets")


@reference_app.command("create", help="Create reference dataset splits")
def create_reference(
    dataset: str = typer.Option(None, "--dataset", "-d", help="Dataset name"),
    data_path: Path = typer.Option(cli_defaults['data_path'], "--data-path", help="Path to raw data directory"),
    reference_path: Path = typer.Option(cli_defaults['reference_path'], "--reference-path", help="Path to save reference files"),
    strategy: str = typer.Option("random", "--strategy", help="Split strategy (random, la2m)"),
    reference_ratio: float = typer.Option(0.5, "--reference-ratio", help="Ratio of reference data (D0)"),
    remove_duplicates: bool = typer.Option(False, "--remove-duplicates", help="Remove duplicate answers"),
    select_top_1: bool = typer.Option(True, "--select-top-1/--select-all", help="Select top-1 or all relevant answers"),
    force: bool = typer.Option(False, "--force", help="Force regeneration"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive mode"),
    verbose: bool = typer.Option(cli_defaults['verbose'], "--verbose", "-v", help="Verbose output"),
):
    """Create reference dataset splits for mapping."""
    
    # Set random seed
    set_seed()
    
    # Interactive mode
    if interactive:
        if not dataset:
            dataset = select_dataset_interactively()
        
        # Ask for strategy
        if not strategy:
            strategy_options = ["random", "la2m"]
            rprint("[cyan]Available split strategies:[/cyan]")
            for i, s in enumerate(strategy_options, 1):
                rprint(f"  {i}. {s}")
            
            while True:
                try:
                    choice = typer.prompt("Select strategy (index or name)", default="random")
                    if choice.isdigit():
                        idx = int(choice) - 1
                        if 0 <= idx < len(strategy_options):
                            strategy = strategy_options[idx]
                            break
                    elif choice.lower() in strategy_options:
                        strategy = choice.lower()
                        break
                    else:
                        rprint(f"[red]Invalid choice: {choice}[/red]")
                except typer.Abort:
                    raise typer.Exit(code=1)
    
    # Validate inputs
    if not dataset:
        display_error_and_exit("Please specify dataset name (or use --interactive)")
    
    if dataset not in SUPPORTED_DATASETS:
        display_error_and_exit(f"Dataset '{dataset}' not supported. Use 'vectormerge list-datasets' to see available datasets.")
    
    # Check if reference already exists
    reference_file = reference_path / f"{dataset}_reference.npz"
    if reference_file.exists() and not force:
        rprint(f"[yellow]Reference already exists:[/yellow] {reference_file}")
        rprint("[yellow]Use --force to regenerate[/yellow]")
        return
    
    # Create reference path
    reference_path.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    rprint(f"[blue]📊 Creating reference for dataset: {dataset}[/blue]")
    rprint(f"[blue]Strategy:[/blue] {strategy}")
    rprint(f"[blue]Reference ratio:[/blue] {reference_ratio}")
    rprint(f"[blue]Output path:[/blue] {reference_file}")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading dataset...", total=None)
        
        try:
            from ..dataset import load_dataset
            dataset_obj = load_dataset(dataset, data_path=data_path)
            
            progress.update(task, description="Creating reference split...")
            
            # Create reference split based on strategy
            if strategy == "random":
                from ..reference import RandomSplit
                splitter = RandomSplit(
                    dataset_name=dataset,
                    internal_index=dataset_obj.get_internal_index(),
                    reference_ratio=reference_ratio,
                    reference_path=str(reference_path)
                )
                d0_index, d1_index, d2_index = splitter.split()
            
            elif strategy == "la2m":
                from ..reference import LA2MSplit
                splitter = LA2MSplit.from_qrels(
                    dataset_name=dataset,
                    dataset_index=dataset_obj.get_dataset_index(),
                    internal_index=dataset_obj.get_internal_index(),
                    dataset_obj=dataset_obj,
                    qrels=dataset_obj.qrels,
                    reference_ratio=reference_ratio,
                    reference_path=str(reference_path),
                    remove_dup_answer=remove_duplicates,
                    select_top_1=select_top_1
                )
                d0_index, d1_index, d2_index = splitter.split()
            
            else:
                display_error_and_exit(f"Unknown strategy: {strategy}")
            
            # Create result dictionary
            result = {
                'd0_size': len(d0_index),
                'd1_size': len(d1_index),
                'd2_size': len(d2_index),
                'dataset_obj': dataset_obj,
                'd0_indices': d0_index.tolist(),
                'd1_indices': d1_index.tolist(),
                'd2_indices': d2_index.tolist(),
            }
            
            # Add strategy-specific constraints info
            if strategy == "la2m":
                result['la2m_constraints'] = {
                    'answers_excluded_from_d0': True,
                    'answers_evenly_distributed': True,
                    'duplicate_answers': 'removed' if remove_duplicates else 'kept'
                }
            
            progress.update(task, description="Reference split created!")
            
        except Exception as e:
            rprint(f"[red]Error creating reference:[/red] {e}")
            import traceback
            if verbose:
                rprint(f"[red]Traceback:[/red] {traceback.format_exc()}")
            raise typer.Exit(code=1)
    
    # Display results
    _show_reference_results(dataset, strategy, reference_file, result)


def _show_reference_results(dataset: str, strategy: str, reference_file: Path, result: dict) -> None:
    """Show reference creation results."""
    
    # Create results table
    results_table = Table(title=f"Reference Creation Results: {dataset}")
    results_table.add_column("Split", style="cyan", no_wrap=True)
    results_table.add_column("Size", style="green", justify="right")
    results_table.add_column("Ratio", style="yellow", justify="right")
    
    total_size = result['d0_size'] + result['d1_size'] + result['d2_size']
    
    results_table.add_row(
        "D0 (Reference)", 
        str(result['d0_size']), 
        f"{result['d0_size']/total_size*100:.1f}%"
    )
    results_table.add_row(
        "D1 (Test 1)", 
        str(result['d1_size']), 
        f"{result['d1_size']/total_size*100:.1f}%"
    )
    results_table.add_row(
        "D2 (Test 2)", 
        str(result['d2_size']), 
        f"{result['d2_size']/total_size*100:.1f}%"
    )
    results_table.add_row(
        "Total", 
        str(total_size), 
        "100.0%",
        style="bold"
    )
    
    console.print(results_table)
    
    # Show top 10 samples from each split
    if 'dataset_obj' in result:
        _show_top10_samples(result['dataset_obj'], result)
    
    # Show strategy-specific info
    if strategy == "la2m" and 'la2m_constraints' in result:
        la2m_info = result['la2m_constraints']
        rprint(f"\n[blue]📋 LA2M Constraints:[/blue]")
        rprint(f"[green]✓ Answer documents excluded from D0:[/green] {la2m_info['answers_excluded_from_d0']}")
        rprint(f"[green]✓ Answer documents evenly distributed:[/green] {la2m_info['answers_evenly_distributed']}")
        if 'duplicate_answers' in la2m_info:
            rprint(f"[yellow]⚠️  Duplicate answers handled:[/yellow] {la2m_info['duplicate_answers']}")
    
    # Success message
    success_text = Text()
    success_text.append("🎉 Reference dataset created successfully!\n\n", style="bold green")
    success_text.append(f"📁 File: {reference_file}\n", style="")
    success_text.append(f"🔧 Strategy: {strategy}\n", style="")
    success_text.append(f"📊 Total documents: {total_size}\n\n", style="")
    success_text.append("💡 What's next?\n", style="bold yellow")
    success_text.append("  🎯 Generate embeddings: ", style="dim")
    success_text.append("vectormerge generate-embedding\n", style="cyan")
    success_text.append("  🗺️ Create mappings: ", style="dim")
    success_text.append("vectormerge map-embedding\n", style="cyan")
    success_text.append("  📈 Run evaluation: ", style="dim")
    success_text.append("vectormerge evaluate", style="cyan")
    
    panel = Panel(
        success_text,
        title="[bold green]✅ Success![/bold green]",
        title_align="left",
        border_style="green",
        padding=(1, 2),
        expand=False
    )
    
    console.print(panel)


def _show_top10_samples(dataset_obj, result: dict) -> None:
    """Show top 10 samples from each split."""
    
    # Load the split indices
    d0_indices = result.get('d0_indices', [])
    d1_indices = result.get('d1_indices', [])
    d2_indices = result.get('d2_indices', [])
    
    def _create_sample_table(title: str, indices: list, color: str, max_samples: int = 10) -> None:
        """Create a sample table for a split."""
        if not indices:
            return
            
        # Get sample indices (first 10 or less)
        sample_indices = indices[:max_samples]
        
        # Create table
        table = Table(title=f"{title} - Top {len(sample_indices)} Samples", 
                     title_style=f"bold {color}", 
                     show_header=True, 
                     header_style="bold white")
        table.add_column("Index", style="dim", width=8)
        table.add_column("Internal ID", style="magenta", width=12)
        table.add_column("Doc ID", style="cyan", width=12)
        table.add_column("Title", style="green", width=30)
        table.add_column("Content Preview", style="white", width=50)
        
        for i, idx in enumerate(sample_indices):
            # Get document ID (convert from internal index if needed)
            if isinstance(idx, int):
                internal_idx = idx
                doc_id = dataset_obj.internal_index_to_original_id(idx)
            else:
                internal_idx = "N/A"
                doc_id = str(idx)
            
            # Get document content
            doc = dataset_obj.corpus.get(doc_id, {})
            title = doc.get('title', 'No title')[:28] + '...' if len(doc.get('title', '')) > 28 else doc.get('title', 'No title')
            content = doc.get('text', 'No content')[:47] + '...' if len(doc.get('text', '')) > 47 else doc.get('text', 'No content')
            
            table.add_row(
                str(i + 1),
                str(internal_idx),
                doc_id[:10] + '...' if len(doc_id) > 10 else doc_id,
                title,
                content
            )
        
        console.print(table)
        console.print()  # Add spacing
    
    # Show samples for each split
    rprint(f"\n[bold blue]📋 Sample Documents from Each Split[/bold blue]")
    
    _create_sample_table("🎯 D0 (Reference Set)", d0_indices, "blue")
    _create_sample_table("🧪 D1 (Test Set 1)", d1_indices, "green") 
    _create_sample_table("🧪 D2 (Test Set 2)", d2_indices, "yellow")


# Add aliases for backward compatibility
# reference_app.command("cr", help="Short alias for create")(create_reference) 