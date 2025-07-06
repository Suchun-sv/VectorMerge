from typing import List, Optional, Dict, Any
import json
import yaml
import shutil

import typer
from rich.console import Console
from rich.prompt import Prompt

from rich import print as rprint
from rich.syntax import Syntax
from . import SUPPORTED_DATASETS, SUPPORTED_MODELS

def select_dataset_interactively() -> str:
    rprint("[bold blue]📥 Dataset Download[/bold blue]")
    rprint("\nAvailable datasets:")
    
    # Show dataset options with descriptions
    dataset_info = {
        "scifact": "Scientific claim verification",
        "nfcorpus": "Nutrition facts corpus", 
        "nq": "Natural Questions",
        "cqadupstack": "Community question answering",
        "arguana": "Argument mining", 
        "scidocs": "Scientific document classification",
        "fiqa": "Financial question answering",
    }
    
    for i, ds in enumerate(SUPPORTED_DATASETS, 1):
        description = dataset_info.get(ds, "")
        rprint(f"  [bold cyan]{i}.[/bold cyan] [bold green]{ds}[/bold green] - {description}")
    
    # Create choices list with both numbers and names
    choices = []
    for i, ds in enumerate(SUPPORTED_DATASETS, 1):
        choices.extend([str(i), ds])
    
    selection = Prompt.ask(
        "\nSelect dataset",
        choices=choices,
        default="1"
    )
    
    # Convert number selection to dataset name
    if selection.isdigit():
        idx = int(selection) - 1
        if 0 <= idx < len(SUPPORTED_DATASETS):
            dataset = SUPPORTED_DATASETS[idx]
        else:
            rprint(f"[red]Error:[/red] Invalid selection: {selection}")
            raise typer.Exit(code=1)
    else:
        dataset = selection
        
    return dataset

def select_model_interactively() -> str:
    rprint("[bold blue]📥 Model Selection[/bold blue]")
    rprint("\nAvailable models:")

    model_info = {
        "openai": "OpenAI-Ada [API]",
        "mistral": "Mistral [API]",
        "glove": "GloVe [Traditional]",
        "fast-text": "FastText [Traditional]",
        "gte": "GTE-Qwen2 [Self-hosted]",
        "nv-embed": "NV-Embed-V2 [Self-hosted]",
    }
    
    # Show model options with descriptions
    for i, model in enumerate(SUPPORTED_MODELS, 1):
        rprint(f"  [bold cyan]{i}.[/bold cyan] [bold green]{model}[/bold green] - {model_info[model]}")
    
    # Create choices list with both numbers and names
    choices = []
    for i, model in enumerate(SUPPORTED_MODELS, 1):
        choices.extend([str(i), model])
    
    selection = Prompt.ask(
        "\nSelect model",
        choices=choices,
        default="1"
    )

    if selection.isdigit():
        idx = int(selection) - 1
        if 0 <= idx < len(SUPPORTED_MODELS):
            model = SUPPORTED_MODELS[idx]
        else:
            rprint(f"[red]Error:[/red] Invalid selection: {selection}")
            raise typer.Exit(code=1)
    else:
        model = selection
    
    return model