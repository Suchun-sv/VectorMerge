"""
Base constants and utilities for VectorMerge CLI.

This module contains shared constants, configuration, and utility functions
used across all CLI commands.
"""

import json
import os
import wandb
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import print as rprint
import typer
from ..clustering import SUPPORTED_CLUSTERING_METHODS
from ..dataset import SUPPORTED_DATASETS
from ..embeddings import SUPPORTED_MODELS, get_embedding
from ..mapping import SUPPORTED_MAPPING_METHODS
from ..reference import get_reference
from ..mapping import VectorSpaceMapper, MappingConfig
from loguru import logger

# Initialize rich console
console = Console()

# Load CLI defaults
def load_cli_defaults() -> Dict[str, Any]:
    """Load CLI default configuration."""
    from .config_loader import ConfigLoader
    config_loader = ConfigLoader().load_config()
    return config_loader.config.to_dict()

cli_defaults = load_cli_defaults()


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    
    random.seed(seed)
    np.random.seed(seed)
    
    if not globals().get('_seed_logged', False):
        rprint(f"[dim]🎲 Random seed set to {seed} for reproducibility[/dim]")
        globals()['_seed_logged'] = True

def set_wandb(wandb_entity: Optional[str], wandb_project: Optional[str], config_dict: Optional[Dict[str, Any]] = None):
    """Set up Weights and Biases for logging."""
    if wandb_entity is None or wandb_project is None:
        console.log("[dim]WandB logging is disabled.[/dim]")
        wandb.init(project=wandb_project, mode="disabled")
    else:
        console.log(f"[bold green]WandB logging enabled for project '{wandb_project}'[/bold green]")
        wandb.init(entity=wandb_entity, project=wandb_project, config=config_dict)

