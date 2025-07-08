"""
Configuration management commands for VectorMerge CLI.

This module contains commands for managing VectorMerge configuration.
"""

from pathlib import Path
from typing import Optional, Any
import click
import typer
from rich import print as rprint
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from .base import console
from .utils import confirm_action, display_error_and_exit, display_success, display_warning, display_info

# Initialize config command group
config_app = typer.Typer(help="Manage VectorMerge configuration")


@config_app.command("check", help="Check current configuration")
def check_config():
    """Check current configuration."""

    _display_config_status()

    try:
        from .config_loader import ConfigLoader
        config_loader = ConfigLoader()

        type_map = {
            "all": None,
            "global": config_loader.global_config_path,
            "local": config_loader.local_config_path,
            "package": config_loader.package_config_path,
        }

        # Deciding type_map[all]:
        if Path(config_loader.local_config_path).exists():
            type_map["all"] = config_loader.local_config_path
        elif Path(config_loader.global_config_path).exists():
            type_map["all"] = config_loader.global_config_path
        elif Path(config_loader.package_config_path).exists():
            type_map["all"] = config_loader.package_config_path
        else:
            display_warning(f"No configuration found in {config_loader.local_config_path}, {config_loader.global_config_path}, {config_loader.package_config_path}")
            return

        # config = config_loader.load_yaml_config(type_map["all"])
        config = config_loader.load_yaml_config(type_map["all"]).config
        rprint(f"[blue]📋 Current Configuration ({type_map['all']})[/blue]")
        rprint(f"[dim]Path: {type_map['all']}[/dim]")
        _display_config(config.to_dict(), "table")

    except Exception as e:
        display_error_and_exit(f"Error loading configuration: {e}")


@config_app.command("generate", help="Generate default configuration file")
def generate_default_config(
    scope: str = typer.Option("local", "--scope", help="Configuration scope (global, local)"),
    force: bool = typer.Option(False, "--force", help="Overwrite existing configuration"),
):
    """Create a new configuration file."""
    from .config_loader import ConfigLoader
    config_loader = ConfigLoader()

    config_yaml_str = config_loader.to_commented_yaml()
    if scope == "global":
        config_path = config_loader.global_config_path
    elif scope == "local":
        config_path = config_loader.local_config_path
    else:
        display_error_and_exit(f"Invalid scope: {scope}. Use 'global' or 'local'.")

    display_info(f"[dim]You are about to generate a default configuration file to [yellow]{config_path}[/yellow], change it through the --scope option.[/dim]")

    config_path.parent.mkdir(parents=True, exist_ok=True)

    if not force:
        if config_path.exists():
            display_warning(f"{config_path} already exists. Manually remove if or set --force to overwrite.")
            return
        else:
            config_path.write_text(config_yaml_str)
            rprint(f"[green]✅ {config_path} created[/green]")
    else:
        config_path.write_text(config_yaml_str)
        rprint(f"[green]✅ {config_path} created[/green]")



@config_app.command("reset", help="Reset configuration to defaults")
def reset_config(
    scope: str = typer.Option("local", "--scope", help="Configuration scope (global, local)"),
    force: bool = typer.Option(False, "--force", help="Force reset without confirmation"),
):
    """Reset configuration to defaults."""
    
    try:
        from .config_loader import ConfigLoader
        config_loader = ConfigLoader()
        
        if scope == "global":
            config_path = config_loader.global_config_path
            config_name = "Global"
        elif scope == "local":
            config_path = config_loader.local_config_path
            config_name = "Local"
        else:
            display_error_and_exit(f"Invalid scope: {scope}. Use 'global' or 'local'.")
        
        # Check if config exists
        if not config_path.exists():
            rprint(f"[yellow]No {config_name.lower()} configuration found[/yellow]")
            return
        
        # Confirm reset
        if not force:
            if not confirm_action(f"Reset {config_name.lower()} configuration to defaults?"):
                rprint("[yellow]Configuration reset cancelled[/yellow]")
                return
        
        # Delete the config file
        config_path.unlink()
        
        rprint(f"[green]✅ {config_name} configuration reset to defaults[/green]")
        
    except Exception as e:
        display_error_and_exit(f"Error resetting configuration: {e}")


def _display_config(config, format_: str) -> None:
    """Display configuration in specified format."""
    
    if hasattr(config, 'to_dict'):
        config_dict = config.to_dict()
    else:
        config_dict = config
    
    if format_ == "table":
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Setting", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        
        for key, value in config_dict.items():
            # Format key for display
            display_key = key.replace('default_', '').replace('_', ' ').title()
            table.add_row(display_key, str(value))
        
        console.print(table)
        
    elif format_ == "yaml":
        import yaml
        yaml_str = yaml.dump(config_dict, default_flow_style=False, sort_keys=False)
        rprint(f"[dim]{yaml_str}[/dim]")
        
    elif format_ == "json":
        import json
        json_str = json.dumps(config_dict, indent=2)
        rprint(f"[dim]{json_str}[/dim]")
        
    else:
        display_error_and_exit(f"Invalid format: {format_}. Use 'table', 'yaml', or 'json'.")


def _interactive_config_creation(current_config) -> dict:
    """Interactive configuration creation."""
    
    rprint("[blue]🔧 Interactive Configuration Setup[/blue]")
    rprint("[dim]Press Enter to keep current value[/dim]\n")
    
    # Data path
    data_path = typer.prompt(
        f"Data path [{current_config.default_data_path}]",
        default=str(current_config.default_data_path)
    )
    
    # Embedding path
    embedding_path = typer.prompt(
        f"Embedding path [{current_config.default_embedding_path}]",
        default=str(current_config.default_embedding_path)
    )
    
    # Reference path
    reference_path = typer.prompt(
        f"Reference path [{current_config.default_reference_path}]",
        default=str(current_config.default_reference_path)
    )
    
    # Type
    type_options = ["corpus", "query", "both"]
    rprint(f"\nAvailable types: {', '.join(type_options)}")
    type_ = typer.prompt(
        f"Default type [{current_config.default_type}]",
        default=current_config.default_type
    )
    
    # Verbose
    verbose = typer.confirm(
        f"Verbose output by default? [{current_config.default_verbose}]",
        default=current_config.default_verbose
    )
    
    # Force download
    force_download = typer.confirm(
        f"Force download by default? [{current_config.default_force_download}]",
        default=current_config.default_force_download
    )
    
    return {
        'default_data_path': data_path,
        'default_embedding_path': embedding_path,
        'default_reference_path': reference_path,
        'default_type': type_,
        'default_verbose': verbose,
        'default_force_download': force_download,
    }


def _display_config_status():
    from .config_loader import ConfigLoader
    config_loader = ConfigLoader()
    configs = [
        ("Local", config_loader.local_config_path),
        ("Package", config_loader.package_config_path),
        ("Global", config_loader.global_config_path),
    ]

    for name, path in configs:
        if Path(path).exists():
            rprint(f"[green]✅ {name} configuration found: {path}[/green]")
        else:
            rprint(f"[yellow]❌ {name} configuration not found: {path}[/yellow]")

def parse_dynamic_config(ctx: click.Context) -> dict[str, Any]:
    extra = ctx.args  # All unknown parameters are here
    dynamic: dict[str, Any] = {}
    for arg in extra:
        if arg.startswith("--") and "=" in arg:
            key_path, val = arg.lstrip("-").split("=", 1)
            ptr = dynamic
            parts = key_path.split(".")
            for p in parts[:-1]:
                ptr = ptr.setdefault(p, {})
            ptr[parts[-1]] = val
    return dynamic