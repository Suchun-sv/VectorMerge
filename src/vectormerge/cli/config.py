"""
Configuration management commands for VectorMerge CLI.

This module contains commands for managing VectorMerge configuration.
"""

from pathlib import Path
from typing import Optional
import typer
from rich import print as rprint
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from .base import console
from .utils import confirm_action, display_error_and_exit, display_success

# Initialize config command group
config_app = typer.Typer(help="Manage VectorMerge configuration")


@config_app.command("show", help="Show current configuration")
def show_config(
    scope: str = typer.Option("all", "--scope", help="Configuration scope (all, global, local, package)"),
    format_: str = typer.Option("table", "--format", help="Output format (table, yaml, json)"),
):
    """Show current configuration."""
    
    try:
        from ..config_loader import VectorMergeConfigLoader
        config_loader = VectorMergeConfigLoader()
        
        if scope == "all":
            # Show merged configuration
            config = config_loader.load_config()
            rprint("[blue]📋 Current Configuration (Merged)[/blue]")
            _display_config(config, format_)
            
            # Show source information
            rprint("\n[dim]Configuration Sources:[/dim]")
            rprint(f"[dim]  Package config: {config_loader.package_config_path}[/dim]")
            rprint(f"[dim]  Global config: {config_loader.global_config_path}[/dim]")
            rprint(f"[dim]  Local config: {config_loader.local_config_path}[/dim]")
            
        elif scope == "global":
            if config_loader.global_config_path.exists():
                config = config_loader._load_yaml_config(config_loader.global_config_path)
                rprint("[blue]📋 Global Configuration[/blue]")
                _display_config(config, format_)
            else:
                rprint("[yellow]No global configuration found[/yellow]")
                
        elif scope == "local":
            if config_loader.local_config_path.exists():
                config = config_loader._load_yaml_config(config_loader.local_config_path)
                rprint("[blue]📋 Local Configuration[/blue]")
                _display_config(config, format_)
            else:
                rprint("[yellow]No local configuration found[/yellow]")
                
        elif scope == "package":
            if config_loader.package_config_path.exists():
                config = config_loader._load_yaml_config(config_loader.package_config_path)
                rprint("[blue]📋 Package Configuration[/blue]")
                _display_config(config, format_)
            else:
                rprint("[yellow]No package configuration found[/yellow]")
                
        else:
            display_error_and_exit(f"Invalid scope: {scope}. Use 'all', 'global', 'local', or 'package'.")
            
    except Exception as e:
        display_error_and_exit(f"Error loading configuration: {e}")


@config_app.command("create", help="Create configuration file")
def create_config(
    scope: str = typer.Option("local", "--scope", help="Configuration scope (global, local)"),
    force: bool = typer.Option(False, "--force", help="Overwrite existing configuration"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive configuration"),
):
    """Create a new configuration file."""
    
    try:
        from ..config_loader import VectorMergeConfigLoader
        config_loader = VectorMergeConfigLoader()
        
        if scope == "global":
            config_path = config_loader.global_config_path
            config_name = "Global"
        elif scope == "local":
            config_path = config_loader.local_config_path
            config_name = "Local"
        else:
            display_error_and_exit(f"Invalid scope: {scope}. Use 'global' or 'local'.")
        
        # Check if config already exists
        if config_path.exists() and not force:
            rprint(f"[yellow]Configuration already exists: {config_path}[/yellow]")
            if not confirm_action("Overwrite existing configuration?"):
                rprint("[yellow]Configuration creation cancelled[/yellow]")
                return
        
        # Create directory if needed
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load current config for defaults
        current_config = config_loader.load_config()
        
        if interactive:
            # Interactive configuration
            new_config = _interactive_config_creation(current_config)
        else:
            # Create with current defaults
            new_config = {
                'default_data_path': str(current_config.default_data_path),
                'default_embedding_path': str(current_config.default_embedding_path),
                'default_reference_path': str(current_config.default_reference_path),
                'default_type': current_config.default_type,
                'default_verbose': current_config.default_verbose,
                'default_force_download': current_config.default_force_download,
            }
        
        # Write configuration
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(new_config, f, default_flow_style=False, sort_keys=False)
        
        rprint(f"[green]✅ {config_name} configuration created: {config_path}[/green]")
        
        # Show the created configuration
        rprint(f"\n[blue]📋 Created Configuration:[/blue]")
        _display_config(new_config, "table")
        
    except Exception as e:
        display_error_and_exit(f"Error creating configuration: {e}")


@config_app.command("edit", help="Edit configuration file")
def edit_config(
    scope: str = typer.Option("local", "--scope", help="Configuration scope (global, local)"),
):
    """Edit configuration file in default editor."""
    
    try:
        from .config_loader import VectorMergeConfigLoader
        config_loader = VectorMergeConfigLoader()
        
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
            if confirm_action(f"{config_name} configuration does not exist. Create it?"):
                create_config(scope=scope, force=False, interactive=True)
                return
            else:
                return
        
        # Open in editor
        import os
        editor = os.environ.get('EDITOR', 'nano')
        os.system(f"{editor} {config_path}")
        
        rprint(f"[green]✅ {config_name} configuration edited[/green]")
        
    except Exception as e:
        display_error_and_exit(f"Error editing configuration: {e}")


@config_app.command("reset", help="Reset configuration to defaults")
def reset_config(
    scope: str = typer.Option("local", "--scope", help="Configuration scope (global, local)"),
    force: bool = typer.Option(False, "--force", help="Force reset without confirmation"),
):
    """Reset configuration to defaults."""
    
    try:
        from .config_loader import VectorMergeConfigLoader
        config_loader = VectorMergeConfigLoader()
        
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


# Add aliases for backward compatibility
# config_app.command("cfg", help="Alias for show")(show_config) 