"""
Configuration loader for VectorMerge.

This module provides functionality to load configuration from multiple sources
with proper precedence: CLI args > project config > global config > defaults.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field

@dataclass
class VectorMergeConfig:
    """Configuration container for VectorMerge."""
    
    # Path settings
    data_path: str = "./data/raw/beir/"
    embedding_path: str = "./data/processed/embeddings/"
    reference_path: str = "./output/references/"
    mapping_path: str = "./output/mappings/"
    
    # Model settings
    model_1: str = "bert-base-uncased"
    model_2: str = "roberta-base"
    dataset: str = "scifact"
    
    # Algorithm settings
    default_mapping_method: str = "procrustes"
    default_reference_strategy: str = "ours"
    default_num_clusters: int = 50
    default_d0_ratio: float = 0.33
    
    # Evaluation settings
    default_k_list: list = field(default_factory=lambda: [10, 100, 1000])
    default_metrics: list = field(default_factory=lambda: ["recall", "ndcg", "map"])
    
    # Advanced settings
    verbose: bool = False
    use_cache: bool = True
    force_download: bool = False
    
    # CLI defaults
    default_model: str = "mistral"
    default_dataset: str = "scifact"
    default_type: str = "all"
    default_batch_size: int = 32
    default_device: str = "auto"
    
    # Embedding models configuration
    embedding_models: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class ConfigLoader:
    """Loads configuration from multiple sources with proper precedence."""
    
    def __init__(self):
        self.config = VectorMergeConfig()
        self._load_configs()
    
    def _load_configs(self):
        """Load configurations in order of precedence."""
        # 1. Load built-in defaults (already in VectorMergeConfig)
        
        # 2. Load from package config
        package_config = self._load_package_config()
        if package_config:
            self._merge_config(package_config)
        
        # 3. Load from global config
        global_config = self._load_global_config()
        if global_config:
            self._merge_config(global_config)
        
        # 4. Load from project config
        project_config = self._load_project_config()
        if project_config:
            self._merge_config(project_config)
    
    def _load_package_config(self) -> Optional[Dict[str, Any]]:
        """Load config from package's configs/config.yaml."""
        try:
            # Get the package directory
            package_dir = Path(__file__).parent.parent.parent
            config_path = package_dir / "configs" / "config.yaml"
            
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load package config: {e}")
        return None
    
    def _load_global_config(self) -> Optional[Dict[str, Any]]:
        """Load config from ~/.vectormerge/config.yaml."""
        try:
            global_config_path = Path.home() / ".vectormerge" / "config.yaml"
            if global_config_path.exists():
                with open(global_config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load global config: {e}")
        return None
    
    def _load_project_config(self) -> Optional[Dict[str, Any]]:
        """Load config from ./.vectormerge/config.yaml."""
        try:
            project_config_path = Path.cwd() / ".vectormerge" / "config.yaml"
            if project_config_path.exists():
                with open(project_config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load project config: {e}")
        return None
    
    def _merge_config(self, config_dict: Dict[str, Any]):
        """Merge configuration dictionary into current config."""
        for key, value in config_dict.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            elif key == 'embedding_models' and hasattr(self.config, 'embedding_models'):
                # Special handling for embedding_models
                self.config.embedding_models.update(value)
    
    def get_model_settings(self, model_name: str) -> Dict[str, Any]:
        """Get settings for a specific model."""
        return self.config.embedding_models.get(model_name, {})
    
    def get_default(self, key: str, fallback: Any = None) -> Any:
        """Get a default value from config."""
        return getattr(self.config, key, fallback)
    
    def get_cli_defaults(self) -> Dict[str, Any]:
        """Get CLI command defaults."""
        return {
            'model': self.config.default_model,
            'dataset': self.config.default_dataset,
            'type': self.config.default_type,
            'batch_size': self.config.default_batch_size,
            'device': self.config.default_device,
            'data_path': self.config.data_path,
            'embedding_path': self.config.embedding_path,
            'verbose': self.config.verbose,
            'use_cache': self.config.use_cache,
            'force_download': self.config.force_download,
        }


# Global config loader instance
_config_loader = None

def get_config() -> ConfigLoader:
    """Get the global config loader instance."""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
    return _config_loader

def reload_config():
    """Reload configuration from all sources."""
    global _config_loader
    _config_loader = ConfigLoader()
    return _config_loader 