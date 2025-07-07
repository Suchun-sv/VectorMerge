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
    reference_path: str = "./data/processed/references/"
    mapping_path: str = "./data/processed/mappings/"
    mapping_param_path: str = "./output/mapping_models/"
    mapping_embedding_path: str = "./output/mapping_embeddings/"
    cluster_path: str = "./data/processed/clusters/"
    
    # Model settings
    model_1: str = "bert-base-uncased"
    model_2: str = "roberta-base"
    dataset: str = "scifact"
    
    # Algorithm settings
    default_mapping_method: str = "procrustes"
    default_reference_strategy: str = "la2m"
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
        self._cached_configs = {}
        self._load_configs()
    
    def _load_configs(self):
        """Load configurations in order of precedence."""
        # Load all config sources
        configs = {
            'package': self._load_package_config(),
            'global': self._load_global_config(), 
            'project': self._load_project_config()
        }
        
        # Merge in order of precedence (package -> global -> project)
        for config_type, config_data in configs.items():
            if config_data:
                self._cached_configs[config_type] = config_data
                self._merge_config(config_data)
    
    def _load_config_file(self, config_path: Path) -> Optional[Dict[str, Any]]:
        """Generic config file loader."""
        try:
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
        return None
    
    def _load_package_config(self) -> Optional[Dict[str, Any]]:
        """Load config from package's configs/config.yaml."""
        package_dir = Path(__file__).parent.parent.parent
        return self._load_config_file(package_dir / "configs" / "config.yaml")
    
    def _load_global_config(self) -> Optional[Dict[str, Any]]:
        """Load config from ~/.vectormerge/config.yaml."""
        return self._load_config_file(Path.home() / ".vectormerge" / "config.yaml")
    
    def _load_project_config(self) -> Optional[Dict[str, Any]]:
        """Load config from ./.vectormerge/config.yaml."""
        return self._load_config_file(Path.cwd() / ".vectormerge" / "config.yaml")
    
    def _merge_config(self, config_dict: Dict[str, Any]):
        """Merge configuration dictionary into current config."""
        for key, value in config_dict.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            elif key == 'embedding_models' and hasattr(self.config, 'embedding_models'):
                self.config.embedding_models.update(value)
    
    def get_model_settings(self, model_name: str) -> Dict[str, Any]:
        """Get settings for a specific model."""
        return self.config.embedding_models.get(model_name, {})
    
    def get_default(self, key: str, fallback: Any = None) -> Any:
        """Get a default value from config."""
        return getattr(self.config, key, fallback)
    
    def get_cli_defaults(self) -> Dict[str, Any]:
        """Get CLI command defaults dynamically."""
        # Define mapping from config attributes to CLI defaults
        cli_mapping = {
            'model': 'default_model',
            'dataset': 'default_dataset', 
            'type': 'default_type',
            'batch_size': 'default_batch_size',
            'device': 'default_device',
            'data_path': 'data_path',
            'embedding_path': 'embedding_path',
            'reference_path': 'reference_path',
            'mapping_path': 'mapping_path',
            'mapping_param_path': 'mapping_param_path',
            'mapping_embedding_path': 'mapping_embedding_path',
            'verbose': 'verbose',
            'cluster_path': 'cluster_path',
            'use_cache': 'use_cache',
            'force_download': 'force_download'
        }
        
        return {
            cli_key: getattr(self.config, config_attr, None)
            for cli_key, config_attr in cli_mapping.items()
        }
    
    def get_mapper_defaults(self, mapper_type: str) -> Dict[str, Any]:
        """Get mapper-specific defaults from config."""
        # Default configurations for each mapper type
        default_configs = {
            'nonlinear': {
                'hidden_size': 512,
                'num_layers': 3,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 100,
                'dropout': 0.1,
                'loss_function': 'mse',
                'save_param': True,
                'save_embedding': True
            },
            'procrustes': {
                'use_gpu': False,
                'approximate': False,
                'save_transformed': True,
                'save_param': True,
                'save_embedding': True
            },
            'la2m': {
                'num_clusters': 50,
                'clustering_method': 'kmeans',
                'local_strategy': 'procrustes',
                'min_cluster_size': 10,
                'save_transformed': True,
                'save_param': True,
                'save_embedding': True
            }
        }
        
        # Get base defaults
        result = default_configs.get(mapper_type, {}).copy()
        
        # Override with config file values if available
        project_config = self._cached_configs.get('project', {})
        mapper_config = project_config.get('mapper', {})
        mapper_defaults = mapper_config.get(mapper_type, {})
        
        # Key mapping from config to CLI parameters
        key_mapping = {
            'hidden_dim': 'hidden_size',
            'num_epochs': 'epochs', 
            'loss_type': 'loss_function',
            'cluster_method': 'clustering_method'
        }
        
        # Apply config overrides
        for config_key, config_value in mapper_defaults.items():
            cli_key = key_mapping.get(config_key, config_key)
            result[cli_key] = config_value
        
        return result


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