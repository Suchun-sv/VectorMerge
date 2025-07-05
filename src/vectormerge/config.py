"""
Configuration management for VectorMerge.

This module provides configuration classes and utilities for managing
library settings, model configurations, and evaluation parameters.
"""

import os
import yaml
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for embedding models."""
    
    name: str
    batch_size: int = 32
    max_seq_length: int = 512
    cache_dir: Optional[str] = None
    device: str = "auto"  # "auto", "cpu", "cuda"
    
    # Model-specific parameters
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # API keys for external services
    api_key: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        
        if self.cache_dir is None:
            self.cache_dir = os.path.expanduser("~/.vectormerge/models")
        
        # Ensure cache directory exists
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)


@dataclass 
class DatasetConfig:
    """Configuration for datasets."""
    
    name: str
    path: Optional[str] = None
    split: str = "test"
    cache_dir: Optional[str] = None
    download: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.cache_dir is None:
            self.cache_dir = os.path.expanduser("~/.vectormerge/datasets")
        
        # Ensure cache directory exists
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class MappingConfig:
    """Configuration for vector space mapping."""
    
    method: str = "procrustes"  # "procrustes", "nonlinear", "ours"
    num_clusters: int = 10
    reduced_dim: int = -1  # -1 means no reduction
    cluster_method: str = "kmeans"  # "kmeans", "hierarchical", "ours"
    
    # Reference creation parameters
    reference_creation_method: str = "ours"  # "ours", "random"
    d0_ratio: float = 0.33
    split_strategy: str = "closest_to_p"
    
    # Auto-dimensionality parameters
    auto_dim_cluster: bool = False
    tau_1: float = 0.8
    tau_2: float = 0.1
    
    # Optimization parameters
    approximate: bool = False
    max_iter: int = 100
    tolerance: float = 1e-6


@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics."""
    
    k_list: List[int] = field(default_factory=lambda: [10, 100, 1000])
    metrics: List[str] = field(default_factory=lambda: ["recall", "ndcg", "map"])
    
    # Extended evaluation parameters
    cal_self_metric: bool = True
    cal_extended_recall: bool = True
    cal_rank_recall: bool = True
    
    # Output configuration
    save_results: bool = True
    results_dir: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.results_dir is None:
            self.results_dir = os.path.expanduser("~/.vectormerge/results")
        
        # Ensure results directory exists
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    
    # File logging
    log_to_file: bool = False
    log_file: Optional[str] = None
    
    # Wandb logging
    use_wandb: bool = False
    wandb_project: str = "vectormerge"
    wandb_entity: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.log_to_file and self.log_file is None:
            self.log_file = os.path.expanduser("~/.vectormerge/logs/vectormerge.log")
        
        if self.log_file:
            # Ensure log directory exists
            Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)


@dataclass
class VectorMergeConfig:
    """Main configuration class for VectorMerge."""
    
    # Sub-configurations
    model_1: ModelConfig
    model_2: ModelConfig
    dataset: DatasetConfig
    mapping: MappingConfig = field(default_factory=MappingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Global settings
    seed: int = 42
    cache_dir: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.cache_dir is None:
            self.cache_dir = os.path.expanduser("~/.vectormerge/cache")
        
        # Ensure cache directory exists
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "VectorMergeConfig":
        """Create configuration from dictionary."""
        # Extract sub-configurations
        model_1_config = ModelConfig(**config_dict.get("model_1", {}))
        model_2_config = ModelConfig(**config_dict.get("model_2", {}))
        dataset_config = DatasetConfig(**config_dict.get("dataset", {}))
        
        # Optional sub-configurations
        mapping_config = MappingConfig(**config_dict.get("mapping", {}))
        evaluation_config = EvaluationConfig(**config_dict.get("evaluation", {}))
        logging_config = LoggingConfig(**config_dict.get("logging", {}))
        
        return cls(
            model_1=model_1_config,
            model_2=model_2_config,
            dataset=dataset_config,
            mapping=mapping_config,
            evaluation=evaluation_config,
            logging=logging_config,
            **{k: v for k, v in config_dict.items() 
               if k not in ["model_1", "model_2", "dataset", "mapping", "evaluation", "logging"]}
        )
    
    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "VectorMergeConfig":
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    def to_yaml(self, output_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dictionary for serialization
        config_dict = {
            "model_1": {
                "name": self.model_1.name,
                "batch_size": self.model_1.batch_size,
                "max_seq_length": self.model_1.max_seq_length,
                "cache_dir": self.model_1.cache_dir,
                "device": self.model_1.device,
                "model_kwargs": self.model_1.model_kwargs,
                "api_key": self.model_1.api_key,
            },
            "model_2": {
                "name": self.model_2.name,
                "batch_size": self.model_2.batch_size,
                "max_seq_length": self.model_2.max_seq_length,
                "cache_dir": self.model_2.cache_dir,
                "device": self.model_2.device,
                "model_kwargs": self.model_2.model_kwargs,
                "api_key": self.model_2.api_key,
            },
            "dataset": {
                "name": self.dataset.name,
                "path": self.dataset.path,
                "split": self.dataset.split,
                "cache_dir": self.dataset.cache_dir,
                "download": self.dataset.download,
            },
            "mapping": {
                "method": self.mapping.method,
                "num_clusters": self.mapping.num_clusters,
                "reduced_dim": self.mapping.reduced_dim,
                "cluster_method": self.mapping.cluster_method,
                "reference_creation_method": self.mapping.reference_creation_method,
                "d0_ratio": self.mapping.d0_ratio,
                "split_strategy": self.mapping.split_strategy,
                "auto_dim_cluster": self.mapping.auto_dim_cluster,
                "tau_1": self.mapping.tau_1,
                "tau_2": self.mapping.tau_2,
                "approximate": self.mapping.approximate,
                "max_iter": self.mapping.max_iter,
                "tolerance": self.mapping.tolerance,
            },
            "evaluation": {
                "k_list": self.evaluation.k_list,
                "metrics": self.evaluation.metrics,
                "cal_self_metric": self.evaluation.cal_self_metric,
                "cal_extended_recall": self.evaluation.cal_extended_recall,
                "cal_rank_recall": self.evaluation.cal_rank_recall,
                "save_results": self.evaluation.save_results,
                "results_dir": self.evaluation.results_dir,
            },
            "logging": {
                "level": self.logging.level,
                "format": self.logging.format,
                "date_format": self.logging.date_format,
                "log_to_file": self.logging.log_to_file,
                "log_file": self.logging.log_file,
                "use_wandb": self.logging.use_wandb,
                "wandb_project": self.logging.wandb_project,
                "wandb_entity": self.logging.wandb_entity,
            },
            "seed": self.seed,
            "cache_dir": self.cache_dir,
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Configuration saved to {output_path}")


def get_default_config() -> VectorMergeConfig:
    """Get default configuration."""
    return VectorMergeConfig(
        model_1=ModelConfig(name="bert-base-uncased"),
        model_2=ModelConfig(name="roberta-base"),
        dataset=DatasetConfig(name="scifact"),
    )


def setup_logging(config: LoggingConfig) -> None:
    """Setup logging configuration."""
    # Configure basic logging
    logging.basicConfig(
        level=getattr(logging, config.level.upper()),
        format=config.format,
        datefmt=config.date_format,
    )
    
    # Add file handler if specified
    if config.log_to_file and config.log_file:
        file_handler = logging.FileHandler(config.log_file)
        file_handler.setFormatter(logging.Formatter(config.format, config.date_format))
        logging.getLogger().addHandler(file_handler)
    
    # Setup wandb if enabled
    if config.use_wandb:
        try:
            import wandb
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
            )
        except ImportError:
            logger.warning("Wandb not installed. Skipping wandb setup.")
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}") 