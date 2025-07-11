"""
Base classes and interfaces for vector space mapping in VectorMerge.

This module provides abstract base classes and data structures for different
mapping strategies and clustering approaches.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field, asdict
from dacite import from_dict
from pathlib import Path
import numpy as np
import torch
from loguru import logger

from vectormerge.mapping.utils.io import save_embeddings

@dataclass
class LA2MConfig():
    """Configuration for LA2M mapping strategy."""
    num_clusters: int = 50
    cluster_method: str = "la2m-cluster"
    d_prime: int = 10
    pca_mapping: bool = True
    pca_dim: int = 14
    use_norm: bool = False

@dataclass
class NonLinearMappingConfig():
    """Configuration for non-linear mapping strategy."""
    num_layers: int = 3
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 100
    hidden_dim: int = 512
    dropout_rate: float = 0.1
    loss_type: str = "mse"

@dataclass
class ProcrustesConfig():
    """Configuration for Procrustes mapping strategy."""
    approximate: bool = False
    q: int = 1500
    with_rotation: bool = True
    with_scaling: bool = True
    use_pca: bool = False
    reduced_dim: int = 0
    procrustes_pca_type: str = "none"
    use_norm: bool = True


@dataclass
class MappingConfig:
    """Configuration for mapping strategies."""
    
    # General settings
    device: str = "auto"
    batch_size: int = 32
    verbose: bool = False
    save_param: bool = True
    save_embedding: bool = True

    # Mapping strategy settings
    la2m_config: LA2MConfig = LA2MConfig()
    nonlinear_config: NonLinearMappingConfig = NonLinearMappingConfig()
    procrustes_config: ProcrustesConfig = ProcrustesConfig()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_string(self) -> str:
        """Convert to string."""
        return str(self.__dict__)
    
    @classmethod
    def from_update_dict(cls, update_dict: Dict[str, Any]) -> 'MappingConfig':
        """init and update config from dictionary."""
        config = cls()
        config._update_config(update_dict)
        return config
    
    def _update_config(self, update_dict: Dict[str, Any]) -> None:
        """Update config from dictionary."""
        for key, value in update_dict.items():
            setattr(self, key, value)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MappingConfig':
        """Create from dictionary."""
        return from_dict(cls, config_dict)


class MappingStrategy(ABC):
    """Abstract base class for embedding mapping strategies."""
    
    def __init__(self, config: MappingConfig):
        """Initialize mapping strategy.
        
        Args:
            config: Configuration object containing all necessary parameters
        """
        self.config = config
        self.device = self._get_device()
        self.is_fitted = False
        self.transformation_matrix: Optional[np.ndarray] = None
        self.metadata: Dict[str, Any] = {}
        
        logger.info(f"Initialized {self.__class__.__name__} with device: {self.device}")
    
    def _get_device(self) -> torch.device:
        """Get the appropriate device for computation."""
        if self.config.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.config.device)
    
    def __repr__(self) -> str:
        """Return a string representation of the mapping strategy."""
        return f"{self.__class__.__name__} is_fitted={self.is_fitted}"
    
    def fit(self, source_embeddings: np.ndarray, target_embeddings: np.ndarray,
            reference_indices: np.ndarray, **kwargs) -> None:
        """Fit the mapping from source to target embeddings using reference data.
        
        Args:
            source_embeddings: Source embedding space (model 1)
            target_embeddings: Target embedding space (model 2)
            reference_indices: Indices of reference points (D0)
            **kwargs: Additional arguments specific to each strategy
        """
        self.metadata['source_dimension'] = source_embeddings.shape[1]
        self.metadata['target_dimension'] = target_embeddings.shape[1]
        self.metadata['num_reference_points'] = len(reference_indices)
        self._fit(source_embeddings, target_embeddings, reference_indices, **kwargs)
        self.is_fitted = True

    @abstractmethod
    def _fit(self, source_embeddings: np.ndarray, target_embeddings: np.ndarray,
             reference_indices: np.ndarray, **kwargs) -> None:
        """Fit the mapping from source to target embeddings using reference data.
        
        Args:
            source_embeddings: Source embedding space (model 1)
            target_embeddings: Target embedding space (model 2) 
            reference_indices: Indices of reference points (D0)
            **kwargs: Additional arguments specific to each strategy
        """
        pass
    
    def transform(self, embeddings: np.ndarray, **kwargs) -> np.ndarray:
        """Transform embeddings using the learned mapping.
        
        Args:
            embeddings: Embeddings to transform
            **kwargs: Additional arguments specific to each strategy
            
        Returns:
            Transformed embeddings
        """
        return self._transform(embeddings, **kwargs)
    
    @abstractmethod
    def _transform(self, embeddings: np.ndarray, **kwargs) -> np.ndarray:
        """Transform embeddings using the learned mapping.
        
        Args:
            embeddings: Embeddings to transform
            **kwargs: Additional arguments specific to each strategy
        """
        pass
    
    def fit_transform(self, source_embeddings: np.ndarray, target_embeddings: np.ndarray,
                     reference_indices: np.ndarray, embeddings_to_transform: Optional[np.ndarray] = None,
                     **kwargs) -> np.ndarray:
        """Fit the mapping and transform embeddings.
        
        Args:
            source_embeddings: Source embedding space
            target_embeddings: Target embedding space
            reference_indices: Reference indices for training
            embeddings_to_transform: Embeddings to transform (if None, use source_embeddings)
            **kwargs: Additional arguments
            
        Returns:
            Transformed embeddings
        """
        self.fit(source_embeddings, target_embeddings, reference_indices, **kwargs)
        
        if embeddings_to_transform is None:
            embeddings_to_transform = source_embeddings
            
        return self.transform(embeddings_to_transform, **kwargs)

    @classmethod
    def check_on_disk(cls, path: Union[str, Path]) -> bool:
        """Check if the mapping is fitted and on disk."""
        return Path(path).exists()
    
    def save(self, path: Union[str, Path]) -> None:
        """Save the fitted mapping to disk.
        
        Args:
            path: Path to save the mapping
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted mapping")
        
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save transformation matrix if available
        if self.transformation_matrix is not None:
            np.save(save_path / "transformation_matrix.npy", self.transformation_matrix)
        
        # Save metadata and config
        metadata = {
            "strategy_type": self.__class__.__name__,
            "config": self.config.to_dict(),
            "metadata": self.metadata,
            "is_fitted": self.is_fitted
        }
        
        import json
        with open(save_path / "mapping_info.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved mapping to {save_path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'MappingStrategy':
        """Load a fitted mapping from disk.
        
        Args:
            path: Path to load the mapping from
            
        Returns:
            Loaded mapping strategy instance
        """
        load_path = Path(path)
        
        # Load metadata
        import json
        with open(load_path / "mapping_info.json", "r") as f:
            metadata = json.load(f)
        
        # Create instance
        config = MappingConfig.from_dict(metadata["config"])
        instance = cls(config)
        
        # Load transformation matrix if available
        transformation_path = load_path / "transformation_matrix.npy"
        if transformation_path.exists():
            instance.transformation_matrix = np.load(transformation_path)
        
        # Restore state
        instance.is_fitted = metadata["is_fitted"]
        instance.metadata = metadata["metadata"]
        
        logger.info(f"Loaded mapping from {load_path}")
        return instance


class MappingResult:
    """Result container for mapping operations."""
    
    def __init__(self, transformed_embeddings: np.ndarray, 
                 mapping_strategy: str, metadata: Dict[str, Any]):
        """Initialize mapping result.
        
        Args:
            transformed_embeddings: The transformed embeddings
            mapping_strategy: Name of the strategy used
            metadata: Additional metadata about the mapping
        """
        self.transformed_embeddings = transformed_embeddings
        self.mapping_strategy = mapping_strategy
        self.metadata = metadata
        self.timestamp = np.datetime64('now')
    
    def save(self, path: Union[str, Path]) -> None:
        """Save mapping result to disk.
        
        Args:
            path: Path to save the result
        """
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings
        np.save(save_path / "transformed_embeddings.npy", self.transformed_embeddings)
        
        # Save metadata
        result_info = {
            "mapping_strategy": self.mapping_strategy,
            "metadata": self.metadata,
            "timestamp": str(self.timestamp),
            "shape": self.transformed_embeddings.shape
        }
        
        import json
        with open(save_path / "result_info.json", "w") as f:
            json.dump(result_info, f, indent=2)
        
        logger.info(f"Saved mapping result to {save_path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'MappingResult':
        """Load mapping result from disk.
        
        Args:
            path: Path to load from
            
        Returns:
            Loaded mapping result
        """
        load_path = Path(path)
        
        # Load embeddings
        embeddings = np.load(load_path / "transformed_embeddings.npy")
        
        # Load metadata
        import json
        with open(load_path / "result_info.json", "r") as f:
            info = json.load(f)
        
        result = cls(embeddings, info["mapping_strategy"], info["metadata"])
        result.timestamp = np.datetime64(info["timestamp"])
        
        logger.info(f"Loaded mapping result from {load_path}")
        return result 