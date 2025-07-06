"""
Main mapper class for VectorMerge.

This module provides the VectorSpaceMapper class, which serves as the main
interface for all embedding mapping strategies in VectorMerge.
"""

import numpy as np
from typing import Union, Dict, Any, Optional, List
from pathlib import Path
import time
from loguru import logger

from .base import MappingStrategy, MappingConfig, MappingResult
from .strategies import (
    ProcrustesMappingStrategy,
    NonLinearMappingStrategy,
    LA2MStrategy
)


class VectorSpaceMapper:
    """
    Main interface for vector space mapping in VectorMerge.
    
    This class provides a unified interface for different mapping strategies
    and handles the complete mapping workflow from training to transformation.
    """
    
    # Available mapping strategies
    AVAILABLE_STRATEGIES = {
        "procrustes": ProcrustesMappingStrategy,
        "nonlinear": NonLinearMappingStrategy,
        "la2m": LA2MStrategy,
    }
    
    def __init__(self, strategy: str = "procrustes", config: Optional[MappingConfig] = None):
        """Initialize the VectorSpaceMapper.
        
        Args:
            strategy: Name of the mapping strategy to use
            config: Configuration object (if None, uses default config)
        """
        if strategy not in self.AVAILABLE_STRATEGIES:
            raise ValueError(f"Unknown strategy '{strategy}'. Available strategies: "
                           f"{list(self.AVAILABLE_STRATEGIES.keys())}")
        
        self.strategy_name = strategy
        self.config = config if config is not None else MappingConfig()
        
        # Initialize the mapping strategy
        strategy_class = self.AVAILABLE_STRATEGIES[strategy]
        self.mapping_strategy: MappingStrategy = strategy_class(self.config)
        
        # Training history and metadata
        self.training_history: Dict[str, Any] = {}
        self.is_fitted = False
        
        logger.info(f"VectorSpaceMapper initialized with strategy: {strategy}")
    
    def fit(self, source_embeddings: np.ndarray, target_embeddings: np.ndarray,
            reference_indices: np.ndarray, **kwargs) -> 'VectorSpaceMapper':
        """Fit the mapping strategy on reference data.
        
        Args:
            source_embeddings: Source embedding space (N x D1)
            target_embeddings: Target embedding space (N x D2)  
            reference_indices: Indices of reference points for training
            **kwargs: Additional arguments for the mapping strategy
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting {self.strategy_name} mapping strategy...")
        
        # Validate inputs
        self._validate_inputs(source_embeddings, target_embeddings, reference_indices)
        
        # Record training start time
        start_time = time.time()
        
        # Fit the mapping strategy
        self.mapping_strategy.fit(
            source_embeddings, target_embeddings, reference_indices, **kwargs
        )
        
        # Record training time
        training_time = time.time() - start_time
        
        # Update training history
        self.training_history = {
            'strategy': self.strategy_name,
            'training_time': training_time,
            'reference_size': len(reference_indices),
            'source_shape': source_embeddings.shape,
            'target_shape': target_embeddings.shape,
            'config': self.config.to_dict(),
            'strategy_metadata': self.mapping_strategy.metadata.copy()
        }
        
        self.is_fitted = True
        
        logger.info(f"Mapping strategy fitted successfully in {training_time:.2f} seconds")
        return self
    
    def transform(self, embeddings: np.ndarray, **kwargs) -> np.ndarray:
        """Transform embeddings using the fitted mapping strategy.
        
        Args:
            embeddings: Embeddings to transform (N x D1)
            **kwargs: Additional arguments for the mapping strategy
            
        Returns:
            Transformed embeddings (N x D2)
        """
        if not self.is_fitted:
            raise ValueError("Mapper must be fitted before transformation")
        
        logger.info(f"Transforming {len(embeddings)} embeddings...")
        
        start_time = time.time()
        transformed = self.mapping_strategy.transform(embeddings, **kwargs)
        transform_time = time.time() - start_time
        
        logger.info(f"Transformation completed in {transform_time:.2f} seconds")
        
        return transformed
    
    def fit_transform(self, source_embeddings: np.ndarray, target_embeddings: np.ndarray,
                     reference_indices: np.ndarray, embeddings_to_transform: Optional[np.ndarray] = None,
                     **kwargs) -> np.ndarray:
        """Fit the mapping and transform embeddings in one step.
        
        Args:
            source_embeddings: Source embedding space
            target_embeddings: Target embedding space
            reference_indices: Reference indices for training
            embeddings_to_transform: Embeddings to transform (if None, uses source_embeddings)
            **kwargs: Additional arguments
            
        Returns:
            Transformed embeddings
        """
        # Fit the mapping
        self.fit(source_embeddings, target_embeddings, reference_indices, **kwargs)
        
        # Transform embeddings
        if embeddings_to_transform is None:
            embeddings_to_transform = source_embeddings
        
        return self.transform(embeddings_to_transform, **kwargs)
    
    def get_mapping_result(self, transformed_embeddings: np.ndarray) -> MappingResult:
        """Create a MappingResult object with metadata.
        
        Args:
            transformed_embeddings: The transformed embeddings
            
        Returns:
            MappingResult object
        """
        if not self.is_fitted:
            raise ValueError("Mapper must be fitted before creating result")
        
        metadata = self.training_history.copy()
        metadata['transform_shape'] = transformed_embeddings.shape
        
        return MappingResult(
            transformed_embeddings=transformed_embeddings,
            mapping_strategy=self.strategy_name,
            metadata=metadata
        )
    
    def evaluate_mapping(self, source_embeddings: np.ndarray, target_embeddings: np.ndarray,
                        test_indices: np.ndarray) -> Dict[str, float]:
        """Evaluate the mapping quality on test data.
        
        Args:
            source_embeddings: Source embeddings
            target_embeddings: Target embeddings  
            test_indices: Indices of test points
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Mapper must be fitted before evaluation")
        
        logger.info(f"Evaluating mapping on {len(test_indices)} test points...")
        
        # Transform test source embeddings
        test_source = source_embeddings[test_indices]
        test_target = target_embeddings[test_indices]
        transformed_test = self.transform(test_source)
        
        # Compute evaluation metrics
        metrics = {}
        
        # Mean Squared Error
        mse = np.mean((transformed_test - test_target) ** 2)
        metrics['mse'] = float(mse)
        
        # Root Mean Squared Error
        metrics['rmse'] = float(np.sqrt(mse))
        
        # Mean Absolute Error
        mae = np.mean(np.abs(transformed_test - test_target))
        metrics['mae'] = float(mae)
        
        # Cosine Similarity
        def cosine_similarity(a, b):
            a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
            b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
            return np.mean(np.sum(a_norm * b_norm, axis=1))
        
        cos_sim = cosine_similarity(transformed_test, test_target)
        metrics['cosine_similarity'] = float(cos_sim)
        
        # Pearson Correlation (if embeddings are 1D or we take mean)
        try:
            if transformed_test.shape[1] == 1:
                corr = np.corrcoef(transformed_test.flatten(), test_target.flatten())[0, 1]
            else:
                # Average correlation across dimensions
                corrs = []
                for i in range(transformed_test.shape[1]):
                    corr = np.corrcoef(transformed_test[:, i], test_target[:, i])[0, 1]
                    if not np.isnan(corr):
                        corrs.append(corr)
                corr = np.mean(corrs) if corrs else 0.0
            
            metrics['correlation'] = float(corr) if not np.isnan(corr) else 0.0
        except:
            metrics['correlation'] = 0.0
        
        logger.info(f"Evaluation completed. MSE: {metrics['mse']:.6f}, "
                   f"Cosine Similarity: {metrics['cosine_similarity']:.6f}")
        
        return metrics
    
    def save(self, path: Union[str, Path]) -> None:
        """Save the fitted mapper to disk.
        
        Args:
            path: Path to save the mapper
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted mapper")
        
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save the mapping strategy
        self.mapping_strategy.save(save_path / "strategy")
        
        # Save mapper metadata
        mapper_info = {
            'strategy_name': self.strategy_name,
            'config': self.config.to_dict(),
            'training_history': self.training_history,
            'is_fitted': self.is_fitted
        }
        
        import json
        with open(save_path / "mapper_info.json", "w") as f:
            json.dump(mapper_info, f, indent=2)
        
        logger.info(f"Saved VectorSpaceMapper to {save_path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'VectorSpaceMapper':
        """Load a fitted mapper from disk.
        
        Args:
            path: Path to load the mapper from
            
        Returns:
            Loaded VectorSpaceMapper instance
        """
        load_path = Path(path)
        
        # Load mapper metadata
        import json
        with open(load_path / "mapper_info.json", "r") as f:
            mapper_info = json.load(f)
        
        # Create mapper instance
        config = MappingConfig.from_dict(mapper_info['config'])
        mapper = cls(strategy=mapper_info['strategy_name'], config=config)
        
        # Load the mapping strategy
        strategy_class = cls.AVAILABLE_STRATEGIES[mapper_info['strategy_name']]
        mapper.mapping_strategy = strategy_class.load(load_path / "strategy")
        
        # Restore state
        mapper.training_history = mapper_info['training_history']
        mapper.is_fitted = mapper_info['is_fitted']
        
        logger.info(f"Loaded VectorSpaceMapper from {load_path}")
        return mapper
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get information about the current mapping strategy.
        
        Returns:
            Dictionary with strategy information
        """
        info = {
            'strategy_name': self.strategy_name,
            'is_fitted': self.is_fitted,
            'config': self.config.to_dict()
        }
        
        if self.is_fitted:
            info['training_history'] = self.training_history
            info['strategy_metadata'] = self.mapping_strategy.metadata
        
        # Add strategy-specific information
        if hasattr(self.mapping_strategy, 'get_cluster_statistics'):
            info['cluster_statistics'] = self.mapping_strategy.get_cluster_statistics()
        
        if hasattr(self.mapping_strategy, 'get_training_history'):
            info['training_losses'] = self.mapping_strategy.get_training_history()
        
        return info
    
    def _validate_inputs(self, source_embeddings: np.ndarray, target_embeddings: np.ndarray,
                        reference_indices: np.ndarray) -> None:
        """Validate input arrays for fitting.
        
        Args:
            source_embeddings: Source embeddings
            target_embeddings: Target embeddings
            reference_indices: Reference indices
        """
        # Check array types
        if not isinstance(source_embeddings, np.ndarray):
            raise TypeError("source_embeddings must be a numpy array")
        if not isinstance(target_embeddings, np.ndarray):
            raise TypeError("target_embeddings must be a numpy array")
        if not isinstance(reference_indices, np.ndarray):
            raise TypeError("reference_indices must be a numpy array")
        
        # Check dimensions
        if len(source_embeddings.shape) != 2:
            raise ValueError("source_embeddings must be 2D array")
        if len(target_embeddings.shape) != 2:
            raise ValueError("target_embeddings must be 2D array")
        
        # Check matching number of samples
        if source_embeddings.shape[0] != target_embeddings.shape[0]:
            raise ValueError("source_embeddings and target_embeddings must have same number of samples")
        
        # Check reference indices
        if len(reference_indices) == 0:
            raise ValueError("reference_indices cannot be empty")
        
        if np.any(reference_indices < 0) or np.any(reference_indices >= source_embeddings.shape[0]):
            raise ValueError("reference_indices contain invalid indices")
        
        # Check for sufficient reference points
        min_points = 2 if self.strategy_name != "la2m" else 4  # LA2M strategy needs more points for clustering
        if len(reference_indices) < min_points:
            raise ValueError(f"At least {min_points} reference points required for {self.strategy_name} strategy")
    
    @classmethod
    def list_available_strategies(cls) -> List[str]:
        """List all available mapping strategies.
        
        Returns:
            List of strategy names
        """
        return list(cls.AVAILABLE_STRATEGIES.keys())
    
    @classmethod
    def get_strategy_description(cls, strategy: str) -> str:
        """Get description of a mapping strategy.
        
        Args:
            strategy: Strategy name
            
        Returns:
            Strategy description
        """
        descriptions = {
            "procrustes": "Procrustes analysis - orthogonal transformation using SVD",
            "nonlinear": "Non-linear neural network - deep network with non-linear activations",
            "la2m": "Clustering-based mapping - learns local mappings for each cluster"
        }
        
        return descriptions.get(strategy, "Unknown strategy")
    
    def __repr__(self) -> str:
        """String representation of the mapper."""
        status = "fitted" if self.is_fitted else "not fitted"
        return f"VectorSpaceMapper(strategy='{self.strategy_name}', status='{status}')" 