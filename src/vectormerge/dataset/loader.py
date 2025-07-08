"""
Unified dataset loader for VectorMerge.

This module provides a unified interface for loading datasets from different sources
(BEIR, custom formats, etc.) in a consistent way.
"""

from typing import Union, List, Optional, Dict, Any
from pathlib import Path

from loguru import logger

from .base import Dataset, DatasetLoader
from .beir import BEIRDatasetLoader
from .custom import CustomDatasetLoader


class UnifiedDatasetLoader:
    """Unified dataset loader that handles multiple dataset types."""
    
    def __init__(self, data_path: Union[str, Path]):
        """Initialize unified dataset loader.
        
        Args:
            data_path: Base path for datasets
        """
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize loaders for different sources
        self.loaders = {
            "beir": BEIRDatasetLoader(self.data_path),
            "custom": CustomDatasetLoader(self.data_path / "custom"),
        }
        
        logger.info(f"UnifiedDatasetLoader initialized with data_path: {self.data_path}")
    
    def load(self, dataset_name: str, split: str = "test", source: str = "beir", **kwargs) -> Dataset:
        """Load a dataset from specified source.
        
        Args:
            dataset_name: Name of the dataset
            split: Dataset split to load
            source: Source type (beir, custom, etc.)
            **kwargs: Additional arguments passed to the specific loader
            
        Returns:
            Dataset object
            
        Raises:
            ValueError: If source is not supported
        """
        if source not in self.loaders:
            raise ValueError(f"Unknown source: {source}. Available: {list(self.loaders.keys())}")
        
        logger.info(f"Loading dataset '{dataset_name}' from source '{source}' (split: {split})")
        
        try:
            dataset = self.loaders[source].load(dataset_name, split, **kwargs)
            logger.info(f"Successfully loaded dataset: {dataset.get_stats()}")
            return dataset
        except Exception as e:
            logger.error(f"Failed to load dataset '{dataset_name}' from source '{source}': {e}")
            raise
    
    def list_available_datasets(self, source: Optional[str] = None) -> Dict[str, List[str]]:
        """List available datasets by source.
        
        Args:
            source: If specified, only return datasets from this source
            
        Returns:
            Dictionary mapping source to list of datasets
        """
        if source:
            if source not in self.loaders:
                raise ValueError(f"Unknown source: {source}. Available: {list(self.loaders.keys())}")
            return {source: self.loaders[source].list_available_datasets()}
        
        result = {}
        for source_name, loader in self.loaders.items():
            try:
                datasets = loader.list_available_datasets()
                result[source_name] = datasets
            except Exception as e:
                logger.warning(f"Failed to list datasets for source '{source_name}': {e}")
                result[source_name] = []
        
        return result
    
    def auto_detect_source(self, dataset_name: str) -> str:
        """Auto-detect the source for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Source name
            
        Raises:
            ValueError: If dataset is not found in any source
        """
        for source, loader in self.loaders.items():
            try:
                if dataset_name in loader.list_available_datasets():
                    logger.info(f"Auto-detected source '{source}' for dataset '{dataset_name}'")
                    return source
            except Exception as e:
                logger.warning(f"Error checking source '{source}' for dataset '{dataset_name}': {e}")
                continue
        
        raise ValueError(f"Dataset '{dataset_name}' not found in any source")
    
    def get_dataset_info(self, dataset_name: str, source: Optional[str] = None) -> Dict[str, Any]:
        """Get information about a dataset.
        
        Args:
            dataset_name: Name of the dataset
            source: Source to check (if None, auto-detect)
            
        Returns:
            Dictionary containing dataset information
        """
        if source is None:
            source = self.auto_detect_source(dataset_name)
        
        if source not in self.loaders:
            raise ValueError(f"Unknown source: {source}")
        
        loader = self.loaders[source]
        
        # Get basic info from the loader if available
        if hasattr(loader, 'get_dataset_info'):
            return loader.get_dataset_info(dataset_name)
        
        # Fallback to basic info
        return {
            "name": dataset_name,
            "source": source,
            "available": loader.is_available(dataset_name),
        }
    
    def add_loader(self, source: str, loader: DatasetLoader) -> None:
        """Add a new dataset loader.
        
        Args:
            source: Source name
            loader: Dataset loader instance
        """
        self.loaders[source] = loader
        logger.info(f"Added new dataset loader for source: {source}")
    
    def remove_loader(self, source: str) -> None:
        """Remove a dataset loader.
        
        Args:
            source: Source name to remove
        """
        if source in self.loaders:
            del self.loaders[source]
            logger.info(f"Removed dataset loader for source: {source}")
        else:
            logger.warning(f"No loader found for source: {source}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about all available datasets.
        
        Returns:
            Dictionary containing loader statistics
        """
        stats = {
            "total_sources": len(self.loaders),
            "sources": list(self.loaders.keys()),
            "datasets_by_source": {},
            "total_datasets": 0,
        }
        
        for source, loader in self.loaders.items():
            try:
                datasets = loader.list_available_datasets()
                stats["datasets_by_source"][source] = len(datasets)
                stats["total_datasets"] += len(datasets)
            except Exception as e:
                logger.warning(f"Failed to get stats for source '{source}': {e}")
                stats["datasets_by_source"][source] = 0
        
        return stats
    
    def validate_dataset(self, dataset_name: str, source: Optional[str] = None) -> Dict[str, Any]:
        """Validate a dataset without fully loading it.
        
        Args:
            dataset_name: Name of the dataset
            source: Source to check (if None, auto-detect)
            
        Returns:
            Dictionary containing validation results
        """
        if source is None:
            try:
                source = self.auto_detect_source(dataset_name)
            except ValueError as e:
                return {
                    "valid": False,
                    "error": str(e),
                    "source": None,
                }
        
        try:
            loader = self.loaders[source]
            available = loader.is_available(dataset_name)
            
            result = {
                "valid": available,
                "source": source,
                "dataset_name": dataset_name,
            }
            
            if available:
                result["info"] = self.get_dataset_info(dataset_name, source)
            else:
                result["error"] = f"Dataset '{dataset_name}' not available in source '{source}'"
            
            return result
            
        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
                "source": source,
                "dataset_name": dataset_name,
            }


# Global dataset loader instance
_global_loader: Optional[UnifiedDatasetLoader] = None


def get_dataset_loader(data_path: Optional[Union[str, Path]] = None) -> UnifiedDatasetLoader:
    """Get the global dataset loader instance.
    
    Args:
        data_path: Path to datasets (uses default if None)
        
    Returns:
        UnifiedDatasetLoader instance
    """
    global _global_loader
    
    if _global_loader is None or data_path is not None:
        if data_path is None:
            data_path = "./data/raw/beir/"
        _global_loader = UnifiedDatasetLoader(data_path)
    
    return _global_loader


def load_dataset(dataset_name: str, split: str = "test", source: str = "beir", 
                data_path: Optional[Union[str, Path]] = None, **kwargs) -> Dataset:
    """Convenience function to load a dataset.
    
    Args:
        dataset_name: Name of the dataset
        split: Dataset split to load
        source: Source type (beir, custom, etc.)
        data_path: Path to datasets (uses default if None)
        **kwargs: Additional arguments passed to the specific loader
        
    Returns:
        Dataset object
    """
    loader = get_dataset_loader(data_path)
    
    # Auto-detect source if not explicitly BEIR and dataset not found
    if source == "beir" and dataset_name not in loader.loaders["beir"].list_available_datasets():
        try:
            detected_source = loader.auto_detect_source(dataset_name)
            if detected_source != "beir":
                source = detected_source
                logger.info(f"Auto-detected source '{source}' for dataset '{dataset_name}'")
        except ValueError:
            pass  # Keep original source
    
    return loader.load(dataset_name, split, source, **kwargs)


def list_datasets(source: Optional[str] = None, data_path: Optional[Union[str, Path]] = None) -> Dict[str, List[str]]:
    """List available datasets.
    
    Args:
        source: If specified, only return datasets from this source
        data_path: Path to datasets (uses default if None)
        
    Returns:
        Dictionary mapping source to list of datasets
    """
    loader = get_dataset_loader(data_path)
    return loader.list_available_datasets(source)


def validate_dataset(dataset_name: str, source: Optional[str] = None, 
                    data_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Validate a dataset.
    
    Args:
        dataset_name: Name of the dataset
        source: Source to check (if None, auto-detect)
        data_path: Path to datasets (uses default if None)
        
    Returns:
        Dictionary containing validation results
    """
    loader = get_dataset_loader(data_path)
    return loader.validate_dataset(dataset_name, source) 