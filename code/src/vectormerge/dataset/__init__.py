"""
VectorMerge Dataset Package

This package provides a unified interface for loading and working with
different types of datasets (BEIR, custom formats, etc.) in a consistent way.

Main classes:
- Dataset: Unified dataset container
- DatasetLoader: Abstract base class for dataset loaders
- BEIRDatasetLoader: Loader for BEIR datasets
- CustomDatasetLoader: Loader for custom dataset formats
- UnifiedDatasetLoader: Unified loader that handles multiple dataset types

Main functions:
- load_dataset: Convenience function to load a dataset
- list_datasets: List available datasets
- validate_dataset: Validate a dataset
- get_dataset_loader: Get the global dataset loader instance
"""

from .base import Dataset, DatasetLoader, EmbeddingDataset

from .beir import BEIRDatasetLoader, load_beir_dataset
from .custom import CustomDatasetLoader, load_custom_dataset

from .loader import (
    UnifiedDatasetLoader,
    get_dataset_loader,
    load_dataset,
    list_datasets,
    validate_dataset,
)

SUPPORTED_DATASETS = [
    "scifact",
    "nfcorpus",
    "arguana",
    "scidocs",
    "fiqa",
]

__all__ = [
    "Dataset",
    "DatasetLoader", 
    "EmbeddingDataset",
    
    "BEIRDatasetLoader",
    "CustomDatasetLoader",
    "UnifiedDatasetLoader",
    
    "load_dataset",
    "load_beir_dataset",
    "load_custom_dataset",
    "list_datasets",
    "validate_dataset",
    "get_dataset_loader",
    "SUPPORTED_DATASETS",
]
