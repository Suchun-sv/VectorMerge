"""VectorMerge Dataset Package."""

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
