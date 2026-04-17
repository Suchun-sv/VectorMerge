"""Pipeline orchestration package for CLI-to-pipeline mapping."""

from .dataset_pipeline import dataset_pipeline
from .embedding_pipeline import embedding_pipeline
from .reference_pipeline import reference_pipeline
from .cluster_pipeline import cluster_pipeline
from .mapping_pipeline import mapping_pipeline
from .evaluation_pipeline import evaluation_pipeline

__all__ = [
    "dataset_pipeline",
    "embedding_pipeline",
    "reference_pipeline",
    "cluster_pipeline",
    "mapping_pipeline",
    "evaluation_pipeline",
]
