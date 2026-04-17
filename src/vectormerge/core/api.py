"""Stable V2 public runtime API."""

from __future__ import annotations

from .runner import prepare_dataset_runner, transform_runner
from .results import DatasetPrepareResult, TransformResult
from .specs import DatasetPrepareSpec, TransformSpec


def prepare_dataset(spec: DatasetPrepareSpec) -> DatasetPrepareResult:
    return prepare_dataset_runner(spec)


def run_transform(spec: TransformSpec) -> TransformResult:
    return transform_runner(spec)
