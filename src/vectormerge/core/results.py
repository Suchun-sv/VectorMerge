"""Core output structures for V2 runtime APIs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class StageStatus:
    name: str
    status: str
    detail: str = ""
    duration_ms: Optional[int] = None


@dataclass(slots=True)
class DatasetPrepareResult:
    run_id: str
    status: str
    dataset: str
    artifacts: Dict[str, str] = field(default_factory=dict)
    stages: List[StageStatus] = field(default_factory=list)
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Dict[str, str]] = None


@dataclass(slots=True)
class TransformResult:
    run_id: str
    status: str
    method: str
    dataset: str
    source_model: str
    target_model: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    stages: List[StageStatus] = field(default_factory=list)
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Dict[str, str]] = None
