"""Execution context for V2 runner and methods."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from .specs import TransformSpec


@dataclass(slots=True)
class RunContext:
    spec: TransformSpec
    config: Dict[str, Any]
    legacy_config: Any
    artifacts: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
