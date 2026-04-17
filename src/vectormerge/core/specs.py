"""Core input specifications for V2 runtime APIs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


SUPPORTED_V2_METHODS = {"la2m", "wasserstein"}
SUPPORTED_MODEL_MODES = {"none", "all", "pair"}


@dataclass(slots=True)
class RunOptions:
    force: bool = False
    resume: bool = True
    dry_run: bool = False
    seed: int = 42
    verbose: bool = False
    run_id: Optional[str] = None


@dataclass(slots=True)
class DatasetPrepareSpec:
    dataset: str
    model_mode: str = "none"
    source_model: Optional[str] = None
    target_model: Optional[str] = None
    overrides: Dict[str, Any] = field(default_factory=dict)
    options: RunOptions = field(default_factory=RunOptions)

    def __post_init__(self) -> None:
        if self.model_mode not in SUPPORTED_MODEL_MODES:
            raise ValueError(
                f"Invalid model_mode '{self.model_mode}'. Supported: {sorted(SUPPORTED_MODEL_MODES)}"
            )
        if self.model_mode == "pair" and (
            not self.source_model or not self.target_model
        ):
            raise ValueError(
                "source_model and target_model are required when model_mode='pair'"
            )


@dataclass(slots=True)
class TransformSpec:
    method: str
    dataset: str
    source_model: str
    target_model: str
    overrides: Dict[str, Any] = field(default_factory=dict)
    eval_enabled: bool = True
    options: RunOptions = field(default_factory=RunOptions)

    def __post_init__(self) -> None:
        if self.method not in SUPPORTED_V2_METHODS:
            raise ValueError(
                f"Invalid method '{self.method}'. Supported: {sorted(SUPPORTED_V2_METHODS)}"
            )
