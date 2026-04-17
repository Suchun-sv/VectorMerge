"""Base method interface for V2 transform methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from vectormerge.core.context import RunContext


class BaseTransformMethod(ABC):
    name: str

    def prepare(self, context: RunContext) -> Dict[str, Any]:
        return {}

    @abstractmethod
    def fit(self, context: RunContext) -> Dict[str, Any]:
        raise NotImplementedError

    def transform(self, context: RunContext) -> Dict[str, Any]:
        return {}

    def evaluate(self, context: RunContext) -> Dict[str, Any]:
        return {}
