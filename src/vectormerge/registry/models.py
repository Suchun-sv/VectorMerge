"""Model registry skeleton for V2 runtime."""

from __future__ import annotations

from typing import Any, Callable, Dict


class ModelRegistry:
    _registry: Dict[str, Callable[..., Any]] = {}

    @classmethod
    def register(cls, name: str, backend: Callable[..., Any]) -> None:
        cls._registry[name] = backend

    @classmethod
    def get_backend(cls, name: str) -> Callable[..., Any]:
        if name not in cls._registry:
            raise KeyError(f"Unknown model backend '{name}'.")
        return cls._registry[name]
