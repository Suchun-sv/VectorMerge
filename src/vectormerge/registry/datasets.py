"""Dataset registry skeleton for V2 runtime."""

from __future__ import annotations

from typing import Any, Callable, Dict


class DatasetRegistry:
    _registry: Dict[str, Callable[..., Any]] = {}

    @classmethod
    def register(cls, name: str, loader: Callable[..., Any]) -> None:
        cls._registry[name] = loader

    @classmethod
    def get_loader(cls, name: str) -> Callable[..., Any]:
        if name not in cls._registry:
            raise KeyError(f"Unknown dataset loader '{name}'.")
        return cls._registry[name]
