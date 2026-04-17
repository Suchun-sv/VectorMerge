"""Method registry for V2 runtime."""

from __future__ import annotations

from typing import Dict, Type

from vectormerge.methods.base import BaseTransformMethod
from vectormerge.methods.la2m import LA2MMethod
from vectormerge.methods.wasserstein import WassersteinMethod


class MethodRegistry:
    _registry: Dict[str, Type[BaseTransformMethod]] = {}

    @classmethod
    def register(cls, name: str, method_cls: Type[BaseTransformMethod]) -> None:
        cls._registry[name] = method_cls

    @classmethod
    def create(cls, name: str) -> BaseTransformMethod:
        if name not in cls._registry:
            raise KeyError(
                f"Unknown method '{name}'. Registered: {sorted(cls._registry)}"
            )
        return cls._registry[name]()

    @classmethod
    def list_methods(cls) -> list[str]:
        return sorted(cls._registry.keys())


MethodRegistry.register("la2m", LA2MMethod)
MethodRegistry.register("wasserstein", WassersteinMethod)
