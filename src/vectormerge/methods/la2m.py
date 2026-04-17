"""LA2M method adapter for V2 runtime."""

from __future__ import annotations

from typing import Any, Dict

from .base import BaseTransformMethod
from vectormerge.core.context import RunContext
from vectormerge.pipeline import evaluation_pipeline


class LA2MMethod(BaseTransformMethod):
    name = "la2m"

    def fit(self, context: RunContext) -> Dict[str, Any]:
        reference_key = context.config.get("reference_key")
        reference_method = context.config.get("reference_strategy", "la2m")
        output = evaluation_pipeline(
            config=context.legacy_config,
            dataset_name=context.spec.dataset,
            source_model=context.spec.source_model,
            target_model=context.spec.target_model,
            mapping_method="la2m",
            reference_key=reference_key,
            reference_method=reference_method,
            force=context.spec.options.force,
            verbose=context.spec.options.verbose,
        )
        context.metadata["metrics"] = output.get("results", {})
        return {"status": "fit_done", "method": self.name}

    def evaluate(self, context: RunContext) -> Dict[str, Any]:
        metrics = dict(context.metadata.get("metrics", {}))
        metrics.setdefault("method", self.name)
        return metrics
