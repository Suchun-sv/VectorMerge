"""Execution runners for V2 API specs."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

from vectormerge.config import load_settings_with_args, settings_to_legacy_config
from vectormerge.dataset import load_dataset
from vectormerge.embeddings.generate import (
    generate_embeddings as generate_embeddings_fn,
)
from vectormerge.registry import MethodRegistry

from .context import RunContext
from .results import DatasetPrepareResult, StageStatus, TransformResult
from .specs import DatasetPrepareSpec, TransformSpec


def _now_ms() -> int:
    return int(time.time() * 1000)


def _normalize_overrides(overrides: Dict[str, Any]) -> Dict[str, Any]:
    nested: Dict[str, Any] = {}
    for key, value in overrides.items():
        if "." not in key:
            nested[key] = value
            continue
        ptr = nested
        parts = key.split(".")
        for part in parts[:-1]:
            ptr = ptr.setdefault(part, {})
        ptr[parts[-1]] = value
    return nested


def _persist_run_manifest(run_id: str, payload: Dict[str, Any]) -> str:
    run_dir = Path("output") / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    return str(manifest_path)


def _ensure_embeddings_for_pair(
    legacy_config: Any, dataset: str, source_model: str, target_model: str, force: bool
) -> None:
    for model in [source_model, target_model]:
        model_settings = legacy_config.embedding_models.load_model_settings(model)
        for type_ in ("corpus", "query"):
            generate_embeddings_fn(
                model_name=model,
                dataset_name=dataset,
                dataset_path=str(legacy_config.data_path),
                cache_dir=str(legacy_config.embedding_cache_path),
                model_settings=model_settings,
                type_=type_,
                force=force,
                embedding_path=str(legacy_config.embedding_path),
                upload=False,
            )


def prepare_dataset_runner(spec: DatasetPrepareSpec) -> DatasetPrepareResult:
    run_id = spec.options.run_id or f"prep-{uuid.uuid4().hex[:10]}"
    stages: list[StageStatus] = []

    try:
        t0 = _now_ms()
        settings = load_settings_with_args(
            overrides=_normalize_overrides(spec.overrides)
        )
        legacy_config = settings_to_legacy_config(settings)
        stages.append(
            StageStatus(
                name="resolve_config", status="success", duration_ms=_now_ms() - t0
            )
        )

        t1 = _now_ms()
        load_dataset(
            spec.dataset,
            data_path=legacy_config.data_path,
            force_download=spec.options.force,
        )
        stages.append(
            StageStatus(
                name="ensure_dataset", status="success", duration_ms=_now_ms() - t1
            )
        )

        t2 = _now_ms()
        detail = f"model_mode={spec.model_mode}"
        status = "skipped"
        if spec.model_mode == "pair" and spec.source_model and spec.target_model:
            _ensure_embeddings_for_pair(
                legacy_config,
                dataset=spec.dataset,
                source_model=spec.source_model,
                target_model=spec.target_model,
                force=spec.options.force,
            )
            status = "success"
            detail = "model_mode=pair"
        stages.append(
            StageStatus(
                name="ensure_embeddings",
                status=status,
                detail=detail,
                duration_ms=_now_ms() - t2,
            )
        )

        artifacts = {
            "data_path": str(legacy_config.data_path),
            "embedding_path": str(legacy_config.embedding_path),
        }
        artifacts["manifest_path"] = _persist_run_manifest(
            run_id,
            {
                "run_id": run_id,
                "status": "success",
                "dataset": spec.dataset,
                "artifacts": artifacts,
                "stages": [asdict(s) for s in stages],
                "config_snapshot": settings.model_dump(),
            },
        )

        return DatasetPrepareResult(
            run_id=run_id,
            status="success",
            dataset=spec.dataset,
            artifacts=artifacts,
            stages=stages,
            config_snapshot=settings.model_dump(),
        )
    except Exception as exc:
        stages.append(
            StageStatus(name="prepare_dataset", status="failed", detail=str(exc))
        )
        artifacts = {
            "manifest_path": _persist_run_manifest(
                run_id,
                {
                    "run_id": run_id,
                    "status": "failed",
                    "dataset": spec.dataset,
                    "stages": [asdict(s) for s in stages],
                    "error": {"message": str(exc), "type": type(exc).__name__},
                },
            )
        }
        return DatasetPrepareResult(
            run_id=run_id,
            status="failed",
            dataset=spec.dataset,
            artifacts=artifacts,
            stages=stages,
            error={"message": str(exc), "type": type(exc).__name__},
        )


def transform_runner(spec: TransformSpec) -> TransformResult:
    run_id = spec.options.run_id or f"tran-{uuid.uuid4().hex[:10]}"
    stages: list[StageStatus] = []

    try:
        t0 = _now_ms()
        settings = load_settings_with_args(
            overrides=_normalize_overrides(spec.overrides)
        )
        legacy_config = settings_to_legacy_config(settings)
        stages.append(
            StageStatus(
                name="resolve_config", status="success", duration_ms=_now_ms() - t0
            )
        )

        t1 = _now_ms()
        load_dataset(
            spec.dataset,
            data_path=legacy_config.data_path,
            force_download=spec.options.force,
        )
        stages.append(
            StageStatus(
                name="ensure_dataset", status="success", duration_ms=_now_ms() - t1
            )
        )

        t2 = _now_ms()
        _ensure_embeddings_for_pair(
            legacy_config,
            dataset=spec.dataset,
            source_model=spec.source_model,
            target_model=spec.target_model,
            force=spec.options.force,
        )
        stages.append(
            StageStatus(
                name="ensure_embeddings", status="success", duration_ms=_now_ms() - t2
            )
        )

        t3 = _now_ms()
        method = MethodRegistry.create(spec.method)
        context = RunContext(
            spec=spec,
            config=settings.model_dump(),
            legacy_config=legacy_config,
            artifacts={
                "data_path": str(legacy_config.data_path),
                "embedding_path": str(legacy_config.embedding_path),
                "reference_path": str(legacy_config.reference_path),
                "mapping_param_path": str(legacy_config.mapping_param_path),
                "mapping_embedding_path": str(legacy_config.mapping_embedding_path),
            },
        )
        method.prepare(context)
        method.fit(context)
        metrics = method.evaluate(context) if spec.eval_enabled else {}
        stages.append(
            StageStatus(
                name="fit_transform_evaluate",
                status="success",
                duration_ms=_now_ms() - t3,
            )
        )

        context.artifacts["manifest_path"] = _persist_run_manifest(
            run_id,
            {
                "run_id": run_id,
                "status": "success",
                "method": spec.method,
                "dataset": spec.dataset,
                "source_model": spec.source_model,
                "target_model": spec.target_model,
                "metrics": metrics,
                "artifacts": context.artifacts,
                "stages": [asdict(s) for s in stages],
                "config_snapshot": settings.model_dump(),
            },
        )

        return TransformResult(
            run_id=run_id,
            status="success",
            method=spec.method,
            dataset=spec.dataset,
            source_model=spec.source_model,
            target_model=spec.target_model,
            metrics=metrics,
            artifacts=context.artifacts,
            stages=stages,
            config_snapshot=settings.model_dump(),
        )
    except Exception as exc:
        stages.append(StageStatus(name="transform", status="failed", detail=str(exc)))
        artifacts = {
            "manifest_path": _persist_run_manifest(
                run_id,
                {
                    "run_id": run_id,
                    "status": "failed",
                    "method": spec.method,
                    "dataset": spec.dataset,
                    "source_model": spec.source_model,
                    "target_model": spec.target_model,
                    "stages": [asdict(s) for s in stages],
                    "error": {"message": str(exc), "type": type(exc).__name__},
                },
            )
        }
        return TransformResult(
            run_id=run_id,
            status="failed",
            method=spec.method,
            dataset=spec.dataset,
            source_model=spec.source_model,
            target_model=spec.target_model,
            artifacts=artifacts,
            stages=stages,
            error={"message": str(exc), "type": type(exc).__name__},
        )
