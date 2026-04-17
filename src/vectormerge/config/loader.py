"""Configuration loader utilities for settings.yaml-based flow."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml
from dotenv import dotenv_values

from .base import VectorMergeConfig
from .models import Settings


def _merge_dict(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge_dict(base[key], value)
        else:
            base[key] = value
    return base


def _parse_scalar(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        parsed = yaml.safe_load(value)
        return parsed
    except Exception:
        return value


def _set_nested(target: Dict[str, Any], dotted_key: str, value: Any) -> None:
    ptr = target
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        ptr = ptr.setdefault(part, {})
    ptr[parts[-1]] = value


def _flatten_env_overrides(prefix: str = "VECTORMERGE") -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    normalized_prefix = f"{prefix}_"
    for key, value in os.environ.items():
        if not key.startswith(normalized_prefix):
            continue
        dot_key = key[len(normalized_prefix) :].lower().replace("__", ".")
        _set_nested(result, dot_key, _parse_scalar(value))
    return result


def _flatten_dotenv_overrides(
    prefix: str = "VECTORMERGE", dotenv_path: Path | None = None
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    path = dotenv_path or (Path.cwd() / ".env")
    if not path.exists():
        return result

    values = dotenv_values(path)
    normalized_prefix = f"{prefix}_"
    for key, value in values.items():
        if key is None or value is None or not key.startswith(normalized_prefix):
            continue
        dot_key = key[len(normalized_prefix) :].lower().replace("__", ".")
        _set_nested(result, dot_key, _parse_scalar(value))
    return result


def parse_overrides(args: Iterable[str]) -> Dict[str, Any]:
    """Parse CLI-style overrides into a nested dictionary.

    Supported formats:
    - ``--key.sub=value``
    - ``--set key.sub=value``
    - ``-s key.sub=value``
    """

    args_list = list(args)
    dynamic: Dict[str, Any] = {}

    i = 0
    while i < len(args_list):
        arg = args_list[i]

        if arg in {"--set", "-s"}:
            if i + 1 < len(args_list) and "=" in args_list[i + 1]:
                key_path, raw = args_list[i + 1].split("=", 1)
                _set_nested(dynamic, key_path, _parse_scalar(raw))
                i += 2
                continue

        if arg.startswith("--") and "=" in arg:
            key_path, raw = arg.lstrip("-").split("=", 1)
            _set_nested(dynamic, key_path, _parse_scalar(raw))
        i += 1

    return dynamic


def load_settings_with_args(
    overrides: Dict[str, Any] | None = None,
    settings_path: Path | None = None,
    env_prefix: str = "VECTORMERGE",
) -> Settings:
    """Load runtime settings with deterministic precedence.

    Precedence (low -> high): defaults -> settings.yaml -> .env -> env vars -> CLI overrides.
    """

    merged: Dict[str, Any] = {}

    path = settings_path or (Path.cwd() / "settings.yaml")
    if path.exists():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict):
            _merge_dict(merged, loaded)

    _merge_dict(merged, _flatten_dotenv_overrides(prefix=env_prefix))
    _merge_dict(merged, _flatten_env_overrides(prefix=env_prefix))

    if overrides:
        _merge_dict(merged, overrides)

    return Settings.model_validate(merged)


def settings_to_legacy_config(settings: Settings) -> VectorMergeConfig:
    """Bridge new settings model to legacy runtime config dataclass."""

    return VectorMergeConfig.from_dict(settings.model_dump())
