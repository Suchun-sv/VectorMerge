"""V2 transform embedding command."""

from __future__ import annotations

from typing import Dict, List

import typer
import yaml
from rich import print as rprint

from vectormerge.core import RunOptions, TransformSpec, run_transform

tran_emb_app = typer.Typer(help="Run transform-embedding methods (V2).")


def _parse_set_pairs(set_values: List[str]) -> Dict[str, object]:
    overrides: Dict[str, object] = {}
    for item in set_values:
        if "=" not in item:
            raise typer.BadParameter(
                f"Invalid --set value '{item}'. Expected key=value."
            )
        key, raw = item.split("=", 1)
        overrides[key] = yaml.safe_load(raw)
    return overrides


@tran_emb_app.callback(invoke_without_command=True)
def tran_emb(
    method: str = typer.Option(
        ..., "--method", "-m", help="Method name (la2m|wasserstein)"
    ),
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset name"),
    source_model: str = typer.Option(..., "--source-model", "-s", help="Source model"),
    target_model: str = typer.Option(..., "--target-model", "-t", help="Target model"),
    set_values: List[str] = typer.Option(
        [], "--set", help="Override with key=value, repeatable"
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Force rerun"),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="Resume mode"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Run embedding transformation with a method-centric API."""

    overrides = _parse_set_pairs(set_values)
    spec = TransformSpec(
        method=method,
        dataset=dataset,
        source_model=source_model,
        target_model=target_model,
        overrides=overrides,
        options=RunOptions(force=force, resume=resume, verbose=verbose),
    )
    result = run_transform(spec)
    if result.status == "failed":
        if result.error:
            rprint(
                f"[red]{result.error.get('type', 'Error')}: {result.error.get('message', 'unknown error')}[/red]"
            )
        raise typer.Exit(code=1)
    rprint(
        f"[green]run_id={result.run_id} status={result.status} method={result.method}[/green]"
    )
    if result.metrics:
        rprint(result.metrics)
