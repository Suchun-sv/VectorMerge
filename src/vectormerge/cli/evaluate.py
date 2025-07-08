import typer
from typing import Any
from click import Context
from .base import SUPPORTED_DATASETS
from .base import SUPPORTED_MODELS

evaluate_app = typer.Typer(
    name="evaluate",
    help="Evaluate the performance of a mapping model, you can pass extra arguments to the command to override the default values. \n E.g. --mapping_config.la2m.num_clusters=50"
)

@evaluate_app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def single_run(
    ctx: Context,
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset name, support: " + ", ".join(SUPPORTED_DATASETS)),
    source_model: str = typer.Option(..., "--source-model", "-s", help="Source model name, support: " + ", ".join(SUPPORTED_MODELS)),
    target_model: str = typer.Option(..., "--target-model", "-t", help="Target model name, support: " + ", ".join(SUPPORTED_MODELS)),
    reference_key: str = typer.Option(..., "--reference-key", "-rk", help="Reference key"),
    force: bool = typer.Option(False, "--force", "-f", help="Force evaluation"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive mode"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Evaluate the performance of a mapping model, you can pass extra arguments to the command to override the default values.
    E.g. --mapping_config.la2m.num_clusters=50
    """
    extra = ctx.args  # 所有未知参数都在这里
    dynamic: dict[str, Any] = {}
    for arg in extra:
        if arg.startswith("--") and "=" in arg:
            key_path, val = arg.lstrip("-").split("=", 1)
            ptr = dynamic
            parts = key_path.split(".")
            for p in parts[:-1]:
                ptr = ptr.setdefault(p, {})
            ptr[parts[-1]] = val

    typer.secho("✅ Parsed known args:", fg="green")
    typer.echo(f" dataset={dataset}, source_model={source_model}, target_model={target_model}")
    typer.echo(f" force={force}, interactive={interactive}, verbose={verbose}")
    typer.secho("🔧 Parsed dynamic config:", fg="cyan")
    typer.echo(dynamic)

    # TODO: 把 dynamic merge 到你的 config 对象里，并继续 evaluate 流程
