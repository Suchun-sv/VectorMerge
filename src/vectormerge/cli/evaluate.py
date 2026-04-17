import typer
from typing import Any
from click import Context
import wandb

from vectormerge.cli.base import (
    SUPPORTED_DATASETS,
    SUPPORTED_MODELS,
    SUPPORTED_MAPPING_METHODS,
    set_wandb,
)
from vectormerge.cli.utils import handle_extra_args

from vectormerge.mapping import VectorSpaceMapper
from vectormerge.dataset import load_dataset
from vectormerge.embeddings import get_embedding
from vectormerge.reference import (
    get_reference,
    split_dataset,
    SUPPORTED_REFERENCE_STRATEGIES,
)
from vectormerge.evaluation import Evaluator
from vectormerge.embedding_dataset import EmbeddingDataset
from vectormerge.pipeline import evaluation_pipeline

evaluate_app = typer.Typer(
    name="evaluate",
    help="Evaluate the performance of a mapping model, you can pass extra arguments to the command to override the default values. \n E.g. --mapping_config.la2m.num_clusters=50",
)


@evaluate_app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def single_run(
    ctx: Context,
    dataset_name: str = typer.Option(
        ...,
        "--dataset",
        "-d",
        help="Dataset name, support: " + ", ".join(SUPPORTED_DATASETS),
    ),
    source_model: str = typer.Option(
        None,
        "--source-model",
        "-s",
        help="Source model name, support: " + ", ".join(SUPPORTED_MODELS),
    ),
    target_model: str = typer.Option(
        None,
        "--target-model",
        "-t",
        help="Target model name, support: " + ", ".join(SUPPORTED_MODELS),
    ),
    src_tar_model: str = typer.Option(
        None,
        "--src-tar-model",
        "-stm",
        help="Combination of source and target model, use `_` to separate the source and target model, e.g. `mistral_openai` support: "
        + ", ".join(SUPPORTED_MODELS),
    ),
    reference_key: str = typer.Option(
        None,
        "--reference-key",
        "-rk",
        help="Reference key, if not provided, the dataset will be split into D0, D1, D2 based on the reference method",
    ),
    reference_method: str = typer.Option(
        "la2m",
        "--reference-method",
        "-rm",
        help="Reference method, support: " + ", ".join(SUPPORTED_REFERENCE_STRATEGIES),
    ),
    mapping_method: str = typer.Option(
        ...,
        "--mapping-method",
        "-mm",
        help="Mapping method, support: " + ", ".join(SUPPORTED_MAPPING_METHODS),
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Force evaluation"),
    interactive: bool = typer.Option(
        False, "--interactive", "-i", help="Interactive mode"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    wandb_entity: str = typer.Option(
        None, "--wandb-entity", "-we", help="Weights and Biases"
    ),
    wandb_project: str = typer.Option(
        "vector-merge", "--wandb-project", "-wp", help="Weights and Biases project"
    ),
):
    """
    Evaluate the performance of a mapping model, you can pass extra arguments to the command to override the default values.
    E.g. --mapping_config.la2m.num_clusters=50
    """
    if src_tar_model is not None:
        source_model = src_tar_model.split("_")[0]
        target_model = src_tar_model.split("_")[1]

    if source_model not in SUPPORTED_MODELS:
        raise ValueError(
            f"Source model is not supported, support: " + ", ".join(SUPPORTED_MODELS)
        )
    if target_model not in SUPPORTED_MODELS:
        raise ValueError(
            f"Target model is not supported, support: " + ", ".join(SUPPORTED_MODELS)
        )

    if source_model is None or target_model is None:
        raise ValueError("Source and target model are required")

    config = handle_extra_args(ctx)

    set_wandb(
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        config_dict=config.to_dict(),
    )

    output = evaluation_pipeline(
        config=config,
        dataset_name=dataset_name,
        source_model=source_model,
        target_model=target_model,
        mapping_method=mapping_method,
        reference_key=reference_key,
        reference_method=reference_method,
        force=force,
        verbose=verbose,
    )
    results = output["results"]
    wandb.log(results)
    from rich.console import Console

    console = Console()
    console.rule("results")
    console.print(results)
