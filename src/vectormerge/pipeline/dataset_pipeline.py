"""Dataset pipeline entrypoint."""

from typing import Any, Dict, List

from vectormerge.dataset import SUPPORTED_DATASETS, load_dataset


def dataset_pipeline(
    config: Any, dataset: str, force: bool = False, verbose: bool = False
) -> Dict[str, Any]:
    """Download one or more datasets through a pipeline boundary."""

    data_path = getattr(config, "data_path", "./data/raw/beir/")
    datasets: List[str] = SUPPORTED_DATASETS if dataset == "all" else [dataset]

    success: List[str] = []
    failed: List[str] = []

    for name in datasets:
        try:
            load_dataset(name, data_path=data_path, force_download=force)
            success.append(name)
        except Exception:
            failed.append(name)
            if verbose:
                raise

    return {
        "pipeline": "dataset",
        "requested": datasets,
        "success": success,
        "failed": failed,
        "data_path": data_path,
    }
