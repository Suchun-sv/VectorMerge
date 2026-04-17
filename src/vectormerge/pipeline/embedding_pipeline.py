"""Embedding pipeline entrypoint."""

from typing import Any, Dict, List, Optional

from vectormerge.dataset import SUPPORTED_DATASETS
from vectormerge.embeddings import SUPPORTED_MODELS
from vectormerge.embeddings.generate import (
    generate_embeddings as generate_embeddings_fn,
)


def embedding_pipeline(
    config: Any,
    model: str,
    dataset: str,
    type_: Optional[str] = None,
    force: bool = False,
    upload: bool = True,
) -> Dict[str, Any]:
    """Generate embeddings through a pipeline boundary."""

    models: List[str] = SUPPORTED_MODELS if model == "all" else [model]
    datasets: List[str] = SUPPORTED_DATASETS if dataset == "all" else [dataset]
    types = [type_] if type_ else ["corpus", "query"]

    total = 0
    errors: List[Dict[str, str]] = []
    for m in models:
        for d in datasets:
            settings = config.embedding_models.load_model_settings(m)
            for t in types:
                total += 1
                try:
                    generate_embeddings_fn(
                        model_name=m,
                        dataset_name=d,
                        dataset_path=str(config.data_path),
                        cache_dir=str(config.embedding_cache_path),
                        model_settings=settings,
                        type_=t,
                        force=force,
                        embedding_path=str(config.embedding_path),
                        upload=upload,
                    )
                except Exception as exc:
                    errors.append(
                        {"model": m, "dataset": d, "type": t, "error": str(exc)}
                    )

    return {
        "pipeline": "embedding",
        "total_jobs": total,
        "errors": errors,
        "success_jobs": total - len(errors),
    }
