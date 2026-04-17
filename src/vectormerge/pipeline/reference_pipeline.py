"""Reference pipeline entrypoint."""

from pathlib import Path
from typing import Any, Dict

from vectormerge.dataset import load_dataset
from vectormerge.reference import LA2MSplit, RandomSplit


def reference_pipeline(
    config: Any,
    dataset: str,
    strategy: str = "random",
    reference_ratio: float = 0.33,
    remove_duplicates: bool = False,
    select_top_1: bool = True,
    force: bool = False,
) -> Dict[str, Any]:
    """Create reference split through a pipeline boundary."""

    reference_path = Path(config.reference_path)
    reference_path.mkdir(parents=True, exist_ok=True)
    reference_file = reference_path / f"{dataset}_reference.npz"
    if reference_file.exists() and not force:
        return {
            "pipeline": "reference",
            "skipped": True,
            "reference_file": str(reference_file),
        }

    dataset_obj = load_dataset(dataset, data_path=Path(config.data_path))
    if strategy == "random":
        splitter = RandomSplit(
            dataset_name=dataset,
            internal_index=dataset_obj.get_internal_index(),
            reference_ratio=reference_ratio,
            reference_path=str(reference_path),
        )
    else:
        splitter = LA2MSplit.from_qrels(
            dataset_name=dataset,
            dataset_index=dataset_obj.get_dataset_index(),
            internal_index=dataset_obj.get_internal_index(),
            dataset_obj=dataset_obj,
            qrels=dataset_obj.qrels,
            reference_ratio=reference_ratio,
            reference_path=str(reference_path),
            remove_dup_answer=remove_duplicates,
            select_top_1=select_top_1,
        )

    d0, d1, d2 = splitter.split()
    return {
        "pipeline": "reference",
        "reference_file": str(reference_file),
        "d0_size": len(d0),
        "d1_size": len(d1),
        "d2_size": len(d2),
        "d0_indices": d0.tolist(),
        "d1_indices": d1.tolist(),
        "d2_indices": d2.tolist(),
    }
