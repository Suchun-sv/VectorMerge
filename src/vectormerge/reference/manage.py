from rich import text
from vectormerge.reference import SplitConfig, LA2MSplitConfig
from vectormerge.embedding_dataset import EmbeddingDataset
from vectormerge.reference import LA2MSplit, RandomSplit
import numpy as np


def get_split_config(split_method_name: str, **kwargs) -> 'SplitConfig':
    if split_method_name == "la2m":
        return LA2MSplitConfig().update(**kwargs)
    elif split_method_name == "random":
        return SplitConfig().update(**kwargs)
    else:
        raise ValueError(f"Invalid split method: {split_method_name}")

def split_dataset(dataset: EmbeddingDataset, reference_method: str, update_split_config: dict = {}) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split_config = get_split_config(reference_method, **update_split_config)
    text_dataset = dataset.text_dataset
    if reference_method == "la2m":
        splitter = LA2MSplit.from_qrels(
            dataset_name=dataset.text_dataset_name,
            dataset_index=text_dataset.get_dataset_index(),
            internal_index=text_dataset.get_internal_index(),
            dataset_obj=text_dataset,
            qrels=text_dataset.qrels,
            reference_ratio=split_config.reference_ratio,
            reference_path=split_config.reference_path,
            remove_dup_answer=split_config.remove_dup_answer,
            select_top_1=split_config.select_top_1
        )
        return splitter.split(save=split_config.save, verbose=split_config.verbose)
    elif reference_method == "random":
        splitter = RandomSplit(
            dataset_name=dataset.text_dataset_name,
            internal_index=text_dataset.get_internal_index(),
            reference_ratio=split_config.reference_ratio,
            reference_path=split_config.reference_path
        )
        return splitter.split(save=split_config.save, verbose=split_config.verbose)
    else:
        raise ValueError(f"Invalid split method: {reference_method}")