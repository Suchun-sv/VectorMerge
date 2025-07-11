from datasets import Dataset
import numpy as np
from vectormerge.dataset import load_dataset, Dataset as TextDataset
from vectormerge.embeddings import align_dimension as align_dimension_func
from vectormerge.reference import SplitConfig, LA2MSplitConfig
from vectormerge.reference import LA2MSplit, RandomSplit
# from vectormerge.reference.manage import split_dataset, get_split_config
from .single_embedding_dataset import SingleEmbeddingDataset

class EmbeddingDataset(Dataset):
    """Single embedding dataset container for VectorMerge."""
    
    def __init__(self, text_dataset_name: str, source_embedding_model_name: str, target_embedding_model_name: str, align_dimension: bool, dataset_path: str = "./data/raw/beir/", embedding_path: str = "./data/processed/embeddings/", reference_method: str = "la2m", reference_config: dict = {}):
        """
        Embedding dataset container for VectorMerge.

        Args:
            text_dataset_name: Name of the text dataset.
            source_embedding_model_name: Name of the source embedding model.
            target_embedding_model_name: Name of the target embedding model.
            align_dimension: Whether to align the dimension of the source and target embeddings. If True, the dimension of the source and target embeddings will be aligned.
            dataset_path: Path to the raw dataset.
            embedding_path: Path to the processed embeddings.
            reference_method: Method for splitting the dataset. See vectormerge.reference.SUPPORTED_REFERENCE_STRATEGIES
            reference_config: Update the reference configuration. See vectormerge.reference.get_split_config
        """
        self.text_dataset_name = text_dataset_name
        self.text_dataset: TextDataset = load_dataset(text_dataset_name, split="test", dataset_path=dataset_path)
        self.source_embedding_dataset = SingleEmbeddingDataset(text_dataset_name, source_embedding_model_name, dataset_path, embedding_path, load_text_dataset=False, type_="corpus")
        self.target_embedding_dataset = SingleEmbeddingDataset(text_dataset_name, target_embedding_model_name, dataset_path, embedding_path, load_text_dataset=False, type_="corpus")

        self.source_query_embedding_dataset = SingleEmbeddingDataset(text_dataset_name, source_embedding_model_name, dataset_path, embedding_path, load_text_dataset=False, type_="query")
        self.target_query_embedding_dataset = SingleEmbeddingDataset(text_dataset_name, target_embedding_model_name, dataset_path, embedding_path, load_text_dataset=False, type_="query")

        if align_dimension:
            self.source_embeddings, self.target_embeddings = align_dimension_func(self.source_embedding_dataset.embedding, self.target_embedding_dataset.embedding)
            self.source_query_embeddings, self.target_query_embeddings = align_dimension_func(self.source_query_embedding_dataset.embedding, self.target_query_embedding_dataset.embedding)
        else:
            self.source_embeddings = self.source_embedding_dataset.embedding
            self.target_embeddings = self.target_embedding_dataset.embedding
            self.source_query_embeddings = self.source_query_embedding_dataset.embedding
            self.target_query_embeddings = self.target_query_embedding_dataset.embedding
        
        self.d0_index, self.d1_index, self.d2_index = split_dataset(self, reference_method, reference_config)




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
        return splitter.split(save=split_config.save)
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