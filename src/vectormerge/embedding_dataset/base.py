from datasets import Dataset
import numpy as np
from vectormerge.dataset import load_dataset, Dataset as TextDataset
from vectormerge.embeddings import align_dimension as align_dimension_func
from vectormerge.reference import SplitConfig, LA2MSplitConfig, split_dataset, get_split_config
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
        self.source_embedding_dataset = SingleEmbeddingDataset(text_dataset_name, source_embedding_model_name, dataset_path, embedding_path, load_text_dataset=False)
        self.target_embedding_dataset = SingleEmbeddingDataset(text_dataset_name, target_embedding_model_name, dataset_path, embedding_path, load_text_dataset=False)

        if align_dimension:
            self.source_embeddings, self.target_embeddings = align_dimension_func(self.source_embedding_dataset.embedding, self.target_embedding_dataset.embedding)
        else:
            self.source_embeddings = self.source_embedding_dataset.embedding
            self.target_embeddings = self.target_embedding_dataset.embedding
        
        self.d0_index, self.d1_index, self.d2_index = split_dataset(self, reference_method, reference_config)
