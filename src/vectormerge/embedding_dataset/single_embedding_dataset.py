from datasets import load_dataset, DatasetDict, Dataset
import numpy as np
from vectormerge.dataset import load_dataset, Dataset as RawDataset
from vectormerge.embeddings import get_embedding

class SingleEmbeddingDataset():
    """Single embedding dataset container for VectorMerge."""
    
    def __init__(self, text_dataset_name: str, embedding_model_name: str, dataset_path: str = "./data/raw/beir/", embedding_path: str = "./data/processed/embeddings/", load_text_dataset: bool = True):
        self.text_dataset_name = text_dataset_name
        if load_text_dataset:
            self.text_dataset = load_dataset(text_dataset_name, split="test", dataset_path=dataset_path)
        else:
            self.text_dataset = None
        self.embedding_model_name = embedding_model_name
        self.embedding: np.ndarray = get_embedding(text_dataset_name, embedding_model_name, embedding_path, type_="corpus", download=True)
