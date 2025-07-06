from typing import Dict, Tuple, Optional, List
import fasttext
import os
import logging
import torch
import tqdm
import argparse
from pathlib import Path
from dataclasses import dataclass
from beir.datasets.data_loader import GenericDataLoader
from beir import util as beir_util
from beir.retrieval import models
from beir import LoggingHandler
import numpy as np
import shutil

from .embedding_generator import get_embedding_generator

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    handlers=[LoggingHandler()]
)
logger = logging.getLogger(__name__)

@dataclass
class DatasetConfig:
    """Configuration for dataset loading and processing"""
    name: str
    path: str
    split: str = "test"
    base_url: str = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"

    @property
    def url(self) -> str:
        return f"{self.base_url}/{self.name}.zip"

class FastTextTrainer:
    """Handles FastText model training and text processing"""
    
    def __init__(self, work_dir: str):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
    def train_model(
        self, 
        corpus_path: str, 
        model_path: str,
        dim: int = 300,
        lr: float = 0.025,
        window_size: int = 5,
        epochs: int = 10,
        min_count: int = 1
    ) -> None:
        """
        Train a FastText model with specified parameters
        
        Args:
            corpus_path: Path to the input corpus file
            model_path: Path to save the trained model
            dim: Embedding dimension
            lr: Learning rate
            window_size: Context window size
            epochs: Number of training epochs
            min_count: Minimum word count threshold
        """
        model = fasttext.train_unsupervised(
            corpus_path,
            model='skipgram',
            dim=dim,
            lr=lr,
            ws=window_size,
            epoch=epochs,
            minCount=min_count
        )
        model.save_model(model_path)
        logger.info(f"Model saved to {model_path}")

    def generate_embeddings(self, model_path: str, corpus: Dict, embeddings_path: str) -> None:
        """
        Generate embeddings for a given corpus using a trained FastText model
        
        Args:
            model_path: Path to the trained FastText model
            corpus: Dictionary containing document IDs and their content
            embeddings_path: Path to save the generated embeddings
        """
        model = fasttext.load_model(model_path)
        embeddings = []
        for doc_id, doc in tqdm.tqdm(corpus.items()):
            text = f"{doc.get('title', '')} {doc.get('text', '')}"
            embedding = model.get_word_vector(text.strip())
            embeddings.append(embedding)
        np.save(embeddings_path, np.array(embeddings))
        logger.info(f"Embeddings saved to {embeddings_path}")

class DatasetLoader:
    """Handles dataset loading and preprocessing"""
    
    SUPPORTED_DATASETS = {
        "scifact": "train",
        "nfcorpus": "test",
        "cqadupstack": "test",
        "arguana": "test",
        "scidocs": "test",
        "fiqa": "test",
        "signal1m": "test",
        "fever": "test"
    }

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)

    def load_dataset(self, dataset_name: str, split: str = "test") -> Tuple[Dict, Dict, Dict]:
        """
        Load and prepare a dataset for training
        
        Args:
            dataset_name: Name of the dataset to load
            split: Split of the dataset to load
            
        Returns:
            Tuple containing corpus, queries, and qrels
            
        Raises:
            ValueError: If dataset is not supported
        """
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(f"Dataset {dataset_name} is not supported. "
                           f"Supported datasets: {list(self.SUPPORTED_DATASETS.keys())}")

        config = DatasetConfig(
            name=dataset_name,
            path=str(self.dataset_path),
            split=split
        )
        
        logger.info(f"Loading dataset: {dataset_name} from {config.path}")
        
        data_path = self._prepare_dataset_path(config)
        data_loader = GenericDataLoader(data_folder=data_path)
        return data_loader.load(split=config.split)

    def _prepare_dataset_path(self, config: DatasetConfig) -> str:
        """Prepare the dataset path and handle special cases"""
        out_dir = self.dataset_path
        if not out_dir.exists():
            raise ValueError(f"Dataset directory {out_dir} does not exist.")
            
        data_path = beir_util.download_and_unzip(config.url, str(out_dir))
        
        if config.name == "cqadupstack":
            return os.path.join(data_path, "english")
        return data_path

def clean_corpus(corpus: Dict[str, Dict[str, str]]) -> str:
    """
    Clean and prepare corpus for training
    
    Args:
        corpus: Dictionary containing document IDs and their content
        
    Returns:
        Cleaned corpus as a single string
    """
    cleaned_texts = []
    for doc_id, doc in corpus.items():
        text = f"{doc.get('title', '')} {doc.get('text', '')}"
        cleaned_texts.append(text.strip())
    return "\n".join(cleaned_texts)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate embeddings for datasets')
    
    # Required arguments
    # parser.add_argument('--model', type=str, default="nv-embed", help='Name of the embedding model to use (e.g., fasttext, sbert)')
    parser.add_argument('--model', type=str, default="mistral", help='Name of the embedding model to use (e.g., fasttext, sbert)')
    # parser.add_argument('--datasets', type=str, nargs='+', default="fever",
    #                   help='List of dataset names to process')
    parser.add_argument('--datasets', type=str, nargs='+', default="scifact",
                      help='List of dataset names to process')
    
    # Optional arguments
    parser.add_argument('--dataset-path', type=str, default='data/raw/beir/',
                      help='Path to the raw datasets directory')
    parser.add_argument('--output-dir', type=str, default='data/processed/embeddings/',
                      help='Directory to save the generated embeddings')
    
    # Model-specific arguments
    parser.add_argument('--model-path', type=str,
                      help='Path to the pre-trained model (if required)')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for embedding generation')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use for computation (cuda/cpu)')
    
    # FastText specific arguments
    parser.add_argument('--dim', type=int, default=300,
                      help='Dimension of embeddings (for FastText)')
    
    return parser.parse_args()


def generate_embeddings(model_name: str, dataset_name: str, dataset_path: str, cache_dir: str, model_settings: Dict, type_: str = "corpus", force: bool = False, embedding_path: Optional[str] = None):
    """
    Generate embeddings for a dataset using specified model.
    
    Args:
        model_name: Name of the embedding model
        dataset_name: Name of the dataset
        dataset_path: Path to the dataset files
        cache_dir: Directory for temporary cache files
        model_settings: Model configuration settings
        type_: Type of embeddings to generate ("corpus" or "query")
        force: Whether to force regeneration of embeddings
        embedding_path: Final path to store embeddings (if None, uses cache_dir)
    """
    # Create output directory if it doesn't exist
    output_dir = Path(cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up embedding path
    if embedding_path is None:
        final_embedding_path = Path(cache_dir)
    else:
        final_embedding_path = Path(embedding_path)
        final_embedding_path.mkdir(parents=True, exist_ok=True)
    
    # Generate final output filename
    final_filename = f"{type_}_embeddings_{model_name}_{dataset_name}.npy"
    final_path = final_embedding_path / final_filename
    
    # Check if final embedding already exists
    if final_path.exists() and not force:
        logger.info(f"Embeddings already exist at {final_path}. Use --force to regenerate.")
        return
    
    # Initialize dataset loader
    dataset_loader = DatasetLoader(dataset_path)
    
    # Get embedding generator with force parameter
    embedding_generator = get_embedding_generator(
        model_name=model_name,
        dataset_name=dataset_name,
        cache_dir=str(output_dir),
        model_settings=model_settings,
        force=force
    )
    
    # Process dataset
    logger.info(f"Processing dataset: {dataset_name} for {type_} embeddings")
        
    corpus, queries, qrels = dataset_loader.load_dataset(dataset_name, split="test")
    
    if type_ == "corpus":
        # Prepare text list for corpus
        text_list = []
        for doc_id, doc in tqdm.tqdm(corpus.items(), desc="Preparing corpus texts"):
            text = f"{doc.get('title', '')} {doc.get('text', '')}"
            text_list.append(text)
        
        # Generate embeddings in cache directory
        cache_key = f"corpus_embeddings_{model_name}_{dataset_name}.npy"
        
        embedding_generator.generate_embeddings(text_list, 
                                                cache_key=cache_key)
        
        logger.info(f"Successfully generated corpus embeddings for {dataset_name}")
    
    elif type_ == "query":
        # Prepare text list for queries
        text_list = []
        for query_id, query in tqdm.tqdm(queries.items(), desc="Preparing query texts"):
            text_list.append(query)
        
        # Generate embeddings in cache directory
        cache_key = f"query_embeddings_{model_name}_{dataset_name}.npy"
        
        embedding_generator.generate_embeddings(text_list, 
                                                cache_key=cache_key,
                                                type="query")
        
        logger.info(f"Successfully generated query embeddings for {dataset_name}")
    
    else:
        raise ValueError(f"Invalid type: {type_}. Must be 'corpus' or 'query'.")
    
    # Copy from cache to final location
    cache_file = output_dir / cache_key
    if cache_file.exists():
        if str(final_embedding_path) != str(output_dir):
            logger.info(f"Copying embeddings from cache to final location: {final_path}")
            shutil.copy2(cache_file, final_path)
            logger.info(f"✅ Embeddings saved to: {final_path}")
        else:
            logger.info(f"✅ Embeddings ready at: {final_path}")
    else:
        logger.error(f"❌ Cache file not found: {cache_file}")
        raise FileNotFoundError(f"Generated embeddings not found in cache: {cache_file}")