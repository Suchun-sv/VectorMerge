from typing import Dict, Tuple, Optional, Union
import fasttext
import os
import logging
import torch
import tqdm
import argparse
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import shutil

from .hf_interface import download_embedding, upload_embedding
from .embedding_generator import get_embedding_generator
from .util import align_dimension
from ..dataset import load_dataset, Dataset

# Configure logging
from beir import LoggingHandler

logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    handlers=[LoggingHandler()]
)
logger = logging.getLogger(__name__)

QUERY_KEY_TEMPLATE = "query_embeddings_{model_name}_{dataset_name}.npy"
CORPUS_KEY_TEMPLATE = "corpus_embeddings_{model_name}_{dataset_name}.npy"

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

# DatasetLoader class removed - now using unified dataset.py loader

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

def _get_single_embedding(model_name: str, dataset_name: str, embedding_path: str, type_: str = "corpus", download: bool = True):
    """
    Simple function to get embeddings from a model and dataset.
    """
    if type_ == "corpus":
        cache_key = CORPUS_KEY_TEMPLATE.format(dataset_name=dataset_name, model_name=model_name)
    elif type_ == "query":
        cache_key = QUERY_KEY_TEMPLATE.format(dataset_name=dataset_name, model_name=model_name)
    else:
        raise ValueError(f"Invalid type: {type_}. Must be 'corpus' or 'query'.")
    
    if download:
        download_embedding(cache_key, Path(embedding_path))
    
    return np.load(embedding_path / Path(cache_key))

def get_embedding(dataset_name: str, model_name: str, embedding_path: str, target_model_name: Optional[str] = None, type_: str = "corpus", align: bool = True, download: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Get embeddings from a model and dataset.
    """
    source_embeddings = _get_single_embedding(model_name, dataset_name, embedding_path, type_=type_, download=download)
    if target_model_name is None:
        return source_embeddings
    if target_model_name is not None:
        target_embeddings = _get_single_embedding(target_model_name, dataset_name, embedding_path, type_=type_, download=download)
    if target_model_name is not None and align:
        source_embeddings, target_embeddings = align_dimension(source_embeddings, target_embeddings)
    
    assert source_embeddings.shape[0] == target_embeddings.shape[0], f"Source and target embeddings have different sample size: {source_embeddings.shape[0]} and {target_embeddings.shape[0]}"

    return source_embeddings, target_embeddings

def get_source_target_embeddings(dataset_name: str, embedding_path: str, source_model_name: str, target_model_name: str, type_: str = "corpus", align: bool = True):
    """
    Get source and target embeddings from a model and dataset.
    """
    source_embeddings = _get_single_embedding(source_model_name, dataset_name, embedding_path, type_=type_)
    target_embeddings = _get_single_embedding(target_model_name, dataset_name, embedding_path, type_=type_)
    if align:
        source_embeddings, target_embeddings = align_dimension(source_embeddings, target_embeddings)
    return source_embeddings, target_embeddings


def generate_embeddings(model_name: str, dataset_name: str, dataset_path: str, cache_dir: str, model_settings: Dict, type_: str = "corpus", force: bool = False, embedding_path: Optional[str] = None, upload: bool = False):
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
        if upload:
            logger.info(f"Uploading embeddings to Hugging Face: {final_path}")
            upload_embedding(final_path, final_path.name)
        return
    
    # Load dataset using unified loader
    dataset = load_dataset(dataset_name, split="test", data_path=dataset_path)
    
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
    
    if type_ == "corpus":
        # Get corpus texts using dataset method
        text_list = dataset.get_corpus_texts(include_title=True)
        
        # Generate embeddings in cache directory
        cache_key = CORPUS_KEY_TEMPLATE.format(dataset_name=dataset_name, model_name=model_name)
        
        embedding_generator.generate_embeddings(text_list, 
                                                cache_key=cache_key)
        
        logger.info(f"Successfully generated corpus embeddings for {dataset_name}")
    
    elif type_ == "query":
        # Get query texts using dataset method
        text_list = dataset.get_query_texts()
        
        # Generate embeddings in cache directory
        cache_key = QUERY_KEY_TEMPLATE.format(dataset_name=dataset_name, model_name=model_name)
        
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
            if upload:
                logger.info(f"Uploading embeddings to Hugging Face: {final_path}")
                upload_embedding(final_path, final_path.name)
            logger.info(f"✅ Embeddings saved to: {final_path}")
        else:
            logger.info(f"✅ Embeddings ready at: {final_path}")
    else:
        logger.error(f"❌ Cache file not found: {cache_file}")
        raise FileNotFoundError(f"Generated embeddings not found in cache: {cache_file}")