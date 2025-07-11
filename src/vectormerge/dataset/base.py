"""
Base dataset classes and abstract interfaces for VectorMerge.

This module contains the core dataset container and abstract loader interfaces.
"""

from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np

from loguru import logger


@dataclass
class Dataset:
    """Unified dataset container for VectorMerge."""
    
    name: str
    corpus: Dict[str, Dict[str, str]]  # doc_id -> {"title": str, "text": str}
    queries: Dict[str, str]  # query_id -> query_text
    qrels: Dict[str, Dict[str, int]]  # query_id -> {doc_id: relevance_score}
    # Additional metadata
    split: str = "test"
    source: str = "unknown"  # beir, custom, etc.
    corpus_ids2index: Dict[str, int] = field(default_factory=dict)
    index2corpus_id: Dict[int, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize corpus ID to index mapping."""
        if not self.corpus_ids2index:
            self.corpus_ids2index = {doc_id: i for i, doc_id in enumerate(self.corpus.keys())}
            self.index2corpus_id = {i: doc_id for doc_id, i in self.corpus_ids2index.items()}
        
        self.query_id2answer_ids: Dict[str, List[str]] = {query_id: [doc_id for doc_id, score in self.qrels[query_id].items() if score > 0] for query_id in self.qrels.keys()}
        self.query_index2answer_index = self._construct_query_index2answer_index(self.query_id2answer_ids)
    
    def _construct_query_index2answer_index(self, query_id2answer_ids: Dict[str, List[str]]):
        """Construct query index to answer index mapping."""
        query_index2answer_index = {}
        for i, query_id in enumerate(self.qrels.keys()):
            try:
                query_index2answer_index[i] = self.batch_original_ids_to_internal_indices(self.query_id2answer_ids[query_id]).tolist()
            except Exception as e:
                logger.error(f"Error constructing query index to answer index mapping for query {query_id}: {e}, skip this query")
        return query_index2answer_index
        

    
    def __len__(self) -> int:
        """Return number of documents in corpus."""
        return len(self.corpus)
    
    @property
    def num_queries(self) -> int:
        """Return number of queries."""
        return len(self.queries)
    
    @property
    def num_qrels(self) -> int:
        """Return number of query-document relevance pairs."""
        return sum(len(docs) for docs in self.qrels.values())
    
    def get_corpus_texts(self, include_title: bool = True) -> List[str]:
        """Get all corpus texts as a list.
        
        Args:
            include_title: Whether to include document titles
            
        Returns:
            List of text strings
        """
        texts = []
        for doc_id, doc in self.corpus.items():
            if include_title:
                text = f"{doc.get('title', '')} {doc.get('text', '')}"
            else:
                text = doc.get('text', '')
            texts.append(text.strip())
        return texts
    
    def get_query_texts(self) -> List[str]:
        """Get all query texts as a list.
        
        Returns:
            List of query strings
        """
        return list(self.queries.values())
    
    def get_dataset_index(self) -> List[str]:
        """Get original document IDs.
        
        Returns:
            List of original document IDs (strings)
        """
        return list(self.corpus.keys())
    
    def get_internal_index(self) -> np.ndarray:
        """Get internal 0-based indices.
        
        Returns:
            Array of internal indices from 0 to len(corpus)-1
        """
        return np.arange(len(self.corpus))
    
    def original_id_to_internal_index(self, doc_id: str) -> int:
        """Convert original document ID to internal index.
        
        Args:
            doc_id: Original document ID
            
        Returns:
            Internal 0-based index
            
        Raises:
            KeyError: If document ID not found
        """
        if doc_id not in self.corpus_ids2index:
            raise KeyError(f"Document ID '{doc_id}' not found in corpus")
        return self.corpus_ids2index[doc_id]
    
    def internal_index_to_original_id(self, index: int) -> str:
        """Convert internal index to original document ID.
        
        Args:
            index: Internal 0-based index
            
        Returns:
            Original document ID
            
        Raises:
            IndexError: If index out of range
        """
        if index < 0 or index >= len(self.corpus):
            raise IndexError(f"Index {index} out of range [0, {len(self.corpus)})")
        if index not in self.index2corpus_id:
            raise IndexError(f"Index {index} not found in mapping")
        return self.index2corpus_id[index]
    
    def batch_original_ids_to_internal_indices(self, doc_ids: List[str]) -> np.ndarray:
        """Convert batch of original document IDs to internal indices.
        
        Args:
            doc_ids: List of original document IDs
            
        Returns:
            Array of internal indices
        """
        return np.array([self.original_id_to_internal_index(doc_id) for doc_id in doc_ids])
    
    def batch_internal_indices_to_original_ids(self, indices: np.ndarray) -> List[str]:
        """Convert batch of internal indices to original document IDs.
        
        Args:
            indices: Array of internal indices
            
        Returns:
            List of original document IDs
        """
        return [self.internal_index_to_original_id(int(idx)) for idx in indices]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics.
        
        Returns:
            Dictionary containing dataset statistics
        """
        return {
            "name": self.name,
            "source": self.source,
            "split": self.split,
            "num_docs": len(self.corpus),
            "num_queries": self.num_queries,
            "num_qrels": self.num_qrels,
            "avg_doc_length": np.mean([len(doc.get('text', '')) for doc in self.corpus.values()]) if self.corpus else 0,
            "avg_query_length": np.mean([len(q) for q in self.queries.values()]) if self.queries else 0,
        }
    
    def filter_corpus(self, doc_ids: List[str]) -> 'Dataset':
        """Create a new dataset with filtered corpus.
        
        Args:
            doc_ids: List of document IDs to keep
            
        Returns:
            New Dataset instance with filtered corpus
        """
        filtered_corpus = {doc_id: self.corpus[doc_id] for doc_id in doc_ids if doc_id in self.corpus}
        
        return Dataset(
            name=f"{self.name}_filtered",
            corpus=filtered_corpus,
            queries=self.queries.copy(),
            qrels=self.qrels.copy(),
            split=self.split,
            source=self.source,
            metadata={**self.metadata, "filtered": True, "original_size": len(self.corpus)}
        )
    
    def get_relevant_docs(self, query_id: str) -> List[str]:
        """Get relevant document IDs for a query.
        
        Args:
            query_id: Query ID
            
        Returns:
            List of relevant document IDs
        """
        if query_id not in self.qrels:
            return []
        
        return [doc_id for doc_id, score in self.qrels[query_id].items() if score > 0]


class DatasetLoader(ABC):
    """Abstract base class for dataset loaders."""
    
    @abstractmethod
    def load(self, dataset_name: str, split: str = "test", **kwargs) -> Dataset:
        """Load a dataset.
        
        Args:
            dataset_name: Name of the dataset to load
            split: Dataset split to load
            **kwargs: Additional arguments
            
        Returns:
            Dataset object
        """
        pass
    
    @abstractmethod
    def list_available_datasets(self) -> List[str]:
        """List available datasets.
        
        Returns:
            List of available dataset names
        """
        pass
    
    def is_available(self, dataset_name: str) -> bool:
        """Check if a dataset is available.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            True if dataset is available
        """
        return dataset_name in self.list_available_datasets()


# Legacy compatibility
EmbeddingDataset = Dataset  # Alias for backward compatibility 