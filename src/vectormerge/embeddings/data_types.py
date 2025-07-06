from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List, Tuple
import numpy as np

@dataclass
class EmbeddingPairDataset:
    """A dataset containing a list of embedding pairs"""
    corpus_emb_1: np.ndarray
    corpus_emb_2: np.ndarray
    query_emb_1: np.ndarray
    query_emb_2: np.ndarray
    metadata: Optional[Any] = None  # For any additional dataset info


