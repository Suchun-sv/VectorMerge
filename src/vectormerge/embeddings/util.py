import numpy as np
from typing import Tuple
from loguru import logger

def align_dimension(emb_1: np.ndarray, emb_2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align the dimension of two embeddings. It not, we pad the shorter one with zeros.
    """
    if emb_1.shape[1] != emb_2.shape[1]:
        max_dim = max(emb_1.shape[1], emb_2.shape[1])
        logger.info(f"Aligning dimension of {emb_1.shape} and {emb_2.shape} to {max_dim}")
        emb_1 = np.pad(emb_1, ((0, 0), (0, max_dim - emb_1.shape[1])), mode="constant", constant_values=0)
        emb_2 = np.pad(emb_2, ((0, 0), (0, max_dim - emb_2.shape[1])), mode="constant", constant_values=0)

    assert emb_1.shape[0] == emb_2.shape[0], f"Source and target embeddings have different sample size: {emb_1.shape[0]} and {emb_2.shape[0]}"
    assert emb_1.shape[1] == emb_2.shape[1], f"Source and target embeddings have different dimension: {emb_1.shape[1]} and {emb_2.shape[1]}"

    return emb_1, emb_2