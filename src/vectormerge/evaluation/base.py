import argparse
import numpy as np
from typing import Dict, List, Tuple
from loguru import logger
import faiss


def get_retrieval_list(query_emb_2: np.ndarray, embeddings: np.ndarray, top_k: int, metric: str = "l2") -> np.ndarray:
    """
    Get the retrieval list using FAISS for efficient similarity search.
    
    Args:
        query_emb_2: Query embeddings with shape (num_queries, embedding_dim)
        embeddings: Corpus embeddings with shape (num_docs, embedding_dim)
        top_k: Number of top documents to retrieve
        metric: Distance metric to use ("l2" or "cosine")
        
    Returns:
        np.ndarray: Array of retrieved document indices with shape (num_queries, top_k)
    """
    dim = embeddings.shape[1]
    
    if metric == "l2":
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings.astype(np.float32)) # type: ignore
        distances, indices = index.search(query_emb_2.astype(np.float32), top_k) # type: ignore
        return indices
    elif metric == "cosine":
        # For cosine similarity, normalize embeddings manually
        embeddings_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_normalized = (embeddings / (embeddings_norms + 1e-8)).astype(np.float32)
        
        query_norms = np.linalg.norm(query_emb_2, axis=1, keepdims=True)  
        query_normalized = (query_emb_2 / (query_norms + 1e-8)).astype(np.float32)
        
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings_normalized) # type: ignore
        distances, indices = index.search(query_normalized, top_k) # type: ignore
        return indices
    else:
        raise ValueError(f"Unsupported metric: {metric}")


class Evaluator:
    def __init__(self, corpus_emb_1: np.ndarray, corpus_emb_2: np.ndarray, query_emb_1: np.ndarray, query_emb_2: np.ndarray, query_index2answer_index: Dict[int, List[int]], d0: np.ndarray, d1: np.ndarray, d2: np.ndarray, corpus_emb_1_transformed: np.ndarray, k_list: List[int] = [10, 50, 100, 500, 1000]):
        self.corpus_emb_1 = corpus_emb_1
        self.corpus_emb_2 = corpus_emb_2
        self.query_emb_1 = query_emb_1
        self.query_emb_2 = query_emb_2
        self.query_index2answer_index = query_index2answer_index
        self.d0 = d0
        self.d1 = d1
        self.d2 = d2
        self.corpus_emb_1_transformed = corpus_emb_1_transformed
        self.k_list = k_list
        self._align_dimensions()
    
    def _align_dimensions(self):
        max_dim = max(self.corpus_emb_1.shape[1], self.corpus_emb_2.shape[1], self.corpus_emb_1_transformed.shape[1], self.query_emb_2.shape[1])
        self.corpus_emb_1 = np.pad(self.corpus_emb_1, ((0, 0), (0, max_dim - self.corpus_emb_1.shape[1])), mode="constant", constant_values=0)
        self.corpus_emb_2 = np.pad(self.corpus_emb_2, ((0, 0), (0, max_dim - self.corpus_emb_2.shape[1])), mode="constant", constant_values=0)
        self.corpus_emb_1_transformed = np.pad(self.corpus_emb_1_transformed, ((0, 0), (0, max_dim - self.corpus_emb_1_transformed.shape[1])), mode="constant", constant_values=0)
        self.query_emb_2 = np.pad(self.query_emb_2, ((0, 0), (0, max_dim - self.query_emb_2.shape[1])), mode="constant", constant_values=0)

    def evaluate(self) -> Dict:
        return self.merge_and_evaluate_embeddings(self.corpus_emb_1, self.corpus_emb_2, self.corpus_emb_1_transformed, self.query_emb_2, self.query_index2answer_index, self.d0, self.d1, self.d2)
    
    def get_retrieval_list(self, query_emb_2: np.ndarray, embeddings: np.ndarray, top_k: int, metric: str = "l2") -> np.ndarray:
        return get_retrieval_list(query_emb_2, embeddings, top_k=top_k, metric=metric)

    def merge_and_evaluate_embeddings(
        self,
        corpus_emb_1: np.ndarray,
        corpus_emb_2: np.ndarray,
        corpus_emb_1_transformed: np.ndarray,
        query_emb_2: np.ndarray,
        query_index2answer_index: Dict[int, List[int]],
        d0: np.ndarray,
        d1: np.ndarray,
        d2: np.ndarray,
    ) -> Dict:
        """
        Merge embedding spaces and evaluate retrieval performance.

        Args:
            corpus_emb_1: Embeddings from first model (source space)
            corpus_emb_2: Embeddings from second model (target space)
            corpus_emb_1_transformed: Transformed embeddings from first model
            query_emb_2: Query embeddings in target space
            query_index2answer_index: Dict mapping query index to list of answer indices
            d0: Reference set indices
            d1: Non-reference set indices for first model
            d2: Non-reference set indices for second model

        Returns:
            Dictionary containing recall metrics for different methods
        """
        non_ref_indices = d1

        merged_embeddings = {
            "target_only": corpus_emb_2.copy(),  # Only use target space embeddings
            "direct_concat": corpus_emb_2.copy(),  # Directly concatenate source and target embeddings
            "our_method": corpus_emb_2.copy()  # Our translation method
        }
        
        # For direct concatenation, use original source embeddings for non-reference indices
        merged_embeddings["direct_concat"][non_ref_indices] = corpus_emb_1[non_ref_indices]
        
        # For our method, use transformed source embeddings for non-reference indices
        merged_embeddings["our_method"][non_ref_indices] = corpus_emb_1_transformed[non_ref_indices]

        recalls = {}
        for top_k in [10, 50, 100, 500, 1000]:
            retrieval_results = {
                method: get_retrieval_list(query_emb_2, embeddings, top_k=top_k)
                for method, embeddings in merged_embeddings.items()
            }

            D1_hit_counts = {method: 0 for method in merged_embeddings.keys()}
            D2_hit_counts = {method: 0 for method in merged_embeddings.keys()}
            hit_counts = {method: 0 for method in merged_embeddings.keys()}
            
            # Iterate through each query
            for query_idx, answer_indices in query_index2answer_index.items():
                for method, results in retrieval_results.items():
                    # Check if any answer is in the retrieved results for this query
                    retrieved_for_query = results[query_idx]
                    for answer_idx in answer_indices:
                        if answer_idx in retrieved_for_query:
                            hit_counts[method] += 1
                            if answer_idx in d1:
                                D1_hit_counts[method] += 1
                            if answer_idx in d2:
                                D2_hit_counts[method] += 1
                            break  # Count only one hit per query
                            
            total_queries = len(query_index2answer_index)
            recalls.update({method+f"@{top_k}": count / total_queries for method, count in hit_counts.items()})

        return recalls
