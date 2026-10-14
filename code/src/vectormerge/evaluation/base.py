import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional
from loguru import logger
import faiss


def get_retrieval_list(query_emb_2: np.ndarray, embeddings: np.ndarray, top_k: int, metric: str = "l2") -> np.ndarray:
    """Get the retrieval list using FAISS for efficient similarity search."""
    dim = embeddings.shape[1]

    if metric == "l2":
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings.astype(np.float32))
        distances, indices = index.search(query_emb_2.astype(np.float32), top_k)
        return indices
    elif metric == "cosine":
        embeddings_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_normalized = (embeddings / (embeddings_norms + 1e-8)).astype(np.float32)

        query_norms = np.linalg.norm(query_emb_2, axis=1, keepdims=True)
        query_normalized = (query_emb_2 / (query_norms + 1e-8)).astype(np.float32)

        index = faiss.IndexFlatIP(dim)
        index.add(embeddings_normalized)
        distances, indices = index.search(query_normalized, top_k)
        return indices
    else:
        raise ValueError(f"Unsupported metric: {metric}")


class Evaluator:
    def __init__(self, corpus_emb_1: np.ndarray, corpus_emb_2: np.ndarray, query_emb_1: np.ndarray, query_emb_2: np.ndarray, query_index2answer_index: Dict[int, List[int]], d0: np.ndarray, d1: np.ndarray, d2: np.ndarray, corpus_emb_1_transformed: np.ndarray, k_list: List[int] = [10, 50, 100, 500, 1000], source_side: Optional[np.ndarray] = None):
        self.corpus_emb_1 = corpus_emb_1
        self.corpus_emb_2 = corpus_emb_2
        self.query_emb_1 = query_emb_1
        self.query_emb_2 = query_emb_2
        self.query_index2answer_index = query_index2answer_index
        for key, value in query_index2answer_index.items():
                self.query_index2answer_index[key] = [value[0]]
        self.d0 = d0
        self.d1 = d1
        self.d2 = d2
        self.source_side = d1 if source_side is None else source_side
        self.corpus_emb_1_transformed = corpus_emb_1_transformed
        self.k_list = k_list
        self._align_dimensions()

    @staticmethod
    def get_rank_of_answer(query_emb: np.ndarray, answer_indices: List[int],
                          corpus_embeddings: np.ndarray, metric: str = "l2") -> int:
        """Get the rank of the first answer document for a given query."""
        if not answer_indices:
            return len(corpus_embeddings) + 1

        if metric == "l2":
            distances = np.linalg.norm(corpus_embeddings - query_emb, axis=1)
            sorted_indices = np.argsort(distances)
        elif metric == "cosine":
            query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
            corpus_norms = np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
            corpus_normalized = corpus_embeddings / (corpus_norms + 1e-8)

            similarities = np.dot(corpus_normalized, query_norm.T).flatten()
            sorted_indices = np.argsort(-similarities)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        for rank, doc_idx in enumerate(sorted_indices, 1):
            if doc_idx in answer_indices:
                return rank

        return len(corpus_embeddings) + 1

    def _align_dimensions(self):
        max_dim = max(self.corpus_emb_1.shape[1], self.corpus_emb_2.shape[1], self.corpus_emb_1_transformed.shape[1], self.query_emb_2.shape[1])
        self.corpus_emb_1 = np.pad(self.corpus_emb_1, ((0, 0), (0, max_dim - self.corpus_emb_1.shape[1])), mode="constant", constant_values=0)
        self.corpus_emb_2 = np.pad(self.corpus_emb_2, ((0, 0), (0, max_dim - self.corpus_emb_2.shape[1])), mode="constant", constant_values=0)
        self.corpus_emb_1_transformed = np.pad(self.corpus_emb_1_transformed, ((0, 0), (0, max_dim - self.corpus_emb_1_transformed.shape[1])), mode="constant", constant_values=0)
        self.query_emb_2 = np.pad(self.query_emb_2, ((0, 0), (0, max_dim - self.query_emb_2.shape[1])), mode="constant", constant_values=0)

    def evaluate(self) -> Dict:
        """Main evaluation method that computes all metrics."""
        results = {}

        recall_results = self.merge_and_evaluate_embeddings(
            self.corpus_emb_1, self.corpus_emb_2, self.corpus_emb_1_transformed,
            self.query_emb_2, self.query_index2answer_index, self.d0, self.d1, self.d2
        )
        results.update(recall_results)

        standard_recall_results = self._evaluate_recall_at_k()
        results.update(standard_recall_results)

        d1_recall_results = self._evaluate_retrieval_D1()
        results.update(d1_recall_results)

        relative_ndcg_results = self._evaluate_relative_ndcg()
        results.update(relative_ndcg_results)

        beta_results = self._evaluate_beta()
        results.update(beta_results)

        return results

    def _evaluate_retrieval_D1(self) -> Dict[str, float]:
        """Evaluate Recall@k[D1] - recall considering only queries where the top answer is in D1."""
        logger.info("Computing Recall@k[D1] metrics...")

        d1_set = set(self.d1.tolist())
        d1_queries = {qi: ans for qi, ans in self.query_index2answer_index.items()
                      if any(a in d1_set for a in ans)}

        if not d1_queries:
            logger.warning("No queries found with answers in D1")
            return {f"recall_d1@{k}": 0.0 for k in self.k_list}

        ranks = self._ranks_against(self._integrated_db(), d1_queries)
        return {f"recall_d1@{k}": sum(r <= k for r in ranks.values()) / len(d1_queries)
                for k in self.k_list}

    def _integrated_db(self) -> np.ndarray:
        """The target vectors with the source-side rows replaced by their mapped counterparts."""
        if getattr(self, "_integrated_db_cache", None) is None:
            db = self.corpus_emb_2.copy()
            db[self.source_side] = self.corpus_emb_1_transformed[self.source_side]
            self._integrated_db_cache = db
        return self._integrated_db_cache

    def _ranks_against(self, corpus: np.ndarray, queries: Dict[int, List[int]]) -> Dict[int, int]:
        """Rank of the best-placed answer of each query, 1-indexed."""
        corpus_sq = np.einsum("ij,ij->i", corpus, corpus)
        ranks = {}
        for query_idx, answer_indices in queries.items():
            if not answer_indices:
                continue
            scores = corpus_sq - 2.0 * (corpus @ self.query_emb_2[query_idx])
            best = scores[answer_indices].min()
            ranks[query_idx] = int((scores < best).sum()) + 1
        return ranks

    def _evaluate_relative_ndcg(self) -> Dict[str, float]:
        """Evaluate Relative NDCG - measures relative rank changes compared to ground truth."""
        logger.info("Computing Relative NDCG metrics...")

        merge_ranks = self._ranks_against(self._integrated_db(), self.query_index2answer_index)
        star_ranks = self._ranks_against(self.corpus_emb_2, self.query_index2answer_index)

        scores = []
        for query_idx, r_merge in merge_ranks.items():
            r_star = star_ranks[query_idx]

            if r_merge == r_star:
                score = 1.0
            else:
                score = 1.0 / np.log2(abs(r_merge - r_star) + 1)

            scores.append(score)

        if not scores:
            logger.warning("No valid queries for relative NDCG computation")
            return {"relative_ndcg": 0.0}

        relative_ndcg = np.mean(scores)
        return {"relative_ndcg": float(relative_ndcg)}

    def _evaluate_beta(self) -> Dict[str, float]:
        """Evaluate integration error β - measures distance between transformed and target embeddings."""
        logger.info("Computing integration error β...")


        d1_emb_1_transformed = self.corpus_emb_1_transformed[self.d1]
        d1_emb_2 = self.corpus_emb_2[self.d1]

        distances = np.linalg.norm(d1_emb_1_transformed - d1_emb_2, axis=1)

        beta = np.mean(distances)
        beta_std = np.std(distances)

        return {
            "integration_error_beta": float(beta),
            "integration_error_beta_std": float(beta_std),
            "integration_error_beta_min": float(np.min(distances)),
            "integration_error_beta_max": float(np.max(distances))
        }

    def _evaluate_recall_at_k(self) -> Dict[str, float]:
        """Evaluate standard Recall@k - measures if the top answer is retrieved within top-k."""
        logger.info("Computing standard Recall@k metrics...")

        ranks = self._ranks_against(self._integrated_db(), self.query_index2answer_index)
        total_queries = len(self.query_index2answer_index)

        return {f"recall_at_{k}": sum(r <= k for r in ranks.values()) / total_queries
                for k in self.k_list}

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
        """Merge embedding spaces and evaluate retrieval performance."""
        non_ref_indices = self.source_side

        merged_embeddings = {
            "target_only": corpus_emb_2.copy(),
            "direct_concat": corpus_emb_2.copy(),
            "our_method": corpus_emb_2.copy()
        }

        merged_embeddings["direct_concat"][non_ref_indices] = corpus_emb_1[non_ref_indices]

        merged_embeddings["our_method"][non_ref_indices] = corpus_emb_1_transformed[non_ref_indices]

        k_values = [10, 50, 100, 500, 1000]

        retrieval_results = {
            method: get_retrieval_list(query_emb_2, embeddings, top_k=max(k_values))
            for method, embeddings in merged_embeddings.items()
        }

        total_queries = len(query_index2answer_index)
        recalls = {}
        for method, results in retrieval_results.items():
            hits = {k: 0 for k in k_values}
            for query_idx, answer_indices in query_index2answer_index.items():
                if not answer_indices:
                    continue
                answer_idx = answer_indices[0]
                positions = np.flatnonzero(results[query_idx] == answer_idx)
                if positions.size == 0:
                    continue
                rank = int(positions[0]) + 1
                for k in k_values:
                    if rank <= k:
                        hits[k] += 1
            recalls.update({f"{method}@{k}": hits[k] / total_queries for k in k_values})

        return recalls
