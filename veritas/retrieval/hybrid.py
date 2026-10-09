"""
Hybrid Retriever combining Dense and Sparse retrievers using Reciprocal Rank Fusion (RRF).
"""

from typing import Dict, List, Optional, Any
from collections import defaultdict
from veritas.retrieval.sparse import BM25Retriever
from veritas.retrieval.dense import DenseRetriever


class HybridRetriever:
    """Hybrid Retriever utilizing Reciprocal Rank Fusion (RRF) over BM25 and Dense embeddings."""

    def __init__(
        self,
        sparse_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
        rrf_k: int = 60,
        sparse_weight: float = 0.5,
        dense_weight: float = 0.5,
    ):
        self.sparse = sparse_retriever
        self.dense = dense_retriever
        self.rrf_k = rrf_k
        self.sparse_weight = sparse_weight
        self.dense_weight = dense_weight

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_pool_size: int = 25,
    ) -> List[Dict[str, Any]]:
        """
        Execute hybrid search.
        Retrieves top candidate_pool_size from sparse and dense, fuses ranks via RRF,
        and returns top_k combined documents.
        """
        sparse_results = self.sparse.search(query, top_k=candidate_pool_size)
        dense_results = self.dense.search(query, top_k=candidate_pool_size)

        rrf_scores: Dict[str, float] = defaultdict(float)
        chunk_map: Dict[str, Dict[str, Any]] = {}

        # Process Sparse ranks
        for item in sparse_results:
            chunk_id = item.get("chunk_id", str(item.get("text", "")[:60]))
            chunk_map[chunk_id] = item
            rank = item.get("sparse_rank", candidate_pool_size)
            rrf_scores[chunk_id] += self.sparse_weight * (1.0 / (self.rrf_k + rank))

        # Process Dense ranks
        for item in dense_results:
            chunk_id = item.get("chunk_id", str(item.get("text", "")[:60]))
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = item
            else:
                chunk_map[chunk_id].update({
                    "dense_score": item.get("dense_score", 0.0),
                    "dense_rank": item.get("dense_rank", candidate_pool_size),
                })
            rank = item.get("dense_rank", candidate_pool_size)
            rrf_scores[chunk_id] += self.dense_weight * (1.0 / (self.rrf_k + rank))

        # Sort by RRF score descending
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        final_results = []
        for rank, (chunk_id, score) in enumerate(sorted_chunks, start=1):
            res = dict(chunk_map[chunk_id])
            res["rrf_score"] = float(score)
            res["final_rank"] = rank
            final_results.append(res)

        return final_results
