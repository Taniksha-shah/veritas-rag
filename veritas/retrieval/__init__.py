"""
Retrieval Module for Veritas RAG Engine.
Includes Sparse (BM25), Dense (Transformer embeddings), and Hybrid (RRF) Retrievers.
"""

from veritas.retrieval.sparse import BM25Retriever
from veritas.retrieval.dense import DenseRetriever
from veritas.retrieval.hybrid import HybridRetriever
from veritas.retrieval.indexer import CorpusIndexer, DocumentChunk

__all__ = [
    "BM25Retriever",
    "DenseRetriever",
    "HybridRetriever",
    "CorpusIndexer",
    "DocumentChunk",
]
