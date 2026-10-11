"""
Dense Vector Retriever using Hugging Face Transformer Embeddings.
Supports mean pooling, cosine similarity ranking, disk serialization,
and resilient offline fallback.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class DenseRetriever:
    """Dense retriever using PyTorch + Hugging Face Transformers."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None
        self.embeddings: Optional[np.ndarray] = None
        self.corpus: List[Dict[str, Any]] = []
        self._is_initialized = False
        self._fallback_tfidf = None

    def _lazy_init(self) -> bool:
        """Lazily initialize the model and tokenizer."""
        if self._is_initialized:
            return True

        try:
            import torch
            from transformers import AutoTokenizer, AutoModel

            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            self._is_initialized = True
            return True
        
        except Exception as e:
            # Fall back to TF-IDF and fit it on the stored corpus.
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._fallback_tfidf = TfidfVectorizer(
                max_features=512,
                stop_words="english"
            )

            # Critical: a loaded retriever must fit the fallback
            # vectorizer before transforming a query.
            if self.corpus:
                texts = [
                    chunk.get("text", "")
                    for chunk in self.corpus
                ]
                if any(text.strip() for text in texts):
                    self._fallback_tfidf.fit(texts)

            self._is_initialized = True
            return False

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts into L2-normalized float32 vectors."""
        self._lazy_init()

        if self.model is not None and self.tokenizer is not None:
            import torch

            all_embeddings = []
            batch_size = 32

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt"
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**encoded)
                    # Mean pooling with attention mask
                    token_embeddings = outputs.last_hidden_state
                    attention_mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
                    sum_embeddings = torch.sum(token_embeddings * attention_mask, dim=1)
                    sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
                    pooled = sum_embeddings / sum_mask
                    # Normalize
                    normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
                    all_embeddings.append(normalized.cpu().numpy())

            if all_embeddings:
                return np.vstack(all_embeddings).astype(np.float32)
            return np.zeros((0, 384), dtype=np.float32)

        # Fallback to TF-IDF dense vectors
        if self._fallback_tfidf is not None:
            sparse_mat = self._fallback_tfidf.transform(texts)
            dense_mat = sparse_mat.toarray().astype(np.float32)
            # Normalize
            norms = np.linalg.norm(dense_mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return dense_mat / norms

        return np.zeros((len(texts), 128), dtype=np.float32)

    def fit(self, corpus_chunks: List[Dict[str, Any]]) -> "DenseRetriever":
        """Compute and store embeddings for all corpus chunks."""
        self.corpus = corpus_chunks
        texts = [chunk.get("text", "") for chunk in corpus_chunks]

        self._lazy_init()
        if self._fallback_tfidf is not None:
            # Fit TF-IDF on corpus
            self._fallback_tfidf.fit(texts)

        self.embeddings = self._embed_texts(texts)
        return self

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search the corpus using cosine similarity against query embedding."""
        if self.embeddings is None or len(self.corpus) == 0:
            return []

        query_vec = self._embed_texts([query])  # Shape (1, dim)
        if query_vec.shape[0] == 0:
            return []

        # Cosine similarity is simply dot product because vectors are L2-normalized
        sims = np.dot(self.embeddings, query_vec.T).squeeze(-1)  # Shape (N,)

        top_indices = np.argsort(-sims)[:top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            chunk = dict(self.corpus[idx])
            chunk["dense_score"] = float(sims[idx])
            chunk["dense_rank"] = rank
            results.append(chunk)

        return results

    def save(self, filepath_prefix: Path) -> None:
        """Save embeddings (.npy) and metadata (.json)."""
        filepath_prefix = Path(filepath_prefix)
        filepath_prefix.parent.mkdir(parents=True, exist_ok=True)

        if self.embeddings is not None:
            np.save(str(filepath_prefix) + "_emb.npy", self.embeddings)

        meta = {
            "model_name": self.model_name,
            "corpus": self.corpus,
            "has_embeddings": self.embeddings is not None,
        }
        with open(str(filepath_prefix) + "_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)

    @classmethod
    def load(cls, filepath_prefix: Path) -> "DenseRetriever":
        """Load saved embeddings and metadata."""
        filepath_prefix = Path(filepath_prefix)
        meta_file = Path(str(filepath_prefix) + "_meta.json")
        emb_file = Path(str(filepath_prefix) + "_emb.npy")

        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        instance = cls(model_name=meta.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"))
        instance.corpus = meta.get("corpus", [])

        if emb_file.exists():
            instance.embeddings = np.load(str(emb_file))

        return instance
