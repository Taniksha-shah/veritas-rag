"""
Sparse BM25 Retriever Implementation.
Pure Python, dependency-free, high-performance Okapi BM25 with inverted index.
"""

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Standard English stopwords
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once",
    "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
    "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd",
    "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where",
    "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
    "yourselves"
}

TOKEN_PATTERN = re.compile(r"\b[a-zA-Z0-9_\-]{2,}\b")


def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    """Tokenize and normalize text."""
    tokens = [t.lower() for t in TOKEN_PATTERN.findall(text)]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


class BM25Retriever:
    """Okapi BM25 sparse retriever with inverted index for efficient ranking."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_count: int = 0
        self.avg_doc_len: float = 0.0
        self.doc_lengths: List[int] = []
        self.idf: Dict[str, float] = {}
        # Inverted index: token -> list of (doc_idx, tf)
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        # Stored metadata / passages
        self.corpus: List[Dict[str, Any]] = []

    def fit(self, corpus_chunks: List[Dict[str, Any]]) -> "BM25Retriever":
        """Build the inverted index from a list of document chunks."""
        self.corpus = corpus_chunks
        self.doc_count = len(corpus_chunks)
        self.doc_lengths = []
        self.inverted_index.clear()
        doc_frequencies: Dict[str, int] = defaultdict(int)

        total_words = 0
        for doc_idx, chunk in enumerate(corpus_chunks):
            text = chunk.get("text", "")
            tokens = tokenize(text)
            doc_len = len(tokens)
            self.doc_lengths.append(doc_len)
            total_words += doc_len

            term_counts = Counter(tokens)
            for token, tf in term_counts.items():
                self.inverted_index[token].append((doc_idx, tf))
                doc_frequencies[token] += 1

        self.avg_doc_len = total_words / self.doc_count if self.doc_count > 0 else 0.0

        # Compute Robertson-Spärck Jones IDF
        self.idf = {}
        for token, df in doc_frequencies.items():
            # Standard smooth BM25 IDF
            val = (self.doc_count - df + 0.5) / (df + 0.5) + 1.0
            self.idf[token] = math.log(val) if val > 0 else 0.0

        return self

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search the indexed corpus and return top-k matches with scores."""
        if not self.corpus or self.doc_count == 0:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores: Dict[int, float] = defaultdict(float)

        for token in query_tokens:
            if token not in self.inverted_index:
                continue

            idf = self.idf.get(token, 0.0)
            postings = self.inverted_index[token]

            for doc_idx, tf in postings:
                doc_len = self.doc_lengths[doc_idx]
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / (self.avg_doc_len or 1.0)))
                numerator = tf * (self.k1 + 1.0)
                scores[doc_idx] += idf * (numerator / (denominator or 1.0))

        # Rank documents by score descending
        ranked_indices = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

        results = []
        for rank, (doc_idx, score) in enumerate(ranked_indices, start=1):
            chunk = dict(self.corpus[doc_idx])
            chunk["sparse_score"] = float(score)
            chunk["sparse_rank"] = rank
            results.append(chunk)

        return results

    def save(self, filepath: Path) -> None:
        """Save the fitted BM25 model to disk."""
        data = {
            "k1": self.k1,
            "b": self.b,
            "doc_count": self.doc_count,
            "avg_doc_len": self.avg_doc_len,
            "doc_lengths": self.doc_lengths,
            "idf": self.idf,
            "inverted_index": {k: v for k, v in self.inverted_index.items()},
            "corpus": self.corpus,
        }
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, filepath: Path) -> "BM25Retriever":
        """Load a saved BM25 model from disk."""
        filepath = Path(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        instance = cls(k1=data["k1"], b=data["b"])
        instance.doc_count = data["doc_count"]
        instance.avg_doc_len = data["avg_doc_len"]
        instance.doc_lengths = data["doc_lengths"]
        instance.idf = data["idf"]
        instance.inverted_index = defaultdict(list, {k: [tuple(x) for x in v] for k, v in data["inverted_index"].items()})
        instance.corpus = data["corpus"]
        return instance
