"""
Corpus Indexer & Document Chunker for Veritas RAG Engine.
Handles ingestion of text/PDF files, sliding-window chunking with offset tracking,
and building/persisting BM25 and Dense indexes.
"""

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

from veritas.retrieval.sparse import BM25Retriever
from veritas.retrieval.dense import DenseRetriever
from veritas.retrieval.hybrid import HybridRetriever


@dataclass
class DocumentChunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    start_char: int
    end_char: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CorpusIndexer:
    """Manages document chunking, indexing, and persistent storage."""

    def __init__(
        self,
        chunk_size_words: int = 250,
        chunk_overlap_words: int = 50,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.chunk_size_words = chunk_size_words
        self.chunk_overlap_words = chunk_overlap_words
        self.embedding_model = embedding_model
        self.chunks: List[DocumentChunk] = []

    def chunk_document(
        self,
        text: str,
        doc_id: str,
        title: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """Split a document text into sliding-window word chunks with character offsets."""
        text = text.strip()
        if not text:
            return []

        words = text.split()
        if not words:
            return []

        chunks: List[DocumentChunk] = []
        step = max(1, self.chunk_size_words - self.chunk_overlap_words)

        for i in range(0, len(words), step):
            chunk_words = words[i : i + self.chunk_size_words]
            chunk_text = " ".join(chunk_words)

            # Locate character span within original text
            approx_start = max(0, text.find(chunk_words[0]))
            approx_end = approx_start + len(chunk_text)

            chunk_id = f"{doc_id}#c{len(chunks):04d}"
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    title=title or doc_id,
                    text=chunk_text,
                    start_char=approx_start,
                    end_char=approx_end,
                    metadata=metadata or {},
                )
            )

            if i + self.chunk_size_words >= len(words):
                break

        return chunks

    def ingest_text_file(self, filepath: Path) -> List[DocumentChunk]:
        """Read a text file and generate chunks."""
        filepath = Path(filepath)
        doc_id = filepath.stem
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception:
            return []

        # Extract title from first line if available
        lines = [l.strip() for l in content.splitlines() if l.strip()]
        title = lines[0] if lines else doc_id

        return self.chunk_document(content, doc_id=doc_id, title=title, metadata={"source_path": str(filepath)})

    def ingest_pdf_file(self, filepath: Path) -> List[DocumentChunk]:
        """Read a PDF file and generate chunks."""
        filepath = Path(filepath)
        doc_id = filepath.stem
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(filepath))
            pages_text = [page.extract_text() or "" for page in reader.pages]
            full_text = "\n\n".join(pages_text)
        except Exception:
            return []

        lines = [l.strip() for l in full_text.splitlines() if l.strip()]
        title = lines[0] if lines else doc_id

        return self.chunk_document(full_text, doc_id=doc_id, title=title, metadata={"source_path": str(filepath), "pages": len(pages_text)})

    def index_directory(
        self,
        directory_path: Path,
        max_docs: Optional[int] = None,
    ) -> List[DocumentChunk]:
        """Ingest all supported documents (.txt, .pdf) in a directory."""
        directory_path = Path(directory_path)
        all_chunks: List[DocumentChunk] = []

        files = sorted(list(directory_path.glob("*.txt")) + list(directory_path.glob("*.pdf")))
        if max_docs:
            files = files[:max_docs]

        for file in files:
            if file.suffix.lower() == ".txt":
                chunks = self.ingest_text_file(file)
            elif file.suffix.lower() == ".pdf":
                chunks = self.ingest_pdf_file(file)
            else:
                chunks = []
            all_chunks.extend(chunks)

        self.chunks = all_chunks
        return all_chunks

    def build_and_save_index(
        self,
        output_dir: Path,
        chunks: Optional[List[DocumentChunk]] = None,
    ) -> HybridRetriever:
        """Fit BM25 and Dense retrievers and save index to disk."""
        target_chunks = chunks or self.chunks
        if not target_chunks:
            raise ValueError("No document chunks to index.")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        chunk_dicts = [c.to_dict() for c in target_chunks]

        # 1. Fit & save BM25
        sparse = BM25Retriever()
        sparse.fit(chunk_dicts)
        sparse.save(output_dir / "bm25_index.json")

        # 2. Fit & save Dense
        dense = DenseRetriever(model_name=self.embedding_model)
        dense.fit(chunk_dicts)
        dense.save(output_dir / "dense_index")

        # 3. Save raw chunks
        with open(output_dir / "corpus_chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunk_dicts, f)

        return HybridRetriever(sparse_retriever=sparse, dense_retriever=dense)

    @classmethod
    def load_index(
        cls,
        index_dir: Path,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> HybridRetriever:
        """Load an existing HybridRetriever from disk."""
        index_dir = Path(index_dir)
        sparse = BM25Retriever.load(index_dir / "bm25_index.json")
        dense = DenseRetriever.load(index_dir / "dense_index")
        return HybridRetriever(sparse_retriever=sparse, dense_retriever=dense)
