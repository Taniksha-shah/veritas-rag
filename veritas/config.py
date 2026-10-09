"""
Global Configuration and Settings for Veritas RAG Engine.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

# Automatically load environment variables from .env
load_dotenv()


@dataclass
class VeritasConfig:
    """Configuration class for Veritas RAG Engine components."""

    # Base Directories
    workspace_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    raw_papers_dir: Path = field(default_factory=lambda: Path("raw_papers"))
    index_dir: Path = field(default_factory=lambda: Path("veritas_index"))
    audit_log_dir: Path = field(default_factory=lambda: Path("audit_logs"))

    # API Keys & Endpoints
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    groq_base_url: str = "https://api.groq.com/openai/v1"

    # Generation Models
    # Recommended Groq models: llama-3.3-70b-versatile, llama-3.1-8b-instant, mixtral-8x7b-32768
    generator_model: str = "llama-3.3-70b-versatile"
    verifier_model: str = "llama-3.3-70b-versatile"
    fallback_generator_model: str = "llama-3.1-8b-instant"

    # Local Fallback Models
    local_decomposer_model: str = "google/flan-t5-small"
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Retrieval Hyperparameters
    chunk_size_words: int = 250
    chunk_overlap_words: int = 50
    top_k_retrieval: int = 5
    rrf_k: int = 60
    bm25_b: float = 0.75
    bm25_k1: float = 1.5

    # Recursive Verification Hyperparameters
    max_verification_iterations: int = 3
    entailment_confidence_threshold: float = 0.75
    strict_safety_pruning: bool = True  # If claim still unverified after max_iter, excise or qualify it

    # Citation Attribution Hyperparameters
    min_citation_overlap_score: float = 0.35
    citation_window_padding: int = 30  # characters around exact matched span

    def __post_init__(self):
        # Resolve relative directories against workspace_root
        if not self.raw_papers_dir.is_absolute():
            self.raw_papers_dir = self.workspace_root / self.raw_papers_dir
        if not self.index_dir.is_absolute():
            self.index_dir = self.workspace_root / self.index_dir
        if not self.audit_log_dir.is_absolute():
            self.audit_log_dir = self.workspace_root / self.audit_log_dir

        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.audit_log_dir.mkdir(parents=True, exist_ok=True)
