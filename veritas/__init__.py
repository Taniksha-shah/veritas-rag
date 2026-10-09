"""
Veritas RAG Engine
Hallucination-Free Domain-Specific RAG with Recursive Citation Verification
"""

__version__ = "1.0.0"
__author__ = "Veritas AI Research Team"

from veritas.config import VeritasConfig
from veritas.pipeline.engine import VeritasEngine, VeritasResponse

__all__ = ["VeritasEngine", "VeritasResponse", "VeritasConfig"]
