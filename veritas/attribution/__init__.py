"""
Citation Attribution Module for Veritas RAG Engine.
Maps each generated sentence to precise source document character spans.
"""

from veritas.attribution.citation import CitationAttributor, AttributedOutput, SentenceCitation

__all__ = ["CitationAttributor", "AttributedOutput", "SentenceCitation"]
