"""
Atomic Claim Decomposition Module for Veritas RAG Engine.
Decomposes generated drafts into self-contained, verifiable propositions.
"""

from veritas.decomposition.decomposer import ClaimDecomposer, AtomicClaim

__all__ = ["ClaimDecomposer", "AtomicClaim"]
