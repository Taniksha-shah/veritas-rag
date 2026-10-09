"""
Verification & Textual Entailment Module for Veritas RAG Engine.
Evaluates claim-to-evidence support and synthesizes discrepancy feedback.
"""

from veritas.verification.entailment import EntailmentVerifier, VerificationResult
from veritas.verification.discrepancy import DiscrepancyGenerator, DiscrepancyReport

__all__ = [
    "EntailmentVerifier",
    "VerificationResult",
    "DiscrepancyGenerator",
    "DiscrepancyReport",
]
