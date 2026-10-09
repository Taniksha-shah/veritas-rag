"""
Discrepancy Generator for Recursive Cross-Verification Feedback.
Synthesizes precise, actionable discrepancy explanations for claims
that failed verification, guiding the generator's refinement.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from veritas.verification.entailment import VerificationResult


@dataclass
class DiscrepancyReport:
    iteration: int
    total_claims: int
    supported_claims: int
    contradicted_claims: int
    unverified_claims: int
    is_fully_attested: bool
    failed_details: List[Dict[str, Any]] = field(default_factory=list)
    feedback_prompt: str = ""

    @property
    def attestation_rate(self) -> float:
        return (self.supported_claims / self.total_claims) if self.total_claims > 0 else 0.0

    @property
    def hallucination_rate(self) -> float:
        return ((self.contradicted_claims + self.unverified_claims) / self.total_claims) if self.total_claims > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "total_claims": self.total_claims,
            "supported_claims": self.supported_claims,
            "contradicted_claims": self.contradicted_claims,
            "unverified_claims": self.unverified_claims,
            "is_fully_attested": self.is_fully_attested,
            "attestation_rate": round(self.attestation_rate, 4),
            "hallucination_rate": round(self.hallucination_rate, 4),
            "failed_details": self.failed_details,
            "feedback_prompt": self.feedback_prompt,
        }


class DiscrepancyGenerator:
    """Constructs discrepancy reports and feedback prompts for the iterative refinement loop."""

    def generate_report(
        self,
        verification_results: List[VerificationResult],
        iteration: int = 1,
    ) -> DiscrepancyReport:
        """Analyze verification results and produce a discrepancy report."""
        total = len(verification_results)
        supported = [r for r in verification_results if r.status == "SUPPORTED"]
        contradicted = [r for r in verification_results if r.status == "CONTRADICTED"]
        unverified = [r for r in verification_results if r.status == "UNVERIFIED"]

        is_fully_attested = (len(contradicted) == 0 and len(unverified) == 0)

        failed_details = []
        feedback_items = []

        for r in contradicted:
            directive = (
                f"- CONTRADICTION in claim: '{r.claim_text}'\n"
                f"  Evidence [{r.evidence_doc_id}] states: \"{r.evidence_quote}\"\n"
                f"  Correction: Remove or correct this assertion because the evidence contradicts it ({r.rationale})."
            )
            feedback_items.append(directive)
            failed_details.append({
                "claim_id": r.claim_id,
                "claim_text": r.claim_text,
                "status": "CONTRADICTED",
                "evidence_doc_id": r.evidence_doc_id,
                "evidence_quote": r.evidence_quote,
                "rationale": r.rationale,
                "directive": directive,
            })

        for r in unverified:
            directive = (
                f"- UNVERIFIED assertion: '{r.claim_text}'\n"
                f"  Discrepancy: The provided context does NOT contain verifiable evidence to attest this proposition.\n"
                f"  Correction: Excise this claim completely or qualify it strictly based only on attested source data."
            )
            feedback_items.append(directive)
            failed_details.append({
                "claim_id": r.claim_id,
                "claim_text": r.claim_text,
                "status": "UNVERIFIED",
                "evidence_doc_id": r.evidence_doc_id,
                "evidence_quote": "",
                "rationale": r.rationale,
                "directive": directive,
            })

        feedback_prompt = ""
        if not is_fully_attested:
            feedback_prompt = (
                "The previous draft contained factual discrepancies that failed verification:\n\n"
                + "\n\n".join(feedback_items)
                + "\n\nPlease rewrite the answer addressing every discrepancy above. "
                "Retain all supported facts, remove all contradicted or unverified assertions, "
                "and ensure every statement in your revision is strictly grounded in the provided context."
            )

        return DiscrepancyReport(
            iteration=iteration,
            total_claims=total,
            supported_claims=len(supported),
            contradicted_claims=len(contradicted),
            unverified_claims=len(unverified),
            is_fully_attested=is_fully_attested,
            failed_details=failed_details,
            feedback_prompt=feedback_prompt,
        )
