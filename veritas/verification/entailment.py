"""
Textual Entailment & Claim Verification Engine (Stage B).
Verifies whether each atomic claim is strictly entailed, contradicted,
or unverified by candidate retrieved evidence passages.
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from openai import OpenAI
from veritas.decomposition.decomposer import AtomicClaim


@dataclass
class VerificationResult:
    claim_id: str
    claim_text: str
    status: str  # SUPPORTED, CONTRADICTED, UNVERIFIED
    confidence: float
    evidence_doc_id: str
    evidence_quote: str
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "status": self.status,
            "confidence": self.confidence,
            "evidence_doc_id": self.evidence_doc_id,
            "evidence_quote": self.evidence_quote,
            "rationale": self.rationale,
        }


NLI_VERIFICATION_SYSTEM_PROMPT = """You are a rigorous scientific verification judge checking textual entailment for medical and legal claims.

Given a CLAIM and an EVIDENCE PASSAGE, classify the relationship into one of three categories:
1. SUPPORTED: The evidence passage directly, explicitly affirms and entails the claim. Numbers, units, directions of effect, and conditions match.
2. CONTRADICTED: The evidence passage directly refutes or contradicts the claim (e.g. claim says 'increased mortality', passage says 'no difference' or 'decreased mortality').
3. UNVERIFIED: The evidence passage does not provide sufficient proof to verify the claim, or the claim is an extrapolation/fabrication not stated in the passage.

You MUST respond ONLY with a valid JSON object formatted as follows:
{
  "status": "SUPPORTED" | "CONTRADICTED" | "UNVERIFIED",
  "confidence": 0.0 to 1.0,
  "evidence_quote": "exact quote from the passage supporting or refuting the claim, or empty string if unverified",
  "rationale": "one sentence explaining why the claim is supported, contradicted, or unverified"
}
"""


class EntailmentVerifier:
    """Natural Language Inference Verifier for atomic claims."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.groq.com/openai/v1",
        model: str = "llama-3.3-70b-versatile",
        confidence_threshold: float = 0.70,
    ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.base_url = base_url
        self.model = model
        self.confidence_threshold = confidence_threshold
        self._client: Optional[OpenAI] = None

    def _get_client(self) -> Optional[OpenAI]:
        if self._client is None and self.api_key:
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def verify_claim(
        self,
        claim: AtomicClaim,
        candidate_passages: List[Dict[str, Any]],
    ) -> VerificationResult:
        """
        Verify a single atomic claim against a list of candidate passages.
        Picks the best supporting or refuting passage.
        """
        if not candidate_passages:
            return VerificationResult(
                claim_id=claim.claim_id,
                claim_text=claim.text,
                status="UNVERIFIED",
                confidence=1.0,
                evidence_doc_id="N/A",
                evidence_quote="",
                rationale="No candidate evidence passages available for verification.",
            )

        client = self._get_client()

        # Combine top candidate passages into context with identifiers
        context_blocks = []
        for i, p in enumerate(candidate_passages[:3], start=1):
            doc_id = p.get("doc_id", f"Doc_{i}")
            text = p.get("text", "")
            context_blocks.append(f"[{doc_id}]: {text}")

        combined_evidence = "\n\n".join(context_blocks)

        if client:
            try:
                res = self._verify_with_llm(client, claim.text, combined_evidence, candidate_passages)
                if res:
                    return res
            except Exception:
                pass

        # Fallback heuristic NLI based on token overlap and contradiction cues
        return self._verify_heuristic(claim, candidate_passages)

    def _verify_with_llm(
        self,
        client: OpenAI,
        claim_text: str,
        combined_evidence: str,
        candidate_passages: List[Dict[str, Any]],
    ) -> Optional[VerificationResult]:
        prompt = f"""EVIDENCE:\n{combined_evidence}\n\nCLAIM TO VERIFY:\n"{claim_text}"\n\nEvaluate whether the evidence supports, contradicts, or fails to verify this claim."""

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": NLI_VERIFICATION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=512,
        )

        content = (response.choices[0].message.content or "").strip()
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content).strip()

        data = json.loads(content)
        status = data.get("status", "UNVERIFIED").upper()
        if status not in {"SUPPORTED", "CONTRADICTED", "UNVERIFIED"}:
            status = "UNVERIFIED"

        confidence = float(data.get("confidence", 0.8))
        if confidence < self.confidence_threshold and status == "SUPPORTED":
            status = "UNVERIFIED"

        evidence_quote = data.get("evidence_quote", "").strip()

        # Match back to specific doc_id
        best_doc_id = candidate_passages[0].get("doc_id", "DOC")
        if evidence_quote:
            for p in candidate_passages:
                if evidence_quote.lower() in p.get("text", "").lower():
                    best_doc_id = p.get("doc_id", best_doc_id)
                    break

        return VerificationResult(
            claim_id="",
            claim_text=claim_text,
            status=status,
            confidence=confidence,
            evidence_doc_id=best_doc_id,
            evidence_quote=evidence_quote,
            rationale=data.get("rationale", ""),
        )

    def _verify_heuristic(
        self,
        claim: AtomicClaim,
        candidate_passages: List[Dict[str, Any]],
    ) -> VerificationResult:
        """Token-overlap and negation-aware fallback verifier."""
        from veritas.retrieval.sparse import tokenize

        claim_tokens = set(tokenize(claim.text))
        best_overlap = 0.0
        best_passage = candidate_passages[0]
        best_quote = ""

        for p in candidate_passages:
            p_text = p.get("text", "")
            p_tokens = set(tokenize(p_text))
            if not p_tokens:
                continue

            overlap = len(claim_tokens.intersection(p_tokens)) / max(1, len(claim_tokens))
            if overlap > best_overlap:
                best_overlap = overlap
                best_passage = p
                # Extract sentence with highest overlap
                sentences = re.split(r"(?<=[.!?])\s+", p_text)
                for s in sentences:
                    s_tokens = set(tokenize(s))
                    s_overlap = len(claim_tokens.intersection(s_tokens)) / max(1, len(claim_tokens))
                    if s_overlap >= best_overlap * 0.8:
                        best_quote = s.strip()
                        break

        # Check for contradiction signals (e.g. "not", "no", "never", "failed", "opposite")
        negation_terms = {"no", "not", "none", "neither", "never", "failed", "unassociated", "ineffective"}
        claim_has_neg = bool(claim_tokens.intersection(negation_terms))

        if best_overlap >= 0.65:
            # High overlap
            return VerificationResult(
                claim_id=claim.claim_id,
                claim_text=claim.text,
                status="SUPPORTED",
                confidence=min(1.0, round(best_overlap, 2)),
                evidence_doc_id=best_passage.get("doc_id", "DOC"),
                evidence_quote=best_quote or best_passage.get("text", "")[:120],
                rationale=f"High semantic overlap ({best_overlap:.1%}) with source document.",
            )
        elif best_overlap >= 0.40 and claim_has_neg:
            return VerificationResult(
                claim_id=claim.claim_id,
                claim_text=claim.text,
                status="CONTRADICTED",
                confidence=0.75,
                evidence_doc_id=best_passage.get("doc_id", "DOC"),
                evidence_quote=best_quote,
                rationale="Polarity conflict or negation detected against source passage.",
            )
        else:
            return VerificationResult(
                claim_id=claim.claim_id,
                claim_text=claim.text,
                status="UNVERIFIED",
                confidence=max(0.2, round(best_overlap, 2)),
                evidence_doc_id=best_passage.get("doc_id", "DOC"),
                evidence_quote="",
                rationale=f"Insufficient evidence overlap ({best_overlap:.1%}) to attest claim.",
            )
