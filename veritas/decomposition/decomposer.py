"""
Atomic Claim Decomposition Engine (Stage A).
Converts generated answers or paragraphs into atomic, checkable factual claims.
Supports Groq/OpenAI LLM decomposition, local fine-tuned Seq2Seq models (Flan-T5),
and rule-based fallback.
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from openai import OpenAI


@dataclass
class AtomicClaim:
    claim_id: str
    text: str
    source_sentence: str = ""
    status: str = "PENDING"  # PENDING, SUPPORTED, CONTRADICTED, UNVERIFIED
    evidence_snippet: Optional[str] = None
    evidence_doc_id: Optional[str] = None
    confidence: float = 0.0
    discrepancy_explanation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "source_sentence": self.source_sentence,
            "status": self.status,
            "evidence_snippet": self.evidence_snippet,
            "evidence_doc_id": self.evidence_doc_id,
            "confidence": self.confidence,
            "discrepancy_explanation": self.discrepancy_explanation,
        }


DECOMPOSITION_SYSTEM_PROMPT = """You are an expert at decomposing scientific text into atomic factual claims.

An atomic claim is a single, self-contained, independently checkable proposition.

Rules:
- Each claim must express exactly ONE independently verifiable proposition.
- Split compound statements only when each resulting claim is independently meaningful and checkable.
- Do not split a claim merely because it contains multiple related details. Keep details together when they describe the same proposition, relationship, or measurement.
- Do not split a scientific finding from numerical or statistical evidence that describes or quantifies it.
- Resolve all pronouns and implicit references (e.g. "it", "this drug") using context from the passage, so each claim can be understood in isolation.
- Preserve the original meaning and certainty of every statement (preserve epistemic qualifiers such as "may", "suggests", "associated with").
- Preserve numbers, units, percentages, ranges, and named entities exactly.
- Do not include opinions, rhetorical filler, or metadata.

Output ONLY a valid JSON list of strings, e.g. ["Claim 1", "Claim 2"]. Do not include markdown codeblocks or extra text.
"""


class ClaimDecomposer:
    """Decomposes text into atomic claims using LLM, local Seq2Seq model, or heuristic fallback."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.groq.com/openai/v1",
        model: str = "llama-3.3-70b-versatile",
        local_model_path: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.base_url = base_url
        self.model = model
        self.local_model_path = local_model_path
        self._client: Optional[OpenAI] = None
        self._local_pipeline = None

    def _get_client(self) -> Optional[OpenAI]:
        if self._client is None and self.api_key:
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def decompose(self, text: str) -> List[AtomicClaim]:
        """Decompose text into atomic claims."""
        text = text.strip()
        if not text:
            return []

        # Attempt 1: Call LLM API if key is present
        client = self._get_client()
        if client:
            try:
                claims = self._decompose_with_llm(client, text)
                if claims:
                    return [
                        AtomicClaim(claim_id=f"c_{i:02d}", text=c)
                        for i, c in enumerate(claims, start=1)
                    ]
            except Exception:
                pass

        # Attempt 2: Local seq2seq model if path provided
        if self.local_model_path and os.path.exists(self.local_model_path):
            try:
                claims = self._decompose_with_local_model(text)
                if claims:
                    return [
                        AtomicClaim(claim_id=f"c_{i:02d}", text=c)
                        for i, c in enumerate(claims, start=1)
                    ]
            except Exception:
                pass

        # Attempt 3: Rule-based sentence/clause heuristic fallback
        return self._decompose_heuristic(text)

    def _decompose_with_llm(self, client: OpenAI, text: str) -> List[str]:
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": DECOMPOSITION_SYSTEM_PROMPT},
                {"role": "user", "content": f"Decompose this text into atomic claims:\n\n{text}"},
            ],
            temperature=0.0,
            max_tokens=1024,
        )
        content = (response.choices[0].message.content or "").strip()
        # Clean potential markdown wrapping
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content).strip()

        # Parse JSON
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return [str(c).strip() for c in parsed if str(c).strip()]
        return []

    def _decompose_with_local_model(self, text: str) -> List[str]:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        tokenizer = AutoTokenizer.from_pretrained(self.local_model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.local_model_path)

        prompt = f"decompose: {text}"
        inputs = tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True)
        outputs = model.generate(**inputs, max_length=256)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

        claims = [c.strip() for c in decoded.split("<CLAIM_SEP>") if c.strip()]
        return claims

    def _decompose_heuristic(self, text: str) -> List[AtomicClaim]:
        """Simple rule-based splitter into sentences and major clauses."""
        # Split on sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+", text)
        claims: List[AtomicClaim] = []
        idx = 1

        for s in sentences:
            s_clean = s.strip()
            if not s_clean:
                continue

            # Check if sentence has compound semicolon or 'furthermore/moreover'
            clauses = re.split(r";\s*|\b(?:furthermore|moreover|additionally),\s*", s_clean, flags=re.IGNORECASE)
            for clause in clauses:
                clause = clause.strip()
                if len(clause.split()) >= 3:
                    # Capitalize first letter and ensure ending period
                    claim_text = clause[0].upper() + clause[1:]
                    if not claim_text.endswith("."):
                        claim_text += "."
                    claims.append(
                        AtomicClaim(
                            claim_id=f"c_{idx:02d}",
                            text=claim_text,
                            source_sentence=s_clean,
                        )
                    )
                    idx += 1

        return claims
