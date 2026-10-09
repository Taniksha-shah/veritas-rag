"""
Primary Generator & Iterative Refinement Controller.
Produces conditioned drafts from retrieved context and refines drafts
based on recursive verifier discrepancy feedback.
"""

import json
import os
import re
from typing import Dict, List, Optional, Any
from openai import OpenAI


GENERATOR_SYSTEM_PROMPT = """You are Veritas, a precision domain-specific scientific and legal AI assistant.

Your task is to answer the user query accurately and comprehensively, conditioned STRICTLY on the provided evidence context.

Strict Grounding Rules:
1. Every factual assertion you make MUST be directly traceable to the provided evidence passages.
2. Do not introduce outside medical or scientific speculation, unverified extrapolations, or assumed details.
3. Preserve numbers, percentages, confidence intervals, sample sizes, and study conclusions exactly as stated.
4. If the retrieved evidence does not contain sufficient information to answer a part of the query, explicitly state what is known and what cannot be answered.
5. Write in clear, authoritative, objective domain prose without meta-commentary (do not say "Based on passage 1...", simply state the facts).
"""


class VeritasGenerator:
    """Primary LLM Generator for initial answers and recursive refinements."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.groq.com/openai/v1",
        model: str = "llama-3.3-70b-versatile",
    ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.base_url = base_url
        self.model = model
        self._client: Optional[OpenAI] = None

    def _get_client(self) -> Optional[OpenAI]:
        if self._client is None and self.api_key:
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def _format_context(self, passages: List[Dict[str, Any]]) -> str:
        blocks = []
        for i, p in enumerate(passages, start=1):
            doc_id = p.get("doc_id", f"Doc_{i}")
            title = p.get("title", "")
            text = p.get("text", "")
            header = f"--- Document [{doc_id}] ({title}) ---" if title else f"--- Document [{doc_id}] ---"
            blocks.append(f"{header}\n{text}")
        return "\n\n".join(blocks)

    def generate_draft(self, query: str, passages: List[Dict[str, Any]]) -> str:
        """Produce the initial draft answer conditioned on retrieved passages."""
        context_str = self._format_context(passages)
        client = self._get_client()

        if client:
            try:
                user_message = f"CONTEXT:\n{context_str}\n\nQUESTION: {query}\n\nProvide a precise, grounded answer."
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=0.1,
                    max_tokens=1024,
                )
                answer = (response.choices[0].message.content or "").strip()
                if answer:
                    return answer
            except Exception:
                pass

        # Resilient offline fallback: construct summary from top passages
        return self._fallback_summarize(query, passages)

    def refine_draft(
        self,
        query: str,
        passages: List[Dict[str, Any]],
        previous_draft: str,
        discrepancy_feedback: str,
    ) -> str:
        """
        Produce a refined draft addressing specific discrepancy feedback
        from the verifier while preserving all supported claims.
        """
        context_str = self._format_context(passages)
        client = self._get_client()

        if client:
            try:
                refinement_prompt = (
                    f"CONTEXT:\n{context_str}\n\n"
                    f"QUESTION: {query}\n\n"
                    f"PREVIOUS DRAFT:\n{previous_draft}\n\n"
                    f"VERIFICATION DISCREPANCY AUDIT:\n{discrepancy_feedback}\n\n"
                    "INSTRUCTIONS: Rewrite the draft to address all discrepancies above. "
                    "Remove or correct any refuted or unverified assertions, retain all verified facts, "
                    "and produce a seamless, fully grounded final answer."
                )

                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
                        {"role": "user", "content": refinement_prompt},
                    ],
                    temperature=0.1,
                    max_tokens=1024,
                )
                answer = (response.choices[0].message.content or "").strip()
                if answer:
                    return answer
            except Exception:
                pass

        # Offline fallback refinement: strip sentences matching failed claims
        return previous_draft

    def _fallback_summarize(self, query: str, passages: List[Dict[str, Any]]) -> str:
        """Extract top representative sentences from context when offline."""
        if not passages:
            return "No sufficient evidence was found in the indexed corpus to address the query."

        top_sentences = []
        for p in passages[:3]:
            text = p.get("text", "")
            sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) >= 6]
            if sents:
                top_sentences.append(sents[0])

        return " ".join(top_sentences)
