"""
Sentence-to-Source-Span Citation Attribution Engine.
Segments verified output into sentences, resolves fine-grained character span
offsets in the source documents, and produces auditable citation markers and tables.
"""

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Any, Tuple
from veritas.retrieval.sparse import tokenize


@dataclass
class SentenceCitation:
    sentence_index: int
    sentence_text: str
    doc_id: str
    doc_title: str
    chunk_id: str
    span_start: int
    span_end: int
    matched_snippet: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sentence_index": self.sentence_index,
            "sentence_text": self.sentence_text,
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "chunk_id": self.chunk_id,
            "span_start": self.span_start,
            "span_end": self.span_end,
            "matched_snippet": self.matched_snippet,
            "confidence": round(self.confidence, 4),
        }


@dataclass
class AttributedOutput:
    annotated_text: str
    raw_text: str
    citations: List[SentenceCitation] = field(default_factory=list)
    audit_table_markdown: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "annotated_text": self.annotated_text,
            "raw_text": self.raw_text,
            "citations": [c.to_dict() for c in self.citations],
            "audit_table_markdown": self.audit_table_markdown,
        }


class CitationAttributor:
    """Calculates sentence-to-source-span alignment and generates citations."""

    def __init__(self, min_confidence: float = 0.25):
        self.min_confidence = min_confidence

    def attribute(
        self,
        text: str,
        retrieved_passages: Optional[List[Dict[str, Any]]] = None,
        passages: Optional[List[Dict[str, Any]]] = None,
    ) -> AttributedOutput:
        """
        Segment text into sentences, locate exact supporting spans across passages,
        and generate inline citation badges and an audit table.
        """
        target_passages = retrieved_passages if retrieved_passages is not None else (passages or [])
        text = text.strip()
        if not text:
            return AttributedOutput(annotated_text="", raw_text="")

        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", text)
            if s.strip() and len(s.split()) >= 3
        ]

        if not sentences:
            sentences = [text]

        citations: List[SentenceCitation] = []
        annotated_sentences: List[str] = []

        citation_idx = 1
        for sent_idx, sentence in enumerate(sentences, start=1):
            best_match = self._find_best_span(sentence, target_passages)

            if best_match and best_match["confidence"] >= self.min_confidence:
                badge = f"[^{citation_idx}]"
                annotated_sentences.append(f"{sentence} {badge}")

                citations.append(
                    SentenceCitation(
                        sentence_index=sent_idx,
                        sentence_text=sentence,
                        doc_id=best_match["doc_id"],
                        doc_title=best_match["title"],
                        chunk_id=best_match["chunk_id"],
                        span_start=best_match["span_start"],
                        span_end=best_match["span_end"],
                        matched_snippet=best_match["snippet"],
                        confidence=best_match["confidence"],
                    )
                )
                citation_idx += 1
            else:
                annotated_sentences.append(sentence)

        annotated_text = " ".join(annotated_sentences)
        audit_table = self._format_audit_table(citations)

        return AttributedOutput(
            annotated_text=annotated_text,
            raw_text=text,
            citations=citations,
            audit_table_markdown=audit_table,
        )

    def _find_best_span(
        self,
        sentence: str,
        passages: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Find the exact passage and character span with highest alignment to the sentence."""
        sent_tokens = set(tokenize(sentence))
        if not sent_tokens:
            return None

        best_score = 0.0
        best_data: Optional[Dict[str, Any]] = None

        for p in passages:
            p_text = p.get("text", "")
            if not p_text:
                continue

            # Check sentence-level alignment within the passage
            p_sentences = [
                s.strip()
                for s in re.split(r"(?<=[.!?])\s+", p_text)
                if s.strip()
            ]

            for ps in p_sentences:
                ps_tokens = set(tokenize(ps))
                if not ps_tokens:
                    continue

                # Jaccard overlap on content words
                jaccard = len(sent_tokens.intersection(ps_tokens)) / len(sent_tokens.union(ps_tokens))
                # Longest contiguous sequence matcher ratio
                seq_ratio = SequenceMatcher(None, sentence.lower(), ps.lower()).ratio()
                combined_score = 0.6 * jaccard + 0.4 * seq_ratio

                if combined_score > best_score:
                    best_score = combined_score

                    # Find exact char span in passage text
                    char_start = max(0, p_text.find(ps))
                    char_end = char_start + len(ps)

                    # Global document offset if chunk metadata is available
                    chunk_base_offset = p.get("start_char", 0)

                    best_data = {
                        "doc_id": p.get("doc_id", "DOC"),
                        "title": p.get("title", p.get("doc_id", "Unknown Document")),
                        "chunk_id": p.get("chunk_id", "c000"),
                        "span_start": chunk_base_offset + char_start,
                        "span_end": chunk_base_offset + char_end,
                        "snippet": ps,
                        "confidence": combined_score,
                    }

        return best_data

    def _format_audit_table(self, citations: List[SentenceCitation]) -> str:
        """Format citations into a GitHub-flavored Markdown table for auditing."""
        if not citations:
            return ""

        lines = [
            "### 🔍 Evidence Citation Audit Table",
            "| Citation | Source Document | Character Span | Evidence Snippet | Attestation Confidence |",
            "| :---: | :--- | :---: | :--- | :---: |",
        ]

        for i, c in enumerate(citations, start=1):
            snippet_cleaned = re.sub(r"\s+", " ", c.matched_snippet)
            if len(snippet_cleaned) > 120:
                snippet_cleaned = snippet_cleaned[:117] + "..."
            snippet_quoted = f'"{snippet_cleaned}"'

            conf_pct = f"{c.confidence * 100:.1f}%"
            span_str = f"[{c.span_start} : {c.span_end}]"

            lines.append(
                f"| [^{i}] | **{c.doc_id}** ({c.doc_title[:30]}) | `{span_str}` | {snippet_quoted} | `{conf_pct}` |"
            )

        return "\n".join(lines)
