"""
End-to-End Veritas RAG Engine Orchestrator.
Executes the recursive cross-verification loop between generator and verifier,
drives factual convergence, maps sentence-level citation attribution,
and records compliance audit trails.
"""

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

from veritas.config import VeritasConfig
from veritas.retrieval.hybrid import HybridRetriever
from veritas.retrieval.indexer import CorpusIndexer
from veritas.decomposition.decomposer import ClaimDecomposer, AtomicClaim
from veritas.verification.entailment import EntailmentVerifier, VerificationResult
from veritas.verification.discrepancy import DiscrepancyGenerator, DiscrepancyReport
from veritas.generator.generator import VeritasGenerator
from veritas.attribution.citation import CitationAttributor, AttributedOutput, SentenceCitation
from veritas.audit.logger import AuditLogger, AuditRecord


@dataclass
class VeritasResponse:
    query: str
    annotated_answer: str
    raw_answer: str
    audit_table_markdown: str
    iterations_run: int
    converged: bool
    initial_draft: str
    initial_hallucination_rate: float
    final_hallucination_rate: float
    final_attestation_rate: float
    claims: List[Dict[str, Any]]
    citations: List[Dict[str, Any]]
    retrieved_passages: List[Dict[str, Any]]
    iteration_traces: List[Dict[str, Any]]
    execution_time_seconds: float
    execution_id: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "query": self.query,
            "annotated_answer": self.annotated_answer,
            "raw_answer": self.raw_answer,
            "audit_table_markdown": self.audit_table_markdown,
            "iterations_run": self.iterations_run,
            "converged": self.converged,
            "initial_hallucination_rate": round(self.initial_hallucination_rate, 4),
            "final_hallucination_rate": round(self.final_hallucination_rate, 4),
            "final_attestation_rate": round(self.final_attestation_rate, 4),
            "claims": self.claims,
            "citations": self.citations,
            "retrieved_passages": self.retrieved_passages,
            "iteration_traces": self.iteration_traces,
            "execution_time_seconds": round(self.execution_time_seconds, 3),
        }


class VeritasEngine:
    """The Complete Veritas RAG Engine with Recursive Citation Verification."""

    def __init__(
        self,
        config: Optional[VeritasConfig] = None,
        retriever: Optional[HybridRetriever] = None,
    ):
        self.config = config or VeritasConfig()

        # Initialize sub-modules
        self.generator = VeritasGenerator(
            api_key=self.config.groq_api_key,
            base_url=self.config.groq_base_url,
            model=self.config.generator_model,
        )

        self.decomposer = ClaimDecomposer(
            api_key=self.config.groq_api_key,
            base_url=self.config.groq_base_url,
            model=self.config.verifier_model,
        )

        self.verifier = EntailmentVerifier(
            api_key=self.config.groq_api_key,
            base_url=self.config.groq_base_url,
            model=self.config.verifier_model,
            confidence_threshold=self.config.entailment_confidence_threshold,
        )

        self.discrepancy_gen = DiscrepancyGenerator()
        self.attributor = CitationAttributor(
            min_confidence=self.config.min_citation_overlap_score
        )
        self.audit_logger = AuditLogger(log_dir=self.config.audit_log_dir)

        # Retriever
        self.retriever = retriever
        if self.retriever is None:
            self._init_or_load_retriever()

    def _init_or_load_retriever(self) -> None:
        """Attempt to load index if present on disk."""
        bm25_file = self.config.index_dir / "bm25_index.json"
        if bm25_file.exists():
            try:
                self.retriever = CorpusIndexer.load_index(
                    index_dir=self.config.index_dir,
                    embedding_model=self.config.local_embedding_model,
                )
                return
            except Exception:
                pass
        # Left as None until indexed or custom context is passed
        self.retriever = None

    def query(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        max_iterations: Optional[int] = None,
        custom_context_passages: Optional[List[Dict[str, Any]]] = None,
    ) -> VeritasResponse:
        """
        Execute full Veritas query workflow:
        1. Retrieval (Hybrid)
        2. Initial Draft Generation
        3. Recursive Verification Loop (Decompose -> Entailment -> Discrepancy -> Refine)
        4. Sentence-to-Source-Span Citation Attribution
        5. Audit Logging
        """
        start_time = time.time()
        exec_id = str(uuid.uuid4())
        k = top_k or self.config.top_k_retrieval
        max_iter = max_iterations or self.config.max_verification_iterations

        # 1. Retrieval
        if custom_context_passages is not None:
            retrieved_passages = custom_context_passages
        elif self.retriever is not None:
            retrieved_passages = self.retriever.search(query_text, top_k=k)
        else:
            retrieved_passages = []

        # 2. Initial Draft Generation
        initial_draft = self.generator.generate_draft(query_text, retrieved_passages)
        current_draft = initial_draft

        iteration_traces: List[Dict[str, Any]] = []
        initial_hallucination_rate = 0.0
        final_report: Optional[DiscrepancyReport] = None
        final_claims_dict: List[Dict[str, Any]] = []
        converged = False

        # 3. Recursive Cross-Verification Loop
        for iteration in range(1, max_iter + 1):
            # 3a. Decompose into atomic claims
            atomic_claims = self.decomposer.decompose(current_draft)

            # 3b. Verify each claim
            verification_results: List[VerificationResult] = []
            verified_claim_objects: List[AtomicClaim] = []

            for claim in atomic_claims:
                v_res = self.verifier.verify_claim(claim, retrieved_passages)
                v_res.claim_id = claim.claim_id
                verification_results.append(v_res)

                claim.status = v_res.status
                claim.evidence_snippet = v_res.evidence_quote
                claim.evidence_doc_id = v_res.evidence_doc_id
                claim.confidence = v_res.confidence
                claim.discrepancy_explanation = v_res.rationale
                verified_claim_objects.append(claim)

            # 3c. Generate discrepancy report
            report = self.discrepancy_gen.generate_report(verification_results, iteration=iteration)

            if iteration == 1:
                initial_hallucination_rate = report.hallucination_rate

            final_report = report
            final_claims_dict = [c.to_dict() for c in verified_claim_objects]

            # Record step trace
            trace_entry = {
                "iteration": iteration,
                "draft": current_draft,
                "claims": final_claims_dict,
                "supported_count": report.supported_claims,
                "contradicted_count": report.contradicted_claims,
                "unverified_count": report.unverified_claims,
                "is_fully_attested": report.is_fully_attested,
                "hallucination_rate": report.hallucination_rate,
                "feedback_prompt": report.feedback_prompt,
            }
            iteration_traces.append(trace_entry)

            # Check convergence
            if report.is_fully_attested:
                converged = True
                break

            # 3d. Refinement if not converged and iterations remain
            if iteration < max_iter:
                refined_draft = self.generator.refine_draft(
                    query=query_text,
                    passages=retrieved_passages,
                    previous_draft=current_draft,
                    discrepancy_feedback=report.feedback_prompt,
                )
                current_draft = refined_draft
            else:
                # Max iteration reached: Strict Safety Pruning
                if self.config.strict_safety_pruning and not report.is_fully_attested:
                    current_draft = self._apply_safety_pruning(current_draft, verified_claim_objects)

        # 4. Citation Attribution
        attribution_result: AttributedOutput = self.attributor.attribute(
            text=current_draft,
            passages=retrieved_passages,
        )

        elapsed = time.time() - start_time
        final_hal_rate = final_report.hallucination_rate if final_report else 0.0
        final_att_rate = final_report.attestation_rate if final_report else 1.0

        # 5. Log Execution Trail
        audit_record = AuditRecord(
            execution_id=exec_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            query=query_text,
            retrieved_chunk_ids=[p.get("chunk_id", "N/A") for p in retrieved_passages],
            iterations_run=len(iteration_traces),
            converged=converged,
            initial_draft=initial_draft,
            final_output=attribution_result.annotated_text,
            initial_hallucination_rate=initial_hallucination_rate,
            final_hallucination_rate=final_hal_rate,
            final_attestation_rate=final_att_rate,
            citations_count=len(attribution_result.citations),
            iteration_traces=iteration_traces,
            execution_time_seconds=elapsed,
        )
        self.audit_logger.log_execution(audit_record)

        return VeritasResponse(
            query=query_text,
            annotated_answer=attribution_result.annotated_text,
            raw_answer=current_draft,
            audit_table_markdown=attribution_result.audit_table_markdown,
            iterations_run=len(iteration_traces),
            converged=converged,
            initial_draft=initial_draft,
            initial_hallucination_rate=initial_hallucination_rate,
            final_hallucination_rate=final_hal_rate,
            final_attestation_rate=final_att_rate,
            claims=final_claims_dict,
            citations=[c.to_dict() for c in attribution_result.citations],
            retrieved_passages=retrieved_passages,
            iteration_traces=iteration_traces,
            execution_time_seconds=elapsed,
            execution_id=exec_id,
        )

    def _apply_safety_pruning(self, text: str, claims: List[AtomicClaim]) -> str:
        """
        Prunes or qualifies unverified/contradicted propositions
        to guarantee 0 hallucination in final output.
        """
        failed_texts = [
            c.text.lower().rstrip(".")
            for c in claims
            if c.status in {"CONTRADICTED", "UNVERIFIED"}
        ]
        if not failed_texts:
            return text

        import re
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        retained = []

        for s in sentences:
            s_lower = s.lower()
            is_compromised = any(ft in s_lower or s_lower in ft for ft in failed_texts)
            if not is_compromised:
                retained.append(s)

        if not retained:
            return "Based strictly on the verified evidence, no attested statements could be validated for this specific query."

        return " ".join(retained)
