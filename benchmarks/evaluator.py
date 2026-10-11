"""
Benchmark Evaluation Engine for Veritas RAG.

Compares three configurations on a domain QA set:
  1. Standard Baseline RAG        (single-pass retrieval + generation)
  2. Post-Hoc Single-Pass Check   (one critique-and-revise pass, RARR-style)
  3. Veritas Recursive Verification (iterative refinement loop)

What is MEASURED here
  - Hallucination / attestation rate (as judged by the engine's own verifier)
  - Mean claims per answer, mean iterations, mean latency
  - How often a revision actually ran (single pass) / the loop needed >1 round
  - Seeded-error check: does the verifier flag an answer after one fabricated
    claim is injected? (a sanity check on the verifier itself)

What is NOT measured (reported as None / "not measured")
  - Citation precision and citation recall. Earlier versions of this file
    filled these with constants (0.85 / 0.92 / 0.98) and a claim-count ratio,
    which are not measurements. They need citation spans compared against
    ground-truth spans.

Known limits of this setup (also stated in the generated report)
  - Every question is answered from ONE gold passage (open-book, no retrieval
    from the corpus, no distractor passages).
  - The same verifier that drives Veritas also judges all three systems.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from veritas.pipeline.engine import VeritasEngine, VeritasResponse
from veritas.config import VeritasConfig  # noqa: F401  (kept for callers that import it from here)


# Fabricated statements that appear in none of the evaluation passages.
# One is appended (cycling) to each Standard-RAG draft for the seeded-error check.
SEEDED_FALSE_CLAIMS: List[str] = [
    "The recommended starting dose is 500 mg every 15 minutes.",
    "The study was funded by a $4.2 million grant from the National Cancer Institute.",
    "The treatment was approved by the FDA in 1987 for veterinary use only.",
    "The authors reported that symptoms resolved within 24 hours in every patient.",
    "A follow-up period of 15 years was required before any benefit could be observed.",
]


def _round(value: Optional[float], digits: int) -> Optional[float]:
    return None if value is None else round(value, digits)


@dataclass
class SystemMetrics:
    system_name: str
    sample_count: int
    mean_hallucination_rate: float
    mean_attestation_rate: float
    mean_iterations: float
    mean_latency_seconds: float
    mean_claims_per_answer: float
    total_claims_evaluated: int
    unsupported_claims_count: int
    # How many samples triggered a revision (None = not applicable to this system)
    revisions_triggered: Optional[int] = None
    # Not measured in this benchmark; kept so older readers of the JSON don't break.
    mean_citation_precision: Optional[float] = None
    mean_citation_recall: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "system_name": self.system_name,
            "sample_count": self.sample_count,
            "mean_hallucination_rate": _round(self.mean_hallucination_rate, 4),
            "mean_attestation_rate": _round(self.mean_attestation_rate, 4),
            "mean_claims_per_answer": _round(self.mean_claims_per_answer, 2),
            "mean_iterations": _round(self.mean_iterations, 2),
            "mean_latency_seconds": _round(self.mean_latency_seconds, 2),
            "total_claims_evaluated": self.total_claims_evaluated,
            "unsupported_claims_count": self.unsupported_claims_count,
            "revisions_triggered": self.revisions_triggered,
            "mean_citation_precision": _round(self.mean_citation_precision, 4),  # null = not measured
            "mean_citation_recall": _round(self.mean_citation_recall, 4),        # null = not measured
        }


def _new_accumulator() -> Dict[str, Any]:
    return {
        "hal_rates": [], "att_rates": [], "iters": [], "latencies": [],
        "total_claims": 0, "failed_claims": 0, "revisions": 0,
    }


class BenchmarkEvaluator:
    """Evaluates and compares Veritas against baselines on domain benchmark QA sets."""

    def __init__(self, engine: VeritasEngine, seeded_check: bool = True):
        self.engine = engine
        self.seeded_check = seeded_check
        # Filled during run_benchmark(); saved by the runner as benchmark_checks.json
        self.checks: Dict[str, Any] = {
            "seeded_error_check": {
                "enabled": seeded_check,
                "tested": 0,
                "detected": 0,
                "errors": 0,
                "details": [],
            }
        }

    # ------------------------------------------------------------------
    def _seeded_error_check(self, idx: int, qid: str, draft: str, passages: List[dict]) -> None:
        """Append one fabricated claim to a draft and see whether the verifier flags the answer."""
        stats = self.checks["seeded_error_check"]
        false_claim = SEEDED_FALSE_CLAIMS[(idx - 1) % len(SEEDED_FALSE_CLAIMS)]
        try:
            poisoned = f"{draft.rstrip()} {false_claim}"
            claims = self.engine.decomposer.decompose(poisoned)
            results = [self.engine.verifier.verify_claim(c, passages) for c in claims]
            report = self.engine.discrepancy_gen.generate_report(results, iteration=1)
        except Exception as exc:  # report it; never silently count as a pass
            stats["errors"] += 1
            print(f"  [seeded check] failed on {qid}: {exc}")
            return

        detected = not report.is_fully_attested
        stats["tested"] += 1
        stats["detected"] += int(detected)
        stats["details"].append({"id": qid, "injected_claim": false_claim, "flagged": detected})

    # ------------------------------------------------------------------
    def run_benchmark(
        self,
        eval_dataset_path: Path,
        max_samples: Optional[int] = None,
    ) -> Dict[str, SystemMetrics]:
        """Run the benchmark on a dataset and return comparative metrics for all 3 systems."""
        with open(eval_dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        if max_samples:
            dataset = dataset[:max_samples]

        print(f"Loaded {len(dataset)} evaluation samples.")

        results = {
            "Standard_RAG": _new_accumulator(),
            "Single_Pass_Verification": _new_accumulator(),
            "Veritas_Recursive": _new_accumulator(),
        }

        for idx, item in enumerate(dataset, start=1):
            qid = item["id"]
            question = item["question"]
            context_passage = item["context_passage"]
            doc_id = item["doc_id"]

            candidate_passages = [
                {
                    "chunk_id": f"{doc_id}#c001",
                    "doc_id": doc_id,
                    "title": doc_id,
                    "text": context_passage,
                    "start_char": 0,
                    "end_char": len(context_passage),
                }
            ]

            print(f"[{idx}/{len(dataset)}] Evaluating sample {qid}...")

            # ---------------------------------------------------------
            # 1. Standard RAG (single pass, no revision)
            # ---------------------------------------------------------
            t0 = time.time()
            raw_draft = self.engine.generator.generate_draft(question, candidate_passages)
            t_std = time.time() - t0

            draft_claims = self.engine.decomposer.decompose(raw_draft)
            v_results_std = [self.engine.verifier.verify_claim(c, candidate_passages) for c in draft_claims]
            rep_std = self.engine.discrepancy_gen.generate_report(v_results_std, iteration=1)

            acc = results["Standard_RAG"]
            acc["hal_rates"].append(rep_std.hallucination_rate)
            acc["att_rates"].append(rep_std.attestation_rate)
            acc["iters"].append(1.0)
            acc["latencies"].append(t_std)
            acc["total_claims"] += len(draft_claims)
            acc["failed_claims"] += rep_std.contradicted_claims + rep_std.unverified_claims

            if self.seeded_check:
                self._seeded_error_check(idx, qid, raw_draft, candidate_passages)

            # ---------------------------------------------------------
            # 2. Post-hoc single-pass verification (one critique + edit)
            #    The revision only runs when the first check flagged something.
            # ---------------------------------------------------------
            t0 = time.time()
            revision_ran = not rep_std.is_fully_attested
            if revision_ran:
                refined_sp = self.engine.generator.refine_draft(
                    query=question,
                    passages=candidate_passages,
                    previous_draft=raw_draft,
                    discrepancy_feedback=rep_std.feedback_prompt,
                )
            else:
                refined_sp = raw_draft
            t_sp = time.time() - t0 + t_std

            sp_claims = self.engine.decomposer.decompose(refined_sp)
            v_results_sp = [self.engine.verifier.verify_claim(c, candidate_passages) for c in sp_claims]
            rep_sp = self.engine.discrepancy_gen.generate_report(v_results_sp, iteration=1)

            acc = results["Single_Pass_Verification"]
            acc["hal_rates"].append(rep_sp.hallucination_rate)
            acc["att_rates"].append(rep_sp.attestation_rate)
            acc["iters"].append(1.0)
            acc["latencies"].append(t_sp)
            acc["total_claims"] += len(sp_claims)
            acc["failed_claims"] += rep_sp.contradicted_claims + rep_sp.unverified_claims
            acc["revisions"] += int(revision_ran)

            # ---------------------------------------------------------
            # 3. Veritas recursive cross-verification (full engine)
            # ---------------------------------------------------------
            v_resp: VeritasResponse = self.engine.query(
                query_text=question,
                max_iterations=3,
                custom_context_passages=candidate_passages,
            )

            acc = results["Veritas_Recursive"]
            acc["hal_rates"].append(v_resp.final_hallucination_rate)
            acc["att_rates"].append(v_resp.final_attestation_rate)
            acc["iters"].append(float(v_resp.iterations_run))
            acc["latencies"].append(v_resp.execution_time_seconds)
            acc["total_claims"] += len(v_resp.claims)
            acc["failed_claims"] += int(len(v_resp.claims) * v_resp.final_hallucination_rate)
            acc["revisions"] += int(v_resp.iterations_run > 1)

        # -------------------------------------------------------------
        # Summaries
        # -------------------------------------------------------------
        summary: Dict[str, SystemMetrics] = {}
        n = len(dataset)

        for sys_name, data in results.items():
            summary[sys_name] = SystemMetrics(
                system_name=sys_name,
                sample_count=n,
                mean_hallucination_rate=sum(data["hal_rates"]) / n if n else 0.0,
                mean_attestation_rate=sum(data["att_rates"]) / n if n else 0.0,
                mean_iterations=sum(data["iters"]) / n if n else 1.0,
                mean_latency_seconds=sum(data["latencies"]) / n if n else 0.0,
                mean_claims_per_answer=data["total_claims"] / n if n else 0.0,
                total_claims_evaluated=data["total_claims"],
                unsupported_claims_count=data["failed_claims"],
                revisions_triggered=None if sys_name == "Standard_RAG" else data["revisions"],
            )

        return summary