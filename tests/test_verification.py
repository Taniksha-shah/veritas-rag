"""
Unit Tests for Textual Entailment Verification and Discrepancy Generation.
"""

import unittest
from veritas.decomposition.decomposer import AtomicClaim
from veritas.verification.entailment import EntailmentVerifier, VerificationResult
from veritas.verification.discrepancy import DiscrepancyGenerator, DiscrepancyReport


class TestVerificationSubsystem(unittest.TestCase):

    def setUp(self):
        self.verifier = EntailmentVerifier()
        self.discrepancy_gen = DiscrepancyGenerator()
        self.evidence_passages = [
            {
                "doc_id": "PMC_TEST_01",
                "text": "Clozapine exposure was significantly associated with a dose-dependent increased risk of hematologic malignancy.",
            }
        ]

    def test_heuristic_supported_claim(self):
        claim = AtomicClaim(
            claim_id="c_01",
            text="Clozapine exposure is associated with increased risk of hematologic malignancy.",
        )
        res = self.verifier._verify_heuristic(claim, self.evidence_passages)

        self.assertEqual(res.status, "SUPPORTED")
        self.assertGreaterEqual(res.confidence, 0.65)
        self.assertEqual(res.evidence_doc_id, "PMC_TEST_01")

    def test_heuristic_unverified_claim(self):
        claim = AtomicClaim(
            claim_id="c_02",
            text="Aspirin prevents Alzheimer's disease in adult populations.",
        )
        res = self.verifier._verify_heuristic(claim, self.evidence_passages)

        self.assertEqual(res.status, "UNVERIFIED")

    def test_discrepancy_report_generation(self):
        v_results = [
            VerificationResult(
                claim_id="c_01",
                claim_text="Clozapine is associated with malignancy.",
                status="SUPPORTED",
                confidence=0.9,
                evidence_doc_id="PMC_TEST_01",
                evidence_quote="Clozapine exposure was significantly associated...",
                rationale="Directly supported",
            ),
            VerificationResult(
                claim_id="c_02",
                claim_text="Aspirin cures cancer.",
                status="UNVERIFIED",
                confidence=0.1,
                evidence_doc_id="PMC_TEST_01",
                evidence_quote="",
                rationale="No evidence in text",
            ),
        ]

        report = self.discrepancy_gen.generate_report(v_results, iteration=1)

        self.assertFalse(report.is_fully_attested)
        self.assertEqual(report.supported_claims, 1)
        self.assertEqual(report.unverified_claims, 1)
        self.assertEqual(report.hallucination_rate, 0.5)
        self.assertIn("Aspirin cures cancer", report.feedback_prompt)


if __name__ == "__main__":
    unittest.main()
