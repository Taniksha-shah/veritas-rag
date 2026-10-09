"""
Unit Tests for Sentence-to-Source-Span Citation Attribution.
"""

import unittest
from veritas.attribution.citation import CitationAttributor, AttributedOutput, SentenceCitation


class TestCitationAttribution(unittest.TestCase):

    def setUp(self):
        self.attributor = CitationAttributor(min_confidence=0.20)
        self.passages = [
            {
                "chunk_id": "PMC_101#c001",
                "doc_id": "PMC_101",
                "title": "Clozapine Pharmacokinetics",
                "text": "Risk was dose-dependent, where cumulative clozapine exposures over 5000 DDD were significantly associated with malignancy risk. Patients with shorter exposure showed no significant elevation.",
                "start_char": 100,
                "end_char": 310,
            }
        ]

    def test_sentence_attribution_and_badges(self):
        answer = "Cumulative clozapine exposures over 5000 DDD were significantly associated with malignancy risk."
        res: AttributedOutput = self.attributor.attribute(answer, self.passages)

        self.assertIn("[^1]", res.annotated_text)
        self.assertEqual(len(res.citations), 1)
        self.assertEqual(res.citations[0].doc_id, "PMC_101")
        self.assertGreater(res.citations[0].span_end, res.citations[0].span_start)
        self.assertIn("Evidence Citation Audit Table", res.audit_table_markdown)


if __name__ == "__main__":
    unittest.main()
