"""
Integration Test for the Veritas RAG Engine Pipeline.
"""

import unittest
from veritas.pipeline.engine import VeritasEngine, VeritasResponse
from veritas.config import VeritasConfig


class TestVeritasEngine(unittest.TestCase):

    def setUp(self):
        self.config = VeritasConfig()
        self.config.max_verification_iterations = 2
        self.config.strict_safety_pruning = True
        self.engine = VeritasEngine(config=self.config)

        self.custom_passages = [
            {
                "chunk_id": "PMC_TEST#c001",
                "doc_id": "PMC_TEST",
                "title": "Adverse Drug Reactions",
                "text": "Treatment of cardiac immune-related adverse events is mainly supportive and includes corticosteroids.",
                "start_char": 0,
                "end_char": 105,
            }
        ]

    def test_end_to_end_query_execution(self):
        query = "What is the primary treatment for cardiac immune-related adverse events?"
        response: VeritasResponse = self.engine.query(
            query_text=query,
            custom_context_passages=self.custom_passages,
        )

        self.assertIsNotNone(response.annotated_answer)
        self.assertGreater(len(response.annotated_answer), 0)
        self.assertGreaterEqual(response.iterations_run, 1)
        self.assertIsInstance(response.final_attestation_rate, float)
        self.assertIsInstance(response.final_hallucination_rate, float)
        self.assertGreater(len(response.iteration_traces), 0)


if __name__ == "__main__":
    unittest.main()
