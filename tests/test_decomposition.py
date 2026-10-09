"""
Unit Tests for Claim Decomposition Subsystem.
"""

import unittest
from veritas.decomposition.decomposer import ClaimDecomposer, AtomicClaim


class TestClaimDecomposer(unittest.TestCase):

    def setUp(self):
        self.decomposer = ClaimDecomposer()

    def test_heuristic_decomposition(self):
        text = (
            "Clozapine is an atypical antipsychotic. "
            "It is associated with dose-dependent hematologic malignancy; "
            "furthermore, patients require close monitoring."
        )
        claims = self.decomposer._decompose_heuristic(text)

        self.assertGreaterEqual(len(claims), 2)
        for c in claims:
            self.assertIsInstance(c, AtomicClaim)
            self.assertTrue(c.text.endswith("."))
            self.assertGreater(len(c.text.split()), 2)

    def test_empty_text(self):
        claims = self.decomposer.decompose("")
        self.assertEqual(len(claims), 0)


if __name__ == "__main__":
    unittest.main()
