"""
Unit Tests for Veritas Retrieval Subsystem.
"""

import unittest
from veritas.retrieval.sparse import BM25Retriever, tokenize
from veritas.retrieval.dense import DenseRetriever
from veritas.retrieval.hybrid import HybridRetriever
from veritas.retrieval.indexer import CorpusIndexer, DocumentChunk


class TestRetrievalSubsystem(unittest.TestCase):

    def setUp(self):
        self.sample_chunks = [
            {
                "chunk_id": "doc1#c001",
                "doc_id": "doc1",
                "title": "Clozapine study",
                "text": "Clozapine exposure was associated with dose-dependent increased risk of hematologic malignancy.",
                "start_char": 0,
                "end_char": 95,
            },
            {
                "chunk_id": "doc2#c001",
                "doc_id": "doc2",
                "title": "Cardiac adverse events",
                "text": "Treatment of cardiac immune-related adverse events is mainly supportive with corticosteroids.",
                "start_char": 0,
                "end_char": 92,
            },
            {
                "chunk_id": "doc3#c001",
                "doc_id": "doc3",
                "title": "Respiratory support",
                "text": "Non-invasive respiratory support using HFNC reduced intubation in selected pulmonary patients.",
                "start_char": 0,
                "end_char": 94,
            },
        ]

    def test_tokenization(self):
        tokens = tokenize("Clozapine exposure; was dose-dependent!")
        self.assertIn("clozapine", tokens)
        self.assertIn("exposure", tokens)
        self.assertIn("dose-dependent", tokens)
        self.assertNotIn("was", tokens)  # stopword removed

    def test_bm25_retrieval(self):
        bm25 = BM25Retriever()
        bm25.fit(self.sample_chunks)
        results = bm25.search("hematologic malignancy clozapine", top_k=2)

        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["doc_id"], "doc1")
        self.assertGreater(results[0]["sparse_score"], 0.0)

    def test_chunker_sliding_window(self):
        indexer = CorpusIndexer(chunk_size_words=10, chunk_overlap_words=3)
        sample_text = "Word " * 25
        chunks = indexer.chunk_document(sample_text, doc_id="test_doc", title="Test Title")

        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0].doc_id, "test_doc")
        self.assertGreater(chunks[0].end_char, chunks[0].start_char)

    def test_hybrid_rrf_retrieval(self):
        sparse = BM25Retriever().fit(self.sample_chunks)
        dense = DenseRetriever().fit(self.sample_chunks)
        hybrid = HybridRetriever(sparse, dense, rrf_k=60)

        results = hybrid.search("corticosteroids cardiac irAEs", top_k=2)
        self.assertGreater(len(results), 0)
        self.assertIn("rrf_score", results[0])
        self.assertEqual(results[0]["doc_id"], "doc2")


if __name__ == "__main__":
    unittest.main()
