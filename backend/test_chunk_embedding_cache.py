from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np

from extraction_retrieval import ExtractionRetriever, FIELD_QUERIES, source_passages
from qa_chunk_embedding_cache import ChunkEmbeddingCache, ChunkCachingEncoder
from qa_embedding_cache_benchmark import assert_same_evidence, compare_vectors, observed_peak_rss


class ChunkEmbeddingCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.identity = {"alias": "minilm", "model": "repo/minilm", "revision": "rev1",
                         "dimension": 2, "normalize_embeddings": True, "dtype": "torch.float32"}

    def cache(self, identity=None, document="invoice-1", mode="legacy"):
        return ChunkEmbeddingCache(self.root, identity or self.identity, document, mode)

    def test_model_revision_dimension_normalization_and_mode_required(self):
        for key in self.identity:
            identity = dict(self.identity)
            del identity[key]
            with self.assertRaises(ValueError):
                self.cache(identity)
        for identity in ({**self.identity, "normalize_embeddings": False},
                         {**self.identity, "revision": ""}, {**self.identity, "dimension": True}):
            with self.assertRaises(ValueError):
                self.cache(identity)
        for document, mode in (("", "legacy"), (None, "legacy"), ("doc", "unknown")):
            with self.assertRaises(ValueError):
                self.cache(document=document, mode=mode)

    def test_document_model_revision_dimension_and_chunking_separate_entries(self):
        base = self.cache()
        base.put(["x", "y"], np.eye(2, dtype=np.float32))
        variants = [self.cache(document="invoice-2"), self.cache(mode="layout")]
        variants += [self.cache({**self.identity, **change}) for change in (
            {"alias": "bge-m3", "model": "BAAI/bge-m3", "dimension": 1024},
            {"model": "repo/other"}, {"revision": "rev2"}, {"dimension": 3}, {"dtype": "float64"})]
        for other in variants:
            self.assertNotEqual(base.path, other.path)
            self.assertFalse(other.contains(["x", "y"]))
            with self.assertRaises(FileNotFoundError):
                other.get(["x", "y"])

    def test_ordered_text_and_modified_text_invalidate_cache(self):
        cache = self.cache()
        cache.put(["x", "y"], np.eye(2, dtype=np.float32))
        for text in (["y", "x"], ["x changed", "y"], ["x"]):
            self.assertFalse(cache.contains(text))
        self.assertTrue(cache.contains(["x", "y"]))
        np.testing.assert_array_equal(cache.get(["x", "y"]), np.eye(2, dtype=np.float32))

    def test_incomplete_entry_refused(self):
        cache = self.cache()
        (cache.path / f"{cache.key(['x'])}.npy").write_bytes(b"incomplete")
        with self.assertRaises(ValueError):
            cache.contains(["x"])

    def test_foreign_document_metadata_refused_even_if_copied_into_namespace(self):
        cache = self.cache()
        cache.put(["x", "y"], np.eye(2, dtype=np.float32))
        path = cache.path / f"{cache.key(['x', 'y'])}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["identity"]["document_id"] = "another"
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaises(ValueError):
            cache.get(["x", "y"])

    def test_corrupted_file_refused(self):
        cache = self.cache()
        cache.put(["x", "y"], np.eye(2, dtype=np.float32))
        (cache.path / f"{cache.key(['x', 'y'])}.npy").write_bytes(b"bad")
        with self.assertRaises(ValueError):
            cache.get(["x", "y"])

    def test_queries_recomputed_and_chunks_encoded_once_across_calls(self):
        model = Mock()
        model.encode.side_effect = lambda texts, **kw: np.tile(np.array([[1, 0]], dtype=np.float32), (len(texts), 1))
        encoder = ChunkCachingEncoder(model, self.root, self.identity, ["query"])
        for repeat in range(3):
            encoder.begin_document("doc", ["chunk"])
            encoder.encode(["query"], normalize_embeddings=True)
            encoder.encode(["chunk"], normalize_embeddings=True)
            self.assertFalse(encoder.events[0]["cache_hit"])
            self.assertEqual(encoder.events[1]["cache_hit"], repeat > 0)
        calls = [call.args[0] for call in model.encode.call_args_list]
        self.assertEqual(calls, [["query"], ["chunk"], ["query"], ["query"]])
        self.assertEqual(len(list(self.root.rglob("*.npy"))), 1)

    def test_unexpected_encode_order_rejected(self):
        encoder = ChunkCachingEncoder(Mock(), self.root, self.identity, ["query"])
        encoder.begin_document("doc", ["chunk"])
        with self.assertRaises(ValueError):
            encoder.encode(["chunk"], normalize_embeddings=True)
        with self.assertRaises(ValueError):
            encoder.encode(["query"], normalize_embeddings=False)

    def test_retriever_output_strictly_identical_with_chunk_cache(self):
        class DeterministicEncoder:
            def __init__(self):
                self.calls = []

            def encode(self, texts, normalize_embeddings=False):
                self.calls.append(list(texts))
                vectors = np.array([[len(t) % 5 + 1, len(t) % 7 + 1] for t in texts], dtype=np.float32)
                return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

        text = "Supplier Alpha\nMF 1234567A\nFacture 001\nDate 01/01/2026\nTotal HT 10\nTVA 1.9\nTotal TTC 11.9"
        plain_model = DeterministicEncoder()
        plain = ExtractionRetriever()
        plain.load_attempted, plain.model, plain.backend = True, plain_model, "bm25+sentence_transformers"
        expected = plain.retrieve(text, top_k=5, chunking_mode="legacy", document_id="doc")
        model = DeterministicEncoder()
        encoder = ChunkCachingEncoder(model, self.root, self.identity, list(FIELD_QUERIES.values()))
        retriever = ExtractionRetriever()
        retriever.load_attempted, retriever.model, retriever.backend = True, encoder, plain.backend
        for _ in range(3):
            encoder.begin_document("doc", [c["text"] for c in source_passages(text)])
            actual = retriever.retrieve(text, top_k=5, chunking_mode="legacy", document_id="doc")
            self.assertEqual(actual, expected)
            self.assertIs(retriever.model, encoder)
        self.assertEqual(len(model.calls), 4)

    def test_equivalence_rejects_score_or_ranking_changes(self):
        expected = {"field": [{"source_id": "1", "score": .03}, {"source_id": "2", "score": .02}]}
        assert_same_evidence(deepcopy(expected), expected)
        for actual in ({"field": list(reversed(expected["field"]))},
                       {"field": [{"source_id": "1", "score": .030001}, expected["field"][1]]}):
            with self.assertRaises(ValueError):
                assert_same_evidence(actual, expected)

    def test_vector_tolerance_does_not_hide_dimension_precision_or_value_changes(self):
        original = np.eye(2, dtype=np.float32)
        self.assertTrue(compare_vectors(original, original)["bit_exact"])
        self.assertFalse(compare_vectors(original + np.float32(1e-8), original)["bit_exact"])
        for other in (original.astype(np.float64), np.eye(3, dtype=np.float32), original + .001):
            with self.assertRaises(ValueError):
                compare_vectors(other, original)

    def test_report_peak_covers_independent_stage_samplers(self):
        run = {"memory": {"peak_rss_bytes": 100}, "initialization": {"memory": {"peak_rss_bytes": 90}},
               "documents": [{"cold": {"memory": {"peak_rss_bytes": 110}},
                              "warm": [{"memory": {"peak_rss_bytes": 95}}]}]}
        self.assertEqual(observed_peak_rss(run), 110)


if __name__ == "__main__":
    unittest.main()
