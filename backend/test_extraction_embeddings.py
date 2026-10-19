import json
import hashlib
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

import extraction_embeddings as embeddings
from extraction_retrieval import ExtractionRetriever
from qa_embedding_benchmark import VectorCache, compare_units, metrics


class EmbeddingSelectionTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"RAG_EMBEDDING_MODEL": "minilm", "EXTRACTION_SEMANTIC_RAG": "true"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.cache = patch.dict(embeddings._models, {}, clear=True)
        self.cache.start()
        self.addCleanup(self.cache.stop)

    def test_default_is_minilm_without_environment_setting(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(ExtractionRetriever().embedding_model, "minilm")

    def test_model_loaded_once_shared_across_retrievers_on_cpu(self):
        model = Mock()
        model.get_sentence_embedding_dimension.return_value = 384
        factory = Mock(return_value=model)
        with patch.dict(sys.modules, {"sentence_transformers": SimpleNamespace(SentenceTransformer=factory)}):
            first, second = ExtractionRetriever(), ExtractionRetriever()
            first._load()
            first._load()
            second._load()
        self.assertIs(first.model, second.model)
        factory.assert_called_once_with("sentence-transformers/all-MiniLM-L6-v2", device="cpu", local_files_only=True)

    def test_bge_and_minilm_use_distinct_loaded_models(self):
        small, big = Mock(), Mock()
        small.get_sentence_embedding_dimension.return_value = 384
        big.get_sentence_embedding_dimension.return_value = 1024
        factory = Mock(side_effect=[small, big])
        with patch.dict(sys.modules, {"sentence_transformers": SimpleNamespace(SentenceTransformer=factory)}):
            first, second = ExtractionRetriever("minilm"), ExtractionRetriever("bge-m3")
            first._load()
            second._load()
        self.assertIs(first.model, small)
        self.assertIs(second.model, big)
        self.assertEqual(factory.call_args_list[1].kwargs, {"device": "cpu", "local_files_only": True})

    def test_model_choice_captured_per_instance(self):
        first = ExtractionRetriever()
        with patch.dict(os.environ, {"RAG_EMBEDDING_MODEL": "bge-m3"}):
            second = ExtractionRetriever()
        self.assertEqual(first.embedding_model, "minilm")
        self.assertEqual(second.embedding_model, "bge-m3")

    def test_unknown_model_rejected(self):
        with self.assertRaises(ValueError):
            ExtractionRetriever("foreign-model")

    def test_wrong_model_dimension_not_cached_or_silently_used(self):
        model = Mock()
        model.get_sentence_embedding_dimension.return_value = 384
        factory = Mock(return_value=model)
        with patch.dict(sys.modules, {"sentence_transformers": SimpleNamespace(SentenceTransformer=factory)}):
            with self.assertRaises(ValueError):
                embeddings.load_embedding_model("bge-m3")
        self.assertNotIn("bge-m3", embeddings._models)

    def test_bm25_path_does_not_load_embedding_model(self):
        with patch.dict(os.environ, {"EXTRACTION_SEMANTIC_RAG": "false"}), patch("extraction_retrieval.load_embedding_model") as loader:
            retriever = ExtractionRetriever()
            output = retriever.retrieve("TOTAL TTC 12.600", top_k=5, chunking_mode="legacy")
        loader.assert_not_called()
        self.assertEqual(retriever.backend, "bm25")
        self.assertTrue(output["total_amount"])


class VectorCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.identity = {"alias": "minilm", "model": "repo/a", "revision": "abc", "dimension": 2,
                         "normalize_embeddings": True, "dtype": "float32"}

    def test_identity_separates_model_revision_configuration_and_dimension(self):
        base = VectorCache(self.root, self.identity)
        for change in ({"alias": "bge-m3", "model": "repo/b"}, {"revision": "def"},
                       {"dimension": 3}, {"normalize_embeddings": False}):
            other = VectorCache(self.root, {**self.identity, **change})
            self.assertNotEqual(base.path, other.path)

    def test_roundtrip_and_text_order_are_checked(self):
        cache = VectorCache(self.root, self.identity)
        vectors = np.eye(2, dtype=np.float32)
        cache.put(["a", "b"], vectors)
        np.testing.assert_array_equal(cache.get(["a", "b"]), vectors)
        with self.assertRaises(FileNotFoundError):
            cache.get(["b", "a"])

    def test_foreign_metadata_refused(self):
        cache = VectorCache(self.root, self.identity)
        cache.put(["a", "b"], np.eye(2, dtype=np.float32))
        path = cache.path / f"{cache.key(['a', 'b'])}.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata["identity"]["model"] = "foreign/model"
        path.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaises(ValueError):
            cache.get(["a", "b"])

    def test_bad_dimension_precision_and_nonfinite_vectors_refused(self):
        cache = VectorCache(self.root, self.identity)
        for values in (np.ones((2, 3), dtype=np.float32), np.eye(2, dtype=np.float64),
                       np.full((2, 2), np.nan, dtype=np.float32), np.zeros((2, 2), dtype=np.float32)):
            with self.assertRaises(ValueError):
                cache.put(["a", "b"], values)

    def test_corrupt_array_refused_without_pickle(self):
        cache = VectorCache(self.root, self.identity)
        cache.put(["a", "b"], np.eye(2, dtype=np.float32))
        path = cache.path / f"{cache.key(['a', 'b'])}.npy"
        path.write_bytes(b"corrupted")
        with self.assertRaises(ValueError):
            cache.get(["a", "b"])


class EmbeddingBenchmarkTests(unittest.TestCase):
    def test_changes_classify_top5_exits_and_common_successes(self):
        rows = [{"document_id": "doc", "field": "field", "units": [{"id": str(i)} for i in range(5)], "ranks": [1, 4, None, 2, None]}]
        newer = [{**rows[0], "ranks": [2, 1, 3, None, None]}]
        cases = compare_units(rows, newer)
        self.assertTrue(cases[0]["A_both_succeed"])
        self.assertTrue(cases[0]["C_minilm_better"])
        self.assertTrue(cases[1]["B_bge_better"])
        self.assertEqual(cases[2]["D_leaves_top5"], "minilm")
        self.assertEqual(cases[3]["D_leaves_top5"], "bge-m3")
        self.assertTrue(cases[4]["both_fail"])

    def test_rank_metrics_preserve_missing_proof_denominator(self):
        rows = [{"ranks": [1, None], "recall": {"1": .5, "3": .5, "5": .5}}]
        value = metrics(rows)
        self.assertEqual(value["mean_rank_found_only"], 1)
        self.assertEqual(value["mean_censored_rank_6"], 3.5)
        self.assertEqual(value["mrr_at_5"], .5)

    def test_rag_scope_remains_frozen_during_extraction_fixes(self):
        from extraction_retrieval import FIELD_QUERIES
        from qa_chunking_benchmark import configured_mode
        root = Path(__file__).resolve().parent
        phase3b = root / "qa-results/embedding-phase3b"
        read = lambda p: json.loads(p.read_text(encoding="utf-8"))
        baseline = read(phase3b / "runs" / read(phase3b / "latest.json")["run_id"] / "INTEGRITY.json")["after"]
        audit = baseline["phase3_integrity"]
        files = {**audit["stable_app_files"], **audit["chatbot"],
                 **audit["frozen_sources"], **baseline["files"]}
        # Approved transport-only checkpoint/retry update; request prompts are
        # compared against the frozen smoke trace in test_groq_resume.
        files["groq_analyzer.py"] = "fc270874544e51a2940e3a78beac64d98d12c7648daae81d2881f84ab7030833"
        # Approved functional fixes affect grounding, mapping and financial checks.
        # Retrieval, OCR source fixtures and chatbot methods remain frozen.
        for name, expected in files.items():
            if name in {"paddle_text.py", "ollama_analyzer.py", "extraction_guards.py", "invoice_rules.py", "main.py"}:
                continue
            with self.subTest(file=name):
                self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), expected)
        self.assertEqual(FIELD_QUERIES, audit["queries"])
        self.assertEqual(configured_mode()["code_default"], "legacy")
        self.assertEqual(embeddings.embedding_spec().name, "minilm")


if __name__ == "__main__":
    unittest.main()
