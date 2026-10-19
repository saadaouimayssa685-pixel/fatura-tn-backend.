import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np

from qa_ingestion_cache_benchmark import EncodeMonitor, cache_snapshot, ingest_document


class IngestionCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.identity = {"alias": "minilm", "model": "repo/minilm", "revision": "rev1",
            "dimension": 2, "normalize_embeddings": True, "dtype": "torch.float32",
            "max_seq_length": 256, "batch_size": 32, "prompts": {}, "default_prompt_name": None}
        self.texts = ["Facture 0001", "Total TTC 12.600"]
        self.model = Mock()
        self.model.encode.side_effect = lambda texts, **kwargs: np.tile(np.array([[1., 0.]], dtype=np.float32), (len(texts), 1))
        self.monitor = EncodeMonitor(self.model)

    def test_cold_then_warm_one_encode_and_no_rewrite(self):
        cold, original = ingest_document(self.root, self.identity, "doc", self.texts, self.monitor)
        before = cache_snapshot(self.root)
        warm, cached = ingest_document(self.root, self.identity, "doc", self.texts, self.monitor)
        self.assertFalse(cold["cache_hit"])
        self.assertTrue(warm["cache_hit"])
        self.assertEqual(warm["chunk_embedding_seconds"], 0)
        self.assertEqual(warm["cache_write_seconds"], 0)
        self.assertEqual(self.monitor.calls, [{"count": 2, "normalize_embeddings": True}])
        self.assertEqual(before, cache_snapshot(self.root))
        np.testing.assert_array_equal(cached, original)
        self.assertGreaterEqual(cold["ingestion_seconds"], cold["chunk_embedding_seconds"] + cold["cache_write_seconds"])

    def test_all_chunks_sent_in_single_encode_call(self):
        texts = [f"Ligne {i}" for i in range(11)]
        record, _ = ingest_document(self.root, self.identity, "doc", texts, self.monitor)
        self.model.encode.assert_called_once_with(texts, normalize_embeddings=True)
        self.assertEqual(record["chunk_count"], 11)
        self.assertEqual(self.monitor.calls[0]["count"], 11)

    def test_new_instance_reuses_disk_without_any_encode(self):
        ingest_document(self.root, self.identity, "doc", self.texts, self.monitor)
        before = cache_snapshot(self.root)
        forbidden = EncodeMonitor(self.model, forbid=True)
        record, _ = ingest_document(self.root, self.identity, "doc", self.texts, forbidden)
        self.assertTrue(record["cache_hit"])
        self.assertEqual(forbidden.calls, [])
        self.assertEqual(before, cache_snapshot(self.root))

    def test_separate_python_process_reads_persisted_matrix(self):
        ingest_document(self.root, self.identity, "doc", self.texts, self.monitor)
        before = cache_snapshot(self.root)
        code = """
import json, os, sys
from pathlib import Path
from qa_ingestion_cache_benchmark import EncodeMonitor, ingest_document
record, vectors = ingest_document(Path(sys.argv[1]), json.loads(sys.argv[2]), 'doc',
                                  json.loads(sys.argv[3]), EncodeMonitor(None, forbid=True))
assert record['cache_hit'] and record['chunk_embedding_seconds'] == 0 and record['cache_write_seconds'] == 0
print(json.dumps({'pid': os.getpid(), 'vectors': vectors.tolist()}))
"""
        result = subprocess.run([sys.executable, "-c", code, str(self.root), json.dumps(self.identity),
            json.dumps(self.texts)], cwd=Path(__file__).resolve().parent, check=True, capture_output=True, text=True)
        actual = json.loads(result.stdout)
        import os
        self.assertNotEqual(actual["pid"], os.getpid())
        self.assertEqual(actual["vectors"], [[1., 0.], [1., 0.]])
        self.assertEqual(before, cache_snapshot(self.root))

    def test_restart_guard_fails_instead_of_reencoding_missing_cache(self):
        guard = EncodeMonitor(self.model, forbid=True)
        with self.assertRaises(RuntimeError):
            ingest_document(self.root, self.identity, "missing", self.texts, guard)
        self.assertFalse(self.model.encode.called)

    def test_changed_order_or_exact_text_creates_new_entry(self):
        original, _ = ingest_document(self.root, self.identity, "doc", self.texts, self.monitor)
        for texts in (list(reversed(self.texts)), [self.texts[0], "Total TTC 12.601"], [self.texts[0] + " ", self.texts[1]]):
            record, _ = ingest_document(self.root, self.identity, "doc", texts, self.monitor)
            self.assertFalse(record["cache_hit"])
            self.assertNotEqual(record["text_sha256"], original["text_sha256"])
        self.assertEqual(len(self.monitor.calls), 4)
        self.assertEqual(len(list(self.root.rglob("*.npy"))), 4)

    def test_document_and_chunking_mode_do_not_share_vectors(self):
        original, _ = ingest_document(self.root, self.identity, "doc", self.texts, self.monitor)
        for doc, mode in (("other", "legacy"), ("doc", "layout")):
            record, _ = ingest_document(self.root, self.identity, doc, self.texts, self.monitor, mode)
            self.assertFalse(record["cache_hit"])
            self.assertNotEqual(record["cache_path"], original["cache_path"])

    def test_revision_model_or_embedding_configuration_invalidates(self):
        original, _ = ingest_document(self.root, self.identity, "doc", self.texts, self.monitor)
        for change in ({"model": "repo/other", "alias": "bge-m3"}, {"revision": "rev2"},
                       {"max_seq_length": 8192}, {"batch_size": 16}, {"prompts": {"query": "p"}}):
            record, _ = ingest_document(self.root, {**self.identity, **change}, "doc", self.texts, self.monitor)
            self.assertFalse(record["cache_hit"])
            self.assertNotEqual(record["cache_path"], original["cache_path"])

    def test_corrupt_cache_is_not_silently_reused_or_overwritten(self):
        ingest_document(self.root, self.identity, "doc", self.texts, self.monitor)
        next(self.root.rglob("*.npy")).write_bytes(b"corrupted")
        before = cache_snapshot(self.root)
        with self.assertRaises(ValueError):
            ingest_document(self.root, self.identity, "doc", self.texts, self.monitor)
        self.assertEqual(len(self.monitor.calls), 1)
        self.assertEqual(before, cache_snapshot(self.root))


if __name__ == "__main__":
    unittest.main()
