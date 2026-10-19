from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np

from extraction_retrieval import ExtractionRetriever, FIELD_QUERIES, source_passages
from qa_chunk_embedding_cache import ChunkEmbeddingCache
from qa_query_embedding_cache import QueryEmbeddingCache, QueryEncoder, QueryCachingEncoder
from qa_query_cache_benchmark import query_inventory


class QueryCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.identity = {"alias": "minilm", "model": "repo/minilm", "revision": "rev1",
            "dimension": 2, "normalize_embeddings": True, "dtype": "torch.float32",
            "max_seq_length": 256, "batch_size": 32, "prompts": {}, "default_prompt_name": None}
        self.queries = ["total TTC", "numero facture"]
        self.model = Mock()
        self.model.encode.side_effect = lambda texts, **kwargs: np.tile(np.array([[1., 0.]], dtype=np.float32), (len(texts), 1))

    def encoder(self, mode="memory", identity=None):
        return QueryEncoder(self.model, self.root, identity or self.identity, storage=mode)

    def test_inventory_has_22_literal_fixed_queries_no_dynamic_variants(self):
        inventory = query_inventory()
        self.assertEqual(inventory["count"], 22)
        self.assertEqual(inventory["fixed"], inventory["field_specific_fixed"])
        self.assertEqual(inventory["document_dependent"], 0)
        self.assertEqual(inventory["dynamically_generated"], 0)
        self.assertEqual([q["text"] for q in inventory["ordered_queries"]], list(FIELD_QUERIES.values()))

    def test_off_recomputes_despite_existing_cache(self):
        self.encoder("disk").encode(self.queries, normalize_embeddings=True)
        off = self.encoder("off")
        for _ in range(2):
            off.encode(self.queries, normalize_embeddings=True)
            self.assertFalse(off.last_event["cache_hit"])
        self.assertEqual(self.model.encode.call_count, 3)

    def test_memory_encodes_once_and_returns_immutable_matrix(self):
        encoder = self.encoder()
        first = encoder.encode(self.queries, normalize_embeddings=True)
        second = encoder.encode(self.queries, normalize_embeddings=True)
        self.assertIs(first, second)
        self.assertEqual(encoder.last_event["hit_source"], "memory")
        self.model.encode.assert_called_once_with(self.queries, normalize_embeddings=True)
        with self.assertRaises(ValueError):
            second[0, 0] = 0
        with self.assertRaises(ValueError):
            second.setflags(write=True)
        self.assertEqual(encoder.memory_bytes(), 16)

    def test_restart_preloads_from_disk_without_encoding(self):
        original = self.encoder("disk").encode(self.queries, normalize_embeddings=True)
        restarted = self.encoder("memory")
        np.testing.assert_array_equal(restarted.encode(self.queries, normalize_embeddings=True), original)
        self.assertEqual(restarted.last_event["hit_source"], "disk")
        restarted.encode(self.queries, normalize_embeddings=True)
        self.assertEqual(restarted.last_event["hit_source"], "memory")
        self.assertEqual(self.model.encode.call_count, 1)

    def test_disk_mode_validates_reads_every_time(self):
        encoder = self.encoder("disk")
        encoder.encode(self.queries, normalize_embeddings=True)
        for _ in range(2):
            encoder.encode(self.queries, normalize_embeddings=True)
            self.assertEqual(encoder.last_event["hit_source"], "disk")
        self.assertEqual(self.model.encode.call_count, 1)
        self.assertEqual(encoder.memory_bytes(), 0)

    def test_exact_text_case_whitespace_and_order_invalidate(self):
        encoder = self.encoder()
        variants = [self.queries, list(reversed(self.queries)), ["total ttc", "numero facture"],
                    ["total TTC ", "numero facture"], ["total TTC", "numero FACTURE"]]
        for texts in variants:
            encoder.encode(texts, normalize_embeddings=True)
            self.assertFalse(encoder.last_event["cache_hit"])
        self.assertEqual(self.model.encode.call_count, 5)
        self.assertEqual(len(encoder.memory), 4)
        encoder.encode(self.queries, normalize_embeddings=True)
        self.assertEqual(encoder.last_event["hit_source"], "disk")

    def test_model_revision_dimension_and_configuration_namespaces_separated(self):
        base = QueryEmbeddingCache(self.root, self.identity)
        base.put(self.queries, np.eye(2, dtype=np.float32))
        for change in ({"model": "BAAI/bge-m3", "alias": "bge-m3", "dimension": 1024},
                       {"revision": "rev2"}, {"dimension": 3}, {"max_seq_length": 8192},
                       {"batch_size": 16}, {"prompts": {"query": "prefix"}},
                       {"default_prompt_name": "query"}, {"pooling_sha256": "different"}):
            other = QueryEmbeddingCache(self.root, {**self.identity, **change})
            self.assertNotEqual(base.path, other.path)
            self.assertFalse(other.contains(self.queries))
            with self.assertRaises(FileNotFoundError):
                other.get(self.queries)

    def test_text_change_for_same_field_is_a_miss(self):
        encoder = self.encoder()
        fields = {"total_amount": "total TTC"}
        encoder.encode(list(fields.values()), normalize_embeddings=True)
        fields["total_amount"] = "net a payer"
        encoder.encode(list(fields.values()), normalize_embeddings=True)
        self.assertFalse(encoder.last_event["cache_hit"])
        self.assertEqual(self.model.encode.call_count, 2)

    def test_missing_identity_or_wrong_dtype_normalization_refused(self):
        for key in self.identity:
            identity = deepcopy(self.identity)
            del identity[key]
            with self.assertRaises(ValueError):
                self.encoder(identity=identity)
        for change in ({"dtype": "torch.float16"}, {"normalize_embeddings": False},
                       {"revision": ""}, {"max_seq_length": 0}):
            with self.assertRaises(ValueError):
                self.encoder(identity={**self.identity, **change})

    def test_foreign_metadata_corruption_and_partial_entry_refused(self):
        encoder = self.encoder("disk")
        encoder.encode(self.queries, normalize_embeddings=True)
        cache = encoder.cache
        metadata_path = cache.path / f"{cache.key(self.queries)}.json"
        original = metadata_path.read_text(encoding="utf-8")
        changed = json.loads(original)
        changed["identity"]["model"] = "BAAI/bge-m3"
        metadata_path.write_text(json.dumps(changed), encoding="utf-8")
        with self.assertRaises(ValueError):
            encoder.encode(self.queries, normalize_embeddings=True)
        metadata_path.write_text(original, encoding="utf-8")
        (cache.path / f"{cache.key(self.queries)}.npy").write_bytes(b"bad")
        with self.assertRaises(ValueError):
            encoder.encode(self.queries, normalize_embeddings=True)
        (cache.path / f"{cache.key(['partial'])}.npy").write_bytes(b"incomplete")
        with self.assertRaises(ValueError):
            encoder.encode(["partial"], normalize_embeddings=True)
        self.assertEqual(self.model.encode.call_count, 1)

    def test_chunk_cache_cannot_be_read_as_query_cache(self):
        chunks = ChunkEmbeddingCache(self.root, self.identity, "doc", "legacy")
        chunks.put(self.queries, np.eye(2, dtype=np.float32))
        queries = QueryEmbeddingCache(self.root, self.identity)
        self.assertNotEqual(chunks.path, queries.path)
        self.assertFalse(queries.contains(self.queries))

    def test_concurrent_memory_read_encodes_only_once(self):
        encoder = self.encoder()
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: encoder.encode(self.queries, normalize_embeddings=True), range(8)))
        self.assertEqual(self.model.encode.call_count, 1)
        self.assertTrue(all(value is results[0] for value in results))

    def test_no_cache_mode_silently_replaces_invalid_configuration(self):
        with self.assertRaises(ValueError):
            self.encoder("unknown")
        encoder = self.encoder()
        for texts in ([], [None], [""]):
            with self.assertRaises(ValueError):
                encoder.encode(texts, normalize_embeddings=True)
        with self.assertRaises(ValueError):
            encoder.encode(self.queries, normalize_embeddings=False)

    def test_every_storage_mode_keeps_retriever_outputs_and_scores(self):
        text = "Fournisseur ABC\nFACTURE 123\nDate 01/01/2026\nTotal HT 10\nTVA 1.9\nTotal TTC 11.9"
        texts = [p["text"] for p in source_passages(text)]
        chunk_root = self.root / "chunks"
        ChunkEmbeddingCache(chunk_root, self.identity, "doc", "legacy").put(texts, self.model.encode(texts))
        original = ExtractionRetriever()
        original.load_attempted, original.model, original.backend = True, self.model, "bm25+sentence_transformers"
        expected = original.retrieve(text, top_k=5, chunking_mode="legacy")
        for mode in ("off", "disk", "memory"):
            query = self.encoder(mode)
            adapter = QueryCachingEncoder(self.model, chunk_root, self.identity, list(FIELD_QUERIES.values()), query)
            original.model = adapter
            for _ in range(2):
                adapter.begin_document("doc", texts)
                actual = original.retrieve(text, top_k=5, chunking_mode="legacy")
                self.assertIs(original.model, adapter)
                self.assertEqual(expected, actual)
                self.assertTrue(adapter.events[1]["cache_hit"])


if __name__ == "__main__":
    unittest.main()
