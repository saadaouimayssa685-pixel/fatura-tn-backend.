"""Experimental document-chunk cache. Never imported by the application pipeline."""
from copy import deepcopy
import time

from qa_embedding_benchmark import VectorCache


class ChunkEmbeddingCache(VectorCache):
    """Path identifies model/config/document/mode; filename hashes ordered texts."""

    def __init__(self, root, identity, document_id, chunking_mode):
        required = {"alias", "model", "revision", "dimension", "normalize_embeddings", "dtype"}
        if not required <= identity.keys():
            raise ValueError("Incomplete model identity")
        if (not isinstance(document_id, str) or not document_id.strip()
                or chunking_mode not in {"legacy", "layout"}
                or not identity["model"] or not identity["revision"]
                or type(identity["dimension"]) is not int or identity["dimension"] <= 0
                or identity["normalize_embeddings"] is not True):
            raise ValueError("Invalid document/cache configuration")
        super().__init__(root, {**deepcopy(identity), "document_id": document_id,
            "chunking_mode": chunking_mode, "cache_schema": "phase3b-chunks-v1",
            "role": "document_chunks"})

    def contains(self, texts):
        key = self.key(texts)
        array = (self.path / f"{key}.npy").exists()
        metadata = (self.path / f"{key}.json").exists()
        if array != metadata:
            raise ValueError("Incomplete chunk cache entry")
        return array


class ChunkCachingEncoder:
    """Wrap only the QA model.encode calls; queries are ALWAYS encoded again."""

    def __init__(self, model, root, identity, queries):
        self.model, self.root, self.identity = model, root, deepcopy(identity)
        self.queries = list(queries)
        self.cache = None
        self.call_index = 0
        self.events = []
        self.arrays = {}

    def begin_document(self, document_id, chunk_texts, chunking_mode="legacy"):
        if self.cache is not None and self.call_index != 2:
            raise ValueError("Previous retrieval did not complete both encode calls")
        self.cache = ChunkEmbeddingCache(self.root, self.identity, document_id, chunking_mode)
        self.chunk_texts = list(chunk_texts)
        self.call_index = 0
        self.events = []
        self.arrays = {}

    def encode(self, texts, normalize_embeddings=False):
        if self.cache is None or not normalize_embeddings or self.call_index >= 2:
            raise ValueError("Unexpected encoder invocation")
        texts = list(texts)
        # Check the retriever's two known calls instead of guessing roles from text.
        kind = "queries" if self.call_index == 0 else "chunks"
        if texts != (self.queries if kind == "queries" else self.chunk_texts):
            raise ValueError("Retriever inputs changed; refusing to cache foreign texts")
        self.call_index += 1
        event = {"kind": kind, "count": len(texts), "cache_hit": False,
                 "embedding_seconds": 0.0, "cache_read_seconds": 0.0, "cache_write_seconds": 0.0}
        if kind == "chunks" and self.cache.contains(texts):
            started = time.perf_counter()
            vectors = self.cache.get(texts)
            event.update(cache_hit=True, cache_read_seconds=time.perf_counter() - started)
        else:
            started = time.perf_counter()
            vectors = self.model.encode(texts, normalize_embeddings=True)
            event["embedding_seconds"] = time.perf_counter() - started
            if kind == "chunks":
                started = time.perf_counter()
                self.cache.put(texts, vectors)
                event["cache_write_seconds"] = time.perf_counter() - started
        self.events.append(event)
        self.arrays[kind] = vectors
        return vectors
