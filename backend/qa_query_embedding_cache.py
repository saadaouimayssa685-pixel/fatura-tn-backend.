"""Phase 3C experiment only: exact query batches, never field-name-only keys."""
from collections import OrderedDict
from copy import deepcopy
import threading
import time

from qa_embedding_benchmark import VectorCache
from qa_chunk_embedding_cache import ChunkCachingEncoder


class QueryEmbeddingCache(VectorCache):
    """Whole-batch identity preserves original encode padding/order and precision."""

    def __init__(self, root, identity):
        required = {"alias", "model", "revision", "dimension", "normalize_embeddings", "dtype",
                    "max_seq_length", "batch_size", "prompts", "default_prompt_name"}
        if not required <= identity.keys():
            raise ValueError("Incomplete query embedding identity")
        if (not identity["model"] or not identity["revision"]
                or type(identity["dimension"]) is not int or identity["dimension"] <= 0
                or type(identity["max_seq_length"]) is not int or identity["max_seq_length"] <= 0
                or identity["normalize_embeddings"] is not True
                or identity["dtype"] != "torch.float32"):
            raise ValueError("Unsupported query embedding configuration")
        super().__init__(root, {**deepcopy(identity), "role": "fixed_query_batch",
                               "cache_schema": "phase3c-queries-v1"})

    def contains(self, texts):
        key = self.key(texts)
        matrix = (self.path / f"{key}.npy").exists()
        metadata = (self.path / f"{key}.json").exists()
        if matrix != metadata:
            raise ValueError("Incomplete query cache entry")
        return matrix


class QueryEncoder:
    """off: encode; disk: validated read; memory: immutable batch loaded once."""

    def __init__(self, model, root, identity, storage="off", max_memory_batches=4):
        if storage not in {"off", "disk", "memory"} or max_memory_batches < 1:
            raise ValueError("Invalid experimental query cache mode")
        self.model = model
        self.storage = storage
        self.cache = QueryEmbeddingCache(root, identity)
        self.memory = OrderedDict()
        self.max_memory_batches = max_memory_batches
        self.lock = threading.Lock()
        self.last_event = None

    def encode(self, texts, normalize_embeddings=False):
        import numpy as np

        if not normalize_embeddings:
            raise ValueError("Query cache requires normalized vectors")
        texts = list(texts)
        if not texts or any(not isinstance(text, str) or not text for text in texts):
            raise ValueError("Expected exact nonempty query strings")
        with self.lock:
            started = time.perf_counter()
            key = self.cache.key(texts)
            event = {"kind": "queries", "count": len(texts), "storage": self.storage,
                     "cache_hit": False, "hit_source": None, "embedding_seconds": 0.0,
                     "cache_read_seconds": 0.0, "cache_write_seconds": 0.0}
            if self.storage == "memory" and key in self.memory:
                vectors = self.memory[key]
                self.memory.move_to_end(key)
                event.update(cache_hit=True, hit_source="memory")
            elif self.storage != "off" and self.cache.contains(texts):
                read_start = time.perf_counter()
                vectors = self.cache.get(texts)
                event.update(cache_hit=True, hit_source="disk",
                             cache_read_seconds=time.perf_counter() - read_start)
            else:
                encode_start = time.perf_counter()
                vectors = self.model.encode(texts, normalize_embeddings=True)
                event["embedding_seconds"] = time.perf_counter() - encode_start
                self.cache.validate(vectors, len(texts))
                if self.storage != "off":
                    write_start = time.perf_counter()
                    self.cache.put(texts, vectors)
                    event["cache_write_seconds"] = time.perf_counter() - write_start
            if self.storage == "memory" and key not in self.memory:
                # Bytes-backed arrays cannot be made writable by the caller.
                vectors = np.frombuffer(vectors.tobytes(order="C"), dtype=np.float32).reshape(vectors.shape)
                self.memory[key] = vectors
                while len(self.memory) > self.max_memory_batches:
                    self.memory.popitem(last=False)
            event["total_seconds"] = time.perf_counter() - started
            self.last_event = event
            return vectors

    def memory_bytes(self):
        return sum(vector.nbytes for vector in self.memory.values())


class QueryCachingEncoder(ChunkCachingEncoder):
    """QA adapter: leave Phase 3B chunk-cache reads unchanged."""

    def __init__(self, model, chunk_root, identity, queries, query_encoder):
        super().__init__(model, chunk_root, identity, queries)
        self.query_encoder = query_encoder

    def encode(self, texts, normalize_embeddings=False):
        if self.call_index != 0:
            return super().encode(texts, normalize_embeddings=normalize_embeddings)
        if self.cache is None or not normalize_embeddings or list(texts) != self.queries:
            raise ValueError("Unexpected query call from retriever")
        vectors = self.query_encoder.encode(texts, normalize_embeddings=True)
        self.call_index = 1
        self.events.append(dict(self.query_encoder.last_event))
        self.arrays["queries"] = vectors
        return vectors
