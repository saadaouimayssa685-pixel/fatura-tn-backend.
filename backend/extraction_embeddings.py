"""CPU embedding selection for invoice extraction, independent of chatbot stores."""
from dataclasses import dataclass
import os
import threading


@dataclass(frozen=True)
class EmbeddingSpec:
    name: str
    repository: str
    dimension: int


SPECS = {
    "minilm": EmbeddingSpec("minilm", "sentence-transformers/all-MiniLM-L6-v2", 384),
    "bge-m3": EmbeddingSpec("bge-m3", "BAAI/bge-m3", 1024),
}
_models = {}
_load_lock = threading.Lock()


def embedding_spec(name=None):
    selected = name if name is not None else os.getenv("RAG_EMBEDDING_MODEL", "minilm")
    if selected not in SPECS:
        raise ValueError("RAG_EMBEDDING_MODEL must be minilm or bge-m3")
    return SPECS[selected]


def load_embedding_model(name=None):
    spec = embedding_spec(name)
    with _load_lock:
        if spec.name not in _models:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(spec.repository, device="cpu", local_files_only=True)
            if model.get_sentence_embedding_dimension() != spec.dimension:
                raise ValueError(f"Unexpected embedding dimension for {spec.name}")
            _models[spec.name] = model
        return _models[spec.name]
