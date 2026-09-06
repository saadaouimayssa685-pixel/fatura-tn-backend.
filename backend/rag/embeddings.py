import os
import hashlib
import math
import re
from typing import List


class EmbeddingProvider:
    """Embeds text with sentence-transformers when available, otherwise uses hashing vectors."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", dimensions: int = 384):
        self.model_name = model_name
        self.dimensions = dimensions
        self.backend = "hashing"
        self.model = None
        self.enable_sentence_transformers = os.getenv("ENABLE_SENTENCE_TRANSFORMERS", "false").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

        if not self.enable_sentence_transformers:
            return

        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name)
            self.backend = "sentence_transformers"
        except Exception as error:
            print(f"⚠️ sentence-transformers indisponible, fallback hashing embeddings: {error}")

    def embed(self, text: str) -> List[float]:
        if self.model is not None:
            vector = self.model.encode(text, normalize_embeddings=True)
            return [float(value) for value in vector]

        return self._hashing_embedding(text)

    def _hashing_embedding(self, text: str) -> List[float]:
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[\wÀ-ÿ]+", text.lower())

        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            index = int(digest[:8], 16) % self.dimensions
            sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector

        return [value / norm for value in vector]


def cosine_similarity(left: List[float], right: List[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0

    return sum(a * b for a, b in zip(left, right))
