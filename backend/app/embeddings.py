"""Embedding backends behind one small interface."""

import hashlib
import math
import re
from functools import lru_cache
from typing import Protocol

from app.config import get_settings

_TOKEN = re.compile(r"[\w']+", re.UNICODE)


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    """Dense multilingual embeddings. The model is loaded once per process."""

    def __init__(self, model_name: str, dim: int):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        actual = self._model.get_sentence_embedding_dimension()
        if actual != dim:
            raise ValueError(
                f"{model_name} produces {actual}-dim vectors but EMBEDDING_DIM={dim}; "
                "the database column must match the model."
            )
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
        )
        return [v.tolist() for v in vectors]


class HashingEmbedder:
    """Deterministic feature-hashing embedder (unigrams + bigrams, signed, L2-normalised).

    It has no semantic understanding, but cosine similarity still tracks lexical overlap, which
    makes it a fast, offline stand-in for CI and for trying the stack without downloading a model.
    """

    def __init__(self, dim: int):
        self.dim = dim

    def _features(self, text: str) -> list[str]:
        tokens = [t.lower() for t in _TOKEN.findall(text)]
        return tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)]

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            for feature in self._features(text):
                digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
                h = int.from_bytes(digest, "little")
                vec[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            out.append([x / norm for x in vec])
        return out


@lru_cache
def get_embedder() -> Embedder:
    s = get_settings()
    if s.embedder == "hashing":
        return HashingEmbedder(s.embedding_dim)
    return SentenceTransformerEmbedder(s.embedding_model, s.embedding_dim)
