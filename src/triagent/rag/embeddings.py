"""Embeddings behind a tiny interface, so the retrieval layer runs offline in CI
(deterministic hash embedder) and swaps to a real sentence-transformers model in
production by flipping one env var.

The fake embedder is not random: it produces stable, L2-normalized vectors from
token hashing, so semantically-overlapping texts (shared tokens) land closer than
unrelated ones. That is enough to exercise and *evaluate* the retrieval plumbing
deterministically — the eval harness reports real recall@k on it — while the prod
embedder is what you actually ship.
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod

import numpy as np

from triagent.config import Settings, get_settings

_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(ABC):
    dim: int

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalized row vectors."""

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


class HashEmbedder(Embedder):
    """Deterministic bag-of-hashed-tokens embedder. No model download, no network."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        tokens = _TOKEN.findall(text.lower())
        if not tokens:
            return v
        for tok in tokens:
            h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "big")
            idx = h % self.dim
            sign = 1.0 if (h >> 63) & 1 else -1.0
            v[idx] += sign
        return v

    def encode(self, texts: list[str]) -> np.ndarray:
        mat = np.vstack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim))
        return _l2_normalize(mat)


class SentenceTransformerEmbedder(Embedder):
    """Production embedder. Lazy-imports sentence-transformers (extra: models)."""

    def __init__(self, model_name: str, dim: int) -> None:
        from sentence_transformers import SentenceTransformer  # lazy

        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension() or dim

    def encode(self, texts: list[str]) -> np.ndarray:
        emb = self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        return emb.astype(np.float32)


def get_embedder(settings: Settings | None = None) -> Embedder:
    settings = settings or get_settings()
    if settings.embedder == "sentence-transformers":
        return SentenceTransformerEmbedder(settings.embed_model, settings.embed_dim)
    return HashEmbedder(dim=settings.embed_dim)
