"""Reranking stage. Retrieval optimizes recall (get the right doc *somewhere* in
the top-k); reranking optimizes precision (get it to the *top*). A cross-encoder
scores each (query, doc) pair jointly, which is far more accurate than the
bi-encoder cosine used for first-stage retrieval — and the single biggest quality
lever in a RAG system.

Interface stays swappable: ``off`` (passthrough), ``identity`` (keep retrieval
order, offline default), or ``cross-encoder`` (production, lazy-imports models).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from triagent.config import Settings, get_settings
from triagent.rag.store import Retrieved


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, hits: list[Retrieved], top_k: int) -> list[Retrieved]: ...


class IdentityReranker(Reranker):
    """Preserve first-stage order; just truncate to top_k. Offline default."""

    def rerank(self, query: str, hits: list[Retrieved], top_k: int) -> list[Retrieved]:
        return hits[:top_k]


class CrossEncoderReranker(Reranker):
    """Production reranker. Lazy-imports sentence-transformers CrossEncoder."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder  # lazy

        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, hits: list[Retrieved], top_k: int) -> list[Retrieved]:
        if not hits:
            return []
        pairs = [(query, h.document.text) for h in hits]
        scores = self._model.predict(pairs)
        reranked = sorted(
            (Retrieved(document=h.document, score=float(s)) for h, s in zip(hits, scores, strict=True)),
            key=lambda r: r.score,
            reverse=True,
        )
        return reranked[:top_k]


def get_reranker(settings: Settings | None = None) -> Reranker:
    settings = settings or get_settings()
    if settings.reranker == "cross-encoder":
        return CrossEncoderReranker(settings.rerank_model)
    # "off" and "identity" both preserve order; truncation happens in the caller.
    return IdentityReranker()
