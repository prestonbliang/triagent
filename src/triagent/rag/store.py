"""Vector store abstraction with two backends:

- ``memory``  : NumPy cosine search, zero dependencies, used in tests/CI and demos.
- ``pgvector``: Postgres + pgvector for production, with *hybrid* search — dense
  cosine similarity blended with a lexical signal so exact identifiers (a CVE id,
  a product name, a CWE) aren't lost by pure embeddings.

Both implement the same :class:`VectorStore` interface, so the agent graph and eval
harness never know which one they're talking to.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from triagent.config import Settings, get_settings
from triagent.rag.embeddings import Embedder, get_embedder
from triagent.schemas import Document


@dataclass
class Retrieved:
    document: Document
    score: float  # cosine similarity in [-1, 1], typically [0, 1] for normalized text


class VectorStore(ABC):
    @abstractmethod
    def add(self, docs: list[Document]) -> int: ...

    @abstractmethod
    def search(self, query: str, k: int) -> list[Retrieved]: ...

    @abstractmethod
    def count(self) -> int: ...


def _lexical_overlap(query: str, text: str) -> float:
    """Jaccard overlap of lowercased token sets — cheap lexical signal for hybrid."""
    import re

    tok = re.compile(r"[a-z0-9]+")
    q = set(tok.findall(query.lower()))
    t = set(tok.findall(text.lower()))
    if not q or not t:
        return 0.0
    return len(q & t) / len(q | t)


class MemoryStore(VectorStore):
    def __init__(self, embedder: Embedder, hybrid_alpha: float = 0.85) -> None:
        self.embedder = embedder
        self.hybrid_alpha = hybrid_alpha  # weight on dense vs lexical
        self._docs: list[Document] = []
        self._mat: np.ndarray | None = None

    def add(self, docs: list[Document]) -> int:
        if not docs:
            return 0
        vecs = self.embedder.encode([d.text for d in docs])
        self._mat = vecs if self._mat is None else np.vstack([self._mat, vecs])
        self._docs.extend(docs)
        return len(docs)

    def search(self, query: str, k: int) -> list[Retrieved]:
        if self._mat is None or not self._docs:
            return []
        q = self.embedder.encode_one(query)
        dense = self._mat @ q  # cosine, both sides normalized
        scored: list[Retrieved] = []
        for i, doc in enumerate(self._docs):
            lex = _lexical_overlap(query, doc.text)
            blended = self.hybrid_alpha * float(dense[i]) + (1 - self.hybrid_alpha) * lex
            scored.append(Retrieved(document=doc, score=blended))
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:k]

    def count(self) -> int:
        return len(self._docs)


class PgVectorStore(VectorStore):
    """Production store. Lazy-imports sqlalchemy/pgvector (extra: postgres).

    Uses pgvector cosine distance for the dense stage and a websearch tsquery for
    the lexical stage, blended in SQL. DDL is created on first use.
    """

    def __init__(self, embedder: Embedder, database_url: str, hybrid_alpha: float = 0.85) -> None:
        from sqlalchemy import create_engine, text  # lazy

        self.embedder = embedder
        self.hybrid_alpha = hybrid_alpha
        self._text = text
        self._engine = create_engine(database_url, pool_pre_ping=True)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        ddl = f"""
        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE TABLE IF NOT EXISTS documents (
            doc_id      TEXT PRIMARY KEY,
            source      TEXT NOT NULL,
            title       TEXT NOT NULL,
            body        TEXT NOT NULL,
            cwe         TEXT[] DEFAULT '{{}}',
            cvss_vector TEXT,
            cvss_score  REAL,
            url         TEXT,
            embedding   vector({self.embedder.dim}),
            tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', body)) STORED
        );
        CREATE INDEX IF NOT EXISTS documents_embedding_idx
            ON documents USING hnsw (embedding vector_cosine_ops);
        CREATE INDEX IF NOT EXISTS documents_tsv_idx ON documents USING gin (tsv);
        """
        with self._engine.begin() as conn:
            for stmt in filter(str.strip, ddl.split(";")):
                conn.execute(self._text(stmt))

    def add(self, docs: list[Document]) -> int:
        if not docs:
            return 0
        vecs = self.embedder.encode([d.text for d in docs])
        rows = [
            {
                "doc_id": d.doc_id,
                "source": d.source,
                "title": d.title,
                "body": d.text,
                "cwe": d.cwe,
                "cvss_vector": d.cvss_vector,
                "cvss_score": d.cvss_score,
                "url": d.url,
                "embedding": vec.tolist(),
            }
            for d, vec in zip(docs, vecs, strict=True)
        ]
        sql = self._text(
            """
            INSERT INTO documents
                (doc_id, source, title, body, cwe, cvss_vector, cvss_score, url, embedding)
            VALUES
                (:doc_id, :source, :title, :body, :cwe, :cvss_vector, :cvss_score, :url, :embedding)
            ON CONFLICT (doc_id) DO UPDATE SET
                body = EXCLUDED.body, embedding = EXCLUDED.embedding,
                cvss_vector = EXCLUDED.cvss_vector, cvss_score = EXCLUDED.cvss_score
            """
        )
        with self._engine.begin() as conn:
            conn.execute(sql, rows)
        return len(rows)

    def search(self, query: str, k: int) -> list[Retrieved]:
        qvec = self.embedder.encode_one(query).tolist()
        # Blend dense cosine similarity (1 - cosine_distance) with lexical rank.
        sql = self._text(
            """
            SELECT doc_id, source, title, body, cwe, cvss_vector, cvss_score, url,
                   :alpha * (1 - (embedding <=> :qvec)) +
                   (1 - :alpha) * COALESCE(ts_rank(tsv, websearch_to_tsquery('english', :q)), 0)
                   AS score
            FROM documents
            ORDER BY score DESC
            LIMIT :k
            """
        )
        with self._engine.connect() as conn:
            result = conn.execute(
                sql, {"qvec": str(qvec), "q": query, "alpha": self.hybrid_alpha, "k": k}
            )
            out: list[Retrieved] = []
            for row in result.mappings():
                out.append(
                    Retrieved(
                        document=Document(
                            doc_id=row["doc_id"],
                            source=row["source"],
                            title=row["title"],
                            text=row["body"],
                            cwe=list(row["cwe"] or []),
                            cvss_vector=row["cvss_vector"],
                            cvss_score=row["cvss_score"],
                            url=row["url"],
                        ),
                        score=float(row["score"]),
                    )
                )
            return out

    def count(self) -> int:
        with self._engine.connect() as conn:
            return int(conn.execute(self._text("SELECT COUNT(*) FROM documents")).scalar_one())


# Process-wide singletons so the in-memory store persists across API/agent calls
# within a run (and so ingestion and retrieval share the same store in-process).
_STORE: VectorStore | None = None


def get_store(settings: Settings | None = None, fresh: bool = False) -> VectorStore:
    global _STORE
    if _STORE is not None and not fresh:
        return _STORE
    settings = settings or get_settings()
    embedder = get_embedder(settings)
    if settings.store == "pgvector":
        _STORE = PgVectorStore(embedder, settings.database_url)
    else:
        _STORE = MemoryStore(embedder)
    return _STORE
