import json
from pathlib import Path

import pytest

from triagent.config import Settings
from triagent.ingest import cve_to_document
from triagent.rag.embeddings import HashEmbedder
from triagent.rag.rerank import IdentityReranker
from triagent.rag.store import MemoryStore

FIXTURE = Path(__file__).parent / "fixtures" / "nvd_sample.json"


@pytest.fixture
def settings() -> Settings:
    return Settings(
        llm_provider="fake",
        embedder="fake",
        reranker="identity",
        store="memory",
        embed_dim=384,
        retrieval_k=10,
        rerank_k=5,
        dup_threshold=0.83,
        max_revisions=2,
    )


@pytest.fixture
def corpus_store() -> MemoryStore:
    store = MemoryStore(HashEmbedder(dim=384))
    docs = [cve_to_document(v) for v in json.loads(FIXTURE.read_text())["vulnerabilities"]]
    store.add(docs)
    return store


@pytest.fixture
def reranker() -> IdentityReranker:
    return IdentityReranker()
