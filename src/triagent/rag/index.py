"""Thin ingestion entrypoint: push normalized Documents into the active store."""

from __future__ import annotations

from triagent.rag.store import get_store
from triagent.schemas import Document


def index_documents(docs: list[Document]) -> int:
    store = get_store()
    return store.add(docs)
