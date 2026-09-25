"""Dedup agent: retrieve prior art and decide whether the finding is likely a
duplicate of something already public. The decision is made in code against a
tuned similarity threshold (auditable, tunable, evaluated); the LLM only writes
the human-readable justification."""

from __future__ import annotations

import json

from triagent.agents.llm import LLM
from triagent.agents.state import TriageState
from triagent.config import Settings
from triagent.rag.rerank import Reranker
from triagent.rag.store import VectorStore
from triagent.schemas import DuplicateMatch

_SYSTEM = """You are the deduplication agent in a vulnerability triage pipeline. [[task:dedup]]
You are given the finding and candidate prior-art documents with similarity scores.
Summarize the relationship in one or two sentences. Respond ONLY with JSON:
{"assessment": str}"""


def _query_text(state: TriageState) -> str:
    f = state["finding"]
    scope = state.get("scope", {})
    parts = [f.title, f.description]
    if scope.get("vuln_class"):
        parts.append(scope["vuln_class"])
    if scope.get("cwe"):
        parts.append(" ".join(scope["cwe"]))
    return "\n".join(p for p in parts if p)


def dedup_node(
    state: TriageState,
    llm: LLM,
    store: VectorStore,
    reranker: Reranker,
    settings: Settings,
) -> TriageState:
    query = _query_text(state)
    hits = store.search(query, k=settings.retrieval_k)
    reranked = reranker.rerank(query, hits, top_k=settings.rerank_k)

    duplicates = [
        DuplicateMatch(
            doc_id=r.document.doc_id,
            title=r.document.title,
            score=round(float(r.score), 4),
            url=r.document.url,
        )
        for r in reranked
    ]
    likely = bool(duplicates) and duplicates[0].score >= settings.dup_threshold

    if duplicates:
        user = json.dumps(
            {
                "finding": {"title": state["finding"].title},
                "candidates": [d.model_dump() for d in duplicates[:5]],
            }
        )
        assessment = llm.complete_json(_SYSTEM, user).get("assessment")
        if assessment:
            duplicates[0].why = assessment

    return {
        "duplicates": duplicates,
        "likely_duplicate": likely,
        "trace": ["dedup"],
    }
