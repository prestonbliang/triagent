"""Retrieval / dedup eval: recall@k and MRR against labeled query->relevant pairs.

recall@k    : fraction of queries whose relevant doc appears in the top-k
mrr         : mean reciprocal rank of the first relevant doc
Measures the retrieval+rerank stack the dedup agent depends on. Swap in the full
CVE-linked disclosed-report set to get production numbers; the harness is identical.
"""

from __future__ import annotations

from typing import Any

from triagent.config import Settings, get_settings
from triagent.eval.datasets import RetrievalExample, load_retrieval_eval
from triagent.rag.rerank import Reranker, get_reranker
from triagent.rag.store import VectorStore


def _rank_of_first_relevant(ranked_ids: list[str], relevant: set[str]) -> int | None:
    for i, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant:
            return i
    return None


def run_retrieval_eval(
    store: VectorStore,
    settings: Settings | None = None,
    reranker: Reranker | None = None,
    examples: list[RetrievalExample] | None = None,
    ks: tuple[int, ...] = (1, 3, 5),
) -> dict[str, Any]:
    settings = settings or get_settings()
    reranker = reranker or get_reranker(settings)
    examples = examples or load_retrieval_eval()

    recall_hits = {k: 0 for k in ks}
    reciprocal_ranks: list[float] = []
    per_example: list[dict[str, Any]] = []

    for ex in examples:
        relevant = set(ex.relevant)
        hits = store.search(ex.query, k=max(settings.retrieval_k, max(ks)))
        reranked = reranker.rerank(ex.query, hits, top_k=max(ks))
        ranked_ids = [r.document.doc_id for r in reranked]

        rank = _rank_of_first_relevant(ranked_ids, relevant)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        for k in ks:
            if rank is not None and rank <= k:
                recall_hits[k] += 1

        per_example.append(
            {"query": ex.query, "relevant": ex.relevant, "top": ranked_ids[:3], "rank": rank}
        )

    n = len(examples)
    return {
        "suite": "retrieval",
        "n": n,
        "recall_at_k": {f"recall@{k}": round(recall_hits[k] / n, 4) for k in ks},
        "mrr": round(sum(reciprocal_ranks) / n, 4),
        "examples": per_example,
    }
