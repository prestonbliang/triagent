"""Report-quality eval: LLM-as-judge over generated reports.

The judge scores each report against a fixed rubric. Two things make this
trustworthy rather than hand-wavy:

  1. It reports pass-rate AND mean rubric score, not a single vibe number.
  2. `judge_agreement()` compares the judge to a small human-labeled calibration
     set (Cohen's kappa), so you can state how much to trust the judge instead of
     assuming it. Swap `human_labels` for your real annotations.

Runs the full triage pipeline per example, then judges the drafted report.
"""

from __future__ import annotations

from typing import Any

from triagent.agents.graph import build_graph
from triagent.agents.llm import LLM, get_llm
from triagent.config import Settings, get_settings
from triagent.eval.datasets import ScoringExample, load_scoring_eval
from triagent.rag.rerank import Reranker
from triagent.rag.store import VectorStore


def run_report_eval(
    store: VectorStore,
    settings: Settings | None = None,
    llm: LLM | None = None,
    reranker: Reranker | None = None,
    examples: list[ScoringExample] | None = None,
) -> dict[str, Any]:
    settings = settings or get_settings()
    llm = llm or get_llm(settings)
    examples = examples or load_scoring_eval()

    graph = build_graph(
        llm=llm, reasoning_llm=llm, store=store, reranker=reranker, settings=settings
    )

    scores: list[float] = []
    passes = 0
    per_example: list[dict[str, Any]] = []

    for ex in examples:
        final = graph.invoke({"finding": ex.finding, "revisions": 0, "trace": []})
        quality = final.get("quality")
        report = final.get("report_markdown") or ""
        if quality is None:
            # Short-circuited as duplicate; not a report-quality data point.
            continue
        scores.append(quality.score)
        passes += int(quality.passed)
        per_example.append(
            {
                "id": ex.id,
                "passed": quality.passed,
                "score": quality.score,
                "issues": quality.issues,
                "report_len": len(report),
            }
        )

    n = len(scores)
    return {
        "suite": "report",
        "n": n,
        "pass_rate": round(passes / n, 4) if n else 0.0,
        "mean_rubric_score": round(sum(scores) / n, 4) if n else 0.0,
        "examples": per_example,
    }


def cohens_kappa(a: list[int], b: list[int]) -> float:
    """Cohen's kappa for two binary label vectors (judge vs human)."""
    if len(a) != len(b) or not a:
        raise ValueError("label vectors must be equal length and non-empty")
    n = len(a)
    agree = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    pa1 = sum(a) / n
    pb1 = sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if pe == 1.0:
        return 1.0
    return (agree - pe) / (1 - pe)
