"""Scoring eval: how close is the model's CVSS judgment to NVD ground truth?

Metrics reported:
  - base_score_mae   : mean |predicted_base - truth_base| over the set
  - base_score_rmse  : root-mean-square error (penalizes large misses)
  - exact_vector_rate : fraction where all 8 base metrics match exactly
  - within_1_rate    : fraction where |predicted - truth| <= 1.0 CVSS point
  - severity_accuracy : fraction with the correct qualitative band
  - per_metric_accuracy : accuracy of each of AV/AC/PR/UI/S/C/I/A
  - confusion         : per-metric predicted-vs-truth confusion counts

The per-metric confusion is the useful artifact — it tells you *which* metric the
model is weak on (typically Scope), so you know exactly what to fix.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from triagent.agents.llm import LLM, get_llm
from triagent.agents.scoring import scoring_node
from triagent.agents.state import TriageState
from triagent.config import Settings, get_settings
from triagent.cvss import base_score, parse_vector
from triagent.eval.datasets import ScoringExample, load_scoring_eval
from triagent.schemas import Severity

_METRICS = ["AV", "AC", "PR", "UI", "S", "C", "I", "A"]


def _predict_vector(example: ScoringExample, llm: LLM) -> str:
    state: TriageState = {"finding": example.finding}
    out = scoring_node(state, llm=llm)
    assessment = out["cvss"]
    assert assessment is not None
    return assessment.vector


def run_scoring_eval(
    settings: Settings | None = None,
    llm: LLM | None = None,
    examples: list[ScoringExample] | None = None,
) -> dict[str, Any]:
    settings = settings or get_settings()
    llm = llm or get_llm(settings, reasoning=True)
    examples = examples or load_scoring_eval()

    abs_errors: list[float] = []
    sq_errors: list[float] = []
    exact = 0
    within_1 = 0
    severity_hits = 0
    metric_hits: dict[str, int] = defaultdict(int)
    confusion: dict[str, dict[str, int]] = {m: defaultdict(int) for m in _METRICS}
    per_example: list[dict[str, Any]] = []

    for ex in examples:
        pred_vec = _predict_vector(ex, llm)
        pred = parse_vector(pred_vec)
        truth = parse_vector(ex.ground_truth_vector)
        pred_score = base_score(pred)
        truth_score = base_score(truth)

        err = abs(pred_score - truth_score)
        abs_errors.append(err)
        sq_errors.append(err * err)
        if err <= 1.0:
            within_1 += 1
        if Severity.from_score(pred_score) is Severity.from_score(truth_score):
            severity_hits += 1

        all_match = True
        for m in _METRICS:
            pv, tv = getattr(pred, m), getattr(truth, m)
            if pv == tv:
                metric_hits[m] += 1
            else:
                all_match = False
            confusion[m][f"{tv}->{pv}"] += 1
        if all_match:
            exact += 1

        per_example.append(
            {
                "id": ex.id,
                "predicted": pred_vec,
                "truth": ex.ground_truth_vector,
                "predicted_score": pred_score,
                "truth_score": truth_score,
                "abs_error": round(err, 3),
            }
        )

    n = len(examples)
    return {
        "suite": "scoring",
        "n": n,
        "base_score_mae": round(sum(abs_errors) / n, 4),
        "base_score_rmse": round((sum(sq_errors) / n) ** 0.5, 4),
        "exact_vector_rate": round(exact / n, 4),
        "within_1_rate": round(within_1 / n, 4),
        "severity_accuracy": round(severity_hits / n, 4),
        "per_metric_accuracy": {m: round(metric_hits[m] / n, 4) for m in _METRICS},
        "confusion": {m: dict(confusion[m]) for m in _METRICS},
        "examples": per_example,
    }
