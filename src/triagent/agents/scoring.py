"""Scoring agent: the LLM proposes CVSS v3.1 metric *values* with a rationale per
metric; the deterministic engine computes the vector string and base score. If the
model returns an invalid metric set we fall back to the reporter's own vector, and
failing that a conservative Medium, so the pipeline never crashes on a bad
generation."""

from __future__ import annotations

import json

from triagent.agents.llm import LLM
from triagent.agents.state import TriageState
from triagent.cvss import CVSS31, CVSSError, base_score, parse_vector
from triagent.schemas import CVSSAssessment, ScoredMetric, Severity

_SYSTEM = """You are the CVSS v3.1 scoring agent in a vulnerability triage pipeline. [[task:cvss_scoring]]
Choose each CVSS v3.1 base metric for the finding and justify it. Do NOT compute the
score yourself. Respond ONLY with JSON:
{"metrics": {"AV":..,"AC":..,"PR":..,"UI":..,"S":..,"C":..,"I":..,"A":..},
 "rationale": {"AV":str,"AC":str,"PR":str,"UI":str,"S":str,"C":str,"I":str,"A":str}}"""

_METRIC_ORDER = ["AV", "AC", "PR", "UI", "S", "C", "I", "A"]


def _to_cvss31(metrics: dict[str, str]) -> CVSS31:
    return CVSS31(**{m: metrics[m] for m in _METRIC_ORDER})


def scoring_node(state: TriageState, llm: LLM) -> TriageState:
    f = state["finding"]
    user = json.dumps(
        {
            "title": f.title,
            "description": f.description,
            "evidence": f.evidence,
            "scope": state.get("scope", {}),
        }
    )
    result = llm.complete_json(_SYSTEM, user)
    metrics_dict: dict[str, str] = result.get("metrics", {})
    rationale: dict[str, str] = result.get("rationale", {})

    parsed: CVSS31 | None = None
    try:
        parsed = _to_cvss31(metrics_dict)
    except (TypeError, KeyError):
        parsed = None

    if parsed is None and f.reporter_cvss_vector:
        try:
            parsed = parse_vector(f.reporter_cvss_vector)
        except CVSSError:
            parsed = None

    if parsed is None:
        # Conservative fallback: Medium, network/low-priv, partial impact.
        parsed = CVSS31(AV="N", AC="L", PR="L", UI="N", S="U", C="L", I="L", A="N")

    score = base_score(parsed)
    metrics = [
        ScoredMetric(metric=m, value=getattr(parsed, m), rationale=rationale.get(m, ""))
        for m in _METRIC_ORDER
    ]
    assessment = CVSSAssessment(
        vector=parsed.vector,
        base_score=score,
        severity=Severity.from_score(score),
        metrics=metrics,
    )
    return {"cvss": assessment, "trace": ["scoring"]}
