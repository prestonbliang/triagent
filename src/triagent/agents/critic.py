"""Critic / quality-gate agent: score the draft against the rubric platforms
actually reject on (missing repro steps, unclear impact, absent remediation). If it
fails and we have revisions left, the graph loops back to drafting."""

from __future__ import annotations

import json

from triagent.agents.llm import LLM
from triagent.agents.state import TriageState
from triagent.schemas import QualityReview

_SYSTEM = """You are the quality-gate agent for vulnerability disclosure reports. [[task:critic]]
Grade the report against this rubric: (1) has a clear Summary, Impact, Steps to
Reproduce, and Remediation; (2) reproduction steps are concrete enough to verify;
(3) impact matches the stated severity. Respond ONLY with JSON:
{"passed": bool, "score": number (0..1), "issues": [str]}"""


def critic_node(state: TriageState, llm: LLM) -> TriageState:
    user = json.dumps({"report_markdown": state.get("report_markdown", "")})
    result = llm.complete_json(_SYSTEM, user)
    review = QualityReview(
        passed=bool(result.get("passed", False)),
        score=float(result.get("score", 0.0)),
        issues=list(result.get("issues", [])),
    )
    revisions = state.get("revisions", 0)
    return {"quality": review, "revisions": revisions + 1, "trace": ["critic"]}
