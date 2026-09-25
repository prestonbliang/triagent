"""Drafting agent: write a submission-ready report grounded in the finding, the
computed CVSS, and retrieved prior art. On a revision pass it receives the critic's
issues and must address them."""

from __future__ import annotations

import json

from triagent.agents.llm import LLM
from triagent.agents.state import TriageState

_SYSTEM = """You are the report-drafting agent in a vulnerability triage pipeline. [[task:drafting]]
Write a clear, reproducible disclosure report in Markdown with these sections:
## Summary, ## Impact, ## Steps to Reproduce, ## Remediation.
Ground every claim in the provided finding, CVSS assessment, and prior art. If prior
issues are provided, fix them. Respond ONLY with JSON: {"report_markdown": str}"""


def drafting_node(state: TriageState, llm: LLM) -> TriageState:
    f = state["finding"]
    cvss = state.get("cvss")
    quality = state.get("quality")
    user = json.dumps(
        {
            "finding": f.model_dump(),
            "scope": state.get("scope", {}),
            "cvss": cvss.model_dump() if cvss else {},
            "duplicates": [d.model_dump() for d in state.get("duplicates", [])[:3]],
            "prior_issues": quality.issues if quality else [],
        }
    )
    report = llm.complete_json(_SYSTEM, user).get("report_markdown", "")
    return {"report_markdown": report, "trace": ["drafting"]}
