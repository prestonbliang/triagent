"""Scope agent: extract the structured facts a triager needs from free text —
affected product, version, endpoint, weakness class, CWE. This is the token-
classification/NER step of the pipeline (fine-tuned extractor in production;
keyword+LLM here)."""

from __future__ import annotations

import json

from triagent.agents.llm import LLM
from triagent.agents.state import TriageState

_SYSTEM = """You are the scope-extraction agent in a vulnerability triage pipeline. [[task:scope]]
Given a security finding, extract structured facts. Respond ONLY with JSON:
{"product": str|null, "version": str|null, "endpoint": str|null,
 "cwe": [str], "vuln_class": str}"""


def scope_node(state: TriageState, llm: LLM) -> TriageState:
    f = state["finding"]
    user = json.dumps(
        {
            "title": f.title,
            "description": f.description,
            "target": f.target,
            "evidence": f.evidence,
        }
    )
    result = llm.complete_json(_SYSTEM, user)
    return {"scope": result, "trace": ["scope"]}
