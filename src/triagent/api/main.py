"""FastAPI service exposing the triage pipeline with a human-in-the-loop review step.

Endpoints:
  GET  /health                 liveness + which providers are active
  POST /triage                 run a finding through the pipeline -> TriageResult
  POST /triage/{id}/decision   record the human's approve/reject decision

Nothing is ever auto-submitted to a bounty platform: /triage returns a *draft* with
awaiting_human_review=True, and a human must approve it. That HITL gate is a
deliberate safety and quality control, not an afterthought.
"""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel

from triagent import __version__
from triagent.agents.graph import build_graph
from triagent.config import get_settings
from triagent.schemas import Finding, TriageResult

app = FastAPI(title="Triagent", version=__version__)

# Build the graph once at startup; it holds the store + model handles.
_GRAPH = None
# In a real deployment this is a database; a dict keeps the reference impl honest.
_DECISIONS: dict[str, dict] = {}


def _graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


class Decision(BaseModel):
    action: Literal["approve", "reject", "needs_changes"]
    note: str | None = None


@app.get("/health")
def health() -> dict:
    s = get_settings()
    return {
        "status": "ok",
        "version": __version__,
        "llm_provider": s.llm_provider,
        "embedder": s.embedder,
        "reranker": s.reranker,
        "store": s.store,
    }


@app.post("/triage", response_model=TriageResult)
def triage_endpoint(finding: Finding) -> TriageResult:
    final = _graph().invoke({"finding": finding, "revisions": 0, "trace": []})
    return TriageResult(
        finding_id=finding.finding_id,
        scope=final.get("scope", {}),
        likely_duplicate=final.get("likely_duplicate", False),
        duplicates=final.get("duplicates", []),
        cvss=final.get("cvss"),
        report_markdown=final.get("report_markdown"),
        quality=final.get("quality"),
        revisions=final.get("revisions", 0),
        trace=final.get("trace", []),
        awaiting_human_review=True,
    )


@app.post("/triage/{finding_id}/decision")
def decision_endpoint(finding_id: str, decision: Decision) -> dict:
    _DECISIONS[finding_id] = decision.model_dump()
    return {"finding_id": finding_id, "recorded": decision.model_dump()}
