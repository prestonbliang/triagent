"""The LangGraph supervisor.

Topology::

    scope → dedup ─┬─(likely duplicate)────────────────► END
                   └─(novel)─► scoring → drafting → critic ─┬─(pass / out of revisions)─► END
                                                    ▲        │
                                                    └───(fail, revise)───┘

Two conditional edges do the real orchestration work: a *short-circuit* that skips
the expensive scoring+drafting when the finding is a near-certain duplicate, and a
*revise loop* that sends a failing draft back to the writer until the critic passes
or the revision budget is spent. Dependencies (LLM, store, reranker, settings) are
bound into the nodes at build time so the graph itself stays pure and the whole
thing runs against the FakeLLM offline.
"""

from __future__ import annotations

from functools import partial

from langgraph.graph import END, START, StateGraph

from triagent.agents.critic import critic_node
from triagent.agents.dedup import dedup_node
from triagent.agents.drafting import drafting_node
from triagent.agents.llm import LLM, get_llm
from triagent.agents.scope import scope_node
from triagent.agents.scoring import scoring_node
from triagent.agents.state import TriageState
from triagent.config import Settings, get_settings
from triagent.rag.rerank import Reranker, get_reranker
from triagent.rag.store import VectorStore, get_store
from triagent.schemas import Finding, TriageResult


def _route_after_dedup(state: TriageState) -> str:
    return "duplicate" if state.get("likely_duplicate") else "novel"


def _route_after_critic(state: TriageState, settings: Settings) -> str:
    quality = state.get("quality")
    if quality and quality.passed:
        return "done"
    if state.get("revisions", 0) >= settings.max_revisions:
        return "done"
    return "revise"


def build_graph(
    *,
    llm: LLM | None = None,
    reasoning_llm: LLM | None = None,
    store: VectorStore | None = None,
    reranker: Reranker | None = None,
    settings: Settings | None = None,
):
    """Compile and return the triage graph. All deps optional (default from config)."""
    settings = settings or get_settings()
    llm = llm or get_llm(settings)
    reasoning_llm = reasoning_llm or get_llm(settings, reasoning=True)
    store = store or get_store(settings)
    reranker = reranker or get_reranker(settings)

    g = StateGraph(TriageState)

    g.add_node("scope", partial(scope_node, llm=llm))
    g.add_node(
        "dedup",
        partial(dedup_node, llm=llm, store=store, reranker=reranker, settings=settings),
    )
    # Scoring and critique use the stronger reasoning model in production.
    g.add_node("scoring", partial(scoring_node, llm=reasoning_llm))
    g.add_node("drafting", partial(drafting_node, llm=llm))
    g.add_node("critic", partial(critic_node, llm=reasoning_llm))

    g.add_edge(START, "scope")
    g.add_edge("scope", "dedup")
    g.add_conditional_edges(
        "dedup", _route_after_dedup, {"duplicate": END, "novel": "scoring"}
    )
    g.add_edge("scoring", "drafting")
    g.add_edge("drafting", "critic")
    g.add_conditional_edges(
        "critic",
        partial(_route_after_critic, settings=settings),
        {"revise": "drafting", "done": END},
    )
    return g.compile()


def _to_result(finding: Finding, state: dict) -> TriageResult:
    return TriageResult(
        finding_id=finding.finding_id,
        scope=state.get("scope", {}),
        likely_duplicate=state.get("likely_duplicate", False),
        duplicates=state.get("duplicates", []),
        cvss=state.get("cvss"),
        report_markdown=state.get("report_markdown"),
        quality=state.get("quality"),
        revisions=state.get("revisions", 0),
        trace=state.get("trace", []),
        awaiting_human_review=True,  # nothing is ever auto-submitted
    )


def triage(finding: Finding, *, graph=None, **build_kwargs) -> TriageResult:
    """Run one finding through the pipeline and return a structured result."""
    graph = graph or build_graph(**build_kwargs)
    final = graph.invoke({"finding": finding, "revisions": 0, "trace": []})
    return _to_result(finding, final)
