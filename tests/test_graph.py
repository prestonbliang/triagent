"""End-to-end orchestration tests, run entirely offline with the FakeLLM.

These prove the graph wiring: the happy path produces a scored, drafted,
critic-approved report; the short-circuit fires on a near-duplicate; and the
revise loop is bounded by the revision budget.
"""

from triagent.agents.graph import build_graph, triage
from triagent.agents.llm import FakeLLM
from triagent.schemas import Finding, Severity


def _graph(settings, store, reranker):
    llm = FakeLLM()
    return build_graph(
        llm=llm, reasoning_llm=llm, store=store, reranker=reranker, settings=settings
    )


def test_full_pipeline_rce(settings, corpus_store, reranker):
    finding = Finding(
        finding_id="F-001",
        title="Unauthenticated RCE via deserialization on API endpoint",
        description=(
            "A remote, unauthenticated attacker can send a crafted JSON payload to the "
            "public API which is deserialized unsafely, resulting in remote code execution "
            "on the server."
        ),
        target="api.example.com",
        program="Example Public Bounty",
        evidence="POST /api/v1/import with serialized object; observed arbitrary command execution.",
    )
    graph = _graph(settings, corpus_store, reranker)
    result = triage(finding, graph=graph)

    # Scope extracted, scored, drafted, and reviewed.
    assert result.scope.get("vuln_class") == "Remote Code Execution"
    assert result.cvss is not None
    # Unauthenticated network RCE -> Critical.
    assert result.cvss.severity is Severity.CRITICAL
    assert result.cvss.base_score >= 9.0
    assert result.report_markdown and "## Steps to Reproduce" in result.report_markdown
    assert result.quality is not None and result.quality.passed
    assert result.awaiting_human_review is True
    # Every stage recorded in the trace, drafting before critic.
    assert result.trace[:3] == ["scope", "dedup", "scoring"]
    assert "drafting" in result.trace and "critic" in result.trace


def test_duplicate_short_circuits(settings, corpus_store, reranker):
    # A finding that restates Heartbleed almost verbatim should trip the dup threshold
    # and skip scoring/drafting entirely.
    settings.dup_threshold = 0.2  # force the short-circuit for the test
    finding = Finding(
        finding_id="F-002",
        title="OpenSSL TLS DTLS Heartbeat buffer over-read Heartbleed",
        description=(
            "The TLS and DTLS implementations in OpenSSL do not properly handle Heartbeat "
            "Extension packets, allowing remote attackers to obtain sensitive information "
            "from process memory via crafted packets that trigger a buffer over-read, aka "
            "Heartbleed."
        ),
        target="vpn.example.com",
    )
    graph = _graph(settings, corpus_store, reranker)
    result = triage(finding, graph=graph)

    assert result.likely_duplicate is True
    assert result.duplicates[0].doc_id == "CVE-2014-0160"
    # Short-circuited: no scoring or drafting ran.
    assert "scoring" not in result.trace
    assert result.report_markdown is None


def test_revise_loop_is_bounded(settings, corpus_store, reranker):
    # A critic that always fails must still terminate at the revision budget.
    class AlwaysFailCritic(FakeLLM):
        def _task_critic(self, p):
            return {"passed": False, "score": 0.1, "issues": ["forced failure"]}

    settings.max_revisions = 2
    llm = AlwaysFailCritic()
    graph = build_graph(
        llm=llm, reasoning_llm=llm, store=corpus_store, reranker=reranker, settings=settings
    )
    finding = Finding(
        finding_id="F-003",
        title="Reflected XSS in search parameter",
        description="A reflected cross-site scripting issue in the q parameter.",
        target="www.example.com",
        evidence="GET /search?q=<script>...",
    )
    result = triage(finding, graph=graph)
    assert result.quality is not None and result.quality.passed is False
    # Bounded: drafting runs at most max_revisions times, no infinite loop.
    assert result.revisions == settings.max_revisions
    assert result.trace.count("drafting") == settings.max_revisions
