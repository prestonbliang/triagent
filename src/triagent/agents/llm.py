"""LLM interface + a deterministic FakeLLM.

The whole agent graph talks to this one interface, so tests and the CI eval gate
run with the FakeLLM (no keys, no network, fully reproducible) and production
flips ``TRIAGENT_LLM_PROVIDER`` to openai/anthropic.

Design choice worth noting for reviewers: the LLM proposes *CVSS metric values*
and writes prose, but it never computes the score — the deterministic engine in
``triagent.cvss`` does. That keeps arithmetic out of the model's hands (the most
common source of wrong CVSS) and makes the eval a clean measure of *judgment*.

The FakeLLM's CVSS proposer is a transparent keyword heuristic. It is deliberately
imperfect, so the eval harness reports a real, non-zero error you can then beat by
swapping in a frontier model — the whole point of the harness.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any

from triagent.config import Settings, get_settings

# Agents tag their system prompt so the FakeLLM can route deterministically.
# Real providers simply ignore the tag.
TASK_TAG = re.compile(r"\[\[task:([a-z_]+)\]\]")


class LLM(ABC):
    @abstractmethod
    def complete(self, system: str, user: str) -> str: ...

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        return _extract_json(self.complete(system, user))


def _extract_json(raw: str) -> dict[str, Any]:
    """Tolerant JSON extraction — real LLMs wrap JSON in prose or code fences."""
    raw = raw.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fence:
        raw = fence.group(1)
    else:
        brace = re.search(r"\{.*\}", raw, re.DOTALL)
        if brace:
            raw = brace.group(0)
    return json.loads(raw)


# --------------------------------------------------------------------------- #
# Deterministic fake
# --------------------------------------------------------------------------- #

_AV_HINTS = [
    (("physical", "usb", "chassis"), "P"),
    (("adjacent", "same network segment", "link-local", "bluetooth", "lan-only"), "A"),
    (("local access", "local privilege", "local attacker", "locally", "cli"), "L"),
    (("remote", "network", "internet", "http", "https", "api", "web", "url", "endpoint"), "N"),
]
_AC_HIGH = ("race condition", "timing", "mitm", "man-in-the-middle", "specific configuration")
_PR_NONE = ("unauthenticated", "no authentication", "without auth", "pre-auth", "anonymous")
_PR_HIGH = ("admin", "administrator", "privileged", "root ")
_PR_LOW = ("authenticated", "logged in", "user account", "low-privilege", "low privilege", "any user")
_UI_REQUIRED = (
    "click", "user interaction", "victim", "phishing",
    "open the", "opens a", "visit", "lure", "social engineering",
)
_SCOPE_CHANGED = ("sandbox escape", "other components", "scope change")
_RCE = ("remote code execution", "rce", "arbitrary code", "command injection", "deserialization")
_DOS = ("denial of service", "dos ", "crash", "resource exhaustion", "unavailable")
_DISCLOSURE = (
    "information disclosure", "sensitive information", "data leak",
    "read arbitrary", "idor", "leak", "exposure",
)


def heuristic_cvss(text: str) -> dict[str, str]:
    """Transparent keyword-based CVSS v3.1 metric proposer used by the FakeLLM."""
    t = text.lower()

    av = "N"
    for hints, val in _AV_HINTS:
        if any(h in t for h in hints):
            av = val
            break

    ac = "H" if any(h in t for h in _AC_HIGH) else "L"

    if any(h in t for h in _PR_NONE):
        pr = "N"
    elif any(h in t for h in _PR_HIGH):
        pr = "H"
    elif any(h in t for h in _PR_LOW):
        pr = "L"
    else:
        pr = "N"

    ui = "R" if any(h in t for h in _UI_REQUIRED) else "N"

    is_xss = "xss" in t or "cross-site scripting" in t or "cross site scripting" in t
    is_ssrf = "ssrf" in t or "server-side request forgery" in t
    scope = "C" if (is_xss or is_ssrf or any(h in t for h in _SCOPE_CHANGED)) else "U"

    rce = any(h in t for h in _RCE)
    sqli = "sql injection" in t or "sqli" in t
    dos = any(h in t for h in _DOS)
    disclosure = any(h in t for h in _DISCLOSURE)

    if rce:
        c = i = a = "H"
    elif sqli:
        c, i, a = "H", "H", "N"
    elif is_xss:
        c, i, a = "L", "L", "N"
    elif dos:
        c, i, a = "N", "N", "H"
    elif disclosure:
        c, i, a = "H", "N", "N"
    else:
        c, i, a = "L", "L", "N"

    return {"AV": av, "AC": ac, "PR": pr, "UI": ui, "S": scope, "C": c, "I": i, "A": a}


class FakeLLM(LLM):
    """Deterministic, offline stand-in. Routes on the [[task:...]] tag."""

    def complete(self, system: str, user: str) -> str:
        m = TASK_TAG.search(system)
        task = m.group(1) if m else "unknown"
        payload = _extract_user_payload(user)
        handler = getattr(self, f"_task_{task}", self._task_unknown)
        return json.dumps(handler(payload))

    # -- per-task handlers ------------------------------------------------- #

    def _task_scope(self, p: dict[str, Any]) -> dict[str, Any]:
        text = f"{p.get('title', '')} {p.get('description', '')} {p.get('evidence', '')}"
        product = _first_match(r"\b([A-Z][A-Za-z0-9_.-]{2,})\b", p.get("target") or text)
        version = _first_match(r"\b(\d+\.\d+(?:\.\d+)?)\b", text)
        endpoint = _first_match(r"(/[A-Za-z0-9_./{}-]{2,})", p.get("evidence") or text)
        cwe = _first_match(r"(CWE-\d+)", text)
        vuln_class = _classify(text)
        return {
            "product": product,
            "version": version,
            "endpoint": endpoint,
            "cwe": [cwe] if cwe else [],
            "vuln_class": vuln_class,
        }

    def _task_dedup(self, p: dict[str, Any]) -> dict[str, Any]:
        # The threshold decision is made in code; the model only explains.
        cands = p.get("candidates", [])
        if cands:
            top = cands[0]
            return {
                "assessment": f"Closest prior art is {top.get('doc_id')} "
                f"(similarity {top.get('score'):.2f}); shares the same weakness pattern.",
            }
        return {"assessment": "No close prior art found in the index."}

    def _task_cvss_scoring(self, p: dict[str, Any]) -> dict[str, Any]:
        text = f"{p.get('title', '')}\n{p.get('description', '')}\n{p.get('evidence', '')}"
        metrics = heuristic_cvss(text)
        rationale = {
            "AV": "Reachability inferred from where the issue is triggered.",
            "AC": "Complexity inferred from preconditions described.",
            "PR": "Privilege level inferred from auth described.",
            "UI": "User interaction inferred from whether a victim action is needed.",
            "S": "Scope inferred from whether impact crosses a security authority.",
            "C": "Confidentiality impact inferred from data exposure described.",
            "I": "Integrity impact inferred from modification/execution described.",
            "A": "Availability impact inferred from disruption described.",
        }
        return {"metrics": metrics, "rationale": rationale}

    def _task_drafting(self, p: dict[str, Any]) -> dict[str, Any]:
        return {"report_markdown": _template_report(p)}

    def _task_critic(self, p: dict[str, Any]) -> dict[str, Any]:
        report = p.get("report_markdown", "") or ""
        required = ["## Summary", "## Impact", "## Steps to Reproduce", "## Remediation"]
        missing = [s for s in required if s.lower() not in report.lower()]
        has_repro = len(report) > 200 and "reproduce" in report.lower()
        issues = []
        if missing:
            issues.append("Missing required section(s): " + ", ".join(missing))
        if not has_repro:
            issues.append("Reproduction steps are too thin to verify.")
        score = max(0.0, 1.0 - 0.25 * len(issues))
        return {"passed": not issues, "score": score, "issues": issues}

    def _task_unknown(self, p: dict[str, Any]) -> dict[str, Any]:
        return {"note": "no handler", "echo": p}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _extract_user_payload(user: str) -> dict[str, Any]:
    try:
        return _extract_json(user)
    except (json.JSONDecodeError, ValueError):
        return {"raw": user}


def _first_match(pattern: str, text: str | None) -> str | None:
    if not text:
        return None
    m = re.search(pattern, text)
    return m.group(1) if m else None


def _classify(text: str) -> str:
    t = text.lower()
    table = [
        ("SQL Injection", ("sql injection", "sqli")),
        ("Cross-Site Scripting", ("xss", "cross-site scripting")),
        ("Remote Code Execution", ("remote code execution", "rce", "arbitrary code", "deserialization")),
        ("SSRF", ("ssrf", "server-side request forgery")),
        ("IDOR / Access Control", ("idor", "broken access control", "authorization bypass")),
        ("Information Disclosure", ("information disclosure", "data leak", "exposure")),
        ("Denial of Service", ("denial of service", "dos", "crash")),
    ]
    for label, hints in table:
        if any(h in t for h in hints):
            return label
    return "Other"


def _template_report(p: dict[str, Any]) -> str:
    finding = p.get("finding", {})
    scope = p.get("scope", {})
    cvss = p.get("cvss", {})
    dups = p.get("duplicates", [])
    title = finding.get("title", "Untitled finding")
    target = finding.get("target", "N/A")
    vector = cvss.get("vector", "N/A")
    score = cvss.get("base_score", "N/A")
    severity = cvss.get("severity", "N/A")
    dup_line = (
        "No strong duplicates found in public prior art."
        if not dups
        else "Potential related prior art: "
        + ", ".join(f"{d.get('doc_id')} ({d.get('score'):.2f})" for d in dups[:3])
    )
    endpoint = scope.get("endpoint") or "the affected endpoint"
    return f"""# {title}

**Target:** {target}
**Severity:** {severity} — CVSS {score} `{vector}`
**Weakness class:** {scope.get('vuln_class', 'Other')}

## Summary
{finding.get('description', '').strip()}

## Impact
An attacker exploiting this issue against `{target}` could achieve the impact
reflected in the CVSS vector above ({vector}). {dup_line}

## Steps to Reproduce
1. Navigate to `{endpoint}` on the in-scope target `{target}`.
2. Perform the action described in the finding while observing the response.
3. Confirm the described behavior is reproducible.

_Evidence:_ {finding.get('evidence') or 'see attached request/response capture.'}

## Remediation
Apply input validation / output encoding / authorization checks appropriate to
the {scope.get('vuln_class', 'identified')} weakness, and add a regression test
covering `{endpoint}`.
"""


# --------------------------------------------------------------------------- #
# Provider factory
# --------------------------------------------------------------------------- #

class LangChainLLM(LLM):
    """Adapter over a LangChain chat model (extra: llm)."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def complete(self, system: str, user: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage  # lazy

        resp = self._model.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        return resp.content if hasattr(resp, "content") else str(resp)


def get_llm(settings: Settings | None = None, *, reasoning: bool = False) -> LLM:
    settings = settings or get_settings()
    if settings.llm_provider == "fake":
        return FakeLLM()
    model_name = settings.reasoning_model if reasoning else settings.llm_model
    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI  # lazy

        return LangChainLLM(ChatOpenAI(model=model_name, temperature=settings.llm_temperature))
    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic  # lazy

        return LangChainLLM(ChatAnthropic(model=model_name, temperature=settings.llm_temperature))
    raise ValueError(f"unknown provider {settings.llm_provider!r}")
