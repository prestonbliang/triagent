"""Exact CVSS v3.1 Base Score implementation.

Follows the FIRST.org CVSS v3.1 specification (section 7.1) precisely, including
the specified ``Roundup`` function, so scores match NVD's published values to the
0.1 decimal. This is deliberately hand-rolled rather than pulled from a library:
the eval harness measures the *LLM's* metric choices against NVD ground truth, and
for that comparison to be trustworthy the score-from-vector math must be exact and
auditable.

Reference: https://www.first.org/cvss/v3.1/specification-document
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# --- Metric weight tables (CVSS v3.1 spec, section 7.4) ---

_AV = {"N": 0.85, "A": 0.62, "L": 0.55, "P": 0.2}
_AC = {"L": 0.77, "H": 0.44}
# Privileges Required is Scope-dependent.
_PR = {
    "U": {"N": 0.85, "L": 0.62, "H": 0.27},  # Scope Unchanged
    "C": {"N": 0.85, "L": 0.68, "H": 0.5},  # Scope Changed
}
_UI = {"N": 0.85, "R": 0.62}
_CIA = {"H": 0.56, "L": 0.22, "N": 0.0}

_ORDER = ["AV", "AC", "PR", "UI", "S", "C", "I", "A"]
_ALLOWED = {
    "AV": set(_AV),
    "AC": set(_AC),
    "PR": {"N", "L", "H"},
    "UI": set(_UI),
    "S": {"U", "C"},
    "C": set(_CIA),
    "I": set(_CIA),
    "A": set(_CIA),
}


class CVSSError(ValueError):
    """Raised on a malformed or incomplete CVSS v3.x base vector."""


@dataclass(frozen=True)
class CVSS31:
    AV: str
    AC: str
    PR: str
    UI: str
    S: str
    C: str
    I: str  # noqa: E741 — CVSS metric name is literally "I"
    A: str

    def as_dict(self) -> dict[str, str]:
        return {m: getattr(self, m) for m in _ORDER}

    @property
    def vector(self) -> str:
        body = "/".join(f"{m}:{getattr(self, m)}" for m in _ORDER)
        return f"CVSS:3.1/{body}"

    @property
    def base_score(self) -> float:
        return base_score(self)

    @property
    def scope_changed(self) -> bool:
        return self.S == "C"


def parse_vector(vector: str) -> CVSS31:
    """Parse a ``CVSS:3.1/...`` (or 3.0) base vector string.

    Temporal/environmental metrics are ignored; only the eight base metrics are
    required. Raises :class:`CVSSError` on anything malformed or incomplete.
    """
    if not vector or "/" not in vector:
        raise CVSSError(f"not a CVSS vector: {vector!r}")

    parts = vector.strip().split("/")
    head = parts[0].upper()
    if head.startswith("CVSS:"):
        version = head.split(":", 1)[1]
        if not version.startswith(("3.0", "3.1")):
            raise CVSSError(f"unsupported CVSS version: {version!r}")
        metric_parts = parts[1:]
    else:
        # Allow bare "AV:N/AC:L/..." without the CVSS:3.1 prefix.
        metric_parts = parts

    found: dict[str, str] = {}
    for token in metric_parts:
        if ":" not in token:
            raise CVSSError(f"malformed metric token: {token!r}")
        key, _, val = token.partition(":")
        key, val = key.upper(), val.upper()
        if key in _ORDER:
            if val not in _ALLOWED[key]:
                raise CVSSError(f"invalid value {val!r} for metric {key}")
            found[key] = val
        # Non-base metrics (E, RL, RC, CR, MAV, ...) are silently ignored.

    missing = [m for m in _ORDER if m not in found]
    if missing:
        raise CVSSError(f"missing base metric(s): {', '.join(missing)}")

    return CVSS31(**found)


def _roundup(x: float) -> float:
    """The exact Roundup from the CVSS v3.1 spec (Appendix A).

    Rounds up to one decimal place while avoiding binary-float artifacts, e.g.
    Roundup(4.02) == 4.1 and Roundup(4.00) == 4.0.
    """
    int_input = round(x * 100_000)
    if int_input % 10_000 == 0:
        return int_input / 100_000.0
    return (math.floor(int_input / 10_000) + 1) / 10.0


def base_score(m: CVSS31) -> float:
    """Compute the CVSS v3.1 Base Score for a parsed vector."""
    iss = 1.0 - ((1 - _CIA[m.C]) * (1 - _CIA[m.I]) * (1 - _CIA[m.A]))

    if m.scope_changed:
        impact = 7.52 * (iss - 0.029) - 3.25 * (iss - 0.02) ** 15
    else:
        impact = 6.42 * iss

    exploitability = (
        8.22 * _AV[m.AV] * _AC[m.AC] * _PR[m.S][m.PR] * _UI[m.UI]
    )

    if impact <= 0:
        return 0.0

    raw = 1.08 * (impact + exploitability) if m.scope_changed else impact + exploitability
    return _roundup(min(raw, 10.0))
