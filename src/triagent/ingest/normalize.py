"""Normalize a raw NVD CVE API 2.0 record into a retrieval :class:`Document`.

Kept pure and side-effect-free so it can be unit-tested against a saved fixture
with no network. The NVD schema is nested and inconsistent across CVEs (English
descriptions, multiple CVSS versions, optional CWE), so this is where that mess
gets tamed once.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from triagent.schemas import Document


def _english_description(cve: dict[str, Any]) -> str:
    for d in cve.get("descriptions", []):
        if d.get("lang") == "en":
            return d.get("value", "").strip()
    descs = cve.get("descriptions", [])
    return descs[0].get("value", "").strip() if descs else ""


def _best_cvss(cve: dict[str, Any]) -> tuple[str | None, float | None]:
    """Prefer CVSS v3.1, then v3.0, then v4.0. Return (vector, base_score)."""
    metrics = cve.get("metrics", {})
    for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV40"):
        entries = metrics.get(key)
        if not entries:
            continue
        data = entries[0].get("cvssData", {})
        vector = data.get("vectorString")
        score = data.get("baseScore")
        if vector is not None:
            return vector, float(score) if score is not None else None
    return None, None


def _cwes(cve: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for w in cve.get("weaknesses", []):
        for d in w.get("description", []):
            val = d.get("value", "")
            if val.startswith("CWE-") and val not in out:
                out.append(val)
    return out


def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def cve_to_document(item: dict[str, Any]) -> Document:
    """Convert one element of the NVD ``vulnerabilities`` array to a Document."""
    cve = item.get("cve", item)  # accept either the wrapped or bare CVE object
    cve_id = cve["id"]
    description = _english_description(cve)
    vector, score = _best_cvss(cve)
    cwes = _cwes(cve)

    # The embedded text is what retrieval matches on — front-load the identifier,
    # weakness class, and description so semantically-similar findings surface.
    text_parts = [cve_id]
    if cwes:
        text_parts.append("Weakness: " + ", ".join(cwes))
    if description:
        text_parts.append(description)
    text = "\n".join(text_parts)

    return Document(
        doc_id=cve_id,
        source="nvd",
        title=cve_id,
        text=text,
        cwe=cwes,
        cvss_vector=vector,
        cvss_score=score,
        published=_parse_ts(cve.get("published")),
        url=f"https://nvd.nist.gov/vuln/detail/{cve_id}",
        metadata={
            "status": cve.get("vulnStatus"),
            "references": [r.get("url") for r in cve.get("references", [])][:10],
        },
    )
