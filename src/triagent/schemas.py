"""Typed contracts used across ingestion, retrieval, the agent graph, and the API."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Severity(str, Enum):
    NONE = "None"
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"

    @classmethod
    def from_score(cls, score: float) -> Severity:
        """CVSS v3.1 qualitative rating scale."""
        if score <= 0.0:
            return cls.NONE
        if score < 4.0:
            return cls.LOW
        if score < 7.0:
            return cls.MEDIUM
        if score < 9.0:
            return cls.HIGH
        return cls.CRITICAL


class Document(BaseModel):
    """A unit of prior art in the vector store: a CVE, advisory, or disclosed report."""

    doc_id: str  # e.g. "CVE-2024-3094", "GHSA-xxxx", "H1-1234567"
    source: str  # "nvd" | "ghsa" | "osv" | "hackerone" | ...
    title: str
    text: str  # the body used for embedding/retrieval
    cwe: list[str] = Field(default_factory=list)
    cvss_vector: str | None = None
    cvss_score: float | None = None
    published: datetime | None = None
    url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Finding(BaseModel):
    """The raw input to triage: an analyst's writeup of a potential vulnerability."""

    finding_id: str
    title: str
    description: str  # free text: what was observed, where, impact
    target: str | None = None  # affected asset (in-scope), e.g. "api.example.com"
    program: str | None = None  # bounty program / scope name
    evidence: str | None = None  # repro notes, request/response excerpts, scanner output
    reporter_cvss_vector: str | None = None  # the hunter's own guess, if any


class ScoredMetric(BaseModel):
    """One CVSS metric with the model's chosen value and its justification."""

    metric: str  # "AV", "AC", "PR", "UI", "S", "C", "I", "A"
    value: str  # "N", "L", "H", ...
    rationale: str


class CVSSAssessment(BaseModel):
    vector: str
    base_score: float
    severity: Severity
    metrics: list[ScoredMetric] = Field(default_factory=list)
    # Real-world exploitability overlays (populated from EPSS / CISA KEV when available).
    epss: float | None = None
    in_kev: bool | None = None


class DuplicateMatch(BaseModel):
    doc_id: str
    title: str
    score: float  # similarity in [0, 1]
    url: str | None = None
    why: str | None = None


class QualityReview(BaseModel):
    passed: bool
    score: float  # 0..1 rubric score
    issues: list[str] = Field(default_factory=list)


class TriageResult(BaseModel):
    """The full output of a triage run."""

    finding_id: str
    scope: dict[str, Any] = Field(default_factory=dict)  # extracted product/version/endpoint/cwe
    likely_duplicate: bool = False
    duplicates: list[DuplicateMatch] = Field(default_factory=list)
    cvss: CVSSAssessment | None = None
    report_markdown: str | None = None
    quality: QualityReview | None = None
    revisions: int = 0
    # Trace of which agents ran, for observability / debugging.
    trace: list[str] = Field(default_factory=list)
    awaiting_human_review: bool = True
