"""Shared LangGraph state. Nodes return partial dicts that LangGraph merges in."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from triagent.schemas import (
    CVSSAssessment,
    DuplicateMatch,
    Finding,
    QualityReview,
)


def _append(left: list[Any], right: list[Any]) -> list[Any]:
    return (left or []) + (right or [])


class TriageState(TypedDict, total=False):
    finding: Finding
    # scope extraction
    scope: dict[str, Any]
    # retrieval / dedup
    duplicates: list[DuplicateMatch]
    likely_duplicate: bool
    # scoring
    cvss: CVSSAssessment
    # drafting + critique loop
    report_markdown: str
    quality: QualityReview
    revisions: int
    # observability: append-only trace of nodes that ran
    trace: Annotated[list[str], _append]
