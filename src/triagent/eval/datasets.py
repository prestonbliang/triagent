"""Load labeled eval datasets. Datasets ship as JSONL fixtures; swap the paths for
your full NVD-derived sets (thousands of CVEs) without touching the harness."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from triagent.schemas import Finding

_DATA = Path(__file__).resolve().parents[3] / "data" / "fixtures"


@dataclass
class ScoringExample:
    id: str
    finding: Finding
    ground_truth_vector: str


@dataclass
class RetrievalExample:
    query: str
    relevant: list[str]


def load_scoring_eval(path: Path | None = None) -> list[ScoringExample]:
    path = path or _DATA / "scoring_eval.jsonl"
    out: list[ScoringExample] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out.append(
            ScoringExample(
                id=row["id"],
                finding=Finding(**row["finding"]),
                ground_truth_vector=row["ground_truth_vector"],
            )
        )
    return out


def load_retrieval_eval(path: Path | None = None) -> list[RetrievalExample]:
    path = path or _DATA / "retrieval_eval.jsonl"
    out: list[RetrievalExample] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out.append(RetrievalExample(query=row["query"], relevant=list(row["relevant"])))
    return out
