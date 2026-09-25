"""Eval CLI + CI gate.

    triagent-eval --suite all --gate --out eval_runs/ci.json

Runs the requested suites, prints a markdown summary, writes a JSON artifact, and
(with --gate) exits non-zero if any headline metric breaches its threshold — this
is what turns the eval harness into a merge gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from triagent.agents.llm import get_llm
from triagent.config import get_settings
from triagent.eval.report_eval import run_report_eval
from triagent.eval.retrieval_eval import run_retrieval_eval
from triagent.eval.scoring_eval import run_scoring_eval
from triagent.rag.rerank import get_reranker

# Regression thresholds. Calibrated to the deterministic fake baseline so CI is
# green today and a real regression (or a model swap that underperforms) fails.
# Tighten these as you swap in stronger models.
GATES: dict[str, dict[str, float]] = {
    "scoring": {"base_score_mae_max": 2.5, "severity_accuracy_min": 0.55},
    "retrieval": {"recall@3_min": 0.80, "mrr_min": 0.70},
    "report": {"pass_rate_min": 0.80},
}


def _corpus_store():
    """Fresh in-memory store loaded with the NVD fixtures for retrieval/report evals."""
    import json as _json

    from triagent.ingest import cve_to_document
    from triagent.rag.embeddings import get_embedder
    from triagent.rag.store import MemoryStore

    settings = get_settings()
    fixture = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "nvd_sample.json"
    store = MemoryStore(get_embedder(settings))
    docs = [cve_to_document(v) for v in _json.loads(fixture.read_text())["vulnerabilities"]]
    store.add(docs)
    return store


def run_all(suites: list[str]) -> dict[str, Any]:
    settings = get_settings()
    results: dict[str, Any] = {}
    store = _corpus_store()
    reranker = get_reranker(settings)

    if "scoring" in suites:
        results["scoring"] = run_scoring_eval(settings, get_llm(settings, reasoning=True))
    if "retrieval" in suites:
        results["retrieval"] = run_retrieval_eval(store, settings, reranker)
    if "report" in suites:
        results["report"] = run_report_eval(store, settings, get_llm(settings), reranker)
    return results


def check_gates(results: dict[str, Any]) -> list[str]:
    """Return a list of gate violations (empty == pass)."""
    failures: list[str] = []
    if "scoring" in results:
        s = results["scoring"]
        g = GATES["scoring"]
        if s["base_score_mae"] > g["base_score_mae_max"]:
            failures.append(f"scoring.base_score_mae {s['base_score_mae']} > {g['base_score_mae_max']}")
        if s["severity_accuracy"] < g["severity_accuracy_min"]:
            failures.append(
                f"scoring.severity_accuracy {s['severity_accuracy']} < {g['severity_accuracy_min']}"
            )
    if "retrieval" in results:
        r = results["retrieval"]
        g = GATES["retrieval"]
        if r["recall_at_k"]["recall@3"] < g["recall@3_min"]:
            failures.append(f"retrieval.recall@3 {r['recall_at_k']['recall@3']} < {g['recall@3_min']}")
        if r["mrr"] < g["mrr_min"]:
            failures.append(f"retrieval.mrr {r['mrr']} < {g['mrr_min']}")
    if "report" in results:
        rp = results["report"]
        g = GATES["report"]
        if rp["pass_rate"] < g["pass_rate_min"]:
            failures.append(f"report.pass_rate {rp['pass_rate']} < {g['pass_rate_min']}")
    return failures


def to_markdown(results: dict[str, Any]) -> str:
    lines = ["# Triagent eval report", ""]
    if "scoring" in results:
        s = results["scoring"]
        lines += [
            "## Scoring (CVSS vs NVD ground truth)",
            f"- examples: **{s['n']}**",
            f"- base-score MAE: **{s['base_score_mae']}**  |  RMSE: {s['base_score_rmse']}",
            f"- within 1.0 point: **{s['within_1_rate']:.0%}**  |  exact vector: {s['exact_vector_rate']:.0%}",
            f"- severity-band accuracy: **{s['severity_accuracy']:.0%}**",
            "- per-metric accuracy: "
            + ", ".join(f"{m} {v:.0%}" for m, v in s["per_metric_accuracy"].items()),
            "",
        ]
    if "retrieval" in results:
        r = results["retrieval"]
        rk = r["recall_at_k"]
        lines += [
            "## Retrieval / dedup",
            f"- queries: **{r['n']}**",
            "- " + "  |  ".join(f"{k}: **{v:.0%}**" for k, v in rk.items()),
            f"- MRR: **{r['mrr']}**",
            "",
        ]
    if "report" in results:
        rp = results["report"]
        lines += [
            "## Report quality (LLM-as-judge rubric)",
            f"- reports: **{rp['n']}**",
            f"- pass rate: **{rp['pass_rate']:.0%}**  |  mean rubric score: {rp['mean_rubric_score']}",
            "",
        ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Triagent eval harness.")
    parser.add_argument(
        "--suite",
        choices=["all", "scoring", "retrieval", "report"],
        default="all",
    )
    parser.add_argument("--gate", action="store_true", help="fail (exit 1) on threshold breach")
    parser.add_argument("--out", type=str, help="write JSON results to this path")
    args = parser.parse_args()

    suites = ["scoring", "retrieval", "report"] if args.suite == "all" else [args.suite]
    results = run_all(suites)

    print(to_markdown(results))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2, default=str))
        print(f"\nWrote {out}")

    if args.gate:
        failures = check_gates(results)
        if failures:
            print("\nEVAL GATE FAILED:")
            for f in failures:
                print(f"  - {f}")
            sys.exit(1)
        print("\nEVAL GATE PASSED")


if __name__ == "__main__":
    main()
