"""Lock the eval harness: shapes, gate logic, and the kappa helper."""

from triagent.eval.report_eval import cohens_kappa
from triagent.eval.run import check_gates, run_all


def test_scoring_eval_shape_and_quality():
    results = run_all(["scoring"])
    s = results["scoring"]
    assert s["n"] == 10
    # The deterministic baseline must stay usable; large drift is a real regression.
    assert s["base_score_mae"] < 1.0
    assert 0.0 <= s["severity_accuracy"] <= 1.0
    # Per-metric accuracy is reported for all eight base metrics.
    assert set(s["per_metric_accuracy"]) == {"AV", "AC", "PR", "UI", "S", "C", "I", "A"}
    # Confusion is populated (the artifact that localizes weakness).
    assert s["confusion"]["S"]


def test_retrieval_and_report_pass_gate():
    results = run_all(["scoring", "retrieval", "report"])
    assert results["retrieval"]["recall_at_k"]["recall@3"] >= 0.8
    assert results["report"]["pass_rate"] >= 0.8
    # The full run passes every configured gate.
    assert check_gates(results) == []


def test_gate_catches_regression():
    bad = {"scoring": {"base_score_mae": 9.9, "severity_accuracy": 0.1}}
    failures = check_gates(bad)
    assert any("base_score_mae" in f for f in failures)
    assert any("severity_accuracy" in f for f in failures)


def test_cohens_kappa():
    # Perfect agreement -> 1.0; independent-ish -> near 0.
    assert cohens_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert abs(cohens_kappa([1, 1, 0, 0], [1, 0, 1, 0])) < 0.6
