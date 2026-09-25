from fastapi.testclient import TestClient

from triagent.api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["llm_provider"] == "fake"  # offline default


def test_triage_endpoint_returns_draft_for_review():
    finding = {
        "finding_id": "API-1",
        "title": "Unauthenticated RCE via deserialization",
        "description": "Remote unauthenticated attacker achieves arbitrary code execution "
        "via unsafe deserialization on the public API.",
        "target": "api.example.com",
        "evidence": "POST /api/import with a serialized payload runs commands.",
    }
    r = client.post("/triage", json=finding)
    assert r.status_code == 200
    body = r.json()
    assert body["finding_id"] == "API-1"
    assert body["cvss"]["severity"] == "Critical"
    assert body["awaiting_human_review"] is True
    assert "## Steps to Reproduce" in body["report_markdown"]


def test_decision_is_recorded():
    r = client.post("/triage/API-1/decision", json={"action": "approve", "note": "verified"})
    assert r.status_code == 200
    assert r.json()["recorded"]["action"] == "approve"
