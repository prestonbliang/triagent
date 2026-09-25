import json
from pathlib import Path

from triagent.ingest import cve_to_document

FIXTURE = Path(__file__).parent / "fixtures" / "nvd_sample.json"


def _load():
    return json.loads(FIXTURE.read_text())["vulnerabilities"]


def test_normalize_log4shell():
    doc = cve_to_document(_load()[0])
    assert doc.doc_id == "CVE-2021-44228"
    assert doc.source == "nvd"
    assert doc.cvss_vector == "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H"
    assert doc.cvss_score == 10.0
    assert "CWE-502" in doc.cwe and "CWE-917" in doc.cwe
    assert doc.url == "https://nvd.nist.gov/vuln/detail/CVE-2021-44228"
    assert "JNDI" in doc.text
    assert doc.published is not None and doc.published.year == 2021


def test_text_frontloads_id_and_cwe():
    doc = cve_to_document(_load()[0])
    first_line = doc.text.splitlines()[0]
    assert first_line == "CVE-2021-44228"
    assert "CWE-502" in doc.text.splitlines()[1]


def test_all_fixtures_normalize():
    docs = [cve_to_document(v) for v in _load()]
    assert {d.doc_id for d in docs} == {
        "CVE-2021-44228",
        "CVE-2014-0160",
        "CVE-2019-0708",
    }
    # Every fixture carries a valid, scored vector.
    for d in docs:
        assert d.cvss_vector and d.cvss_score is not None
