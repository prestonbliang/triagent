"""CVSS v3.1 base-score correctness against published, hand-verified ground truth.

Each case is a real-world vector whose NVD base score is well known. If any of
these drift, the score math is wrong and every downstream eval number is suspect —
so these run first in CI.
"""

import math

import pytest

from triagent.cvss import CVSSError, base_score, parse_vector
from triagent.schemas import Severity

# (vector, expected NVD base score, expected qualitative severity)
CASES = [
    ("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H", 9.8, Severity.CRITICAL),
    ("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H", 10.0, Severity.CRITICAL),
    ("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:N/A:N", 7.5, Severity.HIGH),
    ("CVSS:3.1/AV:L/AC:L/PR:L/UI:N/S:U/C:H/I:H/A:H", 7.8, Severity.HIGH),
    ("CVSS:3.1/AV:N/AC:L/PR:L/UI:N/S:U/C:L/I:L/A:N", 5.4, Severity.MEDIUM),
    ("CVSS:3.1/AV:N/AC:H/PR:N/UI:R/S:U/C:L/I:N/A:N", 3.1, Severity.LOW),
    ("CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:C/C:L/I:L/A:N", 6.1, Severity.MEDIUM),  # stored XSS
    ("CVSS:3.1/AV:P/AC:H/PR:H/UI:R/S:U/C:N/I:N/A:N", 0.0, Severity.NONE),
]


@pytest.mark.parametrize("vector,expected,severity", CASES)
def test_base_score_matches_nvd(vector, expected, severity):
    parsed = parse_vector(vector)
    score = base_score(parsed)
    assert score == pytest.approx(expected, abs=0.001), f"{vector} -> {score}, want {expected}"
    assert Severity.from_score(score) is severity


def test_roundtrip_vector_string():
    v = "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H"
    assert parse_vector(v).vector == v


def test_prefixless_vector_ok():
    assert base_score(parse_vector("AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H")) == 9.8


def test_temporal_metrics_ignored():
    # Trailing temporal/environmental metrics must not break base parsing.
    v = "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H/E:P/RL:O/RC:C"
    assert base_score(parse_vector(v)) == 9.8


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "not-a-vector",
        "CVSS:2.0/AV:N/AC:L/Au:N/C:P/I:P/A:P",  # v2 unsupported
        "CVSS:3.1/AV:X/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",  # invalid AV value
        "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H",  # missing A
    ],
)
def test_invalid_vectors_raise(bad):
    with pytest.raises(CVSSError):
        parse_vector(bad)


def test_roundup_edge():
    # Roundup must round *up* to one decimal, not to-nearest.
    from triagent.cvss.vector import _roundup

    assert _roundup(4.02) == 4.1
    assert _roundup(4.00) == 4.0
    assert _roundup(0.0) == 0.0
    assert math.isclose(_roundup(9.76), 9.8)
