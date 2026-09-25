"""Triagent — an AI vulnerability-triage analyst.

Pipeline: a finding (analyst notes or an authorized-scan export) flows through a
LangGraph supervisor over five agents — scope extraction, retrieval/dedup,
CVSS scoring, report drafting, and a quality-gate critic — producing a triage
decision and a submission-ready disclosure, grounded in retrieved prior art.

Design boundary: this system analyses, deduplicates, scores, and reports on
vulnerability data you are authorized to handle. It does not generate exploits
or scan targets. See README.md#scope-and-ethics.
"""

__version__ = "0.1.0"
