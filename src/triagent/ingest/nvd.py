"""NVD CVE API 2.0 client.

Handles the two things that actually bite you against NVD: pagination (2000
results/page via startIndex/totalResults) and rate limiting (5 req/30s without a
key, 50 with one — NVD asks for a sleep between requests either way). Retries
transient 5xx/timeout with backoff.

CLI:
    triagent-ingest --days 30            # index CVEs modified in the last 30 days
    triagent-ingest --cve CVE-2021-44228 # index a single CVE
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from triagent.config import get_settings
from triagent.ingest.normalize import cve_to_document
from triagent.schemas import Document

log = structlog.get_logger(__name__)

_BASE_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
_PAGE_SIZE = 2000  # NVD max resultsPerPage


class NVDClient:
    def __init__(self, api_key: str | None = None, timeout: float = 60.0) -> None:
        self.api_key = api_key
        # NVD's own guidance: without a key, sleep ~6s between requests; with a
        # key you may go faster. We stay conservative to avoid 403s.
        self._sleep = 0.6 if api_key else 6.0
        self._client = httpx.Client(
            timeout=timeout,
            headers={"apiKey": api_key} if api_key else {},
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> NVDClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        reraise=True,
    )
    def _get(self, params: dict[str, Any]) -> dict[str, Any]:
        resp = self._client.get(_BASE_URL, params=params)
        resp.raise_for_status()
        return resp.json()

    def iter_cves(self, params: dict[str, Any]) -> Iterator[Document]:
        """Yield normalized Documents across all pages for a query."""
        start = 0
        total = None
        while total is None or start < total:
            page_params = {**params, "startIndex": start, "resultsPerPage": _PAGE_SIZE}
            payload = self._get(page_params)
            total = payload.get("totalResults", 0)
            vulns = payload.get("vulnerabilities", [])
            log.info("nvd.page", start=start, total=total, got=len(vulns))
            for item in vulns:
                yield cve_to_document(item)
            if not vulns:
                break
            start += len(vulns)
            if start < total:
                time.sleep(self._sleep)

    def recent(self, days: int) -> Iterator[Document]:
        """CVEs modified within the last ``days`` (NVD caps the window at 120)."""
        if days > 120:
            raise ValueError("NVD lastModified window is capped at 120 days")
        now = datetime.now(UTC)
        start = now - timedelta(days=days)
        return self.iter_cves(
            {
                "lastModStartDate": start.strftime("%Y-%m-%dT%H:%M:%S.000"),
                "lastModEndDate": now.strftime("%Y-%m-%dT%H:%M:%S.000"),
            }
        )

    def by_id(self, cve_id: str) -> Document | None:
        payload = self._get({"cveId": cve_id})
        vulns = payload.get("vulnerabilities", [])
        return cve_to_document(vulns[0]) if vulns else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest NVD CVEs into the vector store.")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--cve", type=str, help="ingest a single CVE id")
    args = parser.parse_args()

    settings = get_settings()
    # Import here so ingestion works even if the RAG extra isn't installed at import time.
    from triagent.rag.index import index_documents

    with NVDClient(api_key=settings.nvd_api_key) as client:
        if args.cve:
            doc = client.by_id(args.cve)
            docs = [doc] if doc else []
        else:
            docs = list(client.recent(args.days))

    n = index_documents(docs)
    log.info("ingest.done", indexed=n)
    print(f"Indexed {n} documents.")


if __name__ == "__main__":
    main()
