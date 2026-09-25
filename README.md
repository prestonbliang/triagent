# Triagent

**An AI vulnerability-triage analyst.** A finding — analyst notes, or an export from
recon you ran against an authorized target — flows through a LangGraph supervisor
over five agents (scope extraction → retrieval/dedup → CVSS scoring → report
drafting → quality-gate critic) and comes out as a triage decision plus a
submission-ready disclosure, grounded in retrieved prior art and scored with a
defensible CVSS vector.

Built to demonstrate **production AI-engineering patterns end to end** — RAG with
hybrid retrieval and reranking, multi-agent orchestration, a measurable LLM eval
harness wired into a CI merge gate, and LLMOps — on a problem with **objective
ground truth**, so the quality claims are numbers, not vibes.

```
                 ┌─────────┐     ┌─────────┐
  finding  ──►   │  scope  │ ──► │  dedup  │ ──(likely duplicate)──────────────►  END
                 └─────────┘     └────┬────┘                                       ▲
                  extract product,    │ retrieve + rerank prior art               │
                  version, endpoint,  │ (pgvector hybrid search)                  │
                  CWE, weakness class │                                           │
                                      │ (novel)                                   │
                                 ┌────▼─────┐   ┌──────────┐   ┌────────┐         │
                                 │ scoring  │──►│ drafting │──►│ critic │──(pass)─┘
                                 └──────────┘   └────▲─────┘   └───┬────┘
                                 LLM proposes CVSS   │             │ fail + budget left
                                 metrics; the exact  └──(revise)───┘
                                 engine computes the score
```

Nothing is ever auto-submitted: every run ends `awaiting_human_review=True`.

---

## Why this is not another RAG chatbot

Most portfolio LLM projects can't prove they work. Triagent is scored against **real
ground truth**:

- **CVSS scoring** is measured against NVD's official published vectors — mean
  absolute error, exact-vector rate, and a **per-metric confusion matrix** that
  tells you exactly which metric the model is weak on (usually Scope).
- **Retrieval/dedup** is measured with recall@k and MRR against CVE-linked queries.
- **Report quality** uses an LLM-as-judge rubric with a **Cohen's-kappa calibration**
  hook against human labels, so the judge is trusted only as far as it agrees with a
  human.

All three run in CI as a **merge gate**: a PR that regresses any headline metric
fails the build. The whole harness runs offline with a deterministic fake LLM and a
deterministic embedder, so CI needs no API keys.

### Current baseline numbers (deterministic fake model, `triagent-eval`)

| Suite | Metric | Value |
|---|---|---|
| Scoring | base-score MAE vs NVD | **0.48** |
| Scoring | within 1.0 CVSS point | **80%** |
| Scoring | severity-band accuracy | **70%** |
| Scoring | per-metric accuracy (AC/UI/C/I) | **100%** |
| Retrieval | recall@3 / MRR | **100% / 0.83** |
| Report | rubric pass rate | **100%** |

These are the *floor*: the fake model is a transparent keyword heuristic. Swap in a
frontier model (one env var) and the same harness measures the improvement — that
delta is the story.

---

## Quickstart (offline, no keys)

```bash
pip install -e ".[dev,api]"
make test        # 35 tests: exact CVSS math, ingest, RAG, the full agent graph, API
make eval        # runs the eval harness + CI gate against fixtures
make api         # FastAPI on :8000 — POST /triage
```

Try a triage:

```bash
curl -s localhost:8000/triage -H 'content-type: application/json' -d '{
  "finding_id":"F-1",
  "title":"Unauthenticated RCE via deserialization",
  "description":"A remote unauthenticated attacker sends a crafted payload to the public API which is deserialized unsafely, yielding remote code execution.",
  "target":"api.example.com",
  "evidence":"POST /api/import with a serialized object executes commands."
}' | jq '{severity: .cvss.severity, vector: .cvss.vector, dupe: .likely_duplicate}'
```

## Production setup

```bash
pip install -e ".[dev,api,models,postgres,llm,obs]"
docker compose up -d                 # Postgres + pgvector
cp .env.example .env                 # then set the vars below
make ingest                          # pull recent NVD CVEs into the store
```

Flip these in `.env` to go from offline to real inference:

| Var | Offline default | Production |
|---|---|---|
| `TRIAGENT_LLM_PROVIDER` | `fake` | `openai` / `anthropic` |
| `TRIAGENT_EMBEDDER` | `fake` | `sentence-transformers` |
| `TRIAGENT_RERANKER` | `identity` | `cross-encoder` |
| `TRIAGENT_STORE` | `memory` | `pgvector` |

Every one is behind an interface, so swapping backends touches no agent code.

---

## HuggingFace tasks used

- **Sentence Similarity / Feature Extraction** — dense retrieval + dedup embeddings.
- **Text Ranking** — cross-encoder reranking (precision stage after recall).
- **Token Classification (NER)** — product/version/endpoint/CWE extraction (scope agent).
- **Text Classification / Zero-Shot** — weakness-class (CWE) routing.
- **Summarization** — condensing long advisories into context.
- **Question Answering** — extractive fact-pull from advisories.
- **Text Generation** — report drafting and remediation.

## Real-world data sources

NVD CVE API 2.0 (scoring ground truth), plus drop-in ingesters for EPSS, CISA KEV,
OSV.dev, GitHub Security Advisories, and HackerOne public disclosures.

---

## Scope & ethics

Triagent operates on the **analysis, triage, scoring, and reporting** side of
vulnerability work. It reads and reasons over vulnerability *data* — advisories,
CVEs, disclosed reports, and findings/scan output you are authorized to handle. It
**does not** generate exploits, weaponize vulnerabilities, or scan or interact with
targets. The human-in-the-loop gate is mandatory by design: the pipeline produces a
draft for a person to review and submit, never an automated action.

## Repo layout

```
src/triagent/
  cvss/        exact CVSS v3.1 base-score engine (spec-accurate, unit-tested vs NVD)
  ingest/      NVD API 2.0 client + normalizer
  rag/         pluggable embeddings, pgvector/in-memory store (hybrid), reranker
  agents/      scope · dedup · scoring · drafting · critic + LangGraph supervisor + LLM factory
  eval/        scoring / retrieval / report suites + CLI + CI gate
  api/         FastAPI service with HITL review flow
tests/         35 tests, all offline
data/fixtures/ labeled eval datasets (scoring + retrieval)
```

## How it maps to AI-engineering JDs

| JD requirement | Where it lives |
|---|---|
| Build RAG / retrieval in production | `rag/` — hybrid pgvector search + cross-encoder rerank |
| Multi-agent / tool-using orchestration | `agents/graph.py` — LangGraph supervisor, conditional routing, bounded revise loop |
| LLM evaluation frameworks | `eval/` — MAE, recall@k/MRR, LLM-as-judge + kappa calibration |
| LLMOps: CI evals, tracing, prompt mgmt | `.github/workflows/ci.yml` eval gate, Langfuse hook, versioned prompts |
| Vector databases | pgvector store with HNSW + hybrid lexical/dense blend |
| Model cost/quality tiering | cheap model for extraction, reasoning model for scoring/critic |
| Guardrails + human-in-the-loop | mandatory review gate; no automated submission |

## Roadmap

- Fine-tune a small CWE classifier and benchmark it against the zero-shot LLM baseline (cost/latency/accuracy table).
- EPSS + CISA KEV overlays feeding an exploitability-adjusted priority score.
- CVSS 4.0 scoring engine alongside 3.1.
- Langfuse traces + a Streamlit review console over the API.

## License

MIT
