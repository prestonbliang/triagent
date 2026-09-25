.PHONY: install install-prod test lint typecheck eval ingest api db-up db-down fmt ci

install:  ## install core + dev (offline-runnable)
	pip install -e ".[dev,api]"

install-prod:  ## install everything for real inference + pgvector
	pip install -e ".[dev,api,models,postgres,llm,obs]"

test:
	pytest

lint:
	ruff check src tests

fmt:
	ruff check --fix src tests
	ruff format src tests

typecheck:
	mypy src

eval:  ## run the full eval harness against fixtures, gate on thresholds
	triagent-eval --suite all --gate

ingest:  ## pull recent NVD CVEs into the vector store
	triagent-ingest --days 30

api:
	uvicorn triagent.api.main:app --reload --port 8000

db-up:
	docker compose up -d

db-down:
	docker compose down

# What CI runs: lint, types, tests, and the eval gate.
ci: lint typecheck test eval
