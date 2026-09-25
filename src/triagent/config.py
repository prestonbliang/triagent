"""Central configuration. Everything defaults to an offline, deterministic mode
so the graph and eval harness run with no secrets and no network."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TRIAGENT_", env_file=".env", extra="ignore"
    )

    # --- LLM ---
    llm_provider: Literal["fake", "openai", "anthropic"] = "fake"
    llm_model: str = "gpt-4o-mini"
    reasoning_model: str = "gpt-4o"  # scoring + critic use the stronger model
    llm_temperature: float = 0.0

    # --- Embeddings ---
    embedder: Literal["fake", "sentence-transformers"] = "fake"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = 384

    # --- Reranker ---
    reranker: Literal["off", "identity", "cross-encoder"] = "identity"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Vector store ---
    store: Literal["memory", "pgvector"] = "memory"
    database_url: str = "postgresql+psycopg://triagent:triagent@localhost:5432/triagent"

    # --- Retrieval / triage tuning ---
    retrieval_k: int = 20
    rerank_k: int = 6
    dup_threshold: float = Field(0.83, ge=0.0, le=1.0)
    max_revisions: int = 2

    # --- NVD ingestion ---
    nvd_api_key: str | None = Field(default=None, alias="NVD_API_KEY")


@lru_cache
def get_settings() -> Settings:
    return Settings()
