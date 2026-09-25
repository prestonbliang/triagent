-- Enable pgvector on first boot of the Postgres container.
CREATE EXTENSION IF NOT EXISTS vector;
