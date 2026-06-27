-- recall/memory/schema.sql
-- PostgreSQL schema for semantic memory (pgvector)
-- Run once via scripts/init_db.py - safe to re-run (idempotent)

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Semantic memory: embedded facts and important notes
CREATE TABLE IF NOT EXISTS memories (
    id          BIGSERIAL PRIMARY KEY,
    content     TEXT NOT NULL,
    source      TEXT NOT NULL,
    session_id  TEXT,
    embedding   VECTOR(512),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata    JSONB DEFAULT '{}'
);

-- HNSW index for fast cosine similarity search
-- No training step needed, builds incrementally as rows are inserted
CREATE INDEX IF NOT EXISTS memory_embedding_hnsw
    ON memories
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Supporting indexes for filtering
CREATE INDEX IF NOT EXISTS memories_source_idx ON memories (source);
CREATE INDEX IF NOT EXISTS memories_created_at_idx ON memories (created_at DESC);