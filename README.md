# Recall

A CLI personal memory agent powered by Claude, built as a hands-on learning project for dual-memory AI architecture.

Unlike a typical chatbot that forgets everything between sessions, Recall maintains a persistent memory layer that grows over time — combining keyword search and semantic search so it can recall both *what was said* and *what it means*.

## Why Two Memory Systems?

Humans search their own memory in two different ways:

- **By words** — "find that conversation about Terraform drift"
- **By meaning** — "what do I know about cloud infrastructure?"

Recall mirrors both:

| System | Technology | Finds things by | Use case |
|---|---|---|---|
| **Episodic memory** | SQLite + FTS5 | Keywords (BM25 ranking) | "What did I say about X?" |
| **Semantic memory** | PostgreSQL + pgvector | Meaning (cosine similarity) | "What do I know about Y?" |

Every conversation is stored in SQLite and indexed for full-text search. Important facts extracted from conversations are embedded via Voyage AI and stored in PostgreSQL for semantic recall. When you send a new message, both systems are queried in parallel, the most relevant memories are merged, and Claude receives that context automatically — before it ever responds.

## Stack

- **Python** — core language
- **Claude (Anthropic SDK)** — conversational backbone
- **PostgreSQL + pgvector** — semantic/associative memory
- **SQLite + FTS5** — episodic/keyword memory (built into Python, no extra install)
- **Voyage AI** — text embeddings
- **Docker** — isolated, reproducible Postgres environment

## What This Project Teaches

- How embeddings represent meaning as vectors
- Vector similarity search and cosine distance
- RAG (Retrieval Augmented Generation) — what it actually is under the hood
- FTS5 full-text search and when keyword search beats vector search
- Agent memory architecture — episodic vs. semantic memory
- Production PostgreSQL patterns (indexing, connection handling, schema design)

## Project Structure

recall/

├── docker/                 # PostgreSQL + pgvector via Docker Compose

├── recall/

│   ├── cli.py              # CLI entry point

│   ├── config.py           # Typed settings (Pydantic)

│   ├── agent.py             # Orchestrates retrieval → Claude → storage

│   ├── context.py          # Builds the memory-enriched system prompt

│   ├── memory/

│   │   ├── episodic.py     # SQLite FTS5 — keyword search

│   │   ├── semantic.py     # pgvector — cosine similarity search

│   │   └── schema.sql      # PostgreSQL schema (DDL)

│   └── embeddings/

│       └── voyage.py       # Voyage AI embedding wrapper

├── scripts/

│   └── init_db.py          # One-time DB schema initialization

└── tests/                  # Unit tests for both memory stores

## Setup

**Prerequisites:** Python 3.11+, Docker, a PostgreSQL client (`psql`), an Anthropic API key, and a Voyage AI API key.

1. Clone the repo and create a virtual environment:
```bash
   git clone https://github.com/jmac052002/recall.git
   cd recall
   python3 -m venv .venv
   source .venv/bin/activate
```

2. Copy the environment template and fill in your credentials:
```bash
   cp .env.example .env
```

3. Install the project in editable mode:
```bash
   pip install -e .
```

4. Start PostgreSQL via Docker:
```bash
   docker compose -f docker/docker-compose.yml up -d
```

5. Initialize the database schema:
```bash
   python scripts/init_db.py
```

## Usage

```bash
recall ask "My name is Joseph and I'm learning DevOps engineering"
recall ask "What do you remember about me?"
```

## Project Status

🚧 Under active development. This is a learning project built incrementally, phase by phase:

- [x] **Phase 1 — Foundation**: project structure, Docker + pgvector, env config
- [ ] **Phase 2 — Embeddings**: Voyage AI integration, embedding pipeline
- [ ] **Phase 3 — Memory retrieval**: vector + keyword dual search
- [ ] **Phase 4 — Agent loop**: full Claude conversation loop with memory injection
- [ ] **Phase 5 — Memory consolidation**: automatic fact extraction & summarization
- [ ] **Phase 6 — CLI polish**: session commands, memory inspection tools

## Acknowledgments

Architecture inspired in part by [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) — specifically its FTS5 session search and skills-based procedural memory approach.
