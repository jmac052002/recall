# recall/memory/semantic.py
import psycopg2
import psycopg2.extras
from contextlib import contextmanager
from typing import Iterator

from recall.config import settings


@contextmanager
def _get_conn() -> Iterator[psycopg2.extensions.connection]:
    """Context manager: commits on success, rolls back on error, always closes."""
    conn = psycopg2.connect(settings.postgres_dsn)
    conn.autocommit = False
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def store_memory(
    content: str,
    embedding: list[float],
    source: str,
    session_id: str | None = None,
    metadata: dict | None = None,
) -> int:
    """Store an embedded memory. Returns the new row ID."""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO memories (content, embedding, source, session_id, metadata)
                VALUES (%s, %s::vector, %s, %s, %s)
                RETURNING id
                """,
                (
                    content,
                    embedding,
                    source,
                    session_id,
                    psycopg2.extras.Json(metadata or {}),
                ),
            )
            return cur.fetchone()[0]


def semantic_search(
    query_embedding: list[float],
    limit: int = 5,
    similarity_threshold: float = 0.75,
) -> list[dict]:
    """
    Cosine similarity search using pgvector.
    <=> is the cosine DISTANCE operator — 0 means identical, 2 means opposite.
    """
    distance_threshold = 1.0 - similarity_threshold

    with _get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    id,
                    content,
                    source,
                    session_id,
                    created_at,
                    1 - (embedding <=> %s::vector) AS similarity
                FROM memories
                WHERE embedding <=> %s::vector < %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (
                    query_embedding,
                    query_embedding,
                    distance_threshold,
                    query_embedding,
                    limit,
                ),
            )
            return [dict(row) for row in cur.fetchall()]