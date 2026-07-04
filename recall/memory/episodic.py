import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager
from typing import Iterator

from recall.config import settings


def _ensure_db_dir() -> Path:
    db_path = settings.recall_db_path.expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


@contextmanager
def _get_conn() -> Iterator[sqlite3.Connection]:
    """Context manager: always commits on success, rolls back on error, always closes."""
    conn = sqlite3.connect(_ensure_db_dir())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA Journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
        
        
def init_schema() -> None:
    """Idempotent - safe to call every time the app starts."""
    with _get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id          TEXT PRIMARY KEY,
                started_at  TEXT NOT NULL,
                summary     TEXT
            );

            CREATE TABLE IF NOT EXISTS messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL REFERENCES sessions(id),
                role        TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content     TEXT NOT NULL,
                created_at  TEXT NOT NULL
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                content,
                session_id UNINDEXED,
                message_id UNINDEXED,
                tokenize='porter ascii'
            );
        """)
        
        
def save_conversation(messages: list[dict]) -> str:
    """Persist a full turn (user + assistant messages) and index it for FTS search."""
    session_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO sessions (id, started_at) VALUES (?, ?)",
            (session_id, now)
        )
        
        for msg in messages:
            cur = conn.execute(
                "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (session_id, msg["role"], msg["content"], now),
            )
            message_id = cur.lastrowid
            conn.execute(
                "INSERT INTO messages_fts (content, session_id, message_id) VALUES (?, ?, ?)",
                (msg["content"], session_id, message_id),
            )
    return session_id 



def keyword_search(query: str, limit: int = 5) -> list[dict]:
    """BM25-ranked full-text search. Lower rank = better match."""
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT
                m.content,
                m.role,
                m.session_id,
                m.created_at,
                bm25(messages_fts) AS rank
            FROM messages_fts
            JOIN messages m ON m.id = messages_fts.message_id
            WHERE messages_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (query, limit),
        ).fetchall()
    return [dict(row) for row in rows] 