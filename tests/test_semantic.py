"""Unit tests for recall.memory.semantic (PostgreSQL + pgvector).

These tests never touch a real PostgreSQL instance. store_memory and
semantic_search are thin, well-defined wrappers around two SQL statements,
so the useful thing to verify is that they build the right query and pass
the right parameters, not that pgvector's cosine distance operator itself
works correctly (that is PostgreSQL's job, not ours). psycopg2.connect is
replaced with a small in-memory fake so these tests run anywhere, without
Docker, and stay fast and deterministic.
"""
import pytest

from recall.memory import semantic


class FakeCursor:
    def __init__(self):
        self.executed = []
        self.fetchone_result = None
        self.fetchall_result = []

    def execute(self, query, params=None):
        self.executed.append((query, params))

    def fetchone(self):
        return self.fetchone_result

    def fetchall(self):
        return self.fetchall_result

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.committed = False
        self.rolled_back = False
        self.closed = False

    def cursor(self, cursor_factory=None):
        return self._cursor

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self.closed = True


@pytest.fixture
def fake_cursor():
    return FakeCursor()


@pytest.fixture
def fake_conn(fake_cursor):
    return FakeConnection(fake_cursor)


@pytest.fixture(autouse=True)
def _patch_connect(monkeypatch, fake_conn):
    monkeypatch.setattr(semantic.psycopg2, "connect", lambda dsn: fake_conn)


def test_store_memory_inserts_and_returns_new_id(fake_cursor):
    fake_cursor.fetchone_result = (42,)

    new_id = semantic.store_memory(
        content="Joseph is learning pgvector.",
        embedding=[0.1, 0.2, 0.3],
        source="test",
        session_id="abc-123",
    )

    assert new_id == 42
    query, params = fake_cursor.executed[0]
    assert "INSERT INTO memories" in query
    assert params[0] == "Joseph is learning pgvector."
    assert params[1] == [0.1, 0.2, 0.3]
    assert params[2] == "test"
    assert params[3] == "abc-123"


def test_store_memory_defaults_metadata_to_empty_dict(fake_cursor):
    fake_cursor.fetchone_result = (1,)

    semantic.store_memory(
        content="No metadata passed.",
        embedding=[0.0],
        source="test",
    )

    _, params = fake_cursor.executed[0]
    # metadata is wrapped in psycopg2.extras.Json before being passed along,
    # so check the value it actually wrapped rather than the wrapper itself.
    assert params[4].adapted == {}


def test_store_memory_commits_and_closes_on_success(fake_cursor, fake_conn):
    fake_cursor.fetchone_result = (1,)

    semantic.store_memory(content="x", embedding=[0.0], source="test")

    assert fake_conn.committed is True
    assert fake_conn.closed is True


def test_semantic_search_returns_rows_from_the_cursor(fake_cursor):
    fake_cursor.fetchall_result = [
        {
            "id": 1,
            "content": "Joseph is learning pgvector.",
            "source": "test",
            "session_id": None,
            "created_at": "2026-07-19T00:00:00Z",
            "similarity": 0.92,
        },
    ]

    results = semantic.semantic_search(query_embedding=[0.1, 0.2, 0.3])

    assert len(results) == 1
    assert results[0]["similarity"] == 0.92


def test_semantic_search_converts_similarity_threshold_to_distance(fake_cursor):
    semantic.semantic_search(query_embedding=[0.1], similarity_threshold=0.8)

    _, params = fake_cursor.executed[0]
    embedding, embedding_again, distance_threshold, embedding_third, limit = params
    # cosine DISTANCE = 1 - cosine SIMILARITY, so an 0.8 similarity floor
    # is a 0.2 distance ceiling.
    assert distance_threshold == pytest.approx(0.2)
    assert limit == 5  # default limit


def test_semantic_search_uses_custom_limit(fake_cursor):
    semantic.semantic_search(query_embedding=[0.1], limit=10)

    _, params = fake_cursor.executed[0]
    assert params[-1] == 10
