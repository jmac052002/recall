"""Unit tests for recall.memory.episodic (SQLite + FTS5 keyword search).

These run against a real, throwaway SQLite file per test (via the
episodic_db_path fixture in conftest.py) instead of mocking sqlite3.
SQLite is embedded and fast enough that an integration-style test here is
both simpler and more honest than mocking the database layer out, since
the thing actually worth verifying is that FTS5 indexing and BM25 ranking
behave the way the rest of Recall assumes they do.
"""
import pytest

from recall.memory import episodic


@pytest.fixture(autouse=True)
def _fresh_schema(episodic_db_path):
    """Every test in this file gets its own schema-initialized SQLite file."""
    episodic.init_schema()
    yield


def test_init_schema_is_idempotent():
    # init_schema uses CREATE TABLE IF NOT EXISTS / CREATE VIRTUAL TABLE IF
    # NOT EXISTS throughout, so calling it again should never raise.
    episodic.init_schema()
    episodic.init_schema()


def test_save_conversation_returns_a_session_id():
    session_id = episodic.save_conversation([
        {"role": "user", "content": "My name is Joseph."},
        {"role": "assistant", "content": "Nice to meet you, Joseph."},
    ])

    assert isinstance(session_id, str)
    assert len(session_id) > 0


def test_keyword_search_finds_a_stored_message():
    episodic.save_conversation([
        {"role": "user", "content": "I am learning Terraform and Kubernetes."},
        {"role": "assistant", "content": "Terraform handles infrastructure as code."},
    ])
    episodic.save_conversation([
        {"role": "user", "content": "What should I eat for dinner tonight?"},
        {"role": "assistant", "content": "How about pasta?"},
    ])

    results = episodic.keyword_search("Terraform")

    assert len(results) >= 1
    assert any("Terraform" in r["content"] for r in results)
    # the unrelated dinner conversation should not show up for this query
    assert all("pasta" not in r["content"].lower() for r in results)


def test_keyword_search_respects_limit():
    for i in range(5):
        episodic.save_conversation([
            {"role": "user", "content": f"Message number {i} about Docker."},
            {"role": "assistant", "content": "Docker containers are isolated."},
        ])

    results = episodic.keyword_search("Docker", limit=2)

    assert len(results) == 2


def test_keyword_search_returns_empty_list_when_nothing_matches():
    episodic.save_conversation([
        {"role": "user", "content": "Let's talk about AWS Lambda."},
        {"role": "assistant", "content": "Lambda is a serverless compute service."},
    ])

    results = episodic.keyword_search("nonexistentword12345")

    assert results == []
