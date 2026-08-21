"""Shared pytest fixtures for Recall's test suite.

recall.config.Settings() runs at import time and requires several
environment-backed fields (API keys, Postgres credentials). None of these
need to be real for the tests in this suite: episodic tests use a
throwaway SQLite file on disk, and semantic tests replace psycopg2.connect
with an in-memory fake, so no live Postgres or external API is ever
contacted. We set placeholder values here, before anything under
`recall/` gets imported, so `Settings()` validates successfully in any
environment (local machine, CI, wherever) without needing a real .env file.
"""
import os

os.environ.setdefault("ANTHROPIC_API_KEY", "test-anthropic-key")
os.environ.setdefault("VOYAGE_API_KEY", "test-voyage-key")
os.environ.setdefault("POSTGRES_USER", "test_user")
os.environ.setdefault("POSTGRES_PASSWORD", "test_password")
os.environ.setdefault("POSTGRES_DB", "test_db")

import pytest


@pytest.fixture
def episodic_db_path(tmp_path, monkeypatch):
    """Point recall's SQLite episodic store at a throwaway file for one test.

    settings is a singleton created once at import time, so tests can't get
    a fresh one from environment variables alone. Patching the attribute
    directly is what actually redirects _ensure_db_dir() in episodic.py.
    """
    from recall.config import settings

    db_path = tmp_path / "recall_test.db"
    monkeypatch.setattr(settings, "recall_db_path", db_path)
    return db_path
