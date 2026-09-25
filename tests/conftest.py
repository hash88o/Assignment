import pytest

from app import db


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    db.init_db()
    return tmp_path / "test.db"


@pytest.fixture()
def props(tmp_db):
    """All 12 seeded properties, by user."""
    return {u: db.list_properties(u) for u in ("U001", "U002", "U003", "U004")}
