from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from marsrag.settings import get_settings

_settings = get_settings()
_pool: ConnectionPool | None = None


def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            conninfo=_settings.database_url,
            min_size=1,
            max_size=5,
            kwargs={"row_factory": dict_row},
            open=True,
        )
    return _pool


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


def fetch_one(query: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
    with get_pool().connection() as conn:
        return conn.execute(query, params).fetchone()


def fetch_all(query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    with get_pool().connection() as conn:
        return list(conn.execute(query, params).fetchall())


def execute(query: str, params: tuple[Any, ...] = ()) -> int:
    with get_pool().connection() as conn:
        cursor = conn.execute(query, params)
        return cursor.rowcount


def check_database() -> bool:
    try:
        return fetch_one("SELECT 1 AS healthy") is not None
    except Exception:
        return False


def migrate() -> None:
    """Apply SQL migrations; vector dimension is safely substituted from validated settings."""
    migration_dir = Path(__file__).resolve().parents[2] / "sql"
    dimension = _settings.embed_dim
    if not 1 <= dimension <= 2000:
        raise ValueError("Unsafe embedding dimension.")
    migrations = sorted(migration_dir.glob("[0-9][0-9][0-9]_*.sql"))
    if not migrations:
        raise FileNotFoundError(f"No SQL migrations found in {migration_dir}")
    with psycopg.connect(_settings.database_url) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations (name TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())")
        for path in migrations:
            applied = conn.execute(
                "SELECT 1 FROM schema_migrations WHERE name = %s", (path.name,)
            ).fetchone()
            if applied:
                continue
            sql = path.read_text(encoding="utf-8").replace("{{EMBED_DIM}}", str(dimension))
            with conn.transaction():
                conn.execute(sql)
                conn.execute("INSERT INTO schema_migrations(name) VALUES (%s)", (path.name,))
            print(f"Applied {path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["migrate", "check"])
    command = parser.parse_args().command
    if command == "migrate":
        migrate()
    else:
        print("database healthy" if check_database() else "database unavailable")
