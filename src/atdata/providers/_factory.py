"""Factory for creating index providers by name.

Examples:
    >>> from atdata.providers._factory import create_provider
    >>> provider = create_provider("sqlite", path="/tmp/index.db")
    >>> provider = create_provider("redis")
    >>> provider = create_provider("postgres", dsn="postgresql://localhost/mydb")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ._base import IndexProvider


def create_provider(
    name: str,
    *,
    path: str | Path | None = None,
    dsn: str | None = None,
    redis: Any = None,
    **kwargs: Any,
) -> IndexProvider:
    """Instantiate an ``IndexProvider`` by backend name.

    Args:
        name: One of ``"redis"``, ``"sqlite"``, or ``"postgres"``.
        path: Database file path (SQLite).  Defaults to
            ``~/.atdata/index.db`` when *name* is ``"sqlite"``.
        dsn: Connection string (PostgreSQL).
        redis: An existing ``redis.Redis`` connection (Redis).  When
            ``None`` and *name* is ``"redis"``, a new connection is
            created from *kwargs*.
        **kwargs: Extra arguments forwarded to the provider constructor
            (e.g. Redis host/port).

    Returns:
        A ready-to-use ``IndexProvider``.

    Raises:
        ValueError: If *name* is not a recognised backend.
    """
    name = name.lower().strip()

    if name == "redis":
        from ._redis import RedisProvider
        from redis import Redis as _Redis

        if redis is not None:
            return RedisProvider(redis)
        return RedisProvider(_Redis(**kwargs))

    if name == "sqlite":
        from ._sqlite import SqliteProvider

        return SqliteProvider(path=path)

    if name in ("postgres", "postgresql"):
        from ._postgres import PostgresProvider

        if dsn is None:
            raise ValueError("dsn is required for the postgres provider")
        return PostgresProvider(dsn=dsn)

    raise ValueError(
        f"Unknown provider {name!r}. Choose from: 'redis', 'sqlite', 'postgres'."
    )
