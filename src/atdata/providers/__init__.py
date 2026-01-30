"""Storage provider backends for the local Index.

This package defines the ``IndexProvider`` abstract base class and concrete
implementations for Redis, SQLite, and PostgreSQL. The ``Index`` class in
``atdata.local`` delegates all persistence to an ``IndexProvider``.

Providers:
    RedisProvider: Redis-backed storage (existing default).
    SqliteProvider: SQLite file-based storage (zero external dependencies).
    PostgresProvider: PostgreSQL storage (requires ``psycopg``).

Examples:
    >>> from atdata.providers import IndexProvider, create_provider
    >>> provider = create_provider("sqlite", path="~/.atdata/index.db")
    >>> from atdata.local import Index
    >>> index = Index(provider=provider)
"""

from ._base import IndexProvider
from ._factory import create_provider

__all__ = [
    "IndexProvider",
    "create_provider",
]
