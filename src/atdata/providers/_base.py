"""Abstract base class for index storage providers.

The ``IndexProvider`` ABC defines the persistence contract that the ``Index``
class delegates to.  Each provider handles storage and retrieval of two entity
types — dataset entries and schema records — using whatever backend it wraps.

Concrete implementations live in sibling modules:
    ``_redis.py``, ``_sqlite.py``, ``_postgres.py``
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from ..local import LocalDatasetEntry


class IndexProvider(ABC):
    """Storage backend for the ``Index`` class.

    Implementations persist ``LocalDatasetEntry`` objects and schema JSON
    records.  The ``Index`` class owns all business logic (CID generation,
    version bumping, schema building); the provider is a pure persistence
    layer.

    Examples:
        >>> from atdata.providers import create_provider
        >>> provider = create_provider("sqlite", path="/tmp/index.db")
        >>> provider.store_schema("MySample", "1.0.0", '{"name": "MySample"}')
        >>> provider.get_schema_json("MySample", "1.0.0")
        '{"name": "MySample"}'
    """

    # ------------------------------------------------------------------
    # Dataset entry operations
    # ------------------------------------------------------------------

    @abstractmethod
    def store_entry(self, entry: LocalDatasetEntry) -> None:
        """Persist a dataset entry (upsert by CID).

        Args:
            entry: The dataset entry to store.  The entry's ``cid`` property
                is used as the primary key.
        """

    @abstractmethod
    def get_entry_by_cid(self, cid: str) -> LocalDatasetEntry:
        """Load a dataset entry by its content identifier.

        Args:
            cid: Content-addressable identifier.

        Returns:
            The matching ``LocalDatasetEntry``.

        Raises:
            KeyError: If no entry exists for *cid*.
        """

    @abstractmethod
    def get_entry_by_name(self, name: str) -> LocalDatasetEntry:
        """Load a dataset entry by its human-readable name.

        Args:
            name: Dataset name.

        Returns:
            The first matching ``LocalDatasetEntry``.

        Raises:
            KeyError: If no entry exists with *name*.
        """

    @abstractmethod
    def iter_entries(self) -> Iterator[LocalDatasetEntry]:
        """Iterate over all stored dataset entries.

        Yields:
            ``LocalDatasetEntry`` objects in unspecified order.
        """

    # ------------------------------------------------------------------
    # Schema operations
    # ------------------------------------------------------------------

    @abstractmethod
    def store_schema(self, name: str, version: str, schema_json: str) -> None:
        """Persist a schema record (upsert by name + version).

        Args:
            name: Schema name (e.g. ``"MySample"``).
            version: Semantic version string (e.g. ``"1.0.0"``).
            schema_json: JSON-serialized schema record.
        """

    @abstractmethod
    def get_schema_json(self, name: str, version: str) -> str | None:
        """Load a schema's JSON by name and version.

        Args:
            name: Schema name.
            version: Semantic version string.

        Returns:
            The JSON string, or ``None`` if not found.
        """

    @abstractmethod
    def iter_schemas(self) -> Iterator[tuple[str, str, str]]:
        """Iterate over all stored schemas.

        Yields:
            Tuples of ``(name, version, schema_json)``.
        """

    @abstractmethod
    def find_latest_version(self, name: str) -> str | None:
        """Find the latest semantic version for a schema name.

        Args:
            name: Schema name to search for.

        Returns:
            The latest version string (e.g. ``"1.2.3"``), or ``None``
            if no schema with *name* exists.
        """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release any resources held by the provider.

        The default implementation is a no-op.  Providers that hold
        connections (SQLite, PostgreSQL) should override this.
        """
