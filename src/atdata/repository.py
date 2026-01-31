"""Repository and atmosphere backend for the unified Index.

A ``Repository`` pairs an ``IndexProvider`` (persistence backend) with an
optional ``AbstractDataStore`` (shard storage), forming a named storage unit
that can be mounted into an ``Index``.

The ``_AtmosphereBackend`` is an internal adapter that wraps an
``AtmosphereClient`` to present the same operational surface as a repository,
but routes through the ATProto network instead of a local provider.

Examples:
    >>> from atdata.repository import Repository, create_repository
    >>> repo = Repository(provider=SqliteProvider("/data/lab.db"))
    >>> repo = create_repository("sqlite", path="/data/lab.db")
    >>>
    >>> # With a data store for shard storage
    >>> repo = Repository(
    ...     provider=SqliteProvider(),
    ...     data_store=S3DataStore(credentials, bucket="lab-data"),
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional, TYPE_CHECKING

from ._protocols import AbstractDataStore

if TYPE_CHECKING:
    from .providers._base import IndexProvider


@dataclass
class Repository:
    """A named storage backend pairing index persistence with optional data storage.

    Repositories are mounted into an ``Index`` by name. The built-in ``"local"``
    repository uses SQLite by default; additional repositories can be added for
    multi-source dataset management.

    Attributes:
        provider: IndexProvider handling dataset/schema persistence.
        data_store: Optional data store for reading/writing dataset shards.
            If present, ``insert_dataset`` will write shards to this store.

    Examples:
        >>> from atdata.providers import create_provider
        >>> from atdata.repository import Repository
        >>>
        >>> provider = create_provider("sqlite", path="/data/lab.db")
        >>> repo = Repository(provider=provider)
        >>>
        >>> # With S3 shard storage
        >>> repo = Repository(
        ...     provider=provider,
        ...     data_store=S3DataStore(credentials, bucket="lab-data"),
        ... )
    """

    provider: IndexProvider
    data_store: AbstractDataStore | None = None


def create_repository(
    provider: str = "sqlite",
    *,
    path: str | Path | None = None,
    dsn: str | None = None,
    redis: Any = None,
    data_store: AbstractDataStore | None = None,
    **kwargs: Any,
) -> Repository:
    """Create a Repository with a provider by name.

    This is a convenience factory that combines ``create_provider`` with
    ``Repository`` construction.

    Args:
        provider: Backend name: ``"sqlite"``, ``"redis"``, or ``"postgres"``.
        path: Database file path (SQLite only).
        dsn: Connection string (PostgreSQL only).
        redis: Existing Redis connection (Redis only).
        data_store: Optional data store for shard storage.
        **kwargs: Extra arguments forwarded to the provider constructor.

    Returns:
        A ready-to-use Repository.

    Raises:
        ValueError: If provider name is not recognised.

    Examples:
        >>> repo = create_repository("sqlite", path="/data/lab.db")
        >>> repo = create_repository(
        ...     "sqlite",
        ...     data_store=S3DataStore(creds, bucket="lab"),
        ... )
    """
    from .providers._factory import create_provider as _create_provider

    backend = _create_provider(provider, path=path, dsn=dsn, redis=redis, **kwargs)
    return Repository(provider=backend, data_store=data_store)


class _AtmosphereBackend:
    """Internal adapter wrapping AtmosphereClient for Index routing.

    This class extracts the operational logic from ``AtmosphereIndex`` into an
    internal component that the unified ``Index`` uses for ATProto resolution.
    It is not part of the public API.

    The backend is lazily initialised -- the publishers/loaders are only
    created when the client is authenticated or when operations require them.
    """

    def __init__(
        self,
        client: Any,  # AtmosphereClient, typed as Any to avoid hard import
        *,
        data_store: Optional[AbstractDataStore] = None,
    ) -> None:
        from .atmosphere.client import AtmosphereClient

        if not isinstance(client, AtmosphereClient):
            raise TypeError(f"Expected AtmosphereClient, got {type(client).__name__}")
        self.client: AtmosphereClient = client
        self._data_store = data_store
        self._schema_publisher: Any = None
        self._schema_loader: Any = None
        self._dataset_publisher: Any = None
        self._dataset_loader: Any = None

    def _ensure_loaders(self) -> None:
        """Lazily create publishers/loaders on first use."""
        if self._schema_loader is not None:
            return
        from .atmosphere.schema import SchemaPublisher, SchemaLoader
        from .atmosphere.records import DatasetPublisher, DatasetLoader

        self._schema_publisher = SchemaPublisher(self.client)
        self._schema_loader = SchemaLoader(self.client)
        self._dataset_publisher = DatasetPublisher(self.client)
        self._dataset_loader = DatasetLoader(self.client)

    @property
    def data_store(self) -> Optional[AbstractDataStore]:
        """The data store for this atmosphere backend, or None."""
        return self._data_store

    # -- Dataset operations --

    def get_dataset(self, ref: str) -> Any:
        """Get a dataset entry by name or AT URI.

        Args:
            ref: Dataset name or AT URI.

        Returns:
            AtmosphereIndexEntry for the dataset.

        Raises:
            ValueError: If record is not a dataset.
        """
        self._ensure_loaders()
        from .atmosphere import AtmosphereIndexEntry

        record = self._dataset_loader.get(ref)
        return AtmosphereIndexEntry(ref, record)

    def list_datasets(self, repo: str | None = None) -> list[Any]:
        """List all dataset entries.

        Args:
            repo: DID of repository. Defaults to authenticated user.

        Returns:
            List of AtmosphereIndexEntry for each dataset.
        """
        self._ensure_loaders()
        from .atmosphere import AtmosphereIndexEntry

        records = self._dataset_loader.list_all(repo=repo)
        return [
            AtmosphereIndexEntry(rec.get("uri", ""), rec.get("value", rec))
            for rec in records
        ]

    def iter_datasets(self, repo: str | None = None) -> Iterator[Any]:
        """Lazily iterate over all dataset entries.

        Args:
            repo: DID of repository. Defaults to authenticated user.

        Yields:
            AtmosphereIndexEntry for each dataset.
        """
        self._ensure_loaders()
        from .atmosphere import AtmosphereIndexEntry

        records = self._dataset_loader.list_all(repo=repo)
        for rec in records:
            uri = rec.get("uri", "")
            yield AtmosphereIndexEntry(uri, rec.get("value", rec))

    def insert_dataset(
        self,
        ds: Any,
        *,
        name: str,
        schema_ref: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Insert a dataset into ATProto.

        Args:
            ds: The Dataset to publish.
            name: Human-readable name.
            schema_ref: Optional schema AT URI. If None, auto-publishes schema.
            **kwargs: Additional options (description, tags, license).

        Returns:
            AtmosphereIndexEntry for the inserted dataset.
        """
        self._ensure_loaders()
        from .atmosphere import AtmosphereIndexEntry

        uri = self._dataset_publisher.publish(
            ds,
            name=name,
            schema_uri=schema_ref,
            description=kwargs.get("description"),
            tags=kwargs.get("tags"),
            license=kwargs.get("license"),
            auto_publish_schema=(schema_ref is None),
        )
        record = self._dataset_loader.get(uri)
        return AtmosphereIndexEntry(str(uri), record)

    # -- Schema operations --

    def publish_schema(
        self,
        sample_type: type,
        *,
        version: str = "1.0.0",
        **kwargs: Any,
    ) -> str:
        """Publish a schema to ATProto.

        Args:
            sample_type: A Packable type.
            version: Semantic version string.
            **kwargs: Additional options.

        Returns:
            AT URI of the schema record.
        """
        self._ensure_loaders()
        uri = self._schema_publisher.publish(
            sample_type,
            version=version,
            description=kwargs.get("description"),
            metadata=kwargs.get("metadata"),
        )
        return str(uri)

    def get_schema(self, ref: str) -> dict:
        """Get a schema record by AT URI.

        Args:
            ref: AT URI of the schema record.

        Returns:
            Schema record dictionary.
        """
        self._ensure_loaders()
        return self._schema_loader.get(ref)

    def list_schemas(self, repo: str | None = None) -> list[dict]:
        """List all schema records.

        Args:
            repo: DID of repository. Defaults to authenticated user.

        Returns:
            List of schema records as dictionaries.
        """
        self._ensure_loaders()
        records = self._schema_loader.list_all(repo=repo)
        return [rec.get("value", rec) for rec in records]

    def iter_schemas(self) -> Iterator[dict]:
        """Lazily iterate over all schema records.

        Yields:
            Schema records as dictionaries.
        """
        self._ensure_loaders()
        records = self._schema_loader.list_all()
        for rec in records:
            yield rec.get("value", rec)

    def decode_schema(self, ref: str) -> type:
        """Reconstruct a Python type from a schema record.

        Args:
            ref: AT URI of the schema record.

        Returns:
            Dynamically generated Packable type.
        """
        from ._schema_codec import schema_to_type

        schema = self.get_schema(ref)
        return schema_to_type(schema)


__all__ = [
    "Repository",
    "create_repository",
]
