"""Repository and atmosphere backend for the unified Index.

A ``Repository`` pairs an ``IndexProvider`` (persistence backend) with an
optional ``AbstractDataStore`` (shard storage), forming a named storage unit
that can be mounted into an ``Index``.

The ``_AtmosphereBackend`` is an internal adapter that wraps an
``Atmosphere`` to present the same operational surface as a repository,
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
    """Internal adapter wrapping Atmosphere for Index routing.

    This class extracts the operational logic from ``AtmosphereIndex`` into an
    internal component that the unified ``Index`` uses for ATProto resolution.
    It is not part of the public API.

    The backend is lazily initialised -- the publishers/loaders are only
    created when the client is authenticated or when operations require them.
    """

    def __init__(
        self,
        client: Any,  # Atmosphere, typed as Any to avoid hard import
        *,
        data_store: Optional[AbstractDataStore] = None,
    ) -> None:
        from .atmosphere.client import Atmosphere

        if not isinstance(client, Atmosphere):
            raise TypeError(f"Expected Atmosphere, got {type(client).__name__}")
        self.client: Atmosphere = client
        self._data_store = data_store
        self._schema_publisher: Any = None
        self._schema_loader: Any = None
        self._dataset_publisher: Any = None
        self._dataset_loader: Any = None
        self._label_publisher: Any = None
        self._label_loader: Any = None

    def _ensure_loaders(self) -> None:
        """Lazily create publishers/loaders on first use."""
        if self._schema_loader is not None:
            return
        from .atmosphere.schema import SchemaPublisher, SchemaLoader
        from .atmosphere.records import DatasetPublisher, DatasetLoader
        from .atmosphere.labels import LabelPublisher, LabelLoader

        self._schema_publisher = SchemaPublisher(self.client)
        self._schema_loader = SchemaLoader(self.client)
        self._dataset_publisher = DatasetPublisher(self.client)
        self._dataset_loader = DatasetLoader(self.client)
        self._label_publisher = LabelPublisher(self.client)
        self._label_loader = LabelLoader(self.client)

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
        data_urls: list[str] | None = None,
        blob_refs: list[dict] | None = None,
        checksums: list | None = None,
        **kwargs: Any,
    ) -> Any:
        """Insert a dataset into ATProto.

        When *blob_refs* is provided the record uses ``storageBlobs`` with
        embedded blob reference objects so the PDS retains the uploaded blobs.

        When *data_urls* is provided (without *blob_refs*) the record uses
        ``storageExternal`` with those URLs.

        Args:
            ds: The Dataset to publish.
            name: Human-readable name.
            schema_ref: Optional schema AT URI. If None, auto-publishes schema.
            data_urls: Explicit shard URLs to store in the record.  When
                provided, these replace whatever ``ds.url`` contains.
            blob_refs: Pre-uploaded blob reference dicts from
                ``PDSBlobStore``.  Takes precedence over *data_urls*.
            checksums: Per-shard ``ShardChecksum`` objects. Forwarded to the
                publisher so each storage entry gets the correct digest.
            **kwargs: Additional options (description, tags, license).

        Returns:
            AtmosphereIndexEntry for the inserted dataset.
        """
        self._ensure_loaders()
        from .atmosphere import AtmosphereIndexEntry

        if blob_refs is not None or data_urls is not None:
            # Ensure schema is published first
            if schema_ref is None:
                from .atmosphere import SchemaPublisher

                sp = SchemaPublisher(self.client)
                schema_uri_obj = sp.publish(
                    ds.sample_type,
                    version=kwargs.get("schema_version", "1.0.0"),
                )
                schema_ref = str(schema_uri_obj)

            metadata = kwargs.get("metadata")
            if metadata is None and hasattr(ds, "_metadata"):
                metadata = ds._metadata

            if blob_refs is not None:
                uri = self._dataset_publisher.publish_with_blob_refs(
                    blob_refs=blob_refs,
                    schema_uri=schema_ref,
                    name=name,
                    description=kwargs.get("description"),
                    tags=kwargs.get("tags"),
                    license=kwargs.get("license"),
                    metadata=metadata,
                    checksums=checksums,
                )
            else:
                uri = self._dataset_publisher.publish_with_urls(
                    urls=data_urls,
                    schema_uri=schema_ref,
                    name=name,
                    description=kwargs.get("description"),
                    tags=kwargs.get("tags"),
                    license=kwargs.get("license"),
                    metadata=metadata,
                )
        else:
            uri = self._dataset_publisher.publish(
                ds,
                name=name,
                schema_uri=schema_ref,
                description=kwargs.get("description"),
                tags=kwargs.get("tags"),
                license=kwargs.get("license"),
                auto_publish_schema=(schema_ref is None),
            )

        # Create a label record for name-based resolution (best-effort;
        # the dataset record is already committed so we log and continue
        # if the label publish fails).
        try:
            self._label_publisher.publish(
                name=name,
                dataset_uri=str(uri),
                version=kwargs.get("version"),
                description=kwargs.get("description"),
            )
        except Exception:
            from ._logging import get_logger

            get_logger().warning(
                "Label publish failed for dataset %s (uri=%s); "
                "dataset was created but label was not.",
                name,
                uri,
                exc_info=True,
            )

        record = self._dataset_loader.get(uri)
        return AtmosphereIndexEntry(str(uri), record)

    # -- Label operations --

    def resolve_label(
        self,
        handle_or_did: str,
        name: str,
        version: str | None = None,
    ) -> str:
        """Resolve a named label to its dataset AT URI.

        Args:
            handle_or_did: DID or handle of the dataset owner.
            name: Label name (e.g. 'mnist').
            version: Specific version to resolve.

        Returns:
            AT URI of the referenced dataset record.

        Raises:
            KeyError: If no matching label is found.
        """
        self._ensure_loaders()
        return self._label_loader.resolve(handle_or_did, name, version)

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

    def get_schema_type(self, ref: str) -> type:
        """Reconstruct a Python type from a schema record.

        Args:
            ref: AT URI of the schema record.

        Returns:
            Dynamically generated Packable type.
        """
        from ._schema_codec import _schema_to_type

        schema = self.get_schema(ref)
        return _schema_to_type(schema)

    def decode_schema(self, ref: str) -> type:
        """Reconstruct a Python type from a schema record.

        .. deprecated::
            Use :meth:`get_schema_type` instead.
        """
        import warnings

        warnings.warn(
            "Repository.decode_schema() is deprecated, use Repository.get_schema_type() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_schema_type(ref)


__all__ = [
    "Repository",
    "create_repository",
]
