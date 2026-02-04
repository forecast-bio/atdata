"""Index class for local dataset management."""

from __future__ import annotations

from atdata import (
    Dataset,
)
from atdata._protocols import AbstractDataStore, Packable

from atdata.index._entry import LocalDatasetEntry
from atdata.index._schema import (
    SchemaNamespace,
    LocalSchemaRecord,
    _schema_ref_from_type,
    _make_schema_ref,
    _parse_schema_ref,
    _parse_lens_ref,
    _increment_patch,
    _build_schema_record,
)

from pathlib import Path
from typing import (
    Any,
    Iterable,
    Type,
    TypeVar,
    Generator,
    TYPE_CHECKING,
)
import json

if TYPE_CHECKING:
    from redis import Redis

    from atdata.providers._base import IndexProvider
    from atdata.repository import Repository, _AtmosphereBackend
    from atdata._protocols import IndexEntry
    from atdata.lens import Lens

T = TypeVar("T", bound=Packable)


def _is_local_path(url: str) -> bool:
    """Check if a URL points to the local filesystem."""
    return (
        url.startswith("/")
        or url.startswith("file://")
        or (len(url) > 1 and url[1] == ":")
    )


def _is_credentialed_source(ds: Dataset) -> bool:
    """Check if a Dataset uses a credentialed source (e.g. S3Source with keys)."""
    from atdata._sources import S3Source

    return isinstance(ds.source, S3Source)


def _merge_checksums(
    metadata: dict | None,
    write_result: list[str],
) -> dict | None:
    """Merge shard checksums from a write result into entry metadata."""
    checksums = getattr(write_result, "checksums", None)
    if not checksums:
        return metadata
    if metadata is None:
        metadata = {}
    return {**metadata, "checksums": checksums}


def _estimate_dataset_bytes(ds: Dataset) -> int:
    """Best-effort total size estimate from local shard files.

    Returns 0 when size cannot be determined (e.g. remote URLs).
    """
    total = 0
    for shard_url in ds.list_shards():
        if _is_local_path(shard_url):
            p = Path(shard_url.removeprefix("file://"))
            if p.exists():
                total += p.stat().st_size
    return total


class Index:
    """Unified index for tracking datasets across multiple repositories.

    Implements the AbstractIndex protocol. Maintains a registry of
    dataset entries across named repositories (always including a built-in
    ``"local"`` repository) and an optional atmosphere (ATProto) backend.

    The ``"local"`` repository is always present and uses the storage backend
    determined by the ``provider`` argument. When no provider is given, defaults
    to SQLite (zero external dependencies). Pass a ``redis`` connection or
    Redis ``**kwargs`` for backwards-compatible Redis behaviour.

    Additional named repositories can be mounted via the ``repos`` parameter,
    each pairing an IndexProvider with an optional data store.

    An Atmosphere is available by default for anonymous read-only
    resolution of ``@handle/dataset`` paths. Pass an authenticated client
    for write operations, or ``atmosphere=None`` to disable.

    Attributes:
        _repos: All repositories keyed by name. ``"local"`` is always present.
        _atmosphere: Optional atmosphere backend for ATProto operations.
    """

    ##

    # Sentinel for default atmosphere behaviour (lazy anonymous client)
    _ATMOSPHERE_DEFAULT = object()

    def __init__(
        self,
        provider: IndexProvider | str | None = None,
        *,
        path: str | Path | None = None,
        dsn: str | None = None,
        redis: Redis | None = None,
        data_store: AbstractDataStore | None = None,
        repos: dict[str, Repository] | None = None,
        atmosphere: Any | None = _ATMOSPHERE_DEFAULT,
        auto_stubs: bool = False,
        stub_dir: Path | str | None = None,
        **kwargs,
    ) -> None:
        """Initialize an index.

        Args:
            provider: Storage backend for the ``"local"`` repository.
                Accepts an ``IndexProvider`` instance or a backend name
                string (``"sqlite"``, ``"redis"``, or ``"postgres"``).
                When ``None``, falls back to *redis* / *kwargs* if given,
                otherwise defaults to SQLite.
            path: Database file path (SQLite only).  Ignored unless
                *provider* is ``"sqlite"``.
            dsn: PostgreSQL connection string.  Required when *provider*
                is ``"postgres"``.
            redis: Redis connection to use (backwards-compat shorthand for
                ``RedisProvider(redis)``).  Ignored when *provider* is given.
            data_store: Optional data store for writing dataset shards in the
                ``"local"`` repository.  If provided, ``insert_dataset()`` will
                write shards to this store.  If None, only indexes existing URLs.
            repos: Named repositories to mount alongside ``"local"``.  Keys are
                repository names (e.g. ``"lab"``, ``"shared"``).  The name
                ``"local"`` is reserved for the built-in repository.
            atmosphere: ATProto client for distributed network operations.
                - Default (sentinel): creates an anonymous read-only client
                  lazily on first access.
                - ``Atmosphere`` instance: uses that client directly.
                - ``None``: disables atmosphere backend entirely.
            auto_stubs: If True, automatically generate .pyi stub files when
                schemas are accessed via get_schema() or decode_schema().
                This enables IDE autocomplete for dynamically decoded types.
            stub_dir: Directory to write stub files. Only used if auto_stubs
                is True or if this parameter is provided (which implies auto_stubs).
                Defaults to ~/.atdata/stubs/ if not specified.
            **kwargs: Additional arguments passed to Redis() constructor when
                *redis* is not given.  If any kwargs are provided (without an
                explicit *provider*), Redis is used instead of the SQLite default.

        Raises:
            TypeError: If provider is not an IndexProvider or valid string.
            ValueError: If repos contains the reserved name ``"local"``.

        Examples:
            >>> # Default: local SQLite + anonymous atmosphere
            >>> index = Index()
            >>>
            >>> # SQLite with explicit path
            >>> index = Index(provider="sqlite", path="~/.atdata/index.db")
            >>>
            >>> # Redis
            >>> index = Index(redis=redis_conn)
            >>>
            >>> # PostgreSQL
            >>> index = Index(provider="postgres", dsn="postgresql://user:pass@host/db")
            >>>
            >>> # Multiple repositories
            >>> from atdata.repository import Repository, create_repository
            >>> index = Index(
            ...     provider="sqlite",
            ...     repos={
            ...         "lab": create_repository("sqlite", path="/data/lab.db"),
            ...     },
            ... )
        """
        ##

        from atdata.providers._base import IndexProvider as _IP
        from atdata.repository import Repository as _Repo

        # Resolve the local provider
        if isinstance(provider, str):
            from atdata.providers._factory import create_provider

            local_provider: _IP = create_provider(
                provider, path=path, dsn=dsn, redis=redis, **kwargs
            )
        elif provider is not None:
            if not isinstance(provider, _IP):
                raise TypeError(
                    f"provider must be an IndexProvider or backend name string, "
                    f"got {type(provider).__name__}"
                )
            local_provider = provider
        elif redis is not None:
            from atdata.providers._redis import RedisProvider

            local_provider = RedisProvider(redis)
        elif kwargs:
            from redis import Redis as _Redis
            from atdata.providers._redis import RedisProvider

            local_provider = RedisProvider(_Redis(**kwargs))
        else:
            from atdata.providers._sqlite import SqliteProvider

            local_provider = SqliteProvider()

        # Build the unified repos dict with "local" always present
        self._repos: dict[str, _Repo] = {
            "local": _Repo(provider=local_provider, data_store=data_store),
        }

        if repos is not None:
            if "local" in repos:
                raise ValueError(
                    '"local" is reserved for the built-in repository. '
                    "Use a different name for your repository."
                )
            for name, repo in repos.items():
                if not isinstance(repo, _Repo):
                    raise TypeError(
                        f"repos[{name!r}] must be a Repository, "
                        f"got {type(repo).__name__}"
                    )
            self._repos.update(repos)

        # Atmosphere backend (lazy or explicit)
        from atdata.repository import _AtmosphereBackend

        if atmosphere is Index._ATMOSPHERE_DEFAULT:
            # Deferred: create anonymous client on first use
            self._atmosphere: _AtmosphereBackend | None = None
            self._atmosphere_deferred = True
        elif atmosphere is None:
            self._atmosphere = None
            self._atmosphere_deferred = False
        else:
            self._atmosphere = _AtmosphereBackend(atmosphere)
            self._atmosphere_deferred = False

        # Initialize stub manager if auto-stubs enabled
        # Providing stub_dir implies auto_stubs=True
        if auto_stubs or stub_dir is not None:
            from atdata._stub_manager import StubManager

            self._stub_manager: StubManager | None = StubManager(stub_dir=stub_dir)
        else:
            self._stub_manager = None

        # Initialize schema namespace for load_schema/schemas API
        self._schema_namespace = SchemaNamespace()

    # -- Repository access --

    def _get_atmosphere(self) -> "_AtmosphereBackend | None":
        """Get the atmosphere backend, lazily creating anonymous client if needed."""
        if self._atmosphere_deferred and self._atmosphere is None:
            try:
                from atdata.atmosphere.client import Atmosphere
                from atdata.repository import _AtmosphereBackend

                client = Atmosphere()
                self._atmosphere = _AtmosphereBackend(client)
            except ImportError:
                # atproto package not installed -- atmosphere unavailable
                self._atmosphere_deferred = False
                return None
        return self._atmosphere

    def _resolve_prefix(self, ref: str) -> tuple[str, str, str | None]:
        """Route a dataset/schema reference to the correct backend.

        Returns:
            Tuple of ``(backend_key, resolved_ref, handle_or_did)``.

            - ``backend_key``: ``"local"``, a named repository, or
              ``"_atmosphere"``.
            - ``resolved_ref``: The dataset/schema name or AT URI to pass
              to the backend.
            - ``handle_or_did``: Populated only for atmosphere paths.
        """
        # AT URIs go to atmosphere
        if ref.startswith("at://"):
            return ("_atmosphere", ref, None)

        # @ prefix -> atmosphere
        if ref.startswith("@"):
            rest = ref[1:]
            parts = rest.split("/", 1)
            if len(parts) == 2:
                return ("_atmosphere", parts[1], parts[0])
            return ("_atmosphere", rest, None)

        # atdata:// full URI
        if ref.startswith("atdata://"):
            path = ref[len("atdata://") :]
            parts = path.split("/")
            # atdata://mount/collection/name  or  atdata://mount/name
            repo_name = parts[0]
            dataset_name = parts[-1]
            if repo_name == "local" or repo_name in self._repos:
                return (repo_name, dataset_name, None)
            # Unknown prefix -- might be an atmosphere handle
            return ("_atmosphere", dataset_name, repo_name)

        # prefix/name where prefix is a known repository
        if "/" in ref:
            prefix, rest = ref.split("/", 1)
            if prefix == "local":
                return ("local", rest, None)
            if prefix in self._repos:
                return (prefix, rest, None)

        # Bare name -> local repository
        return ("local", ref, None)

    @property
    def repos(self) -> dict[str, "Repository"]:
        """All repositories mounted on this index (including ``"local"``)."""
        return dict(self._repos)

    @property
    def atmosphere(self) -> Any:
        """The Atmosphere for this index, or None if disabled.

        Returns the underlying client (not the internal backend wrapper).
        """
        backend = self._get_atmosphere()
        if backend is not None:
            return backend.client
        return None

    @property
    def _provider(self) -> "IndexProvider":  # noqa: F821
        """IndexProvider for the ``"local"`` repository (backward compat)."""
        return self._repos["local"].provider

    @property
    def provider(self) -> "IndexProvider":  # noqa: F821
        """The storage provider backing the ``"local"`` repository."""
        return self._repos["local"].provider

    @property
    def _redis(self) -> Redis:
        """Backwards-compatible access to the underlying Redis connection.

        Raises:
            AttributeError: If the current provider is not Redis-backed.
        """
        from atdata.providers._redis import RedisProvider

        prov = self._repos["local"].provider
        if isinstance(prov, RedisProvider):
            return prov.redis
        raise AttributeError(
            "Index._redis is only available with a Redis provider. "
            "Use index.provider instead."
        )

    @property
    def _data_store(self) -> AbstractDataStore | None:
        """Data store for the ``"local"`` repository (backward compat)."""
        return self._repos["local"].data_store

    @property
    def data_store(self) -> AbstractDataStore | None:
        """The data store for writing shards, or None if index-only."""
        return self._repos["local"].data_store

    @property
    def stub_dir(self) -> Path | None:
        """Directory where stub files are written, or None if auto-stubs disabled.

        Use this path to configure your IDE for type checking support:
        - VS Code/Pylance: Add to python.analysis.extraPaths in settings.json
        - PyCharm: Mark as Sources Root
        - mypy: Add to mypy_path in mypy.ini
        """
        if self._stub_manager is not None:
            return self._stub_manager.stub_dir
        return None

    @property
    def types(self) -> SchemaNamespace:
        """Namespace for accessing loaded schema types.

        After calling :meth:`load_schema`, schema types become available
        as attributes on this namespace.

        Examples:
            >>> index.load_schema("atdata://local/schema/MySample@1.0.0")
            >>> MyType = index.types.MySample
            >>> sample = MyType(name="hello", value=42)

        Returns:
            SchemaNamespace containing all loaded schema types.
        """
        return self._schema_namespace

    def load_schema(self, ref: str) -> Type[Packable]:
        """Load a schema and make it available in the types namespace.

        This method decodes the schema, optionally generates a Python module
        for IDE support (if auto_stubs is enabled), and registers the type
        in the :attr:`types` namespace for easy access.

        Args:
            ref: Schema reference string (atdata://local/schema/... or
                legacy local://schemas/...).

        Returns:
            The decoded PackableSample subclass. Also available via
            ``index.types.<ClassName>`` after this call.

        Raises:
            KeyError: If schema not found.
            ValueError: If schema cannot be decoded.

        Examples:
            >>> # Load and use immediately
            >>> MyType = index.load_schema("atdata://local/schema/MySample@1.0.0")
            >>> sample = MyType(field1="hello", field2=42)
            >>>
            >>> # Or access later via namespace
            >>> index.load_schema("atdata://local/schema/OtherType@1.0.0")
            >>> other = index.types.OtherType(data="test")
        """
        # Decode the schema (uses generated module if auto_stubs enabled)
        cls = self.decode_schema(ref)

        # Register in namespace using the class name
        self._schema_namespace._register(cls.__name__, cls)

        return cls

    def get_import_path(self, ref: str) -> str | None:
        """Get the import path for a schema's generated module.

        When auto_stubs is enabled, this returns the import path that can
        be used to import the schema type with full IDE support.

        Args:
            ref: Schema reference string.

        Returns:
            Import path like "local.MySample_1_0_0", or None if auto_stubs
            is disabled.

        Examples:
            >>> index = Index(auto_stubs=True)
            >>> ref = index.publish_schema(MySample, version="1.0.0")
            >>> index.load_schema(ref)
            >>> print(index.get_import_path(ref))
            local.MySample_1_0_0
            >>> # Then in your code:
            >>> # from local.MySample_1_0_0 import MySample
        """
        if self._stub_manager is None:
            return None

        from atdata._stub_manager import _extract_authority

        name, version = _parse_schema_ref(ref)
        schema_dict = self.get_schema(ref)
        authority = _extract_authority(schema_dict.get("$ref"))

        safe_version = version.replace(".", "_")
        module_name = f"{name}_{safe_version}"

        return f"{authority}.{module_name}"

    def list_entries(self) -> list[LocalDatasetEntry]:
        """Get all index entries as a materialized list.

        Returns:
            List of all LocalDatasetEntry objects in the index.
        """
        return list(self.entries)

    # Legacy alias for backwards compatibility
    @property
    def all_entries(self) -> list[LocalDatasetEntry]:
        """Get all index entries as a list (deprecated, use list_entries())."""
        return self.list_entries()

    @property
    def entries(self) -> Generator[LocalDatasetEntry, None, None]:
        """Iterate over all index entries.

        Yields:
            LocalDatasetEntry objects from the index.
        """
        yield from self._provider.iter_entries()

    def add_entry(
        self,
        ds: Dataset,
        *,
        name: str,
        schema_ref: str | None = None,
        metadata: dict | None = None,
    ) -> LocalDatasetEntry:
        """Add a dataset to the local repository index.

        .. deprecated::
            Use :meth:`insert_dataset` instead.

        Args:
            ds: The dataset to add to the index.
            name: Human-readable name for the dataset.
            schema_ref: Optional schema reference. If None, generates from sample type.
            metadata: Optional metadata dictionary. If None, uses ds._metadata if available.

        Returns:
            The created LocalDatasetEntry object.
        """
        import warnings

        warnings.warn(
            "Index.add_entry() is deprecated, use Index.insert_dataset()",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._insert_dataset_to_provider(
            ds,
            name=name,
            schema_ref=schema_ref,
            provider=self._provider,
            store=None,
            metadata=metadata,
        )

    def get_entry(self, cid: str) -> LocalDatasetEntry:
        """Get an entry by its CID.

        Args:
            cid: Content identifier of the entry.

        Returns:
            LocalDatasetEntry for the given CID.

        Raises:
            KeyError: If entry not found.
        """
        return self._provider.get_entry_by_cid(cid)

    def get_entry_by_name(self, name: str) -> LocalDatasetEntry:
        """Get an entry by its human-readable name.

        Args:
            name: Human-readable name of the entry.

        Returns:
            LocalDatasetEntry with the given name.

        Raises:
            KeyError: If no entry with that name exists.
        """
        return self._provider.get_entry_by_name(name)

    # AbstractIndex protocol methods

    @staticmethod
    def _ensure_schema_stored(
        schema_ref: str,
        sample_type: type,
        provider: "IndexProvider",  # noqa: F821
    ) -> None:
        """Persist the schema definition if not already stored.

        Called during dataset insertion so that ``decode_schema()`` can
        reconstruct the type later without the caller needing to publish
        the schema separately.
        """
        schema_name, version = _parse_schema_ref(schema_ref)
        if provider.get_schema_json(schema_name, version) is None:
            record = _build_schema_record(sample_type, version=version)
            provider.store_schema(schema_name, version, json.dumps(record))

    def _insert_dataset_to_provider(
        self,
        ds: Dataset,
        *,
        name: str,
        schema_ref: str | None = None,
        provider: "IndexProvider",  # noqa: F821
        store: AbstractDataStore | None = None,
        **kwargs,
    ) -> LocalDatasetEntry:
        """Insert a dataset into a specific provider/store pair.

        This is the internal implementation shared by all local and named
        repository inserts.
        """
        from atdata._logging import get_logger

        log = get_logger()
        metadata = kwargs.get("metadata")

        if store is not None:
            prefix = kwargs.get("prefix", name)
            cache_local = kwargs.get("cache_local", False)
            log.debug(
                "_insert_dataset_to_provider: name=%s, store=%s",
                name,
                type(store).__name__,
            )

            written_urls = store.write_shards(
                ds,
                prefix=prefix,
                cache_local=cache_local,
            )
            log.info(
                "_insert_dataset_to_provider: %d shard(s) written for %s",
                len(written_urls),
                name,
            )

            if schema_ref is None:
                schema_ref = _schema_ref_from_type(ds.sample_type, version="1.0.0")

            self._ensure_schema_stored(schema_ref, ds.sample_type, provider)

            entry_metadata = metadata if metadata is not None else ds._metadata
            entry_metadata = _merge_checksums(entry_metadata, written_urls)
            entry = LocalDatasetEntry(
                name=name,
                schema_ref=schema_ref,
                data_urls=written_urls,
                metadata=entry_metadata,
            )
        else:
            # No data store - just index the existing URL
            if schema_ref is None:
                schema_ref = _schema_ref_from_type(ds.sample_type, version="1.0.0")

            self._ensure_schema_stored(schema_ref, ds.sample_type, provider)

            data_urls = [ds.url]
            entry_metadata = metadata if metadata is not None else ds._metadata

            entry = LocalDatasetEntry(
                name=name,
                schema_ref=schema_ref,
                data_urls=data_urls,
                metadata=entry_metadata,
            )

        provider.store_entry(entry)
        provider.store_label(
            name=name,
            cid=entry.cid,
            version=kwargs.get("version"),
            description=kwargs.get("description"),
        )
        log.debug("_insert_dataset_to_provider: entry stored for %s", name)
        return entry

    def insert_dataset(
        self,
        ds: Dataset,
        *,
        name: str,
        schema_ref: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        license: str | None = None,
        data_store: AbstractDataStore | None = None,
        force: bool = False,
        copy: bool = False,
        metadata: dict | None = None,
        _data_urls: list[str] | None = None,
        _blob_refs: list[dict] | None = None,
        **kwargs,
    ) -> "IndexEntry":
        """Insert a dataset into the index.

        The target repository is determined by a prefix in the ``name``
        argument (e.g. ``"lab/mnist"``). If no prefix is given, or the
        prefix is ``"local"``, the built-in local repository is used.

        For atmosphere targets:

        - **Local sources** are uploaded via *data_store* (defaults to
          ``PDSBlobStore``).
        - **Public remote sources** (http/https) are referenced as
          external URLs unless *copy* is ``True``.
        - **Credentialed sources** (e.g. ``S3Source``) raise an error
          unless *copy* is ``True`` or *data_store* is provided, to
          prevent leaking private endpoints.

        Args:
            ds: The Dataset to register.
            name: Human-readable name for the dataset, optionally prefixed
                with a repository name (e.g. ``"lab/mnist"``).
            schema_ref: Optional schema reference.
            description: Optional dataset description (atmosphere only).
            tags: Optional tags for discovery (atmosphere only).
            license: Optional license identifier (atmosphere only).
            data_store: Explicit data store for shard storage. When
                provided, data is always copied through this store.
            force: If True, bypass PDS size limits (50 MB per shard,
                1 GB total). Default: ``False``.
            copy: If True, copy data to the destination store even for
                remote sources. Required for credentialed sources
                targeting the atmosphere. Default: ``False``.
            metadata: Optional metadata dict.

        Returns:
            IndexEntry for the inserted dataset.

        Raises:
            ValueError: If atmosphere limits are exceeded (when
                *force* is ``False``), or if a credentialed source
                targets the atmosphere without *copy*.
        """
        from atdata.atmosphere.store import PDS_TOTAL_DATASET_LIMIT_BYTES

        backend_key, resolved_name, handle_or_did = self._resolve_prefix(name)
        is_atmosphere = backend_key == "_atmosphere"

        if is_atmosphere:
            atmo = self._get_atmosphere()
            if atmo is None:
                raise ValueError(
                    f"Atmosphere backend required for name {name!r} but not available."
                )

            # Providing an explicit data_store implies copy behaviour
            needs_copy = copy or data_store is not None

            # Credentialed source guard
            if _is_credentialed_source(ds) and not needs_copy:
                raise ValueError(
                    "Dataset uses a credentialed source. Referencing "
                    "these URLs in a public atmosphere record would "
                    "leak private endpoints. Pass copy=True to copy "
                    "data to the destination store (default: PDS blobs)."
                )

            # If we already have pre-written URLs (from write_samples),
            # go straight to publish.
            if _data_urls is not None:
                return atmo.insert_dataset(
                    ds,
                    name=resolved_name,
                    schema_ref=schema_ref,
                    data_urls=_data_urls,
                    blob_refs=_blob_refs,
                    description=description,
                    tags=tags,
                    license=license,
                    metadata=metadata,
                    **kwargs,
                )

            # Determine whether data must be copied
            source_is_local = _is_local_path(ds.url)

            if source_is_local or needs_copy:
                # Resolve effective store
                if data_store is not None:
                    effective_store = data_store
                else:
                    from atdata.atmosphere.store import PDSBlobStore

                    effective_store = PDSBlobStore(atmo.client)

                # Size guard
                if not force:
                    total_bytes = _estimate_dataset_bytes(ds)
                    if total_bytes > PDS_TOTAL_DATASET_LIMIT_BYTES:
                        raise ValueError(
                            f"Total dataset size ({total_bytes} bytes) "
                            f"exceeds atmosphere limit "
                            f"({PDS_TOTAL_DATASET_LIMIT_BYTES} bytes). "
                            f"Pass force=True to bypass."
                        )

                result = effective_store.write_shards(ds, prefix=resolved_name)

                # ShardUploadResult carries blob_refs; plain list does not
                blob_refs = getattr(result, "blob_refs", None) or None

                return atmo.insert_dataset(
                    ds,
                    name=resolved_name,
                    schema_ref=schema_ref,
                    data_urls=list(result),
                    blob_refs=blob_refs,
                    description=description,
                    tags=tags,
                    license=license,
                    metadata=_merge_checksums(metadata, result),
                    **kwargs,
                )

            # Public remote source — reference existing URLs
            data_urls = ds.list_shards()
            return atmo.insert_dataset(
                ds,
                name=resolved_name,
                schema_ref=schema_ref,
                data_urls=data_urls,
                description=description,
                tags=tags,
                license=license,
                metadata=metadata,
                **kwargs,
            )

        # --- Local / named repo path ---
        repo = self._repos.get(backend_key)
        if repo is None:
            raise KeyError(f"Unknown repository {backend_key!r} in name {name!r}")

        effective_store = data_store or repo.data_store
        return self._insert_dataset_to_provider(
            ds,
            name=resolved_name,
            schema_ref=schema_ref,
            provider=repo.provider,
            store=effective_store,
            metadata=metadata,
            **kwargs,
        )

    def write_samples(
        self,
        samples: Iterable,
        *,
        name: str,
        schema_ref: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        license: str | None = None,
        maxcount: int = 10_000,
        maxsize: int | None = None,
        metadata: dict | None = None,
        manifest: bool = False,
        data_store: AbstractDataStore | None = None,
        force: bool = False,
    ) -> "IndexEntry":
        """Write samples and create an index entry in one step.

        This is the primary method for publishing data. It serializes
        samples to WebDataset tar files, stores them via the appropriate
        backend, and creates an index entry.

        The target backend is determined by the *name* prefix:

        - Bare name (e.g., ``"mnist"``): writes to the local repository.
        - ``"@handle/name"``: writes and publishes to the atmosphere.
        - ``"repo/name"``: writes to a named repository.

        For atmosphere targets, data is uploaded as PDS blobs by default.
        Shard size is capped at 50 MB and total dataset size at 1 GB
        unless *force* is ``True``.

        When the local backend has no ``data_store`` configured, a
        ``LocalDiskStore`` is created automatically at
        ``~/.atdata/data/`` so that samples have persistent storage.

        Args:
            samples: Iterable of ``Packable`` samples. Must be non-empty.
            name: Dataset name, optionally prefixed with target.
            schema_ref: Optional schema reference. Auto-generated if ``None``.
            description: Optional dataset description (atmosphere only).
            tags: Optional tags for discovery (atmosphere only).
            license: Optional license identifier (atmosphere only).
            maxcount: Max samples per shard. Default: 10,000.
            maxsize: Max bytes per shard. For atmosphere targets defaults
                to 50 MB (PDS blob limit). For local targets defaults to
                ``None`` (unlimited).
            metadata: Optional metadata dict stored with the entry.
            manifest: If True, write per-shard manifest sidecar files
                alongside each tar. Default: ``False``.
            data_store: Explicit data store for shard storage. Overrides
                the repository's default store. For atmosphere targets
                defaults to ``PDSBlobStore``.
            force: If True, bypass PDS size limits (50 MB per shard,
                1 GB total dataset). Default: ``False``.

        Returns:
            IndexEntry for the created dataset.

        Raises:
            ValueError: If *samples* is empty, or if atmosphere size
                limits are exceeded (when *force* is ``False``).

        Examples:
            >>> index = Index()
            >>> samples = [MySample(key="0", text="hello")]
            >>> entry = index.write_samples(samples, name="my-dataset")
        """
        import tempfile

        from atdata.dataset import write_samples as _write_samples
        from atdata.atmosphere.store import (
            PDS_BLOB_LIMIT_BYTES,
            PDS_TOTAL_DATASET_LIMIT_BYTES,
        )
        from atdata._logging import log_operation

        backend_key, resolved_name, _ = self._resolve_prefix(name)
        is_atmosphere = backend_key == "_atmosphere"

        with log_operation("Index.write_samples", name=name):
            # --- Atmosphere size guards ---
            if is_atmosphere and not force:
                if maxsize is not None and maxsize > PDS_BLOB_LIMIT_BYTES:
                    raise ValueError(
                        f"maxsize={maxsize} exceeds PDS blob limit "
                        f"({PDS_BLOB_LIMIT_BYTES} bytes). "
                        f"Pass force=True to bypass."
                    )

            # Default maxsize for atmosphere targets
            effective_maxsize = maxsize
            if is_atmosphere and effective_maxsize is None:
                effective_maxsize = PDS_BLOB_LIMIT_BYTES

            # Resolve the effective data store
            if is_atmosphere:
                atmo = self._get_atmosphere()
                if atmo is None:
                    raise ValueError(
                        f"Atmosphere backend required for name {name!r} but not available."
                    )
                if data_store is None:
                    from atdata.atmosphere.store import PDSBlobStore

                    effective_store: AbstractDataStore | None = PDSBlobStore(
                        atmo.client
                    )
                else:
                    effective_store = data_store
            else:
                repo = self._repos.get(backend_key)
                effective_store = data_store or (
                    repo.data_store if repo is not None else None
                )
                needs_auto_store = repo is not None and effective_store is None
                if needs_auto_store:
                    from atdata.stores._disk import LocalDiskStore

                    effective_store = LocalDiskStore()

            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir) / "data.tar"
                ds = _write_samples(
                    samples,
                    tmp_path,
                    maxcount=maxcount,
                    maxsize=effective_maxsize,
                    manifest=manifest,
                )

                # Atmosphere total-size guard (after writing so we can measure)
                if is_atmosphere and not force:
                    total_bytes = _estimate_dataset_bytes(ds)
                    if total_bytes > PDS_TOTAL_DATASET_LIMIT_BYTES:
                        raise ValueError(
                            f"Total dataset size ({total_bytes} bytes) exceeds "
                            f"atmosphere limit ({PDS_TOTAL_DATASET_LIMIT_BYTES} "
                            f"bytes). Pass force=True to bypass."
                        )

                if is_atmosphere:
                    # Write shards through the store, then publish record
                    # with the resulting URLs (not the temp paths).
                    written_urls = effective_store.write_shards(
                        ds, prefix=resolved_name
                    )

                    # If write_shards returned blob refs (e.g. ShardUploadResult),
                    # use storageBlobs so the PDS retains the uploaded blobs.
                    # Fall back to storageExternal with AT URIs otherwise.
                    blob_refs = getattr(written_urls, "blob_refs", None) or None

                    return self.insert_dataset(
                        ds,
                        name=name,
                        schema_ref=schema_ref,
                        metadata=_merge_checksums(metadata, written_urls),
                        description=description,
                        tags=tags,
                        license=license,
                        data_store=data_store,
                        force=force,
                        _data_urls=written_urls,
                        _blob_refs=blob_refs,
                    )

                # Local / named repo path
                repo = self._repos.get(backend_key)
                if repo is not None and effective_store is not None:
                    return self._insert_dataset_to_provider(
                        ds,
                        name=resolved_name,
                        schema_ref=schema_ref,
                        provider=repo.provider,
                        store=effective_store,
                        metadata=metadata,
                    )

                return self.insert_dataset(
                    ds,
                    name=name,
                    schema_ref=schema_ref,
                    metadata=metadata,
                    description=description,
                    tags=tags,
                    license=license,
                )

    def write(
        self,
        samples: Iterable,
        *,
        name: str,
        **kwargs: Any,
    ) -> "IndexEntry":
        """Write samples and create an index entry.

        .. deprecated::
            Use :meth:`write_samples` instead.
        """
        import warnings

        warnings.warn(
            "Index.write() is deprecated, use Index.write_samples()",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.write_samples(samples, name=name, **kwargs)

    def get_dataset(self, ref: str) -> "IndexEntry":
        """Get a dataset entry by name or prefixed reference.

        Supports repository-prefixed lookups (e.g. ``"lab/mnist"``),
        atmosphere paths (``"@handle/dataset"``), AT URIs, and bare names
        (which default to the ``"local"`` repository).

        Args:
            ref: Dataset name, prefixed name, or AT URI.

        Returns:
            IndexEntry for the dataset.

        Raises:
            KeyError: If dataset not found.
            ValueError: If the atmosphere backend is required but unavailable.
        """
        backend_key, resolved_ref, handle_or_did = self._resolve_prefix(ref)

        if backend_key == "_atmosphere":
            atmo = self._get_atmosphere()
            if atmo is None:
                raise ValueError(
                    f"Atmosphere backend required for path {ref!r} but not available. "
                    "Install 'atproto' or pass an Atmosphere."
                )
            return atmo.get_dataset(resolved_ref)

        repo = self._repos.get(backend_key)
        if repo is None:
            raise KeyError(f"Unknown repository {backend_key!r} in ref {ref!r}")

        # Resolve through label first, fall back to direct name lookup
        try:
            cid, _version = repo.provider.get_label(resolved_ref)
            return repo.provider.get_entry_by_cid(cid)
        except KeyError:
            return repo.provider.get_entry_by_name(resolved_ref)

    @property
    def datasets(self) -> Generator["IndexEntry", None, None]:
        """Lazily iterate over all dataset entries across local repositories.

        Yields entries from all mounted repositories (``"local"`` and named).
        Atmosphere entries are not included (use
        ``list_datasets(repo="_atmosphere")`` for those).

        Yields:
            IndexEntry for each dataset.
        """
        for repo in self._repos.values():
            yield from repo.provider.iter_entries()

    def list_datasets(self, repo: str | None = None) -> list["IndexEntry"]:
        """Get dataset entries as a materialized list (AbstractIndex protocol).

        Args:
            repo: Optional repository filter. If ``None``, aggregates entries
                from ``"local"`` and all named repositories. Use ``"local"``
                for only the built-in repository, a named repo key, or
                ``"_atmosphere"`` for atmosphere entries.

        Returns:
            List of IndexEntry for each dataset.
        """
        if repo is None:
            return list(self.datasets)

        if repo == "_atmosphere":
            atmo = self._get_atmosphere()
            if atmo is None:
                return []
            return atmo.list_datasets()

        named = self._repos.get(repo)
        if named is None:
            raise KeyError(f"Unknown repository {repo!r}")
        return list(named.provider.iter_entries())

    # Label operations

    def label(
        self,
        name: str,
        cid: str,
        *,
        version: str | None = None,
        description: str | None = None,
    ) -> None:
        """Create or update a label mapping a name to a dataset CID.

        Args:
            name: Human-readable label name, optionally prefixed with a
                repository name (e.g. ``"lab/mnist"``).
            cid: Content identifier of the target dataset entry.
            version: Optional version string (e.g. ``"1.0.0"``).
            description: Optional description.

        Examples:
            >>> index.label("mnist", entry.cid, version="2.0.0")
        """
        backend_key, resolved_name, _ = self._resolve_prefix(name)
        repo = self._repos.get(backend_key)
        if repo is None:
            raise KeyError(f"Unknown repository {backend_key!r} in name {name!r}")
        repo.provider.store_label(
            name=resolved_name,
            cid=cid,
            version=version,
            description=description,
        )

    def get_label(self, name: str, version: str | None = None) -> "IndexEntry":
        """Resolve a label to its dataset entry.

        Args:
            name: Label name, optionally prefixed with a repository name.
            version: Specific version to resolve. If ``None``, returns the
                most recently created label.

        Returns:
            IndexEntry for the labeled dataset.

        Raises:
            KeyError: If no label or dataset found.

        Examples:
            >>> entry = index.get_label("mnist", version="1.0.0")
        """
        backend_key, resolved_name, _ = self._resolve_prefix(name)
        repo = self._repos.get(backend_key)
        if repo is None:
            raise KeyError(f"Unknown repository {backend_key!r} in name {name!r}")
        cid, _resolved_version = repo.provider.get_label(resolved_name, version)
        return repo.provider.get_entry_by_cid(cid)

    def list_labels(self, repo: str | None = None) -> list[tuple[str, str, str | None]]:
        """List all labels as ``(name, cid, version)`` tuples.

        Args:
            repo: Optional repository filter. If ``None``, aggregates
                from all local repositories.

        Returns:
            List of ``(name, cid, version)`` tuples.

        Examples:
            >>> for name, cid, version in index.list_labels():
            ...     print(f"{name}@{version} -> {cid}")
        """
        if repo is not None:
            named = self._repos.get(repo)
            if named is None:
                raise KeyError(f"Unknown repository {repo!r}")
            return list(named.provider.iter_labels())

        result: list[tuple[str, str, str | None]] = []
        for r in self._repos.values():
            result.extend(r.provider.iter_labels())
        return result

    # Schema operations

    def _get_latest_schema_version(self, name: str) -> str | None:
        """Get the latest version for a schema by name, or None if not found."""
        return self._provider.find_latest_version(name)

    def publish_schema(
        self,
        sample_type: type,
        *,
        version: str | None = None,
        description: str | None = None,
    ) -> str:
        """Publish a schema for a sample type to Redis.

        Args:
            sample_type: A Packable type (@packable-decorated or PackableSample subclass).
            version: Semantic version string (e.g., '1.0.0'). If None,
                auto-increments from the latest published version (patch bump),
                or starts at '1.0.0' if no previous version exists.
            description: Optional human-readable description. If None, uses
                the class docstring.

        Returns:
            Schema reference string: 'atdata://local/schema/{name}@{version}'.

        Raises:
            ValueError: If sample_type is not a dataclass.
            TypeError: If sample_type doesn't satisfy the Packable protocol,
                or if a field type is not supported.
        """
        # Validate that sample_type satisfies Packable protocol at runtime
        # This catches non-packable types early with a clear error message
        try:
            # Check protocol compliance by verifying required methods exist
            if not (
                hasattr(sample_type, "from_data")
                and hasattr(sample_type, "from_bytes")
                and callable(getattr(sample_type, "from_data", None))
                and callable(getattr(sample_type, "from_bytes", None))
            ):
                raise TypeError(
                    f"{sample_type.__name__} does not satisfy the Packable protocol. "
                    "Use @packable decorator or inherit from PackableSample."
                )
        except AttributeError:
            raise TypeError(
                f"sample_type must be a class, got {type(sample_type).__name__}"
            )

        # Auto-increment version if not specified
        if version is None:
            latest = self._get_latest_schema_version(sample_type.__name__)
            if latest is None:
                version = "1.0.0"
            else:
                version = _increment_patch(latest)

        schema_record = _build_schema_record(
            sample_type,
            version=version,
            description=description,
        )

        schema_ref = _schema_ref_from_type(sample_type, version)
        name, _ = _parse_schema_ref(schema_ref)

        # Store via provider
        schema_json = json.dumps(schema_record)
        self._provider.store_schema(name, version, schema_json)

        return schema_ref

    def get_schema(self, ref: str) -> dict:
        """Get a schema record by reference (AbstractIndex protocol).

        Args:
            ref: Schema reference string. Supports both new format
                (atdata://local/schema/{name}@{version}) and legacy
                format (local://schemas/{module.Class}@{version}).

        Returns:
            Schema record as a dictionary with keys 'name', 'version',
            'fields', '$ref', etc.

        Raises:
            KeyError: If schema not found.
            ValueError: If reference format is invalid.
        """
        name, version = _parse_schema_ref(ref)

        schema_json = self._provider.get_schema_json(name, version)
        if schema_json is None:
            raise KeyError(f"Schema not found: {ref}")

        schema = json.loads(schema_json)
        schema["$ref"] = _make_schema_ref(name, version)

        # Auto-generate stub if enabled
        if self._stub_manager is not None:
            self._stub_manager.ensure_stub(schema)

        return schema

    def get_schema_record(self, ref: str) -> LocalSchemaRecord:
        """Get a schema record as LocalSchemaRecord object.

        Use this when you need the full LocalSchemaRecord with typed properties.
        For Protocol-compliant dict access, use get_schema() instead.

        Args:
            ref: Schema reference string.

        Returns:
            LocalSchemaRecord with schema details.

        Raises:
            KeyError: If schema not found.
            ValueError: If reference format is invalid.
        """
        schema = self.get_schema(ref)
        return LocalSchemaRecord.from_dict(schema)

    @property
    def schemas(self) -> Generator[LocalSchemaRecord, None, None]:
        """Iterate over all schema records in this index.

        Yields:
            LocalSchemaRecord for each schema.
        """
        for name, version, schema_json in self._provider.iter_schemas():
            schema = json.loads(schema_json)
            schema["$ref"] = _make_schema_ref(name, version)
            yield LocalSchemaRecord.from_dict(schema)

    def list_schemas(self) -> list[dict]:
        """Get all schema records as a materialized list (AbstractIndex protocol).

        Returns:
            List of schema records as dictionaries.
        """
        return [record.to_dict() for record in self.schemas]

    def decode_schema(self, ref: str) -> Type[Packable]:
        """Reconstruct a Python PackableSample type from a stored schema.

        This method enables loading datasets without knowing the sample type
        ahead of time. The index retrieves the schema record and dynamically
        generates a PackableSample subclass matching the schema definition.

        If auto_stubs is enabled, a Python module will be generated and the
        class will be imported from it, providing full IDE autocomplete support.
        The returned class has proper type information that IDEs can understand.

        Args:
            ref: Schema reference string (atdata://local/schema/... or
                legacy local://schemas/...).

        Returns:
            A PackableSample subclass - either imported from a generated module
            (if auto_stubs is enabled) or dynamically created.

        Raises:
            KeyError: If schema not found.
            ValueError: If schema cannot be decoded.
        """
        schema_dict = self.get_schema(ref)

        # If auto_stubs is enabled, generate module and import class from it
        if self._stub_manager is not None:
            cls = self._stub_manager.ensure_module(schema_dict)
            if cls is not None:
                return cls

        # Fall back to dynamic type generation
        from atdata._schema_codec import schema_to_type

        return schema_to_type(schema_dict)

    def decode_schema_as(self, ref: str, type_hint: type[T]) -> type[T]:
        """Decode a schema with explicit type hint for IDE support.

        This is a typed wrapper around decode_schema() that preserves the
        type information for IDE autocomplete. Use this when you have a
        stub file for the schema and want full IDE support.

        Args:
            ref: Schema reference string.
            type_hint: The stub type to use for type hints. Import this from
                the generated stub file.

        Returns:
            The decoded type, cast to match the type_hint for IDE support.

        Examples:
            >>> # After enabling auto_stubs and configuring IDE extraPaths:
            >>> from local.MySample_1_0_0 import MySample
            >>>
            >>> # This gives full IDE autocomplete:
            >>> DecodedType = index.decode_schema_as(ref, MySample)
            >>> sample = DecodedType(text="hello", value=42)  # IDE knows signature!

        Note:
            The type_hint is only used for static type checking - at runtime,
            the actual decoded type from the schema is returned. Ensure the
            stub matches the schema to avoid runtime surprises.
        """
        from typing import cast

        return cast(type[T], self.decode_schema(ref))

    def clear_stubs(self) -> int:
        """Remove all auto-generated stub files.

        Only works if auto_stubs was enabled when creating the Index.

        Returns:
            Number of stub files removed, or 0 if auto_stubs is disabled.
        """
        if self._stub_manager is not None:
            return self._stub_manager.clear_stubs()
        return 0

    # -- Atmosphere promotion --

    def promote_entry(
        self,
        entry_name: str,
        *,
        name: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        license: str | None = None,
    ) -> str:
        """Promote a locally-indexed dataset to the atmosphere.

        .. deprecated::
            Use :meth:`insert_dataset` instead.

        Args:
            entry_name: Name of the local dataset entry to promote.
            name: Override name for the atmosphere record. Defaults to
                the local entry name.
            description: Optional description for the dataset.
            tags: Optional tags for discovery.
            license: Optional license identifier.

        Returns:
            AT URI of the created atmosphere dataset record.

        Raises:
            ValueError: If atmosphere backend is not available, or
                the local entry has no data URLs.
            KeyError: If the entry or its schema is not found.

        Examples:
            >>> index = Index(atmosphere=client)
            >>> uri = index.promote_entry("mnist-train")
        """
        import warnings

        warnings.warn(
            "Index.promote_entry() is deprecated, use Index.insert_dataset()",
            DeprecationWarning,
            stacklevel=2,
        )
        from atdata.promote import _find_or_publish_schema
        from atdata.atmosphere import DatasetPublisher
        from atdata._schema_codec import schema_to_type
        from atdata._logging import log_operation

        atmo = self._get_atmosphere()
        if atmo is None:
            raise ValueError("Atmosphere backend required but not available.")

        with log_operation("Index.promote_entry", entry_name=entry_name):
            entry = self.get_entry_by_name(entry_name)
            if not entry.data_urls:
                raise ValueError(f"Local entry {entry_name!r} has no data URLs")

            schema_record = self.get_schema(entry.schema_ref)
            sample_type = schema_to_type(schema_record)
            schema_version = schema_record.get("version", "1.0.0")

            atmosphere_schema_uri = _find_or_publish_schema(
                sample_type,
                schema_version,
                atmo.client,
                description=schema_record.get("description"),
            )

            publisher = DatasetPublisher(atmo.client)
            uri = publisher.publish_with_urls(
                urls=entry.data_urls,
                schema_uri=atmosphere_schema_uri,
                name=name or entry.name,
                description=description,
                tags=tags,
                license=license,
                metadata=entry.metadata,
            )

            # Create a label pointing to the dataset record
            from atdata.atmosphere import LabelPublisher

            label_publisher = LabelPublisher(atmo.client)
            label_publisher.publish(
                name=name or entry.name,
                dataset_uri=str(uri),
                version=schema_version,
                description=description,
            )

            return str(uri)

    def promote_dataset(
        self,
        dataset: Dataset,
        *,
        name: str,
        sample_type: type | None = None,
        schema_version: str = "1.0.0",
        description: str | None = None,
        tags: list[str] | None = None,
        license: str | None = None,
    ) -> str:
        """Publish a Dataset directly to the atmosphere.

        .. deprecated::
            Use :meth:`insert_dataset` instead.

        Args:
            dataset: The Dataset to publish.
            name: Name for the atmosphere dataset record.
            sample_type: Sample type for schema publishing. Inferred from
                ``dataset.sample_type`` if not provided.
            schema_version: Semantic version for the schema. Default: ``"1.0.0"``.
            description: Optional description for the dataset.
            tags: Optional tags for discovery.
            license: Optional license identifier.

        Returns:
            AT URI of the created atmosphere dataset record.

        Raises:
            ValueError: If atmosphere backend is not available.

        Examples:
            >>> index = Index(atmosphere=client)
            >>> ds = atdata.load_dataset("./data.tar", MySample, split="train")
            >>> uri = index.promote_dataset(ds, name="my-dataset")
        """
        import warnings

        warnings.warn(
            "Index.promote_dataset() is deprecated, use Index.insert_dataset()",
            DeprecationWarning,
            stacklevel=2,
        )
        from atdata.promote import _find_or_publish_schema
        from atdata.atmosphere import DatasetPublisher
        from atdata._logging import log_operation

        atmo = self._get_atmosphere()
        if atmo is None:
            raise ValueError("Atmosphere backend required but not available.")

        with log_operation("Index.promote_dataset", name=name):
            st = sample_type or dataset.sample_type

            atmosphere_schema_uri = _find_or_publish_schema(
                st,
                schema_version,
                atmo.client,
                description=description,
            )

            data_urls = dataset.list_shards()

            publisher = DatasetPublisher(atmo.client)
            uri = publisher.publish_with_urls(
                urls=data_urls,
                schema_uri=atmosphere_schema_uri,
                name=name,
                description=description,
                tags=tags,
                license=license,
                metadata=dataset._metadata,
            )

            # Create a label pointing to the dataset record
            from atdata.atmosphere import LabelPublisher

            label_publisher = LabelPublisher(atmo.client)
            label_publisher.publish(
                name=name,
                dataset_uri=str(uri),
                version=schema_version,
                description=description,
            )

            return str(uri)

    # ------------------------------------------------------------------
    # Lens operations
    # ------------------------------------------------------------------

    def store_lens(
        self,
        lens_obj: "Lens",
        *,
        name: str,
        version: str | None = None,
        description: str | None = None,
        source_schema: str | None = None,
        view_schema: str | None = None,
    ) -> str:
        """Persist a lens transformation in the local provider.

        Serializes the lens to JSON and stores it. If the lens uses simple
        field mappings, those are captured declaratively for later
        reconstitution. Otherwise, code references are stored.

        Args:
            lens_obj: The Lens to store.
            name: Human-readable lens name.
            version: Semantic version string. If ``None``, auto-increments
                from the latest version or starts at ``"1.0.0"``.
            description: Optional description.
            source_schema: Source schema name. Auto-detected if ``None``.
            view_schema: View schema name. Auto-detected if ``None``.

        Returns:
            Lens reference string: ``"atdata://local/lens/{name}@{version}"``.

        Examples:
            >>> @lens
            ... def my_lens(s: Source) -> View:
            ...     return View(name=s.name)
            >>> ref = index.store_lens(my_lens, name="my_lens")
        """
        from atdata._lens_codec import lens_to_json

        if version is None:
            latest = self._provider.find_latest_lens_version(name)
            if latest is None:
                version = "1.0.0"
            else:
                version = _increment_patch(latest)

        lens_json = lens_to_json(
            lens_obj,
            name=name,
            version=version,
            description=description,
            source_schema=source_schema,
            view_schema=view_schema,
        )

        self._provider.store_lens(name, version, lens_json)

        # Auto-generate lens stub if enabled
        if self._stub_manager is not None:
            record = json.loads(lens_json)
            self._stub_manager.ensure_lens_stub(record)

        return f"atdata://local/lens/{name}@{version}"

    def get_lens(self, ref: str) -> dict:
        """Get a lens record by reference.

        Args:
            ref: Lens reference string
                (``"atdata://local/lens/{name}@{version}"``).

        Returns:
            Lens record as a dictionary.

        Raises:
            KeyError: If lens not found.
            ValueError: If reference format is invalid.

        Examples:
            >>> record = index.get_lens("atdata://local/lens/my_lens@1.0.0")
        """
        name, version = _parse_lens_ref(ref)

        lens_json = self._provider.get_lens_json(name, version)
        if lens_json is None:
            raise KeyError(f"Lens not found: {ref}")

        record = json.loads(lens_json)
        record["$ref"] = f"atdata://local/lens/{name}@{version}"
        return record

    def load_lens(
        self,
        ref: str,
        *,
        source_type: "Type[Packable] | None" = None,
        view_type: "Type[Packable] | None" = None,
    ) -> "Lens":
        """Reconstitute a Lens object from a stored record.

        Loads the lens definition from the provider and creates a working
        ``Lens`` object. For field-mapping lenses, ``source_type`` and
        ``view_type`` must be provided. For code-reference lenses, the
        types are inferred from the imported functions.

        The reconstituted lens is automatically registered in the global
        ``LensNetwork``.

        Args:
            ref: Lens reference string.
            source_type: The source Packable type. Required for
                field-mapping lens reconstitution.
            view_type: The view Packable type. Required for
                field-mapping lens reconstitution.

        Returns:
            A reconstituted Lens object.

        Raises:
            KeyError: If lens not found.
            ValueError: If lens cannot be reconstituted.

        Examples:
            >>> lens_obj = index.load_lens(
            ...     "atdata://local/lens/my_lens@1.0.0",
            ...     source_type=Source,
            ...     view_type=View,
            ... )
        """
        from atdata._lens_codec import lens_from_record

        record = self.get_lens(ref)
        return lens_from_record(
            record,
            source_type=source_type,
            view_type=view_type,
        )

    @property
    def lenses(self) -> Generator[dict, None, None]:
        """Lazily iterate over all lens records in this index.

        Yields:
            Lens record dicts.
        """
        for name, version, lens_json in self._provider.iter_lenses():
            record = json.loads(lens_json)
            record["$ref"] = f"atdata://local/lens/{name}@{version}"
            yield record

    def list_lenses(self) -> list[dict]:
        """Get all lens records as a materialized list.

        Returns:
            List of lens record dicts.

        Examples:
            >>> for lens_rec in index.list_lenses():
            ...     print(lens_rec["name"], lens_rec["version"])
        """
        return list(self.lenses)

    def find_lenses(
        self,
        source_schema: str,
        view_schema: str | None = None,
    ) -> list[dict]:
        """Find lenses matching source and/or view schema names.

        Args:
            source_schema: Source schema name to match.
            view_schema: Optional view schema name. If ``None``, returns
                all lenses with the given source schema.

        Returns:
            List of matching lens record dicts.

        Examples:
            >>> lenses = index.find_lenses("ImageSample", "GrayscaleSample")
        """
        results = self._provider.find_lenses_by_schemas(source_schema, view_schema)
        records = []
        for name, version, lens_json in results:
            record = json.loads(lens_json)
            record["$ref"] = f"atdata://local/lens/{name}@{version}"
            records.append(record)
        return records
