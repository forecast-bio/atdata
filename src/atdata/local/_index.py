"""Index class for local dataset management."""

from __future__ import annotations

from atdata import (
    Dataset,
)
from atdata._protocols import AbstractDataStore, Packable

from atdata.local._entry import LocalDatasetEntry
from atdata.local._schema import (
    SchemaNamespace,
    LocalSchemaRecord,
    _schema_ref_from_type,
    _make_schema_ref,
    _parse_schema_ref,
    _increment_patch,
    _build_schema_record,
)

from pathlib import Path
from typing import (
    Any,
    Type,
    TypeVar,
    Generator,
    TYPE_CHECKING,
)
from redis import Redis
import json

if TYPE_CHECKING:
    from atdata.providers._base import IndexProvider
    from atdata.repository import Repository, _AtmosphereBackend
    from atdata._protocols import IndexEntry

T = TypeVar("T", bound=Packable)


class Index:
    """Unified index for tracking datasets across multiple repositories.

    Implements the AbstractIndex protocol. Maintains a registry of
    dataset entries across a built-in ``"local"`` repository, optional
    named repositories, and an optional atmosphere (ATProto) backend.

    The ``"local"`` repository is always present and uses the storage backend
    determined by the ``provider`` argument. When no provider is given, defaults
    to SQLite (zero external dependencies). Pass a ``redis`` connection or
    Redis ``**kwargs`` for backwards-compatible Redis behaviour.

    Additional named repositories can be mounted via the ``repos`` parameter,
    each pairing an IndexProvider with an optional data store.

    An AtmosphereClient is available by default for anonymous read-only
    resolution of ``@handle/dataset`` paths. Pass an authenticated client
    for write operations, or ``atmosphere=None`` to disable.

    Attributes:
        _provider: IndexProvider for the built-in ``"local"`` repository.
        _data_store: Optional AbstractDataStore for the local repository.
        _repos: Named repositories beyond ``"local"``.
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
                - ``AtmosphereClient`` instance: uses that client directly.
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

        if isinstance(provider, str):
            # String-based provider selection
            from atdata.providers._factory import create_provider

            self._provider: _IP = create_provider(
                provider, path=path, dsn=dsn, redis=redis, **kwargs
            )
        elif provider is not None:
            if not isinstance(provider, _IP):
                raise TypeError(
                    f"provider must be an IndexProvider or backend name string, "
                    f"got {type(provider).__name__}"
                )
            self._provider = provider
        elif redis is not None:
            # Explicit Redis connection provided
            from atdata.providers._redis import RedisProvider

            self._provider = RedisProvider(redis)
        elif kwargs:
            # kwargs provided — assume Redis constructor args for compat
            from atdata.providers._redis import RedisProvider

            self._provider = RedisProvider(Redis(**kwargs))
        else:
            # Default: zero-dependency SQLite
            from atdata.providers._sqlite import SqliteProvider

            self._provider = SqliteProvider()

        self._data_store = data_store

        # Validate and store named repositories
        from atdata.repository import Repository as _Repo

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
            self._repos: dict[str, _Repo] = dict(repos)
        else:
            self._repos = {}

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
                from atdata.atmosphere.client import AtmosphereClient
                from atdata.repository import _AtmosphereBackend

                client = AtmosphereClient()
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
    def repos(self) -> dict[str, Repository]:
        """Named repositories mounted on this index (excluding ``"local"``)."""
        return dict(self._repos)

    @property
    def atmosphere(self) -> Any:
        """The AtmosphereClient for this index, or None if disabled.

        Returns the underlying client (not the internal backend wrapper).
        """
        backend = self._get_atmosphere()
        if backend is not None:
            return backend.client
        return None

    @property
    def provider(self) -> "IndexProvider":  # noqa: F821
        """The storage provider backing this index."""
        return self._provider

    @property
    def _redis(self) -> Redis:
        """Backwards-compatible access to the underlying Redis connection.

        Raises:
            AttributeError: If the current provider is not Redis-backed.
        """
        from atdata.providers._redis import RedisProvider

        if isinstance(self._provider, RedisProvider):
            return self._provider.redis
        raise AttributeError(
            "Index._redis is only available with a Redis provider. "
            "Use index.provider instead."
        )

    @property
    def data_store(self) -> AbstractDataStore | None:
        """The data store for writing shards, or None if index-only."""
        return self._data_store

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
            >>> index.load_schema("atdata://local/sampleSchema/MySample@1.0.0")
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
            ref: Schema reference string (atdata://local/sampleSchema/... or
                legacy local://schemas/...).

        Returns:
            The decoded PackableSample subclass. Also available via
            ``index.types.<ClassName>`` after this call.

        Raises:
            KeyError: If schema not found.
            ValueError: If schema cannot be decoded.

        Examples:
            >>> # Load and use immediately
            >>> MyType = index.load_schema("atdata://local/sampleSchema/MySample@1.0.0")
            >>> sample = MyType(field1="hello", field2=42)
            >>>
            >>> # Or access later via namespace
            >>> index.load_schema("atdata://local/sampleSchema/OtherType@1.0.0")
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

        Args:
            ds: The dataset to add to the index.
            name: Human-readable name for the dataset.
            schema_ref: Optional schema reference. If None, generates from sample type.
            metadata: Optional metadata dictionary. If None, uses ds._metadata if available.

        Returns:
            The created LocalDatasetEntry object.
        """
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
        metadata = kwargs.get("metadata")

        if store is not None:
            prefix = kwargs.get("prefix", name)
            cache_local = kwargs.get("cache_local", False)

            written_urls = store.write_shards(
                ds,
                prefix=prefix,
                cache_local=cache_local,
            )

            if schema_ref is None:
                schema_ref = _schema_ref_from_type(ds.sample_type, version="1.0.0")

            entry_metadata = metadata if metadata is not None else ds._metadata
            entry = LocalDatasetEntry(
                name=name,
                schema_ref=schema_ref,
                data_urls=written_urls,
                metadata=entry_metadata,
            )
            provider.store_entry(entry)
            return entry

        # No data store - just index the existing URL
        if schema_ref is None:
            schema_ref = _schema_ref_from_type(ds.sample_type, version="1.0.0")

        data_urls = [ds.url]
        entry_metadata = metadata if metadata is not None else ds._metadata

        entry = LocalDatasetEntry(
            name=name,
            schema_ref=schema_ref,
            data_urls=data_urls,
            metadata=entry_metadata,
        )
        provider.store_entry(entry)
        return entry

    def insert_dataset(
        self,
        ds: Dataset,
        *,
        name: str,
        schema_ref: str | None = None,
        **kwargs,
    ) -> "IndexEntry":
        """Insert a dataset into the index (AbstractIndex protocol).

        The target repository is determined by a prefix in the ``name``
        argument (e.g. ``"lab/mnist"``). If no prefix is given, or the
        prefix is ``"local"``, the built-in local repository is used.

        If the target repository has a data_store, shards are written to
        storage first, then indexed. Otherwise, the dataset's existing URL
        is indexed directly.

        Args:
            ds: The Dataset to register.
            name: Human-readable name for the dataset, optionally prefixed
                with a repository name (e.g. ``"lab/mnist"``).
            schema_ref: Optional schema reference.
            **kwargs: Additional options:
                - metadata: Optional metadata dict
                - prefix: Storage prefix (default: dataset name)
                - cache_local: If True, cache writes locally first

        Returns:
            IndexEntry for the inserted dataset.
        """
        backend_key, resolved_name, handle_or_did = self._resolve_prefix(name)

        if backend_key == "_atmosphere":
            atmo = self._get_atmosphere()
            if atmo is None:
                raise ValueError(
                    f"Atmosphere backend required for name {name!r} but not available."
                )
            return atmo.insert_dataset(
                ds, name=resolved_name, schema_ref=schema_ref, **kwargs
            )

        if backend_key == "local":
            return self._insert_dataset_to_provider(
                ds,
                name=resolved_name,
                schema_ref=schema_ref,
                provider=self._provider,
                store=self._data_store,
                **kwargs,
            )

        # Named repository
        repo = self._repos.get(backend_key)
        if repo is None:
            raise KeyError(f"Unknown repository {backend_key!r} in name {name!r}")
        return self._insert_dataset_to_provider(
            ds,
            name=resolved_name,
            schema_ref=schema_ref,
            provider=repo.provider,
            store=repo.data_store,
            **kwargs,
        )

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
                    "Install 'atproto' or pass an AtmosphereClient."
                )
            return atmo.get_dataset(resolved_ref)

        if backend_key == "local":
            return self._provider.get_entry_by_name(resolved_ref)

        # Named repository
        repo = self._repos.get(backend_key)
        if repo is None:
            raise KeyError(f"Unknown repository {backend_key!r} in ref {ref!r}")
        return repo.provider.get_entry_by_name(resolved_ref)

    @property
    def datasets(self) -> Generator["IndexEntry", None, None]:
        """Lazily iterate over all dataset entries across local repositories.

        Yields entries from the ``"local"`` repository and all named
        repositories. Atmosphere entries are not included (use
        ``list_datasets(repo="_atmosphere")`` for those).

        Yields:
            IndexEntry for each dataset.
        """
        yield from self._provider.iter_entries()
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

        if repo == "local":
            return self.list_entries()

        if repo == "_atmosphere":
            atmo = self._get_atmosphere()
            if atmo is None:
                return []
            return atmo.list_datasets()

        named = self._repos.get(repo)
        if named is None:
            raise KeyError(f"Unknown repository {repo!r}")
        return list(named.provider.iter_entries())

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
            Schema reference string: 'atdata://local/sampleSchema/{name}@{version}'.

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
                (atdata://local/sampleSchema/{name}@{version}) and legacy
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
            ref: Schema reference string (atdata://local/sampleSchema/... or
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
