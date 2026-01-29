"""Local storage backend for atdata datasets.

Key classes:

- ``Index``: Unified index with pluggable providers (SQLite default),
  named repositories, and optional atmosphere backend.
- ``LocalDatasetEntry``: Index entry with ATProto-compatible CIDs.
- ``S3DataStore``: S3-compatible shard storage.
- ``LocalIndex()``: Factory for creating Index with a named provider.
"""

##
# Imports

from atdata import (
    PackableSample,
    Dataset,
)
from atdata._cid import generate_cid
from atdata._type_utils import (
    PRIMITIVE_TYPE_MAP,
    unwrap_optional,
    is_ndarray_type,
    extract_ndarray_dtype,
    parse_semver,
)
from atdata._protocols import AbstractDataStore, Packable

from pathlib import Path
from uuid import uuid4
from tempfile import TemporaryDirectory
from dotenv import dotenv_values
import msgpack

from redis import Redis

from s3fs import (
    S3FileSystem,
)

import webdataset as wds

from dataclasses import (
    dataclass,
    field,
)
from typing import (
    Any,
    Type,
    TypeVar,
    Generator,
    Iterator,
    BinaryIO,
    Optional,
    Literal,
    cast,
    get_type_hints,
    get_origin,
    get_args,
)
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
import json
import warnings

T = TypeVar("T", bound=PackableSample)

# Redis key prefixes for index entries and schemas
REDIS_KEY_DATASET_ENTRY = "LocalDatasetEntry"
REDIS_KEY_SCHEMA = "LocalSchema"


class SchemaNamespace:
    """Namespace for accessing loaded schema types as attributes.

    After ``index.load_schema(uri)``, the type is available as an attribute.
    Supports attribute access, iteration, ``len()``, and ``in`` checks.

    Examples:
        >>> index.load_schema("atdata://local/sampleSchema/MySample@1.0.0")
        >>> MyType = index.types.MySample
        >>> sample = MyType(field1="hello", field2=42)

    Note:
        For full IDE autocomplete, enable ``auto_stubs=True`` and add
        ``index.stub_dir`` to your IDE's extraPaths.
    """

    def __init__(self) -> None:
        self._types: dict[str, Type[Packable]] = {}

    def _register(self, name: str, cls: Type[Packable]) -> None:
        """Register a schema type in the namespace."""
        self._types[name] = cls

    def __getattr__(self, name: str) -> Any:
        # Returns Any to avoid IDE complaints about unknown attributes.
        # For full IDE support, import from the generated module instead.
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
        if name not in self._types:
            raise AttributeError(
                f"Schema '{name}' not loaded. "
                f"Call index.load_schema() first to load the schema."
            )
        return self._types[name]

    def __dir__(self) -> list[str]:
        return list(self._types.keys()) + ["_types", "_register", "get"]

    def __iter__(self) -> Iterator[str]:
        return iter(self._types)

    def __len__(self) -> int:
        return len(self._types)

    def __contains__(self, name: str) -> bool:
        return name in self._types

    def __repr__(self) -> str:
        if not self._types:
            return "SchemaNamespace(empty)"
        names = ", ".join(sorted(self._types.keys()))
        return f"SchemaNamespace({names})"

    def get(self, name: str, default: T | None = None) -> Type[Packable] | T | None:
        """Get a type by name, returning default if not found.

        Args:
            name: The schema class name to look up.
            default: Value to return if not found (default: None).

        Returns:
            The schema class, or default if not loaded.
        """
        return self._types.get(name, default)


##
# Schema types


@dataclass
class SchemaFieldType:
    """Schema field type definition for local storage.

    Represents a type in the schema type system, supporting primitives,
    ndarrays, arrays, and references to other schemas.
    """

    kind: Literal["primitive", "ndarray", "ref", "array"]
    """The category of type."""

    primitive: Optional[str] = None
    """For kind='primitive': one of 'str', 'int', 'float', 'bool', 'bytes'."""

    dtype: Optional[str] = None
    """For kind='ndarray': numpy dtype string (e.g., 'float32')."""

    ref: Optional[str] = None
    """For kind='ref': URI of referenced schema."""

    items: Optional["SchemaFieldType"] = None
    """For kind='array': type of array elements."""

    @classmethod
    def from_dict(cls, data: dict) -> "SchemaFieldType":
        """Create from a dictionary (e.g., from Redis storage)."""
        type_str = data.get("$type", "")
        if "#" in type_str:
            kind = type_str.split("#")[-1]
        else:
            kind = data.get("kind", "primitive")

        items = None
        if "items" in data and data["items"]:
            items = cls.from_dict(data["items"])

        return cls(
            kind=kind,  # type: ignore[arg-type]
            primitive=data.get("primitive"),
            dtype=data.get("dtype"),
            ref=data.get("ref"),
            items=items,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        result: dict[str, Any] = {"$type": f"local#{self.kind}"}
        if self.kind == "primitive":
            result["primitive"] = self.primitive
        elif self.kind == "ndarray":
            result["dtype"] = self.dtype
        elif self.kind == "ref":
            result["ref"] = self.ref
        elif self.kind == "array" and self.items:
            result["items"] = self.items.to_dict()
        return result


@dataclass
class SchemaField:
    """Schema field definition for local storage."""

    name: str
    """Field name."""

    field_type: SchemaFieldType
    """Type of this field."""

    optional: bool = False
    """Whether this field can be None."""

    @classmethod
    def from_dict(cls, data: dict) -> "SchemaField":
        """Create from a dictionary."""
        return cls(
            name=data["name"],
            field_type=SchemaFieldType.from_dict(data["fieldType"]),
            optional=data.get("optional", False),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "name": self.name,
            "fieldType": self.field_type.to_dict(),
            "optional": self.optional,
        }


@dataclass
class LocalSchemaRecord:
    """Schema record for local storage.

    Represents a PackableSample schema stored in the local index.
    Aligns with the atmosphere SchemaRecord structure for seamless promotion.
    """

    name: str
    """Schema name (typically the class name)."""

    version: str
    """Semantic version string (e.g., '1.0.0')."""

    fields: list[SchemaField]
    """List of field definitions."""

    ref: str
    """Schema reference URI (atdata://local/sampleSchema/{name}@{version})."""

    description: Optional[str] = None
    """Human-readable description."""

    created_at: Optional[datetime] = None
    """When this schema was published."""

    @classmethod
    def from_dict(cls, data: dict) -> "LocalSchemaRecord":
        """Create from a dictionary (e.g., from Redis storage)."""
        created_at = None
        if "createdAt" in data:
            try:
                created_at = datetime.fromisoformat(data["createdAt"])
            except (ValueError, TypeError):
                created_at = None  # Invalid datetime format, leave as None

        return cls(
            name=data["name"],
            version=data["version"],
            fields=[SchemaField.from_dict(f) for f in data.get("fields", [])],
            ref=data.get("$ref", ""),
            description=data.get("description"),
            created_at=created_at,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        result: dict[str, Any] = {
            "name": self.name,
            "version": self.version,
            "fields": [f.to_dict() for f in self.fields],
            "$ref": self.ref,
        }
        if self.description:
            result["description"] = self.description
        if self.created_at:
            result["createdAt"] = self.created_at.isoformat()
        return result


##
# Helpers


def _kind_str_for_sample_type(st: Type[Packable]) -> str:
    """Return fully-qualified 'module.name' string for a sample type."""
    return f"{st.__module__}.{st.__name__}"


def _create_s3_write_callbacks(
    credentials: dict[str, Any],
    temp_dir: str,
    written_shards: list[str],
    fs: S3FileSystem | None,
    cache_local: bool,
    add_s3_prefix: bool = False,
) -> tuple:
    """Create opener and post callbacks for ShardWriter with S3 upload.

    Args:
        credentials: S3 credentials dict.
        temp_dir: Temporary directory for local caching.
        written_shards: List to append written shard paths to.
        fs: S3FileSystem for direct writes (used when cache_local=False).
        cache_local: If True, write locally then copy to S3.
        add_s3_prefix: If True, prepend 's3://' to shard paths.

    Returns:
        Tuple of (writer_opener, writer_post) callbacks.
    """
    if cache_local:
        import boto3

        s3_client_kwargs = {
            "aws_access_key_id": credentials["AWS_ACCESS_KEY_ID"],
            "aws_secret_access_key": credentials["AWS_SECRET_ACCESS_KEY"],
        }
        if "AWS_ENDPOINT" in credentials:
            s3_client_kwargs["endpoint_url"] = credentials["AWS_ENDPOINT"]
        s3_client = boto3.client("s3", **s3_client_kwargs)

        def _writer_opener(p: str):
            local_path = Path(temp_dir) / p
            local_path.parent.mkdir(parents=True, exist_ok=True)
            return open(local_path, "wb")

        def _writer_post(p: str):
            local_path = Path(temp_dir) / p
            path_parts = Path(p).parts
            bucket = path_parts[0]
            key = str(Path(*path_parts[1:]))

            with open(local_path, "rb") as f_in:
                s3_client.put_object(Bucket=bucket, Key=key, Body=f_in.read())

            local_path.unlink()
            if add_s3_prefix:
                written_shards.append(f"s3://{p}")
            else:
                written_shards.append(p)

        return _writer_opener, _writer_post
    else:
        if fs is None:
            raise ValueError("S3FileSystem required when cache_local=False")

        def _direct_opener(s: str):
            return cast(BinaryIO, fs.open(f"s3://{s}", "wb"))

        def _direct_post(s: str):
            if add_s3_prefix:
                written_shards.append(f"s3://{s}")
            else:
                written_shards.append(s)

        return _direct_opener, _direct_post


##
# Schema helpers

# URI scheme prefixes
_ATDATA_URI_PREFIX = "atdata://local/sampleSchema/"
_LEGACY_URI_PREFIX = "local://schemas/"


def _schema_ref_from_type(sample_type: Type[Packable], version: str) -> str:
    """Generate 'atdata://local/sampleSchema/{name}@{version}' reference."""
    return _make_schema_ref(sample_type.__name__, version)


def _make_schema_ref(name: str, version: str) -> str:
    """Generate schema reference URI from name and version."""
    return f"{_ATDATA_URI_PREFIX}{name}@{version}"


def _parse_schema_ref(ref: str) -> tuple[str, str]:
    """Parse schema reference into (name, version).

    Supports both new format: 'atdata://local/sampleSchema/{name}@{version}'
    and legacy format: 'local://schemas/{module.Class}@{version}'
    """
    if ref.startswith(_ATDATA_URI_PREFIX):
        path = ref[len(_ATDATA_URI_PREFIX) :]
    elif ref.startswith(_LEGACY_URI_PREFIX):
        path = ref[len(_LEGACY_URI_PREFIX) :]
    else:
        raise ValueError(f"Invalid schema reference: {ref}")

    if "@" not in path:
        raise ValueError(f"Schema reference must include version (@version): {ref}")

    name, version = path.rsplit("@", 1)
    # For legacy format, extract just the class name from module.Class
    if "." in name:
        name = name.rsplit(".", 1)[1]
    return name, version


def _increment_patch(version: str) -> str:
    """Increment patch version: 1.0.0 -> 1.0.1"""
    major, minor, patch = parse_semver(version)
    return f"{major}.{minor}.{patch + 1}"


def _python_type_to_field_type(python_type: Any) -> dict:
    """Convert Python type annotation to schema field type dict."""
    if python_type in PRIMITIVE_TYPE_MAP:
        return {
            "$type": "local#primitive",
            "primitive": PRIMITIVE_TYPE_MAP[python_type],
        }

    if is_ndarray_type(python_type):
        return {"$type": "local#ndarray", "dtype": extract_ndarray_dtype(python_type)}

    origin = get_origin(python_type)
    if origin is list:
        args = get_args(python_type)
        items = (
            _python_type_to_field_type(args[0])
            if args
            else {"$type": "local#primitive", "primitive": "str"}
        )
        return {"$type": "local#array", "items": items}

    if is_dataclass(python_type):
        raise TypeError(
            f"Nested dataclass types not yet supported: {python_type.__name__}. "
            "Publish nested types separately and use references."
        )

    raise TypeError(f"Unsupported type for schema field: {python_type}")


def _build_schema_record(
    sample_type: Type[Packable],
    *,
    version: str,
    description: str | None = None,
) -> dict:
    """Build a schema record dict from a PackableSample type.

    Args:
        sample_type: The PackableSample subclass to introspect.
        version: Semantic version string.
        description: Optional human-readable description. If None, uses the
            class docstring.

    Returns:
        Schema record dict suitable for Redis storage.

    Raises:
        ValueError: If sample_type is not a dataclass.
        TypeError: If a field type is not supported.
    """
    if not is_dataclass(sample_type):
        raise ValueError(f"{sample_type.__name__} must be a dataclass (use @packable)")

    # Use docstring as fallback for description
    if description is None:
        description = sample_type.__doc__

    field_defs = []
    type_hints = get_type_hints(sample_type)

    for f in fields(sample_type):
        field_type = type_hints.get(f.name, f.type)
        field_type, is_optional = unwrap_optional(field_type)
        field_type_dict = _python_type_to_field_type(field_type)

        field_defs.append(
            {
                "name": f.name,
                "fieldType": field_type_dict,
                "optional": is_optional,
            }
        )

    return {
        "name": sample_type.__name__,
        "version": version,
        "fields": field_defs,
        "description": description,
        "createdAt": datetime.now(timezone.utc).isoformat(),
    }


##
# Redis object model


@dataclass
class LocalDatasetEntry:
    """Index entry for a dataset stored in the local repository.

    Implements the IndexEntry protocol for compatibility with AbstractIndex.
    Uses dual identity: a content-addressable CID (ATProto-compatible) and
    a human-readable name.

    The CID is generated from the entry's content (schema_ref + data_urls),
    ensuring the same data produces the same CID whether stored locally or
    in the atmosphere. This enables seamless promotion from local to ATProto.

    Attributes:
        name: Human-readable name for this dataset.
        schema_ref: Reference to the schema for this dataset.
        data_urls: WebDataset URLs for the data.
        metadata: Arbitrary metadata dictionary, or None if not set.
    """

    ##

    name: str
    """Human-readable name for this dataset."""

    schema_ref: str
    """Reference to the schema for this dataset."""

    data_urls: list[str]
    """WebDataset URLs for the data."""

    metadata: dict | None = None
    """Arbitrary metadata dictionary, or None if not set."""

    _cid: str | None = field(default=None, repr=False)
    """Content identifier (ATProto-compatible CID). Generated from content if not provided."""

    # Legacy field for backwards compatibility during migration
    _legacy_uuid: str | None = field(default=None, repr=False)
    """Legacy UUID for backwards compatibility with existing Redis entries."""

    def __post_init__(self):
        """Generate CID from content if not provided."""
        if self._cid is None:
            self._cid = self._generate_cid()

    def _generate_cid(self) -> str:
        """Generate ATProto-compatible CID from entry content."""
        # CID is based on schema_ref and data_urls - the identity of the dataset
        content = {
            "schema_ref": self.schema_ref,
            "data_urls": self.data_urls,
        }
        return generate_cid(content)

    @property
    def cid(self) -> str:
        """Content identifier (ATProto-compatible CID)."""
        if self._cid is None:
            raise RuntimeError("CID not initialized; this should not happen after __post_init__")
        return self._cid

    # Legacy compatibility

    @property
    def wds_url(self) -> str:
        """Legacy property: returns first data URL for backwards compatibility."""
        return self.data_urls[0] if self.data_urls else ""

    @property
    def sample_kind(self) -> str:
        """Legacy property: returns schema_ref for backwards compatibility."""
        return self.schema_ref

    def write_to(self, redis: Redis):
        """Persist this index entry to Redis.

        Stores the entry as a Redis hash with key '{REDIS_KEY_DATASET_ENTRY}:{cid}'.

        Args:
            redis: Redis connection to write to.
        """
        save_key = f"{REDIS_KEY_DATASET_ENTRY}:{self.cid}"
        data = {
            "name": self.name,
            "schema_ref": self.schema_ref,
            "data_urls": msgpack.packb(self.data_urls),  # Serialize list
            "cid": self.cid,
        }
        if self.metadata is not None:
            data["metadata"] = msgpack.packb(self.metadata)
        if self._legacy_uuid is not None:
            data["legacy_uuid"] = self._legacy_uuid

        redis.hset(save_key, mapping=data)  # type: ignore[arg-type]

    @classmethod
    def from_redis(cls, redis: Redis, cid: str) -> "LocalDatasetEntry":
        """Load an entry from Redis by CID.

        Args:
            redis: Redis connection to read from.
            cid: Content identifier of the entry to load.

        Returns:
            LocalDatasetEntry loaded from Redis.

        Raises:
            KeyError: If entry not found.
        """
        save_key = f"{REDIS_KEY_DATASET_ENTRY}:{cid}"
        raw_data = redis.hgetall(save_key)
        if not raw_data:
            raise KeyError(f"{REDIS_KEY_DATASET_ENTRY} not found: {cid}")

        # Decode string fields, keep binary fields as bytes for msgpack
        raw_data_typed = cast(dict[bytes, bytes], raw_data)
        name = raw_data_typed[b"name"].decode("utf-8")
        schema_ref = raw_data_typed[b"schema_ref"].decode("utf-8")
        cid_value = raw_data_typed.get(b"cid", b"").decode("utf-8") or None
        legacy_uuid = raw_data_typed.get(b"legacy_uuid", b"").decode("utf-8") or None

        # Deserialize msgpack fields (stored as raw bytes)
        data_urls = msgpack.unpackb(raw_data_typed[b"data_urls"])
        metadata = None
        if b"metadata" in raw_data_typed:
            metadata = msgpack.unpackb(raw_data_typed[b"metadata"])

        return cls(
            name=name,
            schema_ref=schema_ref,
            data_urls=data_urls,
            metadata=metadata,
            _cid=cid_value,
            _legacy_uuid=legacy_uuid,
        )


# Backwards compatibility alias
BasicIndexEntry = LocalDatasetEntry


def _s3_env(credentials_path: str | Path) -> dict[str, Any]:
    """Load S3 credentials from .env file.

    Args:
        credentials_path: Path to .env file containing AWS_ENDPOINT,
            AWS_ACCESS_KEY_ID, and AWS_SECRET_ACCESS_KEY.

    Returns:
        Dict with the three required credential keys.

    Raises:
        ValueError: If any required key is missing from the .env file.
    """
    credentials_path = Path(credentials_path)
    env_values = dotenv_values(credentials_path)

    required_keys = ("AWS_ENDPOINT", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
    missing = [k for k in required_keys if k not in env_values]
    if missing:
        raise ValueError(
            f"Missing required keys in {credentials_path}: {', '.join(missing)}"
        )

    return {k: env_values[k] for k in required_keys}


def _s3_from_credentials(creds: str | Path | dict) -> S3FileSystem:
    """Create S3FileSystem from credentials dict or .env file path."""
    if not isinstance(creds, dict):
        creds = _s3_env(creds)

    # Build kwargs, making endpoint_url optional
    kwargs = {
        "key": creds["AWS_ACCESS_KEY_ID"],
        "secret": creds["AWS_SECRET_ACCESS_KEY"],
    }
    if "AWS_ENDPOINT" in creds:
        kwargs["endpoint_url"] = creds["AWS_ENDPOINT"]

    return S3FileSystem(**kwargs)


##
# Classes


class Repo:
    """Repository for storing and managing atdata datasets.

    .. deprecated::
        Use :class:`Index` with :class:`S3DataStore` instead::

            store = S3DataStore(credentials, bucket="my-bucket")
            index = Index(redis=redis, data_store=store)
            entry = index.insert_dataset(ds, name="my-dataset")

    Provides storage of datasets in S3-compatible object storage with Redis-based
    indexing. Datasets are stored as WebDataset tar files with optional metadata.

    Attributes:
        s3_credentials: S3 credentials dictionary or None.
        bucket_fs: S3FileSystem instance or None.
        hive_path: Path within S3 bucket for storing datasets.
        hive_bucket: Name of the S3 bucket.
        index: Index instance for tracking datasets.
    """

    ##

    def __init__(
        self,
        s3_credentials: str | Path | dict[str, Any] | None = None,
        hive_path: str | Path | None = None,
        redis: Redis | None = None,
    ) -> None:
        """Initialize a repository.

        .. deprecated::
            Use Index with S3DataStore instead.

        Args:
            s3_credentials: Path to .env file with S3 credentials, or dict with
                AWS_ENDPOINT, AWS_ACCESS_KEY_ID, and AWS_SECRET_ACCESS_KEY.
                If None, S3 functionality will be disabled.
            hive_path: Path within the S3 bucket to store datasets.
                Required if s3_credentials is provided.
            redis: Redis connection for indexing. If None, creates a new connection.

        Raises:
            ValueError: If hive_path is not provided when s3_credentials is set.
        """
        warnings.warn(
            "Repo is deprecated. Use Index with S3DataStore instead:\n"
            "  store = S3DataStore(credentials, bucket='my-bucket')\n"
            "  index = Index(redis=redis, data_store=store)\n"
            "  entry = index.insert_dataset(ds, name='my-dataset')",
            DeprecationWarning,
            stacklevel=2,
        )

        if s3_credentials is None:
            self.s3_credentials = None
        elif isinstance(s3_credentials, dict):
            self.s3_credentials = s3_credentials
        else:
            self.s3_credentials = _s3_env(s3_credentials)

        if self.s3_credentials is None:
            self.bucket_fs = None
        else:
            self.bucket_fs = _s3_from_credentials(self.s3_credentials)

        if self.bucket_fs is not None:
            if hive_path is None:
                raise ValueError("Must specify hive path within bucket")
            self.hive_path = Path(hive_path)
            self.hive_bucket = self.hive_path.parts[0]
        else:
            self.hive_path = None
            self.hive_bucket = None

        #

        self.index = Index(redis=redis)

    ##

    def insert(
        self,
        ds: Dataset[T],
        *,
        name: str,
        cache_local: bool = False,
        schema_ref: str | None = None,
        **kwargs,
    ) -> tuple[LocalDatasetEntry, Dataset[T]]:
        """Insert a dataset into the repository.

        Writes the dataset to S3 as WebDataset tar files, stores metadata,
        and creates an index entry in Redis.

        Args:
            ds: The dataset to insert.
            name: Human-readable name for the dataset.
            cache_local: If True, write to local temporary storage first, then
                copy to S3. This can be faster for some workloads.
            schema_ref: Optional schema reference. If None, generates from sample type.
            **kwargs: Additional arguments passed to wds.ShardWriter.

        Returns:
            A tuple of (index_entry, new_dataset) where:
                - index_entry: LocalDatasetEntry for the stored dataset
                - new_dataset: Dataset object pointing to the stored copy

        Raises:
            ValueError: If S3 credentials or hive_path are not configured.
            RuntimeError: If no shards were written.
        """
        if self.s3_credentials is None:
            raise ValueError(
                "S3 credentials required for insert(). Initialize Repo with s3_credentials."
            )
        if self.hive_bucket is None or self.hive_path is None:
            raise ValueError(
                "hive_path required for insert(). Initialize Repo with hive_path."
            )

        new_uuid = str(uuid4())

        hive_fs = _s3_from_credentials(self.s3_credentials)

        # Write metadata
        metadata_path = (
            self.hive_path / "metadata" / f"atdata-metadata--{new_uuid}.msgpack"
        )
        # Note: S3 doesn't need directories created beforehand - s3fs handles this

        if ds.metadata is not None:
            # Use s3:// prefix to ensure s3fs treats this as an S3 path
            with cast(
                BinaryIO, hive_fs.open(f"s3://{metadata_path.as_posix()}", "wb")
            ) as f:
                meta_packed = msgpack.packb(ds.metadata)
                f.write(cast(bytes, meta_packed))

        # Write data
        shard_pattern = (self.hive_path / f"atdata--{new_uuid}--%06d.tar").as_posix()

        written_shards: list[str] = []
        with TemporaryDirectory() as temp_dir:
            writer_opener, writer_post = _create_s3_write_callbacks(
                credentials=self.s3_credentials,
                temp_dir=temp_dir,
                written_shards=written_shards,
                fs=hive_fs,
                cache_local=cache_local,
                add_s3_prefix=False,
            )

            with wds.writer.ShardWriter(
                shard_pattern,
                opener=writer_opener,
                post=writer_post,
                **kwargs,
            ) as sink:
                for sample in ds.ordered(batch_size=None):
                    sink.write(sample.as_wds)

        # Make a new Dataset object for the written dataset copy
        if len(written_shards) == 0:
            raise RuntimeError(
                "Cannot form new dataset entry -- did not write any shards"
            )

        elif len(written_shards) < 2:
            new_dataset_url = (
                self.hive_path / (Path(written_shards[0]).name)
            ).as_posix()

        else:
            shard_s3_format = (
                (self.hive_path / f"atdata--{new_uuid}").as_posix()
            ) + "--{shard_id}.tar"
            shard_id_braced = "{" + f"{0:06d}..{len(written_shards) - 1:06d}" + "}"
            new_dataset_url = shard_s3_format.format(shard_id=shard_id_braced)

        new_dataset = Dataset[ds.sample_type](
            url=new_dataset_url,
            metadata_url=metadata_path.as_posix(),
        )

        # Add to index (use ds._metadata to avoid network requests)
        new_entry = self.index.add_entry(
            new_dataset,
            name=name,
            schema_ref=schema_ref,
            metadata=ds._metadata,
        )

        return new_entry, new_dataset


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
        provider: "IndexProvider | None" = None,  # noqa: F821
        *,
        redis: Redis | None = None,
        data_store: AbstractDataStore | None = None,
        repos: "dict[str, Repository] | None" = None,
        atmosphere: "Any | None" = _ATMOSPHERE_DEFAULT,
        auto_stubs: bool = False,
        stub_dir: Path | str | None = None,
        **kwargs,
    ) -> None:
        """Initialize an index.

        Args:
            provider: Storage backend for the ``"local"`` repository.  If
                provided, all local persistence is delegated to this provider.
                Mutually exclusive with *redis* and extra *kwargs*.
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
            TypeError: If provider is not an IndexProvider.
            ValueError: If repos contains the reserved name ``"local"``.

        Examples:
            >>> # Default: local SQLite + anonymous atmosphere
            >>> index = Index()
            >>>
            >>> # SQLite local + authenticated atmosphere
            >>> from atdata.providers import create_provider
            >>> from atdata.atmosphere import AtmosphereClient
            >>> client = AtmosphereClient()
            >>> client.login("alice.bsky.social", "app-password")
            >>> index = Index(
            ...     provider=create_provider("sqlite"),
            ...     atmosphere=client,
            ... )
            >>>
            >>> # Multiple repositories
            >>> from atdata.repository import Repository, create_repository
            >>> index = Index(
            ...     provider=create_provider("sqlite"),
            ...     repos={
            ...         "lab": create_repository("sqlite", path="/data/lab.db"),
            ...     },
            ... )
        """
        ##

        from .providers._base import IndexProvider as _IP

        if provider is not None:
            if not isinstance(provider, _IP):
                raise TypeError(
                    f"provider must be an IndexProvider, got {type(provider).__name__}"
                )
            self._provider: _IP = provider
        elif redis is not None:
            # Explicit Redis connection provided
            from .providers._redis import RedisProvider

            self._provider = RedisProvider(redis)
        elif kwargs:
            # kwargs provided — assume Redis constructor args for compat
            from .providers._redis import RedisProvider

            self._provider = RedisProvider(Redis(**kwargs))
        else:
            # Default: zero-dependency SQLite
            from .providers._sqlite import SqliteProvider

            self._provider = SqliteProvider()

        self._data_store = data_store

        # Validate and store named repositories
        from .repository import Repository as _Repo

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
        from .repository import _AtmosphereBackend

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
            from ._stub_manager import StubManager

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
                from .atmosphere.client import AtmosphereClient
                from .repository import _AtmosphereBackend

                client = AtmosphereClient()
                self._atmosphere = _AtmosphereBackend(client)
            except ImportError:
                # atproto package not installed -- atmosphere unavailable
                self._atmosphere_deferred = False
                return None
        return self._atmosphere

    def _resolve_prefix(
        self, ref: str
    ) -> tuple[str, str, str | None]:
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
            path = ref[len("atdata://"):]
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
        """Named repositories mounted on this index (excluding ``"local"``)."""
        from .repository import Repository as _Repo

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
        from .providers._redis import RedisProvider

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
            >>> sample = MyType(name="hello", value=42)
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
            >>> index = LocalIndex(auto_stubs=True)
            >>> ref = index.publish_schema(MySample, version="1.0.0")
            >>> index.load_schema(ref)
            >>> print(index.get_import_path(ref))
            local.MySample_1_0_0
            >>> # Then in your code:
            >>> # from local.MySample_1_0_0 import MySample
        """
        if self._stub_manager is None:
            return None

        from ._stub_manager import _extract_authority

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
            record = LocalSchemaRecord.from_dict(schema)
            self._stub_manager.ensure_stub(record)

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


def LocalIndex(
    provider: str = "redis",
    *,
    path: str | Path | None = None,
    dsn: str | None = None,
    redis: Redis | None = None,
    data_store: AbstractDataStore | None = None,
    auto_stubs: bool = False,
    stub_dir: Path | str | None = None,
    **kwargs,
) -> Index:
    """Create an Index with the specified storage backend.

    This is the recommended entry point for creating index instances.
    It wraps the ``Index`` class with a convenient provider-selection API.

    Args:
        provider: Backend type — ``"redis"``, ``"sqlite"``, or ``"postgres"``.
        path: Database file path (SQLite only).  Defaults to
            ``~/.atdata/index.db`` when *provider* is ``"sqlite"``.
        dsn: PostgreSQL connection string (postgres only).
        redis: Existing Redis connection (redis only).  If ``None`` and
            *provider* is ``"redis"``, a new connection is created from
            *kwargs*.
        data_store: Optional data store for writing dataset shards.
        auto_stubs: Enable automatic stub file generation.
        stub_dir: Directory for stub files.
        **kwargs: Passed to the Redis constructor when *provider* is
            ``"redis"`` and *redis* is ``None``.

    Returns:
        A configured ``Index`` instance.

    Raises:
        ValueError: If *provider* is not a recognised backend name.

    Examples:
        >>> # SQLite (zero-dependency local storage)
        >>> index = LocalIndex(provider="sqlite", path="~/.atdata/index.db")

        >>> # Redis (default, backwards-compatible)
        >>> index = LocalIndex()

        >>> # PostgreSQL
        >>> index = LocalIndex(provider="postgres", dsn="postgresql://user:pass@host/db")
    """
    from .providers._factory import create_provider

    backend = create_provider(provider, path=path, dsn=dsn, redis=redis, **kwargs)
    return Index(
        provider=backend,
        data_store=data_store,
        atmosphere=None,
        auto_stubs=auto_stubs,
        stub_dir=stub_dir,
    )


class S3DataStore:
    """S3-compatible data store implementing AbstractDataStore protocol.

    Handles writing dataset shards to S3-compatible object storage and
    resolving URLs for reading.

    Attributes:
        credentials: S3 credentials dictionary.
        bucket: Target bucket name.
        _fs: S3FileSystem instance.
    """

    def __init__(
        self,
        credentials: str | Path | dict[str, Any],
        *,
        bucket: str,
    ) -> None:
        """Initialize an S3 data store.

        Args:
            credentials: Path to .env file or dict with AWS_ACCESS_KEY_ID,
                AWS_SECRET_ACCESS_KEY, and optionally AWS_ENDPOINT.
            bucket: Name of the S3 bucket for storage.
        """
        if isinstance(credentials, dict):
            self.credentials = credentials
        else:
            self.credentials = _s3_env(credentials)

        self.bucket = bucket
        self._fs = _s3_from_credentials(self.credentials)

    def write_shards(
        self,
        ds: Dataset,
        *,
        prefix: str,
        cache_local: bool = False,
        **kwargs,
    ) -> list[str]:
        """Write dataset shards to S3.

        Args:
            ds: The Dataset to write.
            prefix: Path prefix within bucket (e.g., 'datasets/mnist/v1').
            cache_local: If True, write locally first then copy to S3.
            **kwargs: Additional args passed to wds.ShardWriter (e.g., maxcount).

        Returns:
            List of S3 URLs for the written shards.

        Raises:
            RuntimeError: If no shards were written.
        """
        new_uuid = str(uuid4())
        shard_pattern = f"{self.bucket}/{prefix}/data--{new_uuid}--%06d.tar"

        written_shards: list[str] = []

        with TemporaryDirectory() as temp_dir:
            writer_opener, writer_post = _create_s3_write_callbacks(
                credentials=self.credentials,
                temp_dir=temp_dir,
                written_shards=written_shards,
                fs=self._fs,
                cache_local=cache_local,
                add_s3_prefix=True,
            )

            with wds.writer.ShardWriter(
                shard_pattern,
                opener=writer_opener,
                post=writer_post,
                **kwargs,
            ) as sink:
                for sample in ds.ordered(batch_size=None):
                    sink.write(sample.as_wds)

        if len(written_shards) == 0:
            raise RuntimeError("No shards written")

        return written_shards

    def read_url(self, url: str) -> str:
        """Resolve an S3 URL for reading/streaming.

        For S3-compatible stores with custom endpoints (like Cloudflare R2,
        MinIO, etc.), converts s3:// URLs to HTTPS URLs that WebDataset can
        stream directly.

        For standard AWS S3 (no custom endpoint), URLs are returned unchanged
        since WebDataset's built-in s3fs integration handles them.

        Args:
            url: S3 URL to resolve (e.g., 's3://bucket/path/file.tar').

        Returns:
            HTTPS URL if custom endpoint is configured, otherwise unchanged.
            Example: 's3://bucket/path' -> 'https://endpoint.com/bucket/path'
        """
        endpoint = self.credentials.get("AWS_ENDPOINT")
        if endpoint and url.startswith("s3://"):
            # s3://bucket/path -> https://endpoint/bucket/path
            path = url[5:]  # Remove 's3://' prefix
            endpoint = endpoint.rstrip("/")
            return f"{endpoint}/{path}"
        return url

    def supports_streaming(self) -> bool:
        """S3 supports streaming reads.

        Returns:
            True.
        """
        return True


#
