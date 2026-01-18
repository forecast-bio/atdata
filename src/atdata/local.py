"""Local repository storage for atdata datasets.

This module provides a local storage backend for atdata datasets using:
- S3-compatible object storage for dataset tar files and metadata
- Redis for indexing and tracking datasets

The main classes are:
- Repo: Manages dataset storage in S3 with Redis indexing
- LocalIndex: Redis-backed index for tracking dataset metadata
- LocalDatasetEntry: Index entry representing a stored dataset

This is intended for development and small-scale deployment before
migrating to the full atproto PDS infrastructure. The implementation
uses ATProto-compatible CIDs for content addressing, enabling seamless
promotion from local storage to the atmosphere (ATProto network).
"""

##
# Imports

from atdata import (
    PackableSample,
    Dataset,
)
from atdata._cid import generate_cid
from atdata._type_utils import numpy_dtype_to_string, PRIMITIVE_TYPE_MAP
from atdata._protocols import IndexEntry, AbstractDataStore

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
    Union,
    cast,
    get_type_hints,
    get_origin,
    get_args,
)
import types
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
import json
import warnings

T = TypeVar( 'T', bound = PackableSample )

# Redis key prefixes for index entries and schemas
REDIS_KEY_DATASET_ENTRY = "LocalDatasetEntry"
REDIS_KEY_SCHEMA = "LocalSchema"


##
# Helpers

def _kind_str_for_sample_type( st: Type[PackableSample] ) -> str:
    """Return fully-qualified 'module.name' string for a sample type."""
    return f'{st.__module__}.{st.__name__}'


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
            'aws_access_key_id': credentials['AWS_ACCESS_KEY_ID'],
            'aws_secret_access_key': credentials['AWS_SECRET_ACCESS_KEY']
        }
        if 'AWS_ENDPOINT' in credentials:
            s3_client_kwargs['endpoint_url'] = credentials['AWS_ENDPOINT']
        s3_client = boto3.client('s3', **s3_client_kwargs)

        def _writer_opener(p: str):
            local_path = Path(temp_dir) / p
            local_path.parent.mkdir(parents=True, exist_ok=True)
            return open(local_path, 'wb')

        def _writer_post(p: str):
            local_path = Path(temp_dir) / p
            path_parts = Path(p).parts
            bucket = path_parts[0]
            key = str(Path(*path_parts[1:]))

            with open(local_path, 'rb') as f_in:
                s3_client.put_object(Bucket=bucket, Key=key, Body=f_in.read())

            local_path.unlink()
            if add_s3_prefix:
                written_shards.append(f"s3://{p}")
            else:
                written_shards.append(p)

        return _writer_opener, _writer_post
    else:
        assert fs is not None, "S3FileSystem required when cache_local=False"

        def _direct_opener(s: str):
            return cast(BinaryIO, fs.open(f's3://{s}', 'wb'))

        def _direct_post(s: str):
            if add_s3_prefix:
                written_shards.append(f"s3://{s}")
            else:
                written_shards.append(s)

        return _direct_opener, _direct_post

##
# Schema helpers

def _schema_ref_from_type(sample_type: Type[PackableSample], version: str = "1.0.0") -> str:
    """Generate 'local://schemas/{module.Class}@{version}' reference."""
    kind_str = _kind_str_for_sample_type(sample_type)
    return f"local://schemas/{kind_str}@{version}"


def _parse_schema_ref(ref: str) -> tuple[str, str]:
    """Parse 'local://schemas/{module.Class}@{version}' into (module.Class, version)."""
    if not ref.startswith("local://schemas/"):
        raise ValueError(f"Invalid local schema reference: {ref}")

    path = ref[len("local://schemas/"):]
    if "@" not in path:
        raise ValueError(f"Schema reference must include version (@version): {ref}")

    kind_str, version = path.rsplit("@", 1)
    return kind_str, version


def _python_type_to_field_type(python_type: Any) -> dict:
    """Convert Python type annotation to schema field type dict."""
    # Handle primitives
    if python_type in PRIMITIVE_TYPE_MAP:
        return {"$type": "local#primitive", "primitive": PRIMITIVE_TYPE_MAP[python_type]}

    # Check for NDArray
    type_str = str(python_type)
    if "NDArray" in type_str or "ndarray" in type_str.lower():
        dtype = "float32"  # Default
        args = get_args(python_type)
        if args:
            dtype_arg = args[-1] if args else None
            if dtype_arg is not None:
                dtype = numpy_dtype_to_string(dtype_arg)
        return {"$type": "local#ndarray", "dtype": dtype}

    # Check for list/array types
    origin = get_origin(python_type)
    if origin is list:
        args = get_args(python_type)
        if args:
            items = _python_type_to_field_type(args[0])
            return {"$type": "local#array", "items": items}
        else:
            return {"$type": "local#array", "items": {"$type": "local#primitive", "primitive": "str"}}

    # Check for nested dataclass (not yet supported)
    if is_dataclass(python_type):
        raise TypeError(
            f"Nested dataclass types not yet supported: {python_type.__name__}. "
            "Publish nested types separately and use references."
        )

    raise TypeError(f"Unsupported type for schema field: {python_type}")


def _build_schema_record(
    sample_type: Type[PackableSample],
    *,
    version: str = "1.0.0",
    description: str | None = None,
) -> dict:
    """Build a schema record dict from a PackableSample type.

    Args:
        sample_type: The PackableSample subclass to introspect.
        version: Semantic version string.
        description: Optional human-readable description.

    Returns:
        Schema record dict suitable for Redis storage.

    Raises:
        ValueError: If sample_type is not a dataclass.
        TypeError: If a field type is not supported.
    """
    if not is_dataclass(sample_type):
        raise ValueError(f"{sample_type.__name__} must be a dataclass (use @packable)")

    field_defs = []
    type_hints = get_type_hints(sample_type)

    for f in fields(sample_type):
        field_type = type_hints.get(f.name, f.type)

        # Check for Optional types (Union with None)
        is_optional = False
        origin = get_origin(field_type)

        if origin is Union or isinstance(field_type, types.UnionType):
            args = get_args(field_type)
            non_none_args = [a for a in args if a is not type(None)]
            if type(None) in args or len(non_none_args) < len(args):
                is_optional = True
            if len(non_none_args) == 1:
                field_type = non_none_args[0]
            elif len(non_none_args) > 1:
                raise TypeError(f"Complex union types not supported: {field_type}")

        field_type_dict = _python_type_to_field_type(field_type)

        field_defs.append({
            "name": f.name,
            "fieldType": field_type_dict,
            "optional": is_optional,
        })

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
    """
    ##

    _name: str
    """Human-readable name for this dataset."""

    _schema_ref: str
    """Reference to the schema for this dataset (local:// path)."""

    _data_urls: list[str]
    """WebDataset URLs for the data."""

    _metadata: dict | None = None
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
            "schema_ref": self._schema_ref,
            "data_urls": self._data_urls,
        }
        return generate_cid(content)

    # IndexEntry protocol properties

    @property
    def name(self) -> str:
        """Human-readable dataset name."""
        return self._name

    @property
    def schema_ref(self) -> str:
        """Reference to the schema for this dataset."""
        return self._schema_ref

    @property
    def data_urls(self) -> list[str]:
        """WebDataset URLs for the data."""
        return self._data_urls

    @property
    def metadata(self) -> dict | None:
        """Arbitrary metadata dictionary, or None if not set."""
        return self._metadata

    # Additional properties

    @property
    def cid(self) -> str:
        """Content identifier (ATProto-compatible CID)."""
        assert self._cid is not None
        return self._cid

    # Legacy compatibility

    @property
    def wds_url(self) -> str:
        """Legacy property: returns first data URL for backwards compatibility."""
        return self._data_urls[0] if self._data_urls else ""

    @property
    def sample_kind(self) -> str:
        """Legacy property: returns schema_ref for backwards compatibility."""
        return self._schema_ref

    def write_to(self, redis: Redis):
        """Persist this index entry to Redis.

        Stores the entry as a Redis hash with key '{REDIS_KEY_DATASET_ENTRY}:{cid}'.

        Args:
            redis: Redis connection to write to.
        """
        save_key = f'{REDIS_KEY_DATASET_ENTRY}:{self.cid}'
        data = {
            'name': self._name,
            'schema_ref': self._schema_ref,
            'data_urls': msgpack.packb(self._data_urls),  # Serialize list
            'cid': self.cid,
        }
        if self._metadata is not None:
            data['metadata'] = msgpack.packb(self._metadata)
        if self._legacy_uuid is not None:
            data['legacy_uuid'] = self._legacy_uuid

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
        save_key = f'{REDIS_KEY_DATASET_ENTRY}:{cid}'
        raw_data = redis.hgetall(save_key)
        if not raw_data:
            raise KeyError(f"{REDIS_KEY_DATASET_ENTRY} not found: {cid}")

        # Decode string fields, keep binary fields as bytes for msgpack
        raw_data_typed = cast(dict[bytes, bytes], raw_data)
        name = raw_data_typed[b'name'].decode('utf-8')
        schema_ref = raw_data_typed[b'schema_ref'].decode('utf-8')
        cid_value = raw_data_typed.get(b'cid', b'').decode('utf-8') or None
        legacy_uuid = raw_data_typed.get(b'legacy_uuid', b'').decode('utf-8') or None

        # Deserialize msgpack fields (stored as raw bytes)
        data_urls = msgpack.unpackb(raw_data_typed[b'data_urls'])
        metadata = None
        if b'metadata' in raw_data_typed:
            metadata = msgpack.unpackb(raw_data_typed[b'metadata'])

        return cls(
            _name=name,
            _schema_ref=schema_ref,
            _data_urls=data_urls,
            _metadata=metadata,
            _cid=cid_value,
            _legacy_uuid=legacy_uuid,
        )


# Backwards compatibility alias
BasicIndexEntry = LocalDatasetEntry

def _s3_env( credentials_path: str | Path ) -> dict[str, Any]:
    """Load S3 credentials (AWS_ENDPOINT, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY) from .env file."""
    credentials_path = Path( credentials_path )
    env_values = dotenv_values( credentials_path )
    assert 'AWS_ENDPOINT' in env_values
    assert 'AWS_ACCESS_KEY_ID' in env_values
    assert 'AWS_SECRET_ACCESS_KEY' in env_values

    return {
        k: env_values[k]
        for k in (
            'AWS_ENDPOINT',
            'AWS_ACCESS_KEY_ID',
            'AWS_SECRET_ACCESS_KEY',
        )
    }

def _s3_from_credentials( creds: str | Path | dict ) -> S3FileSystem:
    """Create S3FileSystem from credentials dict or .env file path."""
    if not isinstance( creds, dict ):
        creds = _s3_env( creds )

    # Build kwargs, making endpoint_url optional
    kwargs = {
        'key': creds['AWS_ACCESS_KEY_ID'],
        'secret': creds['AWS_SECRET_ACCESS_KEY']
    }
    if 'AWS_ENDPOINT' in creds:
        kwargs['endpoint_url'] = creds['AWS_ENDPOINT']

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

    def __init__( self,
                #
                s3_credentials: str | Path | dict[str, Any] | None = None,
                hive_path: str | Path | None = None,
                redis: Redis | None = None,
                #
                #
                **kwargs
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
            **kwargs: Additional arguments (reserved for future use).

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
        elif isinstance( s3_credentials, dict ):
            self.s3_credentials = s3_credentials
        else:
            self.s3_credentials = _s3_env( s3_credentials )

        if self.s3_credentials is None:
            self.bucket_fs = None
        else:
            self.bucket_fs = _s3_from_credentials( self.s3_credentials )

        if self.bucket_fs is not None:
            if hive_path is None:
                raise ValueError( 'Must specify hive path within bucket' )
            self.hive_path = Path( hive_path )
            self.hive_bucket = self.hive_path.parts[0]
        else:
            self.hive_path = None
            self.hive_bucket = None

        #

        self.index = Index( redis = redis )

    ##

    def insert(self,
               ds: Dataset[T],
               *,
               name: str,
               cache_local: bool = False,
               schema_ref: str | None = None,
               **kwargs
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
            raise ValueError("S3 credentials required for insert(). Initialize Repo with s3_credentials.")
        if self.hive_bucket is None or self.hive_path is None:
            raise ValueError("hive_path required for insert(). Initialize Repo with hive_path.")

        new_uuid = str( uuid4() )

        hive_fs = _s3_from_credentials( self.s3_credentials )

        # Write metadata
        metadata_path = (
            self.hive_path
            / 'metadata'
            / f'atdata-metadata--{new_uuid}.msgpack'
        )
        # Note: S3 doesn't need directories created beforehand - s3fs handles this

        if ds.metadata is not None:
            # Use s3:// prefix to ensure s3fs treats this as an S3 path
            with cast( BinaryIO, hive_fs.open( f's3://{metadata_path.as_posix()}', 'wb' ) ) as f:
                meta_packed = msgpack.packb( ds.metadata )
                assert meta_packed is not None
                f.write( cast( bytes, meta_packed ) )


        # Write data
        shard_pattern = (
            self.hive_path
            / f'atdata--{new_uuid}--%06d.tar'
        ).as_posix()

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
        if len( written_shards ) == 0:
            raise RuntimeError( 'Cannot form new dataset entry -- did not write any shards' )
        
        elif len( written_shards ) < 2:
            new_dataset_url = (
                self.hive_path
                / ( Path( written_shards[0] ).name )
            ).as_posix()

        else:
            shard_s3_format = (
                (
                    self.hive_path
                    / f'atdata--{new_uuid}'
                ).as_posix()
            ) + '--{shard_id}.tar'
            shard_id_braced = '{' + f'{0:06d}..{len( written_shards ) - 1:06d}' + '}'
            new_dataset_url = shard_s3_format.format( shard_id = shard_id_braced )

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
    """Redis-backed index for tracking datasets in a repository.

    Implements the AbstractIndex protocol. Maintains a registry of
    LocalDatasetEntry objects in Redis, allowing enumeration and lookup
    of stored datasets.

    When initialized with a data_store, insert_dataset() will write dataset
    shards to storage before indexing. Without a data_store, insert_dataset()
    only indexes existing URLs.

    Attributes:
        _redis: Redis connection for index storage.
        _data_store: Optional AbstractDataStore for writing dataset shards.
    """

    ##

    def __init__(
        self,
        redis: Redis | None = None,
        data_store: AbstractDataStore | None = None,
        **kwargs,
    ) -> None:
        """Initialize an index.

        Args:
            redis: Redis connection to use. If None, creates a new connection
                using the provided kwargs.
            data_store: Optional data store for writing dataset shards.
                If provided, insert_dataset() will write shards to this store.
                If None, insert_dataset() only indexes existing URLs.
            **kwargs: Additional arguments passed to Redis() constructor if
                redis is None.
        """
        ##

        if redis is not None:
            self._redis = redis
        else:
            self._redis: Redis = Redis(**kwargs)

        self._data_store = data_store

    @property
    def data_store(self) -> AbstractDataStore | None:
        """The data store for writing shards, or None if index-only."""
        return self._data_store

    @property
    def all_entries(self) -> list[LocalDatasetEntry]:
        """Get all index entries as a list.

        Returns:
            List of all LocalDatasetEntry objects in the index.
        """
        return list(self.entries)

    @property
    def entries(self) -> Generator[LocalDatasetEntry, None, None]:
        """Iterate over all index entries.

        Scans Redis for LocalDatasetEntry keys and yields them one at a time.

        Yields:
            LocalDatasetEntry objects from the index.
        """
        prefix = f'{REDIS_KEY_DATASET_ENTRY}:'
        for key in self._redis.scan_iter(match=f'{prefix}*'):
            key_str = key.decode('utf-8') if isinstance(key, bytes) else key
            cid = key_str[len(prefix):]
            yield LocalDatasetEntry.from_redis(self._redis, cid)

    def add_entry(self,
                  ds: Dataset,
                  *,
                  name: str,
                  schema_ref: str | None = None,
                  metadata: dict | None = None,
                  ) -> LocalDatasetEntry:
        """Add a dataset to the index.

        Creates a LocalDatasetEntry for the dataset and persists it to Redis.

        Args:
            ds: The dataset to add to the index.
            name: Human-readable name for the dataset.
            schema_ref: Optional schema reference. If None, generates from sample type.
            metadata: Optional metadata dictionary. If None, uses ds._metadata if available.

        Returns:
            The created LocalDatasetEntry object.
        """
        ##
        if schema_ref is None:
            schema_ref = f"local://schemas/{_kind_str_for_sample_type(ds.sample_type)}@1.0.0"

        # Normalize URL to list
        data_urls = [ds.url]

        # Use provided metadata, or fall back to dataset's cached metadata
        # (avoid triggering network requests via ds.metadata property)
        entry_metadata = metadata if metadata is not None else ds._metadata

        entry = LocalDatasetEntry(
            _name=name,
            _schema_ref=schema_ref,
            _data_urls=data_urls,
            _metadata=entry_metadata,
        )

        entry.write_to(self._redis)

        return entry

    def get_entry(self, cid: str) -> LocalDatasetEntry:
        """Get an entry by its CID.

        Args:
            cid: Content identifier of the entry.

        Returns:
            LocalDatasetEntry for the given CID.

        Raises:
            KeyError: If entry not found.
        """
        return LocalDatasetEntry.from_redis(self._redis, cid)

    def get_entry_by_name(self, name: str) -> LocalDatasetEntry:
        """Get an entry by its human-readable name.

        Args:
            name: Human-readable name of the entry.

        Returns:
            LocalDatasetEntry with the given name.

        Raises:
            KeyError: If no entry with that name exists.
        """
        for entry in self.entries:
            if entry.name == name:
                return entry
        raise KeyError(f"No entry with name: {name}")

    # AbstractIndex protocol methods

    def insert_dataset(
        self,
        ds: Dataset,
        *,
        name: str,
        schema_ref: str | None = None,
        **kwargs,
    ) -> LocalDatasetEntry:
        """Insert a dataset into the index (AbstractIndex protocol).

        If a data_store was provided at initialization, writes dataset shards
        to storage first, then indexes the new URLs. Otherwise, indexes the
        dataset's existing URL.

        Args:
            ds: The Dataset to register.
            name: Human-readable name for the dataset.
            schema_ref: Optional schema reference.
            **kwargs: Additional options:
                - metadata: Optional metadata dict
                - prefix: Storage prefix (default: dataset name)
                - cache_local: If True, cache writes locally first

        Returns:
            IndexEntry for the inserted dataset.
        """
        metadata = kwargs.get('metadata')

        if self._data_store is not None:
            # Write shards to data store, then index the new URLs
            prefix = kwargs.get('prefix', name)
            cache_local = kwargs.get('cache_local', False)

            written_urls = self._data_store.write_shards(
                ds,
                prefix=prefix,
                cache_local=cache_local,
            )

            # Generate schema_ref if not provided
            if schema_ref is None:
                schema_ref = _schema_ref_from_type(ds.sample_type)

            # Create entry with the written URLs
            entry_metadata = metadata if metadata is not None else ds._metadata
            entry = LocalDatasetEntry(
                _name=name,
                _schema_ref=schema_ref,
                _data_urls=written_urls,
                _metadata=entry_metadata,
            )
            entry.write_to(self._redis)
            return entry

        # No data store - just index the existing URL
        return self.add_entry(ds, name=name, schema_ref=schema_ref, metadata=metadata)

    def get_dataset(self, ref: str) -> LocalDatasetEntry:
        """Get a dataset entry by name (AbstractIndex protocol).

        Args:
            ref: Dataset name.

        Returns:
            IndexEntry for the dataset.

        Raises:
            KeyError: If dataset not found.
        """
        return self.get_entry_by_name(ref)

    def list_datasets(self) -> Iterator[LocalDatasetEntry]:
        """List all dataset entries (AbstractIndex protocol).

        Yields:
            IndexEntry for each dataset.
        """
        return self.entries

    # Schema operations

    def publish_schema(
        self,
        sample_type: Type[PackableSample],
        *,
        version: str = "1.0.0",
        description: str | None = None,
    ) -> str:
        """Publish a schema for a sample type to Redis.

        Args:
            sample_type: The PackableSample subclass to publish.
            version: Semantic version string (e.g., '1.0.0').
            description: Optional human-readable description.

        Returns:
            Schema reference string: 'local://schemas/{module.Class}@{version}'.

        Raises:
            ValueError: If sample_type is not a dataclass.
            TypeError: If a field type is not supported.
        """
        schema_record = _build_schema_record(
            sample_type,
            version=version,
            description=description,
        )

        schema_ref = _schema_ref_from_type(sample_type, version)
        kind_str, _ = _parse_schema_ref(schema_ref)

        # Store in Redis
        redis_key = f"{REDIS_KEY_SCHEMA}:{kind_str}@{version}"
        schema_json = json.dumps(schema_record)
        self._redis.set(redis_key, schema_json)

        return schema_ref

    def get_schema(self, ref: str) -> dict:
        """Get a schema record by reference.

        Args:
            ref: Schema reference string (local://schemas/...).

        Returns:
            Schema record as a dictionary.

        Raises:
            KeyError: If schema not found.
            ValueError: If reference format is invalid.
        """
        kind_str, version = _parse_schema_ref(ref)
        redis_key = f"{REDIS_KEY_SCHEMA}:{kind_str}@{version}"

        schema_json = self._redis.get(redis_key)
        if schema_json is None:
            raise KeyError(f"Schema not found: {ref}")

        if isinstance(schema_json, bytes):
            schema_json = schema_json.decode('utf-8')

        schema = json.loads(schema_json)
        # Add $ref for decode_schema compatibility
        schema['$ref'] = ref
        return schema

    def list_schemas(self) -> Generator[dict, None, None]:
        """List all schema records in this index.

        Yields:
            Schema records as dictionaries.
        """
        prefix = f'{REDIS_KEY_SCHEMA}:'
        for key in self._redis.scan_iter(match=f'{prefix}*'):
            key_str = key.decode('utf-8') if isinstance(key, bytes) else key
            # Extract kind_str@version from key
            schema_id = key_str[len(prefix):]

            schema_json = self._redis.get(key)
            if schema_json is None:
                continue

            if isinstance(schema_json, bytes):
                schema_json = schema_json.decode('utf-8')

            schema = json.loads(schema_json)
            schema['$ref'] = f"local://schemas/{schema_id}"
            yield schema

    def decode_schema(self, ref: str) -> Type[PackableSample]:
        """Reconstruct a Python PackableSample type from a stored schema.

        This method enables loading datasets without knowing the sample type
        ahead of time. The index retrieves the schema record and dynamically
        generates a PackableSample subclass matching the schema definition.

        Args:
            ref: Schema reference string (local://schemas/...).

        Returns:
            A dynamically generated PackableSample subclass.

        Raises:
            KeyError: If schema not found.
            ValueError: If schema cannot be decoded.
        """
        from atdata._schema_codec import schema_to_type

        schema = self.get_schema(ref)
        return schema_to_type(schema)


# Backwards compatibility alias
LocalIndex = Index


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
        """Resolve an S3 URL for reading.

        For S3, URLs are returned as-is (WebDataset handles s3:// directly).

        Args:
            url: S3 URL to resolve.

        Returns:
            The URL unchanged.
        """
        return url

    def supports_streaming(self) -> bool:
        """S3 supports streaming reads.

        Returns:
            True.
        """
        return True


#