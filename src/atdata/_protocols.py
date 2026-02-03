"""Protocol definitions for atdata index and storage abstractions.

Defines the abstract protocols that enable interchangeable index backends
(local SQLite/Redis vs ATProto PDS) and data stores (S3, local disk, PDS blobs).

Protocols:
    Packable: Structural interface for packable sample types
    IndexEntry: Common interface for dataset index entries
    AbstractIndex: Protocol for index operations (schemas, datasets, lenses)
    AbstractDataStore: Protocol for data storage operations
    DataSource: Protocol for streaming shard data

Examples:
    >>> def process_datasets(index: AbstractIndex) -> None:
    ...     for entry in index.list_datasets():
    ...         print(f"{entry.name}: {entry.data_urls}")
"""

from typing import (
    IO,
    Any,
    Iterable,
    Iterator,
    Optional,
    Protocol,
    Type,
    TYPE_CHECKING,
    runtime_checkable,
)

if TYPE_CHECKING:
    from .dataset import Dataset


##
# Packable Protocol (for lens type compatibility)


@runtime_checkable
class Packable(Protocol):
    """Structural protocol for packable sample types.

    This protocol allows classes decorated with ``@packable`` to be recognized
    as valid types for lens transformations and schema operations, even though
    the decorator doesn't change the class's nominal type at static analysis time.

    Both ``PackableSample`` subclasses and ``@packable``-decorated classes
    satisfy this protocol structurally.

    The protocol captures the full interface needed for:
    - Lens type transformations (as_wds, from_data)
    - Schema publishing (class introspection via dataclass fields)
    - Serialization/deserialization (packed, from_bytes)

    Examples:
        >>> @packable
        ... class MySample:
        ...     name: str
        ...     value: int
        ...
        >>> def process(sample_type: Type[Packable]) -> None:
        ...     # Type checker knows sample_type has from_bytes, packed, etc.
        ...     instance = sample_type.from_bytes(data)
        ...     print(instance.packed)
    """

    @classmethod
    def from_data(cls, data: dict[str, Any]) -> "Packable": ...

    @classmethod
    def from_bytes(cls, bs: bytes) -> "Packable": ...

    @property
    def packed(self) -> bytes: ...

    @property
    def as_wds(self) -> dict[str, Any]: ...


##
# IndexEntry Protocol


@runtime_checkable
class IndexEntry(Protocol):
    """Common interface for index entries (local or atmosphere).

    Both LocalDatasetEntry and atmosphere DatasetRecord-based entries
    should satisfy this protocol, enabling code that works with either.

    Properties:
        name: Human-readable dataset name
        schema_ref: Reference to schema (local:// path or AT URI)
        data_urls: WebDataset URLs for the data
        metadata: Arbitrary metadata dict, or None
    """

    @property
    def name(self) -> str: ...

    @property
    def schema_ref(self) -> str:
        """Schema reference string.

        Local: ``local://schemas/{module.Class}@{version}``
        Atmosphere: ``at://did:plc:.../ac.foundation.dataset.schema/...``
        """
        ...

    @property
    def data_urls(self) -> list[str]:
        """WebDataset URLs for the data.

        These are the URLs that can be passed to atdata.Dataset() or
        used with WebDataset directly. May use brace notation for shards.
        """
        ...

    @property
    def metadata(self) -> Optional[dict]: ...


##
# AbstractIndex Protocol


class AbstractIndex(Protocol):
    """Protocol for index operations — implemented by Index and AtmosphereIndex.

    Manages dataset metadata: publishing/retrieving schemas, inserting/listing
    datasets. A single index holds datasets of many sample types, tracked via
    schema references.

    Examples:
        >>> def publish_and_list(index: AbstractIndex) -> None:
        ...     index.publish_schema(ImageSample, version="1.0.0")
        ...     index.insert_dataset(image_ds, name="images")
        ...     for entry in index.list_datasets():
        ...         print(f"{entry.name} -> {entry.schema_ref}")
    """

    @property
    def data_store(self) -> Optional["AbstractDataStore"]:
        """Optional data store for reading/writing shards.

        If present, ``load_dataset`` uses it for credential resolution.
        Not all implementations provide a data_store; check with
        ``getattr(index, 'data_store', None)``.
        """
        ...

    # Dataset operations

    def write(
        self,
        samples: Iterable,
        *,
        name: str,
        schema_ref: Optional[str] = None,
        **kwargs,
    ) -> IndexEntry:
        """Write samples and create an index entry in one step.

        Serializes samples to WebDataset tar files, stores them via the
        appropriate backend, and creates an index entry.

        Args:
            samples: Iterable of Packable samples. Must be non-empty.
            name: Dataset name, optionally prefixed with target backend.
            schema_ref: Optional schema reference.
            **kwargs: Backend-specific options (maxcount, description, etc.).

        Returns:
            IndexEntry for the created dataset.
        """
        ...

    def insert_dataset(
        self,
        ds: "Dataset",
        *,
        name: str,
        schema_ref: Optional[str] = None,
        **kwargs,
    ) -> IndexEntry:
        """Register an existing dataset in the index.

        Args:
            ds: The Dataset to register.
            name: Human-readable name.
            schema_ref: Explicit schema ref; auto-published if ``None``.
            **kwargs: Backend-specific options.
        """
        ...

    def get_dataset(self, ref: str) -> IndexEntry:
        """Get a dataset entry by name or reference.

        Raises:
            KeyError: If dataset not found.
        """
        ...

    @property
    def datasets(self) -> Iterator[IndexEntry]: ...

    def list_datasets(self) -> list[IndexEntry]: ...

    # Schema operations

    def publish_schema(
        self,
        sample_type: type,
        *,
        version: str = "1.0.0",
        **kwargs,
    ) -> str:
        """Publish a schema for a sample type.

        Args:
            sample_type: A Packable type (``@packable``-decorated or subclass).
            version: Semantic version string.
            **kwargs: Backend-specific options.

        Returns:
            Schema reference string (``local://...`` or ``at://...``).
        """
        ...

    def get_schema(self, ref: str) -> dict:
        """Get a schema record by reference.

        Raises:
            KeyError: If schema not found.
        """
        ...

    @property
    def schemas(self) -> Iterator[dict]: ...

    def list_schemas(self) -> list[dict]: ...

    def decode_schema(self, ref: str) -> Type[Packable]:
        """Reconstruct a Packable type from a stored schema.

        Raises:
            KeyError: If schema not found.
            ValueError: If schema has unsupported field types.

        Examples:
            >>> SampleType = index.decode_schema(entry.schema_ref)
            >>> ds = Dataset[SampleType](entry.data_urls[0])
        """
        ...


##
# AbstractDataStore Protocol


class AbstractDataStore(Protocol):
    """Protocol for data storage backends (S3, local disk, PDS blobs).

    Separates index (metadata) from data store (shard files), enabling
    flexible deployment combinations.

    Examples:
        >>> store = S3DataStore(credentials, bucket="my-bucket")
        >>> urls = store.write_shards(dataset, prefix="training/v1")
    """

    def write_shards(
        self,
        ds: "Dataset",
        *,
        prefix: str,
        **kwargs,
    ) -> list[str]:
        """Write dataset shards to storage.

        Args:
            ds: The Dataset to write.
            prefix: Path prefix (e.g., ``'datasets/mnist/v1'``).
            **kwargs: Backend-specific options (``maxcount``, ``maxsize``, etc.).

        Returns:
            List of shard URLs suitable for ``atdata.Dataset()``.
        """
        ...

    def read_url(self, url: str) -> str:
        """Resolve a storage URL for reading (e.g., sign S3 URLs)."""
        ...

    def supports_streaming(self) -> bool: ...


##
# DataSource Protocol


@runtime_checkable
class DataSource(Protocol):
    """Protocol for data sources that stream shard data to Dataset.

    Implementations (URLSource, S3Source, BlobSource) yield
    ``(identifier, stream)`` pairs fed to WebDataset's tar expander,
    bypassing URL resolution. This enables private S3, custom endpoints,
    and ATProto blob streaming.

    Examples:
        >>> source = S3Source(bucket="my-bucket", keys=["data-000.tar"])
        >>> ds = Dataset[MySample](source)
    """

    @property
    def shards(self) -> Iterator[tuple[str, IO[bytes]]]:
        """Lazily yield ``(shard_id, stream)`` pairs for each shard."""
        ...

    def list_shards(self) -> list[str]:
        """Shard identifiers without opening streams."""
        ...

    def open_shard(self, shard_id: str) -> IO[bytes]:
        """Open a single shard for random access (e.g., DataLoader splitting).

        Raises:
            KeyError: If *shard_id* is not in ``list_shards()``.
        """
        ...


##
# Module exports

__all__ = [
    "Packable",
    "IndexEntry",
    "AbstractIndex",
    "AbstractDataStore",
    "DataSource",
]
