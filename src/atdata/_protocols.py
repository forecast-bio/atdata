"""Protocol definitions for atdata index and storage abstractions.

This module defines the abstract protocols that enable interchangeable
index backends (local Redis vs ATProto PDS) and data stores (S3 vs PDS blobs).

The key insight is that both local and atmosphere implementations solve the
same problem: indexed dataset storage with external data URLs. These protocols
formalize that common interface.

Note:
    Protocol methods use ``...`` (Ellipsis) as the body per PEP 544. This is
    the standard Python syntax for Protocol definitions - these are interface
    specifications, not stub implementations. Concrete classes (LocalIndex,
    AtmosphereIndex, etc.) provide the actual implementations.

Protocols:
    Packable: Structural interface for packable sample types (lens compatibility)
    IndexEntry: Common interface for dataset index entries
    AbstractIndex: Protocol for index operations (schemas, datasets, lenses)
    AbstractDataStore: Protocol for data storage operations

Example:
    >>> def process_datasets(index: AbstractIndex) -> None:
    ...     for entry in index.list_datasets():
    ...         print(f"{entry.name}: {entry.data_urls}")
    ...
    >>> # Works with either LocalIndex or AtmosphereIndex
    >>> process_datasets(local_index)
    >>> process_datasets(atmosphere_index)
"""

from typing import (
    IO,
    Any,
    ClassVar,
    Iterator,
    Optional,
    Protocol,
    Type,
    TYPE_CHECKING,
    runtime_checkable,
)

if TYPE_CHECKING:
    from .dataset import PackableSample, Dataset


##
# Packable Protocol (for lens type compatibility)


@runtime_checkable
class Packable(Protocol):
    """Structural protocol for packable sample types.

    This protocol allows classes decorated with ``@packable`` to be recognized
    as valid types for lens transformations, even though the decorator doesn't
    change the class's nominal type at static analysis time.

    Both ``PackableSample`` subclasses and ``@packable``-decorated classes
    satisfy this protocol structurally.

    Note:
        This protocol is intentionally minimal - it only requires what's needed
        for structural compatibility. The actual ``PackableSample`` class has
        more methods, but lenses only need to know the type is packable.
    """

    @property
    def as_wds(self) -> dict[str, Any]:
        """WebDataset-compatible representation."""
        ...


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
    def name(self) -> str:
        """Human-readable dataset name."""
        ...

    @property
    def schema_ref(self) -> str:
        """Reference to the schema for this dataset.

        For local: 'local://schemas/{module.Class}@{version}'
        For atmosphere: 'at://did:plc:.../ac.foundation.dataset.sampleSchema/...'
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
    def metadata(self) -> Optional[dict]:
        """Arbitrary metadata dictionary, or None if not set."""
        ...


##
# AbstractIndex Protocol


class AbstractIndex(Protocol):
    """Protocol for index operations - implemented by LocalIndex and AtmosphereIndex.

    This protocol defines the common interface for managing dataset metadata:
    - Publishing and retrieving schemas
    - Inserting and listing datasets
    - (Future) Publishing and retrieving lenses

    A single index can hold datasets of many different sample types. The sample
    type is tracked via schema references, not as a generic parameter on the index.

    Example:
        >>> def publish_and_list(index: AbstractIndex) -> None:
        ...     # Publish schemas for different types
        ...     schema1 = index.publish_schema(ImageSample, version="1.0.0")
        ...     schema2 = index.publish_schema(TextSample, version="1.0.0")
        ...
        ...     # Insert datasets of different types
        ...     index.insert_dataset(image_ds, name="images")
        ...     index.insert_dataset(text_ds, name="texts")
        ...
        ...     # List all datasets (mixed types)
        ...     for entry in index.list_datasets():
        ...         print(f"{entry.name} -> {entry.schema_ref}")
    """

    # Dataset operations

    def insert_dataset(
        self,
        ds: "Dataset",
        *,
        name: str,
        schema_ref: Optional[str] = None,
        **kwargs,
    ) -> IndexEntry:
        """Insert a dataset into the index.

        The sample type is inferred from ``ds.sample_type``. If schema_ref is not
        provided, the schema may be auto-published based on the sample type.

        Args:
            ds: The Dataset to register in the index (any sample type).
            name: Human-readable name for the dataset.
            schema_ref: Optional explicit schema reference. If not provided,
                the schema may be auto-published or inferred from ds.sample_type.
            **kwargs: Additional backend-specific options.

        Returns:
            IndexEntry for the inserted dataset.
        """
        ...

    def get_dataset(self, ref: str) -> IndexEntry:
        """Get a dataset entry by name or reference.

        Args:
            ref: Dataset name, path, or full reference string.

        Returns:
            IndexEntry for the dataset.

        Raises:
            KeyError: If dataset not found.
        """
        ...

    def list_datasets(self) -> Iterator[IndexEntry]:
        """List all dataset entries in this index.

        Yields:
            IndexEntry for each dataset (may be of different sample types).
        """
        ...

    # Schema operations

    def publish_schema(
        self,
        sample_type: "Type[PackableSample]",
        *,
        version: str = "1.0.0",
        **kwargs,
    ) -> str:
        """Publish a schema for a sample type.

        Args:
            sample_type: The PackableSample subclass to publish.
            version: Semantic version string for the schema.
            **kwargs: Additional backend-specific options.

        Returns:
            Schema reference string:
            - Local: 'local://schemas/{module.Class}@{version}'
            - Atmosphere: 'at://did:plc:.../ac.foundation.dataset.sampleSchema/...'
        """
        ...

    def get_schema(self, ref: str) -> dict:
        """Get a schema record by reference.

        Args:
            ref: Schema reference string (local:// or at://).

        Returns:
            Schema record as a dictionary with fields like 'name', 'version',
            'fields', etc.

        Raises:
            KeyError: If schema not found.
        """
        ...

    def list_schemas(self) -> Iterator[dict]:
        """List all schema records in this index.

        Yields:
            Schema records as dictionaries.
        """
        ...

    def decode_schema(self, ref: str) -> "Type[PackableSample]":
        """Reconstruct a Python PackableSample type from a stored schema.

        This method enables loading datasets without knowing the sample type
        ahead of time. The index retrieves the schema record and dynamically
        generates a PackableSample subclass matching the schema definition.

        Args:
            ref: Schema reference string (local:// or at://).

        Returns:
            A dynamically generated PackableSample subclass with fields
            matching the schema definition. The class can be used with
            ``Dataset[T]`` to load and iterate over samples.

        Raises:
            KeyError: If schema not found.
            ValueError: If schema cannot be decoded (unsupported field types).

        Example:
            >>> entry = index.get_dataset("my-dataset")
            >>> SampleType = index.decode_schema(entry.schema_ref)
            >>> ds = Dataset[SampleType](entry.data_urls[0])
            >>> for sample in ds.ordered():
            ...     print(sample)  # sample is instance of SampleType
        """
        ...


##
# AbstractDataStore Protocol


class AbstractDataStore(Protocol):
    """Protocol for data storage operations.

    This protocol abstracts over different storage backends for dataset data:
    - S3DataStore: S3-compatible object storage
    - PDSBlobStore: ATProto PDS blob storage (future)

    The separation of index (metadata) from data store (actual files) allows
    flexible deployment: local index with S3 storage, atmosphere index with
    S3 storage, or atmosphere index with PDS blobs.

    Example:
        >>> store = S3DataStore(credentials, bucket="my-bucket")
        >>> urls = store.write_shards(dataset, prefix="training/v1")
        >>> print(urls)
        ['s3://my-bucket/training/v1/shard-000000.tar', ...]
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
            prefix: Path prefix for the shards (e.g., 'datasets/mnist/v1').
            **kwargs: Backend-specific options (e.g., maxcount for shard size).

        Returns:
            List of URLs for the written shards, suitable for use with
            WebDataset or atdata.Dataset().
        """
        ...

    def read_url(self, url: str) -> str:
        """Resolve a storage URL for reading.

        Some storage backends may need to transform URLs (e.g., signing S3 URLs
        or resolving blob references). This method returns a URL that can be
        used directly with WebDataset.

        Args:
            url: Storage URL to resolve.

        Returns:
            WebDataset-compatible URL for reading.
        """
        ...

    def supports_streaming(self) -> bool:
        """Whether this store supports streaming reads.

        Returns:
            True if the store supports efficient streaming (like S3),
            False if data must be fully downloaded first.
        """
        ...


##
# DataSource Protocol


@runtime_checkable
class DataSource(Protocol):
    """Protocol for data sources that provide streams to Dataset.

    A DataSource abstracts over different ways of accessing dataset shards:
    - URLSource: Standard WebDataset-compatible URLs (http, https, pipe, gs, etc.)
    - S3Source: S3-compatible storage with explicit credentials
    - BlobSource: ATProto blob references (future)

    The key method is ``shards()``, which yields (identifier, stream) pairs.
    These are fed directly to WebDataset's tar_file_expander, bypassing URL
    resolution entirely. This enables:
    - Private S3 repos with credentials
    - Custom endpoints (Cloudflare R2, MinIO)
    - ATProto blob streaming
    - Any other source that can provide file-like objects

    Example:
        >>> source = S3Source(
        ...     bucket="my-bucket",
        ...     keys=["data-000.tar", "data-001.tar"],
        ...     endpoint="https://r2.example.com",
        ...     credentials=creds,
        ... )
        >>> ds = Dataset[MySample](source)
        >>> for sample in ds.ordered():
        ...     print(sample)
    """

    def shards(self) -> Iterator[tuple[str, IO[bytes]]]:
        """Yield (identifier, stream) pairs for each shard.

        The identifier is used for error messages and __url__ metadata.
        The stream must be a file-like object that can be read by tarfile.

        Yields:
            Tuple of (shard_identifier, file_like_stream).

        Example:
            >>> for shard_id, stream in source.shards():
            ...     print(f"Processing {shard_id}")
            ...     data = stream.read()
        """
        ...

    @property
    def shard_list(self) -> list[str]:
        """List of shard identifiers without opening streams.

        Used for metadata queries like counting shards without actually
        streaming data. Implementations should return identifiers that
        match what shards() would yield.

        Returns:
            List of shard identifier strings.
        """
        ...

    def open_shard(self, shard_id: str) -> IO[bytes]:
        """Open a single shard by its identifier.

        This method enables random access to individual shards, which is
        required for PyTorch DataLoader worker splitting. Each worker opens
        only its assigned shards rather than iterating all shards.

        Args:
            shard_id: Shard identifier from shard_list.

        Returns:
            File-like stream for reading the shard.

        Raises:
            KeyError: If shard_id is not in shard_list.
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
