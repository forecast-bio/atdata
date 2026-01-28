"""Core dataset and sample infrastructure for typed WebDatasets.

This module provides the core components for working with typed, msgpack-serialized
samples in WebDataset format:

- ``PackableSample``: Base class for msgpack-serializable samples with automatic
  NDArray handling
- ``SampleBatch``: Automatic batching with attribute aggregation
- ``Dataset``: Generic typed dataset wrapper for WebDataset tar files
- ``@packable``: Decorator to convert regular classes into PackableSample subclasses

The implementation handles automatic conversion between numpy arrays and bytes
during serialization, enabling efficient storage of numerical data in WebDataset
archives.

Examples:
    >>> @packable
    ... class ImageSample:
    ...     image: NDArray
    ...     label: str
    ...
    >>> ds = Dataset[ImageSample]("data-{000000..000009}.tar")
    >>> for batch in ds.shuffled(batch_size=32):
    ...     images = batch.image  # Stacked numpy array (32, H, W, C)
    ...     labels = batch.label  # List of 32 strings
"""

##
# Imports

import webdataset as wds

from pathlib import Path
import uuid

import dataclasses
import types
from dataclasses import (
    dataclass,
    asdict,
)
from abc import ABC

from ._sources import URLSource
from ._protocols import DataSource

from tqdm import tqdm
import numpy as np
import pandas as pd
import requests

import typing
from typing import (
    Any,
    Optional,
    Dict,
    Iterator,
    Sequence,
    Iterable,
    Callable,
    Self,
    Generic,
    Type,
    TypeVar,
    TypeAlias,
    dataclass_transform,
    overload,
)
from numpy.typing import NDArray

import msgpack
import ormsgpack
from . import _helpers as eh
from .lens import Lens, LensNetwork


##
# Typing help

Pathlike = str | Path

# WebDataset sample/batch dictionaries (contain __key__, msgpack, etc.)
WDSRawSample: TypeAlias = Dict[str, Any]
WDSRawBatch: TypeAlias = Dict[str, Any]

SampleExportRow: TypeAlias = Dict[str, Any]
SampleExportMap: TypeAlias = Callable[["PackableSample"], SampleExportRow]


##
# Main base classes

DT = TypeVar("DT")


def _make_packable(x):
    """Convert numpy arrays to bytes; pass through other values unchanged."""
    if isinstance(x, np.ndarray):
        return eh.array_to_bytes(x)
    return x


def _is_possibly_ndarray_type(t):
    """Return True if type annotation is NDArray or Optional[NDArray]."""
    if t == NDArray:
        return True
    if isinstance(t, types.UnionType):
        return any(x == NDArray for x in t.__args__)
    return False


class DictSample:
    """Dynamic sample type providing dict-like access to raw msgpack data.

    This class is the default sample type for datasets when no explicit type is
    specified. It stores the raw unpacked msgpack data and provides both
    attribute-style (``sample.field``) and dict-style (``sample["field"]``)
    access to fields.

    ``DictSample`` is useful for:
    - Exploring datasets without defining a schema first
    - Working with datasets that have variable schemas
    - Prototyping before committing to a typed schema

    To convert to a typed schema, use ``Dataset.as_type()`` with a
    ``@packable``-decorated class. Every ``@packable`` class automatically
    registers a lens from ``DictSample``, making this conversion seamless.

    Examples:
        >>> ds = load_dataset("path/to/data.tar")  # Returns Dataset[DictSample]
        >>> for sample in ds.ordered():
        ...     print(sample.some_field)      # Attribute access
        ...     print(sample["other_field"])  # Dict access
        ...     print(sample.keys())          # Inspect available fields
        ...
        >>> # Convert to typed schema
        >>> typed_ds = ds.as_type(MyTypedSample)

    Note:
        NDArray fields are stored as raw bytes in DictSample. They are only
        converted to numpy arrays when accessed through a typed sample class.
    """

    __slots__ = ("_data",)

    def __init__(self, _data: dict[str, Any] | None = None, **kwargs: Any) -> None:
        """Create a DictSample from a dictionary or keyword arguments.

        Args:
            _data: Raw data dictionary. If provided, kwargs are ignored.
            **kwargs: Field values if _data is not provided.
        """
        if _data is not None:
            object.__setattr__(self, "_data", _data)
        else:
            object.__setattr__(self, "_data", kwargs)

    @classmethod
    def from_data(cls, data: dict[str, Any]) -> "DictSample":
        """Create a DictSample from unpacked msgpack data.

        Args:
            data: Dictionary with field names as keys.

        Returns:
            New DictSample instance wrapping the data.
        """
        return cls(_data=data)

    @classmethod
    def from_bytes(cls, bs: bytes) -> "DictSample":
        """Create a DictSample from raw msgpack bytes.

        Args:
            bs: Raw bytes from a msgpack-serialized sample.

        Returns:
            New DictSample instance with the unpacked data.
        """
        return cls.from_data(ormsgpack.unpackb(bs))

    def __getattr__(self, name: str) -> Any:
        """Access a field by attribute name.

        Args:
            name: Field name to access.

        Returns:
            The field value.

        Raises:
            AttributeError: If the field doesn't exist.
        """
        # Avoid infinite recursion for _data lookup
        if name == "_data":
            raise AttributeError(name)
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(
                f"'{type(self).__name__}' has no field '{name}'. "
                f"Available fields: {list(self._data.keys())}"
            ) from None

    def __getitem__(self, key: str) -> Any:
        """Access a field by dict key.

        Args:
            key: Field name to access.

        Returns:
            The field value.

        Raises:
            KeyError: If the field doesn't exist.
        """
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        """Check if a field exists."""
        return key in self._data

    def keys(self) -> list[str]:
        """Return list of field names."""
        return list(self._data.keys())

    def values(self) -> list[Any]:
        """Return list of field values."""
        return list(self._data.values())

    def items(self) -> list[tuple[str, Any]]:
        """Return list of (field_name, value) tuples."""
        return list(self._data.items())

    def get(self, key: str, default: Any = None) -> Any:
        """Get a field value with optional default.

        Args:
            key: Field name to access.
            default: Value to return if field doesn't exist.

        Returns:
            The field value or default.
        """
        return self._data.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        """Return a copy of the underlying data dictionary."""
        return dict(self._data)

    @property
    def packed(self) -> bytes:
        """Pack this sample's data into msgpack bytes.

        Returns:
            Raw msgpack bytes representing this sample's data.
        """
        return msgpack.packb(self._data)

    @property
    def as_wds(self) -> "WDSRawSample":
        """Pack this sample's data for writing to WebDataset.

        Returns:
            A dictionary with ``__key__`` and ``msgpack`` fields.
        """
        return {
            "__key__": str(uuid.uuid1(0, 0)),
            "msgpack": self.packed,
        }

    def __repr__(self) -> str:
        fields = ", ".join(f"{k}=..." for k in self._data.keys())
        return f"DictSample({fields})"


@dataclass
class PackableSample(ABC):
    """Base class for samples that can be serialized with msgpack.

    This abstract base class provides automatic serialization/deserialization
    for dataclass-based samples. Fields annotated as ``NDArray`` or
    ``NDArray | None`` are automatically converted between numpy arrays and
    bytes during packing/unpacking.

    Subclasses should be defined either by:
    1. Direct inheritance with the ``@dataclass`` decorator
    2. Using the ``@packable`` decorator (recommended)

    Examples:
        >>> @packable
        ... class MyData:
        ...     name: str
        ...     embeddings: NDArray
        ...
        >>> sample = MyData(name="test", embeddings=np.array([1.0, 2.0]))
        >>> packed = sample.packed  # Serialize to bytes
        >>> restored = MyData.from_bytes(packed)  # Deserialize
    """

    def _ensure_good(self):
        """Convert bytes to NDArray for fields annotated as NDArray or NDArray | None."""

        # Auto-convert known types when annotated
        # for var_name, var_type in vars( self.__class__ )['__annotations__'].items():
        for field in dataclasses.fields(self):
            var_name = field.name
            var_type = field.type

            # Annotation for this variable is to be an NDArray
            if _is_possibly_ndarray_type(var_type):
                # ... so, we'll always auto-convert to numpy

                var_cur_value = getattr(self, var_name)

                # Execute the appropriate conversion for intermediate data
                # based on what is provided

                if isinstance(var_cur_value, np.ndarray):
                    # Already the correct type, no conversion needed
                    continue

                elif isinstance(var_cur_value, bytes):
                    # Design note: bytes in NDArray-typed fields are always interpreted
                    # as serialized arrays. This means raw bytes fields must not be
                    # annotated as NDArray.
                    setattr(self, var_name, eh.bytes_to_array(var_cur_value))

    def __post_init__(self):
        self._ensure_good()

    ##

    @classmethod
    def from_data(cls, data: WDSRawSample) -> Self:
        """Create a sample instance from unpacked msgpack data.

        Args:
            data: Dictionary with keys matching the sample's field names.

        Returns:
            New instance with NDArray fields auto-converted from bytes.
        """
        return cls(**data)

    @classmethod
    def from_bytes(cls, bs: bytes) -> Self:
        """Create a sample instance from raw msgpack bytes.

        Args:
            bs: Raw bytes from a msgpack-serialized sample.

        Returns:
            A new instance of this sample class deserialized from the bytes.
        """
        return cls.from_data(ormsgpack.unpackb(bs))

    @property
    def packed(self) -> bytes:
        """Pack this sample's data into msgpack bytes.

        NDArray fields are automatically converted to bytes before packing.
        All other fields are packed as-is if they're msgpack-compatible.

        Returns:
            Raw msgpack bytes representing this sample's data.

        Raises:
            RuntimeError: If msgpack serialization fails.
        """

        # Make sure that all of our (possibly unpackable) data is in a packable
        # format
        o = {k: _make_packable(v) for k, v in vars(self).items()}

        ret = msgpack.packb(o)

        if ret is None:
            raise RuntimeError(f"Failed to pack sample to bytes: {o}")

        return ret

    @property
    def as_wds(self) -> WDSRawSample:
        """Pack this sample's data for writing to WebDataset.

        Returns:
            A dictionary with ``__key__`` (UUID v1 for sortable keys) and
            ``msgpack`` (packed sample data) fields suitable for WebDataset.

        Note:
            Keys are auto-generated as UUID v1 for time-sortable ordering.
            Custom key specification is not currently supported.
        """
        return {
            # Generates a UUID that is timelike-sortable
            "__key__": str(uuid.uuid1(0, 0)),
            "msgpack": self.packed,
        }


def _batch_aggregate(xs: Sequence):
    """Stack arrays into numpy array with batch dim; otherwise return list."""
    if not xs:
        return []
    if isinstance(xs[0], np.ndarray):
        return np.array(list(xs))
    return list(xs)


class SampleBatch(Generic[DT]):
    """A batch of samples with automatic attribute aggregation.

    This class wraps a sequence of samples and provides magic ``__getattr__``
    access to aggregate sample attributes. When you access an attribute that
    exists on the sample type, it automatically aggregates values across all
    samples in the batch.

    NDArray fields are stacked into a numpy array with a batch dimension.
    Other fields are aggregated into a list.

    Parameters:
        DT: The sample type, must derive from ``PackableSample``.

    Attributes:
        samples: The list of sample instances in this batch.

    Examples:
        >>> batch = SampleBatch[MyData]([sample1, sample2, sample3])
        >>> batch.embeddings  # Returns stacked numpy array of shape (3, ...)
        >>> batch.names  # Returns list of names

    Note:
        This class uses Python's ``__orig_class__`` mechanism to extract the
        type parameter at runtime. Instances must be created using the
        subscripted syntax ``SampleBatch[MyType](samples)`` rather than
        calling the constructor directly with an unsubscripted class.
    """

    # Design note: The docstring uses "Parameters:" for type parameters because
    # quartodoc doesn't yet support "Type Parameters:" sections in generated docs.

    def __init__(self, samples: Sequence[DT]):
        """Create a batch from a sequence of samples.

        Args:
            samples: A sequence of sample instances to aggregate into a batch.
                Each sample must be an instance of a type derived from
                ``PackableSample``.
        """
        self.samples = list(samples)
        self._aggregate_cache = dict()
        self._sample_type_cache: Type | None = None

    @property
    def sample_type(self) -> Type:
        """The type of each sample in this batch.

        Returns:
            The type parameter ``DT`` used when creating this ``SampleBatch[DT]``.
        """
        if self._sample_type_cache is None:
            self._sample_type_cache = typing.get_args(self.__orig_class__)[0]
            assert self._sample_type_cache is not None
        return self._sample_type_cache

    def __getattr__(self, name):
        """Aggregate an attribute across all samples in the batch.

        This magic method enables attribute-style access to aggregated sample
        fields. Results are cached for efficiency.

        Args:
            name: The attribute name to aggregate across samples.

        Returns:
            For NDArray fields: a stacked numpy array with batch dimension.
            For other fields: a list of values from each sample.

        Raises:
            AttributeError: If the attribute doesn't exist on the sample type.
        """
        # Aggregate named params of sample type
        if name in vars(self.sample_type)["__annotations__"]:
            if name not in self._aggregate_cache:
                self._aggregate_cache[name] = _batch_aggregate(
                    [getattr(x, name) for x in self.samples]
                )

            return self._aggregate_cache[name]

        raise AttributeError(f"No sample attribute named {name}")


ST = TypeVar("ST", bound=PackableSample)
RT = TypeVar("RT", bound=PackableSample)


class _ShardListStage(wds.utils.PipelineStage):
    """Pipeline stage that yields {url: shard_id} dicts from a DataSource.

    This is analogous to SimpleShardList but works with any DataSource.
    Used as the first stage before split_by_worker.
    """

    def __init__(self, source: DataSource):
        self.source = source

    def run(self):
        """Yield {url: shard_id} dicts for each shard."""
        for shard_id in self.source.list_shards():
            yield {"url": shard_id}


class _StreamOpenerStage(wds.utils.PipelineStage):
    """Pipeline stage that opens streams from a DataSource.

    Takes {url: shard_id} dicts and adds a stream using source.open_shard().
    This replaces WebDataset's url_opener stage.
    """

    def __init__(self, source: DataSource):
        self.source = source

    def run(self, src):
        """Open streams for each shard dict."""
        for sample in src:
            shard_id = sample["url"]
            stream = self.source.open_shard(shard_id)
            sample["stream"] = stream
            yield sample


class Dataset(Generic[ST]):
    """A typed dataset built on WebDataset with lens transformations.

    This class wraps WebDataset tar archives and provides type-safe iteration
    over samples of a specific ``PackableSample`` type. Samples are stored as
    msgpack-serialized data within WebDataset shards.

    The dataset supports:
    - Ordered and shuffled iteration
    - Automatic batching with ``SampleBatch``
    - Type transformations via the lens system (``as_type()``)
    - Export to parquet format

    Parameters:
        ST: The sample type for this dataset, must derive from ``PackableSample``.

    Attributes:
        url: WebDataset brace-notation URL for the tar file(s).

    Examples:
        >>> ds = Dataset[MyData]("path/to/data-{000000..000009}.tar")
        >>> for sample in ds.ordered(batch_size=32):
        ...     # sample is SampleBatch[MyData] with batch_size samples
        ...     embeddings = sample.embeddings  # shape: (32, ...)
        ...
        >>> # Transform to a different view
        >>> ds_view = ds.as_type(MyDataView)

    Note:
        This class uses Python's ``__orig_class__`` mechanism to extract the
        type parameter at runtime. Instances must be created using the
        subscripted syntax ``Dataset[MyType](url)`` rather than calling the
        constructor directly with an unsubscripted class.
    """

    # Design note: The docstring uses "Parameters:" for type parameters because
    # quartodoc doesn't yet support "Type Parameters:" sections in generated docs.

    @property
    def sample_type(self) -> Type:
        """The type of each returned sample from this dataset's iterator.

        Returns:
            The type parameter ``ST`` used when creating this ``Dataset[ST]``.
        """
        if self._sample_type_cache is None:
            self._sample_type_cache = typing.get_args(self.__orig_class__)[0]
            assert self._sample_type_cache is not None
        return self._sample_type_cache

    @property
    def batch_type(self) -> Type:
        """The type of batches produced by this dataset.

        Returns:
            ``SampleBatch[ST]`` where ``ST`` is this dataset's sample type.
        """
        return SampleBatch[self.sample_type]

    def __init__(
        self,
        source: DataSource | str | None = None,
        metadata_url: str | None = None,
        *,
        url: str | None = None,
    ) -> None:
        """Create a dataset from a DataSource or URL.

        Args:
            source: Either a DataSource implementation or a WebDataset-compatible
                URL string. If a string is provided, it's wrapped in URLSource
                for backward compatibility.

                Examples:
                    - String URL: ``"path/to/file-{000000..000009}.tar"``
                    - URLSource: ``URLSource("https://example.com/data.tar")``
                    - S3Source: ``S3Source(bucket="my-bucket", keys=["data.tar"])``

            metadata_url: Optional URL to msgpack-encoded metadata for this dataset.
            url: Deprecated. Use ``source`` instead. Kept for backward compatibility.
        """
        super().__init__()

        # Handle backward compatibility: url= keyword argument
        if source is None and url is not None:
            source = url
        elif source is None:
            raise TypeError("Dataset() missing required argument: 'source' or 'url'")

        # Normalize source: strings become URLSource for backward compatibility
        if isinstance(source, str):
            self._source: DataSource = URLSource(source)
            self.url = source
        else:
            self._source = source
            # For compatibility, expose URL if source has list_shards
            shards = source.list_shards()
            # Design note: Using first shard as url for legacy compatibility.
            # Full shard list is available via list_shards() method.
            self.url = shards[0] if shards else ""

        self._metadata: dict[str, Any] | None = None
        self.metadata_url: str | None = metadata_url
        """Optional URL to msgpack-encoded metadata for this dataset."""

        self._output_lens: Lens | None = None
        self._sample_type_cache: Type | None = None

    @property
    def source(self) -> DataSource:
        """The underlying data source for this dataset."""
        return self._source

    def as_type(self, other: Type[RT]) -> "Dataset[RT]":
        """View this dataset through a different sample type using a registered lens.

        Args:
            other: The target sample type to transform into. Must be a type
                derived from ``PackableSample``.

        Returns:
            A new ``Dataset`` instance that yields samples of type ``other``
            by applying the appropriate lens transformation from the global
            ``LensNetwork`` registry.

        Raises:
            ValueError: If no registered lens exists between the current
                sample type and the target type.
        """
        ret = Dataset[other](self._source)
        # Get the singleton lens registry
        lenses = LensNetwork()
        ret._output_lens = lenses.transform(self.sample_type, ret.sample_type)
        return ret

    @property
    def shards(self) -> Iterator[str]:
        """Lazily iterate over shard identifiers.

        Yields:
            Shard identifiers (e.g., 'train-000000.tar', 'train-000001.tar').

        Examples:
            >>> for shard in ds.shards:
            ...     print(f"Processing {shard}")
        """
        return iter(self._source.list_shards())

    def list_shards(self) -> list[str]:
        """Get list of individual dataset shards.

        Returns:
            A full (non-lazy) list of the individual ``tar`` files within the
            source WebDataset.
        """
        return self._source.list_shards()

    # Legacy alias for backwards compatibility
    @property
    def shard_list(self) -> list[str]:
        """List of individual dataset shards (deprecated, use list_shards()).

        .. deprecated::
            Use :meth:`list_shards` instead.
        """
        import warnings

        warnings.warn(
            "shard_list is deprecated, use list_shards() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.list_shards()

    @property
    def metadata(self) -> dict[str, Any] | None:
        """Fetch and cache metadata from metadata_url.

        Returns:
            Deserialized metadata dictionary, or None if no metadata_url is set.

        Raises:
            requests.HTTPError: If metadata fetch fails.
        """
        if self.metadata_url is None:
            return None

        if self._metadata is None:
            with requests.get(self.metadata_url, stream=True) as response:
                response.raise_for_status()
                self._metadata = msgpack.unpackb(response.content, raw=False)

        # Use our cached values
        return self._metadata

    @overload
    def ordered(
        self,
        batch_size: None = None,
    ) -> Iterable[ST]: ...

    @overload
    def ordered(
        self,
        batch_size: int,
    ) -> Iterable[SampleBatch[ST]]: ...

    def ordered(
        self,
        batch_size: int | None = None,
    ) -> Iterable[ST] | Iterable[SampleBatch[ST]]:
        """Iterate over the dataset in order.

        Args:
            batch_size: The size of iterated batches. Default: None (unbatched).
                If ``None``, iterates over one sample at a time with no batch
                dimension.

        Returns:
            A data pipeline that iterates over the dataset in its original
            sample order. When ``batch_size`` is ``None``, yields individual
            samples of type ``ST``. When ``batch_size`` is an integer, yields
            ``SampleBatch[ST]`` instances containing that many samples.

        Examples:
            >>> for sample in ds.ordered():
            ...     process(sample)  # sample is ST
            >>> for batch in ds.ordered(batch_size=32):
            ...     process(batch)  # batch is SampleBatch[ST]
        """
        if batch_size is None:
            return wds.pipeline.DataPipeline(
                _ShardListStage(self._source),
                wds.shardlists.split_by_worker,
                _StreamOpenerStage(self._source),
                wds.tariterators.tar_file_expander,
                wds.tariterators.group_by_keys,
                wds.filters.map(self.wrap),
            )

        return wds.pipeline.DataPipeline(
            _ShardListStage(self._source),
            wds.shardlists.split_by_worker,
            _StreamOpenerStage(self._source),
            wds.tariterators.tar_file_expander,
            wds.tariterators.group_by_keys,
            wds.filters.batched(batch_size),
            wds.filters.map(self.wrap_batch),
        )

    @overload
    def shuffled(
        self,
        buffer_shards: int = 100,
        buffer_samples: int = 10_000,
        batch_size: None = None,
    ) -> Iterable[ST]: ...

    @overload
    def shuffled(
        self,
        buffer_shards: int = 100,
        buffer_samples: int = 10_000,
        *,
        batch_size: int,
    ) -> Iterable[SampleBatch[ST]]: ...

    def shuffled(
        self,
        buffer_shards: int = 100,
        buffer_samples: int = 10_000,
        batch_size: int | None = None,
    ) -> Iterable[ST] | Iterable[SampleBatch[ST]]:
        """Iterate over the dataset in random order.

        Args:
            buffer_shards: Number of shards to buffer for shuffling at the
                shard level. Larger values increase randomness but use more
                memory. Default: 100.
            buffer_samples: Number of samples to buffer for shuffling within
                shards. Larger values increase randomness but use more memory.
                Default: 10,000.
            batch_size: The size of iterated batches. Default: None (unbatched).
                If ``None``, iterates over one sample at a time with no batch
                dimension.

        Returns:
            A data pipeline that iterates over the dataset in randomized order.
            When ``batch_size`` is ``None``, yields individual samples of type
            ``ST``. When ``batch_size`` is an integer, yields ``SampleBatch[ST]``
            instances containing that many samples.

        Examples:
            >>> for sample in ds.shuffled():
            ...     process(sample)  # sample is ST
            >>> for batch in ds.shuffled(batch_size=32):
            ...     process(batch)  # batch is SampleBatch[ST]
        """
        if batch_size is None:
            return wds.pipeline.DataPipeline(
                _ShardListStage(self._source),
                wds.filters.shuffle(buffer_shards),
                wds.shardlists.split_by_worker,
                _StreamOpenerStage(self._source),
                wds.tariterators.tar_file_expander,
                wds.tariterators.group_by_keys,
                wds.filters.shuffle(buffer_samples),
                wds.filters.map(self.wrap),
            )

        return wds.pipeline.DataPipeline(
            _ShardListStage(self._source),
            wds.filters.shuffle(buffer_shards),
            wds.shardlists.split_by_worker,
            _StreamOpenerStage(self._source),
            wds.tariterators.tar_file_expander,
            wds.tariterators.group_by_keys,
            wds.filters.shuffle(buffer_samples),
            wds.filters.batched(batch_size),
            wds.filters.map(self.wrap_batch),
        )

    # Design note: Uses pandas for parquet export. Could be replaced with
    # direct fastparquet calls to reduce dependencies if needed.
    def to_parquet(
        self,
        path: Pathlike,
        sample_map: Optional[SampleExportMap] = None,
        maxcount: Optional[int] = None,
        **kwargs,
    ):
        """Export dataset contents to parquet format.

        Converts all samples to a pandas DataFrame and saves to parquet file(s).
        Useful for interoperability with data analysis tools.

        Args:
            path: Output path for the parquet file. If ``maxcount`` is specified,
                files are named ``{stem}-{segment:06d}.parquet``.
            sample_map: Optional function to convert samples to dictionaries.
                Defaults to ``dataclasses.asdict``.
            maxcount: If specified, split output into multiple files with at most
                this many samples each. Recommended for large datasets.
            **kwargs: Additional arguments passed to ``pandas.DataFrame.to_parquet()``.
                Common options include ``compression``, ``index``, ``engine``.

        Warning:
            **Memory Usage**: When ``maxcount=None`` (default), this method loads
            the **entire dataset into memory** as a pandas DataFrame before writing.
            For large datasets, this can cause memory exhaustion.

            For datasets larger than available RAM, always specify ``maxcount``::

                # Safe for large datasets - processes in chunks
                ds.to_parquet("output.parquet", maxcount=10000)

            This creates multiple parquet files: ``output-000000.parquet``,
            ``output-000001.parquet``, etc.

        Examples:
            >>> ds = Dataset[MySample]("data.tar")
            >>> # Small dataset - load all at once
            >>> ds.to_parquet("output.parquet")
            >>>
            >>> # Large dataset - process in chunks
            >>> ds.to_parquet("output.parquet", maxcount=50000)
        """
        ##

        # Normalize args
        path = Path(path)
        if sample_map is None:
            sample_map = asdict

        verbose = kwargs.get("verbose", False)

        it = self.ordered(batch_size=None)
        if verbose:
            it = tqdm(it)

        #

        if maxcount is None:
            # Load and save full dataset
            df = pd.DataFrame([sample_map(x) for x in self.ordered(batch_size=None)])
            df.to_parquet(path, **kwargs)

        else:
            # Load and save dataset in segments of size `maxcount`

            cur_segment = 0
            cur_buffer = []
            path_template = (
                path.parent / f"{path.stem}-{{:06d}}{path.suffix}"
            ).as_posix()

            for x in self.ordered(batch_size=None):
                cur_buffer.append(sample_map(x))

                if len(cur_buffer) >= maxcount:
                    # Write current segment
                    cur_path = path_template.format(cur_segment)
                    df = pd.DataFrame(cur_buffer)
                    df.to_parquet(cur_path, **kwargs)

                    cur_segment += 1
                    cur_buffer = []

            if len(cur_buffer) > 0:
                # Write one last segment with remainder
                cur_path = path_template.format(cur_segment)
                df = pd.DataFrame(cur_buffer)
                df.to_parquet(cur_path, **kwargs)

    def wrap(self, sample: WDSRawSample) -> ST:
        """Wrap a raw msgpack sample into the appropriate dataset-specific type.

        Args:
            sample: A dictionary containing at minimum a ``'msgpack'`` key with
                serialized sample bytes.

        Returns:
            A deserialized sample of type ``ST``, optionally transformed through
            a lens if ``as_type()`` was called.
        """
        if "msgpack" not in sample:
            raise ValueError(
                f"Sample missing 'msgpack' key, got keys: {list(sample.keys())}"
            )
        if not isinstance(sample["msgpack"], bytes):
            raise ValueError(
                f"Expected sample['msgpack'] to be bytes, got {type(sample['msgpack']).__name__}"
            )

        if self._output_lens is None:
            return self.sample_type.from_bytes(sample["msgpack"])

        source_sample = self._output_lens.source_type.from_bytes(sample["msgpack"])
        return self._output_lens(source_sample)

    def wrap_batch(self, batch: WDSRawBatch) -> SampleBatch[ST]:
        """Wrap a batch of raw msgpack samples into a typed SampleBatch.

        Args:
            batch: A dictionary containing a ``'msgpack'`` key with a list of
                serialized sample bytes.

        Returns:
            A ``SampleBatch[ST]`` containing deserialized samples, optionally
            transformed through a lens if ``as_type()`` was called.

        Note:
            This implementation deserializes samples one at a time, then
            aggregates them into a batch.
        """

        if "msgpack" not in batch:
            raise ValueError(
                f"Batch missing 'msgpack' key, got keys: {list(batch.keys())}"
            )

        if self._output_lens is None:
            batch_unpacked = [
                self.sample_type.from_bytes(bs) for bs in batch["msgpack"]
            ]
            return SampleBatch[self.sample_type](batch_unpacked)

        batch_source = [
            self._output_lens.source_type.from_bytes(bs) for bs in batch["msgpack"]
        ]
        batch_view = [self._output_lens(s) for s in batch_source]
        return SampleBatch[self.sample_type](batch_view)


_T = TypeVar("_T")


@dataclass_transform()
def packable(cls: type[_T]) -> type[PackableSample]:
    """Decorator to convert a regular class into a ``PackableSample``.

    This decorator transforms a class into a dataclass that inherits from
    ``PackableSample``, enabling automatic msgpack serialization/deserialization
    with special handling for NDArray fields.

    The resulting class satisfies the ``Packable`` protocol, making it compatible
    with all atdata APIs that accept packable types (e.g., ``publish_schema``,
    lens transformations, etc.).

    Type Checking:
        The return type is annotated as ``type[PackableSample]`` so that IDEs
        and type checkers recognize the ``PackableSample`` methods (``packed``,
        ``as_wds``, ``from_bytes``, etc.). The ``@dataclass_transform()``
        decorator ensures that field access from the original class is also
        preserved for type checking.

    Args:
        cls: The class to convert. Should have type annotations for its fields.

    Returns:
        A new dataclass that inherits from ``PackableSample`` with the same
        name and annotations as the original class. The class satisfies the
        ``Packable`` protocol and can be used with ``Type[Packable]`` signatures.

    Examples:
        >>> @packable
        ... class MyData:
        ...     name: str
        ...     values: NDArray
        ...
        >>> sample = MyData(name="test", values=np.array([1, 2, 3]))
        >>> bytes_data = sample.packed
        >>> restored = MyData.from_bytes(bytes_data)
        >>>
        >>> # Works with Packable-typed APIs
        >>> index.publish_schema(MyData, version="1.0.0")  # Type-safe
    """

    ##

    class_name = cls.__name__
    class_annotations = cls.__annotations__

    # Add in dataclass niceness to original class
    as_dataclass = dataclass(cls)

    # This triggers a bunch of behind-the-scenes stuff for the newly annotated class
    @dataclass
    class as_packable(as_dataclass, PackableSample):
        def __post_init__(self):
            return PackableSample.__post_init__(self)

    # Restore original class identity for better repr/debugging
    as_packable.__name__ = class_name
    as_packable.__qualname__ = class_name
    as_packable.__module__ = cls.__module__
    as_packable.__annotations__ = class_annotations
    if cls.__doc__:
        as_packable.__doc__ = cls.__doc__

    # Fix qualnames of dataclass-generated methods so they don't show
    # 'packable.<locals>.as_packable' in help() and IDE hints
    old_qualname_prefix = "packable.<locals>.as_packable"
    for attr_name in ("__init__", "__repr__", "__eq__", "__post_init__"):
        attr = getattr(as_packable, attr_name, None)
        if attr is not None and hasattr(attr, "__qualname__"):
            if attr.__qualname__.startswith(old_qualname_prefix):
                attr.__qualname__ = attr.__qualname__.replace(
                    old_qualname_prefix, class_name, 1
                )

    # Auto-register lens from DictSample to this type
    # This enables ds.as_type(MyType) when ds is Dataset[DictSample]
    def _dict_to_typed(ds: DictSample) -> as_packable:
        return as_packable.from_data(ds._data)

    _dict_lens = Lens(_dict_to_typed)
    LensNetwork().register(_dict_lens)

    ##

    return as_packable
