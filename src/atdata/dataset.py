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
import itertools
import uuid

import dataclasses
import types
from dataclasses import (
    dataclass,
    asdict,
)
from abc import ABC

from ._sources import URLSource
from ._protocols import DataSource, Packable
from ._exceptions import SampleKeyError, PartialFailureError, LensNotFoundError

import numpy as np

import typing
from typing import (
    TYPE_CHECKING,
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

if TYPE_CHECKING:
    import pandas
    import pandas as pd
    from .manifest._proxy import Predicate
    from .manifest._query import SampleLocation
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
    """Convert numpy arrays to bytes; coerce numpy scalars to Python natives."""
    if isinstance(x, np.ndarray):
        return eh.array_to_bytes(x)
    if isinstance(x, np.generic):
        return x.item()
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
        """Create a DictSample from unpacked msgpack data."""
        return cls(_data=data)

    @classmethod
    def from_bytes(cls, bs: bytes) -> "DictSample":
        """Create a DictSample from raw msgpack bytes."""
        return cls.from_data(ormsgpack.unpackb(bs))

    def __getattr__(self, name: str) -> Any:
        """Access a field by attribute name.

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
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def keys(self) -> list[str]:
        """Return list of field names."""
        return list(self._data.keys())

    def values(self) -> list[Any]:
        return list(self._data.values())

    def items(self) -> list[tuple[str, Any]]:
        return list(self._data.items())

    def get(self, key: str, default: Any = None) -> Any:
        """Get a field value, returning *default* if missing."""
        return self._data.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        """Return a copy of the underlying data dictionary."""
        return dict(self._data)

    @property
    def packed(self) -> bytes:
        """Serialize to msgpack bytes."""
        return msgpack.packb(self._data)

    @property
    def as_wds(self) -> "WDSRawSample":
        """Serialize for writing to WebDataset (``__key__`` + ``msgpack``)."""
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
        for field in dataclasses.fields(self):
            if _is_possibly_ndarray_type(field.type):
                value = getattr(self, field.name)
                if isinstance(value, np.ndarray):
                    continue
                elif isinstance(value, bytes):
                    setattr(self, field.name, eh.bytes_to_array(value))

    def __post_init__(self):
        self._ensure_good()

    ##

    @classmethod
    def from_data(cls, data: WDSRawSample) -> Self:
        """Create an instance from unpacked msgpack data."""
        return cls(**data)

    @classmethod
    def from_bytes(cls, bs: bytes) -> Self:
        """Create an instance from raw msgpack bytes."""
        return cls.from_data(ormsgpack.unpackb(bs))

    @property
    def packed(self) -> bytes:
        """Serialize to msgpack bytes. NDArray fields are auto-converted."""
        o = {k: _make_packable(v) for k, v in vars(self).items()}
        return msgpack.packb(o)

    @property
    def as_wds(self) -> WDSRawSample:
        """Serialize for writing to WebDataset (``__key__`` + ``msgpack``)."""
        return {
            "__key__": str(uuid.uuid1(0, 0)),
            "msgpack": self.packed,
        }


def _batch_aggregate(xs: Sequence):
    """Stack arrays into numpy array with batch dim; otherwise return list."""
    if not xs:
        return []
    if isinstance(xs[0], np.ndarray):
        return np.stack(xs)
    return list(xs)


class SampleBatch(Generic[DT]):
    """A batch of samples with automatic attribute aggregation.

    Accessing an attribute aggregates that field across all samples:
    NDArray fields are stacked into a numpy array with a batch dimension;
    other fields are collected into a list. Results are cached.

    Parameters:
        DT: The sample type, must derive from ``PackableSample``.

    Examples:
        >>> batch = SampleBatch[MyData]([sample1, sample2, sample3])
        >>> batch.embeddings  # Stacked numpy array of shape (3, ...)
        >>> batch.names  # List of names
    """

    def __init__(self, samples: Sequence[DT]):
        """Create a batch from a sequence of samples."""
        self.samples = list(samples)
        self._aggregate_cache = dict()
        self._sample_type_cache: Type | None = None

    @property
    def sample_type(self) -> Type:
        """The type parameter ``DT`` used when creating this batch."""
        if self._sample_type_cache is None:
            self._sample_type_cache = typing.get_args(self.__orig_class__)[0]
            if self._sample_type_cache is None:
                raise TypeError(
                    "SampleBatch requires a type parameter, e.g. SampleBatch[MySample]"
                )
        return self._sample_type_cache

    def __getattr__(self, name):
        """Aggregate a field across all samples (cached)."""
        # Aggregate named params of sample type
        if name in vars(self.sample_type)["__annotations__"]:
            if name not in self._aggregate_cache:
                self._aggregate_cache[name] = _batch_aggregate(
                    [getattr(x, name) for x in self.samples]
                )

            return self._aggregate_cache[name]

        raise AttributeError(f"No sample attribute named {name}")


ST = TypeVar("ST", bound=Packable)
RT = TypeVar("RT", bound=Packable)


def _make_structural_lens(
    source_type: Type,
    target_type: Type,
) -> "Lens | None":
    """Create a field-mapping lens if source and target types are structurally compatible.

    Two types are structurally compatible when the target's required fields
    are a subset of the source's fields (by name). This enables ``as_type()``
    to work between a dynamically-generated schema type and a user-defined
    type with the same field layout, without requiring an explicit lens
    registration.

    Returns ``None`` if the types are not compatible.
    """
    if not dataclasses.is_dataclass(source_type) or not dataclasses.is_dataclass(
        target_type
    ):
        # DictSample -> typed: @packable auto-registers a lens for this, so
        # this branch is a safety net for target types that weren't decorated.
        if source_type is DictSample and dataclasses.is_dataclass(target_type):

            def _dict_convert(src: DictSample):
                return target_type.from_data(src._data)

            result: Lens = object.__new__(Lens)
            result._getter = _dict_convert
            result._putter = lambda v, s: s
            result.source_type = source_type
            result.view_type = target_type
            return result
        return None

    source_fields = {f.name for f in dataclasses.fields(source_type)}
    target_fields = {f.name for f in dataclasses.fields(target_type)}

    # Target fields must be a subset of source fields
    if not target_fields.issubset(source_fields):
        return None

    field_names = [f.name for f in dataclasses.fields(target_type)]

    def _structural_get(src):
        kwargs = {name: getattr(src, name) for name in field_names}
        return target_type(**kwargs)

    result = object.__new__(Lens)
    result._getter = _structural_get
    result._putter = lambda v, s: s
    result.source_type = source_type
    result.view_type = target_type
    return result


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
        """The type parameter ``ST`` used when creating this dataset."""
        if self._sample_type_cache is None:
            self._sample_type_cache = typing.get_args(self.__orig_class__)[0]
            if self._sample_type_cache is None:
                raise TypeError(
                    "Dataset requires a type parameter, e.g. Dataset[MySample]"
                )
        return self._sample_type_cache

    @property
    def batch_type(self) -> Type:
        """``SampleBatch[ST]`` where ``ST`` is this dataset's sample type."""
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

        if source is None and url is not None:
            source = url
        elif source is None:
            raise TypeError("Dataset() missing required argument: 'source' or 'url'")

        if isinstance(source, str):
            self._source: DataSource = URLSource(source)
            self.url = source
        else:
            self._source = source
            shards = source.list_shards()
            self.url = shards[0] if shards else ""

        self._metadata: dict[str, Any] | None = None
        self.metadata_url: str | None = metadata_url
        self._output_lens: Lens | None = None
        self._sample_type_cache: Type | None = None
        self._content_metadata: "Packable | dict[str, Any] | None" = None

    @property
    def source(self) -> DataSource:
        """The underlying data source for this dataset."""
        return self._source

    def as_type(self, other: Type[RT]) -> "Dataset[RT]":
        """View this dataset through a different sample type via a registered lens.

        Falls back to structural field mapping when no lens is registered but the
        source and target types have compatible field names.

        Raises:
            LensNotFoundError: If no lens exists and types are not structurally
                compatible.
        """
        ret = Dataset[other](self._source)
        lenses = LensNetwork()
        try:
            ret._output_lens = lenses.transform(self.sample_type, ret.sample_type)
        except LensNotFoundError:
            structural = _make_structural_lens(self.sample_type, ret.sample_type)
            if structural is None:
                raise
            ret._output_lens = structural
        return ret

    @property
    def shards(self) -> Iterator[str]:
        """Lazily iterate over shard identifiers."""
        return iter(self._source.list_shards())

    def list_shards(self) -> list[str]:
        """Return all shard paths/URLs as a list."""
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
        """Fetch and cache metadata from metadata_url, or ``None`` if unset."""
        if self.metadata_url is None:
            return None

        if self._metadata is None:
            import requests

            with requests.get(self.metadata_url, stream=True, timeout=30) as response:
                response.raise_for_status()
                self._metadata = msgpack.unpackb(response.content, raw=False)

        # Use our cached values
        return self._metadata

    @property
    def content_metadata(self) -> "Packable | dict[str, Any] | None":
        """Dataset-level content metadata (e.g., instrument settings).

        Returns a ``Packable`` instance if typed metadata was set via
        ``write_samples(..., content_metadata=MyMetadata(...))``, a plain
        ``dict`` if untyped metadata was provided, or ``None`` if absent.

        Examples:
            >>> ds = write_samples(samples, "out.tar", content_metadata=meta)
            >>> ds.content_metadata
            MyMetadata(instrument='Zeiss LSM 880', ...)
        """
        return self._content_metadata

    @content_metadata.setter
    def content_metadata(self, value: "Packable | dict[str, Any] | None") -> None:
        self._content_metadata = value

    ##
    # Convenience methods (GH#38 developer experience)

    @property
    def schema(self) -> dict[str, type]:
        """Field names and types for this dataset's sample type.

        Examples:
            >>> ds = Dataset[MyData]("data.tar")
            >>> ds.schema
            {'name': <class 'str'>, 'embedding': numpy.ndarray}
        """
        st = self.sample_type
        if st is DictSample:
            return {"_data": dict}
        if dataclasses.is_dataclass(st):
            return {f.name: f.type for f in dataclasses.fields(st)}
        return {}

    @property
    def column_names(self) -> list[str]:
        """List of field names for this dataset's sample type."""
        st = self.sample_type
        if dataclasses.is_dataclass(st):
            return [f.name for f in dataclasses.fields(st)]
        return []

    def __iter__(self) -> Iterator[ST]:
        """Shorthand for ``ds.ordered()``."""
        return iter(self.ordered())

    def __len__(self) -> int:
        """Total sample count (iterates all shards on first call, then cached)."""
        if not hasattr(self, "_len_cache"):
            self._len_cache: int = sum(1 for _ in self.ordered())
        return self._len_cache

    def head(self, n: int = 5) -> list[ST]:
        """Return the first *n* samples from the dataset.

        Args:
            n: Number of samples to return. Default: 5.

        Returns:
            List of up to *n* samples in shard order.

        Examples:
            >>> samples = ds.head(3)
            >>> len(samples)
            3
        """
        return list(itertools.islice(self.ordered(), n))

    def get(self, key: str) -> ST:
        """Retrieve a single sample by its ``__key__``.

        Scans shards sequentially until a sample with a matching key is found.
        This is O(n) for streaming datasets.

        Args:
            key: The WebDataset ``__key__`` string to search for.

        Returns:
            The matching sample.

        Raises:
            SampleKeyError: If no sample with the given key exists.

        Examples:
            >>> sample = ds.get("00000001-0001-1000-8000-010000000000")
        """
        pipeline = wds.pipeline.DataPipeline(
            _ShardListStage(self._source),
            wds.shardlists.split_by_worker,
            _StreamOpenerStage(self._source),
            wds.tariterators.tar_file_expander,
            wds.tariterators.group_by_keys,
        )
        for raw_sample in pipeline:
            if raw_sample.get("__key__") == key:
                return self.wrap(raw_sample)
        raise SampleKeyError(key)

    def describe(self) -> dict[str, Any]:
        """Summary statistics: sample_type, fields, num_shards, shards, url, metadata."""
        shards = self.list_shards()
        return {
            "sample_type": self.sample_type.__name__,
            "fields": self.schema,
            "num_shards": len(shards),
            "shards": shards,
            "url": self.url,
            "metadata": self.metadata,
        }

    def filter(self, predicate: Callable[[ST], bool]) -> "Dataset[ST]":
        """Return a new dataset that yields only samples matching *predicate*.

        The filter is applied lazily during iteration — no data is copied.

        Args:
            predicate: A function that takes a sample and returns ``True``
                to keep it or ``False`` to discard it.

        Returns:
            A new ``Dataset`` whose iterators apply the filter.

        Examples:
            >>> long_names = ds.filter(lambda s: len(s.name) > 10)
            >>> for sample in long_names:
            ...     assert len(sample.name) > 10
        """
        filtered = Dataset[self.sample_type](self._source, self.metadata_url)
        filtered._sample_type_cache = self._sample_type_cache
        filtered._output_lens = self._output_lens
        filtered._filter_fn = predicate
        # Preserve any existing filters
        parent_filters = getattr(self, "_filter_fn", None)
        if parent_filters is not None:
            outer = parent_filters
            filtered._filter_fn = lambda s: outer(s) and predicate(s)
        # Preserve any existing map
        if hasattr(self, "_map_fn"):
            filtered._map_fn = self._map_fn
        return filtered

    def map(self, fn: Callable[[ST], Any]) -> "Dataset":
        """Return a new dataset that applies *fn* to each sample during iteration.

        The mapping is applied lazily during iteration — no data is copied.

        Args:
            fn: A function that takes a sample of type ``ST`` and returns
                a transformed value.

        Returns:
            A new ``Dataset`` whose iterators apply the mapping.

        Examples:
            >>> names = ds.map(lambda s: s.name)
            >>> for name in names:
            ...     print(name)
        """
        mapped = Dataset[self.sample_type](self._source, self.metadata_url)
        mapped._sample_type_cache = self._sample_type_cache
        mapped._output_lens = self._output_lens
        mapped._map_fn = fn
        # Preserve any existing map
        if hasattr(self, "_map_fn"):
            outer = self._map_fn
            mapped._map_fn = lambda s: fn(outer(s))
        # Preserve any existing filter
        if hasattr(self, "_filter_fn"):
            mapped._filter_fn = self._filter_fn
        return mapped

    def process_shards(
        self,
        fn: Callable[[list[ST]], Any],
        *,
        shards: list[str] | None = None,
        checkpoint: Path | str | None = None,
        on_shard_error: Callable[[str, Exception], None] | None = None,
    ) -> dict[str, Any]:
        """Process each shard independently, collecting per-shard results.

        Unlike :meth:`map` (which is lazy and per-sample), this method eagerly
        processes each shard in turn, calling *fn* with the full list of samples
        from that shard. If some shards fail, raises
        :class:`~atdata._exceptions.PartialFailureError` containing both the
        successful results and the per-shard errors.

        Args:
            fn: Function receiving a list of samples from one shard and
                returning an arbitrary result.
            shards: Optional list of shard identifiers to process. If ``None``,
                processes all shards in the dataset. Useful for retrying only
                the failed shards from a previous ``PartialFailureError``.
            checkpoint: Optional path to a checkpoint file. If provided,
                already-succeeded shard IDs are loaded from this file and
                skipped. Each newly succeeded shard is appended. On full
                success the file is deleted. On partial failure it remains
                for resume.
            on_shard_error: Optional callback invoked as
                ``on_shard_error(shard_id, exception)`` for each failed shard,
                enabling dead-letter logging or alerting.

        Returns:
            Dict mapping shard identifier to *fn*'s return value for each shard.

        Raises:
            PartialFailureError: If at least one shard fails. The exception
                carries ``.succeeded_shards``, ``.failed_shards``, ``.errors``,
                and ``.results`` for inspection and retry.

        Examples:
            >>> results = ds.process_shards(lambda samples: len(samples))
            >>> # On partial failure, retry just the failed shards:
            >>> try:
            ...     results = ds.process_shards(expensive_fn)
            ... except PartialFailureError as e:
            ...     retry = ds.process_shards(expensive_fn, shards=e.failed_shards)

            >>> # With checkpoint for crash recovery:
            >>> results = ds.process_shards(expensive_fn, checkpoint="progress.txt")
        """
        from ._logging import get_logger, log_operation

        log = get_logger()
        shard_ids = shards or self.list_shards()

        # Load checkpoint: skip already-succeeded shards
        checkpoint_path: Path | None = None
        if checkpoint is not None:
            checkpoint_path = Path(checkpoint)
            if checkpoint_path.exists():
                already_done = set(checkpoint_path.read_text().splitlines())
                log.info(
                    "process_shards: loaded checkpoint, %d shards already done",
                    len(already_done),
                )
                shard_ids = [s for s in shard_ids if s not in already_done]
                if not shard_ids:
                    log.info("process_shards: all shards already checkpointed")
                    return {}

        succeeded: list[str] = []
        failed: list[str] = []
        errors: dict[str, Exception] = {}
        results: dict[str, Any] = {}

        with log_operation("process_shards", total_shards=len(shard_ids)):
            for shard_id in shard_ids:
                try:
                    shard_ds = Dataset[self.sample_type](shard_id)
                    shard_ds._sample_type_cache = self._sample_type_cache
                    samples = list(shard_ds.ordered())
                    results[shard_id] = fn(samples)
                    succeeded.append(shard_id)
                    log.debug("process_shards: shard ok %s", shard_id)
                    if checkpoint_path is not None:
                        with open(checkpoint_path, "a") as f:
                            f.write(shard_id + "\n")
                except Exception as exc:
                    failed.append(shard_id)
                    errors[shard_id] = exc
                    log.warning("process_shards: shard failed %s: %s", shard_id, exc)
                    if on_shard_error is not None:
                        on_shard_error(shard_id, exc)

            if failed:
                raise PartialFailureError(
                    succeeded_shards=succeeded,
                    failed_shards=failed,
                    errors=errors,
                    results=results,
                )

        # All shards succeeded; clean up checkpoint file
        if checkpoint_path is not None and checkpoint_path.exists():
            checkpoint_path.unlink()
            log.debug("process_shards: checkpoint file removed (all shards done)")

        return results

    def select(self, indices: Sequence[int]) -> list[ST]:
        """Return samples at the given integer indices.

        Iterates through the dataset in order and collects samples whose
        positional index matches. This is O(n) for streaming datasets.

        Args:
            indices: Sequence of zero-based indices to select.

        Returns:
            List of samples at the requested positions, in index order.

        Examples:
            >>> samples = ds.select([0, 5, 10])
            >>> len(samples)
            3
        """
        if not indices:
            return []
        target = set(indices)
        max_idx = max(indices)
        result: dict[int, ST] = {}
        count = 0
        for count, sample in enumerate(self.ordered()):
            if count in target:
                result[count] = sample
            if count >= max_idx:
                break
        missing = [idx for idx in indices if idx not in result]
        if missing:
            total = count + 1 if result or count > 0 else 0
            raise IndexError(
                f"Indices {missing} not found in dataset (dataset has {total} samples)"
            )
        return [result[i] for i in indices]

    @property
    def fields(self) -> "Any":
        """Typed field proxy for manifest queries on this dataset.

        Returns an object whose attributes are ``FieldProxy`` instances,
        one per manifest-eligible field of this dataset's sample type.

        Examples:
            >>> ds = atdata.Dataset[MySample](url)
            >>> Q = ds.fields
            >>> results = ds.query(where=(Q.confidence > 0.9))
        """
        from .manifest._proxy import query_fields

        return query_fields(self.sample_type)

    def query(
        self,
        where: "Callable[[pd.DataFrame], pd.Series] | Predicate",
    ) -> "list[SampleLocation]":
        """Query this dataset using per-shard manifest metadata.

        Requires manifests to have been generated during shard writing.
        Discovers manifest files alongside the tar shards, loads them,
        and executes a two-phase query (shard-level aggregate pruning,
        then sample-level parquet filtering).

        The *where* argument accepts either a lambda/function that operates
        on a pandas DataFrame, or a ``Predicate`` built from the proxy DSL.

        Args:
            where: Predicate function or ``Predicate`` object that selects
                matching rows from the per-sample manifest DataFrame.

        Returns:
            List of ``SampleLocation`` for matching samples.

        Raises:
            FileNotFoundError: If no manifest files are found alongside shards.

        Examples:
            >>> locs = ds.query(where=lambda df: df["confidence"] > 0.9)
            >>> len(locs)
            42

            >>> Q = ds.fields
            >>> locs = ds.query(where=(Q.confidence > 0.9))
        """
        from .manifest import QueryExecutor

        shard_urls = self.list_shards()
        executor = QueryExecutor.from_shard_urls(shard_urls)
        return executor.query(where=where)

    def to_pandas(self, limit: int | None = None) -> "pandas.DataFrame":
        """Materialize the dataset (or first *limit* samples) as a DataFrame.

        Args:
            limit: Maximum number of samples to include. ``None`` means all
                samples (may use significant memory for large datasets).

        Returns:
            A pandas DataFrame with one row per sample and columns matching
            the sample fields.

        Warning:
            With ``limit=None`` this loads the entire dataset into memory.

        Examples:
            >>> df = ds.to_pandas(limit=100)
            >>> df.columns.tolist()
            ['name', 'embedding']
        """
        samples = self.head(limit) if limit is not None else list(self.ordered())
        rows = [
            asdict(s) if dataclasses.is_dataclass(s) else s.to_dict() for s in samples
        ]
        import pandas as pd

        return pd.DataFrame(rows)

    def to_dict(self, limit: int | None = None) -> dict[str, list[Any]]:
        """Materialize the dataset as a column-oriented dictionary.

        Args:
            limit: Maximum number of samples to include. ``None`` means all.

        Returns:
            Dictionary mapping field names to lists of values (one entry
            per sample).

        Warning:
            With ``limit=None`` this loads the entire dataset into memory.

        Examples:
            >>> d = ds.to_dict(limit=10)
            >>> d.keys()
            dict_keys(['name', 'embedding'])
            >>> len(d['name'])
            10
        """
        samples = self.head(limit) if limit is not None else list(self.ordered())
        if not samples:
            return {}
        if dataclasses.is_dataclass(samples[0]):
            fields = [f.name for f in dataclasses.fields(samples[0])]
            return {f: [getattr(s, f) for s in samples] for f in fields}
        # DictSample path
        keys = samples[0].keys()
        return {k: [s[k] for s in samples] for k in keys}

    def _post_wrap_stages(self) -> list:
        """Build extra pipeline stages for filter/map set via .filter()/.map()."""
        stages: list = []
        filter_fn = getattr(self, "_filter_fn", None)
        if filter_fn is not None:
            stages.append(wds.filters.select(filter_fn))
        map_fn = getattr(self, "_map_fn", None)
        if map_fn is not None:
            stages.append(wds.filters.map(map_fn))
        return stages

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
                *self._post_wrap_stages(),
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
                *self._post_wrap_stages(),
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
        """Export dataset to parquet file(s).

        Args:
            path: Output path. With *maxcount*, files are named
                ``{stem}-{segment:06d}.parquet``.
            sample_map: Convert sample to dict. Defaults to ``dataclasses.asdict``.
            maxcount: Split into files of at most this many samples.
                Without it, the entire dataset is loaded into memory.
            **kwargs: Passed to ``pandas.DataFrame.to_parquet()``.

        Examples:
            >>> ds.to_parquet("output.parquet", maxcount=50000)
        """
        import pandas as pd

        path = Path(path)
        if sample_map is None:
            sample_map = asdict

        if maxcount is None:
            df = pd.DataFrame([sample_map(x) for x in self.ordered(batch_size=None)])
            df.to_parquet(path, **kwargs)
        else:
            cur_segment = 0
            cur_buffer: list = []
            path_template = (
                path.parent / f"{path.stem}-{{:06d}}{path.suffix}"
            ).as_posix()

            for x in self.ordered(batch_size=None):
                cur_buffer.append(sample_map(x))
                if len(cur_buffer) >= maxcount:
                    cur_path = path_template.format(cur_segment)
                    pd.DataFrame(cur_buffer).to_parquet(cur_path, **kwargs)
                    cur_segment += 1
                    cur_buffer = []

            if cur_buffer:
                cur_path = path_template.format(cur_segment)
                pd.DataFrame(cur_buffer).to_parquet(cur_path, **kwargs)

    def wrap(self, sample: WDSRawSample) -> ST:
        """Deserialize a raw WDS sample dict into type ``ST``."""
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
        """Deserialize a raw WDS batch dict into ``SampleBatch[ST]``."""

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
def packable(cls: type[_T]) -> type[_T]:
    """Convert a class into a ``PackableSample`` dataclass with msgpack serialization.

    The resulting class gains ``packed``, ``as_wds``, ``from_bytes``, and
    ``from_data`` methods, and satisfies the ``Packable`` protocol.
    NDArray fields are automatically handled during serialization.

    Examples:
        >>> @packable
        ... class MyData:
        ...     name: str
        ...     values: NDArray
        ...
        >>> sample = MyData(name="test", values=np.array([1, 2, 3]))
        >>> restored = MyData.from_bytes(sample.packed)
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


# ---------------------------------------------------------------------------
# write_samples — convenience function for writing samples to tar files
# ---------------------------------------------------------------------------


def write_samples(
    samples: Iterable[ST],
    path: str | Path,
    *,
    maxcount: int | None = None,
    maxsize: int | None = None,
    manifest: bool = False,
    content_metadata: "Packable | dict[str, Any] | None" = None,
) -> "Dataset[ST]":
    """Write an iterable of samples to WebDataset tar file(s).

    Args:
        samples: Iterable of ``PackableSample`` instances. Must be non-empty.
        path: Output path for the tar file. For sharded output (when
            *maxcount* or *maxsize* is set), a ``%06d`` pattern is
            auto-appended if the path does not already contain ``%``.
        maxcount: Maximum samples per shard. Triggers multi-shard output.
        maxsize: Maximum bytes per shard. Triggers multi-shard output.
        manifest: If True, write per-shard manifest sidecar files
            (``.manifest.json`` + ``.manifest.parquet``) alongside each
            tar file. Manifests enable metadata queries via
            ``QueryExecutor`` without opening the tars.
        content_metadata: Optional dataset-level content metadata. Accepts
            either a ``Packable`` instance (typed, schema-derivable) or a
            plain ``dict`` (validated at ATProto publish time). When a
            ``dict`` is provided, keys must be JSON-serializable.

    Returns:
        A ``Dataset`` wrapping the written file(s), typed to the sample
        type of the input samples. The returned dataset's
        ``content_metadata`` property is set to the provided value.

    Raises:
        ValueError: If *samples* is empty.
        TypeError: If *content_metadata* is not a Packable, dict, or None.

    Examples:
        >>> samples = [MySample(key="0", text="hello")]
        >>> ds = write_samples(samples, "out.tar")
        >>> list(ds.ordered())
        [MySample(key='0', text='hello')]

        >>> ds = write_samples(samples, "out.tar", content_metadata={"instrument": "Zeiss"})
        >>> ds.content_metadata
        {'instrument': 'Zeiss'}
    """
    # Validate content_metadata type early, before writing any files.
    if content_metadata is not None:
        if not isinstance(content_metadata, (dict, Packable)):
            raise TypeError(
                f"content_metadata must be a Packable instance or dict, "
                f"got {type(content_metadata).__name__}"
            )

    from ._hf_api import _shards_to_wds_url
    from ._logging import get_logger, log_operation

    if manifest:
        from .manifest._builder import ManifestBuilder
        from .manifest._writer import ManifestWriter

    log = get_logger()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    use_shard_writer = maxcount is not None or maxsize is not None
    sample_type: type | None = None
    written_paths: list[str] = []

    with log_operation(
        "write_samples", path=str(path), sharded=use_shard_writer, manifest=manifest
    ):
        # Manifest tracking state
        _current_builder: list = []  # single-element list for nonlocal mutation
        _builders: list[tuple[str, "ManifestBuilder"]] = []
        _running_offset: list[int] = [0]

        def _finalize_builder() -> None:
            """Finalize the current manifest builder and stash it."""
            if _current_builder:
                shard_path = written_paths[-1] if written_paths else ""
                _builders.append((shard_path, _current_builder[0]))
                _current_builder.clear()

        def _start_builder(shard_path: str) -> None:
            """Start a new manifest builder for a shard."""
            _finalize_builder()
            shard_id = Path(shard_path).stem
            _current_builder.append(
                ManifestBuilder(sample_type=sample_type, shard_id=shard_id)
            )
            _running_offset[0] = 0

        def _record_sample(sample: "PackableSample", wds_dict: dict) -> None:
            """Record a sample in the active manifest builder."""
            if not _current_builder:
                return
            packed_bytes = wds_dict["msgpack"]
            size = len(packed_bytes)
            _current_builder[0].add_sample(
                key=wds_dict["__key__"],
                offset=_running_offset[0],
                size=size,
                sample=sample,
            )
            _running_offset[0] += size

        if use_shard_writer:
            # Build shard pattern from path
            if "%" not in str(path):
                pattern = str(path.parent / f"{path.stem}-%06d{path.suffix}")
            else:
                pattern = str(path)

            writer_kwargs: dict[str, Any] = {}
            if maxcount is not None:
                writer_kwargs["maxcount"] = maxcount
            if maxsize is not None:
                writer_kwargs["maxsize"] = maxsize

            def _track(p: str) -> None:
                written_paths.append(str(Path(p).resolve()))
                if manifest and sample_type is not None:
                    _start_builder(p)

            with wds.writer.ShardWriter(pattern, post=_track, **writer_kwargs) as sink:
                for sample in samples:
                    if sample_type is None:
                        sample_type = type(sample)
                    wds_dict = sample.as_wds
                    sink.write(wds_dict)
                    if manifest:
                        # The first sample triggers _track before we get here when
                        # ShardWriter opens the first shard, but just in case:
                        if not _current_builder and sample_type is not None:
                            _start_builder(str(path))
                        _record_sample(sample, wds_dict)
        else:
            with wds.writer.TarWriter(str(path)) as sink:
                for sample in samples:
                    if sample_type is None:
                        sample_type = type(sample)
                    wds_dict = sample.as_wds
                    sink.write(wds_dict)
                    if manifest:
                        if not _current_builder and sample_type is not None:
                            _current_builder.append(
                                ManifestBuilder(
                                    sample_type=sample_type, shard_id=path.stem
                                )
                            )
                        _record_sample(sample, wds_dict)
            written_paths.append(str(path.resolve()))

        if sample_type is None:
            raise ValueError("samples must be non-empty")

        # Finalize and write manifests
        if manifest:
            _finalize_builder()
            for shard_path, builder in _builders:
                m = builder.build()
                base = str(Path(shard_path).with_suffix(""))
                writer = ManifestWriter(base)
                writer.write(m)

        log.info(
            "write_samples: wrote %d shard(s), sample_type=%s",
            len(written_paths),
            sample_type.__name__,
        )

    url = _shards_to_wds_url(written_paths)
    ds: Dataset = Dataset(url)
    ds._sample_type_cache = sample_type
    ds._content_metadata = content_metadata
    return ds
