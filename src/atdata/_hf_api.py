"""HuggingFace Datasets-style API for atdata.

This module provides a familiar `load_dataset()` interface inspired by the
HuggingFace Datasets library, adapted for atdata's typed WebDataset approach.

Key differences from HuggingFace Datasets:
- Requires explicit `sample_type` parameter (typed dataclass)
- Returns atdata.Dataset[ST] instead of HF Dataset
- Built on WebDataset for efficient streaming of large datasets
- No Arrow caching layer (WebDataset handles remote/local transparently)

Examples:
    >>> import atdata
    >>> from atdata import load_dataset
    >>>
    >>> @atdata.packable
    ... class MyData:
    ...     text: str
    ...     label: int
    >>>
    >>> # Load a single split
    >>> ds = load_dataset("path/to/train-{000000..000099}.tar", MyData, split="train")
    >>>
    >>> # Load all splits (returns DatasetDict)
    >>> ds_dict = load_dataset("path/to/{train,test}-*.tar", MyData)
    >>> train_ds = ds_dict["train"]
"""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Generic,
    Mapping,
    Optional,
    Type,
    TypeVar,
    overload,
)

from .dataset import Dataset, DictSample
from ._sources import URLSource, S3Source
from ._protocols import DataSource, Packable

if TYPE_CHECKING:
    from ._protocols import AbstractIndex

##
# Type variables

ST = TypeVar("ST", bound=Packable)


##
# Default Index singleton

_default_index: "Index | None" = None  # noqa: F821 (forward ref)
_default_index_lock = threading.Lock()


def get_default_index() -> "Index":  # noqa: F821
    """Get or create the module-level default Index.

    The default Index uses Redis for local storage (backwards-compatible
    default) and an anonymous AtmosphereClient for read-only public data
    resolution.

    The default is created lazily on first access and cached for the
    lifetime of the process.

    Returns:
        The default Index instance.

    Examples:
        >>> index = get_default_index()
        >>> entry = index.get_dataset("local/mnist")
    """
    global _default_index
    if _default_index is None:
        with _default_index_lock:
            if _default_index is None:
                from .local import Index

                _default_index = Index()
    return _default_index


def set_default_index(index: "Index") -> None:  # noqa: F821
    """Override the module-level default Index.

    Use this to configure a custom default Index with specific repositories,
    an authenticated atmosphere client, or non-default providers.

    Args:
        index: The Index instance to use as the default.

    Examples:
        >>> from atdata.local import Index
        >>> from atdata.providers import create_provider
        >>> custom = Index(provider=create_provider("sqlite"))
        >>> set_default_index(custom)
    """
    global _default_index
    _default_index = index


##
# DatasetDict - container for multiple splits


class DatasetDict(Generic[ST], dict):
    """A dictionary of split names to Dataset instances.

    Similar to HuggingFace's DatasetDict, this provides a container for
    multiple dataset splits (train, test, validation, etc.) with convenience
    methods that operate across all splits.

    Parameters:
        ST: The sample type for all datasets in this dict.

    Examples:
        >>> ds_dict = load_dataset("path/to/data", MyData)
        >>> train = ds_dict["train"]
        >>> test = ds_dict["test"]
        >>>
        >>> # Iterate over all splits
        >>> for split_name, dataset in ds_dict.items():
        ...     print(f"{split_name}: {len(dataset.list_shards())} shards")
    """

    # Note: The docstring uses "Parameters:" for type parameters as a workaround
    # for quartodoc not supporting "Type Parameters:" sections.

    def __init__(
        self,
        splits: Mapping[str, Dataset[ST]] | None = None,
        sample_type: Type[ST] | None = None,
        streaming: bool = False,
    ) -> None:
        """Create a DatasetDict from a mapping of split names to datasets.

        Args:
            splits: Mapping of split names to Dataset instances.
            sample_type: The sample type for datasets in this dict. If not
                provided, inferred from the first dataset in splits.
            streaming: Whether this DatasetDict was loaded in streaming mode.
        """
        super().__init__(splits or {})
        self._sample_type = sample_type
        self._streaming = streaming

    @property
    def sample_type(self) -> Type[ST] | None:
        """The sample type for datasets in this dict."""
        if self._sample_type is not None:
            return self._sample_type
        # Infer from first dataset
        if self:
            first_ds = next(iter(self.values()))
            return first_ds.sample_type
        return None

    def __getitem__(self, key: str) -> Dataset[ST]:
        """Get a dataset by split name."""
        return super().__getitem__(key)

    def __setitem__(self, key: str, value: Dataset[ST]) -> None:
        """Set a dataset for a split name."""
        super().__setitem__(key, value)

    @property
    def streaming(self) -> bool:
        """Whether this DatasetDict was loaded in streaming mode."""
        return self._streaming

    @property
    def num_shards(self) -> dict[str, int]:
        """Number of shards in each split.

        Returns:
            Dict mapping split names to shard counts.

        Note:
            This property accesses the shard list, which may trigger
            shard enumeration for remote datasets.
        """
        return {name: len(ds.list_shards()) for name, ds in self.items()}


##
# Path resolution utilities


def _is_brace_pattern(path: str) -> bool:
    """Check if path contains WebDataset brace expansion notation like {000..099}."""
    return bool(re.search(r"\{[^}]+\}", path))


def _is_glob_pattern(path: str) -> bool:
    """Check if path contains glob wildcards (* or ?)."""
    return "*" in path or "?" in path


def _is_remote_url(path: str) -> bool:
    """Check if path is a remote URL (s3://, gs://, http://, https://, az://)."""
    return path.startswith(("s3://", "gs://", "http://", "https://", "az://"))


def _expand_local_glob(pattern: str) -> list[str]:
    """Expand local glob pattern to sorted list of matching file paths."""
    base_path = Path(pattern).parent
    glob_part = Path(pattern).name

    if not base_path.exists():
        return []

    matches = sorted(base_path.glob(glob_part))
    return [str(p) for p in matches if p.is_file()]


# Pre-compiled split name patterns (pattern, split_name)
_SPLIT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Patterns like "dataset-train-000000.tar" (split in middle with delimiters)
    (re.compile(r"[_-](train|training)[_-]"), "train"),
    (re.compile(r"[_-](test|testing)[_-]"), "test"),
    (re.compile(r"[_-](val|valid|validation)[_-]"), "validation"),
    (re.compile(r"[_-](dev|development)[_-]"), "validation"),
    # Patterns at start of filename like "train-000.tar" or "test_data.tar"
    (re.compile(r"^(train|training)[_-]"), "train"),
    (re.compile(r"^(test|testing)[_-]"), "test"),
    (re.compile(r"^(val|valid|validation)[_-]"), "validation"),
    (re.compile(r"^(dev|development)[_-]"), "validation"),
    # Patterns in directory path like "/path/train/shard-000.tar"
    (re.compile(r"[/\\](train|training)[/\\]"), "train"),
    (re.compile(r"[/\\](test|testing)[/\\]"), "test"),
    (re.compile(r"[/\\](val|valid|validation)[/\\]"), "validation"),
    (re.compile(r"[/\\](dev|development)[/\\]"), "validation"),
    # Patterns at start of path like "train/shard-000.tar"
    (re.compile(r"^(train|training)[/\\]"), "train"),
    (re.compile(r"^(test|testing)[/\\]"), "test"),
    (re.compile(r"^(val|valid|validation)[/\\]"), "validation"),
    (re.compile(r"^(dev|development)[/\\]"), "validation"),
]


def _detect_split_from_path(path: str) -> str | None:
    """Detect split name (train/test/validation) from file path."""
    filename = Path(path).name
    path_lower = path.lower()
    filename_lower = filename.lower()

    # Check filename first (more specific)
    for pattern, split_name in _SPLIT_PATTERNS:
        if pattern.search(filename_lower):
            return split_name

    # Fall back to full path (catches directory patterns)
    for pattern, split_name in _SPLIT_PATTERNS:
        if pattern.search(path_lower):
            return split_name

    return None


def _resolve_shards(
    path: str,
    data_files: str | list[str] | dict[str, str | list[str]] | None = None,
) -> dict[str, list[str]]:
    """Resolve path specification to dict of split -> shard URLs.

    Handles:
    - WebDataset brace notation: "path/{train,test}-{000..099}.tar"
    - Glob patterns: "path/*.tar"
    - Explicit data_files mapping

    Args:
        path: Base path or pattern.
        data_files: Optional explicit mapping of splits to files.

    Returns:
        Dict mapping split names to lists of shard URLs.
    """
    # If explicit data_files provided, use those
    if data_files is not None:
        return _resolve_data_files(path, data_files)

    # WebDataset brace notation - pass through as-is
    # WebDataset handles expansion internally
    if _is_brace_pattern(path):
        # Try to detect split from the pattern itself
        split = _detect_split_from_path(path)
        split_name = split or "train"
        return {split_name: [path]}

    # Local glob pattern
    if not _is_remote_url(path) and _is_glob_pattern(path):
        shards = _expand_local_glob(path)
        return _group_shards_by_split(shards)

    # Local directory - scan for .tar files
    if not _is_remote_url(path) and Path(path).is_dir():
        shards = _expand_local_glob(str(Path(path) / "*.tar"))
        return _group_shards_by_split(shards)

    # Single file or remote URL - treat as single shard
    split = _detect_split_from_path(path)
    split_name = split or "train"
    return {split_name: [path]}


def _resolve_data_files(
    base_path: str,
    data_files: str | list[str] | dict[str, str | list[str]],
) -> dict[str, list[str]]:
    """Resolve explicit data_files specification.

    Args:
        base_path: Base path for relative file references.
        data_files: File specification - can be:
            - str: Single file pattern
            - list[str]: List of file patterns
            - dict[str, ...]: Mapping of split names to patterns

    Returns:
        Dict mapping split names to lists of resolved file paths.
    """
    base = Path(base_path) if not _is_remote_url(base_path) else None

    if isinstance(data_files, str):
        # Single pattern -> "train" split
        if base and not Path(data_files).is_absolute():
            data_files = str(base / data_files)
        return {"train": [data_files]}

    if isinstance(data_files, list):
        # List of patterns -> "train" split
        resolved = []
        for f in data_files:
            if base and not Path(f).is_absolute():
                f = str(base / f)
            resolved.append(f)
        return {"train": resolved}

    # Dict mapping splits to patterns
    result: dict[str, list[str]] = {}
    for split_name, files in data_files.items():
        if isinstance(files, str):
            files = [files]
        resolved = []
        for f in files:
            if base and not Path(f).is_absolute():
                f = str(base / f)
            resolved.append(f)
        result[split_name] = resolved

    return result


def _shards_to_wds_url(shards: list[str]) -> str:
    """Convert a list of shard paths to a WebDataset URL.

    WebDataset supports brace expansion, so we convert multiple shards
    into brace notation when they share a common prefix/suffix.

    Args:
        shards: List of shard file paths.

    Returns:
        WebDataset-compatible URL string.

    Examples:
        >>> _shards_to_wds_url(["data-000.tar", "data-001.tar", "data-002.tar"])
        "data-{000,001,002}.tar"
        >>> _shards_to_wds_url(["train.tar"])
        "train.tar"
    """
    import os.path

    if len(shards) == 0:
        raise ValueError("Cannot create URL from empty shard list")

    if len(shards) == 1:
        return shards[0]

    # Find common prefix using os.path.commonprefix (O(n) vs O(n²))
    prefix = os.path.commonprefix(shards)

    # Find common suffix by reversing strings
    reversed_shards = [s[::-1] for s in shards]
    suffix = os.path.commonprefix(reversed_shards)[::-1]

    prefix_len = len(prefix)
    suffix_len = len(suffix)

    # Ensure prefix and suffix don't overlap
    min_shard_len = min(len(s) for s in shards)
    if prefix_len + suffix_len > min_shard_len:
        # Overlapping - prefer prefix, reduce suffix
        suffix_len = max(0, min_shard_len - prefix_len)
        suffix = shards[0][-suffix_len:] if suffix_len > 0 else ""

    if prefix_len > 0 or suffix_len > 0:
        # Extract the varying middle parts
        middles = []
        for s in shards:
            if suffix_len > 0:
                middle = s[prefix_len:-suffix_len]
            else:
                middle = s[prefix_len:]
            middles.append(middle)

        # Only use brace notation if we have meaningful variation
        if all(middles):
            return f"{prefix}{{{','.join(middles)}}}{suffix}"

    # Fallback: space-separated URLs for WebDataset
    return " ".join(shards)


def _group_shards_by_split(shards: list[str]) -> dict[str, list[str]]:
    """Group a list of shard paths by detected split.

    Args:
        shards: List of shard file paths.

    Returns:
        Dict mapping split names to lists of shards. Files with no
        detected split are placed in "train".
    """
    result: dict[str, list[str]] = {}

    for shard in shards:
        split = _detect_split_from_path(shard)
        split_name = split or "train"
        if split_name not in result:
            result[split_name] = []
        result[split_name].append(shard)

    return result


##
# Index-based path resolution


def _is_indexed_path(path: str) -> bool:
    """Check if path uses @handle/dataset notation for index lookup.

    Examples:
        >>> _is_indexed_path("@maxine.science/mnist")
        True
        >>> _is_indexed_path("@did:plc:abc123/my-dataset")
        True
        >>> _is_indexed_path("s3://bucket/data.tar")
        False
    """
    return path.startswith("@")


def _parse_indexed_path(path: str) -> tuple[str, str]:
    """Parse @handle/dataset path into (handle_or_did, dataset_name).

    Args:
        path: Path in format "@handle/dataset" or "@did:plc:xxx/dataset"

    Returns:
        Tuple of (handle_or_did, dataset_name)

    Raises:
        ValueError: If path format is invalid.
    """
    if not path.startswith("@"):
        raise ValueError(f"Not an indexed path: {path}")

    # Remove leading @
    rest = path[1:]

    # Split on first / (handle can contain . but dataset name is after /)
    if "/" not in rest:
        raise ValueError(
            f"Invalid indexed path format: {path}. "
            "Expected @handle/dataset or @did:plc:xxx/dataset"
        )

    # Find the split point - for DIDs, the format is did:plc:xxx/dataset
    # For handles, it's handle.domain/dataset
    parts = rest.split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Invalid indexed path: {path}")

    return parts[0], parts[1]


def _resolve_indexed_path(
    path: str,
    index: "AbstractIndex",
) -> tuple[DataSource, str]:
    """Resolve @handle/dataset path to DataSource and schema_ref via index lookup.

    Args:
        path: Path in @handle/dataset format.
        index: Index to use for lookup.

    Returns:
        Tuple of (DataSource, schema_ref). The DataSource is configured with
        appropriate credentials when the index has an S3DataStore.

    Raises:
        KeyError: If dataset not found in index.
    """
    handle_or_did, dataset_name = _parse_indexed_path(path)

    # For AtmosphereIndex, we need to resolve handle to DID first
    # For local Index, the handle is ignored and we just look up by name
    entry = index.get_dataset(dataset_name)
    data_urls = entry.data_urls

    # Check if index has a data store
    if hasattr(index, "data_store") and index.data_store is not None:
        store = index.data_store

        # Import here to avoid circular imports at module level
        from .local import S3DataStore

        # For S3DataStore with S3 URLs, create S3Source with credentials
        if isinstance(store, S3DataStore):
            if data_urls and all(url.startswith("s3://") for url in data_urls):
                source = S3Source.from_urls(
                    data_urls,
                    endpoint=store.credentials.get("AWS_ENDPOINT"),
                    access_key=store.credentials.get("AWS_ACCESS_KEY_ID"),
                    secret_key=store.credentials.get("AWS_SECRET_ACCESS_KEY"),
                    region=store.credentials.get("AWS_REGION"),
                )
                return source, entry.schema_ref

        # For any data store, use read_url to transform URLs if needed
        # (handles endpoint URL conversion for HTTPS access, etc.)
        transformed_urls = [store.read_url(url) for url in data_urls]
        url = _shards_to_wds_url(transformed_urls)
        return URLSource(url), entry.schema_ref

    # Default: URL-based source without credentials
    url = _shards_to_wds_url(data_urls)
    return URLSource(url), entry.schema_ref


##
# Main load_dataset function


# Overload: explicit type with split -> Dataset[ST]
@overload
def load_dataset(
    path: str,
    sample_type: Type[ST],
    *,
    split: str,
    data_files: str | list[str] | dict[str, str | list[str]] | None = None,
    streaming: bool = False,
    index: Optional["AbstractIndex"] = None,
) -> Dataset[ST]: ...


# Overload: explicit type without split -> DatasetDict[ST]
@overload
def load_dataset(
    path: str,
    sample_type: Type[ST],
    *,
    split: None = None,
    data_files: str | list[str] | dict[str, str | list[str]] | None = None,
    streaming: bool = False,
    index: Optional["AbstractIndex"] = None,
) -> DatasetDict[ST]: ...


# Overload: no type with split -> Dataset[DictSample]
@overload
def load_dataset(
    path: str,
    sample_type: None = None,
    *,
    split: str,
    data_files: str | list[str] | dict[str, str | list[str]] | None = None,
    streaming: bool = False,
    index: Optional["AbstractIndex"] = None,
) -> Dataset[DictSample]: ...


# Overload: no type without split -> DatasetDict[DictSample]
@overload
def load_dataset(
    path: str,
    sample_type: None = None,
    *,
    split: None = None,
    data_files: str | list[str] | dict[str, str | list[str]] | None = None,
    streaming: bool = False,
    index: Optional["AbstractIndex"] = None,
) -> DatasetDict[DictSample]: ...


def load_dataset(
    path: str,
    sample_type: Type[ST] | None = None,
    *,
    split: str | None = None,
    data_files: str | list[str] | dict[str, str | list[str]] | None = None,
    streaming: bool = False,
    index: Optional["AbstractIndex"] = None,
) -> Dataset[ST] | DatasetDict[ST]:
    """Load a dataset from local files, remote URLs, or an index.

    This function provides a HuggingFace Datasets-style interface for loading
    atdata typed datasets. It handles path resolution, split detection, and
    returns either a single Dataset or a DatasetDict depending on the split
    parameter.

    When no ``sample_type`` is provided, returns a ``Dataset[DictSample]`` that
    provides dynamic dict-like access to fields. Use ``.as_type(MyType)`` to
    convert to a typed schema.

    Args:
        path: Path to dataset. Can be:
            - Index lookup: "@handle/dataset-name" or "@local/dataset-name"
            - WebDataset brace notation: "path/to/{train,test}-{000..099}.tar"
            - Local directory: "./data/" (scans for .tar files)
            - Glob pattern: "path/to/*.tar"
            - Remote URL: "s3://bucket/path/data-*.tar"
            - Single file: "path/to/data.tar"

        sample_type: The PackableSample subclass defining the schema. If None,
            returns ``Dataset[DictSample]`` with dynamic field access. Can also
            be resolved from an index when using @handle/dataset syntax.

        split: Which split to load. If None, returns a DatasetDict with all
            detected splits. If specified (e.g., "train", "test"), returns
            a single Dataset for that split.

        data_files: Optional explicit mapping of data files. Can be:
            - str: Single file pattern
            - list[str]: List of file patterns (assigned to "train")
            - dict[str, str | list[str]]: Explicit split -> files mapping

        streaming: If True, explicitly marks the dataset for streaming mode.
            Note: atdata Datasets are already lazy/streaming via WebDataset
            pipelines, so this parameter primarily signals intent.

        index: Optional AbstractIndex for dataset lookup. Required when using
            @handle/dataset syntax. When provided with an indexed path, the
            schema can be auto-resolved from the index.

    Returns:
        If split is None: DatasetDict with all detected splits.
        If split is specified: Dataset for that split.
        Type is ``ST`` if sample_type provided, otherwise ``DictSample``.

    Raises:
        ValueError: If the specified split is not found.
        FileNotFoundError: If no data files are found at the path.
        KeyError: If dataset not found in index.

    Examples:
        >>> # Load without type - get DictSample for exploration
        >>> ds = load_dataset("./data/train.tar", split="train")
        >>> for sample in ds.ordered():
        ...     print(sample.keys())  # Explore fields
        ...     print(sample["text"]) # Dict-style access
        ...     print(sample.label)   # Attribute access
        >>>
        >>> # Convert to typed schema
        >>> typed_ds = ds.as_type(TextData)
        >>>
        >>> # Or load with explicit type directly
        >>> train_ds = load_dataset("./data/train-*.tar", TextData, split="train")
        >>>
        >>> # Load from index with auto-type resolution
        >>> index = Index()
        >>> ds = load_dataset("@local/my-dataset", index=index, split="train")
    """
    # Handle @handle/dataset indexed path resolution
    if _is_indexed_path(path):
        if index is None:
            index = get_default_index()

        source, schema_ref = _resolve_indexed_path(path, index)

        # Resolve sample_type from schema if not provided
        resolved_type: Type = (
            sample_type if sample_type is not None else index.decode_schema(schema_ref)
        )

        # Create dataset from the resolved source (includes credentials if S3)
        ds = Dataset[resolved_type](source)

        if split is not None:
            # Indexed datasets are single-split by default
            return ds

        return DatasetDict(
            {"train": ds}, sample_type=resolved_type, streaming=streaming
        )

    # Use DictSample as default when no type specified
    resolved_type = sample_type if sample_type is not None else DictSample

    # Resolve path to split -> shard URL mapping
    splits_shards = _resolve_shards(path, data_files)

    if not splits_shards:
        raise FileNotFoundError(f"No data files found at path: {path}")

    # Build Dataset for each split
    datasets: dict[str, Dataset] = {}
    for split_name, shards in splits_shards.items():
        url = _shards_to_wds_url(shards)
        ds = Dataset[resolved_type](url)
        datasets[split_name] = ds

    # Return single Dataset or DatasetDict
    if split is not None:
        if split not in datasets:
            available = list(datasets.keys())
            raise ValueError(
                f"Split '{split}' not found. Available splits: {available}"
            )
        return datasets[split]

    return DatasetDict(datasets, sample_type=resolved_type, streaming=streaming)


##
# Convenience re-exports (will be exposed in __init__.py)

__all__ = [
    "load_dataset",
    "DatasetDict",
]
