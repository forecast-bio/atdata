"""HuggingFace Datasets-style API for atdata.

This module provides a familiar `load_dataset()` interface inspired by the
HuggingFace Datasets library, adapted for atdata's typed WebDataset approach.

Key differences from HuggingFace Datasets:
- Requires explicit `sample_type` parameter (typed dataclass)
- Returns atdata.Dataset[ST] instead of HF Dataset
- Built on WebDataset for efficient streaming of large datasets
- No Arrow caching layer (WebDataset handles remote/local transparently)

Example:
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
from pathlib import Path
from typing import (
    Any,
    Generic,
    Iterator,
    Mapping,
    Type,
    TypeVar,
    Union,
    overload,
)

from .dataset import Dataset, PackableSample

##
# Type variables

ST = TypeVar("ST", bound=PackableSample)


##
# DatasetDict - container for multiple splits


class DatasetDict(Generic[ST], dict):
    """A dictionary of split names to Dataset instances.

    Similar to HuggingFace's DatasetDict, this provides a container for
    multiple dataset splits (train, test, validation, etc.) with convenience
    methods that operate across all splits.

    Type Parameters:
        ST: The sample type for all datasets in this dict.

    Example:
        >>> ds_dict = load_dataset("path/to/data", MyData)
        >>> train = ds_dict["train"]
        >>> test = ds_dict["test"]
        >>>
        >>> # Iterate over all splits
        >>> for split_name, dataset in ds_dict.items():
        ...     print(f"{split_name}: {len(dataset.shard_list)} shards")
    """

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
        return {name: len(ds.shard_list) for name, ds in self.items()}


##
# Path resolution utilities


def _is_brace_pattern(path: str) -> bool:
    """Check if path contains WebDataset brace expansion notation.

    Examples:
        >>> _is_brace_pattern("data-{000000..000099}.tar")
        True
        >>> _is_brace_pattern("data-{train,test}.tar")
        True
        >>> _is_brace_pattern("data-000000.tar")
        False
    """
    return bool(re.search(r"\{[^}]+\}", path))


def _is_glob_pattern(path: str) -> bool:
    """Check if path contains glob wildcards.

    Examples:
        >>> _is_glob_pattern("data-*.tar")
        True
        >>> _is_glob_pattern("data-000000.tar")
        False
    """
    return "*" in path or "?" in path


def _is_remote_url(path: str) -> bool:
    """Check if path is a remote URL (s3, http, etc.).

    Examples:
        >>> _is_remote_url("s3://bucket/path")
        True
        >>> _is_remote_url("https://example.com/data.tar")
        True
        >>> _is_remote_url("/local/path/data.tar")
        False
    """
    return path.startswith(("s3://", "gs://", "http://", "https://", "az://"))


def _expand_local_glob(pattern: str) -> list[str]:
    """Expand a local glob pattern to list of paths.

    Args:
        pattern: Glob pattern like "path/to/*.tar"

    Returns:
        Sorted list of matching file paths.
    """
    base_path = Path(pattern).parent
    glob_part = Path(pattern).name

    if not base_path.exists():
        return []

    matches = sorted(base_path.glob(glob_part))
    return [str(p) for p in matches if p.is_file()]


# Common split name patterns in filenames
_SPLIT_PATTERNS = [
    # Patterns like "dataset-train-000000.tar" (split in middle with delimiters)
    (r"[_-](train|training)[_-]", "train"),
    (r"[_-](test|testing)[_-]", "test"),
    (r"[_-](val|valid|validation)[_-]", "validation"),
    (r"[_-](dev|development)[_-]", "validation"),
    # Patterns at start of filename like "train-000.tar" or "test_data.tar"
    (r"^(train|training)[_-]", "train"),
    (r"^(test|testing)[_-]", "test"),
    (r"^(val|valid|validation)[_-]", "validation"),
    (r"^(dev|development)[_-]", "validation"),
    # Patterns in directory path like "/path/train/shard-000.tar"
    (r"[/\\](train|training)[/\\]", "train"),
    (r"[/\\](test|testing)[/\\]", "test"),
    (r"[/\\](val|valid|validation)[/\\]", "validation"),
    (r"[/\\](dev|development)[/\\]", "validation"),
    # Patterns at start of path like "train/shard-000.tar"
    (r"^(train|training)[/\\]", "train"),
    (r"^(test|testing)[/\\]", "test"),
    (r"^(val|valid|validation)[/\\]", "validation"),
    (r"^(dev|development)[/\\]", "validation"),
]


def _detect_split_from_path(path: str) -> str | None:
    """Attempt to detect split name from a file path.

    Args:
        path: File path to analyze.

    Returns:
        Detected split name ("train", "test", "validation") or None.
    """
    # Extract just the filename for pattern matching on full paths
    filename = Path(path).name
    path_lower = path.lower()
    filename_lower = filename.lower()

    # Check filename first (more specific)
    for pattern, split_name in _SPLIT_PATTERNS:
        if re.search(pattern, filename_lower):
            return split_name

    # Fall back to full path (catches directory patterns like "train/...")
    for pattern, split_name in _SPLIT_PATTERNS:
        if re.search(pattern, path_lower):
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
    if len(shards) == 0:
        raise ValueError("Cannot create URL from empty shard list")

    if len(shards) == 1:
        return shards[0]

    # Find common prefix across ALL shards
    prefix = shards[0]
    for s in shards[1:]:
        # Shorten prefix until it matches
        while not s.startswith(prefix) and prefix:
            prefix = prefix[:-1]

    # Find common suffix across ALL shards
    suffix = shards[0]
    for s in shards[1:]:
        # Shorten suffix until it matches
        while not s.endswith(suffix) and suffix:
            suffix = suffix[1:]

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
# Main load_dataset function


@overload
def load_dataset(
    path: str,
    sample_type: Type[ST],
    *,
    split: str,
    data_files: str | list[str] | dict[str, str | list[str]] | None = None,
    streaming: bool = False,
) -> Dataset[ST]: ...


@overload
def load_dataset(
    path: str,
    sample_type: Type[ST],
    *,
    split: None = None,
    data_files: str | list[str] | dict[str, str | list[str]] | None = None,
    streaming: bool = False,
) -> DatasetDict[ST]: ...


def load_dataset(
    path: str,
    sample_type: Type[ST],
    *,
    split: str | None = None,
    data_files: str | list[str] | dict[str, str | list[str]] | None = None,
    streaming: bool = False,
) -> Dataset[ST] | DatasetDict[ST]:
    """Load a dataset from local files or remote URLs.

    This function provides a HuggingFace Datasets-style interface for loading
    atdata typed datasets. It handles path resolution, split detection, and
    returns either a single Dataset or a DatasetDict depending on the split
    parameter.

    Args:
        path: Path to dataset. Can be:
            - WebDataset brace notation: "path/to/{train,test}-{000..099}.tar"
            - Local directory: "./data/" (scans for .tar files)
            - Glob pattern: "path/to/*.tar"
            - Remote URL: "s3://bucket/path/data-*.tar"
            - Single file: "path/to/data.tar"

        sample_type: The PackableSample subclass defining the schema for
            samples in this dataset. This is required (unlike HF Datasets)
            because atdata uses typed dataclasses.

        split: Which split to load. If None, returns a DatasetDict with all
            detected splits. If specified (e.g., "train", "test"), returns
            a single Dataset for that split.

        data_files: Optional explicit mapping of data files. Can be:
            - str: Single file pattern
            - list[str]: List of file patterns (assigned to "train")
            - dict[str, str | list[str]]: Explicit split -> files mapping

        streaming: If True, explicitly marks the dataset for streaming mode.
            Note: atdata Datasets are already lazy/streaming via WebDataset
            pipelines, so this parameter primarily signals intent. When True,
            shard list precomputation is skipped. Default False.

    Returns:
        If split is None: DatasetDict[ST] with all detected splits.
        If split is specified: Dataset[ST] for that split.

    Raises:
        ValueError: If the specified split is not found.
        FileNotFoundError: If no data files are found at the path.

    Example:
        >>> @atdata.packable
        ... class TextData:
        ...     text: str
        ...     label: int
        >>>
        >>> # Load single split
        >>> train_ds = load_dataset("./data/train-*.tar", TextData, split="train")
        >>>
        >>> # Load all splits
        >>> ds_dict = load_dataset("./data/", TextData)
        >>> train_ds = ds_dict["train"]
        >>> test_ds = ds_dict["test"]
        >>>
        >>> # Explicit data files
        >>> ds_dict = load_dataset("./data/", TextData, data_files={
        ...     "train": "train-*.tar",
        ...     "test": "test-*.tar",
        ... })
    """
    # Resolve path to split -> shard URL mapping
    splits_shards = _resolve_shards(path, data_files)

    if not splits_shards:
        raise FileNotFoundError(f"No data files found at path: {path}")

    # Build Dataset for each split
    datasets: dict[str, Dataset[ST]] = {}
    for split_name, shards in splits_shards.items():
        url = _shards_to_wds_url(shards)
        ds = Dataset[sample_type](url)
        datasets[split_name] = ds

    # Return single Dataset or DatasetDict
    if split is not None:
        if split not in datasets:
            available = list(datasets.keys())
            raise ValueError(
                f"Split '{split}' not found. Available splits: {available}"
            )
        return datasets[split]

    return DatasetDict(datasets, sample_type=sample_type, streaming=streaming)


##
# Convenience re-exports (will be exposed in __init__.py)

__all__ = [
    "load_dataset",
    "DatasetDict",
]
