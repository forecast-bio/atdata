"""``atdata inspect`` command — show dataset summary information."""

from __future__ import annotations

import sys
from typing import Any


def inspect_dataset(url: str) -> int:
    """Print summary information for a dataset at the given URL.

    Args:
        url: WebDataset URL, local path, or atmosphere URI.

    Returns:
        Exit code (0 success, 1 failure).
    """
    try:
        from ..dataset import Dataset, DictSample

        ds = Dataset[DictSample](url)
    except Exception as exc:
        print(f"Error opening dataset: {exc}", file=sys.stderr)
        return 1

    try:
        shards = ds.list_shards()
        print(f"URL:      {url}")
        print(f"Shards:   {len(shards)}")
        for shard in shards:
            print(f"  - {shard}")

        # Read first sample to infer schema
        samples = ds.head(1)
        if samples:
            sample = samples[0]
            print("Schema:   (inferred from first sample)")
            for key in sample.keys():
                val = sample[key]
                print(f"  {key}: {_describe_value(val)}")
        else:
            print("Schema:   (no samples found)")

        # Count samples — scan all shards
        count = sum(1 for _ in ds.ordered())
        print(f"Samples:  {count}")
    except Exception as exc:
        print(f"Error reading dataset: {exc}", file=sys.stderr)
        return 1

    return 0


def _describe_value(val: Any) -> str:
    """Human-readable type description for a sample field value."""
    import numpy as np

    if isinstance(val, np.ndarray):
        return f"ndarray dtype={val.dtype} shape={val.shape}"
    if isinstance(val, bytes):
        return f"bytes len={len(val)}"
    if isinstance(val, str):
        truncated = val[:60] + ("..." if len(val) > 60 else "")
        return f'str "{truncated}"'
    if isinstance(val, (int, float, bool)):
        return f"{type(val).__name__} {val}"
    if isinstance(val, list):
        return f"list len={len(val)}"
    return type(val).__name__
