"""``atdata preview`` command — render first N samples of a dataset."""

from __future__ import annotations

import sys
from typing import Any


def preview_dataset(url: str, limit: int = 5) -> int:
    """Print a human-readable preview of the first *limit* samples.

    Args:
        url: Dataset URL, local path, or atmosphere URI.
        limit: Number of samples to show. Default: 5.

    Returns:
        Exit code (0 success, 1 failure).
    """
    try:
        from ..dataset import Dataset, DictSample

        ds = Dataset[DictSample](url)
    except Exception as exc:
        print(f"Error opening dataset: {exc}", file=sys.stderr)
        return 1

    samples = ds.head(limit)
    if not samples:
        print("No samples found.", file=sys.stderr)
        return 1

    print(f"Preview of {url} ({len(samples)} sample(s)):")
    print()

    for i, sample in enumerate(samples):
        print(f"--- Sample {i} ---")
        for key in sample.keys():
            val = sample[key]
            print(f"  {key}: {_format_value(val)}")
        print()

    return 0


def _format_value(val: Any) -> str:
    """Format a value for preview, truncating large data."""
    import numpy as np

    if isinstance(val, np.ndarray):
        return f"ndarray shape={val.shape} dtype={val.dtype}"
    if isinstance(val, bytes):
        if len(val) <= 40:
            return repr(val)
        return f"bytes[{len(val)}] {val[:20]!r}..."
    if isinstance(val, str):
        if len(val) <= 80:
            return repr(val)
        return repr(val[:77] + "...")
    if isinstance(val, list):
        if len(val) <= 5:
            return repr(val)
        return f"[{val[0]!r}, {val[1]!r}, ... ({len(val)} items)]"
    return repr(val)
