"""``atdata schema`` commands — show and diff dataset schemas."""

from __future__ import annotations

import sys


def schema_show(dataset_ref: str) -> int:
    """Display the schema of a dataset.

    Args:
        dataset_ref: Dataset URL, local path, or index reference
            (e.g. ``@local/my-dataset``).

    Returns:
        Exit code (0 success, 1 failure).
    """
    try:
        from ..dataset import Dataset, DictSample

        ds = Dataset[DictSample](dataset_ref)
    except Exception as exc:
        print(f"Error opening dataset: {exc}", file=sys.stderr)
        return 1

    samples = ds.head(1)
    if not samples:
        print("No samples found — cannot infer schema.", file=sys.stderr)
        return 1

    sample = samples[0]
    print(f"Schema for: {dataset_ref}")
    print(f"Fields ({len(sample.keys())}):")
    for key in sample.keys():
        val = sample[key]
        print(f"  {key}: {_type_label(val)}")

    return 0


def schema_diff(url_a: str, url_b: str) -> int:
    """Compare schemas of two datasets and print differences.

    Args:
        url_a: First dataset URL / path.
        url_b: Second dataset URL / path.

    Returns:
        Exit code (0 identical, 1 different, 2 error).
    """
    try:
        from ..dataset import Dataset, DictSample

        ds_a = Dataset[DictSample](url_a)
        ds_b = Dataset[DictSample](url_b)
    except Exception as exc:
        print(f"Error opening dataset: {exc}", file=sys.stderr)
        return 2

    samples_a = ds_a.head(1)
    samples_b = ds_b.head(1)

    if not samples_a:
        print(f"No samples in {url_a}", file=sys.stderr)
        return 2
    if not samples_b:
        print(f"No samples in {url_b}", file=sys.stderr)
        return 2

    fields_a = {k: _type_label(samples_a[0][k]) for k in samples_a[0].keys()}
    fields_b = {k: _type_label(samples_b[0][k]) for k in samples_b[0].keys()}

    keys_a = set(fields_a)
    keys_b = set(fields_b)

    added = sorted(keys_b - keys_a)
    removed = sorted(keys_a - keys_b)
    common = sorted(keys_a & keys_b)
    changed = [k for k in common if fields_a[k] != fields_b[k]]

    if not added and not removed and not changed:
        print("Schemas are identical.")
        return 0

    if added:
        print("Added:")
        for k in added:
            print(f"  + {k}: {fields_b[k]}")
    if removed:
        print("Removed:")
        for k in removed:
            print(f"  - {k}: {fields_a[k]}")
    if changed:
        print("Changed:")
        for k in changed:
            print(f"  ~ {k}: {fields_a[k]} -> {fields_b[k]}")

    return 1


def _type_label(val: object) -> str:
    """Short type label for schema display."""
    import numpy as np

    if isinstance(val, np.ndarray):
        return f"ndarray[{val.dtype}]"
    if isinstance(val, bytes):
        return "bytes"
    return type(val).__name__
