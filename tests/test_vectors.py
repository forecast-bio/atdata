"""Cross-repo test vector runner.

Executes shard-roundtrip test vectors from ``test-vectors/shard-roundtrip/``
in the vendored lexicons directory.  Each vector specifies a schema, a set of
input samples, and expected outputs.  The runner writes samples to a shard,
reads them back, and verifies against expected values.

Skips gracefully if the test-vectors directory does not exist yet.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import atdata

# Try two locations: sibling test-vectors dir from atdata-lexicon, or vendored
_REPO_ROOT = Path(__file__).resolve().parent.parent
_CANDIDATE_PATHS = [
    _REPO_ROOT / "test-vectors" / "shard-roundtrip",
    _REPO_ROOT / "src" / "atdata" / "lexicons" / ".." / ".." / ".." / ".."
    / "test-vectors" / "shard-roundtrip",
]

VECTORS_DIR: Path | None = None
for _p in _CANDIDATE_PATHS:
    _resolved = _p.resolve()
    if _resolved.is_dir():
        VECTORS_DIR = _resolved
        break


def _collect_vector_files() -> list[Path]:
    """Collect all vector JSON files, or return empty if dir missing."""
    if VECTORS_DIR is None:
        return []
    return sorted(VECTORS_DIR.glob("*.json"))


def _make_packable_class(
    schema_def: dict[str, Any],
) -> type:
    """Dynamically create a packable dataclass from a vector schema definition."""
    fields_def = schema_def["fields"]
    annotations: dict[str, type] = {}
    for f in fields_def:
        type_str = f["type"]
        if type_str == "str":
            annotations[f["name"]] = str
        elif type_str == "int":
            annotations[f["name"]] = int
        elif type_str == "float":
            annotations[f["name"]] = float
        elif type_str == "dict":
            annotations[f["name"]] = dict
        elif type_str == "ndarray":
            annotations[f["name"]] = np.ndarray
        else:
            annotations[f["name"]] = Any

    ns: dict[str, Any] = {"__annotations__": annotations}
    cls = type("VectorSample", (), ns)
    cls = dataclass(cls)
    cls = atdata.packable(cls)
    return cls


def _make_sample(cls: type, sample_data: dict[str, Any], schema_def: dict) -> Any:
    """Create a sample instance from vector data."""
    kwargs: dict[str, Any] = {}
    field_types = {f["name"]: f for f in schema_def["fields"]}

    for fname, fdef in field_types.items():
        val = sample_data.get(fname)
        if fdef["type"] == "ndarray" and val is not None:
            dtype = fdef.get("dtype", "float32")
            val = np.array(val, dtype=dtype)
        kwargs[fname] = val

    kwargs["__key__"] = sample_data["__key__"]
    return cls(**kwargs)


def _check_field_value(
    actual: Any, expected: Any, field_name: str, key: str
) -> None:
    """Assert a single field value matches the expected value."""
    if isinstance(expected, dict) and "dtype" in expected and "values" in expected:
        # NDArray check
        assert isinstance(actual, np.ndarray), (
            f"Sample {key}.{field_name}: expected ndarray, got {type(actual)}"
        )
        np.testing.assert_array_almost_equal(
            actual,
            np.array(expected["values"], dtype=expected["dtype"]),
            err_msg=f"Sample {key}.{field_name} values mismatch",
        )
        assert list(actual.shape) == expected["shape"], (
            f"Sample {key}.{field_name} shape mismatch: "
            f"{list(actual.shape)} != {expected['shape']}"
        )
    else:
        assert actual == expected, (
            f"Sample {key}.{field_name}: {actual!r} != {expected!r}"
        )


_vector_files = _collect_vector_files()


@pytest.mark.skipif(
    not _vector_files,
    reason="No test vectors found (test-vectors/shard-roundtrip/ not present)",
)
@pytest.mark.parametrize(
    "vector_path",
    _vector_files,
    ids=[p.stem for p in _vector_files],
)
def test_shard_roundtrip_vector(vector_path: Path, tmp_path: Path) -> None:
    """Write samples from a test vector to a shard, read back, verify."""
    vector = json.loads(vector_path.read_text())

    schema_def = vector["inputs"]["schema"]
    sample_cls = _make_packable_class(schema_def)

    # Create sample instances
    samples = [
        _make_sample(sample_cls, s, schema_def)
        for s in vector["inputs"]["samples"]
    ]

    # Write to shard
    shard_path = tmp_path / "test-shard-000000.tar"
    atdata.write_samples(samples, str(shard_path))

    # Read back
    ds = atdata.Dataset[sample_cls](str(shard_path))
    read_samples = list(ds)

    # Verify expected sample count
    expected = vector["expected"]
    assert len(read_samples) == expected["sample_count"], (
        f"Expected {expected['sample_count']} samples, got {len(read_samples)}"
    )

    # Verify keys
    read_keys = [s.__key__ for s in read_samples]
    assert read_keys == expected["keys"], (
        f"Key mismatch: {read_keys} != {expected['keys']}"
    )

    # Verify field values
    samples_by_key = {s.__key__: s for s in read_samples}
    for key, field_checks in expected.get("field_checks", {}).items():
        sample = samples_by_key[key]
        for field_name, expected_val in field_checks.items():
            actual_val = getattr(sample, field_name)
            _check_field_value(actual_val, expected_val, field_name, key)
