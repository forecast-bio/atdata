"""Coverage tests for atdata.manifest._query.

Targets uncovered lines: 81, 121, 138-150.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from numpy.typing import NDArray

import atdata
from atdata.manifest import (
    ManifestBuilder,
    ManifestField,
    ManifestWriter,
    QueryExecutor,
    ShardManifest,
)


# ---------------------------------------------------------------------------
# Test sample type
# ---------------------------------------------------------------------------


@atdata.packable
class QueryCovSample:
    """Sample type used across all tests in this module."""

    data: NDArray
    label: Annotated[str, ManifestField("categorical")]
    score: Annotated[float, ManifestField("numeric")]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sample(label: str, score: float) -> QueryCovSample:
    return QueryCovSample(
        data=np.zeros((2, 2), dtype=np.float32),
        label=label,
        score=score,
    )


def _build_manifest(shard_id: str, samples: list[QueryCovSample]) -> ShardManifest:
    builder = ManifestBuilder(
        sample_type=QueryCovSample,
        shard_id=shard_id,
    )
    for i, s in enumerate(samples):
        builder.add_sample(key=f"sample_{i:05d}", offset=i * 1024, size=1024, sample=s)
    return builder.build()


def _write_manifest(tmp_path: Path, shard_name: str, manifest: ShardManifest) -> Path:
    """Write manifest files and return the base path (without extension)."""
    base = tmp_path / shard_name
    writer = ManifestWriter(str(base))
    writer.write(manifest)
    return base


# ---------------------------------------------------------------------------
# Tests — line 81: empty manifest skipping
# ---------------------------------------------------------------------------


def test_query_skips_empty_manifest(tmp_path: Path) -> None:
    """QueryExecutor skips manifests whose samples DataFrame is empty."""
    empty_manifest = ShardManifest(
        shard_id="empty-shard",
        schema_type="QueryCovSample",
        schema_version="1.0.0",
        num_samples=0,
        size_bytes=0,
        created_at=pd.Timestamp.now(tz="UTC").to_pydatetime(),
        aggregates={},
        samples=pd.DataFrame(),  # empty
    )
    populated = _build_manifest("pop-shard", [_make_sample("cat", 0.9)])

    executor = QueryExecutor([empty_manifest, populated])
    results = executor.query(where=lambda df: df["label"] == "cat")

    # Only the populated manifest should contribute results
    assert len(results) == 1
    assert results[0].shard == "pop-shard"
    assert results[0].key == "sample_00000"


# ---------------------------------------------------------------------------
# Tests — line 121: from_directory with JSON-only manifest
# ---------------------------------------------------------------------------


def test_from_directory_json_only(tmp_path: Path) -> None:
    """from_directory loads a manifest via from_json_only when parquet is absent."""
    manifest = _build_manifest("shard-000000", [_make_sample("dog", 0.5)])

    # Write both files, then delete the parquet one
    base = _write_manifest(tmp_path, "shard-000000", manifest)
    parquet = base.with_suffix(".manifest.parquet")
    assert parquet.exists()
    parquet.unlink()

    executor = QueryExecutor.from_directory(tmp_path)

    # The loaded manifest has an empty DataFrame (json-only)
    assert len(executor._manifests) == 1
    assert executor._manifests[0].shard_id == "shard-000000"
    assert executor._manifests[0].samples.empty

    # Query returns nothing because samples DF is empty
    results = executor.query(where=lambda df: df["label"] == "dog")
    assert results == []


def test_from_directory_both_files(tmp_path: Path) -> None:
    """from_directory loads a manifest with both JSON and parquet present."""
    manifest = _build_manifest("shard-000001", [_make_sample("cat", 0.8)])
    _write_manifest(tmp_path, "shard-000001", manifest)

    executor = QueryExecutor.from_directory(tmp_path)

    assert len(executor._manifests) == 1
    assert not executor._manifests[0].samples.empty
    results = executor.query(where=lambda df: df["label"] == "cat")
    assert len(results) == 1


# ---------------------------------------------------------------------------
# Tests — lines 138-150: from_shard_urls
# ---------------------------------------------------------------------------


def test_from_shard_urls_both_files(tmp_path: Path) -> None:
    """from_shard_urls loads when both .manifest.json and .manifest.parquet exist."""
    manifest = _build_manifest(
        "shard-000010",
        [
            _make_sample("bird", 0.7),
            _make_sample("fish", 0.3),
        ],
    )
    _write_manifest(tmp_path, "shard-000010", manifest)

    shard_url = str(tmp_path / "shard-000010.tar")
    executor = QueryExecutor.from_shard_urls([shard_url])

    assert len(executor._manifests) == 1
    assert not executor._manifests[0].samples.empty
    results = executor.query(where=lambda df: df["score"] > 0.5)
    assert len(results) == 1
    assert results[0].key == "sample_00000"


def test_from_shard_urls_json_only(tmp_path: Path) -> None:
    """from_shard_urls falls back to json-only when parquet is missing."""
    manifest = _build_manifest("shard-000020", [_make_sample("ant", 0.1)])
    base = _write_manifest(tmp_path, "shard-000020", manifest)

    # Remove parquet
    base.with_suffix(".manifest.parquet").unlink()

    shard_url = str(tmp_path / "shard-000020.tar")
    executor = QueryExecutor.from_shard_urls([shard_url])

    assert len(executor._manifests) == 1
    assert executor._manifests[0].samples.empty


def test_from_shard_urls_no_manifest(tmp_path: Path) -> None:
    """from_shard_urls skips shards with no manifest files at all."""
    shard_url = str(tmp_path / "nonexistent-shard.tar")
    executor = QueryExecutor.from_shard_urls([shard_url])

    assert len(executor._manifests) == 0


def test_from_shard_urls_mixed(tmp_path: Path) -> None:
    """from_shard_urls handles a mix of present, json-only, and missing manifests."""
    # Shard A: both files
    manifest_a = _build_manifest("shard-a", [_make_sample("x", 1.0)])
    _write_manifest(tmp_path, "shard-a", manifest_a)

    # Shard B: json only
    manifest_b = _build_manifest("shard-b", [_make_sample("y", 2.0)])
    base_b = _write_manifest(tmp_path, "shard-b", manifest_b)
    base_b.with_suffix(".manifest.parquet").unlink()

    # Shard C: nothing at all (no files written)

    shard_urls = [
        str(tmp_path / "shard-a.tar"),
        str(tmp_path / "shard-b.tar"),
        str(tmp_path / "shard-c.tar"),
    ]
    executor = QueryExecutor.from_shard_urls(shard_urls)

    # A (full) + B (json-only) loaded; C skipped
    assert len(executor._manifests) == 2
    assert not executor._manifests[0].samples.empty  # shard-a: full
    assert executor._manifests[1].samples.empty  # shard-b: json-only

    results = executor.query(where=lambda df: df["score"] >= 1.0)
    # Only shard-a has queryable samples
    assert len(results) == 1
    assert results[0].shard == "shard-a"
