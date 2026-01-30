"""Unit tests for the manifest system.

Tests ManifestField resolution, aggregate collectors, ManifestBuilder,
ManifestWriter round-trips, and QueryExecutor filtering.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

import atdata
from atdata.manifest import (
    CategoricalAggregate,
    ManifestBuilder,
    ManifestField,
    ManifestWriter,
    NumericAggregate,
    QueryExecutor,
    SetAggregate,
    ShardManifest,
    create_aggregate,
    resolve_manifest_fields,
)


# =============================================================================
# Test Sample Types
# =============================================================================


@atdata.packable
class ExplicitManifestSample:
    """Sample with explicit ManifestField annotations."""

    image: NDArray
    label: Annotated[str, ManifestField("categorical")]
    confidence: Annotated[float, ManifestField("numeric")]
    tags: Annotated[list[str], ManifestField("set")]


@atdata.packable
class AutoInferSample:
    """Sample relying on auto-inference for manifest fields."""

    name: str
    score: float
    count: int
    flag: bool
    tags: list[str]
    data: NDArray
    raw: bytes


@atdata.packable
class ExcludedFieldSample:
    """Sample with an explicitly excluded field."""

    label: Annotated[str, ManifestField("categorical")]
    secret: Annotated[str, ManifestField("categorical", exclude=True)]


@atdata.packable
class OptionalFieldSample:
    """Sample with optional fields."""

    name: str
    score: Optional[float] = None


# =============================================================================
# ManifestField Resolution Tests
# =============================================================================


class TestResolveManifestFields:
    def test_explicit_annotations(self):
        fields = resolve_manifest_fields(ExplicitManifestSample)
        assert "label" in fields
        assert fields["label"].aggregate == "categorical"
        assert "confidence" in fields
        assert fields["confidence"].aggregate == "numeric"
        assert "tags" in fields
        assert fields["tags"].aggregate == "set"
        # NDArray excluded
        assert "image" not in fields

    def test_auto_inference(self):
        fields = resolve_manifest_fields(AutoInferSample)
        assert fields["name"].aggregate == "categorical"
        assert fields["score"].aggregate == "numeric"
        assert fields["count"].aggregate == "numeric"
        assert fields["flag"].aggregate == "categorical"
        assert fields["tags"].aggregate == "set"
        # NDArray and bytes excluded
        assert "data" not in fields
        assert "raw" not in fields

    def test_explicit_exclude(self):
        fields = resolve_manifest_fields(ExcludedFieldSample)
        assert "label" in fields
        assert "secret" not in fields

    def test_optional_fields(self):
        fields = resolve_manifest_fields(OptionalFieldSample)
        assert "name" in fields
        assert fields["name"].aggregate == "categorical"
        assert "score" in fields
        assert fields["score"].aggregate == "numeric"

    def test_non_dataclass_raises(self):
        with pytest.raises(TypeError):
            resolve_manifest_fields(str)


# =============================================================================
# Aggregate Tests
# =============================================================================


class TestCategoricalAggregate:
    def test_add_and_counts(self):
        agg = CategoricalAggregate()
        agg.add("dog")
        agg.add("cat")
        agg.add("dog")
        assert agg.cardinality == 2
        assert agg.value_counts == {"dog": 2, "cat": 1}

    def test_to_dict(self):
        agg = CategoricalAggregate()
        agg.add("a")
        d = agg.to_dict()
        assert d["type"] == "categorical"
        assert d["cardinality"] == 1
        assert d["value_counts"] == {"a": 1}

    def test_empty(self):
        agg = CategoricalAggregate()
        assert agg.cardinality == 0
        assert agg.to_dict()["value_counts"] == {}


class TestNumericAggregate:
    def test_add_and_stats(self):
        agg = NumericAggregate()
        agg.add(1.0)
        agg.add(3.0)
        agg.add(2.0)
        assert agg.min == 1.0
        assert agg.max == 3.0
        assert agg.mean == pytest.approx(2.0)
        assert agg.count == 3

    def test_to_dict(self):
        agg = NumericAggregate()
        agg.add(5)
        agg.add(10)
        d = agg.to_dict()
        assert d["type"] == "numeric"
        assert d["min"] == 5.0
        assert d["max"] == 10.0
        assert d["mean"] == pytest.approx(7.5)
        assert d["count"] == 2

    def test_empty_mean_is_zero(self):
        agg = NumericAggregate()
        assert agg.mean == 0.0

    def test_single_value(self):
        agg = NumericAggregate()
        agg.add(42)
        assert agg.min == 42.0
        assert agg.max == 42.0
        assert agg.mean == 42.0


class TestSetAggregate:
    def test_add_and_union(self):
        agg = SetAggregate()
        agg.add(["outdoor", "day"])
        agg.add(["indoor"])
        agg.add(["outdoor"])  # duplicate
        assert agg.all_values == {"outdoor", "day", "indoor"}

    def test_to_dict_sorted(self):
        agg = SetAggregate()
        agg.add(["c", "a", "b"])
        d = agg.to_dict()
        assert d["type"] == "set"
        assert d["all_values"] == ["a", "b", "c"]

    def test_empty(self):
        agg = SetAggregate()
        assert agg.all_values == set()


class TestCreateAggregate:
    def test_categorical(self):
        assert isinstance(create_aggregate("categorical"), CategoricalAggregate)

    def test_numeric(self):
        assert isinstance(create_aggregate("numeric"), NumericAggregate)

    def test_set(self):
        assert isinstance(create_aggregate("set"), SetAggregate)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown aggregate kind"):
            create_aggregate("invalid")


# =============================================================================
# ManifestBuilder Tests
# =============================================================================


class TestManifestBuilder:
    def test_build_basic(self):
        builder = ManifestBuilder(
            sample_type=ExplicitManifestSample,
            shard_id="test/shard-000000",
            schema_version="1.0.0",
        )

        sample = ExplicitManifestSample(
            image=np.zeros((2, 2)),
            label="dog",
            confidence=0.95,
            tags=["outdoor", "day"],
        )
        builder.add_sample(key="s0", offset=0, size=1024, sample=sample)

        sample2 = ExplicitManifestSample(
            image=np.ones((2, 2)),
            label="cat",
            confidence=0.80,
            tags=["indoor"],
        )
        builder.add_sample(key="s1", offset=1024, size=2048, sample=sample2)

        manifest = builder.build()
        assert manifest.shard_id == "test/shard-000000"
        assert manifest.schema_type == "ExplicitManifestSample"
        assert manifest.schema_version == "1.0.0"
        assert manifest.num_samples == 2
        assert manifest.size_bytes == 1024 + 2048

        # Check aggregates
        assert manifest.aggregates["label"]["type"] == "categorical"
        assert manifest.aggregates["label"]["value_counts"]["dog"] == 1
        assert manifest.aggregates["label"]["value_counts"]["cat"] == 1
        assert manifest.aggregates["confidence"]["type"] == "numeric"
        assert manifest.aggregates["confidence"]["min"] == 0.80
        assert manifest.aggregates["confidence"]["max"] == 0.95
        assert manifest.aggregates["tags"]["type"] == "set"
        assert set(manifest.aggregates["tags"]["all_values"]) == {
            "outdoor",
            "day",
            "indoor",
        }

        # Check samples DataFrame
        assert len(manifest.samples) == 2
        assert list(manifest.samples["__key__"]) == ["s0", "s1"]
        assert "__offset__" in manifest.samples.columns
        assert "__size__" in manifest.samples.columns
        assert "label" in manifest.samples.columns

    def test_provenance(self):
        builder = ManifestBuilder(
            sample_type=AutoInferSample,
            shard_id="test/shard-000000",
            source_job_id="job-123",
            parent_shards=["input/shard-000000"],
            pipeline_version="2.0.0",
        )
        manifest = builder.build()
        assert manifest.provenance["source_job_id"] == "job-123"
        assert manifest.provenance["parent_shards"] == ["input/shard-000000"]
        assert manifest.provenance["pipeline_version"] == "2.0.0"

    def test_empty_build(self):
        builder = ManifestBuilder(
            sample_type=ExplicitManifestSample,
            shard_id="test/empty",
        )
        manifest = builder.build()
        assert manifest.num_samples == 0
        assert manifest.samples.empty


# =============================================================================
# ManifestWriter + ShardManifest Round-Trip Tests
# =============================================================================


class TestManifestWriterRoundTrip:
    def _make_manifest(self) -> ShardManifest:
        """Create a test manifest with known data."""
        samples = pd.DataFrame(
            {
                "__key__": ["s0", "s1", "s2"],
                "__offset__": [0, 1024, 3072],
                "__size__": [1024, 2048, 512],
                "label": ["dog", "cat", "dog"],
                "confidence": [0.95, 0.80, 0.70],
            }
        )
        return ShardManifest(
            shard_id="test/shard-000000",
            schema_type="TestSample",
            schema_version="1.0.0",
            num_samples=3,
            size_bytes=3584,
            created_at=datetime(2025, 1, 22, 14, 32, 0, tzinfo=timezone.utc),
            aggregates={
                "label": {
                    "type": "categorical",
                    "cardinality": 2,
                    "value_counts": {"dog": 2, "cat": 1},
                },
                "confidence": {
                    "type": "numeric",
                    "min": 0.70,
                    "max": 0.95,
                    "mean": 0.8167,
                    "count": 3,
                },
            },
            samples=samples,
        )

    def test_write_creates_files(self, tmp_path: Path):
        manifest = self._make_manifest()
        writer = ManifestWriter(tmp_path / "shard-000000")
        json_path, parquet_path = writer.write(manifest)

        assert json_path.exists()
        assert parquet_path.exists()
        assert json_path.name == "shard-000000.manifest.json"
        assert parquet_path.name == "shard-000000.manifest.parquet"

    def test_round_trip(self, tmp_path: Path):
        original = self._make_manifest()
        writer = ManifestWriter(tmp_path / "shard-000000")
        json_path, parquet_path = writer.write(original)

        loaded = ShardManifest.from_files(json_path, parquet_path)
        assert loaded.shard_id == original.shard_id
        assert loaded.schema_type == original.schema_type
        assert loaded.schema_version == original.schema_version
        assert loaded.num_samples == original.num_samples
        assert loaded.size_bytes == original.size_bytes
        assert loaded.aggregates == original.aggregates
        assert len(loaded.samples) == len(original.samples)
        assert list(loaded.samples["__key__"]) == list(original.samples["__key__"])

    def test_json_only_loading(self, tmp_path: Path):
        manifest = self._make_manifest()
        writer = ManifestWriter(tmp_path / "shard-000000")
        json_path, _ = writer.write(manifest)

        header_only = ShardManifest.from_json_only(json_path)
        assert header_only.shard_id == manifest.shard_id
        assert header_only.num_samples == manifest.num_samples
        assert header_only.aggregates == manifest.aggregates
        assert header_only.samples.empty

    def test_header_dict_format(self):
        manifest = self._make_manifest()
        header = manifest.header_dict()
        assert "manifest_version" in header
        assert header["shard_id"] == "test/shard-000000"
        assert header["num_samples"] == 3
        assert "aggregates" in header


# =============================================================================
# QueryExecutor Tests
# =============================================================================


class TestQueryExecutor:
    def _make_manifests(self) -> list[ShardManifest]:
        """Create two test manifests for query testing."""
        samples1 = pd.DataFrame(
            {
                "__key__": ["s0", "s1"],
                "__offset__": [0, 1024],
                "__size__": [1024, 1024],
                "label": ["dog", "cat"],
                "confidence": [0.95, 0.40],
            }
        )
        m1 = ShardManifest(
            shard_id="shard-000000",
            schema_type="TestSample",
            schema_version="1.0.0",
            num_samples=2,
            size_bytes=2048,
            created_at=datetime.now(timezone.utc),
            aggregates={
                "label": {
                    "type": "categorical",
                    "cardinality": 2,
                    "value_counts": {"dog": 1, "cat": 1},
                },
                "confidence": {
                    "type": "numeric",
                    "min": 0.40,
                    "max": 0.95,
                    "mean": 0.675,
                    "count": 2,
                },
            },
            samples=samples1,
        )

        samples2 = pd.DataFrame(
            {
                "__key__": ["s2", "s3"],
                "__offset__": [0, 512],
                "__size__": [512, 512],
                "label": ["bird", "dog"],
                "confidence": [0.60, 0.85],
            }
        )
        m2 = ShardManifest(
            shard_id="shard-000001",
            schema_type="TestSample",
            schema_version="1.0.0",
            num_samples=2,
            size_bytes=1024,
            created_at=datetime.now(timezone.utc),
            aggregates={
                "label": {
                    "type": "categorical",
                    "cardinality": 2,
                    "value_counts": {"bird": 1, "dog": 1},
                },
                "confidence": {
                    "type": "numeric",
                    "min": 0.60,
                    "max": 0.85,
                    "mean": 0.725,
                    "count": 2,
                },
            },
            samples=samples2,
        )

        return [m1, m2]

    def test_query_numeric_filter(self):
        manifests = self._make_manifests()
        executor = QueryExecutor(manifests)
        results = executor.query(where=lambda df: df["confidence"] > 0.80)

        keys = [r.key for r in results]
        assert "s0" in keys  # 0.95
        assert "s3" in keys  # 0.85
        assert "s1" not in keys  # 0.40
        assert "s2" not in keys  # 0.60

    def test_query_categorical_filter(self):
        manifests = self._make_manifests()
        executor = QueryExecutor(manifests)
        results = executor.query(where=lambda df: df["label"] == "dog")

        keys = [r.key for r in results]
        assert keys == ["s0", "s3"]

    def test_query_combined_filter(self):
        manifests = self._make_manifests()
        executor = QueryExecutor(manifests)
        results = executor.query(
            where=lambda df: (df["label"] == "dog") & (df["confidence"] > 0.90)
        )

        assert len(results) == 1
        assert results[0].key == "s0"
        assert results[0].shard == "shard-000000"

    def test_query_no_matches(self):
        manifests = self._make_manifests()
        executor = QueryExecutor(manifests)
        results = executor.query(where=lambda df: df["confidence"] > 0.99)
        assert results == []

    def test_query_all_match(self):
        manifests = self._make_manifests()
        executor = QueryExecutor(manifests)
        results = executor.query(where=lambda df: df["confidence"] > 0.0)
        assert len(results) == 4

    def test_sample_location_fields(self):
        manifests = self._make_manifests()
        executor = QueryExecutor(manifests)
        results = executor.query(where=lambda df: df["__key__"] == "s0")

        assert len(results) == 1
        loc = results[0]
        assert loc.shard == "shard-000000"
        assert loc.key == "s0"
        assert loc.offset == 0

    def test_empty_manifests(self):
        executor = QueryExecutor([])
        results = executor.query(where=lambda df: df["confidence"] > 0.5)
        assert results == []

    def test_from_directory(self, tmp_path: Path):
        """Test loading manifests from a directory."""
        manifests = self._make_manifests()
        for m in manifests:
            writer = ManifestWriter(tmp_path / m.shard_id)
            writer.write(m)

        executor = QueryExecutor.from_directory(tmp_path)
        results = executor.query(where=lambda df: df["label"] == "dog")
        assert len(results) == 2
