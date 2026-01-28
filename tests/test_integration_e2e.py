"""End-to-end integration tests for atdata data flow pipeline.

Tests the complete workflow: Create → Store → Load → Iterate → Verify.

These tests verify:
- Full pipeline with various sample types
- Multi-shard datasets with brace notation
- Large batch handling and memory efficiency
- Metadata round-trip preservation
- Parquet export with transformations
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import webdataset as wds

import atdata


##
# Test sample types


@atdata.packable
class SimpleSample:
    """Basic sample with primitive types only."""

    name: str
    value: int
    score: float
    active: bool


@atdata.packable
class NDArraySample:
    """Sample with multiple NDArray fields of different shapes."""

    label: int
    image: NDArray
    features: NDArray


@atdata.packable
class OptionalNDArraySample:
    """Sample with optional NDArray fields."""

    label: int
    image: NDArray
    embeddings: NDArray | None = None


@atdata.packable
class BytesSample:
    """Sample with bytes field."""

    name: str
    raw_data: bytes


@atdata.packable
class ListSample:
    """Sample with list fields."""

    tags: list[str]
    scores: list[float]
    ids: list[int]


@dataclass
class InheritanceSample(atdata.PackableSample):
    """Sample using inheritance syntax instead of decorator."""

    title: str
    count: int
    measurements: NDArray


##
# Helper functions


def create_simple_samples(n: int) -> list[SimpleSample]:
    """Create n simple samples with distinct values."""
    return [
        SimpleSample(
            name=f"sample_{i}",
            value=i * 10,
            score=float(i) * 0.5,
            active=(i % 2 == 0),
        )
        for i in range(n)
    ]


def create_ndarray_samples(n: int, img_shape: tuple = (64, 64)) -> list[NDArraySample]:
    """Create n NDArray samples with distinct values."""
    return [
        NDArraySample(
            label=i,
            image=np.random.randn(*img_shape).astype(np.float32),
            features=np.random.randn(128).astype(np.float32),
        )
        for i in range(n)
    ]


def create_optional_samples(
    n: int, include_optional: bool
) -> list[OptionalNDArraySample]:
    """Create samples with or without optional embeddings."""
    return [
        OptionalNDArraySample(
            label=i,
            image=np.random.randn(32, 32).astype(np.float32),
            embeddings=np.random.randn(64).astype(np.float32)
            if include_optional
            else None,
        )
        for i in range(n)
    ]


def write_single_shard(path: Path, samples: list) -> str:
    """Write samples to a single tar file, return path."""
    tar_path = path.as_posix()
    with wds.writer.TarWriter(tar_path) as sink:
        for sample in samples:
            sink.write(sample.as_wds)
    return tar_path


def write_multi_shard(
    base_path: Path,
    samples: list,
    samples_per_shard: int = 10,
) -> tuple[str, int]:
    """Write samples to multiple shards, return brace pattern and shard count."""
    pattern = (base_path / "shard-%06d.tar").as_posix()
    with wds.writer.ShardWriter(pattern=pattern, maxcount=samples_per_shard) as sink:
        for sample in samples:
            sink.write(sample.as_wds)

    n_shards = (len(samples) + samples_per_shard - 1) // samples_per_shard
    brace_pattern = (base_path / f"shard-{{000000..{n_shards - 1:06d}}}.tar").as_posix()
    return brace_pattern, n_shards


##
# Full Pipeline Tests


class TestFullPipelineSimple:
    """End-to-end tests with simple primitive-only samples."""

    def test_create_store_load_iterate_single_shard(self, tmp_path):
        """Full pipeline: create → store → load → iterate (single shard)."""
        n_samples = 50
        samples = create_simple_samples(n_samples)

        # Store
        tar_path = write_single_shard(tmp_path / "simple.tar", samples)

        # Load
        dataset = atdata.Dataset[SimpleSample](tar_path)

        # Iterate without batching
        loaded = list(dataset.ordered(batch_size=None))

        # Verify
        assert len(loaded) == n_samples
        for i, sample in enumerate(loaded):
            assert isinstance(sample, SimpleSample)
            assert sample.name == f"sample_{i}"
            assert sample.value == i * 10
            assert sample.score == float(i) * 0.5
            assert sample.active == (i % 2 == 0)

    def test_create_store_load_iterate_batched(self, tmp_path):
        """Full pipeline with batching."""
        n_samples = 100
        batch_size = 16
        samples = create_simple_samples(n_samples)

        tar_path = write_single_shard(tmp_path / "batched.tar", samples)
        dataset = atdata.Dataset[SimpleSample](tar_path)

        # Iterate with batching
        batches = list(dataset.ordered(batch_size=batch_size))

        # Verify batch structure (WebDataset drops incomplete final batch)
        total_samples = sum(len(b.samples) for b in batches)
        assert total_samples >= (n_samples // batch_size) * batch_size

        for batch in batches:
            assert isinstance(batch, atdata.SampleBatch)
            assert batch.sample_type == SimpleSample
            assert len(batch.samples) <= batch_size

            # Verify aggregated attributes
            names = batch.name
            values = batch.value
            assert isinstance(names, list)
            assert isinstance(values, list)
            assert len(names) == len(batch.samples)
            assert len(values) == len(batch.samples)

    def test_inheritance_syntax_pipeline(self, tmp_path):
        """Full pipeline using inheritance-style sample definition."""
        n_samples = 25
        samples = [
            InheritanceSample(
                title=f"doc_{i}",
                count=i * 5,
                measurements=np.random.randn(10).astype(np.float32),
            )
            for i in range(n_samples)
        ]

        tar_path = write_single_shard(tmp_path / "inheritance.tar", samples)
        dataset = atdata.Dataset[InheritanceSample](tar_path)

        loaded = list(dataset.ordered(batch_size=None))

        assert len(loaded) == n_samples
        for i, sample in enumerate(loaded):
            assert isinstance(sample, InheritanceSample)
            assert sample.title == f"doc_{i}"
            assert sample.count == i * 5
            assert isinstance(sample.measurements, np.ndarray)


class TestFullPipelineNDArray:
    """End-to-end tests with NDArray samples."""

    def test_ndarray_serialization_roundtrip(self, tmp_path):
        """NDArray fields should serialize and deserialize exactly."""
        n_samples = 20
        samples = create_ndarray_samples(n_samples, img_shape=(32, 32))

        tar_path = write_single_shard(tmp_path / "ndarray.tar", samples)
        dataset = atdata.Dataset[NDArraySample](tar_path)

        loaded = list(dataset.ordered(batch_size=None))

        assert len(loaded) == n_samples
        for original, loaded_sample in zip(samples, loaded):
            assert loaded_sample.label == original.label
            np.testing.assert_array_almost_equal(loaded_sample.image, original.image)
            np.testing.assert_array_almost_equal(
                loaded_sample.features, original.features
            )

    def test_ndarray_batch_stacking(self, tmp_path):
        """NDArray fields should stack into batch dimension."""
        n_samples = 32
        batch_size = 8
        img_shape = (16, 16)
        feature_dim = 64

        samples = [
            NDArraySample(
                label=i,
                image=np.full(img_shape, i, dtype=np.float32),
                features=np.full(feature_dim, i * 0.1, dtype=np.float32),
            )
            for i in range(n_samples)
        ]

        tar_path = write_single_shard(tmp_path / "stacking.tar", samples)
        dataset = atdata.Dataset[NDArraySample](tar_path)

        batches = list(dataset.ordered(batch_size=batch_size))

        for batch_idx, batch in enumerate(batches):
            # Check stacked shapes
            assert batch.image.shape == (batch_size, *img_shape)
            assert batch.features.shape == (batch_size, feature_dim)

            # Check values
            for i in range(batch_size):
                sample_idx = batch_idx * batch_size + i
                np.testing.assert_array_equal(
                    batch.image[i],
                    np.full(img_shape, sample_idx, dtype=np.float32),
                )

    def test_optional_ndarray_with_values(self, tmp_path):
        """Optional NDArray with actual values should roundtrip."""
        n_samples = 15
        samples = create_optional_samples(n_samples, include_optional=True)

        tar_path = write_single_shard(tmp_path / "optional_filled.tar", samples)
        dataset = atdata.Dataset[OptionalNDArraySample](tar_path)

        loaded = list(dataset.ordered(batch_size=None))

        for original, loaded_sample in zip(samples, loaded):
            assert loaded_sample.embeddings is not None
            np.testing.assert_array_almost_equal(
                loaded_sample.embeddings,
                original.embeddings,
            )

    def test_optional_ndarray_with_none(self, tmp_path):
        """Optional NDArray with None should roundtrip."""
        n_samples = 15
        samples = create_optional_samples(n_samples, include_optional=False)

        tar_path = write_single_shard(tmp_path / "optional_none.tar", samples)
        dataset = atdata.Dataset[OptionalNDArraySample](tar_path)

        loaded = list(dataset.ordered(batch_size=None))

        for loaded_sample in loaded:
            assert loaded_sample.embeddings is None

    def test_mixed_dtypes(self, tmp_path):
        """Various numpy dtypes should serialize correctly."""

        @atdata.packable
        class MultiDtypeSample:
            f32: NDArray
            f64: NDArray
            i32: NDArray
            i64: NDArray
            u8: NDArray

        samples = [
            MultiDtypeSample(
                f32=np.array([1.0, 2.0, 3.0], dtype=np.float32),
                f64=np.array([1.0, 2.0, 3.0], dtype=np.float64),
                i32=np.array([1, 2, 3], dtype=np.int32),
                i64=np.array([1, 2, 3], dtype=np.int64),
                u8=np.array([255, 128, 0], dtype=np.uint8),
            )
            for _ in range(10)
        ]

        tar_path = write_single_shard(tmp_path / "multidtype.tar", samples)
        dataset = atdata.Dataset[MultiDtypeSample](tar_path)

        loaded = list(dataset.ordered(batch_size=None))

        for original, loaded_sample in zip(samples, loaded):
            assert loaded_sample.f32.dtype == np.float32
            assert loaded_sample.f64.dtype == np.float64
            assert loaded_sample.i32.dtype == np.int32
            assert loaded_sample.i64.dtype == np.int64
            assert loaded_sample.u8.dtype == np.uint8
            np.testing.assert_array_equal(loaded_sample.f32, original.f32)


class TestMultiShardPipeline:
    """End-to-end tests with multi-shard datasets using brace notation."""

    def test_multi_shard_ordered_iteration(self, tmp_path):
        """Multi-shard dataset should iterate all samples in order."""
        n_samples = 100
        samples_per_shard = 10
        samples = create_simple_samples(n_samples)

        brace_pattern, n_shards = write_multi_shard(
            tmp_path,
            samples,
            samples_per_shard=samples_per_shard,
        )

        assert n_shards == 10

        dataset = atdata.Dataset[SimpleSample](brace_pattern)
        loaded = list(dataset.ordered(batch_size=None))

        assert len(loaded) == n_samples

        # Verify ordering within each shard
        for i, sample in enumerate(loaded):
            assert sample.name == f"sample_{i}"

    def test_multi_shard_batched(self, tmp_path):
        """Multi-shard dataset with batching should work correctly."""
        n_samples = 120
        samples_per_shard = 15
        batch_size = 8
        samples = create_simple_samples(n_samples)

        brace_pattern, n_shards = write_multi_shard(
            tmp_path,
            samples,
            samples_per_shard=samples_per_shard,
        )

        dataset = atdata.Dataset[SimpleSample](brace_pattern)
        batches = list(dataset.ordered(batch_size=batch_size))

        # Total samples should match
        total_samples = sum(len(b.samples) for b in batches)
        assert total_samples == (n_samples // batch_size) * batch_size

    def test_multi_shard_shuffled(self, tmp_path):
        """Multi-shard shuffled iteration should work."""
        n_samples = 50
        samples_per_shard = 10
        samples = create_simple_samples(n_samples)

        brace_pattern, _ = write_multi_shard(
            tmp_path,
            samples,
            samples_per_shard=samples_per_shard,
        )

        dataset = atdata.Dataset[SimpleSample](brace_pattern)

        # Collect some samples from shuffled iteration
        shuffled_samples = []
        for sample in dataset.shuffled(batch_size=None):
            shuffled_samples.append(sample)
            if len(shuffled_samples) >= 30:
                break

        assert len(shuffled_samples) == 30

        # All samples should be valid SimpleSample instances
        for sample in shuffled_samples:
            assert isinstance(sample, SimpleSample)
            assert sample.name.startswith("sample_")

    def test_single_shard_via_brace_pattern(self, tmp_path):
        """Single shard via brace pattern should work."""
        n_samples = 25
        samples = create_simple_samples(n_samples)

        # Create exactly one shard
        brace_pattern, n_shards = write_multi_shard(
            tmp_path,
            samples,
            samples_per_shard=100,  # More than samples, so single shard
        )

        assert n_shards == 1

        dataset = atdata.Dataset[SimpleSample](brace_pattern)
        loaded = list(dataset.ordered(batch_size=None))

        assert len(loaded) == n_samples


class TestLargeBatchHandling:
    """Tests for handling large batches and many samples."""

    def test_large_batch_size(self, tmp_path):
        """Large batch sizes should work correctly."""
        n_samples = 200
        batch_size = 64
        samples = create_simple_samples(n_samples)

        tar_path = write_single_shard(tmp_path / "large_batch.tar", samples)
        dataset = atdata.Dataset[SimpleSample](tar_path)

        batches = list(dataset.ordered(batch_size=batch_size))

        # Verify we got the expected number of complete batches
        total_samples = sum(len(b.samples) for b in batches)
        assert total_samples >= (n_samples // batch_size) * batch_size
        for batch in batches:
            assert len(batch.samples) <= batch_size

    def test_many_samples_single_shard(self, tmp_path):
        """Many samples in single shard should work."""
        n_samples = 500
        samples = create_simple_samples(n_samples)

        tar_path = write_single_shard(tmp_path / "many.tar", samples)
        dataset = atdata.Dataset[SimpleSample](tar_path)

        loaded = list(dataset.ordered(batch_size=None))
        assert len(loaded) == n_samples

    def test_large_ndarray_samples(self, tmp_path):
        """Large NDArray fields should serialize correctly."""
        n_samples = 10
        large_shape = (256, 256)  # Larger images

        samples = create_ndarray_samples(n_samples, img_shape=large_shape)

        tar_path = write_single_shard(tmp_path / "large_ndarray.tar", samples)
        dataset = atdata.Dataset[NDArraySample](tar_path)

        loaded = list(dataset.ordered(batch_size=None))

        for original, loaded_sample in zip(samples, loaded):
            assert loaded_sample.image.shape == large_shape
            np.testing.assert_array_almost_equal(
                loaded_sample.image,
                original.image,
            )


class TestBytesAndListSamples:
    """Tests for bytes and list field types."""

    def test_bytes_field_roundtrip(self, tmp_path):
        """Bytes fields should roundtrip correctly."""
        samples = [
            BytesSample(
                name=f"item_{i}",
                raw_data=f"binary_data_{i}".encode("utf-8"),
            )
            for i in range(20)
        ]

        tar_path = write_single_shard(tmp_path / "bytes.tar", samples)
        dataset = atdata.Dataset[BytesSample](tar_path)

        loaded = list(dataset.ordered(batch_size=None))

        for original, loaded_sample in zip(samples, loaded):
            assert loaded_sample.name == original.name
            assert loaded_sample.raw_data == original.raw_data

    def test_list_fields_roundtrip(self, tmp_path):
        """List fields should roundtrip correctly."""
        samples = [
            ListSample(
                tags=[f"tag_{j}" for j in range(3)],
                scores=[float(j) * 0.1 for j in range(5)],
                ids=[i * 10 + j for j in range(4)],
            )
            for i in range(15)
        ]

        tar_path = write_single_shard(tmp_path / "lists.tar", samples)
        dataset = atdata.Dataset[ListSample](tar_path)

        loaded = list(dataset.ordered(batch_size=None))

        for original, loaded_sample in zip(samples, loaded):
            assert loaded_sample.tags == original.tags
            assert loaded_sample.scores == original.scores
            assert loaded_sample.ids == original.ids


class TestMetadataRoundTrip:
    """Tests for metadata preservation through the pipeline."""

    def test_dataset_with_metadata_url(self, tmp_path):
        """Dataset with metadata_url should fetch and cache metadata."""
        from unittest.mock import Mock, patch, MagicMock
        import msgpack

        samples = create_simple_samples(10)
        tar_path = write_single_shard(tmp_path / "meta.tar", samples)

        test_metadata = {
            "version": "1.0.0",
            "created_by": "test",
            "sample_count": 10,
            "nested": {"key": "value"},
        }

        # Create a proper mock that supports context manager protocol
        mock_response = MagicMock()
        mock_response.content = msgpack.packb(test_metadata)
        mock_response.raise_for_status = Mock()
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)

        with patch("atdata.dataset.requests.get", return_value=mock_response):
            dataset = atdata.Dataset[SimpleSample](
                tar_path,
                metadata_url="http://example.com/meta.msgpack",
            )

            # Fetch metadata
            metadata = dataset.metadata

            assert metadata == test_metadata
            assert metadata["version"] == "1.0.0"
            assert metadata["nested"]["key"] == "value"

            # Second access should use cache
            metadata2 = dataset.metadata
            assert metadata2 == test_metadata


class TestParquetExport:
    """Tests for Parquet export functionality."""

    def test_simple_parquet_export(self, tmp_path):
        """Simple samples should export to Parquet correctly."""
        import pandas as pd

        n_samples = 50
        samples = create_simple_samples(n_samples)

        tar_path = write_single_shard(tmp_path / "for_parquet.tar", samples)
        dataset = atdata.Dataset[SimpleSample](tar_path)

        parquet_path = tmp_path / "output.parquet"
        dataset.to_parquet(parquet_path)

        # Verify Parquet file
        df = pd.read_parquet(parquet_path)
        assert len(df) == n_samples
        assert list(df.columns) == ["name", "value", "score", "active"]
        assert df["name"].iloc[0] == "sample_0"
        assert df["value"].iloc[0] == 0

    def test_parquet_export_with_maxcount(self, tmp_path):
        """Parquet export with maxcount should create segments."""
        import pandas as pd

        n_samples = 45
        maxcount = 10
        samples = create_simple_samples(n_samples)

        tar_path = write_single_shard(tmp_path / "segmented.tar", samples)
        dataset = atdata.Dataset[SimpleSample](tar_path)

        parquet_path = tmp_path / "segments.parquet"
        dataset.to_parquet(parquet_path, maxcount=maxcount)

        # Should create 5 segment files (45 samples / 10 per file)
        segment_files = list(tmp_path.glob("segments-*.parquet"))
        assert len(segment_files) == 5

        # Total rows should match
        total_rows = sum(len(pd.read_parquet(f)) for f in segment_files)
        assert total_rows == n_samples


class TestIterationModes:
    """Tests for different iteration modes."""

    def test_ordered_is_deterministic(self, tmp_path):
        """Ordered iteration should be deterministic across multiple passes."""
        n_samples = 30
        samples = create_simple_samples(n_samples)

        tar_path = write_single_shard(tmp_path / "ordered.tar", samples)
        dataset = atdata.Dataset[SimpleSample](tar_path)

        # Two passes should yield identical results
        pass1 = [s.name for s in dataset.ordered(batch_size=None)]
        pass2 = [s.name for s in dataset.ordered(batch_size=None)]

        assert pass1 == pass2

    def test_shuffled_changes_order(self, tmp_path):
        """Shuffled iteration should change order (with high probability)."""
        n_samples = 100
        samples = create_simple_samples(n_samples)

        tar_path = write_single_shard(tmp_path / "shuffle_test.tar", samples)
        dataset = atdata.Dataset[SimpleSample](tar_path)

        # Collect samples from multiple shuffled passes
        passes = []
        for _ in range(3):
            names = []
            for sample in dataset.shuffled(batch_size=None):
                names.append(sample.name)
                if len(names) >= n_samples:
                    break
            passes.append(names)

        # At least two passes should differ (very high probability with 100 samples)
        # Note: This could theoretically fail, but probability is astronomically low
        assert (
            passes[0] != passes[1] or passes[1] != passes[2] or passes[0] != passes[2]
        )

    def test_batch_size_one(self, tmp_path):
        """batch_size=1 should return single-element batches."""
        n_samples = 10
        samples = create_simple_samples(n_samples)

        tar_path = write_single_shard(tmp_path / "batch1.tar", samples)
        dataset = atdata.Dataset[SimpleSample](tar_path)

        batches = list(dataset.ordered(batch_size=1))

        assert len(batches) == n_samples
        for batch in batches:
            assert isinstance(batch, atdata.SampleBatch)
            assert len(batch.samples) == 1
