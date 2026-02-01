"""Tests for atdata.write_samples() function."""

from pathlib import Path

import numpy as np
import pytest

import atdata
from conftest import SharedBasicSample, SharedNumpySample


class TestWriteSamplesSingleTar:
    """Tests for single-file (non-sharded) write_samples."""

    def test_basic_roundtrip(self, tmp_path: Path):
        samples = [SharedBasicSample(name=f"s{i}", value=i) for i in range(5)]
        ds = atdata.write_samples(samples, tmp_path / "out.tar")

        result = list(ds.ordered())
        assert len(result) == 5
        for i, s in enumerate(result):
            assert s.name == f"s{i}"
            assert s.value == i

    def test_returns_typed_dataset(self, tmp_path: Path):
        samples = [SharedBasicSample(name="x", value=1)]
        ds = atdata.write_samples(samples, tmp_path / "out.tar")

        assert isinstance(ds, atdata.Dataset)
        assert ds.sample_type is SharedBasicSample

    def test_numpy_roundtrip(self, tmp_path: Path):
        arrays = [np.random.randn(4, 4).astype(np.float32) for _ in range(3)]
        samples = [
            SharedNumpySample(data=arr, label=f"arr{i}")
            for i, arr in enumerate(arrays)
        ]
        ds = atdata.write_samples(samples, tmp_path / "out.tar")

        result = list(ds.ordered())
        assert len(result) == 3
        for i, s in enumerate(result):
            assert s.label == f"arr{i}"
            np.testing.assert_array_almost_equal(s.data, arrays[i])

    def test_creates_parent_dirs(self, tmp_path: Path):
        samples = [SharedBasicSample(name="x", value=0)]
        out = tmp_path / "nested" / "deep" / "out.tar"
        ds = atdata.write_samples(samples, out)

        assert out.exists()
        assert len(list(ds.ordered())) == 1

    def test_single_sample(self, tmp_path: Path):
        ds = atdata.write_samples(
            [SharedBasicSample(name="only", value=42)],
            tmp_path / "out.tar",
        )
        result = list(ds.ordered())
        assert len(result) == 1
        assert result[0].name == "only"


class TestWriteSamplesSharded:
    """Tests for sharded (multi-file) write_samples."""

    def test_maxcount_creates_multiple_shards(self, tmp_path: Path):
        samples = [SharedBasicSample(name=f"s{i}", value=i) for i in range(10)]
        ds = atdata.write_samples(
            samples, tmp_path / "data.tar", maxcount=3
        )

        # Should have created multiple shard files
        tar_files = list(tmp_path.glob("data-*.tar"))
        assert len(tar_files) >= 2

        # All samples should be readable
        result = list(ds.ordered())
        assert len(result) == 10

    def test_sharded_preserves_data(self, tmp_path: Path):
        samples = [SharedBasicSample(name=f"s{i}", value=i * 10) for i in range(8)]
        ds = atdata.write_samples(
            samples, tmp_path / "data.tar", maxcount=3
        )

        result = sorted(list(ds.ordered()), key=lambda s: s.value)
        for i, s in enumerate(result):
            assert s.name == f"s{i}"
            assert s.value == i * 10

    def test_custom_pattern_with_percent(self, tmp_path: Path):
        samples = [SharedBasicSample(name=f"s{i}", value=i) for i in range(6)]
        pattern = tmp_path / "shard-%04d.tar"
        ds = atdata.write_samples(
            samples, pattern, maxcount=3
        )

        # Check that shards were created with the custom pattern
        assert (tmp_path / "shard-0000.tar").exists()
        result = list(ds.ordered())
        assert len(result) == 6


class TestWriteSamplesEdgeCases:
    """Tests for error handling and edge cases."""

    def test_empty_samples_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="non-empty"):
            atdata.write_samples([], tmp_path / "empty.tar")

    def test_generator_input(self, tmp_path: Path):
        def gen():
            for i in range(5):
                yield SharedBasicSample(name=f"g{i}", value=i)

        ds = atdata.write_samples(gen(), tmp_path / "out.tar")
        assert len(list(ds.ordered())) == 5
