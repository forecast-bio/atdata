"""Performance benchmarks for dataset read/write operations.

Measures shard writing throughput, read iteration speed, serialization
overhead, and round-trip performance for basic and numpy sample types.
"""

from __future__ import annotations

import numpy as np
import pytest
import webdataset as wds

import atdata

from .conftest import (
    IMAGE_DTYPE,
    IMAGE_SHAPE,
    TSERIES_DTYPE,
    TSERIES_SHAPE,
    BenchBasicSample,
    BenchManifestSample,
    BenchNumpySample,
    generate_basic_samples,
    generate_manifest_samples,
    generate_numpy_samples,
    write_tar,
    write_tar_with_manifest,
)


# =============================================================================
# Write Benchmarks
# =============================================================================


@pytest.mark.bench_io
class TestShardWriteBenchmarks:
    """Shard writing throughput benchmarks."""

    PARAM_LABELS = {"n": "samples per shard"}

    @pytest.mark.parametrize("n", [100, 1000, 10000], ids=["100", "1k", "10k"])
    def test_write_basic_shard(self, benchmark, tmp_path, n):
        benchmark.extra_info["n_samples"] = n
        samples = generate_basic_samples(n)

        def _write():
            tar_path = tmp_path / f"basic-{n}.tar"
            write_tar(tar_path, samples)

        benchmark(_write)

    @pytest.mark.parametrize("n", [100, 1000], ids=["100", "1k"])
    def test_write_numpy_shard(self, benchmark, tmp_path, n):
        benchmark.extra_info["n_samples"] = n
        samples = generate_numpy_samples(n)

        def _write():
            tar_path = tmp_path / f"numpy-{n}.tar"
            write_tar(tar_path, samples)

        benchmark(_write)

    def test_write_large_numpy_shard(self, benchmark, tmp_path):
        benchmark.extra_info["n_samples"] = 10
        samples = generate_numpy_samples(10, shape=TSERIES_SHAPE, dtype=TSERIES_DTYPE)

        def _write():
            tar_path = tmp_path / "numpy-large.tar"
            write_tar(tar_path, samples)

        benchmark(_write)

    @pytest.mark.parametrize("n", [100, 1000], ids=["100", "1k"])
    def test_write_with_manifest(self, benchmark, tmp_path, n):
        benchmark.extra_info["n_samples"] = n
        samples = generate_manifest_samples(n)
        counter = [0]

        def _write():
            idx = counter[0]
            counter[0] += 1
            tar_path = tmp_path / f"manifest-{n}-{idx}.tar"
            write_tar_with_manifest(tar_path, samples, BenchManifestSample)

        benchmark(_write)

    def test_write_multi_shard(self, benchmark, tmp_path):
        benchmark.extra_info["n_samples"] = 10000
        samples = generate_basic_samples(10000)
        counter = [0]

        def _write():
            idx = counter[0]
            counter[0] += 1
            out_dir = tmp_path / f"multi-{idx}"
            out_dir.mkdir(exist_ok=True)
            pattern = str(out_dir / "shard-%06d.tar")
            with wds.writer.ShardWriter(pattern, maxcount=1000) as sink:
                for sample in samples:
                    sink.write(sample.as_wds)

        benchmark(_write)


# =============================================================================
# Read Benchmarks
# =============================================================================


@pytest.mark.bench_io
class TestShardReadBenchmarks:
    """Shard reading and iteration benchmarks."""

    PARAM_LABELS = {"n": "samples in dataset", "batch_size": "samples per batch"}

    @pytest.mark.parametrize("n", [100, 1000, 10000], ids=["100", "1k", "10k"])
    def test_read_ordered(self, benchmark, tmp_path, n):
        benchmark.extra_info["n_samples"] = n
        samples = generate_basic_samples(n)
        tar_path = write_tar(tmp_path / f"read-ordered-{n}.tar", samples)
        ds = atdata.Dataset[BenchBasicSample](url=str(tar_path))

        def _read():
            count = 0
            for _ in ds.ordered():
                count += 1
            return count

        result = benchmark(_read)
        assert result == n

    @pytest.mark.parametrize("n", [100, 1000], ids=["100", "1k"])
    def test_read_shuffled(self, benchmark, tmp_path, n):
        benchmark.extra_info["n_samples"] = n
        samples = generate_basic_samples(n)
        tar_path = write_tar(tmp_path / f"read-shuffled-{n}.tar", samples)
        ds = atdata.Dataset[BenchBasicSample](url=str(tar_path))

        def _read():
            count = 0
            for _ in ds.shuffled():
                count += 1
            return count

        result = benchmark(_read)
        assert result == n

    @pytest.mark.parametrize(
        "batch_size", [32, 128], ids=["batch32", "batch128"]
    )
    def test_read_batched(self, benchmark, tmp_path, batch_size):
        n = 1000
        benchmark.extra_info["n_samples"] = n
        samples = generate_basic_samples(n)
        tar_path = write_tar(tmp_path / f"read-batched-{batch_size}.tar", samples)
        ds = atdata.Dataset[BenchBasicSample](url=str(tar_path))

        def _read():
            count = 0
            for batch in ds.ordered(batch_size=batch_size):
                count += 1
            return count

        benchmark(_read)

    @pytest.mark.parametrize("n", [100, 1000], ids=["100", "1k"])
    def test_read_numpy_ordered(self, benchmark, tmp_path, n):
        benchmark.extra_info["n_samples"] = n
        samples = generate_numpy_samples(n)
        tar_path = write_tar(tmp_path / f"read-numpy-{n}.tar", samples)
        ds = atdata.Dataset[BenchNumpySample](url=str(tar_path))

        def _read():
            count = 0
            for _ in ds.ordered():
                count += 1
            return count

        result = benchmark(_read)
        assert result == n


# =============================================================================
# Serialization Benchmarks (No I/O)
# =============================================================================


@pytest.mark.bench_serial
class TestSerializationBenchmarks:
    """Pure serialization/deserialization without disk I/O."""

    def test_serialize_basic_sample(self, benchmark):
        sample = BenchBasicSample(name="bench_sample", value=42)
        benchmark(lambda: sample.packed)

    def test_deserialize_basic_sample(self, benchmark):
        sample = BenchBasicSample(name="bench_sample", value=42)
        packed = sample.packed
        benchmark(BenchBasicSample.from_bytes, packed)

    def test_serialize_numpy_sample(self, benchmark):
        sample = BenchNumpySample(
            data=np.random.randint(0, 256, size=IMAGE_SHAPE, dtype=IMAGE_DTYPE),
            label="bench",
        )
        benchmark(lambda: sample.packed)

    def test_deserialize_numpy_sample(self, benchmark):
        sample = BenchNumpySample(
            data=np.random.randint(0, 256, size=IMAGE_SHAPE, dtype=IMAGE_DTYPE),
            label="bench",
        )
        packed = sample.packed
        benchmark(BenchNumpySample.from_bytes, packed)

    def test_serialize_large_numpy(self, benchmark):
        sample = BenchNumpySample(
            data=np.random.randn(*TSERIES_SHAPE).astype(TSERIES_DTYPE),
            label="large",
        )
        benchmark(lambda: sample.packed)

    def test_deserialize_large_numpy(self, benchmark):
        sample = BenchNumpySample(
            data=np.random.randn(*TSERIES_SHAPE).astype(TSERIES_DTYPE),
            label="large",
        )
        packed = sample.packed
        benchmark(BenchNumpySample.from_bytes, packed)

    def test_as_wds_basic(self, benchmark):
        sample = BenchBasicSample(name="bench_sample", value=42)
        benchmark(lambda: sample.as_wds)

    def test_as_wds_numpy(self, benchmark):
        sample = BenchNumpySample(
            data=np.random.randint(0, 256, size=IMAGE_SHAPE, dtype=IMAGE_DTYPE),
            label="bench",
        )
        benchmark(lambda: sample.as_wds)


# =============================================================================
# Round-Trip Benchmarks
# =============================================================================


@pytest.mark.bench_io
class TestRoundTripBenchmarks:
    """Full write-then-read round-trip benchmarks."""

    PARAM_LABELS = {"n": "samples round-tripped"}

    @pytest.mark.parametrize("n", [100, 1000], ids=["100", "1k"])
    def test_roundtrip_basic(self, benchmark, tmp_path, n):
        benchmark.extra_info["n_samples"] = n
        samples = generate_basic_samples(n)
        counter = [0]

        def _roundtrip():
            idx = counter[0]
            counter[0] += 1
            tar_path = tmp_path / f"rt-basic-{n}-{idx}.tar"
            write_tar(tar_path, samples)
            ds = atdata.Dataset[BenchBasicSample](url=str(tar_path))
            count = 0
            for _ in ds.ordered():
                count += 1
            return count

        result = benchmark(_roundtrip)
        assert result == n

    @pytest.mark.parametrize("n", [100, 500], ids=["100", "500"])
    def test_roundtrip_numpy(self, benchmark, tmp_path, n):
        benchmark.extra_info["n_samples"] = n
        samples = generate_numpy_samples(n)
        counter = [0]

        def _roundtrip():
            idx = counter[0]
            counter[0] += 1
            tar_path = tmp_path / f"rt-numpy-{n}-{idx}.tar"
            write_tar(tar_path, samples)
            ds = atdata.Dataset[BenchNumpySample](url=str(tar_path))
            count = 0
            for _ in ds.ordered():
                count += 1
            return count

        result = benchmark(_roundtrip)
        assert result == n
