"""Performance benchmarks for the manifest query system.

Measures query execution speed across different predicate types,
manifest loading performance, and scaling behavior with increasing
shard counts and sample sizes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from atdata.manifest import ManifestBuilder, ManifestWriter, QueryExecutor, ShardManifest

from .conftest import (
    BenchManifestSample,
    generate_manifest_samples,
    create_sharded_dataset,
    write_tar_with_manifest,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def query_dataset_small(tmp_path):
    """Small query dataset: 2 shards x 50 samples = 100 total."""
    samples = generate_manifest_samples(100)
    create_sharded_dataset(
        tmp_path, samples, 50, BenchManifestSample, with_manifests=True
    )
    executor = QueryExecutor.from_directory(tmp_path)
    return executor, tmp_path


@pytest.fixture
def query_dataset_medium(tmp_path):
    """Medium query dataset: 10 shards x 100 samples = 1000 total."""
    samples = generate_manifest_samples(1000)
    create_sharded_dataset(
        tmp_path, samples, 100, BenchManifestSample, with_manifests=True
    )
    executor = QueryExecutor.from_directory(tmp_path)
    return executor, tmp_path


@pytest.fixture
def query_dataset_large(tmp_path):
    """Large query dataset: 10 shards x 1000 samples = 10000 total."""
    samples = generate_manifest_samples(10000)
    create_sharded_dataset(
        tmp_path, samples, 1000, BenchManifestSample, with_manifests=True
    )
    executor = QueryExecutor.from_directory(tmp_path)
    return executor, tmp_path


# =============================================================================
# Query Predicate Benchmarks
# =============================================================================


class TestQueryPredicateBenchmarks:
    """Benchmark different query predicate types on a medium dataset."""

    def test_query_simple_equality(self, benchmark, query_dataset_medium):
        executor, _ = query_dataset_medium
        benchmark(executor.query, where=lambda df: df["label"] == "dog")

    def test_query_numeric_range(self, benchmark, query_dataset_medium):
        executor, _ = query_dataset_medium
        benchmark(executor.query, where=lambda df: df["confidence"] > 0.8)

    def test_query_combined(self, benchmark, query_dataset_medium):
        executor, _ = query_dataset_medium
        benchmark(
            executor.query,
            where=lambda df: (df["label"] == "dog") & (df["confidence"] > 0.8),
        )

    def test_query_isin(self, benchmark, query_dataset_medium):
        executor, _ = query_dataset_medium
        benchmark(
            executor.query,
            where=lambda df: df["label"].isin(["dog", "cat"]),
        )

    def test_query_no_results(self, benchmark, query_dataset_medium):
        executor, _ = query_dataset_medium
        benchmark(
            executor.query,
            where=lambda df: df["confidence"] > 999.0,
        )

    def test_query_all_results(self, benchmark, query_dataset_medium):
        executor, _ = query_dataset_medium
        benchmark(
            executor.query,
            where=lambda df: df["confidence"] >= 0.0,
        )


# =============================================================================
# Scale Benchmarks
# =============================================================================


class TestQueryScaleBenchmarks:
    """Benchmark query performance at different scales."""

    def test_query_small(self, benchmark, query_dataset_small):
        executor, _ = query_dataset_small
        benchmark(executor.query, where=lambda df: df["confidence"] > 0.5)

    def test_query_medium(self, benchmark, query_dataset_medium):
        executor, _ = query_dataset_medium
        benchmark(executor.query, where=lambda df: df["confidence"] > 0.5)

    def test_query_large(self, benchmark, query_dataset_large):
        executor, _ = query_dataset_large
        benchmark(executor.query, where=lambda df: df["confidence"] > 0.5)


# =============================================================================
# Manifest Loading Benchmarks
# =============================================================================


class TestManifestLoadBenchmarks:
    """Benchmark manifest loading from disk."""

    @pytest.mark.parametrize("n_shards", [2, 5, 10, 20], ids=["2s", "5s", "10s", "20s"])
    def test_load_from_directory(self, benchmark, tmp_path, n_shards):
        total = n_shards * 100
        samples = generate_manifest_samples(total)
        create_sharded_dataset(
            tmp_path, samples, 100, BenchManifestSample, with_manifests=True
        )
        benchmark(QueryExecutor.from_directory, tmp_path)

    @pytest.mark.parametrize("n_shards", [2, 5, 10], ids=["2s", "5s", "10s"])
    def test_load_from_shard_urls(self, benchmark, tmp_path, n_shards):
        total = n_shards * 100
        samples = generate_manifest_samples(total)
        tar_paths = create_sharded_dataset(
            tmp_path, samples, 100, BenchManifestSample, with_manifests=True
        )
        shard_urls = [str(p) for p in tar_paths]
        benchmark(QueryExecutor.from_shard_urls, shard_urls)


# =============================================================================
# Manifest Build Benchmarks
# =============================================================================


class TestManifestBuildBenchmarks:
    """Benchmark manifest construction from samples."""

    @pytest.mark.parametrize("n", [100, 1000, 5000], ids=["100", "1k", "5k"])
    def test_manifest_build(self, benchmark, n):
        samples = generate_manifest_samples(n)

        def _build():
            builder = ManifestBuilder(
                sample_type=BenchManifestSample,
                shard_id="bench-shard-000000",
            )
            offset = 0
            for i, sample in enumerate(samples):
                packed_size = len(sample.packed)
                builder.add_sample(
                    key=f"sample_{i:06d}",
                    offset=offset,
                    size=packed_size,
                    sample=sample,
                )
                offset += 512 + packed_size
            return builder.build()

        manifest = benchmark(_build)
        assert manifest.num_samples == n

    @pytest.mark.parametrize("n", [100, 1000], ids=["100", "1k"])
    def test_manifest_write(self, benchmark, tmp_path, n):
        samples = generate_manifest_samples(n)

        builder = ManifestBuilder(
            sample_type=BenchManifestSample,
            shard_id="bench-write-000000",
        )
        offset = 0
        for i, sample in enumerate(samples):
            packed_size = len(sample.packed)
            builder.add_sample(
                key=f"sample_{i:06d}",
                offset=offset,
                size=packed_size,
                sample=sample,
            )
            offset += 512 + packed_size
        manifest = builder.build()

        counter = [0]

        def _write():
            idx = counter[0]
            counter[0] += 1
            writer = ManifestWriter(tmp_path / f"mw-{idx:06d}")
            writer.write(manifest)

        benchmark(_write)
