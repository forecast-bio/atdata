"""Performance benchmarks for remote storage backends.

Covers S3DataStore (via moto mock) and Atmosphere/ATProto (network-gated).
S3 benchmarks use moto for reproducible local measurement.
Atmosphere benchmarks are marked ``network`` and skip unless a live PDS is available.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from moto import mock_aws

import atdata

from .conftest import (
    IMAGE_SHAPE,
    BenchBasicSample,
    BenchManifestSample,
    BenchNumpySample,
    generate_basic_samples,
    generate_manifest_samples,
    generate_numpy_samples,
    write_tar,
)


# =============================================================================
# S3 Fixtures
# =============================================================================


@pytest.fixture
def mock_s3():
    """Provide mock S3 environment using moto."""
    with mock_aws():
        import boto3

        creds = {
            "AWS_ACCESS_KEY_ID": "testing",
            "AWS_SECRET_ACCESS_KEY": "testing",
        }
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=creds["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=creds["AWS_SECRET_ACCESS_KEY"],
            region_name="us-east-1",
        )
        bucket_name = "bench-bucket"
        s3_client.create_bucket(Bucket=bucket_name)
        yield {
            "credentials": creds,
            "bucket": bucket_name,
        }


def _make_s3_store(mock_s3_env):
    from atdata.local._s3 import S3DataStore

    return S3DataStore(
        credentials=mock_s3_env["credentials"],
        bucket=mock_s3_env["bucket"],
    )


def _make_source_dataset(tmp_path, samples):
    """Create a local dataset from samples for use as S3 write source."""
    tar_path = write_tar(tmp_path / "source-000000.tar", samples)
    sample_type = type(samples[0])
    return atdata.Dataset[sample_type](url=str(tar_path))


# =============================================================================
# S3 Write Benchmarks
# =============================================================================


@pytest.mark.bench_s3
@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
@pytest.mark.filterwarnings("ignore:coroutine.*was never awaited:RuntimeWarning")
class TestS3WriteBenchmarks:
    """S3 shard writing benchmarks via moto mock."""

    PARAM_LABELS = {"n": "samples per shard"}

    @pytest.mark.parametrize("n", [100, 500], ids=["100", "500"])
    def test_s3_write_shards(self, benchmark, tmp_path, mock_s3, n):
        benchmark.extra_info["n_samples"] = n
        samples = generate_basic_samples(n)
        ds = _make_source_dataset(tmp_path, samples)
        store = _make_s3_store(mock_s3)
        counter = [0]

        def _write():
            idx = counter[0]
            counter[0] += 1
            store.write_shards(ds, prefix=f"bench/basic-{n}-{idx}")

        benchmark(_write)

    def test_s3_write_with_manifest(self, benchmark, tmp_path, mock_s3):
        benchmark.extra_info["n_samples"] = 200
        samples = generate_manifest_samples(200)
        ds = _make_source_dataset(tmp_path, samples)
        store = _make_s3_store(mock_s3)
        counter = [0]

        def _write():
            idx = counter[0]
            counter[0] += 1
            store.write_shards(
                ds, prefix=f"bench/manifest-{idx}", manifest=True,
                cache_local=True,
            )

        benchmark(_write)

    def test_s3_write_cache_local(self, benchmark, tmp_path, mock_s3):
        benchmark.extra_info["n_samples"] = 200
        samples = generate_basic_samples(200)
        ds = _make_source_dataset(tmp_path, samples)
        store = _make_s3_store(mock_s3)
        counter = [0]

        def _write():
            idx = counter[0]
            counter[0] += 1
            store.write_shards(
                ds, prefix=f"bench/cache-{idx}", cache_local=True
            )

        benchmark(_write)

    def test_s3_write_direct(self, benchmark, tmp_path, mock_s3):
        benchmark.extra_info["n_samples"] = 200
        samples = generate_basic_samples(200)
        ds = _make_source_dataset(tmp_path, samples)
        store = _make_s3_store(mock_s3)
        counter = [0]

        def _write():
            idx = counter[0]
            counter[0] += 1
            store.write_shards(
                ds, prefix=f"bench/direct-{idx}", cache_local=False
            )

        benchmark(_write)

    def test_s3_write_numpy(self, benchmark, tmp_path, mock_s3):
        benchmark.extra_info["n_samples"] = 100
        samples = generate_numpy_samples(100)
        ds = _make_source_dataset(tmp_path, samples)
        store = _make_s3_store(mock_s3)
        counter = [0]

        def _write():
            idx = counter[0]
            counter[0] += 1
            store.write_shards(ds, prefix=f"bench/numpy-{idx}")

        benchmark(_write)


# =============================================================================
# Atmosphere Benchmarks (network-gated)
# =============================================================================


@pytest.mark.network
class TestAtmosphereBenchmarks:
    """Atmosphere/ATProto benchmarks. Require live PDS access.

    Run with: just bench -m network
    """

    def test_atmosphere_publish_dataset(self, benchmark, tmp_path):
        """End-to-end dataset publish to Atmosphere."""
        import os

        handle = os.environ.get("ATDATA_BENCH_ATP_HANDLE")
        password = os.environ.get("ATDATA_BENCH_ATP_PASSWORD")
        if not handle or not password:
            pytest.skip("ATDATA_BENCH_ATP_HANDLE/PASSWORD not set")

        from atdata.atmosphere.client import AtmosphereClient

        client = AtmosphereClient(handle=handle, password=password)

        samples = generate_basic_samples(10)
        tar_path = write_tar(tmp_path / "atmo-000000.tar", samples)
        ds = atdata.Dataset[BenchBasicSample](url=str(tar_path))

        counter = [0]

        def _publish():
            idx = counter[0]
            counter[0] += 1
            from atdata.atmosphere.records import DatasetPublisher

            publisher = DatasetPublisher(client)
            publisher.publish(ds, name=f"bench-atmo-{idx}")

        benchmark(_publish)

    def test_atmosphere_resolve_dataset(self, benchmark):
        """Resolve a dataset record from Atmosphere (read-only, anonymous)."""
        import os

        ref = os.environ.get("ATDATA_BENCH_ATP_DATASET_REF")
        if not ref:
            pytest.skip("ATDATA_BENCH_ATP_DATASET_REF not set")

        from atdata.local._index import Index

        index = Index()

        benchmark(index.get_dataset, ref)
