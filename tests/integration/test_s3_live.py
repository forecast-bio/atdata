"""Live S3/MinIO integration tests.

Exercises ``S3DataStore`` against a real MinIO service container.
Requires ``MINIO_ENDPOINT``, ``MINIO_ACCESS_KEY``, and
``MINIO_SECRET_KEY`` environment variables.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import atdata
from atdata.stores._s3 import S3DataStore
from numpy.typing import NDArray

from .conftest import unique_name

# ── Sample types ──────────────────────────────────────────────────


@atdata.packable
class S3BasicSample:
    name: str
    value: int


@atdata.packable
class S3ArraySample:
    label: str
    data: NDArray


# ── Write shards ──────────────────────────────────────────────────


class TestWriteShards:
    """Write dataset shards to MinIO and verify they land correctly."""

    def test_write_basic_samples(
        self,
        minio_credentials: dict[str, str],
        minio_bucket: str,
        tmp_path: Path,
    ):
        store = S3DataStore(credentials=minio_credentials, bucket=minio_bucket)

        samples = [S3BasicSample(name=f"s-{i}", value=i) for i in range(20)]
        tar_path = tmp_path / "basic.tar"
        ds = atdata.write_samples(samples, tar_path)

        prefix = f"test/{unique_name('basic')}"
        urls = store.write_shards(ds, prefix=prefix)

        assert len(urls) >= 1
        for url in urls:
            assert url.startswith("s3://")
            assert minio_bucket in url

    def test_write_array_samples(
        self,
        minio_credentials: dict[str, str],
        minio_bucket: str,
        tmp_path: Path,
    ):
        store = S3DataStore(credentials=minio_credentials, bucket=minio_bucket)

        samples = [
            S3ArraySample(
                label=f"arr-{i}", data=np.random.randn(8, 8).astype(np.float32)
            )
            for i in range(5)
        ]
        tar_path = tmp_path / "arrays.tar"
        ds = atdata.write_samples(samples, tar_path)

        prefix = f"test/{unique_name('array')}"
        urls = store.write_shards(ds, prefix=prefix)

        assert len(urls) >= 1

    def test_write_with_cache_local(
        self,
        minio_credentials: dict[str, str],
        minio_bucket: str,
        tmp_path: Path,
    ):
        store = S3DataStore(credentials=minio_credentials, bucket=minio_bucket)

        samples = [S3BasicSample(name=f"cached-{i}", value=i) for i in range(10)]
        tar_path = tmp_path / "cached.tar"
        ds = atdata.write_samples(samples, tar_path)

        prefix = f"test/{unique_name('cached')}"
        urls = store.write_shards(ds, prefix=prefix, cache_local=True)

        assert len(urls) >= 1
        for url in urls:
            assert url.startswith("s3://")

    def test_write_with_maxcount(
        self,
        minio_credentials: dict[str, str],
        minio_bucket: str,
        tmp_path: Path,
    ):
        """Verify maxcount splits into multiple shards."""
        store = S3DataStore(credentials=minio_credentials, bucket=minio_bucket)

        samples = [S3BasicSample(name=f"split-{i}", value=i) for i in range(20)]
        tar_path = tmp_path / "split.tar"
        ds = atdata.write_samples(samples, tar_path)

        prefix = f"test/{unique_name('split')}"
        urls = store.write_shards(ds, prefix=prefix, maxcount=5)

        assert len(urls) == 4  # 20 samples / 5 per shard


# ── Read-back via URL resolution ──────────────────────────────────


class TestURLResolution:
    """Verify ``read_url`` produces usable HTTPS URLs for MinIO."""

    def test_read_url_converts_to_https(
        self,
        minio_credentials: dict[str, str],
        minio_bucket: str,
    ):
        store = S3DataStore(credentials=minio_credentials, bucket=minio_bucket)
        resolved = store.read_url(f"s3://{minio_bucket}/some/path.tar")

        endpoint = minio_credentials["AWS_ENDPOINT"].rstrip("/")
        assert resolved == f"{endpoint}/{minio_bucket}/some/path.tar"

    def test_supports_streaming(
        self,
        minio_credentials: dict[str, str],
        minio_bucket: str,
    ):
        store = S3DataStore(credentials=minio_credentials, bucket=minio_bucket)
        assert store.supports_streaming() is True


# ── Write + read round-trip via Index ─────────────────────────────


class TestS3IndexRoundTrip:
    """Write samples through Index with S3DataStore, then read back."""

    def test_index_write_and_read(
        self,
        minio_credentials: dict[str, str],
        minio_bucket: str,
        tmp_path: Path,
    ):
        store = S3DataStore(credentials=minio_credentials, bucket=minio_bucket)
        index = atdata.Index(
            provider="sqlite",
            path=str(tmp_path / "index.db"),
            data_store=store,
            atmosphere=None,
        )

        samples = [S3BasicSample(name=f"idx-{i}", value=i * 10) for i in range(10)]
        name = unique_name("idx-rt")
        entry = index.write_samples(samples, name=name)

        assert entry.name == name
        assert len(entry.data_urls) >= 1
        for url in entry.data_urls:
            assert url.startswith("s3://")

        # Retrieve entry by name
        fetched = index.get_dataset(name)
        assert fetched.name == name
        assert fetched.data_urls == entry.data_urls


# ── Manifest generation ──────────────────────────────────────────


class TestS3Manifest:
    """Verify manifest files are written alongside shards."""

    def test_write_with_manifest(
        self,
        minio_credentials: dict[str, str],
        minio_bucket: str,
        tmp_path: Path,
    ):
        store = S3DataStore(credentials=minio_credentials, bucket=minio_bucket)

        samples = [S3BasicSample(name=f"mfst-{i}", value=i) for i in range(10)]
        tar_path = tmp_path / "manifest.tar"
        ds = atdata.write_samples(samples, tar_path)

        prefix = f"test/{unique_name('manifest')}"
        urls = store.write_shards(ds, prefix=prefix, manifest=True)

        assert len(urls) >= 1

        # Check that manifest files exist in the bucket
        import boto3

        s3 = boto3.client(
            "s3",
            endpoint_url=minio_credentials["AWS_ENDPOINT"],
            aws_access_key_id=minio_credentials["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=minio_credentials["AWS_SECRET_ACCESS_KEY"],
        )
        resp = s3.list_objects_v2(Bucket=minio_bucket, Prefix="test/")
        keys = [obj["Key"] for obj in resp.get("Contents", [])]
        manifest_keys = [k for k in keys if ".manifest." in k]
        assert len(manifest_keys) >= 1
