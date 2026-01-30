"""Shared fixtures and helpers for performance benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pytest
import webdataset as wds
from numpy.typing import NDArray, DTypeLike

import atdata
from atdata.manifest import ManifestBuilder, ManifestField, ManifestWriter


# =============================================================================
# Benchmark Sample Types
# =============================================================================


@atdata.packable
class BenchBasicSample:
    """Lightweight sample for throughput benchmarks."""

    name: str
    value: int


@atdata.packable
class BenchNumpySample:
    """Sample with NDArray for array serialization benchmarks."""

    data: NDArray
    label: str


@atdata.packable
class BenchManifestSample:
    """Sample with manifest-annotated fields for query benchmarks."""

    data: NDArray
    label: Annotated[str, ManifestField("categorical")]
    confidence: Annotated[float, ManifestField("numeric")]
    tags: Annotated[list[str], ManifestField("set")]


# =============================================================================
# Benchmark Constants
# =============================================================================

# Standard image: 3-channel 224x224 uint8 (ImageNet-style)
IMAGE_SHAPE = (3, 224, 224)
IMAGE_DTYPE = np.uint8

# Large biological timeseries: 1024x1024 spatial x 600 frames, float32
TSERIES_SHAPE = (1024, 1024, 60)
TSERIES_DTYPE = np.float32

# Small array for manifest/overhead benchmarks (keeps manifests fast)
MANIFEST_ARRAY_SHAPE = (4, 4)
MANIFEST_ARRAY_DTYPE = np.float32


# =============================================================================
# Sample Generators
# =============================================================================

LABELS = ["dog", "cat", "bird", "fish", "horse"]
TAG_POOLS = [
    ["outdoor", "day"],
    ["indoor"],
    ["outdoor", "night"],
    ["underwater"],
    ["field", "day"],
]


def generate_basic_samples(n: int) -> list[BenchBasicSample]:
    return [BenchBasicSample(name=f"sample_{i:06d}", value=i) for i in range(n)]


def generate_numpy_samples(
    n: int,
    shape: tuple[int, ...] = IMAGE_SHAPE,
    dtype: np.dtype = IMAGE_DTYPE,
) -> list[BenchNumpySample]:
    return [
        BenchNumpySample(
            data=np.random.randint(0, 256, size=shape, dtype=dtype)
            if np.issubdtype(dtype, np.integer)
            else np.random.randn(*shape).astype(dtype),
            label=f"array_{i:06d}",
        )
        for i in range(n)
    ]


def generate_manifest_samples(
    n: int, shape: tuple[int, ...] = (4, 4)
) -> list[BenchManifestSample]:
    return [
        BenchManifestSample(
            data=np.random.randn(*shape).astype(np.float32),
            label=LABELS[i % len(LABELS)],
            confidence=0.1 + 0.9 * (i % 100) / 100.0,
            tags=TAG_POOLS[i % len(TAG_POOLS)],
        )
        for i in range(n)
    ]


# =============================================================================
# Tar/Dataset Helpers
# =============================================================================


def write_tar(tar_path: Path, samples: list) -> Path:
    """Write samples to a WebDataset tar file."""
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with wds.writer.TarWriter(str(tar_path)) as writer:
        for sample in samples:
            writer.write(sample.as_wds)
    return tar_path


def write_tar_with_manifest(
    tar_path: Path,
    samples: list,
    sample_type: type,
) -> tuple[Path, Path, Path]:
    """Write samples to a tar file and generate companion manifest files.

    Returns:
        Tuple of (tar_path, json_path, parquet_path).
    """
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    shard_name = tar_path.stem
    shard_id = str(tar_path.parent / shard_name)

    builder = ManifestBuilder(sample_type=sample_type, shard_id=shard_id)

    offset = 0
    with wds.writer.TarWriter(str(tar_path)) as writer:
        for sample in samples:
            wds_dict = sample.as_wds
            writer.write(wds_dict)
            packed_size = len(wds_dict.get("msgpack", b""))
            builder.add_sample(
                key=wds_dict["__key__"],
                offset=offset,
                size=packed_size,
                sample=sample,
            )
            offset += 512 + packed_size + (512 - packed_size % 512) % 512

    manifest = builder.build()
    manifest_writer = ManifestWriter(tar_path.parent / shard_name)
    json_path, parquet_path = manifest_writer.write(manifest)

    return tar_path, json_path, parquet_path


def create_sharded_dataset(
    base_dir: Path,
    samples: list,
    samples_per_shard: int,
    sample_type: type,
    with_manifests: bool = False,
) -> list[Path]:
    """Split samples across multiple shards. Returns list of tar paths."""
    tar_paths: list[Path] = []
    for shard_idx in range(0, len(samples), samples_per_shard):
        chunk = samples[shard_idx : shard_idx + samples_per_shard]
        shard_name = f"data-{shard_idx // samples_per_shard:06d}"
        tar_path = base_dir / f"{shard_name}.tar"

        if with_manifests:
            write_tar_with_manifest(tar_path, chunk, sample_type)
        else:
            write_tar(tar_path, chunk)

        tar_paths.append(tar_path)

    return tar_paths


# =============================================================================
# Provider Fixtures
# =============================================================================


@pytest.fixture
def sqlite_provider(tmp_path):
    """Fresh SQLite provider in a temp directory."""
    from atdata.providers._sqlite import SqliteProvider

    provider = SqliteProvider(path=tmp_path / "bench.db")
    yield provider
    provider.close()


@pytest.fixture
def redis_provider():
    """Real Redis provider, skip if unavailable."""
    from redis import Redis

    try:
        conn = Redis()
        conn.ping()
    except Exception:
        pytest.skip("Redis server not available")

    from atdata.providers._redis import RedisProvider

    provider = RedisProvider(conn)

    # Clean up benchmark keys before/after
    def _cleanup():
        for pattern in ("LocalDatasetEntry:bench_*", "LocalSchema:Bench*"):
            for key in conn.scan_iter(match=pattern):
                conn.delete(key)

    _cleanup()
    yield provider
    _cleanup()
    provider.close()


@pytest.fixture
def postgres_provider():
    """Real PostgreSQL provider, skip if unavailable."""
    try:
        import psycopg  # noqa: F401
    except ImportError:
        pytest.skip("psycopg not installed")

    import os

    dsn = os.environ.get("ATDATA_BENCH_POSTGRES_DSN")
    if not dsn:
        pytest.skip("ATDATA_BENCH_POSTGRES_DSN not set")

    from atdata.providers._postgres import PostgresProvider

    provider = PostgresProvider(dsn=dsn)
    yield provider
    provider.close()


@pytest.fixture(
    params=["sqlite", "redis", "postgres"],
    ids=["sqlite", "redis", "postgres"],
)
def any_provider(request, tmp_path):
    """Parametrized fixture that yields each available provider."""
    backend = request.param

    if backend == "sqlite":
        from atdata.providers._sqlite import SqliteProvider

        provider = SqliteProvider(path=tmp_path / "bench.db")
        yield provider
        provider.close()

    elif backend == "redis":
        from redis import Redis

        try:
            conn = Redis()
            conn.ping()
        except Exception:
            pytest.skip("Redis server not available")

        from atdata.providers._redis import RedisProvider

        provider = RedisProvider(conn)

        def _cleanup():
            for pattern in ("LocalDatasetEntry:bench_*", "LocalSchema:Bench*"):
                for key in conn.scan_iter(match=pattern):
                    conn.delete(key)

        _cleanup()
        yield provider
        _cleanup()
        provider.close()

    elif backend == "postgres":
        try:
            import psycopg  # noqa: F401
        except ImportError:
            pytest.skip("psycopg not installed")

        import os

        dsn = os.environ.get("ATDATA_BENCH_POSTGRES_DSN")
        if not dsn:
            pytest.skip("ATDATA_BENCH_POSTGRES_DSN not set")

        from atdata.providers._postgres import PostgresProvider

        provider = PostgresProvider(dsn=dsn)
        yield provider
        provider.close()


# =============================================================================
# Dataset Fixtures
# =============================================================================


@pytest.fixture
def small_basic_dataset(tmp_path):
    """100-sample basic dataset (1 shard)."""
    samples = generate_basic_samples(100)
    tar_path = write_tar(tmp_path / "small-000000.tar", samples)
    return atdata.Dataset[BenchBasicSample](url=str(tar_path)), samples


@pytest.fixture
def medium_basic_dataset(tmp_path):
    """1000-sample basic dataset (1 shard)."""
    samples = generate_basic_samples(1000)
    tar_path = write_tar(tmp_path / "medium-000000.tar", samples)
    return atdata.Dataset[BenchBasicSample](url=str(tar_path)), samples


@pytest.fixture
def small_numpy_dataset(tmp_path):
    """100-sample numpy dataset (1 shard, 10x10 arrays)."""
    samples = generate_numpy_samples(100)
    tar_path = write_tar(tmp_path / "numpy-000000.tar", samples)
    return atdata.Dataset[BenchNumpySample](url=str(tar_path)), samples


@pytest.fixture
def manifest_dataset_small(tmp_path):
    """100-sample manifest dataset with 2 shards."""
    samples = generate_manifest_samples(100)
    create_sharded_dataset(
        tmp_path, samples, 50, BenchManifestSample, with_manifests=True
    )
    return tmp_path, samples
