"""Pytest configuration for atdata tests.

This module provides shared fixtures and sample types for the test suite.
"""

from pathlib import Path
from typing import Optional, TypeVar

import numpy as np
import pytest
import webdataset as wds
from numpy.typing import NDArray
from redis import Redis

import atdata


# =============================================================================
# Shared Sample Types
# =============================================================================
#
# These shared sample types reduce duplication across test files. Use them
# when your test doesn't require specific field names or structures.
#
# When to use shared types:
#   - General serialization/deserialization tests
#   - Dataset creation and iteration tests
#   - Batch aggregation tests
#   - Integration tests that don't depend on specific field names
#
# When to define local types:
#   - Tests that verify specific field name serialization
#   - Tests that need particular field orderings
#   - Tests for edge cases with unusual field combinations
#   - Tests where the type name is significant (e.g., schema name tests)
#
# =============================================================================


@atdata.packable
class SharedBasicSample:
    """Basic sample with primitive fields for general testing.

    Fields: name (str), value (int)
    """

    name: str
    value: int


@atdata.packable
class SharedNumpySample:
    """Sample with NDArray field for array serialization testing.

    Fields: data (NDArray), label (str)
    """

    data: NDArray
    label: str


@atdata.packable
class SharedOptionalSample:
    """Sample with optional fields for null handling testing.

    Fields: required (str), optional_int (int|None), optional_array (NDArray|None)
    """

    required: str
    optional_int: Optional[int] = None
    optional_array: Optional[NDArray] = None


@atdata.packable
class SharedAllTypesSample:
    """Sample with all supported primitive types.

    Fields: str_field, int_field, float_field, bool_field, bytes_field
    """

    str_field: str
    int_field: int
    float_field: float
    bool_field: bool
    bytes_field: bytes


@atdata.packable
class SharedListSample:
    """Sample with list fields for array-of-primitives testing.

    Fields: tags (list[str]), scores (list[float])
    """

    tags: list[str]
    scores: list[float]


@atdata.packable
class SharedMetadataSample:
    """Sample for testing metadata handling.

    Fields: id (int), content (str), score (float)
    """

    id: int
    content: str
    score: float


# =============================================================================
# Tar Creation Helpers
# =============================================================================
#
# These helpers centralize common WebDataset tar file creation patterns.
# Import and use these instead of duplicating TarWriter boilerplate.
#
# =============================================================================

ST = TypeVar("ST")


def create_tar_with_samples(tar_path: Path, samples: list) -> None:
    """Create a WebDataset tar file from a list of PackableSample instances.

    Args:
        tar_path: Path where the tar file will be created.
        samples: List of PackableSample instances to write.

    Example:
        samples = [SharedBasicSample(name=f"s{i}", value=i) for i in range(10)]
        create_tar_with_samples(tmp_path / "data.tar", samples)
    """
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with wds.writer.TarWriter(str(tar_path)) as writer:
        for sample in samples:
            writer.write(sample.as_wds)


def create_basic_dataset(
    tmp_path: Path,
    num_samples: int = 10,
    name: str = "test",
) -> "atdata.Dataset[SharedBasicSample]":
    """Create a dataset with SharedBasicSample instances.

    Args:
        tmp_path: Temporary directory for the tar file.
        num_samples: Number of samples to create.
        name: Prefix for the tar filename.

    Returns:
        Dataset configured to read the created tar file.
    """
    tar_path = tmp_path / f"{name}-000000.tar"
    samples = [
        SharedBasicSample(name=f"sample_{i}", value=i * 10) for i in range(num_samples)
    ]
    create_tar_with_samples(tar_path, samples)
    return atdata.Dataset[SharedBasicSample](url=str(tar_path))


def create_numpy_dataset(
    tmp_path: Path,
    num_samples: int = 5,
    array_shape: tuple = (10, 10),
    name: str = "array",
) -> "atdata.Dataset[SharedNumpySample]":
    """Create a dataset with SharedNumpySample instances containing random arrays.

    Args:
        tmp_path: Temporary directory for the tar file.
        num_samples: Number of samples to create.
        array_shape: Shape of the random numpy arrays.
        name: Prefix for the tar filename.

    Returns:
        Dataset configured to read the created tar file.
    """
    tar_path = tmp_path / f"{name}-000000.tar"
    samples = [
        SharedNumpySample(
            data=np.random.randn(*array_shape).astype(np.float32),
            label=f"array_{i}",
        )
        for i in range(num_samples)
    ]
    create_tar_with_samples(tar_path, samples)
    return atdata.Dataset[SharedNumpySample](url=str(tar_path))


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def redis_connection():
    """Provide a Redis connection, skip test if Redis is not available."""
    try:
        redis = Redis()
        redis.ping()
        yield redis
    except Exception:
        pytest.skip("Redis server not available")


@pytest.fixture
def clean_redis(redis_connection):
    """Provide a Redis connection with automatic cleanup of test keys.

    Clears LocalDatasetEntry, BasicIndexEntry (legacy), and LocalSchema keys
    before and after each test to ensure test isolation.
    """

    def _clear_all():
        for pattern in ("LocalDatasetEntry:*", "BasicIndexEntry:*", "LocalSchema:*"):
            for key in redis_connection.scan_iter(match=pattern):
                redis_connection.delete(key)

    _clear_all()
    yield redis_connection
    _clear_all()
