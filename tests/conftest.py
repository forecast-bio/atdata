"""Pytest configuration for atdata tests."""

import pytest
from redis import Redis
from typing import Optional

import numpy as np
from numpy.typing import NDArray

import atdata


# =============================================================================
# Shared sample types for testing
# =============================================================================

@atdata.packable
class SharedBasicSample:
    """Basic sample with primitive fields for general testing."""
    name: str
    value: int


@atdata.packable
class SharedNumpySample:
    """Sample with NDArray field for array serialization testing."""
    data: NDArray
    label: str


@atdata.packable
class SharedOptionalSample:
    """Sample with optional fields for null handling testing."""
    required: str
    optional_int: Optional[int] = None
    optional_array: Optional[NDArray] = None


@atdata.packable
class SharedAllTypesSample:
    """Sample with all supported primitive types."""
    str_field: str
    int_field: int
    float_field: float
    bool_field: bool
    bytes_field: bytes


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
        for pattern in ('LocalDatasetEntry:*', 'BasicIndexEntry:*', 'LocalSchema:*'):
            for key in redis_connection.scan_iter(match=pattern):
                redis_connection.delete(key)

    _clear_all()
    yield redis_connection
    _clear_all()
