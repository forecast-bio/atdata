"""Pytest configuration for atdata tests."""

import pytest
from redis import Redis


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
