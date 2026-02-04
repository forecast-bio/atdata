"""Shared fixtures for live integration tests.

Every test in this package is automatically marked with
``@pytest.mark.integration`` via the ``pytestmark`` below.

Fixtures read credentials from environment variables and skip
tests when the required services are unavailable.  A per-run
UUID prefix isolates resources created during the test run.
"""

from __future__ import annotations

import os
import uuid
from typing import Generator

import pytest

# ── Auto-mark every test in this package ────────────────────────────
pytestmark = pytest.mark.integration

# ── Per-run isolation prefix ────────────────────────────────────────
RUN_ID: str = uuid.uuid4().hex[:12]
"""Short unique prefix for all resources created during this test run."""


# ── ATProto fixtures ───────────────────────────────────────────────


@pytest.fixture(scope="session")
def atproto_credentials() -> tuple[str, str]:
    """Return (handle, app_password) from env or skip."""
    handle = os.environ.get("ATPROTO_TEST_HANDLE", "")
    password = os.environ.get("ATPROTO_TEST_PASSWORD", "")
    if not handle or not password:
        pytest.skip(
            "ATProto credentials not configured "
            "(set ATPROTO_TEST_HANDLE and ATPROTO_TEST_PASSWORD)"
        )
    return handle, password


@pytest.fixture(scope="session")
def atproto_client(atproto_credentials: tuple[str, str]):
    """Session-scoped authenticated Atmosphere client."""
    from atdata.atmosphere import Atmosphere

    handle, password = atproto_credentials
    client = Atmosphere.login(handle, password)
    yield client


# ── S3 / MinIO fixtures ───────────────────────────────────────────


@pytest.fixture(scope="session")
def minio_credentials() -> dict[str, str]:
    """Return MinIO credentials from env or skip."""
    endpoint = os.environ.get("MINIO_ENDPOINT", "")
    access_key = os.environ.get("MINIO_ACCESS_KEY", "")
    secret_key = os.environ.get("MINIO_SECRET_KEY", "")
    if not endpoint or not access_key or not secret_key:
        pytest.skip(
            "MinIO credentials not configured "
            "(set MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY)"
        )
    return {
        "AWS_ENDPOINT": endpoint,
        "AWS_ACCESS_KEY_ID": access_key,
        "AWS_SECRET_ACCESS_KEY": secret_key,
    }


@pytest.fixture(scope="session")
def minio_bucket(minio_credentials: dict[str, str]) -> str:
    """Ensure the integration-test bucket exists and return its name."""
    import boto3

    bucket_name = f"atdata-integration-{RUN_ID}"
    s3 = boto3.client(
        "s3",
        endpoint_url=minio_credentials["AWS_ENDPOINT"],
        aws_access_key_id=minio_credentials["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=minio_credentials["AWS_SECRET_ACCESS_KEY"],
    )
    s3.create_bucket(Bucket=bucket_name)
    yield bucket_name

    # Teardown: delete all objects then the bucket
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket_name):
        for obj in page.get("Contents", []):
            s3.delete_object(Bucket=bucket_name, Key=obj["Key"])
    s3.delete_bucket(Bucket=bucket_name)


# ── PostgreSQL fixtures ───────────────────────────────────────────


@pytest.fixture(scope="session")
def postgres_dsn() -> str:
    """Return a PostgreSQL DSN from env or skip."""
    dsn = os.environ.get("POSTGRES_DSN", "")
    if not dsn:
        pytest.skip("PostgreSQL DSN not configured (set POSTGRES_DSN)")
    return dsn


@pytest.fixture()
def postgres_provider(
    postgres_dsn: str,
) -> Generator:
    """Create a PostgresProvider and clean up tables after test."""
    from atdata.providers._postgres import PostgresProvider

    provider = PostgresProvider(dsn=postgres_dsn)
    yield provider

    # Teardown: truncate all tables so each test starts clean
    with provider._conn.cursor() as cur:
        cur.execute(
            "TRUNCATE dataset_entries, schemas, labels RESTART IDENTITY CASCADE"
        )
    provider._conn.commit()
    provider.close()


# ── Redis fixtures ───────────────────────────────────────────────


@pytest.fixture(scope="session")
def redis_url() -> str:
    """Return a Redis URL from env or skip."""
    url = os.environ.get("REDIS_URL", "")
    if not url:
        pytest.skip("Redis URL not configured (set REDIS_URL)")
    return url


@pytest.fixture()
def redis_provider(redis_url: str) -> Generator:
    """Create a RedisProvider and flush test keys after each test."""
    from redis import Redis

    from atdata.providers._redis import RedisProvider

    conn = Redis.from_url(redis_url)
    provider = RedisProvider(redis=conn)
    yield provider

    # Teardown: flush the entire test database (assumes dedicated db)
    conn.flushdb()
    provider.close()


# ── Helpers ────────────────────────────────────────────────────────


def unique_name(prefix: str = "integ") -> str:
    """Generate a unique name for a test resource."""
    return f"{prefix}-{RUN_ID}-{uuid.uuid4().hex[:8]}"
