"""Tests for index storage providers (Redis, SQLite, PostgreSQL).

These tests validate that all IndexProvider implementations behave
identically by parametrizing across backends.  SQLite tests always run;
Redis and PostgreSQL require external services and are skipped when
unavailable.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import pytest
import webdataset as wds
from numpy.typing import NDArray  # used by ProviderArraySample

import atdata
import atdata.local as atlocal
from atdata.providers._base import IndexProvider
from atdata.providers._sqlite import SqliteProvider

# ---------------------------------------------------------------------------
# Sample types
# ---------------------------------------------------------------------------


@dataclass
class ProviderTestSample(atdata.PackableSample):
    """Sample type used across provider tests."""

    name: str
    value: int


@dataclass
class ProviderArraySample(atdata.PackableSample):
    """Sample with NDArray for schema field-type coverage."""

    label: str
    data: NDArray


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sqlite_provider(tmp_path: Path) -> SqliteProvider:
    """Provide a fresh SQLite provider."""
    return SqliteProvider(path=tmp_path / "test-index.db")


@pytest.fixture
def redis_provider():
    """Provide a fresh Redis provider, skip if Redis is unavailable."""
    try:
        from redis import Redis

        redis = Redis()
        redis.ping()
    except Exception:
        pytest.skip("Redis server not available")

    from atdata.providers._redis import RedisProvider

    provider = RedisProvider(redis)

    # Clean test keys before and after
    def _clear():
        for pattern in ("LocalDatasetEntry:*", "LocalSchema:*"):
            for key in redis.scan_iter(match=pattern):
                redis.delete(key)

    _clear()
    yield provider
    _clear()


@pytest.fixture
def postgres_provider():
    """Provide a fresh PostgreSQL provider, skip if unavailable."""
    dsn = os.environ.get("ATDATA_TEST_PG_DSN")
    if not dsn:
        pytest.skip("PostgreSQL not configured (set ATDATA_TEST_PG_DSN)")

    try:
        import psycopg  # noqa: F401
    except ImportError:
        pytest.skip("psycopg not installed")

    from atdata.providers._postgres import PostgresProvider

    provider = PostgresProvider(dsn=dsn)

    # Clean tables before and after
    def _clear():
        with provider._conn.cursor() as cur:
            cur.execute("DELETE FROM dataset_entries")
            cur.execute("DELETE FROM schemas")
        provider._conn.commit()

    _clear()
    yield provider
    _clear()
    provider.close()


@pytest.fixture(params=["sqlite", "redis", "postgres"])
def provider(request, tmp_path: Path) -> IndexProvider:
    """Parametrized fixture yielding each available provider."""
    if request.param == "sqlite":
        return SqliteProvider(path=tmp_path / "test-index.db")

    if request.param == "redis":
        try:
            from redis import Redis

            redis = Redis()
            redis.ping()
        except Exception:
            pytest.skip("Redis server not available")

        from atdata.providers._redis import RedisProvider

        prov = RedisProvider(redis)

        def _clear():
            for pattern in ("LocalDatasetEntry:*", "LocalSchema:*"):
                for key in redis.scan_iter(match=pattern):
                    redis.delete(key)

        _clear()
        request.addfinalizer(_clear)
        return prov

    if request.param == "postgres":
        dsn = os.environ.get("ATDATA_TEST_PG_DSN")
        if not dsn:
            pytest.skip("PostgreSQL not configured (set ATDATA_TEST_PG_DSN)")

        try:
            import psycopg  # noqa: F401
        except ImportError:
            pytest.skip("psycopg not installed")

        from atdata.providers._postgres import PostgresProvider

        prov = PostgresProvider(dsn=dsn)

        def _clear_pg():
            with prov._conn.cursor() as cur:
                cur.execute("DELETE FROM dataset_entries")
                cur.execute("DELETE FROM schemas")
            prov._conn.commit()

        _clear_pg()
        request.addfinalizer(_clear_pg)
        request.addfinalizer(prov.close)
        return prov

    pytest.skip(f"Unknown provider: {request.param}")


@pytest.fixture
def sample_dataset(tmp_path: Path):
    """Create a sample WebDataset for testing."""
    dataset_path = tmp_path / "test-dataset-000000.tar"
    with wds.writer.TarWriter(str(dataset_path)) as sink:
        for i in range(5):
            sample = ProviderTestSample(name=f"s_{i}", value=i * 10)
            sink.write(sample.as_wds)
    return atdata.Dataset[ProviderTestSample](url=str(dataset_path))


# ---------------------------------------------------------------------------
# Provider-level tests (parametrized across backends)
# ---------------------------------------------------------------------------


class TestProviderEntries:
    """Dataset entry CRUD operations."""

    def test_store_and_get_by_cid(self, provider: IndexProvider):
        entry = atlocal.LocalDatasetEntry(
            name="ds-1",
            schema_ref="atdata://local/sampleSchema/Test@1.0.0",
            data_urls=["s3://bucket/data-000000.tar"],
        )
        provider.store_entry(entry)
        loaded = provider.get_entry_by_cid(entry.cid)

        assert loaded.name == "ds-1"
        assert loaded.schema_ref == "atdata://local/sampleSchema/Test@1.0.0"
        assert loaded.data_urls == ["s3://bucket/data-000000.tar"]
        assert loaded.cid == entry.cid

    def test_store_and_get_by_name(self, provider: IndexProvider):
        entry = atlocal.LocalDatasetEntry(
            name="my-dataset",
            schema_ref="atdata://local/sampleSchema/X@1.0.0",
            data_urls=["file:///data.tar"],
        )
        provider.store_entry(entry)
        loaded = provider.get_entry_by_name("my-dataset")

        assert loaded.name == "my-dataset"
        assert loaded.cid == entry.cid

    def test_get_by_cid_missing_raises(self, provider: IndexProvider):
        with pytest.raises(KeyError):
            provider.get_entry_by_cid("nonexistent-cid")

    def test_get_by_name_missing_raises(self, provider: IndexProvider):
        with pytest.raises(KeyError):
            provider.get_entry_by_name("no-such-dataset")

    def test_iter_entries(self, provider: IndexProvider):
        for i in range(3):
            entry = atlocal.LocalDatasetEntry(
                name=f"ds-{i}",
                schema_ref=f"atdata://local/sampleSchema/T@1.0.{i}",
                data_urls=[f"s3://bucket/data-{i}.tar"],
            )
            provider.store_entry(entry)

        entries = list(provider.iter_entries())
        assert len(entries) == 3
        names = {e.name for e in entries}
        assert names == {"ds-0", "ds-1", "ds-2"}

    def test_entry_with_metadata(self, provider: IndexProvider):
        meta = {"author": "test", "version": 2}
        entry = atlocal.LocalDatasetEntry(
            name="meta-ds",
            schema_ref="atdata://local/sampleSchema/M@1.0.0",
            data_urls=["s3://bucket/meta.tar"],
            metadata=meta,
        )
        provider.store_entry(entry)
        loaded = provider.get_entry_by_cid(entry.cid)

        assert loaded.metadata == meta

    def test_entry_with_none_metadata(self, provider: IndexProvider):
        entry = atlocal.LocalDatasetEntry(
            name="no-meta",
            schema_ref="atdata://local/sampleSchema/N@1.0.0",
            data_urls=["file:///data.tar"],
            metadata=None,
        )
        provider.store_entry(entry)
        loaded = provider.get_entry_by_cid(entry.cid)

        assert loaded.metadata is None

    def test_entry_with_multiple_urls(self, provider: IndexProvider):
        urls = [f"s3://bucket/shard-{i:06d}.tar" for i in range(5)]
        entry = atlocal.LocalDatasetEntry(
            name="multi-shard",
            schema_ref="atdata://local/sampleSchema/S@1.0.0",
            data_urls=urls,
        )
        provider.store_entry(entry)
        loaded = provider.get_entry_by_cid(entry.cid)

        assert loaded.data_urls == urls

    def test_upsert_overwrites(self, provider: IndexProvider):
        entry1 = atlocal.LocalDatasetEntry(
            name="ds-upsert",
            schema_ref="atdata://local/sampleSchema/U@1.0.0",
            data_urls=["s3://bucket/v1.tar"],
        )
        provider.store_entry(entry1)

        # Same CID, updated name shouldn't change CID but let's test upsert
        # with a different entry sharing the same CID
        entry2 = atlocal.LocalDatasetEntry(
            name="ds-upsert",
            schema_ref="atdata://local/sampleSchema/U@1.0.0",
            data_urls=["s3://bucket/v1.tar"],
            metadata={"updated": True},
        )
        provider.store_entry(entry2)

        loaded = provider.get_entry_by_cid(entry1.cid)
        assert loaded.metadata == {"updated": True}


class TestProviderSchemas:
    """Schema CRUD operations."""

    def test_store_and_get_schema(self, provider: IndexProvider):
        schema = json.dumps({"name": "TestSample", "version": "1.0.0", "fields": []})
        provider.store_schema("TestSample", "1.0.0", schema)
        loaded = provider.get_schema_json("TestSample", "1.0.0")

        assert loaded is not None
        assert json.loads(loaded)["name"] == "TestSample"

    def test_get_schema_missing_returns_none(self, provider: IndexProvider):
        assert provider.get_schema_json("NoSuch", "1.0.0") is None

    def test_iter_schemas(self, provider: IndexProvider):
        for i in range(3):
            schema = json.dumps({"name": "S", "version": f"1.0.{i}", "fields": []})
            provider.store_schema("S", f"1.0.{i}", schema)

        results = list(provider.iter_schemas())
        assert len(results) == 3
        versions = {r[1] for r in results}
        assert versions == {"1.0.0", "1.0.1", "1.0.2"}

    def test_find_latest_version(self, provider: IndexProvider):
        for v in ["1.0.0", "1.2.0", "1.1.5", "2.0.0", "1.9.9"]:
            provider.store_schema("MyType", v, json.dumps({"name": "MyType"}))

        assert provider.find_latest_version("MyType") == "2.0.0"

    def test_find_latest_version_missing(self, provider: IndexProvider):
        assert provider.find_latest_version("NoSuchSchema") is None

    def test_schema_upsert(self, provider: IndexProvider):
        provider.store_schema("Up", "1.0.0", '{"v": 1}')
        provider.store_schema("Up", "1.0.0", '{"v": 2}')

        loaded = provider.get_schema_json("Up", "1.0.0")
        assert loaded is not None
        assert json.loads(loaded)["v"] == 2

    def test_multiple_schema_names(self, provider: IndexProvider):
        provider.store_schema("Alpha", "1.0.0", '{"name": "Alpha"}')
        provider.store_schema("Beta", "1.0.0", '{"name": "Beta"}')

        assert provider.find_latest_version("Alpha") == "1.0.0"
        assert provider.find_latest_version("Beta") == "1.0.0"
        assert provider.find_latest_version("Gamma") is None


# ---------------------------------------------------------------------------
# Index integration tests with providers
# ---------------------------------------------------------------------------


class TestIndexWithProvider:
    """Test the Index class using different providers."""

    def test_index_with_sqlite(self, tmp_path: Path, sample_dataset):
        provider = SqliteProvider(path=tmp_path / "idx.db")
        index = atlocal.Index(provider=provider)

        ref = index.publish_schema(ProviderTestSample, version="1.0.0")
        assert "ProviderTestSample" in ref
        assert "1.0.0" in ref

        entry = index.insert_dataset(sample_dataset, name="test-ds", schema_ref=ref)
        assert entry.name == "test-ds"

        loaded = index.get_dataset("test-ds")
        assert loaded.cid == entry.cid

    def test_index_list_datasets_sqlite(self, tmp_path: Path):
        provider = SqliteProvider(path=tmp_path / "idx.db")
        index = atlocal.Index(provider=provider)

        # Create two distinct datasets (different URLs → different CIDs)
        for label in ("a", "b"):
            tar_path = tmp_path / f"ds-{label}-000000.tar"
            with wds.writer.TarWriter(str(tar_path)) as sink:
                sink.write(ProviderTestSample(name=label, value=0).as_wds)
            ds = atdata.Dataset[ProviderTestSample](url=str(tar_path))
            index.insert_dataset(ds, name=f"ds-{label}")

        datasets = index.list_datasets()
        names = {d.name for d in datasets}
        assert names == {"ds-a", "ds-b"}

    def test_index_schema_auto_version(self, tmp_path: Path):
        provider = SqliteProvider(path=tmp_path / "idx.db")
        index = atlocal.Index(provider=provider)

        ref1 = index.publish_schema(ProviderTestSample, version="1.0.0")
        ref2 = index.publish_schema(ProviderTestSample)  # auto-increment

        assert "1.0.0" in ref1
        assert "1.0.1" in ref2

    def test_index_list_schemas_sqlite(self, tmp_path: Path):
        provider = SqliteProvider(path=tmp_path / "idx.db")
        index = atlocal.Index(provider=provider)

        index.publish_schema(ProviderTestSample, version="1.0.0")
        index.publish_schema(ProviderArraySample, version="1.0.0")

        schemas = index.list_schemas()
        assert len(schemas) == 2

    def test_index_decode_schema_sqlite(self, tmp_path: Path):
        provider = SqliteProvider(path=tmp_path / "idx.db")
        index = atlocal.Index(provider=provider)

        ref = index.publish_schema(ProviderTestSample, version="1.0.0")
        decoded = index.decode_schema(ref)

        assert decoded.__name__ == "ProviderTestSample"

    def test_index_get_dataset_not_found(self, tmp_path: Path):
        provider = SqliteProvider(path=tmp_path / "idx.db")
        index = atlocal.Index(provider=provider)

        with pytest.raises(KeyError):
            index.get_dataset("nonexistent")

    def test_index_schemas_iteration_sqlite(self, tmp_path: Path):
        provider = SqliteProvider(path=tmp_path / "idx.db")
        index = atlocal.Index(provider=provider)

        index.publish_schema(ProviderTestSample, version="1.0.0")
        records = list(index.schemas)
        assert len(records) == 1
        assert records[0].name == "ProviderTestSample"


class TestStringProviderSelection:
    """Test Index with string-based provider selection."""

    def test_sqlite_by_name(self, tmp_path: Path):
        index = atlocal.Index(provider="sqlite", path=tmp_path / "factory.db")
        assert isinstance(index, atlocal.Index)

    def test_invalid_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            atlocal.Index(provider="mongodb")

    def test_postgres_requires_dsn(self):
        with pytest.raises(ValueError, match="dsn is required"):
            atlocal.Index(provider="postgres")


# ---------------------------------------------------------------------------
# SQLite-specific tests
# ---------------------------------------------------------------------------


class TestSqliteProvider:
    """SQLite-specific behaviour."""

    def test_default_path(self):
        """Default path should be ~/.atdata/index.db."""
        provider = SqliteProvider()
        expected = Path.home() / ".atdata" / "index.db"
        assert provider.path == expected
        provider.close()

    def test_wal_mode(self, tmp_path: Path):
        """SQLite should use WAL journal mode for concurrency."""
        provider = SqliteProvider(path=tmp_path / "wal-test.db")
        cursor = provider._conn.execute("PRAGMA journal_mode")
        mode = cursor.fetchone()[0]
        assert mode == "wal"
        provider.close()

    def test_close_and_reopen(self, tmp_path: Path):
        """Data should persist across close/reopen."""
        db_path = tmp_path / "persist.db"

        provider = SqliteProvider(path=db_path)
        provider.store_schema("Persist", "1.0.0", '{"name": "Persist"}')
        provider.close()

        provider2 = SqliteProvider(path=db_path)
        loaded = provider2.get_schema_json("Persist", "1.0.0")
        assert loaded is not None
        assert json.loads(loaded)["name"] == "Persist"
        provider2.close()
