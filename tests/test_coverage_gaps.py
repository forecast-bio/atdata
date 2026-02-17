"""Tests targeting specific coverage gaps across multiple modules.

Covers missed lines in:
- providers/_factory.py (postgres path, unknown provider)
- dataset.py (DictSample edges, Dataset init errors, chained filter/map,
  wrap_batch error, write_samples maxsize, to_dict DictSample)
- local/_index.py (bad provider type, _redis attr, publish_schema type check,
  get_schema_record)
- local/_disk.py (RuntimeError on no shards written)
"""

from unittest.mock import MagicMock, patch

import pytest

import atdata
import atdata.local as atlocal
from atdata.dataset import DictSample
from atdata.providers._sqlite import SqliteProvider
from conftest import SharedBasicSample, create_tar_with_samples


# ---------------------------------------------------------------------------
# providers/_factory.py
# ---------------------------------------------------------------------------


class TestProviderFactory:
    def test_create_sqlite_provider(self, tmp_path):
        from atdata.providers._factory import create_provider

        p = create_provider("sqlite", path=tmp_path / "test.db")
        assert p is not None

    def test_create_postgres_requires_dsn(self):
        from atdata.providers._factory import create_provider

        with pytest.raises(ValueError, match="dsn is required"):
            create_provider("postgres")

    def test_create_postgres_with_dsn(self):
        from atdata.providers._factory import create_provider

        with patch("atdata.providers._postgres.PostgresProvider") as mock_pg:
            mock_pg.return_value = MagicMock()
            p = create_provider("postgres", dsn="postgresql://localhost/test")
            mock_pg.assert_called_once_with(dsn="postgresql://localhost/test")
            assert p is not None

    def test_create_postgresql_alias(self):
        from atdata.providers._factory import create_provider

        with patch("atdata.providers._postgres.PostgresProvider") as mock_pg:
            mock_pg.return_value = MagicMock()
            create_provider("postgresql", dsn="postgresql://localhost/db")
            mock_pg.assert_called_once_with(dsn="postgresql://localhost/db")

    def test_unknown_provider_raises(self):
        from atdata.providers._factory import create_provider

        with pytest.raises(ValueError, match="Unknown provider"):
            create_provider("mongodb")

    def test_redis_with_existing_connection(self):
        from atdata.providers._factory import create_provider

        mock_redis = MagicMock()
        p = create_provider("redis", redis=mock_redis)
        assert p is not None

    def test_redis_creates_new_connection(self):
        from atdata.providers._factory import create_provider

        with patch("redis.Redis") as mock_cls:
            mock_cls.return_value = MagicMock()
            p = create_provider("redis", host="localhost", port=6379)
            mock_cls.assert_called_once_with(host="localhost", port=6379)
            assert p is not None


# ---------------------------------------------------------------------------
# DictSample edge cases
# ---------------------------------------------------------------------------


class TestDictSampleEdges:
    def test_getattr_missing_field(self):
        ds = DictSample(_data={"x": 1})
        with pytest.raises(AttributeError, match="has no field 'missing'"):
            _ = ds.missing

    def test_getattr_data_recursion_guard(self):
        """Accessing _data before it's set raises AttributeError."""
        ds = object.__new__(DictSample)
        with pytest.raises(AttributeError, match="_data"):
            _ = ds._data


# ---------------------------------------------------------------------------
# Dataset init and edge cases
# ---------------------------------------------------------------------------


class TestDatasetEdgeCases:
    def test_no_source_or_url_raises(self):
        with pytest.raises(TypeError, match="missing required argument"):
            atdata.Dataset[SharedBasicSample]()

    def test_shards_property(self, tmp_path):
        tar_path = tmp_path / "test-000000.tar"
        create_tar_with_samples(tar_path, [SharedBasicSample(name="a", value=1)])
        ds = atdata.Dataset[SharedBasicSample](url=str(tar_path))
        shard_ids = list(ds.shards)
        assert len(shard_ids) >= 1

    def test_schema_returns_empty_for_non_dataclass(self, tmp_path):
        tar_path = tmp_path / "test-000000.tar"
        create_tar_with_samples(tar_path, [SharedBasicSample(name="a", value=1)])
        ds = atdata.Dataset[SharedBasicSample](url=str(tar_path))
        # Force sample_type to a non-dataclass
        ds._sample_type_cache = int
        assert ds.schema == {}

    def test_chained_filter(self, tmp_path):
        """filter() on an already-filtered dataset chains predicates."""
        tar_path = tmp_path / "data-000000.tar"
        samples = [SharedBasicSample(name=f"s{i}", value=i) for i in range(10)]
        create_tar_with_samples(tar_path, samples)
        ds = atdata.Dataset[SharedBasicSample](url=str(tar_path))
        # First filter: value > 3, second filter: value < 8
        filtered = ds.filter(lambda s: s.value > 3).filter(lambda s: s.value < 8)
        result = list(filtered.ordered(batch_size=None))
        values = sorted([s.value for s in result])
        assert values == [4, 5, 6, 7]

    def test_chained_map(self, tmp_path):
        """map() on an already-mapped dataset chains transforms."""
        tar_path = tmp_path / "data-000000.tar"
        samples = [SharedBasicSample(name=f"s{i}", value=i) for i in range(3)]
        create_tar_with_samples(tar_path, samples)
        ds = atdata.Dataset[SharedBasicSample](url=str(tar_path))
        # Chain two maps: first doubles value, second adds 1
        mapped = ds.map(
            lambda s: SharedBasicSample(name=s.name, value=s.value * 2)
        ).map(lambda s: SharedBasicSample(name=s.name, value=s.value + 1))
        result = sorted(list(mapped.ordered(batch_size=None)), key=lambda s: s.value)
        assert [s.value for s in result] == [1, 3, 5]

    def test_filter_preserves_map(self, tmp_path):
        """filter() on a mapped dataset preserves the map function."""
        tar_path = tmp_path / "data-000000.tar"
        samples = [SharedBasicSample(name=f"s{i}", value=i) for i in range(5)]
        create_tar_with_samples(tar_path, samples)
        ds = atdata.Dataset[SharedBasicSample](url=str(tar_path))
        result = ds.map(
            lambda s: SharedBasicSample(name=s.name, value=s.value * 10)
        ).filter(lambda s: s.value >= 20)
        items = list(result.ordered(batch_size=None))
        assert all(s.value >= 20 for s in items)

    def test_map_preserves_filter(self, tmp_path):
        """map() on a filtered dataset preserves the filter function."""
        tar_path = tmp_path / "data-000000.tar"
        samples = [SharedBasicSample(name=f"s{i}", value=i) for i in range(5)]
        create_tar_with_samples(tar_path, samples)
        ds = atdata.Dataset[SharedBasicSample](url=str(tar_path))
        result = ds.filter(lambda s: s.value > 2).map(
            lambda s: SharedBasicSample(name=s.name, value=s.value * 10)
        )
        items = list(result.ordered(batch_size=None))
        assert all(s.value >= 30 for s in items)

    def test_wrap_batch_missing_msgpack(self, tmp_path):
        tar_path = tmp_path / "data-000000.tar"
        create_tar_with_samples(tar_path, [SharedBasicSample(name="a", value=1)])
        ds = atdata.Dataset[SharedBasicSample](url=str(tar_path))
        with pytest.raises(ValueError, match="missing 'msgpack' key"):
            ds.wrap_batch({"__key__": ["k1"]})

    def test_to_dict_with_dict_sample(self, tmp_path):
        """to_dict works with DictSample datasets."""
        tar_path = tmp_path / "data-000000.tar"
        samples = [SharedBasicSample(name=f"s{i}", value=i) for i in range(3)]
        create_tar_with_samples(tar_path, samples)
        ds = atdata.Dataset[DictSample](url=str(tar_path))
        d = ds.to_dict(limit=3)
        assert isinstance(d, dict)
        assert len(d) == 2  # name + value columns


class TestWriteSamplesMaxsize:
    def test_maxsize_creates_shards(self, tmp_path):
        """write_samples with maxsize triggers ShardWriter path."""
        samples = [SharedBasicSample(name=f"s{i}", value=i) for i in range(10)]
        ds = atdata.write_samples(samples, tmp_path / "data.tar", maxsize=100)
        result = list(ds.ordered())
        assert len(result) == 10


# ---------------------------------------------------------------------------
# local/_index.py edge cases
# ---------------------------------------------------------------------------


class TestIndexEdgeCases:
    def test_bad_provider_type_raises(self):
        with pytest.raises(TypeError, match="provider must be"):
            atlocal.Index(provider=42, atmosphere=None)

    def test_redis_property_on_sqlite_raises(self, tmp_path):
        provider = SqliteProvider(path=tmp_path / "test.db")
        index = atlocal.Index(provider=provider, atmosphere=None)
        with pytest.raises(AttributeError, match="only available with a Redis"):
            _ = index._redis

    def test_publish_schema_non_packable_raises(self, tmp_path):
        provider = SqliteProvider(path=tmp_path / "test.db")
        index = atlocal.Index(provider=provider, atmosphere=None)

        class NotPackable:
            x: int = 0

        with pytest.raises(TypeError, match="does not satisfy the Packable protocol"):
            index.publish_schema(NotPackable, version="1.0.0")

    def test_publish_schema_not_a_class_raises(self, tmp_path):
        provider = SqliteProvider(path=tmp_path / "test.db")
        index = atlocal.Index(provider=provider, atmosphere=None)
        with pytest.raises(TypeError, match="sample_type must be a class"):
            index.publish_schema("not-a-class", version="1.0.0")

    def test_get_schema_record(self, tmp_path):
        provider = SqliteProvider(path=tmp_path / "test.db")
        index = atlocal.Index(provider=provider, atmosphere=None)
        ref = index.publish_schema(SharedBasicSample, version="1.0.0")
        record = index.get_schema_record(ref)
        assert record.name == "SharedBasicSample"
        assert record.version == "1.0.0"

    def test_get_schema_at_uri_routes_to_atmosphere(self, tmp_path):
        """get_schema with at:// URI routes to atmosphere backend."""
        provider = SqliteProvider(path=tmp_path / "test.db")
        mock_atmo = MagicMock()
        mock_atmo.get_schema.return_value = {
            "name": "RemoteSample",
            "version": "1.0.0",
            "schemaType": "jsonSchema",
            "schema": {
                "schemaBody": {
                    "properties": {"label": {"type": "integer"}},
                    "required": ["label"],
                },
            },
        }

        index = atlocal.Index(provider=provider, atmosphere=None)
        index._atmosphere = mock_atmo
        index._atmosphere_deferred = False

        at_uri = "at://did:plc:abc/ac.foundation.dataset.schema/rkey123"
        result = index.get_schema(at_uri)
        assert result["name"] == "RemoteSample"
        mock_atmo.get_schema.assert_called_once_with(at_uri)

    def test_get_schema_at_uri_no_atmosphere_raises(self, tmp_path):
        """get_schema with at:// URI raises when atmosphere unavailable."""
        provider = SqliteProvider(path=tmp_path / "test.db")
        index = atlocal.Index(provider=provider, atmosphere=None)

        with pytest.raises(ValueError, match="Atmosphere backend required"):
            index.get_schema("at://did:plc:abc/ac.foundation.dataset.schema/rkey")

    def test_get_schema_handle_ref_routes_to_atmosphere(self, tmp_path):
        """get_schema with @handle/Type@version routes to atmosphere backend."""
        provider = SqliteProvider(path=tmp_path / "test.db")
        mock_atmo = MagicMock()
        mock_atmo.get_schema.return_value = {
            "name": "MnistSample",
            "version": "1.0.0",
            "schemaType": "jsonSchema",
            "schema": {
                "schemaBody": {
                    "properties": {"label": {"type": "integer"}},
                    "required": ["label"],
                },
            },
        }

        index = atlocal.Index(provider=provider, atmosphere=None)
        index._atmosphere = mock_atmo
        index._atmosphere_deferred = False

        ref = "@foundation.ac/MnistSample@1.0.0"
        result = index.get_schema(ref)
        assert result["name"] == "MnistSample"
        mock_atmo.get_schema.assert_called_once_with(ref)

    def test_get_schema_handle_ref_no_atmosphere_raises(self, tmp_path):
        """get_schema with @handle ref raises when atmosphere unavailable."""
        provider = SqliteProvider(path=tmp_path / "test.db")
        index = atlocal.Index(provider=provider, atmosphere=None)

        with pytest.raises(ValueError, match="Atmosphere backend required"):
            index.get_schema("@foundation.ac/MnistSample@1.0.0")

    def test_insert_dataset_atmosphere_path(self, tmp_path):
        """insert_dataset with at:// prefix routes to atmosphere."""
        provider = SqliteProvider(path=tmp_path / "test.db")
        mock_atmo = MagicMock()
        mock_entry = MagicMock()
        mock_atmo.insert_dataset.return_value = mock_entry

        index = atlocal.Index(provider=provider, atmosphere=None)
        index._atmosphere = mock_atmo
        index._atmosphere_deferred = False

        ds = atdata.Dataset[SharedBasicSample](url="s3://fake/data.tar")
        result = index.insert_dataset(ds, name="at://did:plc:abc/test")
        assert result is mock_entry

    def test_get_dataset_atmosphere_path(self, tmp_path):
        """get_dataset with at:// prefix routes to atmosphere."""
        provider = SqliteProvider(path=tmp_path / "test.db")
        mock_atmo = MagicMock()
        mock_entry = MagicMock()
        mock_atmo.get_dataset.return_value = mock_entry

        index = atlocal.Index(provider=provider, atmosphere=None)
        index._atmosphere = mock_atmo
        index._atmosphere_deferred = False

        result = index.get_dataset("at://did:plc:abc/coll/rkey")
        assert result is mock_entry

    def test_get_dataset_no_atmosphere_raises(self, tmp_path):
        """get_dataset with at:// prefix but no atmosphere raises."""
        provider = SqliteProvider(path=tmp_path / "test.db")
        index = atlocal.Index(provider=provider, atmosphere=None)
        with pytest.raises(ValueError, match="Atmosphere backend required"):
            index.get_dataset("at://did:plc:abc/coll/rkey")

    def test_list_datasets_atmosphere_repo(self, tmp_path):
        """list_datasets with repo='_atmosphere' delegates to atmosphere."""
        provider = SqliteProvider(path=tmp_path / "test.db")
        mock_atmo = MagicMock()
        mock_atmo.list_datasets.return_value = []

        index = atlocal.Index(provider=provider, atmosphere=None)
        index._atmosphere = mock_atmo
        index._atmosphere_deferred = False

        result = index.list_datasets(repo="_atmosphere")
        assert result == []
        mock_atmo.list_datasets.assert_called_once()

    def test_list_datasets_no_atmosphere_returns_empty(self, tmp_path):
        """list_datasets with repo='_atmosphere' but no backend returns []."""
        provider = SqliteProvider(path=tmp_path / "test.db")
        index = atlocal.Index(provider=provider, atmosphere=None)
        result = index.list_datasets(repo="_atmosphere")
        assert result == []

    def test_explicit_atmosphere_client(self, tmp_path):
        """Index accepts an explicit atmosphere client."""
        from atdata.atmosphere.client import Atmosphere

        provider = SqliteProvider(path=tmp_path / "test.db")
        mock_client = MagicMock(spec=Atmosphere)
        index = atlocal.Index(provider=provider, atmosphere=mock_client)
        assert index.atmosphere is mock_client
