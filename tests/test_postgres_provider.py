"""Tests for PostgresProvider with fully mocked psycopg connections."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import msgpack
import pytest


@pytest.fixture
def mock_psycopg():
    """Mock psycopg module and connection.

    The PostgresProvider lazily imports psycopg inside __init__, so we
    inject a MagicMock into sys.modules so that ``import psycopg``
    resolves to our mock.
    """
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    mock_pg = MagicMock()
    mock_pg.connect.return_value = mock_conn

    with patch.dict("sys.modules", {"psycopg": mock_pg}):
        yield {
            "module": mock_pg,
            "conn": mock_conn,
            "cursor": mock_cursor,
        }


def _make_provider(mock_psycopg):
    """Create a PostgresProvider using the mocked psycopg fixture."""
    from atdata.providers._postgres import PostgresProvider

    return PostgresProvider(dsn="postgresql://localhost/test")


def _make_entry(**overrides):
    """Create a LocalDatasetEntry with sensible defaults."""
    from atdata.local._entry import LocalDatasetEntry

    defaults = dict(
        name="test_dataset",
        schema_ref="atdata://local/sampleSchema/TestSample@1.0.0",
        data_urls=["/data/shard-000000.tar"],
        metadata={"split": "train"},
    )
    defaults.update(overrides)
    return LocalDatasetEntry(**defaults)


# ------------------------------------------------------------------
# __init__
# ------------------------------------------------------------------


class TestInit:
    def test_connects_and_creates_tables(self, mock_psycopg):
        provider = _make_provider(mock_psycopg)
        conn = mock_psycopg["conn"]
        cursor = mock_psycopg["cursor"]

        mock_psycopg["module"].connect.assert_called_once_with(
            "postgresql://localhost/test", autocommit=False
        )
        cursor.execute.assert_called_once()  # CREATE TABLE statements
        conn.commit.assert_called_once()
        assert provider._conn is conn

    def test_import_error_when_psycopg_missing(self):
        """When psycopg is not installed, __init__ raises ImportError."""
        # Remove psycopg from sys.modules so the real import fails,
        # and also ensure no mock is in place.
        with patch.dict("sys.modules", {"psycopg": None}):
            from atdata.providers._postgres import PostgresProvider

            with pytest.raises(ImportError, match="psycopg"):
                PostgresProvider(dsn="postgresql://localhost/test")


# ------------------------------------------------------------------
# store_entry
# ------------------------------------------------------------------


class TestStoreEntry:
    def test_inserts_entry_and_commits(self, mock_psycopg):
        provider = _make_provider(mock_psycopg)
        cursor = mock_psycopg["cursor"]
        conn = mock_psycopg["conn"]
        conn.commit.reset_mock()

        entry = _make_entry()
        provider.store_entry(entry)

        cursor.execute.assert_called()
        args = cursor.execute.call_args
        sql = args[0][0]
        params = args[0][1]

        assert "INSERT INTO dataset_entries" in sql
        assert "ON CONFLICT" in sql
        assert params[0] == entry.cid
        assert params[1] == entry.name
        assert params[2] == entry.schema_ref
        assert params[3] == msgpack.packb(entry.data_urls)
        assert params[4] == msgpack.packb(entry.metadata)
        assert params[5] == entry._legacy_uuid
        conn.commit.assert_called_once()

    def test_stores_entry_with_none_metadata(self, mock_psycopg):
        provider = _make_provider(mock_psycopg)
        cursor = mock_psycopg["cursor"]

        entry = _make_entry(metadata=None)
        provider.store_entry(entry)

        params = cursor.execute.call_args[0][1]
        # metadata param should be None when entry.metadata is None
        assert params[4] is None


# ------------------------------------------------------------------
# get_entry_by_cid
# ------------------------------------------------------------------


class TestGetEntryByCid:
    def test_returns_entry_when_found(self, mock_psycopg):
        provider = _make_provider(mock_psycopg)
        cursor = mock_psycopg["cursor"]

        entry = _make_entry()
        row = (
            entry.cid,
            entry.name,
            entry.schema_ref,
            msgpack.packb(entry.data_urls),
            msgpack.packb(entry.metadata),
            None,  # legacy_uuid
        )
        cursor.fetchone.return_value = row

        result = provider.get_entry_by_cid(entry.cid)

        assert result.cid == entry.cid
        assert result.name == entry.name
        assert result.schema_ref == entry.schema_ref
        assert result.data_urls == entry.data_urls
        assert result.metadata == entry.metadata

    def test_raises_key_error_when_not_found(self, mock_psycopg):
        provider = _make_provider(mock_psycopg)
        cursor = mock_psycopg["cursor"]
        cursor.fetchone.return_value = None

        with pytest.raises(KeyError, match="LocalDatasetEntry not found"):
            provider.get_entry_by_cid("nonexistent_cid")

    def test_handles_none_metadata_blob(self, mock_psycopg):
        provider = _make_provider(mock_psycopg)
        cursor = mock_psycopg["cursor"]

        entry = _make_entry(metadata=None)
        row = (
            entry.cid,
            entry.name,
            entry.schema_ref,
            msgpack.packb(entry.data_urls),
            None,  # metadata blob
            None,  # legacy_uuid
        )
        cursor.fetchone.return_value = row

        result = provider.get_entry_by_cid(entry.cid)
        assert result.metadata is None


# ------------------------------------------------------------------
# get_entry_by_name
# ------------------------------------------------------------------


class TestGetEntryByName:
    def test_returns_entry_when_found(self, mock_psycopg):
        provider = _make_provider(mock_psycopg)
        cursor = mock_psycopg["cursor"]

        entry = _make_entry()
        row = (
            entry.cid,
            entry.name,
            entry.schema_ref,
            msgpack.packb(entry.data_urls),
            msgpack.packb(entry.metadata),
            None,
        )
        cursor.fetchone.return_value = row

        result = provider.get_entry_by_name("test_dataset")

        sql = cursor.execute.call_args[0][0]
        assert "WHERE name = %s" in sql
        assert result.name == "test_dataset"

    def test_raises_key_error_when_not_found(self, mock_psycopg):
        provider = _make_provider(mock_psycopg)
        cursor = mock_psycopg["cursor"]
        cursor.fetchone.return_value = None

        with pytest.raises(KeyError, match="No entry with name"):
            provider.get_entry_by_name("missing_dataset")


# ------------------------------------------------------------------
# iter_entries
# ------------------------------------------------------------------


class TestIterEntries:
    def test_yields_all_entries(self, mock_psycopg):
        provider = _make_provider(mock_psycopg)
        cursor = mock_psycopg["cursor"]

        entry1 = _make_entry(name="ds1", data_urls=["/data/shard-000000.tar"])
        entry2 = _make_entry(name="ds2", data_urls=["/data/shard-000001.tar"])

        rows = [
            (
                entry1.cid,
                entry1.name,
                entry1.schema_ref,
                msgpack.packb(entry1.data_urls),
                msgpack.packb(entry1.metadata),
                None,
            ),
            (
                entry2.cid,
                entry2.name,
                entry2.schema_ref,
                msgpack.packb(entry2.data_urls),
                msgpack.packb(entry2.metadata),
                None,
            ),
        ]
        cursor.__iter__ = MagicMock(return_value=iter(rows))

        results = list(provider.iter_entries())

        assert len(results) == 2
        assert results[0].name == "ds1"
        assert results[1].name == "ds2"

    def test_yields_nothing_when_empty(self, mock_psycopg):
        provider = _make_provider(mock_psycopg)
        cursor = mock_psycopg["cursor"]
        cursor.__iter__ = MagicMock(return_value=iter([]))

        results = list(provider.iter_entries())
        assert results == []


# ------------------------------------------------------------------
# store_schema
# ------------------------------------------------------------------


class TestStoreSchema:
    def test_inserts_schema_and_commits(self, mock_psycopg):
        provider = _make_provider(mock_psycopg)
        cursor = mock_psycopg["cursor"]
        conn = mock_psycopg["conn"]
        conn.commit.reset_mock()

        provider.store_schema("TestSample", "1.0.0", '{"name": "TestSample"}')

        args = cursor.execute.call_args
        sql = args[0][0]
        params = args[0][1]

        assert "INSERT INTO schemas" in sql
        assert "ON CONFLICT" in sql
        assert params == ("TestSample", "1.0.0", '{"name": "TestSample"}')
        conn.commit.assert_called_once()


# ------------------------------------------------------------------
# get_schema_json
# ------------------------------------------------------------------


class TestGetSchemaJson:
    def test_returns_schema_when_found(self, mock_psycopg):
        provider = _make_provider(mock_psycopg)
        cursor = mock_psycopg["cursor"]

        cursor.fetchone.return_value = ('{"name": "TestSample"}',)

        result = provider.get_schema_json("TestSample", "1.0.0")

        assert result == '{"name": "TestSample"}'
        sql = cursor.execute.call_args[0][0]
        assert "SELECT schema_json FROM schemas" in sql
        params = cursor.execute.call_args[0][1]
        assert params == ("TestSample", "1.0.0")

    def test_returns_none_when_not_found(self, mock_psycopg):
        provider = _make_provider(mock_psycopg)
        cursor = mock_psycopg["cursor"]
        cursor.fetchone.return_value = None

        result = provider.get_schema_json("NonExistent", "0.0.0")
        assert result is None


# ------------------------------------------------------------------
# iter_schemas
# ------------------------------------------------------------------


class TestIterSchemas:
    def test_yields_all_schemas(self, mock_psycopg):
        provider = _make_provider(mock_psycopg)
        cursor = mock_psycopg["cursor"]

        rows = [
            ("SampleA", "1.0.0", '{"name": "SampleA"}'),
            ("SampleB", "2.1.0", '{"name": "SampleB"}'),
        ]
        cursor.__iter__ = MagicMock(return_value=iter(rows))

        results = list(provider.iter_schemas())

        assert len(results) == 2
        assert results[0] == ("SampleA", "1.0.0", '{"name": "SampleA"}')
        assert results[1] == ("SampleB", "2.1.0", '{"name": "SampleB"}')

    def test_yields_nothing_when_empty(self, mock_psycopg):
        provider = _make_provider(mock_psycopg)
        cursor = mock_psycopg["cursor"]
        cursor.__iter__ = MagicMock(return_value=iter([]))

        results = list(provider.iter_schemas())
        assert results == []


# ------------------------------------------------------------------
# find_latest_version
# ------------------------------------------------------------------


class TestFindLatestVersion:
    def test_returns_latest_semver(self, mock_psycopg):
        provider = _make_provider(mock_psycopg)
        cursor = mock_psycopg["cursor"]

        versions = [("1.0.0",), ("1.2.0",), ("1.1.0",)]
        cursor.__iter__ = MagicMock(return_value=iter(versions))

        result = provider.find_latest_version("TestSample")

        assert result == "1.2.0"
        sql = cursor.execute.call_args[0][0]
        assert "SELECT version FROM schemas WHERE name = %s" in sql

    def test_returns_none_when_no_versions(self, mock_psycopg):
        provider = _make_provider(mock_psycopg)
        cursor = mock_psycopg["cursor"]
        cursor.__iter__ = MagicMock(return_value=iter([]))

        result = provider.find_latest_version("NonExistent")
        assert result is None

    def test_skips_invalid_semver_strings(self, mock_psycopg):
        provider = _make_provider(mock_psycopg)
        cursor = mock_psycopg["cursor"]

        versions = [("not-a-version",), ("2.0.0",), ("also-bad",)]
        cursor.__iter__ = MagicMock(return_value=iter(versions))

        result = provider.find_latest_version("TestSample")
        assert result == "2.0.0"

    def test_returns_none_when_all_versions_invalid(self, mock_psycopg):
        provider = _make_provider(mock_psycopg)
        cursor = mock_psycopg["cursor"]

        versions = [("bad",), ("worse",)]
        cursor.__iter__ = MagicMock(return_value=iter(versions))

        result = provider.find_latest_version("TestSample")
        assert result is None


# ------------------------------------------------------------------
# close
# ------------------------------------------------------------------


class TestClose:
    def test_closes_connection(self, mock_psycopg):
        provider = _make_provider(mock_psycopg)
        conn = mock_psycopg["conn"]

        provider.close()

        conn.close.assert_called_once()
