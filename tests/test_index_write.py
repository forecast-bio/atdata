"""Tests for Index.write(), Index.promote_entry(), and Index.promote_dataset()."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import atdata
import atdata.local as atlocal
from atdata.providers._sqlite import SqliteProvider
from conftest import SharedBasicSample, SharedNumpySample

import numpy as np


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sqlite_provider(tmp_path: Path):
    return SqliteProvider(path=tmp_path / "test.db")


@pytest.fixture
def index(sqlite_provider):
    return atlocal.Index(provider=sqlite_provider, atmosphere=None)


@pytest.fixture
def index_with_store(sqlite_provider, tmp_path: Path):
    store = atdata.LocalDiskStore(root=tmp_path / "store")
    return atlocal.Index(
        provider=sqlite_provider,
        data_store=store,
        atmosphere=None,
    )


# ---------------------------------------------------------------------------
# Index.write() tests
# ---------------------------------------------------------------------------


class TestIndexWrite:
    """Tests for Index.write() method."""

    def test_write_basic_samples(self, index):
        samples = [SharedBasicSample(name=f"s{i}", value=i) for i in range(5)]
        entry = index.write(samples, name="basic-ds")

        assert entry.name == "basic-ds"
        assert len(entry.data_urls) >= 1

    def test_write_creates_readable_dataset(self, index):
        samples = [SharedBasicSample(name=f"s{i}", value=i) for i in range(5)]
        entry = index.write(samples, name="readable-ds")

        ds = atdata.Dataset[SharedBasicSample](url=entry.data_urls[0])
        result = list(ds.ordered())
        assert len(result) == 5

    def test_write_preserves_data(self, index):
        samples = [SharedBasicSample(name=f"s{i}", value=i * 10) for i in range(3)]
        entry = index.write(samples, name="preserve-ds")

        ds = atdata.Dataset[SharedBasicSample](url=entry.data_urls[0])
        result = sorted(list(ds.ordered()), key=lambda s: s.value)
        for i, s in enumerate(result):
            assert s.name == f"s{i}"
            assert s.value == i * 10

    def test_write_numpy_samples(self, index):
        arrays = [np.random.randn(3, 3).astype(np.float32) for _ in range(3)]
        samples = [
            SharedNumpySample(data=arr, label=f"a{i}") for i, arr in enumerate(arrays)
        ]
        entry = index.write(samples, name="numpy-ds")

        ds = atdata.Dataset[SharedNumpySample](url=entry.data_urls[0])
        result = list(ds.ordered())
        assert len(result) == 3
        for s in result:
            assert s.data.shape == (3, 3)

    def test_write_sets_schema_ref(self, index):
        samples = [SharedBasicSample(name="x", value=1)]
        entry = index.write(samples, name="schema-ds")

        # write() should set a schema_ref derived from the sample type
        assert entry.schema_ref is not None
        assert "SharedBasicSample" in entry.schema_ref

    def test_write_indexes_entry(self, index):
        samples = [SharedBasicSample(name="x", value=1)]
        index.write(samples, name="indexed-ds")

        # Should be retrievable by name
        entry = index.get_dataset("indexed-ds")
        assert entry.name == "indexed-ds"

    def test_write_with_explicit_store(self, index_with_store):
        samples = [SharedBasicSample(name=f"s{i}", value=i) for i in range(3)]
        entry = index_with_store.write(samples, name="stored-ds")

        assert entry.name == "stored-ds"
        assert len(entry.data_urls) >= 1
        # Data should be in the store's root
        for url in entry.data_urls:
            assert Path(url).exists()

    def test_write_auto_creates_local_disk_store(self, index):
        """When no data_store is configured, write() creates a LocalDiskStore."""
        samples = [SharedBasicSample(name="x", value=1)]
        entry = index.write(samples, name="auto-store-ds")

        # Should have persisted to ~/.atdata/data/ or similar
        assert len(entry.data_urls) >= 1
        for url in entry.data_urls:
            assert Path(url).exists()

    def test_write_with_maxcount(self, index):
        samples = [SharedBasicSample(name=f"s{i}", value=i) for i in range(10)]
        entry = index.write(samples, name="sharded-ds", maxcount=3)

        # All 10 samples should be readable regardless of shard layout
        ds = atdata.Dataset[SharedBasicSample](url=entry.data_urls[0])
        result = list(ds.ordered())
        assert len(result) == 10

    def test_write_empty_raises(self, index):
        with pytest.raises(ValueError, match="non-empty"):
            index.write([], name="empty-ds")

    def test_write_with_metadata(self, index):
        samples = [SharedBasicSample(name="x", value=1)]
        meta = {"source": "test", "version": 2}
        index.write(samples, name="meta-ds", metadata=meta)

        retrieved = index.get_dataset("meta-ds")
        assert retrieved.metadata is not None
        assert retrieved.metadata["source"] == "test"
        assert retrieved.metadata["version"] == 2

    def test_write_multiple_datasets(self, index):
        """Write multiple datasets and verify they coexist."""
        for i in range(3):
            samples = [SharedBasicSample(name=f"ds{i}-s{j}", value=j) for j in range(3)]
            index.write(samples, name=f"multi-{i}")

        entries = index.list_datasets()
        assert len(entries) == 3


# ---------------------------------------------------------------------------
# Index.promote_entry() tests
# ---------------------------------------------------------------------------


class TestIndexPromoteEntry:
    """Tests for Index.promote_entry() - atmosphere promotion via entry name."""

    def test_no_atmosphere_raises(self, index):
        """promote_entry requires atmosphere backend."""
        with pytest.raises(ValueError, match="Atmosphere backend required"):
            index.promote_entry("nonexistent")

    def test_missing_entry_raises(self, sqlite_provider, tmp_path: Path):
        """promote_entry raises KeyError for unknown entry names."""
        # Create an index with a mock atmosphere
        mock_atmo = MagicMock()
        with patch.object(atlocal.Index, "_get_atmosphere", return_value=mock_atmo):
            idx = atlocal.Index(provider=sqlite_provider, atmosphere=None)
            with pytest.raises(KeyError):
                idx.promote_entry("no-such-entry")

    def test_promote_entry_calls_atmosphere(self, sqlite_provider, tmp_path: Path):
        """promote_entry delegates to atmosphere publisher when backend is available."""
        idx = atlocal.Index(provider=sqlite_provider, atmosphere=None)

        # Write a real dataset and publish its schema so promote_entry can find both
        samples = [SharedBasicSample(name="x", value=1)]
        idx.write(samples, name="promotable")
        idx.publish_schema(SharedBasicSample, version="1.0.0")

        # Mock the atmosphere backend and publisher
        mock_atmo = MagicMock()
        mock_atmo.client = MagicMock()

        mock_publisher_instance = MagicMock()
        mock_publisher_instance.publish_with_urls.return_value = (
            "at://did:plc:abc/test/123"
        )

        with (
            patch.object(atlocal.Index, "_get_atmosphere", return_value=mock_atmo),
            patch(
                "atdata.atmosphere.DatasetPublisher",
                return_value=mock_publisher_instance,
            ),
            patch(
                "atdata.promote._find_or_publish_schema", return_value="at://schema/1"
            ),
        ):
            uri = idx.promote_entry("promotable")

        assert uri == "at://did:plc:abc/test/123"
        mock_publisher_instance.publish_with_urls.assert_called_once()


# ---------------------------------------------------------------------------
# Index.promote_dataset() tests
# ---------------------------------------------------------------------------


class TestIndexPromoteDataset:
    """Tests for Index.promote_dataset() - direct Dataset to atmosphere."""

    def test_no_atmosphere_raises(self, index, tmp_path: Path):
        """promote_dataset requires atmosphere backend."""
        ds = atdata.Dataset[SharedBasicSample](url="s3://fake/data.tar")
        with pytest.raises(ValueError, match="Atmosphere backend required"):
            index.promote_dataset(ds, name="test-ds")
