"""Tests for Index.write_samples(), Index.write() (deprecated),
Index.promote_entry(), and Index.promote_dataset()."""

import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import atdata
import atdata.local as atlocal
from atdata._sources import S3Source
from atdata.atmosphere.store import PDS_BLOB_LIMIT_BYTES, PDS_TOTAL_DATASET_LIMIT_BYTES
from atdata.index._index import _is_local_path, _is_credentialed_source
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

    def test_write_persists_schema(self, index):
        """write() should store the schema so decode_schema() works."""
        samples = [SharedBasicSample(name="x", value=1)]
        entry = index.write(samples, name="schema-persist")

        schema = index.get_schema(entry.schema_ref)
        assert schema["name"] == "SharedBasicSample"
        assert "fields" in schema

    def test_write_schema_decode_round_trip(self, index):
        """Schema stored by write() should be decodable back to a type."""
        samples = [SharedBasicSample(name="x", value=1)]
        entry = index.write(samples, name="decode-rt")

        decoded_type = index.decode_schema(entry.schema_ref)
        instance = decoded_type(name="test", value=42)
        assert instance.name == "test"
        assert instance.value == 42

    def test_write_does_not_overwrite_existing_schema(self, index):
        """If schema already exists, write() should not overwrite it."""
        ref = index.publish_schema(SharedBasicSample, version="1.0.0")
        original = index.get_schema(ref)

        samples = [SharedBasicSample(name="x", value=1)]
        index.write(samples, name="no-overwrite", schema_ref=ref)

        after = index.get_schema(ref)
        assert original == after


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
        call_kwargs = mock_publisher_instance.publish_with_urls.call_args[1]
        assert call_kwargs["name"] == "promotable"
        assert call_kwargs["schema_uri"] == "at://schema/1"


# ---------------------------------------------------------------------------
# Index.promote_dataset() tests
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestIndexPromoteDataset:
    """Tests for deprecated Index.promote_dataset()."""

    def test_no_atmosphere_raises(self, index, tmp_path: Path):
        """promote_dataset requires atmosphere backend."""
        ds = atdata.Dataset[SharedBasicSample](url="s3://fake/data.tar")
        with pytest.raises(ValueError, match="Atmosphere backend required"):
            index.promote_dataset(ds, name="test-ds")


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestHelpers:
    """Tests for _is_local_path, _is_credentialed_source, etc."""

    @pytest.mark.parametrize(
        "url,expected",
        [
            ("/tmp/data.tar", True),
            ("/Users/me/datasets/shard.tar", True),
            ("file:///tmp/data.tar", True),
            ("C:\\data\\shard.tar", True),
            ("https://example.com/data.tar", False),
            ("http://example.com/data.tar", False),
            ("s3://bucket/key.tar", False),
            ("at://did:plc:abc/blob/cid123", False),
        ],
    )
    def test_is_local_path(self, url, expected):
        assert _is_local_path(url) == expected

    def test_is_credentialed_source_s3(self):
        source = S3Source(bucket="b", keys=["k.tar"], access_key="AK", secret_key="SK")
        ds = MagicMock()
        ds.source = source
        # Patch Dataset check since MagicMock isn't a real Dataset
        with patch("atdata.index._index.Dataset", MagicMock):
            assert _is_credentialed_source(ds) is True

    def test_is_credentialed_source_url(self):
        ds = MagicMock()
        ds.source = MagicMock(spec=[])  # not an S3Source
        with patch("atdata.index._index.Dataset", MagicMock):
            assert _is_credentialed_source(ds) is False


# ---------------------------------------------------------------------------
# Index.write_samples() tests
# ---------------------------------------------------------------------------


class TestWriteSamples:
    """Tests for the renamed Index.write_samples() method."""

    def test_write_samples_basic(self, index):
        samples = [SharedBasicSample(name=f"s{i}", value=i) for i in range(5)]
        entry = index.write_samples(samples, name="ws-basic")

        assert entry.name == "ws-basic"
        assert len(entry.data_urls) >= 1

    def test_write_samples_readable(self, index):
        samples = [SharedBasicSample(name=f"s{i}", value=i) for i in range(5)]
        entry = index.write_samples(samples, name="ws-readable")

        ds = atdata.Dataset[SharedBasicSample](url=entry.data_urls[0])
        result = list(ds.ordered())
        assert len(result) == 5

    def test_write_samples_with_store(self, index_with_store):
        samples = [SharedBasicSample(name=f"s{i}", value=i) for i in range(3)]
        entry = index_with_store.write_samples(samples, name="ws-stored")

        assert entry.name == "ws-stored"
        for url in entry.data_urls:
            assert Path(url).exists()

    def test_write_samples_with_explicit_data_store(self, sqlite_provider, tmp_path):
        """data_store kwarg overrides the repo's default store."""
        store = atdata.LocalDiskStore(root=tmp_path / "custom-store")
        idx = atlocal.Index(provider=sqlite_provider, atmosphere=None)
        samples = [SharedBasicSample(name="x", value=1)]
        entry = idx.write_samples(samples, name="ws-custom", data_store=store)

        for url in entry.data_urls:
            assert str(tmp_path / "custom-store") in url


# ---------------------------------------------------------------------------
# Atmosphere size guard tests
# ---------------------------------------------------------------------------


class TestAtmosphereSizeGuards:
    """Tests for PDS size limits on atmosphere targets."""

    def test_maxsize_over_blob_limit_raises(self, sqlite_provider):
        mock_atmo = MagicMock()
        mock_atmo.client = MagicMock()
        mock_atmo.client.did = "did:plc:test"

        idx = atlocal.Index(provider=sqlite_provider, atmosphere=None)
        samples = [SharedBasicSample(name="x", value=1)]

        with patch.object(
            atlocal.Index, "_resolve_prefix", return_value=("_atmosphere", "test", None)
        ):
            with patch.object(atlocal.Index, "_get_atmosphere", return_value=mock_atmo):
                with pytest.raises(ValueError, match="exceeds PDS blob limit"):
                    idx.write_samples(
                        samples,
                        name="@handle/test",
                        maxsize=PDS_BLOB_LIMIT_BYTES + 1,
                    )

    def test_maxsize_over_blob_limit_force_ok(self, sqlite_provider, tmp_path):
        """force=True bypasses the per-shard limit check."""
        mock_atmo = MagicMock()
        mock_atmo.client = MagicMock()
        mock_atmo.client.did = "did:plc:test"

        mock_store = MagicMock()
        mock_store.write_shards.return_value = ["at://did:plc:test/blob/abc"]

        mock_entry = MagicMock()
        mock_atmo.insert_dataset.return_value = mock_entry

        idx = atlocal.Index(provider=sqlite_provider, atmosphere=None)
        samples = [SharedBasicSample(name="x", value=1)]

        with (
            patch.object(
                atlocal.Index,
                "_resolve_prefix",
                return_value=("_atmosphere", "test", None),
            ),
            patch.object(atlocal.Index, "_get_atmosphere", return_value=mock_atmo),
        ):
            entry = idx.write_samples(
                samples,
                name="@handle/test",
                maxsize=PDS_BLOB_LIMIT_BYTES + 1,
                force=True,
                data_store=mock_store,
            )
            assert entry == mock_entry

    def test_total_size_guard(self, sqlite_provider, tmp_path):
        """Datasets exceeding 1GB total raise ValueError for atmosphere."""
        mock_atmo = MagicMock()
        mock_atmo.client = MagicMock()
        mock_atmo.client.did = "did:plc:test"

        mock_store = MagicMock()
        mock_store.write_shards.return_value = ["at://did:plc:test/blob/abc"]

        idx = atlocal.Index(provider=sqlite_provider, atmosphere=None)
        samples = [SharedBasicSample(name="x", value=1)]

        with (
            patch.object(
                atlocal.Index,
                "_resolve_prefix",
                return_value=("_atmosphere", "test", None),
            ),
            patch.object(atlocal.Index, "_get_atmosphere", return_value=mock_atmo),
            patch(
                "atdata.index._index._estimate_dataset_bytes",
                return_value=PDS_TOTAL_DATASET_LIMIT_BYTES + 1,
            ),
        ):
            with pytest.raises(ValueError, match="exceeds atmosphere limit"):
                idx.write_samples(
                    samples,
                    name="@handle/test",
                    data_store=mock_store,
                )


# ---------------------------------------------------------------------------
# insert_dataset atmosphere behaviour tests
# ---------------------------------------------------------------------------


class TestInsertDatasetAtmosphere:
    """Tests for insert_dataset with atmosphere targets."""

    def test_local_source_defaults_to_pds_blob_store(self, sqlite_provider, tmp_path):
        """Local filesystem source should auto-upload via PDSBlobStore and use storageBlobs."""
        mock_atmo = MagicMock()
        mock_atmo.client = MagicMock()
        mock_atmo.client.did = "did:plc:test"
        mock_entry = MagicMock()
        mock_atmo.insert_dataset.return_value = mock_entry

        idx = atlocal.Index(provider=sqlite_provider, atmosphere=None)

        # Create a real local dataset
        samples = [SharedBasicSample(name="x", value=1)]
        ds = atdata.write_samples(samples, tmp_path / "data.tar")

        mock_blob_ref = {
            "$type": "blob",
            "ref": {"$link": "bafyreiabc"},
            "mimeType": "application/x-tar",
            "size": 1024,
        }

        def fake_write_shards(self_store, ds, *, prefix, **kwargs):
            from atdata.atmosphere.store import ShardUploadResult

            return ShardUploadResult(
                ["at://did:plc:test/blob/bafyreiabc"], [mock_blob_ref]
            )

        with (
            patch.object(
                atlocal.Index,
                "_resolve_prefix",
                return_value=("_atmosphere", "test", None),
            ),
            patch.object(atlocal.Index, "_get_atmosphere", return_value=mock_atmo),
            patch(
                "atdata.atmosphere.store.PDSBlobStore.write_shards",
                autospec=True,
                side_effect=fake_write_shards,
            ) as mock_ws,
        ):
            entry = idx.insert_dataset(ds, name="@handle/test")

        assert entry == mock_entry
        mock_ws.assert_called_once()
        # Verify blob_refs were passed to atmosphere backend for storageBlobs
        call_kwargs = mock_atmo.insert_dataset.call_args[1]
        assert call_kwargs["blob_refs"] == [mock_blob_ref]
        assert call_kwargs["data_urls"] == ["at://did:plc:test/blob/bafyreiabc"]

    def test_remote_source_references_urls(self, sqlite_provider):
        """Public http source should reference existing URLs, not copy."""
        mock_atmo = MagicMock()
        mock_atmo.client = MagicMock()
        mock_entry = MagicMock()
        mock_atmo.insert_dataset.return_value = mock_entry

        ds = MagicMock()
        ds.url = "https://example.com/data.tar"
        ds.list_shards.return_value = ["https://example.com/data-000.tar"]
        ds.source = MagicMock(spec=[])  # not S3Source
        ds._metadata = None

        idx = atlocal.Index(provider=sqlite_provider, atmosphere=None)

        with (
            patch.object(
                atlocal.Index,
                "_resolve_prefix",
                return_value=("_atmosphere", "test", None),
            ),
            patch.object(atlocal.Index, "_get_atmosphere", return_value=mock_atmo),
        ):
            entry = idx.insert_dataset(ds, name="@handle/test")

        assert entry == mock_entry
        call_kwargs = mock_atmo.insert_dataset.call_args[1]
        assert call_kwargs["data_urls"] == ["https://example.com/data-000.tar"]

    def test_credentialed_source_errors_by_default(self, sqlite_provider):
        """S3Source with credentials should raise without copy=True."""
        mock_atmo = MagicMock()
        mock_atmo.client = MagicMock()

        s3_source = S3Source(
            bucket="b", keys=["k.tar"], access_key="AK", secret_key="SK"
        )
        ds = MagicMock()
        ds.url = "s3://b/k.tar"
        ds.source = s3_source

        idx = atlocal.Index(provider=sqlite_provider, atmosphere=None)

        with (
            patch.object(
                atlocal.Index,
                "_resolve_prefix",
                return_value=("_atmosphere", "test", None),
            ),
            patch.object(atlocal.Index, "_get_atmosphere", return_value=mock_atmo),
        ):
            with pytest.raises(ValueError, match="credentialed source"):
                idx.insert_dataset(ds, name="@handle/test")

    def test_credentialed_source_with_copy(self, sqlite_provider):
        """S3Source with copy=True should copy via PDSBlobStore."""
        mock_atmo = MagicMock()
        mock_atmo.client = MagicMock()
        mock_atmo.client.did = "did:plc:test"
        mock_entry = MagicMock()
        mock_atmo.insert_dataset.return_value = mock_entry

        s3_source = S3Source(
            bucket="b", keys=["k.tar"], access_key="AK", secret_key="SK"
        )
        ds = MagicMock()
        ds.url = "s3://b/k.tar"
        ds.source = s3_source
        ds.list_shards.return_value = ["s3://b/k.tar"]
        ds._metadata = None

        idx = atlocal.Index(provider=sqlite_provider, atmosphere=None)

        with (
            patch.object(
                atlocal.Index,
                "_resolve_prefix",
                return_value=("_atmosphere", "test", None),
            ),
            patch.object(atlocal.Index, "_get_atmosphere", return_value=mock_atmo),
            patch(
                "atdata.atmosphere.store.PDSBlobStore.write_shards",
                return_value=["at://did:plc:test/blob/abc"],
            ),
        ):
            entry = idx.insert_dataset(ds, name="@handle/test", copy=True)

        assert entry == mock_entry

    def test_credentialed_source_with_data_store(self, sqlite_provider):
        """Explicit data_store implies copy for credentialed sources."""
        mock_atmo = MagicMock()
        mock_atmo.client = MagicMock()
        mock_entry = MagicMock()
        mock_atmo.insert_dataset.return_value = mock_entry

        mock_store = MagicMock()
        mock_store.write_shards.return_value = ["s3://public/output.tar"]

        s3_source = S3Source(
            bucket="b", keys=["k.tar"], access_key="AK", secret_key="SK"
        )
        ds = MagicMock()
        ds.url = "s3://b/k.tar"
        ds.source = s3_source
        ds._metadata = None

        idx = atlocal.Index(provider=sqlite_provider, atmosphere=None)

        with (
            patch.object(
                atlocal.Index,
                "_resolve_prefix",
                return_value=("_atmosphere", "test", None),
            ),
            patch.object(atlocal.Index, "_get_atmosphere", return_value=mock_atmo),
        ):
            entry = idx.insert_dataset(ds, name="@handle/test", data_store=mock_store)

        mock_store.write_shards.assert_called_once()
        assert entry == mock_entry

    def test_public_source_with_copy(self, sqlite_provider):
        """copy=True forces copy even for public http sources."""
        mock_atmo = MagicMock()
        mock_atmo.client = MagicMock()
        mock_atmo.client.did = "did:plc:test"
        mock_entry = MagicMock()
        mock_atmo.insert_dataset.return_value = mock_entry

        ds = MagicMock()
        ds.url = "https://example.com/data.tar"
        ds.source = MagicMock(spec=[])  # not S3Source
        ds._metadata = None

        idx = atlocal.Index(provider=sqlite_provider, atmosphere=None)

        with (
            patch.object(
                atlocal.Index,
                "_resolve_prefix",
                return_value=("_atmosphere", "test", None),
            ),
            patch.object(atlocal.Index, "_get_atmosphere", return_value=mock_atmo),
            patch(
                "atdata.atmosphere.store.PDSBlobStore.write_shards",
                return_value=["at://did:plc:test/blob/abc"],
            ),
        ):
            entry = idx.insert_dataset(ds, name="@handle/test", copy=True)

        assert entry == mock_entry


# ---------------------------------------------------------------------------
# Deprecation warning tests
# ---------------------------------------------------------------------------


class TestDeprecationWarnings:
    """Verify deprecated methods emit warnings."""

    def test_write_emits_deprecation(self, index):
        samples = [SharedBasicSample(name="x", value=1)]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            index.write(samples, name="dep-write")

        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert any("write_samples" in str(dw.message) for dw in dep_warnings)

    def test_add_entry_emits_deprecation(self, index, tmp_path):
        samples = [SharedBasicSample(name="x", value=1)]
        ds = atdata.write_samples(samples, tmp_path / "dep.tar")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            index.add_entry(ds, name="dep-add")

        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert any("insert_dataset" in str(dw.message) for dw in dep_warnings)

    def test_promote_entry_emits_deprecation(self, index):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                index.promote_entry("nonexistent")
            except (ValueError, KeyError):
                pass

        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert any("insert_dataset" in str(dw.message) for dw in dep_warnings)

    def test_promote_dataset_emits_deprecation(self, index):
        ds = MagicMock()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                index.promote_dataset(ds, name="test")
            except (ValueError, KeyError):
                pass

        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert any("insert_dataset" in str(dw.message) for dw in dep_warnings)
