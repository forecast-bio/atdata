"""Tests for atmosphere label integration in Index write/read paths.

Covers:
- Bug 1: _AtmosphereBackend.insert_dataset creates label records
- Bug 2: Index.get_label and get_dataset resolve through atmosphere labels
- End-to-end: write_samples → load_dataset with @handle/name pattern
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

import atdata
from atdata.atmosphere._types import LEXICON_NAMESPACE
from atdata.atmosphere.labels import LabelPublisher, LabelLoader
from atdata.atmosphere import AtmosphereIndexEntry
from atdata.index._index import Index
from atdata.repository import _AtmosphereBackend
from atdata.testing import MockAtmosphere


@atdata.packable
class AtmoLabelSample:
    text: str
    value: int


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_atmo():
    """Authenticated MockAtmosphere for label tests."""
    client = MockAtmosphere(
        did="did:plc:testlabel000000",
        handle="test.label.social",
    )
    client.login("test.label.social", "password")
    yield client
    client.reset()


class _FakeAtmoBackend:
    """Lightweight fake of _AtmosphereBackend using MockAtmosphere.

    Uses real LabelPublisher/LabelLoader but stubs dataset operations
    so tests can focus on label integration without full ATProto setup.
    """

    def __init__(self, mock: MockAtmosphere):
        self.client = mock
        self._label_publisher = LabelPublisher(mock)
        self._label_loader = LabelLoader(mock)
        self._datasets: dict[str, dict] = {}
        self._counter = 0

    def insert_dataset(
        self,
        ds,
        *,
        name,
        schema_ref=None,
        data_urls=None,
        blob_refs=None,
        checksums=None,
        **kwargs,
    ):
        """Stub insert that stores a dataset record and publishes a label."""
        self._counter += 1
        uri = (
            f"at://{self.client.did}/{LEXICON_NAMESPACE}.record/{name}-{self._counter}"
        )
        record = {
            "$type": f"{LEXICON_NAMESPACE}.record",
            "name": name,
            "schemaRef": schema_ref or "",
            "storage": {
                "$type": f"{LEXICON_NAMESPACE}#storageExternal",
                "urls": data_urls or [],
            },
        }
        self._datasets[uri] = record

        # Publish label — mirrors the real _AtmosphereBackend fix
        self._label_publisher.publish(
            name=name,
            dataset_uri=uri,
            version=kwargs.get("version"),
            description=kwargs.get("description"),
        )

        return AtmosphereIndexEntry(uri, record)

    def get_dataset(self, ref):
        """Retrieve a previously inserted dataset by AT URI."""
        if ref not in self._datasets:
            raise KeyError(f"Dataset not found: {ref}")
        return AtmosphereIndexEntry(ref, self._datasets[ref])

    def resolve_label(self, handle_or_did, name, version=None):
        """Delegate to real LabelLoader.resolve()."""
        return self._label_loader.resolve(handle_or_did, name, version)

    @property
    def data_store(self):
        return None


@pytest.fixture
def atmo_index(tmp_path, mock_atmo):
    """Index with injected _FakeAtmoBackend for atmosphere label tests."""
    index = Index(provider="sqlite", path=tmp_path / "index.db", atmosphere=None)
    backend = _FakeAtmoBackend(mock_atmo)
    index._atmosphere = backend
    index._atmosphere_deferred = False
    return index


# ---------------------------------------------------------------------------
# Bug 1: _AtmosphereBackend.insert_dataset creates label records
# ---------------------------------------------------------------------------


class TestAtmosphereBackendLabelCreation:
    """Verify that _AtmosphereBackend.insert_dataset publishes a label record."""

    def test_insert_dataset_creates_label(self, mock_atmo):
        """insert_dataset should publish a label record alongside the dataset."""
        backend = _FakeAtmoBackend(mock_atmo)

        ds = MagicMock()
        ds.sample_type = AtmoLabelSample
        ds.url = "http://example.com/data.tar"
        ds.list_shards.return_value = ["http://example.com/data.tar"]

        entry = backend.insert_dataset(
            ds,
            name="test-mnist",
            data_urls=["http://example.com/data.tar"],
            description="MNIST test",
        )

        # Verify label was created
        labels = mock_atmo.list_labels()
        assert len(labels) == 1
        assert labels[0]["name"] == "test-mnist"
        assert labels[0]["datasetUri"] == entry.uri

    def test_insert_dataset_label_has_version(self, mock_atmo):
        """Label record includes version when provided in kwargs."""
        backend = _FakeAtmoBackend(mock_atmo)

        ds = MagicMock()
        backend.insert_dataset(
            ds,
            name="versioned-ds",
            data_urls=["http://example.com/data.tar"],
            version="2.0.0",
        )

        labels = mock_atmo.list_labels()
        assert len(labels) == 1
        assert labels[0]["version"] == "2.0.0"

    def test_insert_dataset_label_without_version(self, mock_atmo):
        """Label record works without version."""
        backend = _FakeAtmoBackend(mock_atmo)

        ds = MagicMock()
        backend.insert_dataset(
            ds,
            name="unversioned-ds",
            data_urls=["http://example.com/data.tar"],
        )

        labels = mock_atmo.list_labels()
        assert len(labels) == 1
        assert labels[0]["name"] == "unversioned-ds"
        assert "version" not in labels[0]

    def test_real_backend_initializes_label_publisher(self):
        """_AtmosphereBackend._ensure_loaders initializes label publisher/loader."""
        from atdata.atmosphere.client import Atmosphere

        mock_sdk = MagicMock()
        mock_sdk.me = MagicMock(did="did:plc:test123", handle="test.social")
        login_resp = MagicMock(did="did:plc:test123", handle="test.social")
        mock_sdk.login.return_value = login_resp

        atmo = Atmosphere(_client=mock_sdk)
        atmo._login("test.social", "pass")

        backend = _AtmosphereBackend(atmo)
        backend._ensure_loaders()

        assert backend._label_publisher is not None
        assert backend._label_loader is not None
        assert isinstance(backend._label_publisher, LabelPublisher)
        assert isinstance(backend._label_loader, LabelLoader)


# ---------------------------------------------------------------------------
# Bug 2: Index.get_label resolves through atmosphere labels
# ---------------------------------------------------------------------------


class TestIndexGetLabelAtmosphere:
    """Verify that Index.get_label routes @handle/name through atmosphere."""

    def test_get_label_atmosphere_resolves(self, atmo_index, mock_atmo):
        """get_label('@did/name') resolves through atmosphere labels."""
        # Pre-create a dataset and label in the mock
        backend = atmo_index._atmosphere
        ds = MagicMock()
        backend.insert_dataset(
            ds,
            name="test-ds",
            data_urls=["http://example.com/data.tar"],
        )

        # Resolve through the index using DID (avoids handle resolution)
        entry = atmo_index.get_label(f"@{mock_atmo.did}/test-ds")
        assert isinstance(entry, AtmosphereIndexEntry)
        assert entry.name == "test-ds"

    def test_get_label_atmosphere_with_version(self, atmo_index, mock_atmo):
        """get_label with version resolves the correct versioned label."""
        backend = atmo_index._atmosphere
        ds = MagicMock()

        backend.insert_dataset(
            ds,
            name="versioned",
            data_urls=["http://example.com/v1.tar"],
            version="1.0.0",
        )
        backend.insert_dataset(
            ds,
            name="versioned",
            data_urls=["http://example.com/v2.tar"],
            version="2.0.0",
        )

        entry = atmo_index.get_label(f"@{mock_atmo.did}/versioned", version="1.0.0")
        # versioned-1 is v1.0.0, versioned-2 is v2.0.0
        assert entry.uri.endswith("versioned-1")

    def test_get_label_atmosphere_not_found_raises(self, atmo_index, mock_atmo):
        """get_label raises KeyError when no matching atmosphere label exists."""
        with pytest.raises(KeyError, match="No label"):
            atmo_index.get_label(f"@{mock_atmo.did}/nonexistent")

    def test_get_label_atmosphere_unavailable_raises(self, tmp_path):
        """get_label raises ValueError when atmosphere is disabled."""
        index = Index(provider="sqlite", path=tmp_path / "index.db", atmosphere=None)
        with pytest.raises(ValueError, match="Atmosphere backend required"):
            index.get_label("@some.handle/dataset")

    def test_get_label_no_handle_raises(self, tmp_path):
        """get_label raises KeyError for atmosphere path without handle."""
        index = Index(provider="sqlite", path=tmp_path / "index.db", atmosphere=None)
        index._atmosphere = _FakeAtmoBackend(
            MockAtmosphere(did="did:plc:x", handle="x.social")
        )
        index._atmosphere_deferred = False

        # Bare atmosphere ref without handle (edge case)
        # _resolve_prefix("@foo") → ("_atmosphere", "foo", None)
        with pytest.raises(KeyError, match="Cannot resolve atmosphere label"):
            index.get_label("@foo")


# ---------------------------------------------------------------------------
# Bug 2: Index.get_dataset resolves through atmosphere labels
# ---------------------------------------------------------------------------


class TestIndexGetDatasetAtmosphere:
    """Verify that Index.get_dataset routes bare names through labels."""

    def test_get_dataset_bare_name_resolves_via_label(self, atmo_index, mock_atmo):
        """get_dataset('@did/name') resolves through labels for bare names."""
        backend = atmo_index._atmosphere
        ds = MagicMock()
        backend.insert_dataset(
            ds,
            name="my-dataset",
            data_urls=["http://example.com/data.tar"],
        )

        entry = atmo_index.get_dataset(f"@{mock_atmo.did}/my-dataset")
        assert isinstance(entry, AtmosphereIndexEntry)
        assert entry.name == "my-dataset"

    def test_get_dataset_at_uri_bypasses_labels(self, atmo_index, mock_atmo):
        """get_dataset with AT URI goes directly to backend, not labels."""
        backend = atmo_index._atmosphere
        uri = f"at://{mock_atmo.did}/{LEXICON_NAMESPACE}.record/direct-record"
        backend._datasets[uri] = {
            "$type": f"{LEXICON_NAMESPACE}.record",
            "name": "direct",
            "schemaRef": "",
            "storage": {
                "$type": f"{LEXICON_NAMESPACE}#storageExternal",
                "urls": ["http://example.com/direct.tar"],
            },
        }

        entry = atmo_index.get_dataset(uri)
        assert entry.name == "direct"

    def test_get_dataset_no_label_raises(self, atmo_index, mock_atmo):
        """get_dataset with bare name raises KeyError when no label exists."""
        with pytest.raises(KeyError, match="Cannot resolve"):
            atmo_index.get_dataset(f"@{mock_atmo.did}/missing")

    def test_get_dataset_atmosphere_unavailable(self, tmp_path):
        """get_dataset raises ValueError when atmosphere is disabled."""
        index = Index(provider="sqlite", path=tmp_path / "index.db", atmosphere=None)
        with pytest.raises(ValueError, match="Atmosphere backend required"):
            index.get_dataset("@handle/dataset")


# ---------------------------------------------------------------------------
# Integration: write + read cycle via Index
# ---------------------------------------------------------------------------


class TestAtmosphereLabelRoundTrip:
    """End-to-end test of writing and reading via atmosphere labels."""

    def test_write_then_get_label(self, atmo_index, mock_atmo):
        """Data written via insert_dataset is retrievable via get_label."""
        backend = atmo_index._atmosphere
        ds = MagicMock()

        backend.insert_dataset(
            ds,
            name="roundtrip-ds",
            data_urls=["http://example.com/roundtrip.tar"],
            version="1.0.0",
            description="Round-trip test",
        )

        entry = atmo_index.get_label(f"@{mock_atmo.did}/roundtrip-ds", version="1.0.0")
        assert entry.name == "roundtrip-ds"

    def test_write_then_get_dataset(self, atmo_index, mock_atmo):
        """Data written via insert_dataset is retrievable via get_dataset."""
        backend = atmo_index._atmosphere
        ds = MagicMock()

        backend.insert_dataset(
            ds,
            name="get-ds-test",
            data_urls=["http://example.com/data.tar"],
        )

        entry = atmo_index.get_dataset(f"@{mock_atmo.did}/get-ds-test")
        assert entry.name == "get-ds-test"

    def test_multiple_datasets_distinct_labels(self, atmo_index, mock_atmo):
        """Multiple datasets each get their own label records."""
        backend = atmo_index._atmosphere
        ds = MagicMock()

        backend.insert_dataset(
            ds,
            name="ds-alpha",
            data_urls=["http://example.com/alpha.tar"],
        )
        backend.insert_dataset(
            ds,
            name="ds-beta",
            data_urls=["http://example.com/beta.tar"],
        )

        alpha = atmo_index.get_label(f"@{mock_atmo.did}/ds-alpha")
        beta = atmo_index.get_label(f"@{mock_atmo.did}/ds-beta")

        assert alpha.name == "ds-alpha"
        assert beta.name == "ds-beta"
        assert alpha.uri != beta.uri


# ---------------------------------------------------------------------------
# _resolve_indexed_path atmosphere integration
# ---------------------------------------------------------------------------


class TestResolveIndexedPathAtmosphere:
    """Tests for _resolve_indexed_path routing to atmosphere labels."""

    def test_resolve_indexed_path_uses_get_label(self, atmo_index, mock_atmo):
        """_resolve_indexed_path calls get_label for @handle/name paths."""
        from atdata._hf_api import _resolve_indexed_path

        backend = atmo_index._atmosphere
        ds = MagicMock()
        backend.insert_dataset(
            ds,
            name="indexed-ds",
            data_urls=["http://example.com/shard.tar"],
            schema_ref=f"at://{mock_atmo.did}/{LEXICON_NAMESPACE}.schema/s1",
        )

        source, schema_ref = _resolve_indexed_path(
            f"@{mock_atmo.did}/indexed-ds", atmo_index
        )
        assert schema_ref == (f"at://{mock_atmo.did}/{LEXICON_NAMESPACE}.schema/s1")

    def test_resolve_indexed_path_with_version(self, atmo_index, mock_atmo):
        """_resolve_indexed_path passes version through to get_label."""
        from atdata._hf_api import _resolve_indexed_path

        backend = atmo_index._atmosphere
        ds = MagicMock()

        backend.insert_dataset(
            ds,
            name="ver-ds",
            data_urls=["http://example.com/v1.tar"],
            schema_ref=f"at://{mock_atmo.did}/{LEXICON_NAMESPACE}.schema/s1",
            version="1.0.0",
        )
        backend.insert_dataset(
            ds,
            name="ver-ds",
            data_urls=["http://example.com/v2.tar"],
            schema_ref=f"at://{mock_atmo.did}/{LEXICON_NAMESPACE}.schema/s2",
            version="2.0.0",
        )

        source, schema_ref = _resolve_indexed_path(
            f"@{mock_atmo.did}/ver-ds@1.0.0", atmo_index
        )
        # Should resolve to v1's schema ref
        assert "s1" in schema_ref


# ---------------------------------------------------------------------------
# _AtmosphereBackend.resolve_label
# ---------------------------------------------------------------------------


class TestAtmosphereBackendResolveLabel:
    """Tests for _AtmosphereBackend.resolve_label method."""

    def test_resolve_label_method_exists(self):
        """_AtmosphereBackend has resolve_label method."""
        assert hasattr(_AtmosphereBackend, "resolve_label")

    def test_resolve_label_delegates_to_loader(self):
        """resolve_label delegates to _label_loader.resolve()."""
        from atdata.atmosphere.client import Atmosphere

        mock_sdk = MagicMock()
        mock_sdk.me = MagicMock(did="did:plc:test", handle="t.social")
        mock_sdk.login.return_value = MagicMock(did="did:plc:test", handle="t.social")
        atmo = Atmosphere(_client=mock_sdk)
        atmo._login("t.social", "pass")

        backend = _AtmosphereBackend(atmo)
        backend._ensure_loaders()

        # Mock the label loader's resolve method
        backend._label_loader = MagicMock()
        backend._label_loader.resolve.return_value = (
            "at://did:plc:test/ac.foundation.dataset.record/abc"
        )

        result = backend.resolve_label("did:plc:test", "my-ds", "1.0.0")
        backend._label_loader.resolve.assert_called_once_with(
            "did:plc:test", "my-ds", "1.0.0"
        )
        assert result == "at://did:plc:test/ac.foundation.dataset.record/abc"


# ---------------------------------------------------------------------------
# Real _AtmosphereBackend.insert_dataset label integration
# ---------------------------------------------------------------------------


class TestRealAtmosphereBackendLabelPublish:
    """Verify the real _AtmosphereBackend.insert_dataset publishes labels.

    Uses a real Atmosphere wrapping a mock SDK client, so the full
    publisher/loader chain is exercised.
    """

    def _make_backend(self):
        """Create _AtmosphereBackend with mock SDK, tracked records."""
        from atdata.atmosphere.client import Atmosphere

        mock_sdk = MagicMock()
        mock_sdk.me = MagicMock(did="did:plc:real123", handle="real.social")
        login_resp = MagicMock(did="did:plc:real123", handle="real.social")
        mock_sdk.login.return_value = login_resp

        records: dict[str, dict] = {}
        counter = [0]

        def create_record(data):
            counter[0] += 1
            collection = data["collection"]
            rkey = data.get("rkey") or f"rkey{counter[0]}"
            uri = f"at://did:plc:real123/{collection}/{rkey}"
            records[uri] = data["record"]
            resp = MagicMock()
            resp.uri = uri
            return resp

        def get_record(params):
            uri = f"at://{params['repo']}/{params['collection']}/{params['rkey']}"
            record = records.get(uri, {})
            resp = MagicMock()
            # Simulate the value having a to_dict method
            val = MagicMock()
            val.to_dict.return_value = dict(record)
            resp.value = val
            return resp

        mock_sdk.com.atproto.repo.create_record.side_effect = create_record
        mock_sdk.com.atproto.repo.get_record.side_effect = get_record

        atmo = Atmosphere(_client=mock_sdk)
        atmo._login("real.social", "pass")

        backend = _AtmosphereBackend(atmo)
        return backend, records

    def test_insert_with_urls_creates_label(self):
        """insert_dataset with data_urls creates both dataset and label."""
        backend, records = self._make_backend()
        ds = MagicMock()
        ds.sample_type = AtmoLabelSample

        backend.insert_dataset(
            ds,
            name="url-test",
            schema_ref="at://did:plc:real123/ac.foundation.dataset.schema/s1",
            data_urls=["http://example.com/data.tar"],
            description="URL test",
        )

        # Find label records
        label_uris = [uri for uri in records if f"{LEXICON_NAMESPACE}.label" in uri]
        assert len(label_uris) == 1
        label_record = records[label_uris[0]]
        assert label_record["name"] == "url-test"
        assert "datasetUri" in label_record

    def test_insert_with_blob_refs_creates_label(self):
        """insert_dataset with blob_refs creates a label record."""
        backend, records = self._make_backend()
        ds = MagicMock()
        ds.sample_type = AtmoLabelSample

        blob_refs = [
            {
                "$type": "blob",
                "ref": {"$link": "bafyabc123"},
                "mimeType": "application/x-tar",
                "size": 1024,
            }
        ]

        backend.insert_dataset(
            ds,
            name="blob-test",
            schema_ref="at://did:plc:real123/ac.foundation.dataset.schema/s1",
            data_urls=["http://example.com/data.tar"],
            blob_refs=blob_refs,
            description="Blob test",
        )

        label_uris = [uri for uri in records if f"{LEXICON_NAMESPACE}.label" in uri]
        assert len(label_uris) == 1
        assert records[label_uris[0]]["name"] == "blob-test"

    def test_insert_label_has_correct_dataset_uri(self):
        """Label's datasetUri points to the actual dataset record."""
        backend, records = self._make_backend()
        ds = MagicMock()
        ds.sample_type = AtmoLabelSample

        backend.insert_dataset(
            ds,
            name="uri-check",
            schema_ref="at://did:plc:real123/ac.foundation.dataset.schema/s1",
            data_urls=["http://example.com/data.tar"],
        )

        # Find dataset and label URIs
        dataset_uris = [uri for uri in records if f"{LEXICON_NAMESPACE}.record" in uri]
        label_uris = [uri for uri in records if f"{LEXICON_NAMESPACE}.label" in uri]

        assert len(dataset_uris) == 1
        assert len(label_uris) == 1

        label_record = records[label_uris[0]]
        assert label_record["datasetUri"] == dataset_uris[0]
