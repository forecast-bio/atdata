"""Tests for the dataset label system.

Covers:
- LexLabelRecord serialization round-trip
- Provider label CRUD (SQLite)
- Index label flow: write_samples creates label, get_dataset resolves via label
- Multiple versions: same name, different versions, resolve latest
- Atmosphere: LabelPublisher, LabelLoader
- Backward compat: existing entries without labels still resolvable
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone
from pathlib import Path

import atdata
from atdata.atmosphere._lexicon_types import LexLabelRecord
from atdata.atmosphere.labels import LabelPublisher, LabelLoader
from atdata.providers._sqlite import SqliteProvider
from atdata.index._index import Index
from atdata.index._entry import LocalDatasetEntry
from atdata.testing import MockAtmosphere


# ---------------------------------------------------------------------------
# LexLabelRecord serialization
# ---------------------------------------------------------------------------


class TestLexLabelRecord:
    """Tests for LexLabelRecord to_record / from_record round-trip."""

    def test_round_trip_full(self):
        """Full round-trip with all fields populated."""
        ts = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        record = LexLabelRecord(
            name="mnist",
            dataset_uri="at://did:plc:abc/ac.foundation.dataset.record/xyz",
            created_at=ts,
            version="1.0.0",
            description="Initial release of MNIST",
        )

        d = record.to_record()
        assert d["$type"] == "ac.foundation.dataset.label"
        assert d["name"] == "mnist"
        assert d["datasetUri"] == "at://did:plc:abc/ac.foundation.dataset.record/xyz"
        assert d["version"] == "1.0.0"
        assert d["description"] == "Initial release of MNIST"
        assert d["createdAt"] == ts.isoformat()

        restored = LexLabelRecord.from_record(d)
        assert restored.name == record.name
        assert restored.dataset_uri == record.dataset_uri
        assert restored.version == record.version
        assert restored.description == record.description
        assert restored.created_at == record.created_at

    def test_round_trip_minimal(self):
        """Round-trip with only required fields."""
        record = LexLabelRecord(
            name="cifar10",
            dataset_uri="at://did:plc:abc/ac.foundation.dataset.record/123",
        )

        d = record.to_record()
        assert "version" not in d
        assert "description" not in d

        restored = LexLabelRecord.from_record(d)
        assert restored.name == "cifar10"
        assert restored.version is None
        assert restored.description is None


# ---------------------------------------------------------------------------
# SQLite provider label CRUD
# ---------------------------------------------------------------------------


class TestSqliteProviderLabels:
    """Tests for label operations in SqliteProvider."""

    def test_store_and_get_unversioned(self, tmp_path: Path):
        """Store and retrieve a label without version."""
        provider = SqliteProvider(path=tmp_path / "test.db")
        provider.store_label(name="mnist", cid="cid-abc123")

        cid, version = provider.get_label("mnist")
        assert cid == "cid-abc123"
        assert version is None

        provider.close()

    def test_store_and_get_versioned(self, tmp_path: Path):
        """Store and retrieve a label with a specific version."""
        provider = SqliteProvider(path=tmp_path / "test.db")
        provider.store_label(name="mnist", cid="cid-v1", version="1.0.0")
        provider.store_label(name="mnist", cid="cid-v2", version="2.0.0")

        cid, version = provider.get_label("mnist", version="1.0.0")
        assert cid == "cid-v1"
        assert version == "1.0.0"

        cid, version = provider.get_label("mnist", version="2.0.0")
        assert cid == "cid-v2"
        assert version == "2.0.0"

        provider.close()

    def test_get_latest_returns_most_recent(self, tmp_path: Path):
        """When no version specified, get_label returns the latest by created_at."""
        provider = SqliteProvider(path=tmp_path / "test.db")
        provider.store_label(name="dataset", cid="cid-old", version="1.0.0")
        provider.store_label(name="dataset", cid="cid-new", version="2.0.0")

        cid, version = provider.get_label("dataset")
        # Latest by created_at — should be the second one stored
        assert cid == "cid-new"

        provider.close()

    def test_get_nonexistent_raises(self, tmp_path: Path):
        """KeyError when no label matches."""
        provider = SqliteProvider(path=tmp_path / "test.db")
        with pytest.raises(KeyError, match="No label"):
            provider.get_label("nonexistent")

        provider.close()

    def test_get_wrong_version_raises(self, tmp_path: Path):
        """KeyError when version doesn't match."""
        provider = SqliteProvider(path=tmp_path / "test.db")
        provider.store_label(name="ds", cid="cid-1", version="1.0.0")
        with pytest.raises(KeyError, match="No label"):
            provider.get_label("ds", version="9.9.9")

        provider.close()

    def test_iter_labels(self, tmp_path: Path):
        """iter_labels yields all stored labels."""
        provider = SqliteProvider(path=tmp_path / "test.db")
        provider.store_label(name="a", cid="cid-a")
        provider.store_label(name="b", cid="cid-b", version="1.0.0")

        labels = list(provider.iter_labels())
        assert len(labels) == 2
        names = {l[0] for l in labels}
        assert names == {"a", "b"}

        provider.close()

    def test_upsert_replaces(self, tmp_path: Path):
        """Storing a label with same name+version replaces the CID."""
        provider = SqliteProvider(path=tmp_path / "test.db")
        provider.store_label(name="ds", cid="old-cid", version="1.0.0")
        provider.store_label(name="ds", cid="new-cid", version="1.0.0")

        cid, _ = provider.get_label("ds", version="1.0.0")
        assert cid == "new-cid"

        provider.close()

    def test_store_with_description(self, tmp_path: Path):
        """Labels can have descriptions."""
        provider = SqliteProvider(path=tmp_path / "test.db")
        provider.store_label(
            name="ds", cid="cid-1", version="1.0.0",
            description="First version",
        )

        # Description is stored but not returned by get_label
        cid, version = provider.get_label("ds", version="1.0.0")
        assert cid == "cid-1"

        provider.close()


# ---------------------------------------------------------------------------
# Index label integration
# ---------------------------------------------------------------------------


@atdata.packable
class LabelTestSample:
    text: str
    value: int


class TestIndexLabels:
    """Tests for label integration in the Index class."""

    def test_write_samples_creates_label(self, tmp_path: Path):
        """write_samples automatically creates a label for the dataset."""
        index = Index(provider="sqlite", path=tmp_path / "index.db")
        samples = [LabelTestSample(text="hello", value=i) for i in range(5)]
        entry = index.write_samples(samples, name="my-dataset")

        # Label should exist
        labels = index.list_labels()
        assert len(labels) == 1
        name, cid, version = labels[0]
        assert name == "my-dataset"
        assert cid == entry.cid

    def test_get_dataset_resolves_via_label(self, tmp_path: Path):
        """get_dataset resolves through label to the entry."""
        index = Index(provider="sqlite", path=tmp_path / "index.db")
        samples = [LabelTestSample(text="test", value=1)]
        original = index.write_samples(samples, name="labeled-ds")

        resolved = index.get_dataset("labeled-ds")
        assert resolved.cid == original.cid
        assert resolved.data_urls == original.data_urls

    def test_get_label_specific_version(self, tmp_path: Path):
        """get_label with version returns the matching entry."""
        index = Index(provider="sqlite", path=tmp_path / "index.db")
        samples = [LabelTestSample(text="v1", value=1)]
        entry = index.write_samples(samples, name="versioned")

        # Create a versioned label manually
        index.label("versioned", entry.cid, version="1.0.0")

        resolved = index.get_label("versioned", version="1.0.0")
        assert resolved.cid == entry.cid

    def test_manual_label_creation(self, tmp_path: Path):
        """index.label() creates a label manually."""
        index = Index(provider="sqlite", path=tmp_path / "index.db")
        samples = [LabelTestSample(text="test", value=1)]
        entry = index.write_samples(samples, name="original")

        index.label("alias", entry.cid)
        resolved = index.get_label("alias")
        assert resolved.cid == entry.cid

    def test_multiple_versions(self, tmp_path: Path):
        """Multiple versioned labels for the same name."""
        index = Index(provider="sqlite", path=tmp_path / "index.db")

        samples_v1 = [LabelTestSample(text="v1", value=1)]
        entry_v1 = index.write_samples(samples_v1, name="multi-ver")
        index.label("multi-ver", entry_v1.cid, version="1.0.0")

        samples_v2 = [LabelTestSample(text="v2", value=2)]
        entry_v2 = index.write_samples(samples_v2, name="multi-ver")
        index.label("multi-ver", entry_v2.cid, version="2.0.0")

        v1 = index.get_label("multi-ver", version="1.0.0")
        assert v1.cid == entry_v1.cid

        v2 = index.get_label("multi-ver", version="2.0.0")
        assert v2.cid == entry_v2.cid

    def test_backward_compat_no_label(self, tmp_path: Path):
        """Entries created before labels are still resolvable by name."""
        provider = SqliteProvider(path=tmp_path / "index.db")
        # Manually insert an entry without a label
        entry = LocalDatasetEntry(
            name="legacy-entry",
            schema_ref="atdata://local/schema/Test@1.0.0",
            data_urls=["file:///data/test.tar"],
        )
        provider.store_entry(entry)

        index = Index(provider="sqlite", path=tmp_path / "index.db")
        resolved = index.get_dataset("legacy-entry")
        assert resolved.name == "legacy-entry"
        assert resolved.data_urls == ["file:///data/test.tar"]

    def test_list_labels(self, tmp_path: Path):
        """list_labels returns all labels."""
        index = Index(provider="sqlite", path=tmp_path / "index.db")
        samples = [LabelTestSample(text="test", value=1)]
        entry = index.write_samples(samples, name="ds-a")

        samples2 = [LabelTestSample(text="test2", value=2)]
        entry2 = index.write_samples(samples2, name="ds-b")

        labels = index.list_labels()
        assert len(labels) == 2
        names = {l[0] for l in labels}
        assert names == {"ds-a", "ds-b"}


# ---------------------------------------------------------------------------
# Atmosphere labels
# ---------------------------------------------------------------------------


class TestLabelPublisher:
    """Tests for LabelPublisher with MockAtmosphere."""

    def test_publish_creates_record(self):
        """publish() creates a label record in the correct collection."""
        mock = MockAtmosphere()
        mock.login("test.user", "password")
        publisher = LabelPublisher(mock)

        uri = publisher.publish(
            name="mnist",
            dataset_uri="at://did:plc:mock/ac.foundation.dataset.record/abc",
            version="1.0.0",
            description="Test label",
        )

        assert isinstance(uri, str)
        assert "ac.foundation.dataset.label" in uri

        record = mock.get_record(uri)
        assert record["$type"] == "ac.foundation.dataset.label"
        assert record["name"] == "mnist"
        assert record["datasetUri"] == "at://did:plc:mock/ac.foundation.dataset.record/abc"
        assert record["version"] == "1.0.0"
        assert record["description"] == "Test label"

    def test_publish_minimal(self):
        """publish() works with only required fields."""
        mock = MockAtmosphere()
        mock.login("test.user", "password")
        publisher = LabelPublisher(mock)

        uri = publisher.publish(
            name="cifar10",
            dataset_uri="at://did:plc:mock/ac.foundation.dataset.record/xyz",
        )

        record = mock.get_record(uri)
        assert record["name"] == "cifar10"
        assert "version" not in record
        assert "description" not in record


class TestLabelLoader:
    """Tests for LabelLoader with MockAtmosphere."""

    def test_get_label_record(self):
        """get() fetches and validates a label record."""
        mock = MockAtmosphere()
        mock.login("test.user", "password")

        # Create a label record directly
        publisher = LabelPublisher(mock)
        uri = publisher.publish(
            name="test-label",
            dataset_uri="at://did:plc:mock/ac.foundation.dataset.record/abc",
        )

        loader = LabelLoader(mock)
        record = loader.get(uri)
        assert record["name"] == "test-label"

    def test_get_typed(self):
        """get_typed() returns LexLabelRecord instance."""
        mock = MockAtmosphere()
        mock.login("test.user", "password")

        publisher = LabelPublisher(mock)
        uri = publisher.publish(
            name="typed-label",
            dataset_uri="at://did:plc:mock/ac.foundation.dataset.record/abc",
            version="2.0.0",
        )

        loader = LabelLoader(mock)
        label = loader.get_typed(uri)
        assert isinstance(label, LexLabelRecord)
        assert label.name == "typed-label"
        assert label.version == "2.0.0"

    def test_list_labels(self):
        """list_all() returns label records."""
        mock = MockAtmosphere()
        mock.login("test.user", "password")

        publisher = LabelPublisher(mock)
        publisher.publish(name="a", dataset_uri="at://did:plc:mock/ac.foundation.dataset.record/1")
        publisher.publish(name="b", dataset_uri="at://did:plc:mock/ac.foundation.dataset.record/2")

        loader = LabelLoader(mock)
        labels = loader.list_all()
        assert len(labels) == 2


class TestMockAtmosphereLabels:
    """Tests for MockAtmosphere list_labels."""

    def test_list_labels_filters_by_collection(self):
        """list_labels only returns label records, not other types."""
        mock = MockAtmosphere()
        mock.login("test.user", "password")

        # Create a label and a non-label record
        mock.create_record(
            "ac.foundation.dataset.label",
            {"$type": "ac.foundation.dataset.label", "name": "test"},
        )
        mock.create_record(
            "ac.foundation.dataset.record",
            {"$type": "ac.foundation.dataset.record", "name": "dataset"},
        )

        labels = mock.list_labels()
        assert len(labels) == 1
        assert labels[0]["name"] == "test"
