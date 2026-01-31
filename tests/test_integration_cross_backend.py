"""Integration tests for cross-backend interoperability.

Tests that abstract protocols work consistently across:
- Index and AtmosphereIndex (AbstractIndex protocol)
- LocalDatasetEntry and AtmosphereIndexEntry (IndexEntry protocol)
- S3DataStore (AbstractDataStore protocol)
"""

import pytest
from unittest.mock import Mock, MagicMock

from numpy.typing import NDArray

import atdata
from atdata.local import Index, LocalDatasetEntry
from atdata._protocols import IndexEntry
from atdata.atmosphere import (
    AtmosphereClient,
    AtmosphereIndex,
    AtmosphereIndexEntry,
)
from atdata.atmosphere._types import LEXICON_NAMESPACE


##
# Test sample types


@atdata.packable
class CrossBackendSample:
    """Sample for cross-backend tests."""

    name: str
    value: int


@atdata.packable
class CrossBackendArraySample:
    """Sample with NDArray for cross-backend tests."""

    label: str
    data: NDArray


##
# Fixtures


@pytest.fixture
def mock_atproto_client():
    """Create a mock atproto SDK client."""
    mock = Mock()
    mock.me = MagicMock()
    mock.me.did = "did:plc:crossbackend123"
    mock.me.handle = "crossbackend.test.social"

    mock_profile = Mock()
    mock_profile.did = "did:plc:crossbackend123"
    mock_profile.handle = "crossbackend.test.social"
    mock.login.return_value = mock_profile
    mock.export_session_string.return_value = "test-session-export"

    return mock


@pytest.fixture
def authenticated_atmosphere_client(mock_atproto_client):
    """Create an authenticated AtmosphereClient."""
    client = AtmosphereClient(_client=mock_atproto_client)
    client.login("crossbackend.test.social", "test-password")
    return client


@pytest.fixture
def local_index(clean_redis):
    """Create a Index backed by Redis."""
    return Index(redis=clean_redis)


@pytest.fixture
def atmosphere_index(authenticated_atmosphere_client):
    """Create an AtmosphereIndex."""
    return AtmosphereIndex(authenticated_atmosphere_client)


##
# IndexEntry Protocol Tests


class TestIndexEntryProtocol:
    """Tests that LocalDatasetEntry and AtmosphereIndexEntry are interchangeable."""

    def test_local_entry_satisfies_protocol(self):
        """LocalDatasetEntry should satisfy IndexEntry protocol."""
        entry = LocalDatasetEntry(
            name="test-dataset",
            schema_ref="local://schemas/TestSample@1.0.0",
            data_urls=["s3://bucket/test.tar"],
        )

        assert isinstance(entry, IndexEntry)
        assert entry.name == "test-dataset"
        assert entry.schema_ref == "local://schemas/TestSample@1.0.0"
        assert entry.data_urls == ["s3://bucket/test.tar"]
        assert entry.metadata is None

    def test_atmosphere_entry_satisfies_protocol(self):
        """AtmosphereIndexEntry should satisfy IndexEntry protocol."""
        record = {
            "name": "atmo-dataset",
            "schemaRef": "at://did:plc:test/ac.foundation.dataset.sampleSchema/abc",
            "storage": {
                "$type": f"{LEXICON_NAMESPACE}.storageExternal",
                "urls": ["s3://bucket/atmo.tar"],
            },
        }
        entry = AtmosphereIndexEntry("at://did:plc:test/dataset/xyz", record)

        assert isinstance(entry, IndexEntry)
        assert entry.name == "atmo-dataset"
        assert (
            entry.schema_ref
            == "at://did:plc:test/ac.foundation.dataset.sampleSchema/abc"
        )
        assert entry.data_urls == ["s3://bucket/atmo.tar"]
        assert entry.metadata is None

    def test_entries_work_with_common_function(self):
        """Both entry types should work with functions accepting IndexEntry."""

        def process_entry(entry: IndexEntry) -> dict:
            return {
                "name": entry.name,
                "schema": entry.schema_ref,
                "url_count": len(entry.data_urls),
            }

        local_entry = LocalDatasetEntry(
            name="local-ds",
            schema_ref="local://schemas/Test@1.0.0",
            data_urls=["s3://bucket/local.tar"],
        )

        atmo_record = {
            "name": "atmo-ds",
            "schemaRef": "at://did:plc:test/schema/abc",
            "storage": {
                "$type": f"{LEXICON_NAMESPACE}.storageExternal",
                "urls": ["s3://bucket/atmo.tar", "s3://bucket/atmo2.tar"],
            },
        }
        atmo_entry = AtmosphereIndexEntry("at://test", atmo_record)

        local_result = process_entry(local_entry)
        atmo_result = process_entry(atmo_entry)

        assert local_result["name"] == "local-ds"
        assert local_result["url_count"] == 1

        assert atmo_result["name"] == "atmo-ds"
        assert atmo_result["url_count"] == 2

    def test_entries_with_metadata(self):
        """Both entry types should handle metadata consistently."""
        import msgpack

        # Local entry with metadata
        local_entry = LocalDatasetEntry(
            name="local-meta",
            schema_ref="local://schemas/Test@1.0.0",
            data_urls=["s3://bucket/local.tar"],
            metadata={"version": "1.0", "samples": 100},
        )

        # Atmosphere entry with metadata
        atmo_record = {
            "name": "atmo-meta",
            "schemaRef": "at://did:plc:test/schema/abc",
            "storage": {
                "$type": f"{LEXICON_NAMESPACE}.storageExternal",
                "urls": ["s3://bucket/atmo.tar"],
            },
            "metadata": msgpack.packb({"version": "1.0", "samples": 200}),
        }
        atmo_entry = AtmosphereIndexEntry("at://test", atmo_record)

        assert local_entry.metadata["version"] == "1.0"
        assert local_entry.metadata["samples"] == 100

        assert atmo_entry.metadata["version"] == "1.0"
        assert atmo_entry.metadata["samples"] == 200

    def test_entries_with_multiple_urls(self):
        """Both entry types should handle multiple data URLs."""
        urls = [
            "s3://bucket/shard-000000.tar",
            "s3://bucket/shard-000001.tar",
            "s3://bucket/shard-000002.tar",
        ]

        local_entry = LocalDatasetEntry(
            name="multi-shard-local",
            schema_ref="local://schemas/Test@1.0.0",
            data_urls=urls,
        )

        atmo_record = {
            "name": "multi-shard-atmo",
            "schemaRef": "at://did:plc:test/schema/abc",
            "storage": {
                "$type": f"{LEXICON_NAMESPACE}.storageExternal",
                "urls": urls,
            },
        }
        atmo_entry = AtmosphereIndexEntry("at://test", atmo_record)

        assert local_entry.data_urls == urls
        assert atmo_entry.data_urls == urls


##
# AbstractIndex Protocol Tests


class TestAbstractIndexProtocol:
    """Tests that Index and AtmosphereIndex share common behavior."""

    def test_local_index_list_datasets_yields_entries(self, local_index, clean_redis):
        """Index.list_datasets should yield IndexEntry objects."""
        # Insert an entry directly via Redis for testing
        entry = LocalDatasetEntry(
            name="test-list",
            schema_ref="local://schemas/Test@1.0.0",
            data_urls=["s3://bucket/test.tar"],
        )
        entry.write_to(clean_redis)

        entries = list(local_index.list_datasets())

        assert len(entries) >= 1
        found = [e for e in entries if e.name == "test-list"]
        assert len(found) == 1
        assert isinstance(found[0], IndexEntry)

    def test_atmosphere_index_list_datasets_yields_entries(
        self, atmosphere_index, mock_atproto_client
    ):
        """AtmosphereIndex.list_datasets should yield IndexEntry objects."""
        mock_record = Mock()
        mock_record.uri = f"at://did:plc:test/{LEXICON_NAMESPACE}.dataset/d1"
        mock_record.value = {
            "name": "atmo-list",
            "schemaRef": "at://schema",
            "storage": {
                "$type": f"{LEXICON_NAMESPACE}.storageExternal",
                "urls": ["s3://data.tar"],
            },
        }

        mock_response = Mock()
        mock_response.records = [mock_record]
        mock_response.cursor = None
        mock_atproto_client.com.atproto.repo.list_records.return_value = mock_response

        entries = list(atmosphere_index.list_datasets())

        assert len(entries) == 1
        assert isinstance(entries[0], IndexEntry)
        assert entries[0].name == "atmo-list"

    def test_local_index_publish_schema(self, local_index):
        """Index.publish_schema should return schema reference."""
        schema_ref = local_index.publish_schema(CrossBackendSample, version="1.0.0")

        assert schema_ref is not None
        assert "CrossBackendSample" in schema_ref
        assert "1.0.0" in schema_ref

    def test_atmosphere_index_publish_schema(
        self, atmosphere_index, mock_atproto_client
    ):
        """AtmosphereIndex.publish_schema should return AT URI."""
        mock_response = Mock()
        mock_response.uri = f"at://did:plc:test/{LEXICON_NAMESPACE}.sampleSchema/abc"
        mock_atproto_client.com.atproto.repo.create_record.return_value = mock_response

        schema_ref = atmosphere_index.publish_schema(
            CrossBackendSample, version="1.0.0"
        )

        assert schema_ref is not None
        assert "at://" in str(schema_ref)

    def test_local_index_get_schema(self, local_index):
        """Index should retrieve published schemas."""
        schema_ref = local_index.publish_schema(CrossBackendSample, version="2.0.0")
        schema = local_index.get_schema(schema_ref)

        assert schema["name"] == "CrossBackendSample"
        assert schema["version"] == "2.0.0"
        assert len(schema["fields"]) == 2

    def test_atmosphere_index_get_schema(self, atmosphere_index, mock_atproto_client):
        """AtmosphereIndex should retrieve schemas."""
        mock_response = Mock()
        mock_response.value = {
            "$type": f"{LEXICON_NAMESPACE}.sampleSchema",
            "name": "RetrievedSchema",
            "version": "1.0.0",
            "fields": [
                {
                    "name": "field1",
                    "fieldType": {
                        "$type": f"{LEXICON_NAMESPACE}.schemaType#primitive",
                        "primitive": "str",
                    },
                    "optional": False,
                }
            ],
        }
        mock_atproto_client.com.atproto.repo.get_record.return_value = mock_response

        schema = atmosphere_index.get_schema("at://did:plc:test/schema/key")

        assert schema["name"] == "RetrievedSchema"
        assert schema["version"] == "1.0.0"


class TestSchemaPortability:
    """Tests that schemas can be used across backends."""

    def test_schema_field_structure_matches(self, local_index):
        """Schema structure should be consistent regardless of backend."""
        schema_ref = local_index.publish_schema(CrossBackendSample, version="1.0.0")
        schema = local_index.get_schema(schema_ref)

        # Verify schema has expected structure
        assert "name" in schema
        assert "version" in schema
        assert "fields" in schema

        field_names = {f["name"] for f in schema["fields"]}
        assert "name" in field_names
        assert "value" in field_names

    def test_ndarray_schema_field_structure(self, local_index):
        """NDArray fields should be represented consistently."""
        schema_ref = local_index.publish_schema(
            CrossBackendArraySample, version="1.0.0"
        )
        schema = local_index.get_schema(schema_ref)

        field_names = {f["name"] for f in schema["fields"]}
        assert "label" in field_names
        assert "data" in field_names

        # Find the data field and verify it's marked as ndarray type
        data_field = next(f for f in schema["fields"] if f["name"] == "data")
        field_type = data_field["fieldType"]
        # Field type should indicate it's an ndarray
        assert (
            "ndarray" in field_type.get("$type", "").lower()
            or field_type.get("primitive") == "ndarray"
        )


class TestCrossBackendSchemaResolution:
    """Tests for schema resolution across different backends."""

    def test_local_schema_ref_format(self, local_index):
        """Local schema refs should use atdata://local/sampleSchema/ URI scheme."""
        schema_ref = local_index.publish_schema(CrossBackendSample, version="1.0.0")

        assert schema_ref.startswith("atdata://local/sampleSchema/")
        assert "CrossBackendSample" in schema_ref

    def test_atmosphere_schema_ref_format(self, atmosphere_index, mock_atproto_client):
        """Atmosphere schema refs should use at:// URI scheme."""
        mock_response = Mock()
        mock_response.uri = f"at://did:plc:test/{LEXICON_NAMESPACE}.sampleSchema/abc"
        mock_atproto_client.com.atproto.repo.create_record.return_value = mock_response

        schema_ref = atmosphere_index.publish_schema(
            CrossBackendSample, version="1.0.0"
        )

        assert "at://" in str(schema_ref)


class TestIndexEntryCreation:
    """Tests for creating index entries via different backends."""

    def test_local_entry_has_cid(self):
        """LocalDatasetEntry should generate a CID."""
        entry = LocalDatasetEntry(
            name="cid-test",
            schema_ref="local://schemas/Test@1.0.0",
            data_urls=["s3://bucket/test.tar"],
        )

        assert entry.cid is not None
        assert len(entry.cid) > 0

    def test_atmosphere_entry_has_uri(self):
        """AtmosphereIndexEntry should have a URI."""
        record = {
            "name": "uri-test",
            "schemaRef": "at://schema",
            "storage": {
                "$type": f"{LEXICON_NAMESPACE}.storageExternal",
                "urls": ["s3://test.tar"],
            },
        }
        entry = AtmosphereIndexEntry("at://did:plc:test/dataset/xyz", record)

        assert entry.uri == "at://did:plc:test/dataset/xyz"

    def test_same_content_same_local_cid(self):
        """Same content should produce same CID in local entries."""
        entry1 = LocalDatasetEntry(
            name="cid-test-1",
            schema_ref="local://schemas/Test@1.0.0",
            data_urls=["s3://bucket/same.tar"],
        )
        entry2 = LocalDatasetEntry(
            name="cid-test-2",
            schema_ref="local://schemas/Test@1.0.0",
            data_urls=["s3://bucket/same.tar"],
        )

        # Different names but same content should produce same CID
        assert entry1.cid == entry2.cid

    def test_different_content_different_local_cid(self):
        """Different content should produce different CID in local entries."""
        entry1 = LocalDatasetEntry(
            name="cid-diff-1",
            schema_ref="local://schemas/Test@1.0.0",
            data_urls=["s3://bucket/file1.tar"],
        )
        entry2 = LocalDatasetEntry(
            name="cid-diff-2",
            schema_ref="local://schemas/Test@1.0.0",
            data_urls=["s3://bucket/file2.tar"],
        )

        assert entry1.cid != entry2.cid


class TestListingConsistency:
    """Tests that listing operations behave consistently."""

    def test_empty_local_index_lists_no_datasets(self, clean_redis):
        """Empty Index should list no datasets."""
        index = Index(redis=clean_redis)
        entries = list(index.list_datasets())

        # Should be empty or contain only pre-existing entries
        # (clean_redis fixture should clear it)
        assert len(entries) == 0

    def test_empty_atmosphere_index_lists_no_datasets(
        self, atmosphere_index, mock_atproto_client
    ):
        """Empty AtmosphereIndex should list no datasets."""
        mock_response = Mock()
        mock_response.records = []
        mock_response.cursor = None
        mock_atproto_client.com.atproto.repo.list_records.return_value = mock_response

        entries = list(atmosphere_index.list_datasets())

        assert len(entries) == 0


class TestGenericIndexFunction:
    """Tests for functions that work with any AbstractIndex implementation."""

    def count_datasets(self, index) -> int:
        """Count datasets in an index (works with any AbstractIndex)."""
        return sum(1 for _ in index.list_datasets())

    def get_all_names(self, index) -> list[str]:
        """Get all dataset names from an index."""
        return [entry.name for entry in index.list_datasets()]

    def test_count_works_with_local(self, local_index, clean_redis):
        """Dataset count function should work with Index."""
        # Insert some entries
        for i in range(3):
            entry = LocalDatasetEntry(
                name=f"count-test-{i}",
                schema_ref="local://schemas/Test@1.0.0",
                data_urls=[f"s3://bucket/test-{i}.tar"],
            )
            entry.write_to(clean_redis)

        count = self.count_datasets(local_index)
        assert count >= 3

    def test_count_works_with_atmosphere(self, atmosphere_index, mock_atproto_client):
        """Dataset count function should work with AtmosphereIndex."""
        mock_records = []
        for i in range(5):
            record = Mock()
            record.uri = f"at://did:plc:test/{LEXICON_NAMESPACE}.dataset/d{i}"
            record.value = {
                "name": f"count-atmo-{i}",
                "schemaRef": "at://schema",
                "storage": {
                    "$type": f"{LEXICON_NAMESPACE}.storageExternal",
                    "urls": [f"s3://data-{i}.tar"],
                },
            }
            mock_records.append(record)

        mock_response = Mock()
        mock_response.records = mock_records
        mock_response.cursor = None
        mock_atproto_client.com.atproto.repo.list_records.return_value = mock_response

        count = self.count_datasets(atmosphere_index)
        assert count == 5

    def test_get_names_works_with_local(self, local_index, clean_redis):
        """Name retrieval function should work with Index."""
        names = ["alpha", "beta", "gamma"]
        for name in names:
            entry = LocalDatasetEntry(
                name=name,
                schema_ref="local://schemas/Test@1.0.0",
                data_urls=[f"s3://bucket/{name}.tar"],
            )
            entry.write_to(clean_redis)

        retrieved_names = self.get_all_names(local_index)

        for name in names:
            assert name in retrieved_names

    def test_get_names_works_with_atmosphere(
        self, atmosphere_index, mock_atproto_client
    ):
        """Name retrieval function should work with AtmosphereIndex."""
        names = ["delta", "epsilon", "zeta"]
        mock_records = []
        for name in names:
            record = Mock()
            record.uri = f"at://did:plc:test/{LEXICON_NAMESPACE}.dataset/{name}"
            record.value = {
                "name": name,
                "schemaRef": "at://schema",
                "storage": {
                    "$type": f"{LEXICON_NAMESPACE}.storageExternal",
                    "urls": [f"s3://{name}.tar"],
                },
            }
            mock_records.append(record)

        mock_response = Mock()
        mock_response.records = mock_records
        mock_response.cursor = None
        mock_atproto_client.com.atproto.repo.list_records.return_value = mock_response

        retrieved_names = self.get_all_names(atmosphere_index)

        assert retrieved_names == names
