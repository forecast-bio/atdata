"""Integration tests for Atmosphere ATProto workflows.

Tests end-to-end Atmosphere operations including:
- Full publish workflow (login → publish schema → publish dataset → query)
- Session persistence and restoration
- Record discovery and querying
- AtmosphereIndex compliance with AbstractIndex
"""

import pytest
from unittest.mock import Mock, MagicMock

from numpy.typing import NDArray
import msgpack

import atdata
from atdata.atmosphere import (
    AtmosphereClient,
    AtmosphereIndex,
    AtmosphereIndexEntry,
    SchemaPublisher,
    SchemaLoader,
    DatasetPublisher,
    AtUri,
)
from atdata.atmosphere._types import LEXICON_NAMESPACE


##
# Test sample types


@atdata.packable
class AtmoSample:
    """Sample for atmosphere tests."""

    name: str
    value: int


@atdata.packable
class AtmoNDArraySample:
    """Sample with NDArray for atmosphere tests."""

    label: str
    data: NDArray


##
# Fixtures


@pytest.fixture
def mock_atproto_client():
    """Create a mock atproto SDK client."""
    mock = Mock()
    mock.me = MagicMock()
    mock.me.did = "did:plc:integration123"
    mock.me.handle = "integration.test.social"

    mock_profile = Mock()
    mock_profile.did = "did:plc:integration123"
    mock_profile.handle = "integration.test.social"
    mock.login.return_value = mock_profile
    mock.export_session_string.return_value = "test-session-export"

    return mock


@pytest.fixture
def authenticated_client(mock_atproto_client):
    """Create an authenticated AtmosphereClient."""
    client = AtmosphereClient(_client=mock_atproto_client)
    client.login("integration.test.social", "test-password")
    return client


##
# Full Workflow Tests


class TestFullPublishWorkflow:
    """End-to-end tests for publish workflow."""

    def test_login_publish_schema_publish_dataset(self, mock_atproto_client):
        """Full workflow: login → publish schema → publish dataset."""
        # Setup mock responses
        schema_response = Mock()
        schema_response.uri = (
            f"at://did:plc:integration123/{LEXICON_NAMESPACE}.sampleSchema/schema123"
        )

        dataset_response = Mock()
        dataset_response.uri = (
            f"at://did:plc:integration123/{LEXICON_NAMESPACE}.dataset/dataset456"
        )

        mock_atproto_client.com.atproto.repo.create_record.side_effect = [
            schema_response,
            dataset_response,
        ]

        # Execute workflow
        client = AtmosphereClient(_client=mock_atproto_client)
        client.login("test.social", "password")

        # Publish schema
        schema_pub = SchemaPublisher(client)
        schema_uri = schema_pub.publish(AtmoSample, version="1.0.0")

        assert isinstance(schema_uri, AtUri)
        assert schema_uri.collection == f"{LEXICON_NAMESPACE}.sampleSchema"

        # Publish dataset using correct API
        dataset_pub = DatasetPublisher(client)
        dataset_uri = dataset_pub.publish_with_urls(
            urls=["s3://bucket/data.tar"],
            schema_uri=str(schema_uri),
            name="test-dataset",
        )

        assert isinstance(dataset_uri, AtUri)
        assert dataset_uri.collection == f"{LEXICON_NAMESPACE}.dataset"


class TestSessionPersistence:
    """Tests for session export and restoration."""

    def test_export_session_returns_string(self, authenticated_client):
        """Authenticated client should export session."""
        session = authenticated_client.export_session()
        assert isinstance(session, str)
        assert len(session) > 0

    def test_login_with_session_restores_auth(self, mock_atproto_client):
        """Login with session string should restore authentication."""
        client = AtmosphereClient(_client=mock_atproto_client)

        assert not client.is_authenticated

        client.login_with_session("saved-session-string")

        assert client.is_authenticated
        mock_atproto_client.login.assert_called_with(
            session_string="saved-session-string"
        )

    def test_session_round_trip(self, mock_atproto_client):
        """Export then import session should maintain auth."""
        # First client - login and export
        client1 = AtmosphereClient(_client=mock_atproto_client)
        client1.login("user@test.social", "password")
        session = client1.export_session()

        # Second client - restore from session
        mock_atproto_client2 = Mock()
        mock_atproto_client2.me = mock_atproto_client.me
        mock_atproto_client2.login.return_value = mock_atproto_client.login.return_value
        mock_atproto_client2.export_session_string.return_value = session

        client2 = AtmosphereClient(_client=mock_atproto_client2)
        client2.login_with_session(session)

        assert client2.is_authenticated
        assert client2.did == client1.did


class TestRecordDiscovery:
    """Tests for finding and querying records."""

    def test_list_schemas_returns_all(self, authenticated_client, mock_atproto_client):
        """list_schemas should return all schema records."""
        mock_record1 = Mock()
        mock_record1.uri = f"at://did:plc:test/{LEXICON_NAMESPACE}.sampleSchema/s1"
        mock_record1.value = {"name": "Schema1", "version": "1.0.0", "fields": []}

        mock_record2 = Mock()
        mock_record2.uri = f"at://did:plc:test/{LEXICON_NAMESPACE}.sampleSchema/s2"
        mock_record2.value = {"name": "Schema2", "version": "1.0.0", "fields": []}

        mock_response = Mock()
        mock_response.records = [mock_record1, mock_record2]
        mock_response.cursor = None
        mock_atproto_client.com.atproto.repo.list_records.return_value = mock_response

        loader = SchemaLoader(authenticated_client)
        schemas = loader.list_all()

        assert len(schemas) == 2

    def test_get_schema_by_uri(self, authenticated_client, mock_atproto_client):
        """get should retrieve schema by URI."""
        mock_response = Mock()
        mock_response.value = {
            "$type": f"{LEXICON_NAMESPACE}.sampleSchema",
            "name": "FoundSchema",
            "version": "2.0.0",
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

        loader = SchemaLoader(authenticated_client)
        schema = loader.get("at://did:plc:test/schema/key")

        assert schema["name"] == "FoundSchema"
        assert schema["version"] == "2.0.0"


class TestAtmosphereIndex:
    """Tests for AtmosphereIndex AbstractIndex compliance."""

    def test_index_list_datasets_yields_entries(
        self, authenticated_client, mock_atproto_client
    ):
        """list_datasets should yield AtmosphereIndexEntry objects."""
        mock_record = Mock()
        mock_record.uri = f"at://did:plc:test/{LEXICON_NAMESPACE}.dataset/d1"
        mock_record.value = {
            "name": "listed-dataset",
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

        index = AtmosphereIndex(authenticated_client)
        entries = list(index.list_datasets())

        assert len(entries) == 1
        assert isinstance(entries[0], AtmosphereIndexEntry)

    def test_entry_from_record_has_properties(self):
        """AtmosphereIndexEntry should expose IndexEntry properties."""
        record = {
            "name": "test-dataset",
            "schemaRef": "at://did:plc:schema/schema/key",
            "storage": {
                "$type": f"{LEXICON_NAMESPACE}.storageExternal",
                "urls": ["s3://data.tar"],
            },
        }

        entry = AtmosphereIndexEntry("at://test/dataset/key", record)

        assert entry.name == "test-dataset"
        assert entry.schema_ref == "at://did:plc:schema/schema/key"
        assert entry.data_urls == ["s3://data.tar"]
        assert entry.uri == "at://test/dataset/key"

    def test_entry_metadata_unpacking(self):
        """Entry should unpack msgpack metadata."""
        original_meta = {"version": "1.0", "count": 100}
        packed_meta = msgpack.packb(original_meta)

        record = {
            "name": "meta-dataset",
            "schemaRef": "at://schema",
            "storage": {
                "$type": f"{LEXICON_NAMESPACE}.storageExternal",
                "urls": ["s3://data.tar"],
            },
            "metadata": packed_meta,
        }

        entry = AtmosphereIndexEntry("at://test/dataset/key", record)

        assert entry.metadata == original_meta
        assert entry.metadata["version"] == "1.0"

    def test_entry_no_metadata_returns_none(self):
        """Entry without metadata should return None."""
        record = {
            "name": "no-meta",
            "schemaRef": "at://schema",
            "storage": {
                "$type": f"{LEXICON_NAMESPACE}.storageExternal",
                "urls": ["s3://data.tar"],
            },
        }

        entry = AtmosphereIndexEntry("at://test/dataset/key", record)

        assert entry.metadata is None


class TestExternalStorageUrls:
    """Tests for datasets with external storage URLs."""

    def test_publish_with_urls(self, authenticated_client, mock_atproto_client):
        """Publish dataset with external URLs."""
        mock_response = Mock()
        mock_response.uri = f"at://did:plc:test/{LEXICON_NAMESPACE}.dataset/urls"
        mock_atproto_client.com.atproto.repo.create_record.return_value = mock_response

        publisher = DatasetPublisher(authenticated_client)
        uri = publisher.publish_with_urls(
            urls=["s3://bucket/shard-000.tar", "s3://bucket/shard-001.tar"],
            schema_uri="at://did:plc:schema/schema/key",
            name="multi-url-dataset",
        )

        assert uri is not None

        call_args = mock_atproto_client.com.atproto.repo.create_record.call_args
        record = call_args.kwargs["data"]["record"]

        assert record["storage"]["$type"] == f"{LEXICON_NAMESPACE}.storageExternal"
        assert len(record["storage"]["urls"]) == 2

    def test_entry_extracts_external_urls(self):
        """Entry should extract URLs from external storage."""
        record = {
            "name": "external-test",
            "schemaRef": "at://schema",
            "storage": {
                "$type": f"{LEXICON_NAMESPACE}.storageExternal",
                "urls": [
                    "https://cdn.example.com/data-000.tar",
                    "https://cdn.example.com/data-001.tar",
                ],
            },
        }

        entry = AtmosphereIndexEntry("at://test/dataset/key", record)

        assert entry.data_urls == [
            "https://cdn.example.com/data-000.tar",
            "https://cdn.example.com/data-001.tar",
        ]


class TestSchemaPublishing:
    """Tests for schema record publishing."""

    def test_publish_basic_schema(self, authenticated_client, mock_atproto_client):
        """Schema should publish with correct structure."""
        mock_response = Mock()
        mock_response.uri = f"at://did:plc:test/{LEXICON_NAMESPACE}.sampleSchema/basic"
        mock_atproto_client.com.atproto.repo.create_record.return_value = mock_response

        publisher = SchemaPublisher(authenticated_client)
        uri = publisher.publish(AtmoSample, version="1.0.0")

        assert isinstance(uri, AtUri)

        call_args = mock_atproto_client.com.atproto.repo.create_record.call_args
        record = call_args.kwargs["data"]["record"]

        assert record["name"] == "AtmoSample"
        assert record["version"] == "1.0.0"

        field_names = {f["name"] for f in record["fields"]}
        assert "name" in field_names
        assert "value" in field_names

    def test_publish_ndarray_schema(self, authenticated_client, mock_atproto_client):
        """Schema with NDArray field should publish correctly."""
        mock_response = Mock()
        mock_response.uri = (
            f"at://did:plc:test/{LEXICON_NAMESPACE}.sampleSchema/ndarray"
        )
        mock_atproto_client.com.atproto.repo.create_record.return_value = mock_response

        publisher = SchemaPublisher(authenticated_client)
        publisher.publish(AtmoNDArraySample, version="1.0.0")

        call_args = mock_atproto_client.com.atproto.repo.create_record.call_args
        record = call_args.kwargs["data"]["record"]

        # Find the data field
        data_field = next(f for f in record["fields"] if f["name"] == "data")
        assert "ndarray" in data_field["fieldType"]["$type"]


class TestErrorHandling:
    """Tests for error handling in atmosphere operations."""

    def test_not_authenticated_raises_on_publish(self, mock_atproto_client):
        """Publishing without authentication should raise."""
        client = AtmosphereClient(_client=mock_atproto_client)

        publisher = SchemaPublisher(client)

        with pytest.raises(ValueError, match="authenticated"):
            publisher.publish(AtmoSample, version="1.0.0")

    def test_invalid_uri_raises(self):
        """Invalid AT URI should raise ValueError."""
        with pytest.raises(ValueError):
            AtUri.parse("not-a-valid-uri")

        with pytest.raises(ValueError):
            AtUri.parse("https://example.com/path")

    def test_uri_missing_parts_raises(self):
        """AT URI with missing parts should raise."""
        with pytest.raises(ValueError, match="expected authority/collection/rkey"):
            AtUri.parse("at://did:plc:abc/collection")


class TestAtUriParsing:
    """Tests for AT URI parsing and formatting."""

    def test_parse_valid_uri(self):
        """Parse a valid AT URI."""
        uri = AtUri.parse("at://did:plc:abc123/com.example.record/key456")

        assert uri.authority == "did:plc:abc123"
        assert uri.collection == "com.example.record"
        assert uri.rkey == "key456"

    def test_uri_str_roundtrip(self):
        """String conversion should roundtrip."""
        original = "at://did:plc:test123/ac.foundation.dataset.sampleSchema/xyz789"
        uri = AtUri.parse(original)
        assert str(uri) == original

    def test_parse_atdata_namespace(self):
        """Parse URIs in the atdata namespace."""
        uri = AtUri.parse(f"at://did:plc:abc/{LEXICON_NAMESPACE}.sampleSchema/test")

        assert uri.collection == f"{LEXICON_NAMESPACE}.sampleSchema"


class TestPDSBlobStore:
    """Tests for PDSBlobStore blob storage."""

    def test_create_with_client(self, authenticated_client):
        """PDSBlobStore can be created with authenticated client."""
        from atdata.atmosphere import PDSBlobStore

        store = PDSBlobStore(client=authenticated_client)
        assert store.client is authenticated_client

    def test_supports_streaming(self, authenticated_client):
        """PDSBlobStore supports streaming."""
        from atdata.atmosphere import PDSBlobStore

        store = PDSBlobStore(client=authenticated_client)
        assert store.supports_streaming() is True

    def test_write_shards_requires_auth(self, mock_atproto_client):
        """write_shards raises if client not authenticated."""
        from atdata.atmosphere import PDSBlobStore

        # Create client without login
        client = AtmosphereClient(_client=mock_atproto_client)
        # Clear session
        client._session = None

        store = PDSBlobStore(client=client)

        # Create minimal mock dataset
        mock_ds = Mock()
        mock_ds.ordered.return_value = iter([])

        with pytest.raises(ValueError, match="Not authenticated"):
            store.write_shards(mock_ds, prefix="test")

    def test_write_shards_uploads_blobs(
        self, authenticated_client, mock_atproto_client, tmp_path
    ):
        """write_shards uploads each shard as a blob."""
        from atdata.atmosphere import PDSBlobStore
        import webdataset as wds

        # Create a test dataset with samples
        tar_path = tmp_path / "test.tar"
        with wds.writer.TarWriter(str(tar_path)) as sink:
            sample = AtmoSample(name="test", value=42)
            sink.write(sample.as_wds)

        ds = atdata.Dataset[AtmoSample](str(tar_path))

        # Mock upload_blob to return a blob reference
        authenticated_client.upload_blob = Mock(
            return_value={
                "$type": "blob",
                "ref": {"$link": "bafyrei123abc"},
                "mimeType": "application/x-tar",
                "size": 1024,
            }
        )

        store = PDSBlobStore(client=authenticated_client)
        urls = store.write_shards(ds, prefix="test/v1", maxcount=100)

        # Should have uploaded one shard
        assert len(urls) == 1
        assert urls[0] == "at://did:plc:integration123/blob/bafyrei123abc"

        # Verify upload_blob was called with tar data
        authenticated_client.upload_blob.assert_called_once()
        call_args = authenticated_client.upload_blob.call_args
        assert call_args.kwargs["mime_type"] == "application/x-tar"
        # First arg should be bytes (tar data)
        assert isinstance(call_args.args[0], bytes)

    def test_read_url_transforms_at_uri(
        self, authenticated_client, mock_atproto_client
    ):
        """read_url transforms AT URIs to HTTP URLs."""
        from atdata.atmosphere import PDSBlobStore

        authenticated_client.get_blob_url = Mock(
            return_value="https://pds.example.com/xrpc/com.atproto.sync.getBlob?did=did:plc:abc&cid=bafyrei123"
        )

        store = PDSBlobStore(client=authenticated_client)
        url = store.read_url("at://did:plc:abc/blob/bafyrei123")

        assert "https://pds.example.com" in url
        assert "bafyrei123" in url
        authenticated_client.get_blob_url.assert_called_once_with(
            "did:plc:abc", "bafyrei123"
        )

    def test_read_url_passes_non_at_uri(self, authenticated_client):
        """read_url returns non-AT URIs unchanged."""
        from atdata.atmosphere import PDSBlobStore

        store = PDSBlobStore(client=authenticated_client)
        url = store.read_url("https://example.com/data.tar")

        assert url == "https://example.com/data.tar"

    def test_read_url_invalid_format(self, authenticated_client):
        """read_url raises on invalid AT URI format."""
        from atdata.atmosphere import PDSBlobStore

        store = PDSBlobStore(client=authenticated_client)

        with pytest.raises(ValueError, match="Invalid blob AT URI format"):
            store.read_url("at://did:plc:abc/invalid/format/extra")

    def test_create_source_returns_blob_source(self, authenticated_client):
        """create_source returns BlobSource for AT URIs."""
        from atdata.atmosphere import PDSBlobStore
        from atdata._sources import BlobSource

        store = PDSBlobStore(client=authenticated_client)
        source = store.create_source(
            [
                "at://did:plc:abc/blob/bafyrei111",
                "at://did:plc:abc/blob/bafyrei222",
            ]
        )

        assert isinstance(source, BlobSource)
        assert len(source.blob_refs) == 2
        assert source.blob_refs[0]["did"] == "did:plc:abc"
        assert source.blob_refs[0]["cid"] == "bafyrei111"

    def test_create_source_invalid_url(self, authenticated_client):
        """create_source raises on non-AT URIs."""
        from atdata.atmosphere import PDSBlobStore

        store = PDSBlobStore(client=authenticated_client)

        with pytest.raises(ValueError, match="Not an AT URI"):
            store.create_source(["https://example.com/data.tar"])

    def test_atmosphere_index_with_data_store(self, authenticated_client):
        """AtmosphereIndex can be created with PDSBlobStore."""
        from atdata.atmosphere import PDSBlobStore

        store = PDSBlobStore(client=authenticated_client)
        index = AtmosphereIndex(client=authenticated_client, data_store=store)

        assert index.data_store is store

    def test_atmosphere_index_data_store_property(self, authenticated_client):
        """AtmosphereIndex.data_store property returns the store."""
        from atdata.atmosphere import PDSBlobStore

        store = PDSBlobStore(client=authenticated_client)
        index = AtmosphereIndex(client=authenticated_client, data_store=store)

        assert index.data_store is store

    def test_atmosphere_index_without_data_store(self, authenticated_client):
        """AtmosphereIndex without data_store has None."""
        index = AtmosphereIndex(client=authenticated_client)

        assert index.data_store is None
