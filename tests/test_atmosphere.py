"""Tests for the atdata.atmosphere module.

This module contains comprehensive tests for ATProto integration including:
- Type definitions (_types.py)
- Client wrapper (client.py)
- Schema publishing/loading (schema.py)
- Dataset publishing/loading (records.py)
- Lens publishing/loading (lens.py)
"""

from datetime import datetime, timezone
from typing import Optional
from unittest.mock import Mock, MagicMock, patch
import pytest

import numpy as np
from numpy.typing import NDArray

import atdata
from atdata.atmosphere import (
    AtmosphereClient,
    SchemaPublisher,
    SchemaLoader,
    DatasetPublisher,
    DatasetLoader,
    LensPublisher,
    LensLoader,
    AtUri,
    SchemaRecord,
    DatasetRecord,
    LensRecord,
)
from atdata.atmosphere._types import (
    FieldType,
    FieldDef,
    StorageLocation,
    CodeReference,
    LEXICON_NAMESPACE,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def mock_atproto_client():
    """Create a mock atproto SDK client."""
    mock = Mock()
    mock.me = MagicMock()
    mock.me.did = "did:plc:test123456789"
    mock.me.handle = "test.bsky.social"

    # Mock login
    mock_profile = Mock()
    mock_profile.did = "did:plc:test123456789"
    mock_profile.handle = "test.bsky.social"
    mock.login.return_value = mock_profile

    # Mock export_session_string
    mock.export_session_string.return_value = "test-session-string"

    return mock


@pytest.fixture
def authenticated_client(mock_atproto_client):
    """Create an authenticated AtmosphereClient with mocked backend."""
    client = AtmosphereClient(_client=mock_atproto_client)
    client.login("test.bsky.social", "test-password")
    return client


@atdata.packable
class BasicSample:
    """Simple sample type for testing."""
    name: str
    value: int


@atdata.packable
class NumpySample:
    """Sample type with NDArray field."""
    data: NDArray
    label: str


@atdata.packable
class OptionalSample:
    """Sample type with optional fields."""
    required_field: str
    optional_field: Optional[int]
    optional_array: Optional[NDArray]


@atdata.packable
class AllTypesSample:
    """Sample type with all primitive types."""
    str_field: str
    int_field: int
    float_field: float
    bool_field: bool
    bytes_field: bytes


# =============================================================================
# Tests for _types.py - AtUri
# =============================================================================

class TestAtUri:
    """Tests for AtUri parsing and formatting."""

    def test_parse_valid_uri_with_did(self):
        """Parse a valid AT URI with a DID authority."""
        uri = AtUri.parse("at://did:plc:abc123/com.example.record/key456")

        assert uri.authority == "did:plc:abc123"
        assert uri.collection == "com.example.record"
        assert uri.rkey == "key456"

    def test_parse_valid_uri_with_handle(self):
        """Parse a valid AT URI with a handle authority."""
        uri = AtUri.parse("at://alice.bsky.social/app.bsky.feed.post/abc123")

        assert uri.authority == "alice.bsky.social"
        assert uri.collection == "app.bsky.feed.post"
        assert uri.rkey == "abc123"

    def test_parse_uri_with_slashes_in_rkey(self):
        """Parse a URI where rkey contains slashes."""
        uri = AtUri.parse("at://did:plc:abc/collection/path/to/key")

        assert uri.authority == "did:plc:abc"
        assert uri.collection == "collection"
        assert uri.rkey == "path/to/key"

    def test_parse_invalid_uri_no_protocol(self):
        """Reject URIs without at:// protocol."""
        with pytest.raises(ValueError, match="must start with 'at://'"):
            AtUri.parse("https://example.com/path")

    def test_parse_invalid_uri_missing_parts(self):
        """Reject URIs with missing components."""
        with pytest.raises(ValueError, match="expected authority/collection/rkey"):
            AtUri.parse("at://did:plc:abc/collection")

    def test_str_roundtrip(self):
        """Verify __str__ produces valid URI that can be re-parsed."""
        original = "at://did:plc:test123/ac.foundation.dataset.sampleSchema/xyz789"
        uri = AtUri.parse(original)
        assert str(uri) == original

    def test_parse_atdata_namespace(self):
        """Parse URIs in the atdata namespace."""
        uri = AtUri.parse(f"at://did:plc:abc/{LEXICON_NAMESPACE}.sampleSchema/test")

        assert uri.collection == f"{LEXICON_NAMESPACE}.sampleSchema"


# =============================================================================
# Tests for _types.py - FieldType
# =============================================================================

class TestFieldType:
    """Tests for FieldType dataclass."""

    def test_primitive_type(self):
        """Create a primitive field type."""
        ft = FieldType(kind="primitive", primitive="str")

        assert ft.kind == "primitive"
        assert ft.primitive == "str"
        assert ft.dtype is None
        assert ft.shape is None

    def test_ndarray_type(self):
        """Create an ndarray field type."""
        ft = FieldType(kind="ndarray", dtype="float32", shape=[224, 224, 3])

        assert ft.kind == "ndarray"
        assert ft.dtype == "float32"
        assert ft.shape == [224, 224, 3]

    def test_ref_type(self):
        """Create a reference field type."""
        ft = FieldType(kind="ref", ref="at://did:plc:abc/collection/key")

        assert ft.kind == "ref"
        assert ft.ref == "at://did:plc:abc/collection/key"

    def test_array_type(self):
        """Create an array field type with items."""
        items = FieldType(kind="primitive", primitive="str")
        ft = FieldType(kind="array", items=items)

        assert ft.kind == "array"
        assert ft.items is not None
        assert ft.items.kind == "primitive"


# =============================================================================
# Tests for _types.py - FieldDef
# =============================================================================

class TestFieldDef:
    """Tests for FieldDef dataclass."""

    def test_required_field(self):
        """Create a required field definition."""
        fd = FieldDef(
            name="test_field",
            field_type=FieldType(kind="primitive", primitive="str"),
            optional=False,
        )

        assert fd.name == "test_field"
        assert fd.optional is False

    def test_optional_field(self):
        """Create an optional field definition."""
        fd = FieldDef(
            name="optional_field",
            field_type=FieldType(kind="primitive", primitive="int"),
            optional=True,
        )

        assert fd.optional is True

    def test_field_with_description(self):
        """Create a field with description."""
        fd = FieldDef(
            name="described_field",
            field_type=FieldType(kind="primitive", primitive="float"),
            optional=False,
            description="A field with a description",
        )

        assert fd.description == "A field with a description"


# =============================================================================
# Tests for _types.py - SchemaRecord
# =============================================================================

class TestSchemaRecord:
    """Tests for SchemaRecord dataclass and to_record()."""

    def test_to_record_basic(self):
        """Convert a basic schema record to dict."""
        schema = SchemaRecord(
            name="TestSchema",
            version="1.0.0",
            fields=[
                FieldDef(
                    name="field1",
                    field_type=FieldType(kind="primitive", primitive="str"),
                    optional=False,
                ),
            ],
        )

        record = schema.to_record()

        assert record["$type"] == f"{LEXICON_NAMESPACE}.sampleSchema"
        assert record["name"] == "TestSchema"
        assert record["version"] == "1.0.0"
        assert len(record["fields"]) == 1
        assert "createdAt" in record

    def test_to_record_with_description(self):
        """Convert schema record with description."""
        schema = SchemaRecord(
            name="DescribedSchema",
            version="2.0.0",
            description="A schema with description",
            fields=[],
        )

        record = schema.to_record()

        assert record["description"] == "A schema with description"

    def test_to_record_with_metadata(self):
        """Convert schema record with metadata."""
        schema = SchemaRecord(
            name="MetaSchema",
            version="1.0.0",
            fields=[],
            metadata={"author": "test", "tags": ["demo"]},
        )

        record = schema.to_record()

        assert record["metadata"] == {"author": "test", "tags": ["demo"]}

    def test_to_record_field_types(self):
        """Verify field type serialization in to_record()."""
        schema = SchemaRecord(
            name="TypesSchema",
            version="1.0.0",
            fields=[
                FieldDef(
                    name="primitive_field",
                    field_type=FieldType(kind="primitive", primitive="int"),
                    optional=False,
                ),
                FieldDef(
                    name="array_field",
                    field_type=FieldType(kind="ndarray", dtype="float32"),
                    optional=True,
                ),
            ],
        )

        record = schema.to_record()

        # Check primitive field
        prim_field = record["fields"][0]
        assert prim_field["name"] == "primitive_field"
        assert prim_field["fieldType"]["$type"] == f"{LEXICON_NAMESPACE}.schemaType#primitive"
        assert prim_field["fieldType"]["primitive"] == "int"
        assert prim_field["optional"] is False

        # Check ndarray field
        arr_field = record["fields"][1]
        assert arr_field["name"] == "array_field"
        assert arr_field["fieldType"]["$type"] == f"{LEXICON_NAMESPACE}.schemaType#ndarray"
        assert arr_field["fieldType"]["dtype"] == "float32"
        assert arr_field["optional"] is True


# =============================================================================
# Tests for _types.py - StorageLocation
# =============================================================================

class TestStorageLocation:
    """Tests for StorageLocation dataclass."""

    def test_external_storage(self):
        """Create external URL storage location."""
        storage = StorageLocation(
            kind="external",
            urls=["s3://bucket/data-{000000..000009}.tar"],
        )

        assert storage.kind == "external"
        assert storage.urls == ["s3://bucket/data-{000000..000009}.tar"]
        assert storage.blob_refs is None

    def test_blob_storage(self):
        """Create ATProto blob storage location."""
        storage = StorageLocation(
            kind="blobs",
            blob_refs=[{"cid": "bafyabc", "mimeType": "application/octet-stream"}],
        )

        assert storage.kind == "blobs"
        assert storage.blob_refs is not None
        assert len(storage.blob_refs) == 1


# =============================================================================
# Tests for _types.py - DatasetRecord
# =============================================================================

class TestDatasetRecord:
    """Tests for DatasetRecord dataclass and to_record()."""

    def test_to_record_external_storage(self):
        """Convert dataset record with external storage."""
        dataset = DatasetRecord(
            name="TestDataset",
            schema_ref="at://did:plc:abc/ac.foundation.dataset.sampleSchema/xyz",
            storage=StorageLocation(
                kind="external",
                urls=["s3://bucket/data.tar"],
            ),
        )

        record = dataset.to_record()

        assert record["$type"] == f"{LEXICON_NAMESPACE}.record"
        assert record["name"] == "TestDataset"
        assert record["schemaRef"] == "at://did:plc:abc/ac.foundation.dataset.sampleSchema/xyz"
        assert record["storage"]["$type"] == f"{LEXICON_NAMESPACE}.storageExternal"
        assert record["storage"]["urls"] == ["s3://bucket/data.tar"]

    def test_to_record_blob_storage(self):
        """Convert dataset record with blob storage."""
        dataset = DatasetRecord(
            name="BlobDataset",
            schema_ref="at://did:plc:abc/collection/key",
            storage=StorageLocation(
                kind="blobs",
                blob_refs=[{"cid": "bafytest"}],
            ),
        )

        record = dataset.to_record()

        assert record["storage"]["$type"] == f"{LEXICON_NAMESPACE}.storageBlobs"
        assert record["storage"]["blobs"] == [{"cid": "bafytest"}]

    def test_to_record_with_tags_and_license(self):
        """Convert dataset record with tags and license."""
        dataset = DatasetRecord(
            name="TaggedDataset",
            schema_ref="at://did:plc:abc/collection/key",
            storage=StorageLocation(kind="external", urls=[]),
            tags=["ml", "vision", "demo"],
            license="MIT",
        )

        record = dataset.to_record()

        assert record["tags"] == ["ml", "vision", "demo"]
        assert record["license"] == "MIT"

    def test_to_record_with_metadata(self):
        """Convert dataset record with msgpack metadata."""
        import msgpack

        metadata_bytes = msgpack.packb({"size": 1000, "split": "train"})
        dataset = DatasetRecord(
            name="MetaDataset",
            schema_ref="at://did:plc:abc/collection/key",
            storage=StorageLocation(kind="external", urls=[]),
            metadata=metadata_bytes,
        )

        record = dataset.to_record()

        assert record["metadata"] == metadata_bytes


# =============================================================================
# Tests for _types.py - LensRecord
# =============================================================================

class TestLensRecord:
    """Tests for LensRecord dataclass and to_record()."""

    def test_to_record_basic(self):
        """Convert basic lens record."""
        lens = LensRecord(
            name="TestLens",
            source_schema="at://did:plc:abc/collection/source",
            target_schema="at://did:plc:abc/collection/target",
        )

        record = lens.to_record()

        assert record["$type"] == f"{LEXICON_NAMESPACE}.lens"
        assert record["name"] == "TestLens"
        assert record["sourceSchema"] == "at://did:plc:abc/collection/source"
        assert record["targetSchema"] == "at://did:plc:abc/collection/target"
        assert "createdAt" in record

    def test_to_record_with_description(self):
        """Convert lens record with description."""
        lens = LensRecord(
            name="DescribedLens",
            source_schema="at://a",
            target_schema="at://b",
            description="Transforms A to B",
        )

        record = lens.to_record()

        assert record["description"] == "Transforms A to B"

    def test_to_record_with_code_references(self):
        """Convert lens record with code references."""
        lens = LensRecord(
            name="CodeLens",
            source_schema="at://a",
            target_schema="at://b",
            getter_code=CodeReference(
                repository="https://github.com/user/repo",
                commit="abc123def456",
                path="module.lenses:getter_func",
            ),
            putter_code=CodeReference(
                repository="https://github.com/user/repo",
                commit="abc123def456",
                path="module.lenses:putter_func",
            ),
        )

        record = lens.to_record()

        assert record["getterCode"]["repository"] == "https://github.com/user/repo"
        assert record["getterCode"]["commit"] == "abc123def456"
        assert record["getterCode"]["path"] == "module.lenses:getter_func"
        assert record["putterCode"]["path"] == "module.lenses:putter_func"


# =============================================================================
# Tests for client.py - AtmosphereClient
# =============================================================================

class TestAtmosphereClient:
    """Tests for AtmosphereClient."""

    def test_init_default(self):
        """Initialize client with defaults."""
        with patch("atdata.atmosphere.client._get_atproto_client_class") as mock_get:
            mock_class = Mock()
            mock_get.return_value = mock_class

            client = AtmosphereClient()

            mock_class.assert_called_once()
            assert not client.is_authenticated

    def test_init_with_base_url(self):
        """Initialize client with custom base URL."""
        with patch("atdata.atmosphere.client._get_atproto_client_class") as mock_get:
            mock_class = Mock()
            mock_get.return_value = mock_class

            client = AtmosphereClient(base_url="https://custom.pds.example")

            mock_class.assert_called_once_with(base_url="https://custom.pds.example")

    def test_init_with_mock_client(self, mock_atproto_client):
        """Initialize with pre-configured mock client."""
        client = AtmosphereClient(_client=mock_atproto_client)

        assert client._client is mock_atproto_client

    def test_login_success(self, mock_atproto_client):
        """Successful login sets session."""
        client = AtmosphereClient(_client=mock_atproto_client)

        client.login("test.bsky.social", "password123")

        assert client.is_authenticated
        assert client.did == "did:plc:test123456789"
        assert client.handle == "test.bsky.social"
        mock_atproto_client.login.assert_called_once_with("test.bsky.social", "password123")

    def test_login_with_session(self, mock_atproto_client):
        """Login with exported session string."""
        client = AtmosphereClient(_client=mock_atproto_client)

        client.login_with_session("test-session-string")

        assert client.is_authenticated
        mock_atproto_client.login.assert_called_once_with(session_string="test-session-string")

    def test_export_session(self, authenticated_client, mock_atproto_client):
        """Export session string."""
        session = authenticated_client.export_session()

        assert session == "test-session-string"
        mock_atproto_client.export_session_string.assert_called_once()

    def test_export_session_not_authenticated(self, mock_atproto_client):
        """Export session raises when not authenticated."""
        client = AtmosphereClient(_client=mock_atproto_client)

        with pytest.raises(ValueError, match="Not authenticated"):
            client.export_session()

    def test_did_not_authenticated(self, mock_atproto_client):
        """Accessing did raises when not authenticated."""
        client = AtmosphereClient(_client=mock_atproto_client)

        with pytest.raises(ValueError, match="Not authenticated"):
            _ = client.did

    def test_handle_not_authenticated(self, mock_atproto_client):
        """Accessing handle raises when not authenticated."""
        client = AtmosphereClient(_client=mock_atproto_client)

        with pytest.raises(ValueError, match="Not authenticated"):
            _ = client.handle

    def test_create_record(self, authenticated_client, mock_atproto_client):
        """Create a record via the client."""
        mock_response = Mock()
        mock_response.uri = "at://did:plc:test123456789/collection/newkey"
        mock_atproto_client.com.atproto.repo.create_record.return_value = mock_response

        uri = authenticated_client.create_record(
            collection="collection",
            record={"$type": "collection", "data": "test"},
        )

        assert isinstance(uri, AtUri)
        assert uri.authority == "did:plc:test123456789"
        assert uri.collection == "collection"
        assert uri.rkey == "newkey"

    def test_create_record_not_authenticated(self, mock_atproto_client):
        """Create record raises when not authenticated."""
        client = AtmosphereClient(_client=mock_atproto_client)

        with pytest.raises(ValueError, match="must be authenticated"):
            client.create_record(collection="test", record={})

    def test_put_record(self, authenticated_client, mock_atproto_client):
        """Put (create or update) a record."""
        mock_response = Mock()
        mock_response.uri = "at://did:plc:test123456789/collection/specific-key"
        mock_atproto_client.com.atproto.repo.put_record.return_value = mock_response

        uri = authenticated_client.put_record(
            collection="collection",
            rkey="specific-key",
            record={"$type": "collection", "data": "test"},
        )

        assert uri.rkey == "specific-key"

    def test_get_record(self, authenticated_client, mock_atproto_client):
        """Get a record by URI."""
        mock_response = Mock()
        mock_response.value = {"$type": "test", "field": "value"}
        mock_atproto_client.com.atproto.repo.get_record.return_value = mock_response

        record = authenticated_client.get_record("at://did:plc:abc/collection/key")

        assert record["field"] == "value"

    def test_get_record_with_aturi_object(self, authenticated_client, mock_atproto_client):
        """Get a record using AtUri object."""
        mock_response = Mock()
        mock_response.value = {"$type": "test", "data": 123}
        mock_atproto_client.com.atproto.repo.get_record.return_value = mock_response

        uri = AtUri(authority="did:plc:abc", collection="collection", rkey="key")
        record = authenticated_client.get_record(uri)

        assert record["data"] == 123

    def test_delete_record(self, authenticated_client, mock_atproto_client):
        """Delete a record."""
        authenticated_client.delete_record("at://did:plc:test123456789/collection/key")

        mock_atproto_client.com.atproto.repo.delete_record.assert_called_once()

    def test_list_records(self, authenticated_client, mock_atproto_client):
        """List records in a collection."""
        mock_record1 = Mock()
        mock_record1.value = {"name": "record1"}
        mock_record2 = Mock()
        mock_record2.value = {"name": "record2"}

        mock_response = Mock()
        mock_response.records = [mock_record1, mock_record2]
        mock_response.cursor = "next-page"
        mock_atproto_client.com.atproto.repo.list_records.return_value = mock_response

        records, cursor = authenticated_client.list_records("collection", limit=10)

        assert len(records) == 2
        assert records[0]["name"] == "record1"
        assert cursor == "next-page"

    def test_list_schemas_convenience(self, authenticated_client, mock_atproto_client):
        """Test list_schemas convenience method."""
        mock_response = Mock()
        mock_response.records = []
        mock_response.cursor = None
        mock_atproto_client.com.atproto.repo.list_records.return_value = mock_response

        schemas = authenticated_client.list_schemas()

        call_args = mock_atproto_client.com.atproto.repo.list_records.call_args
        assert f"{LEXICON_NAMESPACE}.sampleSchema" in str(call_args)


# =============================================================================
# Tests for schema.py - SchemaPublisher
# =============================================================================

class TestSchemaPublisher:
    """Tests for SchemaPublisher."""

    def test_publish_basic_sample(self, authenticated_client, mock_atproto_client):
        """Publish a basic sample type schema."""
        mock_response = Mock()
        mock_response.uri = f"at://did:plc:test123456789/{LEXICON_NAMESPACE}.sampleSchema/abc"
        mock_atproto_client.com.atproto.repo.create_record.return_value = mock_response

        publisher = SchemaPublisher(authenticated_client)
        uri = publisher.publish(BasicSample, version="1.0.0")

        assert isinstance(uri, AtUri)
        assert uri.collection == f"{LEXICON_NAMESPACE}.sampleSchema"

        # Verify the record structure
        call_args = mock_atproto_client.com.atproto.repo.create_record.call_args
        record = call_args.kwargs["data"]["record"]
        assert record["name"] == "BasicSample"
        assert record["version"] == "1.0.0"
        assert len(record["fields"]) == 2

    def test_publish_with_custom_name(self, authenticated_client, mock_atproto_client):
        """Publish with custom name override."""
        mock_response = Mock()
        mock_response.uri = f"at://did:plc:test/{LEXICON_NAMESPACE}.sampleSchema/abc"
        mock_atproto_client.com.atproto.repo.create_record.return_value = mock_response

        publisher = SchemaPublisher(authenticated_client)
        publisher.publish(BasicSample, name="CustomName", version="2.0.0")

        call_args = mock_atproto_client.com.atproto.repo.create_record.call_args
        record = call_args.kwargs["data"]["record"]
        assert record["name"] == "CustomName"

    def test_publish_numpy_sample(self, authenticated_client, mock_atproto_client):
        """Publish sample type with NDArray field."""
        mock_response = Mock()
        mock_response.uri = f"at://did:plc:test/{LEXICON_NAMESPACE}.sampleSchema/abc"
        mock_atproto_client.com.atproto.repo.create_record.return_value = mock_response

        publisher = SchemaPublisher(authenticated_client)
        publisher.publish(NumpySample, version="1.0.0")

        call_args = mock_atproto_client.com.atproto.repo.create_record.call_args
        record = call_args.kwargs["data"]["record"]

        # Find the data field
        data_field = next(f for f in record["fields"] if f["name"] == "data")
        assert "ndarray" in data_field["fieldType"]["$type"]

    def test_publish_optional_fields(self, authenticated_client, mock_atproto_client):
        """Publish sample type with optional fields."""
        mock_response = Mock()
        mock_response.uri = f"at://did:plc:test/{LEXICON_NAMESPACE}.sampleSchema/abc"
        mock_atproto_client.com.atproto.repo.create_record.return_value = mock_response

        publisher = SchemaPublisher(authenticated_client)
        publisher.publish(OptionalSample, version="1.0.0")

        call_args = mock_atproto_client.com.atproto.repo.create_record.call_args
        record = call_args.kwargs["data"]["record"]

        # Check optional field marking
        required = next(f for f in record["fields"] if f["name"] == "required_field")
        optional = next(f for f in record["fields"] if f["name"] == "optional_field")

        assert required["optional"] is False
        assert optional["optional"] is True

    def test_publish_all_primitive_types(self, authenticated_client, mock_atproto_client):
        """Publish sample with all primitive types."""
        mock_response = Mock()
        mock_response.uri = f"at://did:plc:test/{LEXICON_NAMESPACE}.sampleSchema/abc"
        mock_atproto_client.com.atproto.repo.create_record.return_value = mock_response

        publisher = SchemaPublisher(authenticated_client)
        publisher.publish(AllTypesSample, version="1.0.0")

        call_args = mock_atproto_client.com.atproto.repo.create_record.call_args
        record = call_args.kwargs["data"]["record"]

        # Verify each primitive type
        type_map = {f["name"]: f["fieldType"]["primitive"] for f in record["fields"]}
        assert type_map["str_field"] == "str"
        assert type_map["int_field"] == "int"
        assert type_map["float_field"] == "float"
        assert type_map["bool_field"] == "bool"
        assert type_map["bytes_field"] == "bytes"

    def test_publish_not_dataclass_error(self, authenticated_client):
        """Publishing non-dataclass raises error."""
        publisher = SchemaPublisher(authenticated_client)

        class NotADataclass:
            pass

        with pytest.raises(ValueError, match="must be a dataclass"):
            publisher.publish(NotADataclass, version="1.0.0")


class TestSchemaLoader:
    """Tests for SchemaLoader."""

    def test_get_schema(self, authenticated_client, mock_atproto_client):
        """Get a schema by URI."""
        mock_response = Mock()
        mock_response.value = {
            "$type": f"{LEXICON_NAMESPACE}.sampleSchema",
            "name": "TestSchema",
            "version": "1.0.0",
            "fields": [],
        }
        mock_atproto_client.com.atproto.repo.get_record.return_value = mock_response

        loader = SchemaLoader(authenticated_client)
        schema = loader.get(f"at://did:plc:abc/{LEXICON_NAMESPACE}.sampleSchema/xyz")

        assert schema["name"] == "TestSchema"

    def test_get_schema_wrong_type(self, authenticated_client, mock_atproto_client):
        """Get raises error for wrong record type."""
        mock_response = Mock()
        mock_response.value = {
            "$type": "app.bsky.feed.post",
            "text": "Not a schema",
        }
        mock_atproto_client.com.atproto.repo.get_record.return_value = mock_response

        loader = SchemaLoader(authenticated_client)

        with pytest.raises(ValueError, match="not a schema record"):
            loader.get("at://did:plc:abc/app.bsky.feed.post/xyz")

    def test_list_all_schemas(self, authenticated_client, mock_atproto_client):
        """List all schemas."""
        mock_record = Mock()
        mock_record.value = {"name": "Schema1"}

        mock_response = Mock()
        mock_response.records = [mock_record]
        mock_response.cursor = None
        mock_atproto_client.com.atproto.repo.list_records.return_value = mock_response

        loader = SchemaLoader(authenticated_client)
        schemas = loader.list_all()

        assert len(schemas) == 1
        assert schemas[0]["name"] == "Schema1"


# =============================================================================
# Tests for records.py - DatasetPublisher
# =============================================================================

class TestDatasetPublisher:
    """Tests for DatasetPublisher."""

    def test_publish_with_urls(self, authenticated_client, mock_atproto_client):
        """Publish dataset with explicit URLs."""
        mock_response = Mock()
        mock_response.uri = f"at://did:plc:test/{LEXICON_NAMESPACE}.record/abc"
        mock_atproto_client.com.atproto.repo.create_record.return_value = mock_response

        publisher = DatasetPublisher(authenticated_client)
        uri = publisher.publish_with_urls(
            urls=["s3://bucket/data-{000000..000009}.tar"],
            schema_uri="at://did:plc:abc/schema/xyz",
            name="TestDataset",
            description="A test dataset",
            tags=["test", "demo"],
            license="MIT",
        )

        assert isinstance(uri, AtUri)

        call_args = mock_atproto_client.com.atproto.repo.create_record.call_args
        record = call_args.kwargs["data"]["record"]
        assert record["name"] == "TestDataset"
        assert record["schemaRef"] == "at://did:plc:abc/schema/xyz"
        assert record["tags"] == ["test", "demo"]
        assert record["license"] == "MIT"

    def test_publish_auto_schema(self, authenticated_client, mock_atproto_client):
        """Publish dataset with auto schema publishing."""
        # Mock for schema creation
        schema_response = Mock()
        schema_response.uri = f"at://did:plc:test/{LEXICON_NAMESPACE}.sampleSchema/schema123"

        # Mock for dataset creation
        dataset_response = Mock()
        dataset_response.uri = f"at://did:plc:test/{LEXICON_NAMESPACE}.record/dataset456"

        mock_atproto_client.com.atproto.repo.create_record.side_effect = [
            schema_response,
            dataset_response,
        ]

        # Create a mock dataset
        mock_dataset = Mock()
        mock_dataset.url = "s3://bucket/data.tar"
        mock_dataset.sample_type = BasicSample
        mock_dataset.metadata = None

        publisher = DatasetPublisher(authenticated_client)
        uri = publisher.publish(
            mock_dataset,
            name="AutoSchemaDataset",
            auto_publish_schema=True,
        )

        # Should have called create_record twice (schema + dataset)
        assert mock_atproto_client.com.atproto.repo.create_record.call_count == 2

    def test_publish_explicit_schema_uri(self, authenticated_client, mock_atproto_client):
        """Publish dataset with explicit schema URI (no auto publish)."""
        mock_response = Mock()
        mock_response.uri = f"at://did:plc:test/{LEXICON_NAMESPACE}.record/abc"
        mock_atproto_client.com.atproto.repo.create_record.return_value = mock_response

        mock_dataset = Mock()
        mock_dataset.url = "s3://bucket/data.tar"
        mock_dataset.metadata = None

        publisher = DatasetPublisher(authenticated_client)
        publisher.publish(
            mock_dataset,
            name="ExplicitSchemaDataset",
            schema_uri="at://did:plc:existing/schema/xyz",
            auto_publish_schema=False,
        )

        # Should have called create_record only once (dataset only)
        assert mock_atproto_client.com.atproto.repo.create_record.call_count == 1

    def test_publish_no_schema_error(self, authenticated_client):
        """Publish without schema_uri and auto_publish_schema=False raises."""
        mock_dataset = Mock()
        mock_dataset.url = "s3://bucket/data.tar"

        publisher = DatasetPublisher(authenticated_client)

        with pytest.raises(ValueError, match="schema_uri is required"):
            publisher.publish(
                mock_dataset,
                name="NoSchemaDataset",
                auto_publish_schema=False,
            )


class TestDatasetLoader:
    """Tests for DatasetLoader."""

    def test_get_dataset(self, authenticated_client, mock_atproto_client):
        """Get a dataset record."""
        mock_response = Mock()
        mock_response.value = {
            "$type": f"{LEXICON_NAMESPACE}.record",
            "name": "TestDataset",
            "schemaRef": "at://schema",
            "storage": {
                "$type": f"{LEXICON_NAMESPACE}.storageExternal",
                "urls": ["s3://bucket/data.tar"],
            },
        }
        mock_atproto_client.com.atproto.repo.get_record.return_value = mock_response

        loader = DatasetLoader(authenticated_client)
        record = loader.get(f"at://did:plc:abc/{LEXICON_NAMESPACE}.record/xyz")

        assert record["name"] == "TestDataset"

    def test_get_dataset_wrong_type(self, authenticated_client, mock_atproto_client):
        """Get raises error for wrong record type."""
        mock_response = Mock()
        mock_response.value = {
            "$type": f"{LEXICON_NAMESPACE}.sampleSchema",
            "name": "NotADataset",
        }
        mock_atproto_client.com.atproto.repo.get_record.return_value = mock_response

        loader = DatasetLoader(authenticated_client)

        with pytest.raises(ValueError, match="not a dataset record"):
            loader.get("at://did:plc:abc/collection/xyz")

    def test_get_urls(self, authenticated_client, mock_atproto_client):
        """Get WebDataset URLs from a dataset record."""
        mock_response = Mock()
        mock_response.value = {
            "$type": f"{LEXICON_NAMESPACE}.record",
            "name": "TestDataset",
            "schemaRef": "at://schema",
            "storage": {
                "$type": f"{LEXICON_NAMESPACE}.storageExternal",
                "urls": ["s3://bucket/data-{000000..000009}.tar", "s3://bucket/extra.tar"],
            },
        }
        mock_atproto_client.com.atproto.repo.get_record.return_value = mock_response

        loader = DatasetLoader(authenticated_client)
        urls = loader.get_urls(f"at://did:plc:abc/{LEXICON_NAMESPACE}.record/xyz")

        assert len(urls) == 2
        assert "data-{000000..000009}.tar" in urls[0]

    def test_get_urls_blob_storage_error(self, authenticated_client, mock_atproto_client):
        """Get URLs raises for blob storage datasets."""
        mock_response = Mock()
        mock_response.value = {
            "$type": f"{LEXICON_NAMESPACE}.record",
            "name": "BlobDataset",
            "schemaRef": "at://schema",
            "storage": {
                "$type": f"{LEXICON_NAMESPACE}.storageBlobs",
                "blobs": [{"cid": "bafytest"}],
            },
        }
        mock_atproto_client.com.atproto.repo.get_record.return_value = mock_response

        loader = DatasetLoader(authenticated_client)

        with pytest.raises(ValueError, match="blob storage"):
            loader.get_urls(f"at://did:plc:abc/{LEXICON_NAMESPACE}.record/xyz")

    def test_get_metadata(self, authenticated_client, mock_atproto_client):
        """Get metadata from dataset record."""
        import msgpack

        metadata_bytes = msgpack.packb({"split": "train", "samples": 10000})

        mock_response = Mock()
        mock_response.value = {
            "$type": f"{LEXICON_NAMESPACE}.record",
            "name": "MetaDataset",
            "schemaRef": "at://schema",
            "storage": {"$type": f"{LEXICON_NAMESPACE}.storageExternal", "urls": []},
            "metadata": metadata_bytes,
        }
        mock_atproto_client.com.atproto.repo.get_record.return_value = mock_response

        loader = DatasetLoader(authenticated_client)
        metadata = loader.get_metadata(f"at://did:plc:abc/{LEXICON_NAMESPACE}.record/xyz")

        assert metadata["split"] == "train"
        assert metadata["samples"] == 10000

    def test_get_metadata_none(self, authenticated_client, mock_atproto_client):
        """Get metadata returns None when not present."""
        mock_response = Mock()
        mock_response.value = {
            "$type": f"{LEXICON_NAMESPACE}.record",
            "name": "NoMetaDataset",
            "schemaRef": "at://schema",
            "storage": {"$type": f"{LEXICON_NAMESPACE}.storageExternal", "urls": []},
        }
        mock_atproto_client.com.atproto.repo.get_record.return_value = mock_response

        loader = DatasetLoader(authenticated_client)
        metadata = loader.get_metadata(f"at://did:plc:abc/{LEXICON_NAMESPACE}.record/xyz")

        assert metadata is None

    def test_list_all(self, authenticated_client, mock_atproto_client):
        """List all datasets."""
        mock_record = Mock()
        mock_record.value = {"name": "Dataset1"}

        mock_response = Mock()
        mock_response.records = [mock_record]
        mock_response.cursor = None
        mock_atproto_client.com.atproto.repo.list_records.return_value = mock_response

        loader = DatasetLoader(authenticated_client)
        datasets = loader.list_all()

        assert len(datasets) == 1


# =============================================================================
# Tests for lens.py - LensPublisher
# =============================================================================

class TestLensPublisher:
    """Tests for LensPublisher."""

    def test_publish_with_code_refs(self, authenticated_client, mock_atproto_client):
        """Publish lens with code references."""
        mock_response = Mock()
        mock_response.uri = f"at://did:plc:test/{LEXICON_NAMESPACE}.lens/abc"
        mock_atproto_client.com.atproto.repo.create_record.return_value = mock_response

        publisher = LensPublisher(authenticated_client)
        uri = publisher.publish(
            name="TestLens",
            source_schema_uri="at://did:plc:abc/schema/source",
            target_schema_uri="at://did:plc:abc/schema/target",
            description="Transforms source to target",
            code_repository="https://github.com/user/repo",
            code_commit="abc123def456",
            getter_path="module.lenses:my_getter",
            putter_path="module.lenses:my_putter",
        )

        assert isinstance(uri, AtUri)

        call_args = mock_atproto_client.com.atproto.repo.create_record.call_args
        record = call_args.kwargs["data"]["record"]
        assert record["name"] == "TestLens"
        assert record["sourceSchema"] == "at://did:plc:abc/schema/source"
        assert record["targetSchema"] == "at://did:plc:abc/schema/target"
        assert record["getterCode"]["repository"] == "https://github.com/user/repo"

    def test_publish_without_code_refs(self, authenticated_client, mock_atproto_client):
        """Publish lens without code references."""
        mock_response = Mock()
        mock_response.uri = f"at://did:plc:test/{LEXICON_NAMESPACE}.lens/abc"
        mock_atproto_client.com.atproto.repo.create_record.return_value = mock_response

        publisher = LensPublisher(authenticated_client)
        uri = publisher.publish(
            name="MetadataOnlyLens",
            source_schema_uri="at://source",
            target_schema_uri="at://target",
        )

        call_args = mock_atproto_client.com.atproto.repo.create_record.call_args
        record = call_args.kwargs["data"]["record"]
        assert "getterCode" not in record
        assert "putterCode" not in record

    def test_publish_from_lens_object(self, authenticated_client, mock_atproto_client):
        """Publish lens from an atdata Lens object."""
        mock_response = Mock()
        mock_response.uri = f"at://did:plc:test/{LEXICON_NAMESPACE}.lens/abc"
        mock_atproto_client.com.atproto.repo.create_record.return_value = mock_response

        # Create a real lens
        @atdata.lens
        def test_lens(source: BasicSample) -> NumpySample:
            return NumpySample(
                data=np.array([source.value]),
                label=source.name,
            )

        publisher = LensPublisher(authenticated_client)
        uri = publisher.publish_from_lens(
            test_lens,
            name="FromObjectLens",
            source_schema_uri="at://source",
            target_schema_uri="at://target",
            code_repository="https://github.com/user/repo",
            code_commit="abc123",
        )

        call_args = mock_atproto_client.com.atproto.repo.create_record.call_args
        record = call_args.kwargs["data"]["record"]
        assert "test_lens" in record["getterCode"]["path"]


class TestLensLoader:
    """Tests for LensLoader."""

    def test_get_lens(self, authenticated_client, mock_atproto_client):
        """Get a lens record."""
        mock_response = Mock()
        mock_response.value = {
            "$type": f"{LEXICON_NAMESPACE}.lens",
            "name": "TestLens",
            "sourceSchema": "at://source",
            "targetSchema": "at://target",
        }
        mock_atproto_client.com.atproto.repo.get_record.return_value = mock_response

        loader = LensLoader(authenticated_client)
        record = loader.get(f"at://did:plc:abc/{LEXICON_NAMESPACE}.lens/xyz")

        assert record["name"] == "TestLens"

    def test_get_lens_wrong_type(self, authenticated_client, mock_atproto_client):
        """Get raises error for wrong record type."""
        mock_response = Mock()
        mock_response.value = {
            "$type": f"{LEXICON_NAMESPACE}.record",
            "name": "NotALens",
        }
        mock_atproto_client.com.atproto.repo.get_record.return_value = mock_response

        loader = LensLoader(authenticated_client)

        with pytest.raises(ValueError, match="not a lens record"):
            loader.get("at://did:plc:abc/collection/xyz")

    def test_list_all(self, authenticated_client, mock_atproto_client):
        """List all lens records."""
        mock_record = Mock()
        mock_record.value = {"name": "Lens1"}

        mock_response = Mock()
        mock_response.records = [mock_record]
        mock_response.cursor = None
        mock_atproto_client.com.atproto.repo.list_records.return_value = mock_response

        loader = LensLoader(authenticated_client)
        lenses = loader.list_all()

        assert len(lenses) == 1

    def test_find_by_schemas_source_only(self, authenticated_client, mock_atproto_client):
        """Find lenses by source schema only."""
        mock_records = [
            Mock(value={"sourceSchema": "at://schema/a", "targetSchema": "at://schema/b"}),
            Mock(value={"sourceSchema": "at://schema/a", "targetSchema": "at://schema/c"}),
            Mock(value={"sourceSchema": "at://schema/x", "targetSchema": "at://schema/y"}),
        ]

        mock_response = Mock()
        mock_response.records = mock_records
        mock_response.cursor = None
        mock_atproto_client.com.atproto.repo.list_records.return_value = mock_response

        loader = LensLoader(authenticated_client)
        matches = loader.find_by_schemas(source_schema_uri="at://schema/a")

        assert len(matches) == 2

    def test_find_by_schemas_both(self, authenticated_client, mock_atproto_client):
        """Find lenses by both source and target schema."""
        mock_records = [
            Mock(value={"sourceSchema": "at://schema/a", "targetSchema": "at://schema/b"}),
            Mock(value={"sourceSchema": "at://schema/a", "targetSchema": "at://schema/c"}),
        ]

        mock_response = Mock()
        mock_response.records = mock_records
        mock_response.cursor = None
        mock_atproto_client.com.atproto.repo.list_records.return_value = mock_response

        loader = LensLoader(authenticated_client)
        matches = loader.find_by_schemas(
            source_schema_uri="at://schema/a",
            target_schema_uri="at://schema/b",
        )

        assert len(matches) == 1
        assert matches[0]["targetSchema"] == "at://schema/b"


# =============================================================================
# Additional Edge Case Tests for Coverage
# =============================================================================


class TestFieldTypeEdgeCases:
    """Tests for FieldType and FieldDef edge cases."""

    def test_field_with_description(self):
        """Test FieldDef with description is included in dict output."""
        field_type = FieldType(kind="primitive", primitive="str")
        field_def = FieldDef(
            name="described_field",
            field_type=field_type,
            optional=False,
            description="This is a description",
        )

        # Create a SchemaRecord to test _field_to_dict
        schema = SchemaRecord(
            name="TestSchema",
            version="1.0.0",
            fields=[field_def],
        )
        record = schema.to_record()

        # Check that description is included
        field = record["fields"][0]
        assert field["description"] == "This is a description"

    def test_ndarray_type_with_shape(self):
        """Test FieldType for ndarray with shape."""
        field_type = FieldType(
            kind="ndarray",
            dtype="float32",
            shape=[224, 224, 3],
        )

        schema = SchemaRecord(
            name="ShapedArraySchema",
            version="1.0.0",
            fields=[FieldDef(name="image", field_type=field_type, optional=False)],
        )
        record = schema.to_record()

        field = record["fields"][0]
        assert field["fieldType"]["shape"] == [224, 224, 3]

    def test_ref_type(self):
        """Test FieldType for reference type."""
        field_type = FieldType(
            kind="ref",
            ref="at://did:plc:abc/atdata.sampleSchema/xyz",
        )

        schema = SchemaRecord(
            name="RefSchema",
            version="1.0.0",
            fields=[FieldDef(name="reference", field_type=field_type, optional=False)],
        )
        record = schema.to_record()

        field = record["fields"][0]
        assert "ref" in field["fieldType"]["$type"]
        assert field["fieldType"]["ref"] == "at://did:plc:abc/atdata.sampleSchema/xyz"

    def test_array_type_with_items(self):
        """Test FieldType for array with typed items."""
        items_type = FieldType(kind="primitive", primitive="int")
        field_type = FieldType(kind="array", items=items_type)

        schema = SchemaRecord(
            name="ArraySchema",
            version="1.0.0",
            fields=[FieldDef(name="numbers", field_type=field_type, optional=False)],
        )
        record = schema.to_record()

        field = record["fields"][0]
        assert "array" in field["fieldType"]["$type"]
        assert field["fieldType"]["items"]["primitive"] == "int"


class TestSchemaPublisherEdgeCases:
    """Additional edge case tests for SchemaPublisher."""

    def test_publish_list_field(self, authenticated_client, mock_atproto_client):
        """Publish sample type with List[str] field."""
        from typing import List

        @atdata.packable
        class ListSample:
            tags: List[str]
            values: List[int]

        mock_response = Mock()
        mock_response.uri = f"at://did:plc:test/{LEXICON_NAMESPACE}.sampleSchema/abc"
        mock_atproto_client.com.atproto.repo.create_record.return_value = mock_response

        publisher = SchemaPublisher(authenticated_client)
        publisher.publish(ListSample, version="1.0.0")

        call_args = mock_atproto_client.com.atproto.repo.create_record.call_args
        record = call_args.kwargs["data"]["record"]

        # Find the tags field
        tags_field = next(f for f in record["fields"] if f["name"] == "tags")
        assert "array" in tags_field["fieldType"]["$type"]

    def test_publish_nested_dataclass_error(self, authenticated_client):
        """Publishing sample with nested dataclass raises error."""
        from dataclasses import dataclass

        @dataclass
        class Inner:
            value: int

        @atdata.packable
        class Outer:
            nested: Inner

        publisher = SchemaPublisher(authenticated_client)

        with pytest.raises(TypeError, match="Nested dataclass types not yet supported"):
            publisher.publish(Outer, version="1.0.0")

    def test_publish_unsupported_type_error(self, authenticated_client):
        """Publishing sample with unsupported type raises error."""

        @atdata.packable
        class UnsupportedSample:
            value: complex  # complex is not a supported type

        publisher = SchemaPublisher(authenticated_client)

        with pytest.raises(TypeError, match="Unsupported type"):
            publisher.publish(UnsupportedSample, version="1.0.0")
