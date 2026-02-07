"""Tests for the atdata.atmosphere module.

This module contains comprehensive tests for ATProto integration including:
- Type definitions (_types.py)
- Client wrapper (client.py)
- Schema publishing/loading (schema.py)
- Dataset publishing/loading (records.py)
- Lens publishing/loading (lens.py)
"""

from typing import Optional
from unittest.mock import Mock, MagicMock, patch
import pytest

import numpy as np
from numpy.typing import NDArray

import atdata
from atdata.atmosphere import (
    Atmosphere,
    AtmosphereIndex,
    AtmosphereIndexEntry,
    SchemaPublisher,
    SchemaLoader,
    DatasetPublisher,
    DatasetLoader,
    LensPublisher,
    LensLoader,
    AtUri,
    LexSchemaRecord,
    LexDatasetRecord,
    LexLensRecord,
    LexCodeReference,
    JsonSchemaFormat,
    StorageHttp,
    StorageS3,
    StorageBlobs,
    HttpShardEntry,
    S3ShardEntry,
    ShardChecksum,
    BlobEntry,
    DatasetMetadata,
)
from atdata.atmosphere._lexicon_types import storage_from_record
from atdata.atmosphere._types import LEXICON_NAMESPACE


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
    """Create an authenticated Atmosphere with mocked backend."""
    client = Atmosphere(_client=mock_atproto_client)
    client._login("test.bsky.social", "test-password")
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
        original = "at://did:plc:test123/ac.foundation.dataset.schema/xyz789"
        uri = AtUri.parse(original)
        assert str(uri) == original

    def test_parse_atdata_namespace(self):
        """Parse URIs in the atdata namespace."""
        uri = AtUri.parse(f"at://did:plc:abc/{LEXICON_NAMESPACE}.schema/test")

        assert uri.collection == f"{LEXICON_NAMESPACE}.schema"


# =============================================================================
# Tests for _types.py - FieldType
# =============================================================================


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestFieldType:
    """Tests for deprecated FieldType shim dataclass."""

    def test_primitive_type(self):
        """Create a primitive field type."""
        from atdata.atmosphere._types import FieldType

        ft = FieldType(kind="primitive", primitive="str")

        assert ft.kind == "primitive"
        assert ft.primitive == "str"
        assert ft.dtype is None
        assert ft.shape is None

    def test_ndarray_type(self):
        """Create an ndarray field type."""
        from atdata.atmosphere._types import FieldType

        ft = FieldType(kind="ndarray", dtype="float32", shape=[224, 224, 3])

        assert ft.kind == "ndarray"
        assert ft.dtype == "float32"
        assert ft.shape == [224, 224, 3]

    def test_ref_type(self):
        """Create a reference field type."""
        from atdata.atmosphere._types import FieldType

        ft = FieldType(kind="ref", ref="at://did:plc:abc/collection/key")

        assert ft.kind == "ref"
        assert ft.ref == "at://did:plc:abc/collection/key"

    def test_array_type(self):
        """Create an array field type with items."""
        from atdata.atmosphere._types import FieldType

        items = FieldType(kind="primitive", primitive="str")
        ft = FieldType(kind="array", items=items)

        assert ft.kind == "array"
        assert ft.items is not None
        assert ft.items.kind == "primitive"


# =============================================================================
# Tests for _types.py - FieldDef
# =============================================================================


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestFieldDef:
    """Tests for deprecated FieldDef shim dataclass."""

    def test_required_field(self):
        """Create a required field definition."""
        from atdata.atmosphere._types import FieldType, FieldDef

        fd = FieldDef(
            name="test_field",
            field_type=FieldType(kind="primitive", primitive="str"),
            optional=False,
        )

        assert fd.name == "test_field"
        assert fd.optional is False

    def test_optional_field(self):
        """Create an optional field definition."""
        from atdata.atmosphere._types import FieldType, FieldDef

        fd = FieldDef(
            name="optional_field",
            field_type=FieldType(kind="primitive", primitive="int"),
            optional=True,
        )

        assert fd.optional is True

    def test_field_with_description(self):
        """Create a field with description."""
        from atdata.atmosphere._types import FieldType, FieldDef

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


class TestLexSchemaRecord:
    """Tests for LexSchemaRecord dataclass and to_record()."""

    def test_to_record_basic(self):
        """Convert a basic schema record to dict."""
        schema = LexSchemaRecord(
            name="TestSchema",
            version="1.0.0",
            schema_type="jsonSchema",
            schema=JsonSchemaFormat(
                schema_body={
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "field1": {"type": "string"},
                    },
                    "required": ["field1"],
                },
            ),
        )

        record = schema.to_record()

        assert record["$type"] == f"{LEXICON_NAMESPACE}.schema"
        assert record["name"] == "TestSchema"
        assert record["version"] == "1.0.0"
        assert record["schemaType"] == "jsonSchema"
        assert "field1" in record["schema"]["properties"]
        assert "createdAt" in record

    def test_to_record_with_description(self):
        """Convert schema record with description."""
        schema = LexSchemaRecord(
            name="DescribedSchema",
            version="2.0.0",
            schema_type="jsonSchema",
            schema=JsonSchemaFormat(
                schema_body={
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {},
                },
            ),
            description="A schema with description",
        )

        record = schema.to_record()

        assert record["description"] == "A schema with description"

    def test_to_record_with_metadata(self):
        """Convert schema record with metadata."""
        schema = LexSchemaRecord(
            name="MetaSchema",
            version="1.0.0",
            schema_type="jsonSchema",
            schema=JsonSchemaFormat(
                schema_body={
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {},
                },
            ),
            metadata={"author": "test", "tags": ["demo"]},
        )

        record = schema.to_record()

        assert record["metadata"] == {"author": "test", "tags": ["demo"]}

    def test_to_record_json_schema_properties(self):
        """Verify JSON Schema property serialization in to_record()."""
        schema = LexSchemaRecord(
            name="TypesSchema",
            version="1.0.0",
            schema_type="jsonSchema",
            schema=JsonSchemaFormat(
                schema_body={
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "primitive_field": {"type": "integer"},
                        "array_field": {
                            "$ref": "https://foundation.ac/schemas/atdata-ndarray-bytes/1.0.0#/$defs/ndarray"
                        },
                    },
                    "required": ["primitive_field"],
                },
                array_format_versions={"ndarrayBytes": "1.0.0"},
            ),
        )

        record = schema.to_record()

        # Check primitive field in JSON Schema properties
        props = record["schema"]["properties"]
        assert props["primitive_field"]["type"] == "integer"

        # Check ndarray field uses $ref
        assert "$ref" in props["array_field"]

        # Check required list (array_field is optional since not in required)
        assert "primitive_field" in record["schema"]["required"]
        assert "array_field" not in record["schema"]["required"]

        # Check array format versions
        assert record["schema"]["arrayFormatVersions"] == {"ndarrayBytes": "1.0.0"}


# =============================================================================
# Tests for _types.py - StorageLocation
# =============================================================================


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestStorageLocation:
    """Tests for deprecated StorageLocation shim dataclass."""

    def test_external_storage(self):
        """Create external URL storage location."""
        from atdata.atmosphere._types import StorageLocation

        storage = StorageLocation(
            kind="external",
            urls=["s3://bucket/data-{000000..000009}.tar"],
        )

        assert storage.kind == "external"
        assert storage.urls == ["s3://bucket/data-{000000..000009}.tar"]
        assert storage.blob_refs is None

    def test_blob_storage(self):
        """Create ATProto blob storage location."""
        from atdata.atmosphere._types import StorageLocation

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


class TestLexDatasetRecord:
    """Tests for LexDatasetRecord dataclass and to_record()."""

    def test_to_record_http_storage(self):
        """Convert dataset record with HTTP storage."""
        dataset = LexDatasetRecord(
            name="TestDataset",
            schema_ref="at://did:plc:abc/ac.foundation.dataset.schema/xyz",
            storage=StorageHttp(
                shards=[
                    HttpShardEntry(
                        url="s3://bucket/data.tar",
                        checksum=ShardChecksum(algorithm="none", digest=""),
                    ),
                ],
            ),
        )

        record = dataset.to_record()

        assert record["$type"] == f"{LEXICON_NAMESPACE}.record"
        assert record["name"] == "TestDataset"
        assert (
            record["schemaRef"] == "at://did:plc:abc/ac.foundation.dataset.schema/xyz"
        )
        assert record["storage"]["$type"] == f"{LEXICON_NAMESPACE}.storageHttp"
        assert record["storage"]["shards"][0]["url"] == "s3://bucket/data.tar"

    def test_to_record_blob_storage(self):
        """Convert dataset record with blob storage."""
        dataset = LexDatasetRecord(
            name="BlobDataset",
            schema_ref="at://did:plc:abc/collection/key",
            storage=StorageBlobs(
                blobs=[
                    BlobEntry(
                        blob={
                            "$type": "blob",
                            "ref": {"$link": "bafytest"},
                            "mimeType": "application/x-tar",
                            "size": 1024,
                        },
                        checksum=ShardChecksum(algorithm="sha256", digest="abc123"),
                    ),
                ],
            ),
        )

        record = dataset.to_record()

        assert record["storage"]["$type"] == f"{LEXICON_NAMESPACE}.storageBlobs"
        assert record["storage"]["blobs"][0]["blob"]["ref"]["$link"] == "bafytest"

    def test_to_record_with_tags_and_license(self):
        """Convert dataset record with tags and license."""
        dataset = LexDatasetRecord(
            name="TaggedDataset",
            schema_ref="at://did:plc:abc/collection/key",
            storage=StorageHttp(shards=[]),
            tags=["ml", "vision", "demo"],
            license="MIT",
        )

        record = dataset.to_record()

        assert record["tags"] == ["ml", "vision", "demo"]
        assert record["license"] == "MIT"

    def test_to_record_with_metadata(self):
        """Convert dataset record with typed metadata to structured JSON."""
        meta = DatasetMetadata(split="train", custom={"size": 1000})
        dataset = LexDatasetRecord(
            name="MetaDataset",
            schema_ref="at://did:plc:abc/collection/key",
            storage=StorageHttp(shards=[]),
            metadata=meta,
        )

        record = dataset.to_record()

        # metadata is now a plain JSON object, not $bytes
        assert isinstance(record["metadata"], dict)
        assert "$bytes" not in record["metadata"]
        assert record["metadata"]["split"] == "train"
        assert record["metadata"]["custom"] == {"size": 1000}

    def test_metadata_roundtrip(self):
        """Metadata survives to_record() -> from_record() roundtrip."""
        meta = DatasetMetadata(
            split="train",
            version="2.0",
            source_uri="https://example.com/raw",
            custom={"size": 1000},
        )
        dataset = LexDatasetRecord(
            name="MetaDataset",
            schema_ref="at://did:plc:abc/collection/key",
            storage=StorageHttp(shards=[]),
            metadata=meta,
        )

        record = dataset.to_record()
        restored = LexDatasetRecord.from_record(record)

        assert restored.metadata is not None
        assert restored.metadata.split == "train"
        assert restored.metadata.version == "2.0"
        assert restored.metadata.source_uri == "https://example.com/raw"
        assert restored.metadata.custom == {"size": 1000}

    def test_metadata_legacy_bytes_roundtrip(self):
        """Legacy msgpack $bytes metadata is decoded into DatasetMetadata."""
        import base64
        import msgpack

        legacy_dict = {"size": 1000, "split": "train"}
        legacy_bytes = msgpack.packb(legacy_dict)
        record_dict = {
            "$type": "ac.foundation.dataset.record",
            "name": "LegacyDataset",
            "schemaRef": "at://did:plc:abc/collection/key",
            "storage": {"$type": "ac.foundation.dataset.storageHttp", "shards": []},
            "createdAt": "2025-01-01T00:00:00+00:00",
            "metadata": {"$bytes": base64.b64encode(legacy_bytes).decode("ascii")},
        }

        restored = LexDatasetRecord.from_record(record_dict)
        assert restored.metadata is not None
        assert restored.metadata.split == "train"
        assert restored.metadata.custom == {"size": 1000}

    def test_metadata_legacy_raw_bytes(self):
        """Legacy raw msgpack bytes metadata is decoded into DatasetMetadata."""
        import msgpack

        legacy_dict = {"split": "test", "version": "1.0"}
        legacy_bytes = msgpack.packb(legacy_dict)
        record_dict = {
            "$type": "ac.foundation.dataset.record",
            "name": "LegacyDataset",
            "schemaRef": "at://did:plc:abc/collection/key",
            "storage": {"$type": "ac.foundation.dataset.storageHttp", "shards": []},
            "createdAt": "2025-01-01T00:00:00+00:00",
            "metadata": legacy_bytes,
        }

        restored = LexDatasetRecord.from_record(record_dict)
        assert restored.metadata is not None
        assert restored.metadata.split == "test"
        assert restored.metadata.version == "1.0"


# =============================================================================
# Tests for _lexicon_types.py - DatasetMetadata
# =============================================================================


class TestDatasetMetadata:
    """Tests for DatasetMetadata dataclass."""

    def test_empty_metadata_to_record(self):
        """All-None DatasetMetadata serializes to empty dict."""
        meta = DatasetMetadata()
        assert meta.to_record() == {}

    def test_empty_metadata_roundtrip(self):
        """Empty DatasetMetadata survives to_record -> from_record."""
        meta = DatasetMetadata()
        restored = DatasetMetadata.from_record(meta.to_record())
        assert restored == meta

    def test_all_fields_roundtrip(self):
        """DatasetMetadata with every field set survives roundtrip."""
        meta = DatasetMetadata(
            source_uri="https://example.com/raw",
            created_by="pipeline-v3",
            version="1.2.3",
            processing_steps=["crop", "normalize"],
            split="validation",
            custom={"epochs": 5, "model": "vit"},
        )
        record = meta.to_record()
        restored = DatasetMetadata.from_record(record)
        assert restored == meta

    def test_to_dict_merges_custom(self):
        """to_dict flattens custom keys into top-level dict."""
        meta = DatasetMetadata(split="train", custom={"lr": 0.01})
        d = meta.to_dict()
        assert d == {"split": "train", "lr": 0.01}

    def test_to_dict_custom_does_not_overwrite_known(self):
        """Custom keys with same name as known fields don't clobber them."""
        meta = DatasetMetadata(split="train", custom={"split": "WRONG"})
        d = meta.to_dict()
        assert d["split"] == "train"

    def test_from_dict_unknown_keys_go_to_custom(self):
        """Keys not matching known fields end up in custom."""
        meta = DatasetMetadata.from_dict({"split": "test", "lr": 0.001, "seed": 42})
        assert meta.split == "test"
        assert meta.custom == {"lr": 0.001, "seed": 42}

    def test_from_dict_empty_string_preserved(self):
        """Empty string values for known fields are preserved, not treated as None."""
        meta = DatasetMetadata.from_dict({"sourceUri": "", "version": "1.0"})
        assert meta.source_uri == ""
        assert meta.version == "1.0"

    def test_from_dict_empty_list_preserved(self):
        """Empty list for processingSteps is preserved, not treated as None."""
        meta = DatasetMetadata.from_dict({"processingSteps": []})
        assert meta.processing_steps == []

    def test_from_dict_camel_takes_precedence(self):
        """When both camelCase and snake_case keys exist, camelCase wins."""
        meta = DatasetMetadata.from_dict(
            {
                "sourceUri": "camel",
                "source_uri": "snake",
            }
        )
        assert meta.source_uri == "camel"

    def test_from_dict_explicit_custom_merged(self):
        """Explicit 'custom' key is merged with auto-detected custom keys."""
        meta = DatasetMetadata.from_dict(
            {
                "custom": {"a": 1},
                "unknown_key": 2,
            }
        )
        assert meta.custom == {"a": 1, "unknown_key": 2}

    def test_to_dict_from_dict_roundtrip(self):
        """to_dict -> from_dict preserves all data."""
        meta = DatasetMetadata(
            source_uri="s3://bucket/path",
            split="train",
            custom={"extra": True},
        )
        d = meta.to_dict()
        restored = DatasetMetadata.from_dict(d)
        assert restored.source_uri == meta.source_uri
        assert restored.split == meta.split
        assert restored.custom == {"extra": True}

    def test_metadata_none_omitted_from_record(self):
        """LexDatasetRecord with metadata=None omits metadata key."""
        rec = LexDatasetRecord(
            name="NoMeta",
            schema_ref="at://schema",
            storage=StorageHttp(shards=[]),
        )
        record = rec.to_record()
        assert "metadata" not in record

    def test_empty_processing_steps_serialized(self):
        """processing_steps=[] is serialized (not omitted like None)."""
        meta = DatasetMetadata(processing_steps=[])
        record = meta.to_record()
        assert record["processingSteps"] == []


# =============================================================================
# Tests for _lexicon_types.py - storage_from_record roundtrip
# =============================================================================


class TestStorageFromRecord:
    """Tests for storage_from_record() union deserialization."""

    def test_http_roundtrip(self):
        """StorageHttp survives to_record → storage_from_record roundtrip."""
        original = StorageHttp(
            shards=[
                HttpShardEntry(
                    url="https://example.com/shard-000000.tar",
                    checksum=ShardChecksum("sha256", "abc123"),
                ),
                HttpShardEntry(
                    url="https://example.com/shard-000001.tar",
                    checksum=ShardChecksum("sha256", "def456"),
                ),
            ],
        )
        record = original.to_record()
        restored = storage_from_record(record)

        assert isinstance(restored, StorageHttp)
        assert len(restored.shards) == 2
        assert restored.shards[0].url == "https://example.com/shard-000000.tar"
        assert restored.shards[0].checksum.algorithm == "sha256"
        assert restored.shards[0].checksum.digest == "abc123"
        assert restored.shards[1].url == "https://example.com/shard-000001.tar"

    def test_s3_roundtrip(self):
        """StorageS3 survives to_record → storage_from_record roundtrip."""
        original = StorageS3(
            bucket="my-bucket",
            shards=[
                S3ShardEntry(
                    key="data/shard-000000.tar",
                    checksum=ShardChecksum("sha256", "aaa"),
                ),
            ],
            region="us-east-1",
            endpoint="https://s3.example.com",
        )
        record = original.to_record()
        restored = storage_from_record(record)

        assert isinstance(restored, StorageS3)
        assert restored.bucket == "my-bucket"
        assert restored.region == "us-east-1"
        assert restored.endpoint == "https://s3.example.com"
        assert len(restored.shards) == 1
        assert restored.shards[0].key == "data/shard-000000.tar"

    def test_s3_roundtrip_optional_fields_absent(self):
        """StorageS3 roundtrip with optional region/endpoint omitted."""
        original = StorageS3(
            bucket="minimal-bucket",
            shards=[
                S3ShardEntry(
                    key="shard.tar",
                    checksum=ShardChecksum("none", ""),
                ),
            ],
        )
        record = original.to_record()
        restored = storage_from_record(record)

        assert isinstance(restored, StorageS3)
        assert restored.region is None
        assert restored.endpoint is None

    def test_blobs_roundtrip(self):
        """StorageBlobs survives to_record → storage_from_record roundtrip."""
        original = StorageBlobs(
            blobs=[
                BlobEntry(
                    blob={
                        "$type": "blob",
                        "ref": {"$link": "bafytest123"},
                        "mimeType": "application/x-tar",
                        "size": 2048,
                    },
                    checksum=ShardChecksum("sha256", "deadbeef"),
                ),
            ],
        )
        record = original.to_record()
        restored = storage_from_record(record)

        assert isinstance(restored, StorageBlobs)
        assert len(restored.blobs) == 1
        assert restored.blobs[0].blob["ref"]["$link"] == "bafytest123"
        assert restored.blobs[0].checksum is not None
        assert restored.blobs[0].checksum.digest == "deadbeef"

    def test_blobs_roundtrip_no_checksum(self):
        """StorageBlobs roundtrip when BlobEntry has no checksum."""
        original = StorageBlobs(
            blobs=[
                BlobEntry(
                    blob={
                        "$type": "blob",
                        "ref": {"$link": "bafynocsum"},
                        "mimeType": "application/x-tar",
                        "size": 512,
                    },
                ),
            ],
        )
        record = original.to_record()
        restored = storage_from_record(record)

        assert isinstance(restored, StorageBlobs)
        assert restored.blobs[0].checksum is None

    def test_legacy_storage_external(self):
        """Legacy storageExternal format is converted to StorageHttp."""
        legacy_record = {
            "$type": f"{LEXICON_NAMESPACE}.storageExternal",
            "urls": [
                "s3://bucket/data-000000.tar",
                "s3://bucket/data-000001.tar",
            ],
        }
        restored = storage_from_record(legacy_record)

        assert isinstance(restored, StorageHttp)
        assert len(restored.shards) == 2
        assert restored.shards[0].url == "s3://bucket/data-000000.tar"
        assert restored.shards[0].checksum.algorithm == "none"
        assert restored.shards[1].url == "s3://bucket/data-000001.tar"

    def test_unknown_type_raises(self):
        """Unknown $type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown storage type"):
            storage_from_record({"$type": "com.example.unknownStorage"})

    def test_missing_type_raises(self):
        """Missing $type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown storage type"):
            storage_from_record({"urls": ["http://example.com/data.tar"]})


# =============================================================================
# Tests for _types.py - LensRecord
# =============================================================================


class TestLexLensRecord:
    """Tests for LexLensRecord dataclass and to_record()."""

    def test_to_record_basic(self):
        """Convert basic lens record with required code references."""
        lens = LexLensRecord(
            name="TestLens",
            source_schema="at://did:plc:abc/collection/source",
            target_schema="at://did:plc:abc/collection/target",
            getter_code=LexCodeReference(
                repository="https://github.com/user/repo",
                commit="abc123def456",
                path="module.lenses:getter_func",
            ),
            putter_code=LexCodeReference(
                repository="https://github.com/user/repo",
                commit="abc123def456",
                path="module.lenses:putter_func",
            ),
        )

        record = lens.to_record()

        assert record["$type"] == f"{LEXICON_NAMESPACE}.lens"
        assert record["name"] == "TestLens"
        assert record["sourceSchema"] == "at://did:plc:abc/collection/source"
        assert record["targetSchema"] == "at://did:plc:abc/collection/target"
        assert "createdAt" in record

    def test_to_record_with_description(self):
        """Convert lens record with description."""
        lens = LexLensRecord(
            name="DescribedLens",
            source_schema="at://a",
            target_schema="at://b",
            getter_code=LexCodeReference(
                repository="https://github.com/user/repo",
                commit="abc123",
                path="mod:getter",
            ),
            putter_code=LexCodeReference(
                repository="https://github.com/user/repo",
                commit="abc123",
                path="mod:putter",
            ),
            description="Transforms A to B",
        )

        record = lens.to_record()

        assert record["description"] == "Transforms A to B"

    def test_to_record_with_code_references(self):
        """Convert lens record with code references."""
        lens = LexLensRecord(
            name="CodeLens",
            source_schema="at://a",
            target_schema="at://b",
            getter_code=LexCodeReference(
                repository="https://github.com/user/repo",
                commit="abc123def456",
                path="module.lenses:getter_func",
            ),
            putter_code=LexCodeReference(
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
# Tests for client.py - Atmosphere
# =============================================================================


class TestAtmosphere:
    """Tests for Atmosphere."""

    def test_init_default(self):
        """Initialize client with defaults."""
        with patch("atdata.atmosphere.client._get_atproto_client_class") as mock_get:
            mock_class = Mock()
            mock_get.return_value = mock_class

            client = Atmosphere()

            mock_class.assert_called_once_with()
            assert not client.is_authenticated

    def test_init_with_base_url(self):
        """Initialize client with custom base URL."""
        with patch("atdata.atmosphere.client._get_atproto_client_class") as mock_get:
            mock_class = Mock()
            mock_get.return_value = mock_class

            Atmosphere(base_url="https://custom.pds.example")

            mock_class.assert_called_once_with(base_url="https://custom.pds.example")

    def test_init_with_mock_client(self, mock_atproto_client):
        """Initialize with pre-configured mock client."""
        client = Atmosphere(_client=mock_atproto_client)

        assert client._client is mock_atproto_client

    def test_login_success(self, mock_atproto_client):
        """Successful login sets session."""
        client = Atmosphere(_client=mock_atproto_client)

        client._login("test.bsky.social", "password123")

        assert client.is_authenticated
        assert client.did == "did:plc:test123456789"
        assert client.handle == "test.bsky.social"
        mock_atproto_client.login.assert_called_once_with(
            "test.bsky.social", "password123"
        )

    def test_login_with_session(self, mock_atproto_client):
        """Login with exported session string."""
        client = Atmosphere(_client=mock_atproto_client)

        client._login_with_session("test-session-string")

        assert client.is_authenticated
        mock_atproto_client.login.assert_called_once_with(
            session_string="test-session-string"
        )

    def test_export_session(self, authenticated_client, mock_atproto_client):
        """Export session string."""
        session = authenticated_client.export_session()

        assert session == "test-session-string"
        mock_atproto_client.export_session_string.assert_called_once()

    def test_export_session_not_authenticated(self, mock_atproto_client):
        """Export session raises when not authenticated."""
        client = Atmosphere(_client=mock_atproto_client)

        with pytest.raises(ValueError, match="Not authenticated"):
            client.export_session()

    def test_did_not_authenticated(self, mock_atproto_client):
        """Accessing did raises when not authenticated."""
        client = Atmosphere(_client=mock_atproto_client)

        with pytest.raises(ValueError, match="Not authenticated"):
            _ = client.did

    def test_handle_not_authenticated(self, mock_atproto_client):
        """Accessing handle raises when not authenticated."""
        client = Atmosphere(_client=mock_atproto_client)

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
        client = Atmosphere(_client=mock_atproto_client)

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

    def test_get_record_with_aturi_object(
        self, authenticated_client, mock_atproto_client
    ):
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

        mock_atproto_client.com.atproto.repo.delete_record.assert_called_once_with(
            data={
                "repo": "did:plc:test123456789",
                "collection": "collection",
                "rkey": "key",
            }
        )

    def test_upload_blob(self, authenticated_client, mock_atproto_client):
        """Upload blob returns proper blob reference dict."""
        mock_blob_ref = Mock()
        mock_blob_ref.ref = Mock(link="bafkreitest123")
        mock_blob_ref.mime_type = "application/x-tar"
        mock_blob_ref.size = 1024

        mock_response = Mock()
        mock_response.blob = mock_blob_ref
        mock_atproto_client.com.atproto.repo.upload_blob.return_value = mock_response

        result = authenticated_client.upload_blob(
            b"test data", mime_type="application/x-tar"
        )

        assert result["$type"] == "blob"
        assert result["ref"]["$link"] == "bafkreitest123"
        assert result["mimeType"] == "application/x-tar"
        assert result["size"] == 1024

    def test_upload_blob_not_authenticated(self, mock_atproto_client):
        """Upload blob raises when not authenticated."""
        client = Atmosphere(_client=mock_atproto_client)

        with pytest.raises(ValueError, match="must be authenticated"):
            client.upload_blob(b"data")

    def test_get_blob(self, authenticated_client):
        """Get blob fetches from resolved PDS endpoint."""
        with patch("requests.get") as mock_get:
            mock_did_response = Mock()
            mock_did_response.json.return_value = {
                "service": [
                    {
                        "type": "AtprotoPersonalDataServer",
                        "serviceEndpoint": "https://pds.example.com",
                    }
                ]
            }
            mock_did_response.raise_for_status = Mock()

            mock_blob_response = Mock()
            mock_blob_response.content = b"blob data here"
            mock_blob_response.raise_for_status = Mock()

            mock_get.side_effect = [mock_did_response, mock_blob_response]

            result = authenticated_client.get_blob("did:plc:abc123", "bafkreitest")

            assert result == b"blob data here"
            assert mock_get.call_count == 2

    def test_get_blob_pds_not_found(self, authenticated_client):
        """Get blob raises when PDS cannot be resolved."""
        import requests as req_module

        with patch("requests.get") as mock_get:
            mock_get.side_effect = req_module.RequestException("Network error")

            with pytest.raises(ValueError, match="Could not resolve PDS"):
                authenticated_client.get_blob("did:plc:unknown", "cid123")

    def test_get_blob_url(self, authenticated_client):
        """Get blob URL constructs proper URL."""
        with patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {
                "service": [
                    {
                        "type": "AtprotoPersonalDataServer",
                        "serviceEndpoint": "https://pds.example.com",
                    }
                ]
            }
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            url = authenticated_client.get_blob_url("did:plc:abc", "bafkreitest")

            assert (
                url
                == "https://pds.example.com/xrpc/com.atproto.sync.getBlob?did=did:plc:abc&cid=bafkreitest"
            )

    def test_get_blob_url_pds_not_found(self, authenticated_client):
        """Get blob URL raises when PDS cannot be resolved."""
        import requests as req_module

        with patch("requests.get") as mock_get:
            mock_get.side_effect = req_module.RequestException("Network error")

            with pytest.raises(ValueError, match="Could not resolve PDS"):
                authenticated_client.get_blob_url("did:plc:unknown", "cid123")

    def test_resolve_pds_endpoint_did_web(self):
        """PDS resolution raises ValueError for did:web (not implemented)."""
        from atdata.atmosphere import _resolve_pds_endpoint

        with pytest.raises(ValueError, match="Could not resolve PDS"):
            _resolve_pds_endpoint("did:web:example.com")

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

        authenticated_client.list_schemas()

        call_args = mock_atproto_client.com.atproto.repo.list_records.call_args
        assert f"{LEXICON_NAMESPACE}.schema" in str(call_args)


# =============================================================================
# Tests for Atmosphere client coverage gaps
# =============================================================================


class TestAtmosphereClientEdgeCases:
    """Tests for uncovered client.py paths: model conversion, swap_commit, etc."""

    def test_put_record_with_swap_commit(
        self, authenticated_client, mock_atproto_client
    ):
        """put_record passes swapCommit when provided."""
        mock_response = Mock()
        mock_response.uri = f"at://did:plc:test/{LEXICON_NAMESPACE}.record/abc"
        mock_atproto_client.com.atproto.repo.put_record.return_value = mock_response

        authenticated_client.put_record(
            collection=f"{LEXICON_NAMESPACE}.record",
            rkey="abc",
            record={"name": "test"},
            swap_commit="bafyswap123",
        )

        call_data = mock_atproto_client.com.atproto.repo.put_record.call_args.kwargs[
            "data"
        ]
        assert call_data["swapCommit"] == "bafyswap123"

    def test_delete_record_with_swap_commit(
        self, authenticated_client, mock_atproto_client
    ):
        """delete_record passes swapCommit when provided."""
        uri = f"at://did:plc:test/{LEXICON_NAMESPACE}.record/abc"
        authenticated_client.delete_record(uri, swap_commit="bafyswap456")

        call_data = mock_atproto_client.com.atproto.repo.delete_record.call_args.kwargs[
            "data"
        ]
        assert call_data["swapCommit"] == "bafyswap456"

    def test_get_record_model_dump_fallback(
        self, authenticated_client, mock_atproto_client
    ):
        """get_record uses model_dump() when to_dict() is unavailable."""
        mock_value = Mock(spec=[])  # no to_dict, not a dict
        mock_value.model_dump = Mock(return_value={"name": "from_model_dump"})
        mock_response = Mock()
        mock_response.value = mock_value
        mock_atproto_client.com.atproto.repo.get_record.return_value = mock_response

        result = authenticated_client.get_record(
            f"at://did:plc:test/{LEXICON_NAMESPACE}.record/abc"
        )
        assert result == {"name": "from_model_dump"}

    def test_get_record_dict_fallback(self, authenticated_client, mock_atproto_client):
        """get_record uses __dict__ when no to_dict() or model_dump()."""

        class SimpleObj:
            def __init__(self):
                self.name = "from_dict"

        mock_response = Mock()
        mock_response.value = SimpleObj()
        mock_atproto_client.com.atproto.repo.get_record.return_value = mock_response

        result = authenticated_client.get_record(
            f"at://did:plc:test/{LEXICON_NAMESPACE}.record/abc"
        )
        assert result["name"] == "from_dict"

    def test_list_records_model_dump_fallback(
        self, authenticated_client, mock_atproto_client
    ):
        """list_records uses model_dump() for record values."""
        mock_value = Mock(spec=[])
        mock_value.model_dump = Mock(return_value={"name": "from_model_dump"})
        mock_record = Mock()
        mock_record.value = mock_value

        mock_response = Mock()
        mock_response.records = [mock_record]
        mock_response.cursor = None
        mock_atproto_client.com.atproto.repo.list_records.return_value = mock_response

        records, cursor = authenticated_client.list_records(
            f"{LEXICON_NAMESPACE}.schema"
        )
        assert records == [{"name": "from_model_dump"}]

    def test_list_records_dict_fallback(
        self, authenticated_client, mock_atproto_client
    ):
        """list_records uses __dict__ when no to_dict() or model_dump()."""

        class SimpleObj:
            def __init__(self):
                self.data = "test"

        mock_record = Mock()
        mock_record.value = SimpleObj()

        mock_response = Mock()
        mock_response.records = [mock_record]
        mock_response.cursor = None
        mock_atproto_client.com.atproto.repo.list_records.return_value = mock_response

        records, _ = authenticated_client.list_records(f"{LEXICON_NAMESPACE}.schema")
        assert records[0]["data"] == "test"

    def test_list_records_raw_value_fallback(
        self, authenticated_client, mock_atproto_client
    ):
        """list_records passes through raw value when no conversion available."""
        mock_record = Mock()
        mock_record.value = "raw-string-value"

        mock_response = Mock()
        mock_response.records = [mock_record]
        mock_response.cursor = None
        mock_atproto_client.com.atproto.repo.list_records.return_value = mock_response

        records, _ = authenticated_client.list_records(f"{LEXICON_NAMESPACE}.schema")
        assert records == ["raw-string-value"]

    def test_list_labels(self, authenticated_client, mock_atproto_client):
        """list_labels delegates to list_records with label collection."""
        mock_record = Mock()
        mock_record.value = Mock()
        mock_record.value.to_dict = Mock(
            return_value={"$type": f"{LEXICON_NAMESPACE}.label", "name": "mnist"}
        )

        mock_response = Mock()
        mock_response.records = [mock_record]
        mock_response.cursor = None
        mock_atproto_client.com.atproto.repo.list_records.return_value = mock_response

        labels = authenticated_client.list_labels()
        assert len(labels) == 1
        assert labels[0]["name"] == "mnist"

    def test_login_classmethod(self, mock_atproto_client):
        """login() classmethod creates and authenticates instance."""
        with patch.object(Atmosphere, "_login") as mock_login:
            atmo = Atmosphere.login("alice.bsky.social", "password123")
            mock_login.assert_called_once_with("alice.bsky.social", "password123")
            assert isinstance(atmo, Atmosphere)

    def test_from_session_classmethod(self, mock_atproto_client):
        """from_session() classmethod creates instance from session string."""
        with patch.object(Atmosphere, "_login_with_session") as mock_session:
            atmo = Atmosphere.from_session("session-string-abc")
            mock_session.assert_called_once_with("session-string-abc")
            assert isinstance(atmo, Atmosphere)


# =============================================================================
# Tests for schema.py - SchemaPublisher
# =============================================================================


class TestSchemaPublisher:
    """Tests for SchemaPublisher."""

    def test_publish_basic_sample(self, authenticated_client, mock_atproto_client):
        """Publish a basic sample type schema."""
        mock_response = Mock()
        mock_response.uri = f"at://did:plc:test123456789/{LEXICON_NAMESPACE}.schema/abc"
        mock_atproto_client.com.atproto.repo.create_record.return_value = mock_response

        publisher = SchemaPublisher(authenticated_client)
        uri = publisher.publish(BasicSample, version="1.0.0")

        assert isinstance(uri, AtUri)
        assert uri.collection == f"{LEXICON_NAMESPACE}.schema"

        # Verify the record structure
        call_args = mock_atproto_client.com.atproto.repo.create_record.call_args
        record = call_args.kwargs["data"]["record"]
        assert record["name"] == "BasicSample"
        assert record["version"] == "1.0.0"
        assert record["schemaType"] == "jsonSchema"
        assert "name" in record["schema"]["properties"]
        assert "value" in record["schema"]["properties"]

    def test_publish_with_custom_name(self, authenticated_client, mock_atproto_client):
        """Publish with custom name override."""
        mock_response = Mock()
        mock_response.uri = f"at://did:plc:test/{LEXICON_NAMESPACE}.schema/abc"
        mock_atproto_client.com.atproto.repo.create_record.return_value = mock_response

        publisher = SchemaPublisher(authenticated_client)
        publisher.publish(BasicSample, name="CustomName", version="2.0.0")

        call_args = mock_atproto_client.com.atproto.repo.create_record.call_args
        record = call_args.kwargs["data"]["record"]
        assert record["name"] == "CustomName"

    def test_publish_numpy_sample(self, authenticated_client, mock_atproto_client):
        """Publish sample type with NDArray field."""
        mock_response = Mock()
        mock_response.uri = f"at://did:plc:test/{LEXICON_NAMESPACE}.schema/abc"
        mock_atproto_client.com.atproto.repo.create_record.return_value = mock_response

        publisher = SchemaPublisher(authenticated_client)
        publisher.publish(NumpySample, version="1.0.0")

        call_args = mock_atproto_client.com.atproto.repo.create_record.call_args
        record = call_args.kwargs["data"]["record"]

        # Check the data property uses $ref for ndarray
        data_prop = record["schema"]["properties"]["data"]
        assert "$ref" in data_prop

    def test_publish_optional_fields(self, authenticated_client, mock_atproto_client):
        """Publish sample type with optional fields."""
        mock_response = Mock()
        mock_response.uri = f"at://did:plc:test/{LEXICON_NAMESPACE}.schema/abc"
        mock_atproto_client.com.atproto.repo.create_record.return_value = mock_response

        publisher = SchemaPublisher(authenticated_client)
        publisher.publish(OptionalSample, version="1.0.0")

        call_args = mock_atproto_client.com.atproto.repo.create_record.call_args
        record = call_args.kwargs["data"]["record"]

        # Check required list instead of optional flag
        required_list = record["schema"].get("required", [])
        assert "required_field" in required_list
        assert "optional_field" not in required_list
        assert "optional_array" not in required_list

    def test_publish_all_primitive_types(
        self, authenticated_client, mock_atproto_client
    ):
        """Publish sample with all primitive types."""
        mock_response = Mock()
        mock_response.uri = f"at://did:plc:test/{LEXICON_NAMESPACE}.schema/abc"
        mock_atproto_client.com.atproto.repo.create_record.return_value = mock_response

        publisher = SchemaPublisher(authenticated_client)
        publisher.publish(AllTypesSample, version="1.0.0")

        call_args = mock_atproto_client.com.atproto.repo.create_record.call_args
        record = call_args.kwargs["data"]["record"]

        # Verify each primitive type in JSON Schema format
        props = record["schema"]["properties"]
        assert props["str_field"]["type"] == "string"
        assert props["int_field"]["type"] == "integer"
        assert props["float_field"]["type"] == "number"
        assert props["bool_field"]["type"] == "boolean"
        assert props["bytes_field"]["type"] == "string"
        assert props["bytes_field"]["format"] == "byte"

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
            "$type": f"{LEXICON_NAMESPACE}.schema",
            "name": "TestSchema",
            "version": "1.0.0",
            "fields": [],
        }
        mock_atproto_client.com.atproto.repo.get_record.return_value = mock_response

        loader = SchemaLoader(authenticated_client)
        schema = loader.get(f"at://did:plc:abc/{LEXICON_NAMESPACE}.schema/xyz")

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
        schema_response.uri = f"at://did:plc:test/{LEXICON_NAMESPACE}.schema/schema123"

        # Mock for dataset creation
        dataset_response = Mock()
        dataset_response.uri = (
            f"at://did:plc:test/{LEXICON_NAMESPACE}.record/dataset456"
        )

        mock_atproto_client.com.atproto.repo.create_record.side_effect = [
            schema_response,
            dataset_response,
        ]

        # Create a mock dataset
        mock_dataset = Mock()
        mock_dataset.url = "s3://bucket/data.tar"
        mock_dataset.list_shards.return_value = ["s3://bucket/data.tar"]
        mock_dataset.sample_type = BasicSample
        mock_dataset.metadata = None

        publisher = DatasetPublisher(authenticated_client)
        publisher.publish(
            mock_dataset,
            name="AutoSchemaDataset",
            auto_publish_schema=True,
        )

        # Should have called create_record twice (schema + dataset)
        assert mock_atproto_client.com.atproto.repo.create_record.call_count == 2

    def test_publish_explicit_schema_uri(
        self, authenticated_client, mock_atproto_client
    ):
        """Publish dataset with explicit schema URI (no auto publish)."""
        mock_response = Mock()
        mock_response.uri = f"at://did:plc:test/{LEXICON_NAMESPACE}.record/abc"
        mock_atproto_client.com.atproto.repo.create_record.return_value = mock_response

        mock_dataset = Mock()
        mock_dataset.url = "s3://bucket/data.tar"
        mock_dataset.list_shards.return_value = ["s3://bucket/data.tar"]
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

    def test_publish_with_blob_refs(self, authenticated_client, mock_atproto_client):
        """Publish with pre-uploaded blob refs uses storageBlobs."""
        # Pre-uploaded blob references (as returned by Atmosphere.upload_blob)
        blob_refs = [
            {
                "$type": "blob",
                "ref": {"$link": "bafyrei_shard1"},
                "mimeType": "application/x-tar",
                "size": 4096,
            },
            {
                "$type": "blob",
                "ref": {"$link": "bafyrei_shard2"},
                "mimeType": "application/x-tar",
                "size": 8192,
            },
        ]

        mock_create_response = Mock()
        mock_create_response.uri = (
            f"at://did:plc:test/{LEXICON_NAMESPACE}.record/blobrefds"
        )
        mock_atproto_client.com.atproto.repo.create_record.return_value = (
            mock_create_response
        )

        publisher = DatasetPublisher(authenticated_client)
        uri = publisher.publish_with_blob_refs(
            blob_refs=blob_refs,
            schema_uri="at://did:plc:test/schema/xyz",
            name="BlobRefDataset",
            description="Dataset with pre-uploaded blob refs",
            tags=["blob", "test"],
        )

        assert isinstance(uri, AtUri)
        # Should NOT have uploaded any blobs (already uploaded)
        mock_atproto_client.upload_blob.assert_not_called()
        # Should have created one record
        assert mock_atproto_client.com.atproto.repo.create_record.call_count == 1

        # Verify record uses storageBlobs with embedded refs
        call_args = mock_atproto_client.com.atproto.repo.create_record.call_args
        record = call_args.kwargs["data"]["record"]
        assert record["name"] == "BlobRefDataset"
        assert "storageBlobs" in record["storage"]["$type"]
        assert len(record["storage"]["blobs"]) == 2
        assert record["storage"]["blobs"][0]["blob"]["ref"]["$link"] == "bafyrei_shard1"
        assert record["storage"]["blobs"][1]["blob"]["ref"]["$link"] == "bafyrei_shard2"

    def test_publish_with_blob_refs_with_checksums(
        self, authenticated_client, mock_atproto_client
    ):
        """Publish with blob refs attaches per-shard checksums to BlobEntry objects."""
        blob_refs = [
            {
                "$type": "blob",
                "ref": {"$link": "bafyrei_shard1"},
                "mimeType": "application/x-tar",
                "size": 4096,
            },
            {
                "$type": "blob",
                "ref": {"$link": "bafyrei_shard2"},
                "mimeType": "application/x-tar",
                "size": 8192,
            },
        ]
        checksums = [
            ShardChecksum(algorithm="sha256", digest="aabbcc"),
            ShardChecksum(algorithm="sha256", digest="ddeeff"),
        ]

        mock_create_response = Mock()
        mock_create_response.uri = (
            f"at://did:plc:test/{LEXICON_NAMESPACE}.record/blobchk"
        )
        mock_atproto_client.com.atproto.repo.create_record.return_value = (
            mock_create_response
        )

        publisher = DatasetPublisher(authenticated_client)
        uri = publisher.publish_with_blob_refs(
            blob_refs=blob_refs,
            schema_uri="at://did:plc:test/schema/xyz",
            name="BlobChecksumDataset",
            checksums=checksums,
        )

        assert isinstance(uri, AtUri)
        call_args = mock_atproto_client.com.atproto.repo.create_record.call_args
        record = call_args.kwargs["data"]["record"]
        blobs = record["storage"]["blobs"]
        assert len(blobs) == 2
        assert blobs[0]["checksum"]["algorithm"] == "sha256"
        assert blobs[0]["checksum"]["digest"] == "aabbcc"
        assert blobs[1]["checksum"]["algorithm"] == "sha256"
        assert blobs[1]["checksum"]["digest"] == "ddeeff"

    def test_publish_with_blob_refs_no_checksums_uses_placeholder(
        self, authenticated_client, mock_atproto_client
    ):
        """Publish with blob refs without checksums uses placeholder."""
        blob_refs = [
            {
                "$type": "blob",
                "ref": {"$link": "bafyrei_shard1"},
                "mimeType": "application/x-tar",
                "size": 4096,
            },
        ]

        mock_create_response = Mock()
        mock_create_response.uri = (
            f"at://did:plc:test/{LEXICON_NAMESPACE}.record/blobnocheck"
        )
        mock_atproto_client.com.atproto.repo.create_record.return_value = (
            mock_create_response
        )

        publisher = DatasetPublisher(authenticated_client)
        publisher.publish_with_blob_refs(
            blob_refs=blob_refs,
            schema_uri="at://did:plc:test/schema/xyz",
            name="NoCksumDataset",
        )

        call_args = mock_atproto_client.com.atproto.repo.create_record.call_args
        record = call_args.kwargs["data"]["record"]
        blobs = record["storage"]["blobs"]
        assert blobs[0]["checksum"]["algorithm"] == "none"
        assert blobs[0]["checksum"]["digest"] == ""

    def test_publish_with_blob_refs_checksums_length_mismatch(
        self, authenticated_client
    ):
        """Publish with blob refs raises when checksums length mismatches."""
        blob_refs = [
            {
                "$type": "blob",
                "ref": {"$link": "bafyrei_shard1"},
                "mimeType": "application/x-tar",
                "size": 4096,
            },
        ]
        checksums = [
            ShardChecksum(algorithm="sha256", digest="aabbcc"),
            ShardChecksum(algorithm="sha256", digest="ddeeff"),
        ]

        publisher = DatasetPublisher(authenticated_client)
        with pytest.raises(ValueError, match="checksums length.*must match"):
            publisher.publish_with_blob_refs(
                blob_refs=blob_refs,
                schema_uri="at://did:plc:test/schema/xyz",
                name="MismatchDataset",
                checksums=checksums,
            )

    def test_publish_with_blobs(self, authenticated_client, mock_atproto_client):
        """Publish with blob storage uploads blobs and creates record."""
        # Mock blob upload response
        mock_blob_ref = Mock()
        mock_blob_ref.ref = Mock(link="bafkreiblob123")
        mock_blob_ref.mime_type = "application/x-tar"
        mock_blob_ref.size = 2048

        mock_upload_response = Mock()
        mock_upload_response.blob = mock_blob_ref
        mock_atproto_client.com.atproto.repo.upload_blob.return_value = (
            mock_upload_response
        )

        # Mock create_record response
        mock_create_response = Mock()
        mock_create_response.uri = (
            f"at://did:plc:test/{LEXICON_NAMESPACE}.record/blobds"
        )
        mock_atproto_client.com.atproto.repo.create_record.return_value = (
            mock_create_response
        )

        publisher = DatasetPublisher(authenticated_client)
        uri = publisher.publish_with_blobs(
            blobs=[b"tar data 1", b"tar data 2"],
            schema_uri="at://did:plc:test/schema/xyz",
            name="BlobStoredDataset",
            description="Dataset stored in blobs",
            tags=["blob", "test"],
        )

        assert isinstance(uri, AtUri)
        # Should have uploaded 2 blobs
        assert mock_atproto_client.com.atproto.repo.upload_blob.call_count == 2
        # Should have created one record
        assert mock_atproto_client.com.atproto.repo.create_record.call_count == 1

        # Verify record structure
        call_args = mock_atproto_client.com.atproto.repo.create_record.call_args
        record = call_args.kwargs["data"]["record"]
        assert record["name"] == "BlobStoredDataset"
        assert "storageBlobs" in record["storage"]["$type"]

    def test_publish_with_blobs_with_metadata(
        self, authenticated_client, mock_atproto_client
    ):
        """Publish with blobs includes metadata when provided."""
        mock_blob_ref = Mock()
        mock_blob_ref.ref = Mock(link="bafkreiblob456")
        mock_blob_ref.mime_type = "application/x-tar"
        mock_blob_ref.size = 1024

        mock_upload_response = Mock()
        mock_upload_response.blob = mock_blob_ref
        mock_atproto_client.com.atproto.repo.upload_blob.return_value = (
            mock_upload_response
        )

        mock_create_response = Mock()
        mock_create_response.uri = (
            f"at://did:plc:test/{LEXICON_NAMESPACE}.record/metads"
        )
        mock_atproto_client.com.atproto.repo.create_record.return_value = (
            mock_create_response
        )

        publisher = DatasetPublisher(authenticated_client)
        publisher.publish_with_blobs(
            blobs=[b"data"],
            schema_uri="at://schema",
            name="MetaBlobDataset",
            metadata={"samples": 100, "split": "train"},
        )

        call_args = mock_atproto_client.com.atproto.repo.create_record.call_args
        record = call_args.kwargs["data"]["record"]
        assert "metadata" in record
        # metadata is now a structured JSON object, not $bytes
        assert isinstance(record["metadata"], dict)
        assert "$bytes" not in record["metadata"]
        assert record["metadata"]["split"] == "train"
        assert record["metadata"]["custom"] == {"samples": 100}


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
            "$type": f"{LEXICON_NAMESPACE}.schema",
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
                "urls": [
                    "s3://bucket/data-{000000..000009}.tar",
                    "s3://bucket/extra.tar",
                ],
            },
        }
        mock_atproto_client.com.atproto.repo.get_record.return_value = mock_response

        loader = DatasetLoader(authenticated_client)
        urls = loader.get_urls(f"at://did:plc:abc/{LEXICON_NAMESPACE}.record/xyz")

        assert len(urls) == 2
        assert "data-{000000..000009}.tar" in urls[0]

    def test_get_urls_blob_storage_error(
        self, authenticated_client, mock_atproto_client
    ):
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
        """Get metadata from dataset record (new structured format)."""
        mock_response = Mock()
        mock_response.value = {
            "$type": f"{LEXICON_NAMESPACE}.record",
            "name": "MetaDataset",
            "schemaRef": "at://schema",
            "storage": {"$type": f"{LEXICON_NAMESPACE}.storageExternal", "urls": []},
            "metadata": {"split": "train", "custom": {"samples": 10000}},
        }
        mock_atproto_client.com.atproto.repo.get_record.return_value = mock_response

        loader = DatasetLoader(authenticated_client)
        metadata = loader.get_metadata(
            f"at://did:plc:abc/{LEXICON_NAMESPACE}.record/xyz"
        )

        assert metadata["split"] == "train"
        assert metadata["samples"] == 10000

    def test_get_metadata_legacy_bytes(self, authenticated_client, mock_atproto_client):
        """Get metadata from dataset record with legacy msgpack bytes."""
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
        metadata = loader.get_metadata(
            f"at://did:plc:abc/{LEXICON_NAMESPACE}.record/xyz"
        )

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
        metadata = loader.get_metadata(
            f"at://did:plc:abc/{LEXICON_NAMESPACE}.record/xyz"
        )

        assert metadata is None

    def test_get_metadata_typed(self, authenticated_client, mock_atproto_client):
        """get_metadata_typed returns DatasetMetadata instance."""
        mock_response = Mock()
        mock_response.value = {
            "$type": f"{LEXICON_NAMESPACE}.record",
            "name": "TypedMeta",
            "schemaRef": "at://schema",
            "storage": {"$type": f"{LEXICON_NAMESPACE}.storageHttp", "shards": []},
            "createdAt": "2025-01-01T00:00:00+00:00",
            "metadata": {"split": "train", "version": "3.0", "custom": {"k": "v"}},
        }
        mock_atproto_client.com.atproto.repo.get_record.return_value = mock_response

        loader = DatasetLoader(authenticated_client)
        meta = loader.get_metadata_typed(
            f"at://did:plc:abc/{LEXICON_NAMESPACE}.record/xyz"
        )

        assert isinstance(meta, DatasetMetadata)
        assert meta.split == "train"
        assert meta.version == "3.0"
        assert meta.custom == {"k": "v"}

    def test_get_metadata_typed_none(self, authenticated_client, mock_atproto_client):
        """get_metadata_typed returns None when no metadata."""
        mock_response = Mock()
        mock_response.value = {
            "$type": f"{LEXICON_NAMESPACE}.record",
            "name": "NoMeta",
            "schemaRef": "at://schema",
            "storage": {"$type": f"{LEXICON_NAMESPACE}.storageExternal", "urls": []},
        }
        mock_atproto_client.com.atproto.repo.get_record.return_value = mock_response

        loader = DatasetLoader(authenticated_client)
        meta = loader.get_metadata_typed(
            f"at://did:plc:abc/{LEXICON_NAMESPACE}.record/xyz"
        )
        assert meta is None

    def test_publish_with_typed_metadata(
        self, authenticated_client, mock_atproto_client
    ):
        """DatasetPublisher accepts DatasetMetadata directly."""
        mock_response = Mock()
        mock_response.uri = f"at://did:plc:test/{LEXICON_NAMESPACE}.record/abc"
        mock_atproto_client.com.atproto.repo.create_record.return_value = mock_response

        publisher = DatasetPublisher(authenticated_client)
        meta = DatasetMetadata(split="train", version="2.0")
        publisher.publish_with_urls(
            urls=["https://example.com/data.tar"],
            schema_uri="at://schema",
            name="TypedMetaDataset",
            metadata=meta,
        )

        call_args = mock_atproto_client.com.atproto.repo.create_record.call_args
        record = call_args.kwargs["data"]["record"]
        assert record["metadata"]["split"] == "train"
        assert record["metadata"]["version"] == "2.0"
        assert "$bytes" not in record["metadata"]

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

    def test_get_storage_type_external(self, authenticated_client, mock_atproto_client):
        """Get storage type returns 'external' for external storage."""
        mock_response = Mock()
        mock_response.value = {
            "$type": f"{LEXICON_NAMESPACE}.record",
            "name": "ExternalDataset",
            "schemaRef": "at://schema",
            "storage": {
                "$type": f"{LEXICON_NAMESPACE}.storageExternal",
                "urls": ["s3://bucket/data.tar"],
            },
        }
        mock_atproto_client.com.atproto.repo.get_record.return_value = mock_response

        loader = DatasetLoader(authenticated_client)
        storage_type = loader.get_storage_type(
            f"at://did:plc:abc/{LEXICON_NAMESPACE}.record/xyz"
        )

        assert storage_type == "external"

    def test_get_storage_type_blobs(self, authenticated_client, mock_atproto_client):
        """Get storage type returns 'blobs' for blob storage."""
        mock_response = Mock()
        mock_response.value = {
            "$type": f"{LEXICON_NAMESPACE}.record",
            "name": "BlobDataset",
            "schemaRef": "at://schema",
            "storage": {
                "$type": f"{LEXICON_NAMESPACE}.storageBlobs",
                "blobs": [{"ref": {"$link": "bafkreitest"}}],
            },
        }
        mock_atproto_client.com.atproto.repo.get_record.return_value = mock_response

        loader = DatasetLoader(authenticated_client)
        storage_type = loader.get_storage_type(
            f"at://did:plc:abc/{LEXICON_NAMESPACE}.record/xyz"
        )

        assert storage_type == "blobs"

    def test_get_storage_type_unknown(self, authenticated_client, mock_atproto_client):
        """Get storage type raises for unknown storage type."""
        mock_response = Mock()
        mock_response.value = {
            "$type": f"{LEXICON_NAMESPACE}.record",
            "name": "UnknownStorageDataset",
            "schemaRef": "at://schema",
            "storage": {
                "$type": "some.unknown.storage",
            },
        }
        mock_atproto_client.com.atproto.repo.get_record.return_value = mock_response

        loader = DatasetLoader(authenticated_client)

        with pytest.raises(ValueError, match="Unknown storage type"):
            loader.get_storage_type(f"at://did:plc:abc/{LEXICON_NAMESPACE}.record/xyz")

    def test_get_blobs(self, authenticated_client, mock_atproto_client):
        """Get blobs returns blob entry dicts from storage."""
        blob_entries = [
            {
                "blob": {
                    "$type": "blob",
                    "ref": {"$link": "bafkreitest1"},
                    "mimeType": "application/x-tar",
                    "size": 1024,
                },
                "checksum": {"algorithm": "sha256", "digest": "abc123"},
            },
            {
                "blob": {
                    "$type": "blob",
                    "ref": {"$link": "bafkreitest2"},
                    "mimeType": "application/x-tar",
                    "size": 2048,
                },
                "checksum": {"algorithm": "sha256", "digest": "def456"},
            },
        ]
        mock_response = Mock()
        mock_response.value = {
            "$type": f"{LEXICON_NAMESPACE}.record",
            "name": "BlobDataset",
            "schemaRef": "at://schema",
            "storage": {
                "$type": f"{LEXICON_NAMESPACE}.storageBlobs",
                "blobs": blob_entries,
            },
        }
        mock_atproto_client.com.atproto.repo.get_record.return_value = mock_response

        loader = DatasetLoader(authenticated_client)
        blobs = loader.get_blobs(f"at://did:plc:abc/{LEXICON_NAMESPACE}.record/xyz")

        assert len(blobs) == 2
        assert blobs[0]["blob"]["ref"]["$link"] == "bafkreitest1"
        assert blobs[1]["blob"]["ref"]["$link"] == "bafkreitest2"

    def test_get_blobs_external_storage_error(
        self, authenticated_client, mock_atproto_client
    ):
        """Get blobs raises for external URL storage datasets."""
        mock_response = Mock()
        mock_response.value = {
            "$type": f"{LEXICON_NAMESPACE}.record",
            "name": "ExternalDataset",
            "schemaRef": "at://schema",
            "storage": {
                "$type": f"{LEXICON_NAMESPACE}.storageExternal",
                "urls": ["s3://bucket/data.tar"],
            },
        }
        mock_atproto_client.com.atproto.repo.get_record.return_value = mock_response

        loader = DatasetLoader(authenticated_client)

        with pytest.raises(ValueError, match="does not use blob storage"):
            loader.get_blobs(f"at://did:plc:abc/{LEXICON_NAMESPACE}.record/xyz")

    def test_get_blobs_unknown_storage_error(
        self, authenticated_client, mock_atproto_client
    ):
        """Get blobs raises for unknown storage type."""
        mock_response = Mock()
        mock_response.value = {
            "$type": f"{LEXICON_NAMESPACE}.record",
            "name": "UnknownDataset",
            "schemaRef": "at://schema",
            "storage": {
                "$type": "some.unknown.storage",
            },
        }
        mock_atproto_client.com.atproto.repo.get_record.return_value = mock_response

        loader = DatasetLoader(authenticated_client)

        with pytest.raises(ValueError, match="does not use blob storage"):
            loader.get_blobs(f"at://did:plc:abc/{LEXICON_NAMESPACE}.record/xyz")

    def test_get_blob_urls(self, authenticated_client, mock_atproto_client):
        """Get blob URLs resolves PDS and constructs download URLs."""
        mock_response = Mock()
        mock_response.value = {
            "$type": f"{LEXICON_NAMESPACE}.record",
            "name": "BlobDataset",
            "schemaRef": "at://schema",
            "storage": {
                "$type": f"{LEXICON_NAMESPACE}.storageBlobs",
                "blobs": [
                    {
                        "blob": {
                            "$type": "blob",
                            "ref": {"$link": "bafkreitest1"},
                            "mimeType": "application/x-tar",
                            "size": 1024,
                        },
                        "checksum": {"algorithm": "sha256", "digest": "abc"},
                    },
                    {
                        "blob": {
                            "$type": "blob",
                            "ref": {"$link": "bafkreitest2"},
                            "mimeType": "application/x-tar",
                            "size": 2048,
                        },
                        "checksum": {"algorithm": "sha256", "digest": "def"},
                    },
                ],
            },
        }
        mock_atproto_client.com.atproto.repo.get_record.return_value = mock_response

        # Mock PDS resolution
        with patch("requests.get") as mock_get:
            mock_did_response = Mock()
            mock_did_response.json.return_value = {
                "service": [
                    {
                        "type": "AtprotoPersonalDataServer",
                        "serviceEndpoint": "https://pds.example.com",
                    }
                ]
            }
            mock_did_response.raise_for_status = Mock()
            mock_get.return_value = mock_did_response

            loader = DatasetLoader(authenticated_client)
            urls = loader.get_blob_urls(
                f"at://did:plc:abc123/{LEXICON_NAMESPACE}.record/xyz"
            )

            assert len(urls) == 2
            assert "bafkreitest1" in urls[0]
            assert "bafkreitest2" in urls[1]
            assert "did:plc:abc123" in urls[0]

    def test_get_urls_unknown_storage_error(
        self, authenticated_client, mock_atproto_client
    ):
        """Get URLs raises for unknown storage type."""
        mock_response = Mock()
        mock_response.value = {
            "$type": f"{LEXICON_NAMESPACE}.record",
            "name": "UnknownDataset",
            "schemaRef": "at://schema",
            "storage": {
                "$type": "some.unknown.storage",
            },
        }
        mock_atproto_client.com.atproto.repo.get_record.return_value = mock_response

        loader = DatasetLoader(authenticated_client)

        with pytest.raises(ValueError, match="Unknown storage type"):
            loader.get_urls(f"at://did:plc:abc/{LEXICON_NAMESPACE}.record/xyz")

    def test_get_storage_type_http(self, authenticated_client, mock_atproto_client):
        """get_storage_type returns 'http' for HTTP storage."""
        mock_response = Mock()
        mock_response.value = {
            "$type": f"{LEXICON_NAMESPACE}.record",
            "name": "HttpDataset",
            "schemaRef": "at://schema",
            "storage": {
                "$type": f"{LEXICON_NAMESPACE}.storageHttp",
                "shards": [{"url": "https://cdn.example.com/data.tar"}],
            },
        }
        mock_atproto_client.com.atproto.repo.get_record.return_value = mock_response

        loader = DatasetLoader(authenticated_client)
        assert (
            loader.get_storage_type(f"at://did:plc:abc/{LEXICON_NAMESPACE}.record/xyz")
            == "http"
        )

    def test_get_storage_type_s3(self, authenticated_client, mock_atproto_client):
        """get_storage_type returns 's3' for S3 storage."""
        mock_response = Mock()
        mock_response.value = {
            "$type": f"{LEXICON_NAMESPACE}.record",
            "name": "S3Dataset",
            "schemaRef": "at://schema",
            "storage": {
                "$type": f"{LEXICON_NAMESPACE}.storageS3",
                "bucket": "my-bucket",
                "shards": [{"key": "data.tar"}],
            },
        }
        mock_atproto_client.com.atproto.repo.get_record.return_value = mock_response

        loader = DatasetLoader(authenticated_client)
        assert (
            loader.get_storage_type(f"at://did:plc:abc/{LEXICON_NAMESPACE}.record/xyz")
            == "s3"
        )

    def test_get_urls_http_storage(self, authenticated_client, mock_atproto_client):
        """get_urls extracts URLs from HTTP storage shards."""
        mock_response = Mock()
        mock_response.value = {
            "$type": f"{LEXICON_NAMESPACE}.record",
            "name": "HttpDataset",
            "schemaRef": "at://schema",
            "storage": {
                "$type": f"{LEXICON_NAMESPACE}.storageHttp",
                "shards": [
                    {"url": "https://cdn.example.com/shard-000.tar"},
                    {"url": "https://cdn.example.com/shard-001.tar"},
                ],
            },
        }
        mock_atproto_client.com.atproto.repo.get_record.return_value = mock_response

        loader = DatasetLoader(authenticated_client)
        urls = loader.get_urls(f"at://did:plc:abc/{LEXICON_NAMESPACE}.record/xyz")

        assert urls == [
            "https://cdn.example.com/shard-000.tar",
            "https://cdn.example.com/shard-001.tar",
        ]

    def test_get_urls_s3_storage_no_endpoint(
        self, authenticated_client, mock_atproto_client
    ):
        """get_urls builds s3:// URLs when no endpoint specified."""
        mock_response = Mock()
        mock_response.value = {
            "$type": f"{LEXICON_NAMESPACE}.record",
            "name": "S3Dataset",
            "schemaRef": "at://schema",
            "storage": {
                "$type": f"{LEXICON_NAMESPACE}.storageS3",
                "bucket": "my-bucket",
                "shards": [{"key": "train/shard-000.tar"}],
            },
        }
        mock_atproto_client.com.atproto.repo.get_record.return_value = mock_response

        loader = DatasetLoader(authenticated_client)
        urls = loader.get_urls(f"at://did:plc:abc/{LEXICON_NAMESPACE}.record/xyz")

        assert urls == ["s3://my-bucket/train/shard-000.tar"]

    def test_get_urls_s3_storage_with_endpoint(
        self, authenticated_client, mock_atproto_client
    ):
        """get_urls builds full URLs with custom S3 endpoint."""
        mock_response = Mock()
        mock_response.value = {
            "$type": f"{LEXICON_NAMESPACE}.record",
            "name": "S3Dataset",
            "schemaRef": "at://schema",
            "storage": {
                "$type": f"{LEXICON_NAMESPACE}.storageS3",
                "bucket": "my-bucket",
                "endpoint": "https://s3.us-west-2.amazonaws.com/",
                "shards": [{"key": "shard.tar"}],
            },
        }
        mock_atproto_client.com.atproto.repo.get_record.return_value = mock_response

        loader = DatasetLoader(authenticated_client)
        urls = loader.get_urls(f"at://did:plc:abc/{LEXICON_NAMESPACE}.record/xyz")

        assert urls == ["https://s3.us-west-2.amazonaws.com/my-bucket/shard.tar"]

    def test_get_s3_info(self, authenticated_client, mock_atproto_client):
        """get_s3_info extracts bucket, keys, region, endpoint."""
        mock_response = Mock()
        mock_response.value = {
            "$type": f"{LEXICON_NAMESPACE}.record",
            "name": "S3Dataset",
            "schemaRef": "at://schema",
            "storage": {
                "$type": f"{LEXICON_NAMESPACE}.storageS3",
                "bucket": "my-bucket",
                "region": "us-east-1",
                "endpoint": "https://s3.example.com",
                "shards": [{"key": "a.tar"}, {"key": "b.tar"}],
            },
        }
        mock_atproto_client.com.atproto.repo.get_record.return_value = mock_response

        loader = DatasetLoader(authenticated_client)
        info = loader.get_s3_info(f"at://did:plc:abc/{LEXICON_NAMESPACE}.record/xyz")

        assert info["bucket"] == "my-bucket"
        assert info["keys"] == ["a.tar", "b.tar"]
        assert info["region"] == "us-east-1"
        assert info["endpoint"] == "https://s3.example.com"

    def test_get_s3_info_non_s3_raises(self, authenticated_client, mock_atproto_client):
        """get_s3_info raises ValueError for non-S3 storage."""
        mock_response = Mock()
        mock_response.value = {
            "$type": f"{LEXICON_NAMESPACE}.record",
            "name": "HttpDataset",
            "schemaRef": "at://schema",
            "storage": {
                "$type": f"{LEXICON_NAMESPACE}.storageHttp",
                "shards": [{"url": "https://example.com/data.tar"}],
            },
        }
        mock_atproto_client.com.atproto.repo.get_record.return_value = mock_response

        loader = DatasetLoader(authenticated_client)
        with pytest.raises(ValueError, match="does not use S3 storage"):
            loader.get_s3_info(f"at://did:plc:abc/{LEXICON_NAMESPACE}.record/xyz")

    def test_get_blob_urls_with_aturi_object(
        self, authenticated_client, mock_atproto_client
    ):
        """get_blob_urls accepts AtUri object (not just string)."""
        mock_response = Mock()
        mock_response.value = {
            "$type": f"{LEXICON_NAMESPACE}.record",
            "name": "BlobDataset",
            "schemaRef": "at://schema",
            "storage": {
                "$type": f"{LEXICON_NAMESPACE}.storageBlobs",
                "blobs": [
                    {
                        "blob": {
                            "$type": "blob",
                            "ref": {"$link": "bafkreitest1"},
                            "mimeType": "application/x-tar",
                            "size": 1024,
                        },
                    },
                ],
            },
        }
        mock_atproto_client.com.atproto.repo.get_record.return_value = mock_response

        with patch("requests.get") as mock_get:
            mock_did_response = Mock()
            mock_did_response.json.return_value = {
                "service": [
                    {
                        "type": "AtprotoPersonalDataServer",
                        "serviceEndpoint": "https://pds.example.com",
                    }
                ]
            }
            mock_did_response.raise_for_status = Mock()
            mock_get.return_value = mock_did_response

            loader = DatasetLoader(authenticated_client)
            uri_obj = AtUri.parse(f"at://did:plc:abc123/{LEXICON_NAMESPACE}.record/xyz")
            urls = loader.get_blob_urls(uri_obj)

            assert len(urls) == 1
            assert "bafkreitest1" in urls[0]

    def test_get_typed(self, authenticated_client, mock_atproto_client):
        """get_typed returns LexDatasetRecord."""
        mock_response = Mock()
        mock_response.value = {
            "$type": f"{LEXICON_NAMESPACE}.record",
            "name": "TestDataset",
            "schemaRef": "at://did:plc:abc/schema/xyz",
            "createdAt": "2026-01-01T00:00:00Z",
            "storage": {
                "$type": f"{LEXICON_NAMESPACE}.storageExternal",
                "urls": ["s3://bucket/data.tar"],
            },
        }
        mock_atproto_client.com.atproto.repo.get_record.return_value = mock_response

        loader = DatasetLoader(authenticated_client)
        record = loader.get_typed(f"at://did:plc:abc/{LEXICON_NAMESPACE}.record/xyz")

        assert record.name == "TestDataset"

    def test_to_dataset_http(self, authenticated_client, mock_atproto_client):
        """to_dataset creates Dataset from HTTP storage record."""
        mock_response = Mock()
        mock_response.value = {
            "$type": f"{LEXICON_NAMESPACE}.record",
            "name": "TestDataset",
            "schemaRef": "at://schema",
            "storage": {
                "$type": f"{LEXICON_NAMESPACE}.storageHttp",
                "shards": [{"url": "https://cdn.example.com/data.tar"}],
            },
        }
        mock_atproto_client.com.atproto.repo.get_record.return_value = mock_response

        loader = DatasetLoader(authenticated_client)
        ds = loader.to_dataset(
            f"at://did:plc:abc/{LEXICON_NAMESPACE}.record/xyz",
            BasicSample,
        )

        from atdata import Dataset

        assert isinstance(ds, Dataset)

    def test_to_dataset_empty_urls_raises(
        self, authenticated_client, mock_atproto_client
    ):
        """to_dataset raises ValueError when record has no URLs."""
        mock_response = Mock()
        mock_response.value = {
            "$type": f"{LEXICON_NAMESPACE}.record",
            "name": "EmptyDataset",
            "schemaRef": "at://schema",
            "storage": {
                "$type": f"{LEXICON_NAMESPACE}.storageHttp",
                "shards": [],
            },
        }
        mock_atproto_client.com.atproto.repo.get_record.return_value = mock_response

        loader = DatasetLoader(authenticated_client)
        with pytest.raises(ValueError, match="no storage URLs"):
            loader.to_dataset(
                f"at://did:plc:abc/{LEXICON_NAMESPACE}.record/xyz",
                BasicSample,
            )


class TestDatasetPublisherValidation:
    """Tests for DatasetPublisher checksum validation."""

    def test_publish_with_urls_checksum_mismatch(
        self, authenticated_client, mock_atproto_client
    ):
        """publish_with_urls raises when checksums length mismatches URLs."""
        publisher = DatasetPublisher(authenticated_client)
        with pytest.raises(ValueError, match="checksums length"):
            publisher.publish_with_urls(
                urls=["https://example.com/a.tar", "https://example.com/b.tar"],
                schema_uri="at://did:plc:abc/schema/xyz",
                name="TestDataset",
                checksums=[{"algorithm": "sha256", "digest": "abc"}],  # only 1
            )

    def test_publish_with_s3(self, authenticated_client, mock_atproto_client):
        """publish_with_s3 creates record with S3 storage."""
        mock_response = Mock()
        mock_response.uri = f"at://did:plc:test/{LEXICON_NAMESPACE}.record/abc"
        mock_atproto_client.com.atproto.repo.create_record.return_value = mock_response

        publisher = DatasetPublisher(authenticated_client)
        uri = publisher.publish_with_s3(
            bucket="my-bucket",
            keys=["train/shard-000.tar", "train/shard-001.tar"],
            schema_uri="at://did:plc:abc/schema/xyz",
            name="S3Dataset",
            region="us-east-1",
            endpoint="https://s3.us-east-1.amazonaws.com",
        )

        assert isinstance(uri, AtUri)
        call_args = mock_atproto_client.com.atproto.repo.create_record.call_args
        record = call_args.kwargs["data"]["record"]
        assert "storageS3" in record["storage"]["$type"]
        assert record["storage"]["bucket"] == "my-bucket"

    def test_publish_with_s3_checksum_mismatch(
        self, authenticated_client, mock_atproto_client
    ):
        """publish_with_s3 raises when checksums length mismatches keys."""
        publisher = DatasetPublisher(authenticated_client)
        with pytest.raises(ValueError, match="checksums length"):
            publisher.publish_with_s3(
                bucket="my-bucket",
                keys=["a.tar", "b.tar"],
                schema_uri="at://did:plc:abc/schema/xyz",
                name="TestDataset",
                checksums=[{"algorithm": "sha256", "digest": "abc"}],  # only 1
            )


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

    def test_publish_without_code_refs_raises(self, authenticated_client):
        """Publish lens without code references raises TypeError."""
        publisher = LensPublisher(authenticated_client)

        with pytest.raises(TypeError):
            publisher.publish(
                name="MetadataOnlyLens",
                source_schema_uri="at://source",
                target_schema_uri="at://target",
            )

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
        publisher.publish_from_lens(
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

    def test_find_by_schemas_source_only(
        self, authenticated_client, mock_atproto_client
    ):
        """Find lenses by source schema only."""
        mock_records = [
            Mock(
                value={"sourceSchema": "at://schema/a", "targetSchema": "at://schema/b"}
            ),
            Mock(
                value={"sourceSchema": "at://schema/a", "targetSchema": "at://schema/c"}
            ),
            Mock(
                value={"sourceSchema": "at://schema/x", "targetSchema": "at://schema/y"}
            ),
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
            Mock(
                value={"sourceSchema": "at://schema/a", "targetSchema": "at://schema/b"}
            ),
            Mock(
                value={"sourceSchema": "at://schema/a", "targetSchema": "at://schema/c"}
            ),
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


class TestJsonSchemaEdgeCases:
    """Tests for JSON Schema format edge cases in LexSchemaRecord."""

    def test_schema_with_description_in_metadata(self):
        """Test LexSchemaRecord preserves description in metadata."""
        schema = LexSchemaRecord(
            name="TestSchema",
            version="1.0.0",
            schema_type="jsonSchema",
            schema=JsonSchemaFormat(
                schema_body={
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "described_field": {"type": "string"},
                    },
                    "required": ["described_field"],
                },
            ),
            description="This is a schema description",
        )
        record = schema.to_record()
        assert record["description"] == "This is a schema description"

    def test_ndarray_ref_property(self):
        """Test ndarray property uses $ref in JSON Schema."""
        schema = LexSchemaRecord(
            name="ShapedArraySchema",
            version="1.0.0",
            schema_type="jsonSchema",
            schema=JsonSchemaFormat(
                schema_body={
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "image": {
                            "$ref": "https://foundation.ac/schemas/atdata-ndarray-bytes/1.0.0#/$defs/ndarray"
                        },
                    },
                    "required": ["image"],
                },
                array_format_versions={"ndarrayBytes": "1.0.0"},
            ),
        )
        record = schema.to_record()

        prop = record["schema"]["properties"]["image"]
        assert "$ref" in prop
        assert "ndarray" in prop["$ref"]

    def test_array_type_property(self):
        """Test array type in JSON Schema format."""
        schema = LexSchemaRecord(
            name="ArraySchema",
            version="1.0.0",
            schema_type="jsonSchema",
            schema=JsonSchemaFormat(
                schema_body={
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "numbers": {
                            "type": "array",
                            "items": {"type": "integer"},
                        },
                    },
                    "required": ["numbers"],
                },
            ),
        )
        record = schema.to_record()

        prop = record["schema"]["properties"]["numbers"]
        assert prop["type"] == "array"
        assert prop["items"]["type"] == "integer"

    def test_roundtrip_from_record(self):
        """Test LexSchemaRecord round-trips through to_record/from_record."""
        original = LexSchemaRecord(
            name="RoundtripSchema",
            version="2.0.0",
            schema_type="jsonSchema",
            schema=JsonSchemaFormat(
                schema_body={
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "value": {"type": "integer"},
                    },
                    "required": ["name", "value"],
                },
            ),
            description="A roundtrip test",
        )
        record = original.to_record()
        restored = LexSchemaRecord.from_record(record)

        assert restored.name == "RoundtripSchema"
        assert restored.version == "2.0.0"
        assert restored.schema_type == "jsonSchema"
        assert "name" in restored.schema.schema_body["properties"]
        assert "value" in restored.schema.schema_body["properties"]


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
        mock_response.uri = f"at://did:plc:test/{LEXICON_NAMESPACE}.schema/abc"
        mock_atproto_client.com.atproto.repo.create_record.return_value = mock_response

        publisher = SchemaPublisher(authenticated_client)
        publisher.publish(ListSample, version="1.0.0")

        call_args = mock_atproto_client.com.atproto.repo.create_record.call_args
        record = call_args.kwargs["data"]["record"]

        # Check the tags property in JSON Schema format
        tags_prop = record["schema"]["properties"]["tags"]
        assert tags_prop["type"] == "array"
        assert tags_prop["items"]["type"] == "string"

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


# =============================================================================
# AtmosphereIndex Tests
# =============================================================================


class TestAtmosphereIndexEntry:
    """Tests for AtmosphereIndexEntry wrapper."""

    def test_entry_properties(self):
        """Entry exposes record properties correctly."""
        record = {
            "name": "test-dataset",
            "schemaRef": "at://did:plc:abc/schema/xyz",
            "storage": {
                "$type": f"{LEXICON_NAMESPACE}.storageExternal",
                "urls": ["s3://bucket/data.tar"],
            },
        }

        entry = AtmosphereIndexEntry("at://did:plc:abc/record/123", record)

        assert entry.name == "test-dataset"
        assert entry.schema_ref == "at://did:plc:abc/schema/xyz"
        assert entry.data_urls == ["s3://bucket/data.tar"]
        assert entry.uri == "at://did:plc:abc/record/123"

    def test_entry_empty_storage(self):
        """Entry handles missing storage gracefully."""
        record = {"name": "no-storage"}

        entry = AtmosphereIndexEntry("at://uri", record)

        assert entry.data_urls == []

    @patch("atdata.atmosphere._resolve_pds_endpoint")
    def test_data_urls_resolves_storage_blobs(self, mock_resolve):
        """storageBlobs entries are resolved to PDS HTTP URLs."""
        mock_resolve.return_value = "https://pds.example.com"
        record = {
            "name": "blob-dataset",
            "storage": {
                "$type": f"{LEXICON_NAMESPACE}.storageBlobs",
                "blobs": [
                    {
                        "blob": {
                            "ref": {"$link": "bafyabc"},
                            "mimeType": "application/octet-stream",
                        }
                    },
                    {
                        "blob": {
                            "ref": {"$link": "bafydef"},
                            "mimeType": "application/octet-stream",
                        }
                    },
                ],
            },
        }

        entry = AtmosphereIndexEntry(
            "at://did:plc:testdid/ac.foundation.dataset.record/rkey123",
            record,
        )
        urls = entry.data_urls

        assert len(urls) == 2
        assert urls[0] == (
            "https://pds.example.com/xrpc/com.atproto.sync.getBlob"
            "?did=did:plc:testdid&cid=bafyabc"
        )
        assert urls[1] == (
            "https://pds.example.com/xrpc/com.atproto.sync.getBlob"
            "?did=did:plc:testdid&cid=bafydef"
        )
        mock_resolve.assert_called_once_with("did:plc:testdid")


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestAtmosphereIndex:
    """Tests for AtmosphereIndex unified interface."""

    def test_init(self, authenticated_client):
        """Index initializes with client and creates publishers/loaders."""
        index = AtmosphereIndex(authenticated_client)

        assert index.client is authenticated_client
        assert index._schema_publisher is not None
        assert index._schema_loader is not None
        assert index._dataset_publisher is not None
        assert index._dataset_loader is not None

    def test_has_protocol_methods(self, authenticated_client):
        """Index has all AbstractIndex protocol methods."""
        index = AtmosphereIndex(authenticated_client)

        assert hasattr(index, "insert_dataset")
        assert hasattr(index, "get_dataset")
        assert hasattr(index, "list_datasets")
        assert hasattr(index, "publish_schema")
        assert hasattr(index, "get_schema")
        assert hasattr(index, "list_schemas")
        assert hasattr(index, "get_schema_type")

    def test_publish_schema(self, authenticated_client, mock_atproto_client):
        """publish_schema delegates to SchemaPublisher."""
        mock_response = Mock()
        mock_response.uri = f"at://did:plc:test/{LEXICON_NAMESPACE}.schema/abc"
        mock_atproto_client.com.atproto.repo.create_record.return_value = mock_response

        index = AtmosphereIndex(authenticated_client)
        uri = index.publish_schema(BasicSample, version="2.0.0")

        assert uri == str(mock_response.uri)
        mock_atproto_client.com.atproto.repo.create_record.assert_called_once()
        call_data = mock_atproto_client.com.atproto.repo.create_record.call_args[1][
            "data"
        ]
        assert call_data["collection"] == f"{LEXICON_NAMESPACE}.schema"
        assert call_data["record"]["name"] == "BasicSample"

    def test_get_schema(self, authenticated_client, mock_atproto_client):
        """get_schema delegates to SchemaLoader."""
        mock_response = Mock()
        mock_response.value = {
            "$type": f"{LEXICON_NAMESPACE}.schema",
            "name": "TestSchema",
            "version": "1.0.0",
            "fields": [],
        }
        mock_atproto_client.com.atproto.repo.get_record.return_value = mock_response

        index = AtmosphereIndex(authenticated_client)
        schema = index.get_schema("at://did:plc:test/schema/abc")

        assert schema["name"] == "TestSchema"
