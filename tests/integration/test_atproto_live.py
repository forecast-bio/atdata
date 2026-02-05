"""Live ATProto integration tests.

Exercises the Atmosphere client against a real PDS (bsky.social or
self-hosted).  Requires ``ATPROTO_TEST_HANDLE`` and
``ATPROTO_TEST_PASSWORD`` environment variables.

Every record created is cleaned up in a finalizer so the test account
stays tidy.
"""

import io
import os

import pytest
import webdataset as wds
from numpy.typing import NDArray

import atdata
from atdata.atmosphere import (
    Atmosphere,
    SchemaPublisher,
    SchemaLoader,
    DatasetPublisher,
    DatasetLoader,
)
from atdata.atmosphere._types import LEXICON_NAMESPACE

from .conftest import unique_name, RUN_ID

# ── Sample types ──────────────────────────────────────────────────


@atdata.packable
class IntegBasicSample:
    name: str
    value: int


@atdata.packable
class IntegArraySample:
    label: str
    data: NDArray


# ── Helpers ───────────────────────────────────────────────────────

TEST_COLLECTION_SCHEMA = f"{LEXICON_NAMESPACE}.schema"
TEST_COLLECTION_RECORD = f"{LEXICON_NAMESPACE}.record"


def _cleanup_records(client: Atmosphere, collection: str, prefix: str) -> int:
    """Delete all records in *collection* whose name contains *prefix*."""
    deleted = 0
    records, _ = client.list_records(collection)
    for rec in records:
        rec_name = rec.get("name", "")
        if prefix in rec_name:
            uri = rec.get("uri") or rec.get("$uri")
            if uri:
                try:
                    client.delete_record(uri)
                    deleted += 1
                except Exception:
                    continue  # best-effort: skip failures during cleanup
    return deleted


# ── Authentication ────────────────────────────────────────────────


class TestAuthentication:
    """Login and session management against the live PDS."""

    def test_login_returns_did(self, atproto_client: Atmosphere):
        assert atproto_client.is_authenticated
        assert atproto_client.did.startswith("did:")

    def test_session_export_reimport(self, atproto_credentials: tuple[str, str]):
        handle, password = atproto_credentials
        client1 = Atmosphere.login(handle, password)
        session_str = client1.export_session()
        assert len(session_str) > 0

        client2 = Atmosphere.from_session(session_str)
        assert client2.is_authenticated
        assert client2.did == client1.did

    def test_invalid_credentials_raises(self):
        with pytest.raises(Exception):
            Atmosphere.login("invalid-handle-does-not-exist.test", "wrong")


# ── Blob upload / download ────────────────────────────────────────


class TestBlobOperations:
    """Upload and download blobs via the live PDS."""

    def test_upload_and_download_blob(self, atproto_client: Atmosphere):
        payload = b"hello from atdata integration test " + RUN_ID.encode()
        blob_ref = atproto_client.upload_blob(
            payload, mime_type="application/octet-stream"
        )

        assert blob_ref["$type"] == "blob"
        assert blob_ref["size"] == len(payload)
        assert "ref" in blob_ref

        # Blobs must be referenced by a record for the PDS to serve them.
        name = unique_name("blob-dl")
        record = {
            "$type": TEST_COLLECTION_SCHEMA,
            "name": name,
            "version": "1.0.0",
            "fields": [],
            "blob": blob_ref,
        }
        uri = atproto_client.create_record(TEST_COLLECTION_SCHEMA, record)

        cid = blob_ref["ref"]["$link"]
        downloaded = atproto_client.get_blob(atproto_client.did, cid)
        assert downloaded == payload

        atproto_client.delete_record(uri)

    def test_upload_larger_blob(self, atproto_client: Atmosphere):
        """Upload a ~10 KB blob to verify timeout heuristics work."""
        payload = os.urandom(10_000)
        blob_ref = atproto_client.upload_blob(payload)
        assert blob_ref["size"] == len(payload)


# ── Record CRUD ───────────────────────────────────────────────────


class TestRecordCRUD:
    """Create, get, list, and delete records on the live PDS."""

    def test_create_get_delete_record(self, atproto_client: Atmosphere):
        name = unique_name("crud")
        record = {
            "$type": TEST_COLLECTION_SCHEMA,
            "name": name,
            "version": "1.0.0",
            "fields": [],
        }
        uri = atproto_client.create_record(TEST_COLLECTION_SCHEMA, record)
        assert str(uri).startswith("at://")

        fetched = atproto_client.get_record(uri)
        assert fetched["name"] == name

        atproto_client.delete_record(uri)

        with pytest.raises(Exception):
            atproto_client.get_record(uri)

    def test_list_records(self, atproto_client: Atmosphere):
        records, cursor = atproto_client.list_records(TEST_COLLECTION_SCHEMA)
        assert isinstance(records, list)
        # cursor may be None if < limit records


# ── Schema publish / retrieve ─────────────────────────────────────


class TestSchemaPublishing:
    """Publish and retrieve schemas via SchemaPublisher/Loader."""

    def test_publish_and_get_schema(self, atproto_client: Atmosphere):
        name = unique_name("schema")

        @atdata.packable
        class _Sample:
            text: str
            score: int

        _Sample.__module__ = f"integ.{name}"

        pub = SchemaPublisher(atproto_client)
        uri = pub.publish(_Sample, version="1.0.0")
        assert "at://" in str(uri)

        loader = SchemaLoader(atproto_client)
        schema = loader.get(str(uri))
        assert schema["version"] == "1.0.0"
        properties = schema["schema"]["properties"]
        assert set(properties.keys()) == {"text", "score"}
        assert properties["text"]["type"] == "string"
        assert properties["score"]["type"] == "integer"

        # cleanup
        atproto_client.delete_record(uri)

    def test_schema_with_ndarray_field(self, atproto_client: Atmosphere):
        @atdata.packable
        class _ArraySample:
            embedding: NDArray
            label: str

        # Don't override __module__ here — get_type_hints() needs to resolve
        # NDArray from this module's globals.  Record key uniqueness comes
        # from the auto-generated TID rkey.

        pub = SchemaPublisher(atproto_client)
        uri = pub.publish(_ArraySample, version="1.0.0")

        loader = SchemaLoader(atproto_client)
        schema = loader.get(str(uri))
        properties = schema["schema"]["properties"]
        assert "embedding" in properties
        assert "$ref" in properties["embedding"]
        assert "ndarray" in properties["embedding"]["$ref"].lower()

        atproto_client.delete_record(uri)


# ── Dataset publish with URLs ─────────────────────────────────────


class TestDatasetPublishing:
    """Publish dataset records pointing at external URLs."""

    def test_publish_dataset_with_urls(self, atproto_client: Atmosphere):
        name = unique_name("ds-url")

        @atdata.packable
        class _DSample:
            value: int

        _DSample.__module__ = f"integ.{name}"

        schema_pub = SchemaPublisher(atproto_client)
        schema_uri = schema_pub.publish(_DSample, version="1.0.0")

        ds_pub = DatasetPublisher(atproto_client)
        ds_uri = ds_pub.publish_with_urls(
            urls=["https://example.com/shard-000000.tar"],
            schema_uri=str(schema_uri),
            name=name,
            description="integration test dataset",
        )
        assert "at://" in str(ds_uri)

        loader = DatasetLoader(atproto_client)
        record = loader.get(str(ds_uri))
        assert record["name"] == name

        # cleanup
        atproto_client.delete_record(ds_uri)
        atproto_client.delete_record(schema_uri)


# ── Blob-storage round-trip ───────────────────────────────────────


class TestBlobRoundTrip:
    """Full E2E: write samples → upload as blob → retrieve and iterate."""

    def test_write_upload_iterate(self, atproto_client: Atmosphere):
        name = unique_name("blob-rt")

        @atdata.packable
        class _BlobSample:
            id: int
            message: str

        _BlobSample.__module__ = f"integ.{name}"

        samples = [
            _BlobSample(id=0, message="hello"),
            _BlobSample(id=1, message="world"),
        ]

        # Build tar in memory
        buf = io.BytesIO()
        with wds.writer.TarWriter(buf) as sink:
            for s in samples:
                sink.write(s.as_wds)
        tar_bytes = buf.getvalue()

        # Publish schema + dataset with blob
        schema_pub = SchemaPublisher(atproto_client)
        schema_uri = schema_pub.publish(_BlobSample, version="1.0.0")

        ds_pub = DatasetPublisher(atproto_client)
        ds_uri = ds_pub.publish_with_blobs(
            blobs=[tar_bytes],
            schema_uri=str(schema_uri),
            name=name,
            description="blob round-trip test",
        )

        # Retrieve and iterate
        loader = DatasetLoader(atproto_client)
        assert loader.get_storage_type(str(ds_uri)) == "blobs"

        blob_urls = loader.get_blob_urls(str(ds_uri))
        assert len(blob_urls) == 1

        ds = loader.to_dataset(str(ds_uri), _BlobSample)
        result = list(ds.ordered())
        assert len(result) == 2
        assert result[0].message == "hello"
        assert result[1].message == "world"

        # cleanup
        atproto_client.delete_record(ds_uri)
        atproto_client.delete_record(schema_uri)


# ── Error handling ────────────────────────────────────────────────


class TestErrorHandling:
    """Verify error paths against the live PDS."""

    def test_publish_without_auth_raises(self):
        client = Atmosphere()
        pub = SchemaPublisher(client)
        with pytest.raises(ValueError, match="authenticated"):
            pub.publish(IntegBasicSample, version="1.0.0")

    def test_get_nonexistent_record(self, atproto_client: Atmosphere):
        fake_uri = (
            f"at://{atproto_client.did}/{LEXICON_NAMESPACE}.schema/nonexistent99999"
        )
        loader = SchemaLoader(atproto_client)
        with pytest.raises(Exception):
            loader.get(fake_uri)

    def test_full_e2e_with_local_fixture(self, atproto_client: Atmosphere):
        """Publish schema + dataset with local fixture tar, retrieve and iterate."""
        from pathlib import Path

        @atdata.packable
        class _FixtureSample:
            id: int
            name: str
            value: int

        name = unique_name("e2e-fixture")
        _FixtureSample.__module__ = f"integ.{name}"

        fixture_path = Path(__file__).parent.parent / "fixtures" / "test_samples.tar"
        if not fixture_path.exists():
            pytest.skip("Test fixture not found")
        fixture_url = f"file://{fixture_path.absolute()}"

        schema_pub = SchemaPublisher(atproto_client)
        schema_uri = schema_pub.publish(_FixtureSample, version="1.0.0")

        ds_pub = DatasetPublisher(atproto_client)
        ds_uri = ds_pub.publish_with_urls(
            urls=[fixture_url],
            schema_uri=str(schema_uri),
            name=name,
            description="E2E fixture test",
        )

        loader = DatasetLoader(atproto_client)
        record = loader.get(str(ds_uri))
        assert record["name"] == name

        ds = loader.to_dataset(str(ds_uri), _FixtureSample)
        samples = list(ds.ordered())
        assert len(samples) == 3
        assert samples[0].id == 0
        assert samples[0].name == "test_sample_0"

        # cleanup
        atproto_client.delete_record(ds_uri)
        atproto_client.delete_record(schema_uri)


# ── Sweep cleanup (runs last by naming convention) ────────────────


class TestZZZCleanup:
    """Best-effort cleanup of any leftover test records from this run."""

    def test_cleanup_schemas(self, atproto_client: Atmosphere):
        deleted = _cleanup_records(atproto_client, TEST_COLLECTION_SCHEMA, RUN_ID)
        assert deleted >= 0

    def test_cleanup_datasets(self, atproto_client: Atmosphere):
        deleted = _cleanup_records(atproto_client, TEST_COLLECTION_RECORD, RUN_ID)
        assert deleted >= 0
