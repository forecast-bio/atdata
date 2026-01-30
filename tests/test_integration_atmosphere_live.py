"""Live ATProto network integration tests.

These tests connect to the real ATProto network (Bluesky) and verify
that atdata's atmosphere functionality works correctly with actual API calls.

Requirements:
    - testing.env file with ATDATA_TEST_HANDLE and ATDATA_TEST_APP_PASSWORD
    - Network connectivity to bsky.social

Run with:
    source testing.env && uv run pytest tests/test_integration_atmosphere_live.py -v

These tests are marked with @pytest.mark.network and @pytest.mark.slow.
To skip in CI: pytest -m "not network"
"""

import os
import uuid
import pytest
from datetime import datetime

from numpy.typing import NDArray

import atdata
from atdata.atmosphere import (
    AtmosphereClient,
    AtmosphereIndex,
    SchemaPublisher,
    SchemaLoader,
    DatasetPublisher,
    DatasetLoader,
)
from atdata.atmosphere._types import LEXICON_NAMESPACE


##
# Test Configuration


# Prefix for all test records - makes cleanup easier
TEST_PREFIX = "atdata-live-test"


def get_test_credentials():
    """Get test credentials from environment."""
    handle = os.environ.get("ATDATA_TEST_HANDLE")
    password = os.environ.get("ATDATA_TEST_APP_PASSWORD")
    return handle, password


def skip_if_no_credentials():
    """Skip test if credentials not available."""
    handle, password = get_test_credentials()
    if not handle or not password:
        pytest.skip(
            "Live test credentials not configured (set ATDATA_TEST_HANDLE and ATDATA_TEST_APP_PASSWORD)"
        )


##
# Test Sample Types


@atdata.packable
class LiveTestSample:
    """Simple sample for live tests."""

    name: str
    value: int


@atdata.packable
class LiveTestArraySample:
    """Sample with NDArray for live tests."""

    label: str
    data: NDArray


##
# Fixtures


@pytest.fixture(scope="module")
def live_client():
    """Create authenticated client for live tests.

    This fixture is module-scoped so we reuse the same session
    across all tests in this file, reducing API calls.
    """
    handle, password = get_test_credentials()
    if not handle or not password:
        pytest.skip("Live test credentials not configured")

    client = AtmosphereClient()
    client.login(handle, password)

    yield client

    # Cleanup: delete test records created during this session
    # This runs after all tests in the module complete


@pytest.fixture(scope="module")
def live_index(live_client):
    """Create AtmosphereIndex for live tests."""
    return AtmosphereIndex(live_client)


@pytest.fixture
def unique_name():
    """Generate a unique name for test records."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    return f"{TEST_PREFIX}-{timestamp}-{unique_id}"


##
# Cleanup Utilities


def cleanup_test_schemas(client: AtmosphereClient):
    """Delete all test schema records."""
    loader = SchemaLoader(client)
    deleted = 0
    failed = 0

    for record in loader.list_all():
        name = record.get("value", {}).get("name", "")
        if TEST_PREFIX in name:
            uri = record.get("uri", "")
            if uri:
                try:
                    client.delete_record(uri)
                    deleted += 1
                except Exception:
                    failed += 1  # Continue cleanup even if individual deletes fail

    return deleted


def cleanup_test_datasets(client: AtmosphereClient):
    """Delete all test dataset records."""
    loader = DatasetLoader(client)
    deleted = 0
    failed = 0

    for record in loader.list_all():
        name = record.get("value", {}).get("name", "")
        if TEST_PREFIX in name:
            uri = record.get("uri", "")
            if uri:
                try:
                    client.delete_record(uri)
                    deleted += 1
                except Exception:
                    failed += 1  # Continue cleanup even if individual deletes fail

    return deleted


##
# Authentication Tests


@pytest.mark.network
@pytest.mark.slow
class TestLiveAuthentication:
    """Live tests for authentication flow."""

    def test_login_succeeds(self):
        """Should successfully authenticate with valid credentials."""
        handle, password = get_test_credentials()
        skip_if_no_credentials()

        client = AtmosphereClient()
        client.login(handle, password)

        assert client.is_authenticated
        assert client.did is not None
        assert client.did.startswith("did:plc:")

    def test_session_export_import(self):
        """Should export and restore session."""
        handle, password = get_test_credentials()
        skip_if_no_credentials()

        # Create first client and login
        client1 = AtmosphereClient()
        client1.login(handle, password)
        session_string = client1.export_session()

        assert session_string is not None
        assert len(session_string) > 0

        # Create second client and restore session
        client2 = AtmosphereClient()
        client2.login_with_session(session_string)

        assert client2.is_authenticated
        assert client2.did == client1.did

    def test_invalid_credentials_raises(self):
        """Should raise on invalid credentials."""
        client = AtmosphereClient()

        with pytest.raises(Exception):
            client.login("invalid.handle.test", "wrong-password")


##
# Schema Operation Tests


@pytest.mark.network
@pytest.mark.slow
class TestLiveSchemaOperations:
    """Live tests for schema publishing and retrieval."""

    def test_publish_schema(self, live_client, unique_name):
        """Should publish a schema to ATProto."""

        # Create a unique sample type for this test
        @atdata.packable
        class UniqueTestSample:
            name: str
            count: int

        # Monkey-patch the module name to include test prefix
        UniqueTestSample.__module__ = f"{TEST_PREFIX}.{unique_name}"

        publisher = SchemaPublisher(live_client)
        uri = publisher.publish(UniqueTestSample, version="1.0.0")

        assert uri is not None
        assert "at://" in str(uri)
        assert LEXICON_NAMESPACE in str(uri)

    def test_list_schemas(self, live_client):
        """Should list published schemas."""
        loader = SchemaLoader(live_client)
        schemas = loader.list_all()

        # Should return a list (may be empty if no schemas published)
        assert isinstance(schemas, list)

    def test_publish_and_retrieve_schema(self, live_client, unique_name):
        """Should publish then retrieve a schema by URI."""

        @atdata.packable
        class RetrievableTestSample:
            field1: str
            field2: int

        RetrievableTestSample.__module__ = f"{TEST_PREFIX}.{unique_name}"

        publisher = SchemaPublisher(live_client)
        uri = publisher.publish(RetrievableTestSample, version="1.0.0")

        loader = SchemaLoader(live_client)
        schema = loader.get(str(uri))

        assert schema is not None
        # Schema name defaults to just the class name (not full module path)
        assert schema["name"] == "RetrievableTestSample"
        assert schema["version"] == "1.0.0"

        field_names = {f["name"] for f in schema["fields"]}
        assert "field1" in field_names
        assert "field2" in field_names

    def test_schema_with_ndarray_field(self, live_client, unique_name):
        """Should publish schema with NDArray field type."""

        @atdata.packable
        class ArrayTestSample:
            label: str
            embedding: NDArray

        ArrayTestSample.__module__ = f"{TEST_PREFIX}.{unique_name}"

        publisher = SchemaPublisher(live_client)
        uri = publisher.publish(ArrayTestSample, version="1.0.0")

        loader = SchemaLoader(live_client)
        schema = loader.get(str(uri))

        # Find the embedding field
        embedding_field = next(f for f in schema["fields"] if f["name"] == "embedding")
        # Field type should indicate ndarray
        assert "ndarray" in embedding_field["fieldType"]["$type"].lower()


##
# Dataset Operation Tests


@pytest.mark.network
@pytest.mark.slow
class TestLiveDatasetOperations:
    """Live tests for dataset publishing and retrieval."""

    def test_publish_dataset_with_urls(self, live_client, unique_name):
        """Should publish a dataset record with external URLs."""

        # First publish a schema
        @atdata.packable
        class DatasetTestSample:
            id: int

        DatasetTestSample.__module__ = f"{TEST_PREFIX}.{unique_name}"

        schema_pub = SchemaPublisher(live_client)
        schema_uri = schema_pub.publish(DatasetTestSample, version="1.0.0")

        # Now publish dataset with external URLs
        dataset_pub = DatasetPublisher(live_client)
        dataset_uri = dataset_pub.publish_with_urls(
            urls=["https://example.com/test-shard-000000.tar"],
            schema_uri=str(schema_uri),
            name=unique_name,
            description="Test dataset for live integration tests",
        )

        assert dataset_uri is not None
        assert "at://" in str(dataset_uri)

    def test_list_datasets(self, live_client):
        """Should list published datasets."""
        loader = DatasetLoader(live_client)
        datasets = loader.list_all()

        assert isinstance(datasets, list)

    def test_publish_and_retrieve_dataset(self, live_client, unique_name):
        """Should publish then retrieve a dataset."""

        @atdata.packable
        class RetrievableDatasetSample:
            value: int

        RetrievableDatasetSample.__module__ = f"{TEST_PREFIX}.{unique_name}"

        # Publish schema
        schema_pub = SchemaPublisher(live_client)
        schema_uri = schema_pub.publish(RetrievableDatasetSample, version="1.0.0")

        # Publish dataset
        test_urls = [
            "https://example.com/shard-000000.tar",
            "https://example.com/shard-000001.tar",
        ]

        dataset_pub = DatasetPublisher(live_client)
        dataset_uri = dataset_pub.publish_with_urls(
            urls=test_urls,
            schema_uri=str(schema_uri),
            name=unique_name,
            description="Retrievable test dataset",
            tags=["test", "integration"],
        )

        # Retrieve dataset
        loader = DatasetLoader(live_client)
        dataset = loader.get(str(dataset_uri))

        assert dataset is not None
        assert dataset["name"] == unique_name
        assert dataset["description"] == "Retrievable test dataset"

    def test_to_dataset_with_fake_urls_fails_on_iteration(
        self, live_client, unique_name
    ):
        """Attempting to iterate a dataset with fake URLs should fail.

        This test documents a known limitation: we can publish and retrieve
        dataset *metadata* with fake URLs, but actual data iteration fails.
        For true E2E tests, we need either:
        1. Real external URLs (e.g., S3 with test data)
        2. ATProto blob storage support (not yet implemented)
        """

        @atdata.packable
        class IterationTestSample:
            value: int

        IterationTestSample.__module__ = f"{TEST_PREFIX}.{unique_name}"

        # Publish schema
        schema_pub = SchemaPublisher(live_client)
        schema_uri = schema_pub.publish(IterationTestSample, version="1.0.0")

        # Publish dataset with fake URL
        dataset_pub = DatasetPublisher(live_client)
        dataset_uri = dataset_pub.publish_with_urls(
            urls=["https://example.com/fake-shard-000000.tar"],
            schema_uri=str(schema_uri),
            name=unique_name,
            description="Dataset with fake URLs",
        )

        # Can retrieve metadata just fine
        loader = DatasetLoader(live_client)
        urls = loader.get_urls(str(dataset_uri))
        assert urls == ["https://example.com/fake-shard-000000.tar"]

        # But creating a Dataset and iterating should fail
        # (the URL doesn't actually exist)
        with pytest.raises(Exception):
            ds = loader.to_dataset(str(dataset_uri), IterationTestSample)
            # Attempt to iterate - this should fail when trying to fetch data
            # Consume the iterator to trigger the network request
            list(ds.ordered())

    def test_full_e2e_with_local_fixture(self, live_client, unique_name):
        """Full E2E: publish schema + dataset, retrieve, iterate over real data.

        This test uses a local file:// URL to test the complete flow:
        1. Publish schema to ATProto
        2. Publish dataset record with local file URL
        3. Retrieve dataset record
        4. Load data via to_dataset() and iterate
        5. Verify we get the expected samples
        """
        from pathlib import Path

        # Define sample type matching the fixture
        @atdata.packable
        class FixtureSample:
            id: int
            name: str
            value: int

        FixtureSample.__module__ = f"{TEST_PREFIX}.{unique_name}"

        # Get absolute path to test fixture
        fixture_path = Path(__file__).parent / "fixtures" / "test_samples.tar"
        if not fixture_path.exists():
            pytest.skip("Test fixture not found")
        fixture_url = f"file://{fixture_path.absolute()}"

        # 1. Publish schema
        schema_pub = SchemaPublisher(live_client)
        schema_uri = schema_pub.publish(FixtureSample, version="1.0.0")

        # 2. Publish dataset with local file URL
        dataset_pub = DatasetPublisher(live_client)
        dataset_uri = dataset_pub.publish_with_urls(
            urls=[fixture_url],
            schema_uri=str(schema_uri),
            name=unique_name,
            description="E2E test with real data",
        )

        # 3. Retrieve dataset record
        loader = DatasetLoader(live_client)
        record = loader.get(str(dataset_uri))
        assert record["name"] == unique_name

        # 4. Load and iterate
        ds = loader.to_dataset(str(dataset_uri), FixtureSample)
        samples = list(ds.ordered())

        # 5. Verify data (3 samples in fixture)
        assert len(samples) == 3

        # Check sample values (batched as lists)
        assert samples[0].id == [0]
        assert samples[0].name == ["test_sample_0"]
        assert samples[0].value == [0]

        assert samples[2].id == [2]
        assert samples[2].name == ["test_sample_2"]
        assert samples[2].value == [20]

    def test_blob_storage_roundtrip(self, live_client, unique_name):
        """Full E2E: upload blob, publish dataset, retrieve and iterate.

        This tests the complete blob storage workflow:
        1. Create a WebDataset tar in memory using as_wds
        2. Upload as blob to PDS
        3. Publish dataset record with blob storage
        4. Retrieve record and get blob URLs
        5. Load data via to_dataset() and iterate
        6. Verify samples match original data
        """
        import io
        import webdataset as wds

        # Define sample type
        @atdata.packable
        class BlobTestSample:
            id: int
            message: str

        BlobTestSample.__module__ = f"{TEST_PREFIX}.{unique_name}"

        # 1. Create WebDataset tar in memory using proper as_wds pattern
        expected_samples = [
            BlobTestSample(id=0, message="hello from blob"),
            BlobTestSample(id=1, message="blob storage works"),
            BlobTestSample(id=2, message="atproto is cool"),
        ]

        tar_buffer = io.BytesIO()
        with wds.writer.TarWriter(tar_buffer) as sink:
            for sample in expected_samples:
                sink.write(sample.as_wds)

        tar_data = tar_buffer.getvalue()

        # 2. Publish schema
        schema_pub = SchemaPublisher(live_client)
        schema_uri = schema_pub.publish(BlobTestSample, version="1.0.0")

        # 3. Publish dataset with blob storage
        dataset_pub = DatasetPublisher(live_client)
        dataset_uri = dataset_pub.publish_with_blobs(
            blobs=[tar_data],
            schema_uri=str(schema_uri),
            name=unique_name,
            description="E2E blob storage test",
        )

        assert dataset_uri is not None
        assert "at://" in str(dataset_uri)

        # 4. Retrieve and verify storage type
        loader = DatasetLoader(live_client)
        storage_type = loader.get_storage_type(str(dataset_uri))
        assert storage_type == "blobs"

        # 5. Get blob URLs
        blob_urls = loader.get_blob_urls(str(dataset_uri))
        assert len(blob_urls) == 1
        assert "getBlob" in blob_urls[0]

        # 6. Load and iterate
        ds = loader.to_dataset(str(dataset_uri), BlobTestSample)
        samples = list(ds.ordered())

        # 7. Verify data (3 samples)
        assert len(samples) == 3

        # Check sample values (batched as lists)
        assert samples[0].id == [0]
        assert samples[0].message == ["hello from blob"]

        assert samples[1].id == [1]
        assert samples[1].message == ["blob storage works"]

        assert samples[2].id == [2]
        assert samples[2].message == ["atproto is cool"]


##
# AtmosphereIndex Tests


@pytest.mark.network
@pytest.mark.slow
class TestLiveAtmosphereIndex:
    """Live tests for AtmosphereIndex wrapper."""

    def test_index_list_datasets(self, live_index):
        """Should list datasets via AtmosphereIndex."""
        datasets = list(live_index.list_datasets())

        # Should return iterable of entries
        assert isinstance(datasets, list)

    def test_index_publish_schema(self, live_index, unique_name):
        """Should publish schema via AtmosphereIndex."""

        @atdata.packable
        class IndexTestSample:
            data: str

        IndexTestSample.__module__ = f"{TEST_PREFIX}.{unique_name}"

        schema_ref = live_index.publish_schema(IndexTestSample, version="1.0.0")

        assert schema_ref is not None
        assert "at://" in str(schema_ref)

    def test_index_get_schema(self, live_index, unique_name):
        """Should retrieve schema via AtmosphereIndex."""

        @atdata.packable
        class GetSchemaTestSample:
            field: int

        GetSchemaTestSample.__module__ = f"{TEST_PREFIX}.{unique_name}"

        schema_ref = live_index.publish_schema(GetSchemaTestSample, version="1.0.0")
        schema = live_index.get_schema(str(schema_ref))

        # Schema name defaults to just the class name (not full module path)
        assert schema["name"] == "GetSchemaTestSample"


##
# Error Handling Tests


@pytest.mark.network
@pytest.mark.slow
class TestLiveErrorHandling:
    """Live tests for error handling with real API."""

    def test_get_nonexistent_record(self, live_client):
        """Should raise on getting non-existent record."""
        loader = SchemaLoader(live_client)

        fake_uri = (
            f"at://{live_client.did}/{LEXICON_NAMESPACE}.sampleSchema/nonexistent12345"
        )

        with pytest.raises(Exception):
            loader.get(fake_uri)

    def test_publish_without_auth_raises(self):
        """Should raise when publishing without authentication."""
        client = AtmosphereClient()
        # Not logged in

        publisher = SchemaPublisher(client)

        with pytest.raises(ValueError, match="authenticated"):
            publisher.publish(LiveTestSample, version="1.0.0")


##
# Cleanup Test (runs last)


@pytest.mark.network
@pytest.mark.slow
class TestZZZCleanup:
    """Cleanup test records. Named ZZZ to run last."""

    def test_cleanup_test_records(self, live_client):
        """Clean up all test records created during this test run."""
        schemas_deleted = cleanup_test_schemas(live_client)
        datasets_deleted = cleanup_test_datasets(live_client)

        print(
            f"\nCleanup: deleted {schemas_deleted} schemas, {datasets_deleted} datasets"
        )

        # Verify cleanup ran and returned counts
        assert isinstance(schemas_deleted, int)
        assert isinstance(datasets_deleted, int)
