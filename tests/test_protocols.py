"""Protocol compliance tests for atdata abstractions.

These tests verify that concrete implementations satisfy their protocol
definitions, ensuring interoperability between local and atmosphere backends.
"""

from unittest.mock import Mock

from atdata._protocols import (
    IndexEntry,
)
from atdata.local import LocalDatasetEntry, Index, S3DataStore
from atdata.atmosphere import AtmosphereIndex, AtmosphereIndexEntry


class TestIndexEntryProtocol:
    """Tests for IndexEntry protocol compliance."""

    def test_local_dataset_entry_is_index_entry(self):
        """LocalDatasetEntry should satisfy IndexEntry protocol."""
        entry = LocalDatasetEntry(
            name="test-dataset",
            schema_ref="local://schemas/test@1.0.0",
            data_urls=["s3://bucket/data.tar"],
            metadata={"key": "value"},
        )

        # Protocol compliance via isinstance (runtime_checkable)
        assert isinstance(entry, IndexEntry)

        # Verify required properties exist and work
        assert entry.name == "test-dataset"
        assert entry.schema_ref == "local://schemas/test@1.0.0"
        assert entry.data_urls == ["s3://bucket/data.tar"]
        assert entry.metadata == {"key": "value"}

    def test_atmosphere_index_entry_is_index_entry(self):
        """AtmosphereIndexEntry should satisfy IndexEntry protocol."""
        record = {
            "name": "atmo-dataset",
            "schemaRef": "at://did:plc:test/schema/abc",
            "storage": {
                "$type": "ac.foundation.dataset.storageExternal",
                "urls": ["s3://bucket/data.tar"],
            },
        }
        entry = AtmosphereIndexEntry("at://did:plc:test/record/xyz", record)

        # Protocol compliance
        assert isinstance(entry, IndexEntry)

        # Verify properties
        assert entry.name == "atmo-dataset"
        assert entry.schema_ref == "at://did:plc:test/schema/abc"
        assert entry.data_urls == ["s3://bucket/data.tar"]

    def test_index_entry_with_none_metadata(self):
        """IndexEntry should handle None metadata."""
        entry = LocalDatasetEntry(
            name="no-meta",
            schema_ref="local://schemas/test@1.0.0",
            data_urls=["s3://bucket/data.tar"],
            metadata=None,
        )

        assert entry.metadata is None


class TestAbstractIndexProtocol:
    """Tests for AbstractIndex protocol compliance."""

    def test_local_index_has_required_methods(self):
        """Index should have all AbstractIndex methods."""
        # Can't use isinstance with non-runtime_checkable Protocol
        # So we verify methods exist
        index = Index()

        assert hasattr(index, "insert_dataset")
        assert hasattr(index, "get_dataset")
        assert hasattr(index, "list_datasets")
        assert hasattr(index, "publish_schema")
        assert hasattr(index, "get_schema")
        assert hasattr(index, "list_schemas")
        assert hasattr(index, "decode_schema")

        # Verify methods are callable
        assert callable(index.insert_dataset)
        assert callable(index.get_dataset)
        assert callable(index.list_datasets)
        assert callable(index.publish_schema)
        assert callable(index.get_schema)
        assert callable(index.list_schemas)
        assert callable(index.decode_schema)

    def test_atmosphere_index_has_required_methods(self):
        """AtmosphereIndex should have all AbstractIndex methods."""
        mock_client = Mock()
        mock_client.did = "did:plc:test"
        index = AtmosphereIndex(mock_client)

        assert hasattr(index, "insert_dataset")
        assert hasattr(index, "get_dataset")
        assert hasattr(index, "list_datasets")
        assert hasattr(index, "publish_schema")
        assert hasattr(index, "get_schema")
        assert hasattr(index, "list_schemas")
        assert hasattr(index, "decode_schema")

        assert callable(index.insert_dataset)
        assert callable(index.get_dataset)
        assert callable(index.list_datasets)
        assert callable(index.publish_schema)
        assert callable(index.get_schema)
        assert callable(index.list_schemas)
        assert callable(index.decode_schema)


class TestAbstractDataStoreProtocol:
    """Tests for AbstractDataStore protocol compliance."""

    def test_s3_datastore_has_required_methods(self):
        """S3DataStore should have all AbstractDataStore methods."""
        # Create with mock credentials
        mock_creds = {
            "AWS_ENDPOINT": "http://localhost:9000",
            "AWS_ACCESS_KEY_ID": "test",
            "AWS_SECRET_ACCESS_KEY": "test",
        }

        store = S3DataStore(mock_creds, bucket="test-bucket")

        assert hasattr(store, "write_shards")
        assert hasattr(store, "read_url")
        assert hasattr(store, "supports_streaming")

        assert callable(store.write_shards)
        assert callable(store.read_url)
        assert callable(store.supports_streaming)

    def test_s3_datastore_supports_streaming(self):
        """S3DataStore should report streaming support."""
        mock_creds = {
            "AWS_ENDPOINT": "http://localhost:9000",
            "AWS_ACCESS_KEY_ID": "test",
            "AWS_SECRET_ACCESS_KEY": "test",
        }

        store = S3DataStore(mock_creds, bucket="test-bucket")
        assert store.supports_streaming() is True

    def test_s3_datastore_read_url_passthrough(self):
        """S3DataStore.read_url should return URL unchanged without custom endpoint."""
        mock_creds = {
            "AWS_ACCESS_KEY_ID": "test",
            "AWS_SECRET_ACCESS_KEY": "test",
        }

        store = S3DataStore(mock_creds, bucket="test-bucket")
        url = "s3://bucket/path/data.tar"
        assert store.read_url(url) == url

    def test_s3_datastore_read_url_transforms_with_endpoint(self):
        """S3DataStore.read_url should transform s3:// to https:// with custom endpoint."""
        mock_creds = {
            "AWS_ENDPOINT": "http://localhost:9000",
            "AWS_ACCESS_KEY_ID": "test",
            "AWS_SECRET_ACCESS_KEY": "test",
        }

        store = S3DataStore(mock_creds, bucket="test-bucket")
        url = "s3://bucket/path/data.tar"
        # URL should be transformed to use the custom endpoint
        assert store.read_url(url) == "http://localhost:9000/bucket/path/data.tar"


class TestProtocolInteroperability:
    """Tests verifying different implementations can be used interchangeably."""

    def test_function_accepts_any_index_entry(self):
        """Functions typed with IndexEntry should accept any implementation."""

        def get_dataset_name(entry: IndexEntry) -> str:
            return entry.name

        # LocalDatasetEntry
        local_entry = LocalDatasetEntry(
            name="local-data",
            schema_ref="local://schemas/test@1.0.0",
            data_urls=["s3://bucket/data.tar"],
        )
        assert get_dataset_name(local_entry) == "local-data"

        # AtmosphereIndexEntry
        atmo_entry = AtmosphereIndexEntry(
            "at://did:plc:test/record/xyz",
            {"name": "atmo-data", "schemaRef": "at://schema", "storage": {}},
        )
        assert get_dataset_name(atmo_entry) == "atmo-data"

    def test_function_accepts_any_index(self):
        """Functions typed with AbstractIndex should accept any implementation."""

        def count_datasets(index) -> int:
            """Count datasets in an index."""
            return sum(1 for _ in index.list_datasets())

        # Index with default SQLite
        local_index = Index()
        # Empty index returns 0
        assert count_datasets(local_index) == 0

    def test_index_entry_properties_consistent(self):
        """All IndexEntry implementations should have consistent property types."""
        local_entry = LocalDatasetEntry(
            name="test",
            schema_ref="local://schemas/test@1.0.0",
            data_urls=["url1", "url2"],
            metadata={"k": "v"},
        )

        atmo_entry = AtmosphereIndexEntry(
            "at://test",
            {
                "name": "test",
                "schemaRef": "at://schema",
                "storage": {
                    "$type": "ac.foundation.dataset.storageExternal",
                    "urls": ["url1", "url2"],
                },
            },
        )

        # Both should return str for name
        assert isinstance(local_entry.name, str)
        assert isinstance(atmo_entry.name, str)

        # Both should return str for schema_ref
        assert isinstance(local_entry.schema_ref, str)
        assert isinstance(atmo_entry.schema_ref, str)

        # Both should return list[str] for data_urls
        assert isinstance(local_entry.data_urls, list)
        assert isinstance(atmo_entry.data_urls, list)
        assert all(isinstance(u, str) for u in local_entry.data_urls)
        assert all(isinstance(u, str) for u in atmo_entry.data_urls)


class TestPolymorphicBehavior:
    """Tests that verify actual polymorphic usage patterns work correctly."""

    def test_process_entries_polymorphically(self):
        """Process a mixed list of IndexEntry implementations uniformly."""
        entries: list[IndexEntry] = [
            LocalDatasetEntry(
                name="local-1",
                schema_ref="local://schemas/A@1.0.0",
                data_urls=["s3://bucket/local1.tar"],
                metadata={"source": "local"},
            ),
            AtmosphereIndexEntry(
                "at://did:plc:test/record/1",
                {
                    "name": "atmo-1",
                    "schemaRef": "at://did:plc:test/schema/A",
                    "storage": {
                        "$type": "ac.foundation.dataset.storageExternal",
                        "urls": ["s3://bucket/atmo1.tar"],
                    },
                },
            ),
            LocalDatasetEntry(
                name="local-2",
                schema_ref="local://schemas/B@1.0.0",
                data_urls=["s3://bucket/local2.tar", "s3://bucket/local2-001.tar"],
            ),
        ]

        # Extract all names uniformly
        names = [e.name for e in entries]
        assert names == ["local-1", "atmo-1", "local-2"]

        # Extract all schema refs
        schema_refs = [e.schema_ref for e in entries]
        assert schema_refs == [
            "local://schemas/A@1.0.0",
            "at://did:plc:test/schema/A",
            "local://schemas/B@1.0.0",
        ]

        # Count total shards across all entries
        total_urls = sum(len(e.data_urls) for e in entries)
        assert total_urls == 4

        # Filter by metadata presence
        with_metadata = [e for e in entries if e.metadata is not None]
        assert len(with_metadata) == 1
        assert with_metadata[0].name == "local-1"

    def test_index_entry_in_dict_key(self):
        """IndexEntry.name can be used to build lookup structures."""
        entries: list[IndexEntry] = [
            LocalDatasetEntry(
                name="dataset-a",
                schema_ref="local://schemas/A@1.0.0",
                data_urls=["url1"],
            ),
            AtmosphereIndexEntry(
                "at://test",
                {
                    "name": "dataset-b",
                    "schemaRef": "at://schema",
                    "storage": {
                        "$type": "ac.foundation.dataset.storageExternal",
                        "urls": ["url2"],
                    },
                },
            ),
        ]

        # Build a lookup by name
        by_name: dict[str, IndexEntry] = {e.name: e for e in entries}

        assert "dataset-a" in by_name
        assert "dataset-b" in by_name
        assert by_name["dataset-a"].data_urls == ["url1"]
        assert by_name["dataset-b"].data_urls == ["url2"]

    def test_generic_url_collector(self):
        """A generic function can collect URLs from any IndexEntry."""

        def collect_all_urls(entries: list[IndexEntry]) -> list[str]:
            """Collect all data URLs from a list of entries."""
            all_urls = []
            for entry in entries:
                all_urls.extend(entry.data_urls)
            return all_urls

        mixed_entries: list[IndexEntry] = [
            LocalDatasetEntry(
                name="ds1",
                schema_ref="local://test@1.0.0",
                data_urls=["s3://a/1.tar", "s3://a/2.tar"],
            ),
            AtmosphereIndexEntry(
                "at://x",
                {
                    "name": "ds2",
                    "schemaRef": "at://s",
                    "storage": {
                        "$type": "ac.foundation.dataset.storageExternal",
                        "urls": ["s3://b/1.tar"],
                    },
                },
            ),
        ]

        urls = collect_all_urls(mixed_entries)
        assert urls == ["s3://a/1.tar", "s3://a/2.tar", "s3://b/1.tar"]
