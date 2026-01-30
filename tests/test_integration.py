"""Integration tests for atdata local-atmosphere workflows.

These tests verify end-to-end workflows spanning local and atmosphere
components, using mocks for external services (Redis, ATProto PDS).
"""

from unittest.mock import Mock, patch

import webdataset as wds

import atdata
from atdata.local import LocalDatasetEntry
from atdata.promote import promote_to_atmosphere


@atdata.packable
class IntegrationTestSample:
    """Sample type for integration tests."""

    name: str
    value: int


class TestLocalToAtmosphereRoundTrip:
    """Integration tests for local → atmosphere promotion workflow."""

    def test_promote_preserves_data_urls(self, tmp_path):
        """Promote should preserve data URLs when no data_store provided."""
        # Create a local dataset entry
        local_entry = LocalDatasetEntry(
            name="test-dataset",
            schema_ref="local://schemas/test_integration.IntegrationTestSample@1.0.0",
            data_urls=["s3://bucket/data-000000.tar", "s3://bucket/data-000001.tar"],
            metadata={"source": "test"},
        )

        # Mock local index with schema
        mock_local_index = Mock()
        mock_local_index.get_schema.return_value = {
            "name": "test_integration.IntegrationTestSample",
            "version": "1.0.0",
            "fields": [
                {
                    "name": "name",
                    "fieldType": {"$type": "local#primitive", "primitive": "str"},
                    "optional": False,
                },
                {
                    "name": "value",
                    "fieldType": {"$type": "local#primitive", "primitive": "int"},
                    "optional": False,
                },
            ],
        }

        # Mock atmosphere client
        mock_client = Mock()

        with patch("atdata.promote._find_or_publish_schema") as mock_find_schema:
            mock_find_schema.return_value = "at://did:plc:test/schema/abc"

            with patch("atdata.atmosphere.DatasetPublisher") as MockPublisher:
                mock_publisher = MockPublisher.return_value
                mock_publisher.publish_with_urls.return_value = Mock(
                    __str__=lambda s: "at://did:plc:test/record/xyz"
                )

                promote_to_atmosphere(
                    local_entry,
                    mock_local_index,
                    mock_client,
                )

                # Verify data URLs were preserved
                call_kwargs = mock_publisher.publish_with_urls.call_args[1]
                assert call_kwargs["urls"] == [
                    "s3://bucket/data-000000.tar",
                    "s3://bucket/data-000001.tar",
                ]

                # Verify metadata preserved
                assert call_kwargs["metadata"] == {"source": "test"}

    def test_promote_transfers_schema_metadata(self, tmp_path):
        """Promote should use schema version from local index."""
        local_entry = LocalDatasetEntry(
            name="versioned-dataset",
            schema_ref="local://schemas/MySample@2.1.0",
            data_urls=["s3://bucket/data.tar"],
        )

        mock_local_index = Mock()
        mock_local_index.get_schema.return_value = {
            "name": "MySample",
            "version": "2.1.0",
            "description": "A sample with specific version",
            "fields": [
                {
                    "name": "value",
                    "fieldType": {"$type": "local#primitive", "primitive": "int"},
                    "optional": False,
                },
            ],
        }

        mock_client = Mock()

        with patch("atdata.promote._find_or_publish_schema") as mock_find_schema:
            mock_find_schema.return_value = "at://schema"

            with patch("atdata.atmosphere.DatasetPublisher") as MockPublisher:
                mock_publisher = MockPublisher.return_value
                mock_publisher.publish_with_urls.return_value = Mock(
                    __str__=lambda s: "at://result"
                )

                promote_to_atmosphere(local_entry, mock_local_index, mock_client)

                # Verify _find_or_publish_schema was called with correct version
                call_args = mock_find_schema.call_args
                assert call_args[0][1] == "2.1.0"  # version argument


class TestSchemaDeduplication:
    """Tests for schema deduplication during promotion."""

    def test_existing_schema_reused(self):
        """Promoting with existing schema should reuse it, not create duplicate."""
        from atdata.promote import _find_or_publish_schema

        mock_client = Mock()

        # Mock finding an existing schema
        with patch("atdata.promote._find_existing_schema") as mock_find:
            mock_find.return_value = "at://did:plc:test/schema/existing"

            with patch("atdata.atmosphere.SchemaPublisher") as MockPublisher:
                result = _find_or_publish_schema(
                    IntegrationTestSample,
                    "1.0.0",
                    mock_client,
                )

                # Should return existing URI
                assert result == "at://did:plc:test/schema/existing"

                # Should NOT have called publish
                MockPublisher.return_value.publish.assert_not_called()

    def test_new_schema_published_when_not_found(self):
        """Promoting without existing schema should publish new one."""
        from atdata.promote import _find_or_publish_schema

        mock_client = Mock()

        with patch("atdata.promote._find_existing_schema") as mock_find:
            mock_find.return_value = None  # No existing schema

            with patch("atdata.atmosphere.SchemaPublisher") as MockPublisher:
                mock_publisher = MockPublisher.return_value
                mock_publisher.publish.return_value = Mock(
                    __str__=lambda s: "at://did:plc:test/schema/new"
                )

                result = _find_or_publish_schema(
                    IntegrationTestSample,
                    "1.0.0",
                    mock_client,
                )

                # Should return new URI
                assert result == "at://did:plc:test/schema/new"

                # Should have called publish
                mock_publisher.publish.assert_called_once()

    def test_version_mismatch_creates_new_schema(self):
        """Different version should create new schema even if name matches."""
        from atdata.promote import _find_existing_schema

        mock_client = Mock()

        with patch("atdata.atmosphere.SchemaLoader") as MockLoader:
            mock_loader = MockLoader.return_value
            mock_loader.list_all.return_value = [
                {
                    "uri": "at://did:plc:test/schema/v1",
                    "value": {
                        "name": "test_integration.IntegrationTestSample",
                        "version": "1.0.0",  # Different version
                    },
                }
            ]

            # Looking for version 2.0.0
            result = _find_existing_schema(
                mock_client,
                "test_integration.IntegrationTestSample",
                "2.0.0",
            )

            # Should not find the v1 schema
            assert result is None


class TestLoadDatasetWithIndex:
    """Integration tests for load_dataset with index parameter."""

    def test_load_from_local_index(self, tmp_path):
        """load_dataset with Index should resolve dataset."""
        # Create actual test data
        wds_file = tmp_path / "test-data.tar"
        with wds.writer.TarWriter(str(wds_file)) as sink:
            sample = IntegrationTestSample(name="test", value=42)
            sink.write(sample.as_wds)

        # Create local index entry
        local_entry = LocalDatasetEntry(
            name="my-dataset",
            schema_ref="local://schemas/IntegrationTestSample@1.0.0",
            data_urls=[str(wds_file)],
        )

        # Mock index
        mock_index = Mock()
        mock_index.data_store = None  # No data store, so no URL transformation
        mock_index.get_dataset.return_value = local_entry
        mock_index.decode_schema.return_value = IntegrationTestSample

        # Load via index
        ds = atdata.load_dataset(
            "@local/my-dataset",
            index=mock_index,
            split="train",
        )

        # Should return a Dataset
        assert isinstance(ds, atdata.Dataset)

        # Should be able to iterate (batch_size=None for individual samples)
        samples = list(ds.ordered(batch_size=None))
        assert len(samples) == 1
        assert samples[0].name == "test"
        assert samples[0].value == 42

    def test_load_with_explicit_sample_type(self, tmp_path):
        """load_dataset with explicit sample_type should use it."""
        wds_file = tmp_path / "typed-data.tar"
        with wds.writer.TarWriter(str(wds_file)) as sink:
            sample = IntegrationTestSample(name="explicit", value=100)
            sink.write(sample.as_wds)

        local_entry = LocalDatasetEntry(
            name="typed-dataset",
            schema_ref="local://schemas/IntegrationTestSample@1.0.0",
            data_urls=[str(wds_file)],
        )

        mock_index = Mock()
        mock_index.data_store = None  # No data store, so no URL transformation
        mock_index.get_dataset.return_value = local_entry

        # Load with explicit type (should not call decode_schema)
        ds = atdata.load_dataset(
            "@local/typed-dataset",
            IntegrationTestSample,
            index=mock_index,
            split="train",
        )

        # decode_schema should not be called when type is explicit
        mock_index.decode_schema.assert_not_called()

        samples = list(ds.ordered(batch_size=None))
        assert len(samples) == 1
        assert isinstance(samples[0], IntegrationTestSample)


class TestIndexEntryRoundTrip:
    """Tests for index entry serialization round-trips."""

    def test_local_entry_redis_round_trip(self, clean_redis):
        """LocalDatasetEntry should round-trip through Redis correctly."""
        original = LocalDatasetEntry(
            name="roundtrip-test",
            schema_ref="local://schemas/Test@1.0.0",
            data_urls=["s3://bucket/shard-000.tar", "s3://bucket/shard-001.tar"],
            metadata={"key": "value", "count": 42},
        )

        # Write to Redis
        original.write_to(clean_redis)

        # Read back
        loaded = LocalDatasetEntry.from_redis(clean_redis, original.cid)

        # Verify all fields match
        assert loaded.name == original.name
        assert loaded.schema_ref == original.schema_ref
        assert loaded.data_urls == original.data_urls
        assert loaded.metadata == original.metadata
        assert loaded.cid == original.cid

    def test_local_entry_cid_deterministic(self):
        """Same content should produce same CID."""
        entry1 = LocalDatasetEntry(
            name="deterministic",
            schema_ref="local://schemas/Test@1.0.0",
            data_urls=["s3://bucket/data.tar"],
        )

        entry2 = LocalDatasetEntry(
            name="deterministic",
            schema_ref="local://schemas/Test@1.0.0",
            data_urls=["s3://bucket/data.tar"],
        )

        # CIDs should match (based on schema_ref and data_urls)
        assert entry1.cid == entry2.cid

    def test_local_entry_cid_differs_with_content(self):
        """Different content should produce different CID."""
        entry1 = LocalDatasetEntry(
            name="same-name",
            schema_ref="local://schemas/Test@1.0.0",
            data_urls=["s3://bucket/data-v1.tar"],
        )

        entry2 = LocalDatasetEntry(
            name="same-name",
            schema_ref="local://schemas/Test@1.0.0",
            data_urls=["s3://bucket/data-v2.tar"],  # Different URL
        )

        # CIDs should differ
        assert entry1.cid != entry2.cid
