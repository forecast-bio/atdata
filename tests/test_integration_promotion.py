"""Integration tests for the promotion pipeline (local → atmosphere).

Tests end-to-end promotion workflows including:
- Full promotion with local index and mocked atmosphere
- Schema deduplication across multiple promotions
- Metadata preservation during promotion
- Multi-dataset promotion with shared schemas
- Large dataset handling with many shards
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

from numpy.typing import NDArray
import webdataset as wds

import atdata
from atdata.local import Index, LocalDatasetEntry
from atdata.promote import promote_to_atmosphere
from atdata.atmosphere import AtmosphereClient
from atdata.atmosphere._types import LEXICON_NAMESPACE


##
# Test sample types


@atdata.packable
class PromotionSample:
    """Sample for promotion tests."""

    name: str
    value: int


@atdata.packable
class PromotionArraySample:
    """Sample with NDArray for promotion tests."""

    label: str
    features: NDArray


##
# Fixtures


@pytest.fixture
def mock_atproto_client():
    """Create a mock atproto SDK client."""
    mock = Mock()
    mock.me = MagicMock()
    mock.me.did = "did:plc:promotion123"
    mock.me.handle = "promotion.test.social"

    mock_profile = Mock()
    mock_profile.did = "did:plc:promotion123"
    mock_profile.handle = "promotion.test.social"
    mock.login.return_value = mock_profile
    mock.export_session_string.return_value = "test-session-export"

    return mock


@pytest.fixture
def authenticated_client(mock_atproto_client):
    """Create an authenticated AtmosphereClient."""
    client = AtmosphereClient(_client=mock_atproto_client)
    client.login("promotion.test.social", "test-password")
    return client


@pytest.fixture
def local_index_with_data(clean_redis, tmp_path):
    """Create a Index with a sample dataset."""
    index = Index(redis=clean_redis)

    # Publish schema
    schema_ref = index.publish_schema(PromotionSample, version="1.0.0")

    # Create a tar file with samples
    tar_path = tmp_path / "promotion-test-000000.tar"
    with wds.writer.TarWriter(str(tar_path)) as writer:
        for i in range(5):
            sample = PromotionSample(name=f"sample-{i}", value=i * 10)
            writer.write(sample.as_wds)

    # Create entry
    entry = LocalDatasetEntry(
        name="promotion-test-dataset",
        schema_ref=schema_ref,
        data_urls=[str(tar_path)],
        metadata={"version": "1.0", "sample_count": 5},
    )
    entry.write_to(clean_redis)

    return index, entry


##
# Full Promotion Workflow Tests


class TestFullPromotionWorkflow:
    """End-to-end tests for promotion workflow."""

    def test_promote_local_to_atmosphere(
        self, local_index_with_data, authenticated_client, mock_atproto_client
    ):
        """Full workflow: Index dataset → promote → AtmosphereIndex."""
        local_index, local_entry = local_index_with_data

        # Setup mock responses for atmosphere operations
        schema_response = Mock()
        schema_response.uri = (
            f"at://did:plc:test/{LEXICON_NAMESPACE}.sampleSchema/promoted-schema"
        )

        dataset_response = Mock()
        dataset_response.uri = (
            f"at://did:plc:test/{LEXICON_NAMESPACE}.dataset/promoted-dataset"
        )

        mock_atproto_client.com.atproto.repo.create_record.side_effect = [
            schema_response,
            dataset_response,
        ]

        # Mock list_records to return empty (no existing schema)
        mock_list_response = Mock()
        mock_list_response.records = []
        mock_list_response.cursor = None
        mock_atproto_client.com.atproto.repo.list_records.return_value = (
            mock_list_response
        )

        # Promote
        result_uri = promote_to_atmosphere(
            local_entry,
            local_index,
            authenticated_client,
        )

        assert result_uri is not None
        assert "at://" in result_uri

    def test_promoted_dataset_preserves_name(
        self, local_index_with_data, authenticated_client, mock_atproto_client
    ):
        """Promoted dataset should preserve the original name."""
        local_index, local_entry = local_index_with_data

        # Setup mocks
        mock_list_response = Mock()
        mock_list_response.records = []
        mock_list_response.cursor = None
        mock_atproto_client.com.atproto.repo.list_records.return_value = (
            mock_list_response
        )

        schema_response = Mock()
        schema_response.uri = f"at://did:plc:test/{LEXICON_NAMESPACE}.sampleSchema/s1"

        dataset_response = Mock()
        dataset_response.uri = f"at://did:plc:test/{LEXICON_NAMESPACE}.dataset/d1"

        mock_atproto_client.com.atproto.repo.create_record.side_effect = [
            schema_response,
            dataset_response,
        ]

        promote_to_atmosphere(local_entry, local_index, authenticated_client)

        # Check that dataset was published with correct name
        calls = mock_atproto_client.com.atproto.repo.create_record.call_args_list
        dataset_call = calls[-1]  # Last call is for dataset
        record = dataset_call.kwargs["data"]["record"]
        assert record["name"] == "promotion-test-dataset"

    def test_promoted_dataset_preserves_data_urls(
        self, local_index_with_data, authenticated_client, mock_atproto_client
    ):
        """Promoted dataset should use the original data URLs."""
        local_index, local_entry = local_index_with_data

        # Setup mocks
        mock_list_response = Mock()
        mock_list_response.records = []
        mock_list_response.cursor = None
        mock_atproto_client.com.atproto.repo.list_records.return_value = (
            mock_list_response
        )

        schema_response = Mock()
        schema_response.uri = f"at://did:plc:test/{LEXICON_NAMESPACE}.sampleSchema/s1"

        dataset_response = Mock()
        dataset_response.uri = f"at://did:plc:test/{LEXICON_NAMESPACE}.dataset/d1"

        mock_atproto_client.com.atproto.repo.create_record.side_effect = [
            schema_response,
            dataset_response,
        ]

        promote_to_atmosphere(local_entry, local_index, authenticated_client)

        # Check that dataset was published with original URLs
        calls = mock_atproto_client.com.atproto.repo.create_record.call_args_list
        dataset_call = calls[-1]
        record = dataset_call.kwargs["data"]["record"]
        assert "storage" in record
        assert local_entry.data_urls[0] in str(record["storage"]["urls"])


##
# Schema Deduplication Tests


class TestSchemaDeduplication:
    """Tests for schema deduplication during promotion."""

    def test_reuses_existing_schema(
        self, local_index_with_data, authenticated_client, mock_atproto_client
    ):
        """Promotion should reuse existing schema instead of creating duplicate."""
        local_index, local_entry = local_index_with_data

        # Patch _find_existing_schema to return an existing schema URI
        with patch("atdata.promote._find_existing_schema") as mock_find:
            mock_find.return_value = (
                f"at://did:plc:test/{LEXICON_NAMESPACE}.sampleSchema/existing"
            )

            # Only dataset should be created (schema exists)
            dataset_response = Mock()
            dataset_response.uri = f"at://did:plc:test/{LEXICON_NAMESPACE}.dataset/d1"
            mock_atproto_client.com.atproto.repo.create_record.return_value = (
                dataset_response
            )

            promote_to_atmosphere(local_entry, local_index, authenticated_client)

            # Should only have 1 create_record call (for dataset, not schema)
            assert mock_atproto_client.com.atproto.repo.create_record.call_count == 1

            # Verify it was the dataset call
            call_kwargs = (
                mock_atproto_client.com.atproto.repo.create_record.call_args.kwargs
            )
            assert "dataset" in call_kwargs["data"]["collection"]

    def test_creates_schema_when_not_found(
        self, local_index_with_data, authenticated_client, mock_atproto_client
    ):
        """Promotion should create new schema when none exists."""
        local_index, local_entry = local_index_with_data

        # Mock empty list (no existing schemas)
        mock_list_response = Mock()
        mock_list_response.records = []
        mock_list_response.cursor = None
        mock_atproto_client.com.atproto.repo.list_records.return_value = (
            mock_list_response
        )

        # Both schema and dataset should be created
        schema_response = Mock()
        schema_response.uri = f"at://did:plc:test/{LEXICON_NAMESPACE}.sampleSchema/new"

        dataset_response = Mock()
        dataset_response.uri = f"at://did:plc:test/{LEXICON_NAMESPACE}.dataset/d1"

        mock_atproto_client.com.atproto.repo.create_record.side_effect = [
            schema_response,
            dataset_response,
        ]

        promote_to_atmosphere(local_entry, local_index, authenticated_client)

        # Should have 2 create_record calls (schema + dataset)
        assert mock_atproto_client.com.atproto.repo.create_record.call_count == 2

    def test_version_mismatch_creates_new_schema(
        self, local_index_with_data, authenticated_client, mock_atproto_client
    ):
        """Different version should create new schema even if name matches."""
        local_index, local_entry = local_index_with_data

        # Mock existing schema with different version
        existing_schema = Mock()
        existing_schema.uri = f"at://did:plc:test/{LEXICON_NAMESPACE}.sampleSchema/v1"
        existing_schema.value = {
            "name": "test_integration_promotion.PromotionSample",
            "version": "2.0.0",  # Different version!
        }

        mock_list_response = Mock()
        mock_list_response.records = [existing_schema]
        mock_list_response.cursor = None
        mock_atproto_client.com.atproto.repo.list_records.return_value = (
            mock_list_response
        )

        # Both should be created (version mismatch)
        schema_response = Mock()
        schema_response.uri = (
            f"at://did:plc:test/{LEXICON_NAMESPACE}.sampleSchema/v1new"
        )

        dataset_response = Mock()
        dataset_response.uri = f"at://did:plc:test/{LEXICON_NAMESPACE}.dataset/d1"

        mock_atproto_client.com.atproto.repo.create_record.side_effect = [
            schema_response,
            dataset_response,
        ]

        promote_to_atmosphere(local_entry, local_index, authenticated_client)

        # Should have 2 create_record calls (new schema + dataset)
        assert mock_atproto_client.com.atproto.repo.create_record.call_count == 2


##
# Metadata Preservation Tests


class TestMetadataPreservation:
    """Tests for metadata preservation during promotion."""

    def test_metadata_included_in_promoted_dataset(
        self, local_index_with_data, authenticated_client, mock_atproto_client
    ):
        """Metadata from local entry should be included in promoted dataset."""
        local_index, local_entry = local_index_with_data

        # Setup mocks
        mock_list_response = Mock()
        mock_list_response.records = []
        mock_list_response.cursor = None
        mock_atproto_client.com.atproto.repo.list_records.return_value = (
            mock_list_response
        )

        schema_response = Mock()
        schema_response.uri = f"at://did:plc:test/{LEXICON_NAMESPACE}.sampleSchema/s1"

        dataset_response = Mock()
        dataset_response.uri = f"at://did:plc:test/{LEXICON_NAMESPACE}.dataset/d1"

        mock_atproto_client.com.atproto.repo.create_record.side_effect = [
            schema_response,
            dataset_response,
        ]

        promote_to_atmosphere(local_entry, local_index, authenticated_client)

        # Check metadata was passed
        calls = mock_atproto_client.com.atproto.repo.create_record.call_args_list
        dataset_call = calls[-1]
        record = dataset_call.kwargs["data"]["record"]

        # The metadata should be in the record (may be msgpack encoded)
        assert "metadata" in record

    def test_none_metadata_handled(
        self, clean_redis, authenticated_client, mock_atproto_client
    ):
        """Entry without metadata should promote successfully."""
        index = Index(redis=clean_redis)
        schema_ref = index.publish_schema(PromotionSample, version="1.0.0")

        entry = LocalDatasetEntry(
            name="no-metadata-dataset",
            schema_ref=schema_ref,
            data_urls=["s3://bucket/data.tar"],
            # No _metadata specified
        )
        entry.write_to(clean_redis)

        # Setup mocks
        mock_list_response = Mock()
        mock_list_response.records = []
        mock_list_response.cursor = None
        mock_atproto_client.com.atproto.repo.list_records.return_value = (
            mock_list_response
        )

        schema_response = Mock()
        schema_response.uri = f"at://did:plc:test/{LEXICON_NAMESPACE}.sampleSchema/s1"

        dataset_response = Mock()
        dataset_response.uri = f"at://did:plc:test/{LEXICON_NAMESPACE}.dataset/d1"

        mock_atproto_client.com.atproto.repo.create_record.side_effect = [
            schema_response,
            dataset_response,
        ]

        # Should not raise
        result = promote_to_atmosphere(entry, index, authenticated_client)
        assert isinstance(result, str)


##
# Multi-Dataset Promotion Tests


class TestMultiDatasetPromotion:
    """Tests for promoting multiple datasets."""

    def test_multiple_datasets_share_schema(
        self, clean_redis, authenticated_client, mock_atproto_client
    ):
        """Multiple datasets using same schema should reuse the schema."""
        index = Index(redis=clean_redis)
        schema_ref = index.publish_schema(PromotionSample, version="1.0.0")

        # Create multiple entries with same schema
        entries = []
        for i in range(3):
            entry = LocalDatasetEntry(
                name=f"dataset-{i}",
                schema_ref=schema_ref,
                data_urls=[f"s3://bucket/data-{i}.tar"],
            )
            entry.write_to(clean_redis)
            entries.append(entry)

        # Track whether schema has been "published" to atmosphere
        schema_published = {"value": False}
        schema_uri = f"at://did:plc:test/{LEXICON_NAMESPACE}.sampleSchema/shared"

        def mock_find_existing(client, name, version):
            # Return schema URI after first promotion
            if schema_published["value"]:
                return schema_uri
            return None

        # Setup create_record responses
        schema_response = Mock()
        schema_response.uri = schema_uri

        dataset_responses = [
            Mock(uri=f"at://did:plc:test/{LEXICON_NAMESPACE}.dataset/d{i}")
            for i in range(3)
        ]

        mock_atproto_client.com.atproto.repo.create_record.side_effect = [
            schema_response,  # First promotion creates schema
            dataset_responses[0],  # First dataset
            dataset_responses[1],  # Second dataset
            dataset_responses[2],  # Third dataset
        ]

        with patch(
            "atdata.promote._find_existing_schema", side_effect=mock_find_existing
        ):
            # Promote all three
            for i, entry in enumerate(entries):
                promote_to_atmosphere(entry, index, authenticated_client)
                # After first promotion, schema exists
                if i == 0:
                    schema_published["value"] = True

        # Should have 4 create_record calls: 1 schema + 3 datasets
        assert mock_atproto_client.com.atproto.repo.create_record.call_count == 4


##
# Large Dataset Tests


class TestLargeDatasetPromotion:
    """Tests for promoting datasets with many shards."""

    def test_many_shards_promoted(
        self, clean_redis, authenticated_client, mock_atproto_client
    ):
        """Dataset with many shards should have all URLs promoted."""
        index = Index(redis=clean_redis)
        schema_ref = index.publish_schema(PromotionSample, version="1.0.0")

        # Create entry with many shards
        shard_urls = [f"s3://bucket/shard-{i:06d}.tar" for i in range(100)]
        entry = LocalDatasetEntry(
            name="large-dataset",
            schema_ref=schema_ref,
            data_urls=shard_urls,
        )
        entry.write_to(clean_redis)

        # Setup mocks
        mock_list_response = Mock()
        mock_list_response.records = []
        mock_list_response.cursor = None
        mock_atproto_client.com.atproto.repo.list_records.return_value = (
            mock_list_response
        )

        schema_response = Mock()
        schema_response.uri = f"at://did:plc:test/{LEXICON_NAMESPACE}.sampleSchema/s1"

        dataset_response = Mock()
        dataset_response.uri = f"at://did:plc:test/{LEXICON_NAMESPACE}.dataset/large"

        mock_atproto_client.com.atproto.repo.create_record.side_effect = [
            schema_response,
            dataset_response,
        ]

        promote_to_atmosphere(entry, index, authenticated_client)

        # Verify all 100 URLs were included
        calls = mock_atproto_client.com.atproto.repo.create_record.call_args_list
        dataset_call = calls[-1]
        record = dataset_call.kwargs["data"]["record"]
        storage_urls = record["storage"]["urls"]

        assert len(storage_urls) == 100
        assert storage_urls[0] == "s3://bucket/shard-000000.tar"
        assert storage_urls[99] == "s3://bucket/shard-000099.tar"


##
# Error Handling Tests


class TestPromotionErrors:
    """Tests for error handling during promotion."""

    def test_empty_data_urls_raises(self, clean_redis, authenticated_client):
        """Promotion of entry with no data URLs should raise."""
        index = Index(redis=clean_redis)
        schema_ref = index.publish_schema(PromotionSample, version="1.0.0")

        entry = LocalDatasetEntry(
            name="empty-dataset",
            schema_ref=schema_ref,
            data_urls=[],
        )

        with pytest.raises(ValueError, match="has no data URLs"):
            promote_to_atmosphere(entry, index, authenticated_client)

    def test_missing_schema_raises(
        self, clean_redis, authenticated_client, mock_atproto_client
    ):
        """Promotion with missing local schema should raise."""
        index = Index(redis=clean_redis)

        # Entry references a schema that doesn't exist
        entry = LocalDatasetEntry(
            name="orphan-dataset",
            schema_ref="local://schemas/NonExistent@1.0.0",
            data_urls=["s3://bucket/data.tar"],
        )

        with pytest.raises(KeyError):
            promote_to_atmosphere(entry, index, authenticated_client)


##
# Custom Options Tests


class TestPromotionOptions:
    """Tests for promotion with custom options."""

    def test_custom_name_override(
        self, local_index_with_data, authenticated_client, mock_atproto_client
    ):
        """Custom name should override local entry name."""
        local_index, local_entry = local_index_with_data

        # Setup mocks
        mock_list_response = Mock()
        mock_list_response.records = []
        mock_list_response.cursor = None
        mock_atproto_client.com.atproto.repo.list_records.return_value = (
            mock_list_response
        )

        schema_response = Mock()
        schema_response.uri = f"at://did:plc:test/{LEXICON_NAMESPACE}.sampleSchema/s1"

        dataset_response = Mock()
        dataset_response.uri = f"at://did:plc:test/{LEXICON_NAMESPACE}.dataset/d1"

        mock_atproto_client.com.atproto.repo.create_record.side_effect = [
            schema_response,
            dataset_response,
        ]

        promote_to_atmosphere(
            local_entry,
            local_index,
            authenticated_client,
            name="custom-promoted-name",
        )

        calls = mock_atproto_client.com.atproto.repo.create_record.call_args_list
        dataset_call = calls[-1]
        record = dataset_call.kwargs["data"]["record"]
        assert record["name"] == "custom-promoted-name"

    def test_tags_and_license(
        self, local_index_with_data, authenticated_client, mock_atproto_client
    ):
        """Tags and license should be passed to promoted dataset."""
        local_index, local_entry = local_index_with_data

        # Setup mocks
        mock_list_response = Mock()
        mock_list_response.records = []
        mock_list_response.cursor = None
        mock_atproto_client.com.atproto.repo.list_records.return_value = (
            mock_list_response
        )

        schema_response = Mock()
        schema_response.uri = f"at://did:plc:test/{LEXICON_NAMESPACE}.sampleSchema/s1"

        dataset_response = Mock()
        dataset_response.uri = f"at://did:plc:test/{LEXICON_NAMESPACE}.dataset/d1"

        mock_atproto_client.com.atproto.repo.create_record.side_effect = [
            schema_response,
            dataset_response,
        ]

        promote_to_atmosphere(
            local_entry,
            local_index,
            authenticated_client,
            tags=["ml", "training", "images"],
            license="Apache-2.0",
        )

        calls = mock_atproto_client.com.atproto.repo.create_record.call_args_list
        dataset_call = calls[-1]
        record = dataset_call.kwargs["data"]["record"]

        assert record.get("tags") == ["ml", "training", "images"]
        assert record.get("license") == "Apache-2.0"

    def test_description_passed(
        self, local_index_with_data, authenticated_client, mock_atproto_client
    ):
        """Description should be passed to promoted dataset."""
        local_index, local_entry = local_index_with_data

        # Setup mocks
        mock_list_response = Mock()
        mock_list_response.records = []
        mock_list_response.cursor = None
        mock_atproto_client.com.atproto.repo.list_records.return_value = (
            mock_list_response
        )

        schema_response = Mock()
        schema_response.uri = f"at://did:plc:test/{LEXICON_NAMESPACE}.sampleSchema/s1"

        dataset_response = Mock()
        dataset_response.uri = f"at://did:plc:test/{LEXICON_NAMESPACE}.dataset/d1"

        mock_atproto_client.com.atproto.repo.create_record.side_effect = [
            schema_response,
            dataset_response,
        ]

        promote_to_atmosphere(
            local_entry,
            local_index,
            authenticated_client,
            description="A promoted dataset for testing purposes.",
        )

        calls = mock_atproto_client.com.atproto.repo.create_record.call_args_list
        dataset_call = calls[-1]
        record = dataset_call.kwargs["data"]["record"]

        assert record.get("description") == "A promoted dataset for testing purposes."
