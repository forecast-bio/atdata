"""Tests for the promote module."""

import pytest
from unittest.mock import Mock, patch

import atdata
from atdata.promote import (
    promote_to_atmosphere,
    _find_existing_schema,
    _find_or_publish_schema,
)
from atdata.local import LocalDatasetEntry


@atdata.packable
class PromoteTestSample:
    """Sample type for promotion tests."""

    name: str
    value: int


class TestFindExistingSchema:
    """Tests for _find_existing_schema helper."""

    def test_finds_matching_schema(self):
        """Test finding an existing schema by name and version."""
        mock_client = Mock()

        # Mock SchemaLoader.list_all to return a matching schema
        with patch("atdata.atmosphere.SchemaLoader") as MockLoader:
            mock_loader = MockLoader.return_value
            mock_loader.list_all.return_value = [
                {
                    "uri": "at://did:plc:test/ac.foundation.dataset.sampleSchema/abc",
                    "value": {
                        "name": "test_promote.PromoteTestSample",
                        "version": "1.0.0",
                    },
                }
            ]

            result = _find_existing_schema(
                mock_client, "test_promote.PromoteTestSample", "1.0.0"
            )

            assert result == "at://did:plc:test/ac.foundation.dataset.sampleSchema/abc"

    def test_returns_none_when_not_found(self):
        """Test returns None when no matching schema exists."""
        mock_client = Mock()

        with patch("atdata.atmosphere.SchemaLoader") as MockLoader:
            mock_loader = MockLoader.return_value
            mock_loader.list_all.return_value = [
                {
                    "uri": "at://did:plc:test/ac.foundation.dataset.sampleSchema/abc",
                    "value": {
                        "name": "other.OtherSample",
                        "version": "1.0.0",
                    },
                }
            ]

            result = _find_existing_schema(
                mock_client, "test_promote.PromoteTestSample", "1.0.0"
            )

            assert result is None

    def test_returns_none_when_version_mismatch(self):
        """Test returns None when version doesn't match."""
        mock_client = Mock()

        with patch("atdata.atmosphere.SchemaLoader") as MockLoader:
            mock_loader = MockLoader.return_value
            mock_loader.list_all.return_value = [
                {
                    "uri": "at://did:plc:test/ac.foundation.dataset.sampleSchema/abc",
                    "value": {
                        "name": "test_promote.PromoteTestSample",
                        "version": "2.0.0",  # Different version
                    },
                }
            ]

            result = _find_existing_schema(
                mock_client, "test_promote.PromoteTestSample", "1.0.0"
            )

            assert result is None


class TestFindOrPublishSchema:
    """Tests for _find_or_publish_schema helper."""

    def test_returns_existing_schema(self):
        """Test returns existing schema URI without publishing."""
        mock_client = Mock()

        with patch("atdata.promote._find_existing_schema") as mock_find:
            mock_find.return_value = "at://existing/schema/uri"

            with patch("atdata.atmosphere.SchemaPublisher") as MockPublisher:
                result = _find_or_publish_schema(
                    PromoteTestSample,
                    "1.0.0",
                    mock_client,
                )

                assert result == "at://existing/schema/uri"
                MockPublisher.return_value.publish.assert_not_called()

    def test_publishes_new_schema_when_not_found(self):
        """Test publishes new schema when none exists."""
        mock_client = Mock()

        with patch("atdata.promote._find_existing_schema") as mock_find:
            mock_find.return_value = None  # No existing schema

            with patch("atdata.atmosphere.SchemaPublisher") as MockPublisher:
                mock_publisher = MockPublisher.return_value
                mock_publisher.publish.return_value = Mock(
                    __str__=lambda s: "at://new/schema/uri"
                )

                result = _find_or_publish_schema(
                    PromoteTestSample,
                    "1.0.0",
                    mock_client,
                )

                assert result == "at://new/schema/uri"
                mock_publisher.publish.assert_called_once()


class TestPromoteToAtmosphere:
    """Tests for promote_to_atmosphere function."""

    def test_raises_on_empty_data_urls(self):
        """Test raises ValueError when local entry has no data URLs."""
        entry = LocalDatasetEntry(
            name="test-dataset",
            schema_ref="local://schemas/test@1.0.0",
            data_urls=[],  # Empty!
        )
        mock_index = Mock()
        mock_client = Mock()

        with pytest.raises(ValueError, match="has no data URLs"):
            promote_to_atmosphere(entry, mock_index, mock_client)

    def test_promotes_with_existing_urls(self):
        """Test promotion using existing data URLs."""
        entry = LocalDatasetEntry(
            name="test-dataset",
            schema_ref="local://schemas/test@1.0.0",
            data_urls=["s3://bucket/data-000000.tar"],
            metadata={"key": "value"},
        )

        mock_index = Mock()
        mock_index.get_schema.return_value = {
            "name": "test_promote.PromoteTestSample",
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

        mock_client = Mock()

        with patch("atdata.promote._find_or_publish_schema") as mock_find_schema:
            mock_find_schema.return_value = "at://did:plc:test/schema/abc"

            with patch("atdata.atmosphere.DatasetPublisher") as MockPublisher:
                mock_publisher = MockPublisher.return_value
                mock_uri = Mock(__str__=lambda s: "at://did:plc:test/record/xyz")
                mock_publisher.publish_with_urls.return_value = mock_uri

                result = promote_to_atmosphere(entry, mock_index, mock_client)

                assert result == "at://did:plc:test/record/xyz"

                # Verify publish_with_urls was called with correct args
                mock_publisher.publish_with_urls.assert_called_once()
                call_kwargs = mock_publisher.publish_with_urls.call_args[1]
                assert call_kwargs["urls"] == ["s3://bucket/data-000000.tar"]
                assert call_kwargs["schema_uri"] == "at://did:plc:test/schema/abc"
                assert call_kwargs["name"] == "test-dataset"
                assert call_kwargs["metadata"] == {"key": "value"}

    def test_promotes_with_custom_name(self):
        """Test promotion with overridden name."""
        entry = LocalDatasetEntry(
            name="original-name",
            schema_ref="local://schemas/test@1.0.0",
            data_urls=["s3://bucket/data.tar"],
        )

        mock_index = Mock()
        mock_index.get_schema.return_value = {
            "name": "TestSample",
            "version": "1.0.0",
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

                promote_to_atmosphere(
                    entry,
                    mock_index,
                    mock_client,
                    name="custom-name",
                    tags=["tag1", "tag2"],
                    license="MIT",
                )

                call_kwargs = mock_publisher.publish_with_urls.call_args[1]
                assert call_kwargs["name"] == "custom-name"
                assert call_kwargs["tags"] == ["tag1", "tag2"]
                assert call_kwargs["license"] == "MIT"

    def test_promotes_with_data_store(self):
        """Test promotion with data store for copying data."""
        entry = LocalDatasetEntry(
            name="test-dataset",
            schema_ref="local://schemas/test@1.0.0",
            data_urls=["s3://old-bucket/data.tar"],
        )

        mock_index = Mock()
        mock_index.get_schema.return_value = {
            "name": "TestSample",
            "version": "1.0.0",
            "fields": [
                {
                    "name": "value",
                    "fieldType": {"$type": "local#primitive", "primitive": "int"},
                    "optional": False,
                },
            ],
        }

        mock_client = Mock()
        mock_data_store = Mock()
        mock_data_store.write_shards.return_value = [
            "s3://new-bucket/promoted/test-dataset/shard-000000.tar"
        ]

        with patch("atdata.promote._find_or_publish_schema") as mock_find_schema:
            mock_find_schema.return_value = "at://schema"

            with patch("atdata.atmosphere.DatasetPublisher") as MockPublisher:
                mock_publisher = MockPublisher.return_value
                mock_publisher.publish_with_urls.return_value = Mock(
                    __str__=lambda s: "at://result"
                )

                with patch("atdata.dataset.Dataset"):
                    promote_to_atmosphere(
                        entry,
                        mock_index,
                        mock_client,
                        data_store=mock_data_store,
                    )

                    # Verify data_store.write_shards was called
                    mock_data_store.write_shards.assert_called_once()

                    # Verify new URLs were used
                    call_kwargs = mock_publisher.publish_with_urls.call_args[1]
                    assert "s3://new-bucket/" in call_kwargs["urls"][0]
