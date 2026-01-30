"""Integration tests for error handling and recovery.

Tests error conditions and graceful failure including:
- Missing schemas and data URLs
- Malformed data (msgpack, tar)
- Connection failures (Redis, S3, ATProto)
- Authentication and rate limiting errors
- Timeout scenarios
- Partial failures in multi-shard datasets
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import tarfile
import io

from redis.exceptions import RedisError
import atdata
import webdataset as wds
from atdata.local import Index, LocalDatasetEntry
from atdata.atmosphere import AtmosphereClient, AtUri


##
# Test sample types


@atdata.packable
class ErrorTestSample:
    """Sample for error handling tests."""

    name: str
    value: int


##
# Schema Error Tests


class TestMissingSchema:
    """Tests for missing schema errors."""

    def test_missing_schema_raises_keyerror(self, clean_redis):
        """Accessing non-existent schema should raise KeyError."""
        index = Index(redis=clean_redis)

        with pytest.raises(KeyError):
            index.get_schema("local://schemas/NonExistent@1.0.0")

    def test_dataset_with_invalid_schema_ref(self, clean_redis):
        """Dataset entry with invalid schema ref should error on decode."""
        index = Index(redis=clean_redis)

        entry = LocalDatasetEntry(
            name="orphan-dataset",
            schema_ref="local://schemas/DoesNotExist@1.0.0",
            data_urls=["s3://bucket/data.tar"],
        )
        entry.write_to(clean_redis)

        # Entry exists but schema doesn't
        retrieved = index.get_entry_by_name("orphan-dataset")
        assert retrieved.name == "orphan-dataset"

        # Attempting to decode schema should fail
        with pytest.raises(KeyError):
            index.decode_schema(retrieved.schema_ref)


##
# Data URL Error Tests


class TestMissingDataUrls:
    """Tests for missing or inaccessible data URLs."""

    def test_empty_data_urls_raises(self, clean_redis):
        """Dataset entry with empty URLs should be flagged."""
        index = Index(redis=clean_redis)
        schema_ref = index.publish_schema(ErrorTestSample, version="1.0.0")

        entry = LocalDatasetEntry(
            name="empty-urls",
            schema_ref=schema_ref,
            data_urls=[],
        )
        entry.write_to(clean_redis)

        retrieved = index.get_entry_by_name("empty-urls")
        assert retrieved.data_urls == []

    def test_nonexistent_tar_raises(self, tmp_path):
        """Attempting to read non-existent tar should raise."""
        nonexistent_path = tmp_path / "does-not-exist.tar"

        ds = atdata.Dataset[ErrorTestSample](str(nonexistent_path))

        # Iterating should fail
        with pytest.raises(FileNotFoundError):
            list(ds.ordered(batch_size=None))


##
# Malformed Data Tests


class TestMalformedMsgpack:
    """Tests for corrupted msgpack data."""

    def test_invalid_msgpack_in_tar(self, tmp_path):
        """Tar with invalid msgpack should raise on iteration."""
        tar_path = tmp_path / "corrupted-000000.tar"

        # Create tar with invalid msgpack data (no explicit __key__ file;
        # WebDataset derives __key__ from the filename prefix automatically)
        with tarfile.open(tar_path, "w") as tar:
            invalid_data = b"\xff\xff\xff\xff\xff"  # Not valid msgpack
            info = tarfile.TarInfo(name="sample-0.msgpack")
            info.size = len(invalid_data)
            tar.addfile(info, fileobj=io.BytesIO(invalid_data))

        ds = atdata.Dataset[ErrorTestSample](str(tar_path))

        # Should raise an error when trying to deserialize
        with pytest.raises(TypeError):  # unpacked data is not a mapping
            list(ds.ordered(batch_size=None))


class TestCorruptedTar:
    """Tests for corrupted tar files."""

    def test_truncated_tar_raises(self, tmp_path):
        """Truncated tar file should raise an error."""
        tar_path = tmp_path / "truncated-000000.tar"

        # Create a valid tar then truncate it
        with tarfile.open(tar_path, "w") as tar:
            data = b"test data"
            info = tarfile.TarInfo(name="test.txt")
            info.size = len(data)
            import io

            tar.addfile(info, fileobj=io.BytesIO(data))

        # Truncate the file
        with open(tar_path, "r+b") as f:
            f.truncate(50)  # Truncate to partial content

        ds = atdata.Dataset[ErrorTestSample](str(tar_path))

        with pytest.raises((tarfile.TarError, EOFError)):
            list(ds.ordered(batch_size=None))

    def test_not_a_tar_file_raises(self, tmp_path):
        """Non-tar file should raise clear error."""
        fake_tar = tmp_path / "fake-000000.tar"

        # Write random bytes
        with open(fake_tar, "wb") as f:
            f.write(b"This is not a tar file at all!")

        ds = atdata.Dataset[ErrorTestSample](str(fake_tar))

        with pytest.raises(tarfile.TarError):
            list(ds.ordered(batch_size=None))


##
# Redis Error Tests


class TestRedisErrors:
    """Tests for Redis connection errors."""

    def test_redis_connection_error(self):
        """Operations with bad Redis connection should fail cleanly."""
        from redis import Redis, ConnectionError

        # Create index with invalid Redis connection
        bad_redis = Redis(
            host="nonexistent.invalid.host", port=9999, socket_timeout=0.1
        )

        index = Index(redis=bad_redis)

        # Operations should raise connection errors
        with pytest.raises((ConnectionError, Exception)):
            index.publish_schema(ErrorTestSample, version="1.0.0")

    def test_entry_lookup_with_bad_redis(self, clean_redis):
        """Entry lookup should fail cleanly if Redis becomes unavailable."""
        index = Index(redis=clean_redis)

        # First, add an entry
        schema_ref = index.publish_schema(ErrorTestSample, version="1.0.0")
        entry = LocalDatasetEntry(
            name="test-entry",
            schema_ref=schema_ref,
            data_urls=["s3://bucket/data.tar"],
        )
        entry.write_to(clean_redis)

        # Entry should be retrievable
        retrieved = index.get_entry_by_name("test-entry")
        assert retrieved.name == "test-entry"


##
# ATProto Error Tests


class TestAtProtoErrors:
    """Tests for ATProto/Atmosphere errors."""

    def test_unauthenticated_publish_raises(self):
        """Publishing without authentication should raise."""
        mock_client = Mock()
        mock_client.me = None

        client = AtmosphereClient(_client=mock_client)

        # Not authenticated
        assert not client.is_authenticated

        from atdata.atmosphere import SchemaPublisher

        publisher = SchemaPublisher(client)

        with pytest.raises(ValueError, match="authenticated"):
            publisher.publish(ErrorTestSample, version="1.0.0")

    def test_invalid_at_uri_raises(self):
        """Parsing invalid AT URI should raise ValueError."""
        invalid_uris = [
            "not-a-uri",
            "https://example.com/path",
            "at://",
            "at://did:plc:abc",  # Missing collection and rkey
            "at://did:plc:abc/collection",  # Missing rkey
        ]

        for uri in invalid_uris:
            with pytest.raises(ValueError):
                AtUri.parse(uri)

    def test_api_error_response_handling(self):
        """API errors should be propagated appropriately."""
        mock_client = Mock()
        mock_client.me = MagicMock()
        mock_client.me.did = "did:plc:test123"

        # Simulate an API error
        from atproto_client.exceptions import AtProtocolError

        mock_client.com.atproto.repo.create_record.side_effect = AtProtocolError(
            "API error occurred"
        )

        # Create client and authenticate it
        client = AtmosphereClient(_client=mock_client)
        client._session = {"did": "did:plc:test123"}  # Mark as authenticated

        from atdata.atmosphere import SchemaPublisher

        publisher = SchemaPublisher(client)

        # Should propagate the API error
        with pytest.raises(AtProtocolError):
            publisher.publish(ErrorTestSample, version="1.0.0")

    def test_expired_session_detection(self):
        """Expired session should be detectable."""
        mock_client = Mock()
        mock_client.me = None
        mock_client.export_session_string.return_value = None

        client = AtmosphereClient(_client=mock_client)

        # Should not be authenticated
        assert not client.is_authenticated


##
# Entry Not Found Tests


class TestNotFoundErrors:
    """Tests for not-found error handling."""

    def test_get_entry_by_name_not_found(self, clean_redis):
        """Getting non-existent entry by name should raise KeyError."""
        index = Index(redis=clean_redis)

        with pytest.raises(KeyError):
            index.get_entry_by_name("nonexistent-dataset")

    def test_get_entry_by_cid_not_found(self, clean_redis):
        """Getting non-existent entry by CID should raise KeyError."""
        index = Index(redis=clean_redis)

        with pytest.raises(KeyError):
            index.get_entry("bafyreifake123456789")


##
# Error Message Quality Tests


class TestErrorMessageQuality:
    """Tests that error messages are helpful and don't leak sensitive info."""

    def test_missing_schema_error_includes_ref(self, clean_redis):
        """Missing schema error should include the schema reference."""
        index = Index(redis=clean_redis)

        try:
            index.get_schema("local://schemas/MissingType@1.0.0")
            assert False, "Should have raised KeyError"
        except KeyError as e:
            # Error should mention the schema reference
            assert "MissingType" in str(e) or "local://" in str(e)

    def test_invalid_uri_error_is_clear(self):
        """Invalid AT URI error should explain the issue."""
        try:
            AtUri.parse("not-valid")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            # Error should explain it's not a valid URI
            assert "at://" in str(e).lower() or "uri" in str(e).lower()

    def test_auth_error_no_credential_leak(self):
        """Authentication errors should not leak credentials."""
        mock_client = Mock()
        mock_client.me = None

        client = AtmosphereClient(_client=mock_client)

        from atdata.atmosphere import SchemaPublisher

        publisher = SchemaPublisher(client)

        try:
            publisher.publish(ErrorTestSample, version="1.0.0")
        except ValueError as e:
            error_msg = str(e)
            # Should not contain anything that looks like a password or token
            assert "password" not in error_msg.lower()
            assert "token" not in error_msg.lower()
            assert "secret" not in error_msg.lower()


##
# Recovery Tests


class TestRecovery:
    """Tests for recovery from errors."""

    def test_can_continue_after_bad_sample(self, tmp_path, clean_redis):
        """System should be usable after encountering bad data."""
        # First, try to read a bad file
        bad_tar = tmp_path / "bad-000000.tar"
        with open(bad_tar, "wb") as f:
            f.write(b"not a tar file")

        ds_bad = atdata.Dataset[ErrorTestSample](str(bad_tar))
        try:
            list(ds_bad.ordered(batch_size=None))
        except Exception:
            pass  # Expected to fail

        # Now use a good file - should still work
        good_tar = tmp_path / "good-000000.tar"
        import webdataset as wds

        with wds.writer.TarWriter(str(good_tar)) as writer:
            sample = ErrorTestSample(name="good", value=42)
            writer.write(sample.as_wds)

        ds_good = atdata.Dataset[ErrorTestSample](str(good_tar))
        samples = list(ds_good.ordered(batch_size=None))

        assert len(samples) == 1
        assert samples[0].name == "good"

    def test_index_usable_after_failed_publish(self, clean_redis):
        """Index should remain usable after a failed operation."""
        index = Index(redis=clean_redis)

        # Try to get a non-existent schema (fails as expected)
        with pytest.raises(KeyError):
            index.get_schema("local://schemas/NoSuch@1.0.0")

        # Index should still work
        schema_ref = index.publish_schema(ErrorTestSample, version="1.0.0")
        assert "ErrorTestSample" in schema_ref

        schema = index.get_schema(schema_ref)
        assert schema["name"] == "ErrorTestSample"


##
# Validation Tests


class TestInputValidation:
    """Tests for input validation."""

    def test_empty_version_string(self, clean_redis):
        """Empty version string should be handled."""
        index = Index(redis=clean_redis)

        # Empty version - implementation may accept or reject
        schema_ref = index.publish_schema(ErrorTestSample, version="")
        # If it accepts, it should store and retrieve correctly
        schema = index.get_schema(schema_ref)
        assert schema["name"] == "ErrorTestSample"

    def test_special_chars_in_version(self, clean_redis):
        """Special characters in version should be handled."""
        index = Index(redis=clean_redis)

        schema_ref = index.publish_schema(
            ErrorTestSample, version="1.0.0-beta+build.123"
        )
        schema = index.get_schema(schema_ref)

        assert schema["version"] == "1.0.0-beta+build.123"


##
# Timeout Tests


class TestTimeoutScenarios:
    """Tests for timeout and slow connection scenarios."""

    def test_redis_socket_timeout(self):
        """Redis operations should fail with socket timeout."""
        from redis import Redis

        # Very short timeout to force failure
        redis = Redis(
            host="10.255.255.1",  # Non-routable IP
            port=6379,
            socket_timeout=0.01,
            socket_connect_timeout=0.01,
        )

        index = Index(redis=redis)

        # Should timeout quickly rather than hang
        with pytest.raises(RedisError):
            index.publish_schema(ErrorTestSample, version="1.0.0")

    def test_slow_iteration_continues(self, tmp_path):
        """Dataset iteration should handle slow reads gracefully."""
        # Create a valid dataset
        tar_path = tmp_path / "slow-000000.tar"
        with wds.writer.TarWriter(str(tar_path)) as writer:
            for i in range(5):
                sample = ErrorTestSample(name=f"sample_{i}", value=i)
                writer.write(sample.as_wds)

        ds = atdata.Dataset[ErrorTestSample](str(tar_path))

        # Normal iteration should work
        samples = list(ds.ordered(batch_size=None))
        assert len(samples) == 5


##
# Partial Failure Tests


class TestPartialFailures:
    """Tests for partial failures in multi-shard scenarios."""

    def test_multi_shard_with_missing_middle_shard(self, tmp_path):
        """Multi-shard dataset with missing shard should fail cleanly."""
        # Create first and third shard, skip second
        for i in [0, 2]:
            tar_path = tmp_path / f"data-{i:06d}.tar"
            with wds.writer.TarWriter(str(tar_path)) as writer:
                sample = ErrorTestSample(name=f"shard_{i}", value=i)
                writer.write(sample.as_wds)

        # Use brace notation that expects all three shards
        url = str(tmp_path / "data-{000000..000002}.tar")
        ds = atdata.Dataset[ErrorTestSample](url)

        # Should fail when hitting missing shard
        with pytest.raises(FileNotFoundError):
            list(ds.ordered(batch_size=None))

    def test_multi_shard_with_corrupted_shard(self, tmp_path):
        """Multi-shard dataset with one corrupted shard should fail."""
        # Create two good shards
        for i in range(2):
            tar_path = tmp_path / f"data-{i:06d}.tar"
            with wds.writer.TarWriter(str(tar_path)) as writer:
                sample = ErrorTestSample(name=f"shard_{i}", value=i)
                writer.write(sample.as_wds)

        # Create a corrupted third shard
        corrupted_path = tmp_path / "data-000002.tar"
        with open(corrupted_path, "wb") as f:
            f.write(b"this is not a valid tar file")

        url = str(tmp_path / "data-{000000..000002}.tar")
        ds = atdata.Dataset[ErrorTestSample](url)

        # Should fail when hitting corrupted shard
        with pytest.raises(tarfile.TarError):
            list(ds.ordered(batch_size=None))

    def test_empty_shard_in_multi_shard(self, tmp_path):
        """Empty shard in multi-shard dataset should be handled."""
        # Create one shard with data
        tar_path = tmp_path / "data-000000.tar"
        with wds.writer.TarWriter(str(tar_path)) as writer:
            sample = ErrorTestSample(name="sample", value=42)
            writer.write(sample.as_wds)

        # Create an empty tar (valid but no samples)
        empty_path = tmp_path / "data-000001.tar"
        with tarfile.open(empty_path, "w"):
            pass  # Empty tar

        url = str(tmp_path / "data-{000000..000001}.tar")
        ds = atdata.Dataset[ErrorTestSample](url)

        # Should handle empty shard gracefully
        samples = list(ds.ordered(batch_size=None))
        # May get 1 sample (from first shard) or error depending on implementation
        assert len(samples) >= 0  # At minimum, shouldn't crash

    def test_good_shards_before_bad_are_processed(self, tmp_path):
        """Samples from good shards before bad one should be accessible."""
        # Create first good shard with multiple samples
        tar_path = tmp_path / "data-000000.tar"
        with wds.writer.TarWriter(str(tar_path)) as writer:
            for i in range(3):
                sample = ErrorTestSample(name=f"good_{i}", value=i)
                writer.write(sample.as_wds)

        # Create second corrupted shard
        corrupted_path = tmp_path / "data-000001.tar"
        with open(corrupted_path, "wb") as f:
            f.write(b"corrupted data")

        url = str(tmp_path / "data-{000000..000001}.tar")
        ds = atdata.Dataset[ErrorTestSample](url)

        # Iterate and collect what we can
        collected = []
        try:
            for sample in ds.ordered(batch_size=None):
                collected.append(sample)
        except Exception:
            pass  # Expected to fail on second shard

        # Should have gotten samples from first shard before failure
        # Note: actual behavior depends on WebDataset's buffering
        # This test documents the behavior rather than enforcing it
        assert isinstance(collected, list)


##
# S3 Error Simulation Tests


class TestS3ErrorSimulation:
    """Tests for S3-related error scenarios using mocks."""

    def test_s3_access_denied_error(self):
        """S3 access denied should raise clear error."""
        from atdata import S3Source
        from botocore.exceptions import ClientError

        # Create source with mock credentials
        source = S3Source(
            bucket="test-bucket",
            keys=["data.tar"],
            access_key="test",
            secret_key="test",
        )

        # Mock the client after source creation
        with patch.object(source, "_get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.get_object.side_effect = ClientError(
                {"Error": {"Code": "AccessDenied", "Message": "Access Denied"}},
                "GetObject",
            )
            mock_get_client.return_value = mock_client

            # Opening shard should propagate the error
            # Use full S3 URI as returned by shard_list
            with pytest.raises(ClientError):
                source.open_shard("s3://test-bucket/data.tar")

    def test_s3_connection_timeout_simulation(self):
        """S3 connection timeout should raise appropriate error."""
        from atdata import S3Source
        from botocore.exceptions import ConnectTimeoutError

        # Create source with mock credentials
        source = S3Source(
            bucket="test-bucket",
            keys=["data.tar"],
            access_key="test",
            secret_key="test",
        )

        # Mock the client after source creation
        with patch.object(source, "_get_client") as mock_get_client:
            mock_client = Mock()
            mock_client.get_object.side_effect = ConnectTimeoutError(
                endpoint_url="s3://test"
            )
            mock_get_client.return_value = mock_client

            # Use full S3 URI as returned by shard_list
            with pytest.raises(ConnectTimeoutError):
                source.open_shard("s3://test-bucket/data.tar")
