"""Tests for data source implementations."""

import io
import tarfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest
import webdataset as wds

import atdata
from atdata._sources import URLSource, S3Source
from atdata._protocols import DataSource


# Test sample type
@atdata.packable
class SourceTestSample:
    """Simple sample for testing data sources."""
    name: str
    value: int


def create_test_tar(path: Path, samples: list[dict]) -> None:
    """Create a test tar file with msgpack samples."""
    with wds.writer.TarWriter(str(path)) as sink:
        for i, data in enumerate(samples):
            sample = SourceTestSample(**data)
            sink.write(sample.as_wds)


class TestURLSource:
    """Tests for URLSource."""

    def test_conforms_to_protocol(self):
        """URLSource should satisfy DataSource protocol."""
        source = URLSource("http://example.com/data.tar")
        assert isinstance(source, DataSource)

    def test_shard_list_single_url(self):
        """shard_list returns single URL unchanged."""
        source = URLSource("http://example.com/data.tar")
        assert source.shard_list == ["http://example.com/data.tar"]

    def test_shard_list_brace_expansion(self):
        """shard_list expands brace patterns."""
        source = URLSource("data-{000..002}.tar")
        assert source.shard_list == [
            "data-000.tar",
            "data-001.tar",
            "data-002.tar",
        ]

    def test_shard_list_complex_brace_pattern(self):
        """shard_list handles complex brace patterns."""
        source = URLSource("s3://bucket/{train,test}-{00..01}.tar")
        assert source.shard_list == [
            "s3://bucket/train-00.tar",
            "s3://bucket/train-01.tar",
            "s3://bucket/test-00.tar",
            "s3://bucket/test-01.tar",
        ]

    def test_shards_yields_streams(self, tmp_path):
        """shards() yields (url, stream) pairs."""
        # Create test tar file
        tar_path = tmp_path / "test.tar"
        create_test_tar(tar_path, [{"name": "test", "value": 42}])

        source = URLSource(str(tar_path))
        shards = list(source.shards())

        assert len(shards) == 1
        url, stream = shards[0]
        assert url == str(tar_path)
        assert hasattr(stream, "read")

    def test_open_shard(self, tmp_path):
        """open_shard opens a specific shard."""
        tar_path = tmp_path / "test.tar"
        create_test_tar(tar_path, [{"name": "test", "value": 42}])

        source = URLSource(str(tar_path))
        stream = source.open_shard(str(tar_path))

        assert hasattr(stream, "read")

    def test_open_shard_not_found(self, tmp_path):
        """open_shard raises KeyError for unknown shard."""
        tar_path = tmp_path / "test.tar"
        create_test_tar(tar_path, [{"name": "test", "value": 42}])

        source = URLSource(str(tar_path))

        with pytest.raises(KeyError, match="Shard not found"):
            source.open_shard("nonexistent.tar")

    def test_dataset_integration(self, tmp_path):
        """URLSource works with Dataset."""
        tar_path = tmp_path / "test.tar"
        create_test_tar(tar_path, [
            {"name": "sample1", "value": 1},
            {"name": "sample2", "value": 2},
        ])

        source = URLSource(str(tar_path))
        ds = atdata.Dataset[SourceTestSample](source)

        samples = list(ds.ordered())
        assert len(samples) == 2
        assert samples[0].name == "sample1"
        assert samples[1].value == 2


class TestS3Source:
    """Tests for S3Source."""

    def test_conforms_to_protocol(self):
        """S3Source should satisfy DataSource protocol."""
        source = S3Source(bucket="test", keys=["data.tar"])
        assert isinstance(source, DataSource)

    def test_shard_list(self):
        """shard_list returns S3 URIs."""
        source = S3Source(bucket="my-bucket", keys=["a.tar", "b.tar"])
        assert source.shard_list == [
            "s3://my-bucket/a.tar",
            "s3://my-bucket/b.tar",
        ]

    def test_from_urls(self):
        """from_urls parses S3 URLs correctly."""
        source = S3Source.from_urls([
            "s3://bucket/path/a.tar",
            "s3://bucket/path/b.tar",
        ])

        assert source.bucket == "bucket"
        assert source.keys == ["path/a.tar", "path/b.tar"]

    def test_from_urls_with_credentials(self):
        """from_urls passes credentials through."""
        source = S3Source.from_urls(
            ["s3://bucket/data.tar"],
            endpoint="https://r2.example.com",
            access_key="AKID",
            secret_key="SECRET",
        )

        assert source.endpoint == "https://r2.example.com"
        assert source.access_key == "AKID"
        assert source.secret_key == "SECRET"

    def test_from_urls_empty(self):
        """from_urls raises on empty list."""
        with pytest.raises(ValueError, match="cannot be empty"):
            S3Source.from_urls([])

    def test_from_urls_invalid_scheme(self):
        """from_urls raises on non-s3 URLs."""
        with pytest.raises(ValueError, match="Not an S3 URL"):
            S3Source.from_urls(["https://example.com/data.tar"])

    def test_from_urls_multiple_buckets(self):
        """from_urls raises when URLs span buckets."""
        with pytest.raises(ValueError, match="same bucket"):
            S3Source.from_urls([
                "s3://bucket-a/data.tar",
                "s3://bucket-b/data.tar",
            ])

    def test_from_credentials(self):
        """from_credentials creates source from dict."""
        creds = {
            "AWS_ACCESS_KEY_ID": "AKID",
            "AWS_SECRET_ACCESS_KEY": "SECRET",
            "AWS_ENDPOINT": "https://r2.example.com",
        }

        source = S3Source.from_credentials(creds, "bucket", ["data.tar"])

        assert source.bucket == "bucket"
        assert source.keys == ["data.tar"]
        assert source.endpoint == "https://r2.example.com"
        assert source.access_key == "AKID"
        assert source.secret_key == "SECRET"

    def test_shards_uses_boto3(self):
        """shards() uses boto3 client to fetch objects."""
        mock_body = MagicMock()
        mock_body.read.return_value = b"tar data"

        with patch("boto3.client") as mock_boto:
            mock_client = Mock()
            mock_client.get_object.return_value = {"Body": mock_body}
            mock_boto.return_value = mock_client

            source = S3Source(
                bucket="test-bucket",
                keys=["data.tar"],
                access_key="AKID",
                secret_key="SECRET",
            )

            shards = list(source.shards())

            assert len(shards) == 1
            uri, stream = shards[0]
            assert uri == "s3://test-bucket/data.tar"
            assert stream == mock_body

            mock_client.get_object.assert_called_once_with(
                Bucket="test-bucket",
                Key="data.tar",
            )

    def test_open_shard_uses_boto3(self):
        """open_shard() uses boto3 client to fetch specific object."""
        mock_body = MagicMock()

        with patch("boto3.client") as mock_boto:
            mock_client = Mock()
            mock_client.get_object.return_value = {"Body": mock_body}
            mock_boto.return_value = mock_client

            source = S3Source(
                bucket="test-bucket",
                keys=["a.tar", "b.tar"],
                access_key="AKID",
                secret_key="SECRET",
            )

            stream = source.open_shard("s3://test-bucket/b.tar")

            assert stream == mock_body
            mock_client.get_object.assert_called_once_with(
                Bucket="test-bucket",
                Key="b.tar",
            )

    def test_open_shard_not_found(self):
        """open_shard raises KeyError for unknown shard."""
        source = S3Source(bucket="bucket", keys=["a.tar"])

        with pytest.raises(KeyError, match="Shard not found"):
            source.open_shard("s3://bucket/unknown.tar")

    def test_client_uses_endpoint(self):
        """Client is created with custom endpoint."""
        with patch("boto3.client") as mock_boto:
            mock_boto.return_value = Mock()

            source = S3Source(
                bucket="bucket",
                keys=["data.tar"],
                endpoint="https://custom.endpoint.com",
                access_key="AKID",
                secret_key="SECRET",
            )

            # Trigger client creation
            source._get_client()

            mock_boto.assert_called_once_with(
                "s3",
                endpoint_url="https://custom.endpoint.com",
                aws_access_key_id="AKID",
                aws_secret_access_key="SECRET",
            )

    def test_client_caching(self):
        """Client is cached after first creation."""
        with patch("boto3.client") as mock_boto:
            mock_client = Mock()
            mock_boto.return_value = mock_client

            source = S3Source(
                bucket="bucket",
                keys=["data.tar"],
                access_key="AKID",
                secret_key="SECRET",
            )

            # Call twice
            client1 = source._get_client()
            client2 = source._get_client()

            assert client1 is client2
            assert mock_boto.call_count == 1


class TestDatasetWithDataSource:
    """Integration tests for Dataset with different DataSource types."""

    def test_dataset_accepts_url_source(self, tmp_path):
        """Dataset can be created with URLSource."""
        tar_path = tmp_path / "test.tar"
        create_test_tar(tar_path, [{"name": "test", "value": 42}])

        source = URLSource(str(tar_path))
        ds = atdata.Dataset[SourceTestSample](source)

        assert ds.source is source
        assert ds.shard_list == [str(tar_path)]

    def test_dataset_accepts_string_url(self, tmp_path):
        """Dataset auto-wraps string URLs in URLSource."""
        tar_path = tmp_path / "test.tar"
        create_test_tar(tar_path, [{"name": "test", "value": 42}])

        ds = atdata.Dataset[SourceTestSample](str(tar_path))

        assert isinstance(ds.source, URLSource)
        assert ds.url == str(tar_path)

    def test_dataset_backward_compat_url_kwarg(self, tmp_path):
        """Dataset accepts url= keyword for backward compatibility."""
        tar_path = tmp_path / "test.tar"
        create_test_tar(tar_path, [{"name": "test", "value": 42}])

        ds = atdata.Dataset[SourceTestSample](url=str(tar_path))

        assert isinstance(ds.source, URLSource)
        assert ds.url == str(tar_path)

    def test_dataset_source_property(self, tmp_path):
        """Dataset.source property returns the underlying DataSource."""
        tar_path = tmp_path / "test.tar"
        create_test_tar(tar_path, [{"name": "test", "value": 42}])

        source = URLSource(str(tar_path))
        ds = atdata.Dataset[SourceTestSample](source)

        assert ds.source is source

    def test_dataset_multiple_shards(self, tmp_path):
        """Dataset works with multi-shard sources."""
        # Create two shards
        for i in range(2):
            tar_path = tmp_path / f"data-{i:06d}.tar"
            create_test_tar(tar_path, [{"name": f"shard{i}", "value": i}])

        pattern = str(tmp_path / "data-{000000..000001}.tar")
        ds = atdata.Dataset[SourceTestSample](pattern)

        samples = list(ds.ordered())
        assert len(samples) == 2
        names = {s.name for s in samples}
        assert names == {"shard0", "shard1"}
