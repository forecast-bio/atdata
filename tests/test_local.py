"""Test local repository storage functionality."""

##
# Imports

import pytest

# System
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

# External
import numpy as np
from redis import Redis
from moto import mock_aws

# Local
import atdata
import atdata.local as atlocal
import webdataset as wds

# Typing
from numpy.typing import NDArray
from typing import Any


##
# Test fixtures

@pytest.fixture
def redis_connection():
    """Provide a Redis connection, skip test if Redis is not available."""
    try:
        redis = Redis()
        redis.ping()
        yield redis
    except Exception:
        pytest.skip("Redis server not available")


@pytest.fixture
def clean_redis(redis_connection):
    """Provide a Redis connection with automatic BasicIndexEntry cleanup.

    Clears all BasicIndexEntry keys before and after each test to ensure
    test isolation.
    """
    def _clear_entries():
        for key in redis_connection.scan_iter(match='BasicIndexEntry:*'):
            redis_connection.delete(key)

    _clear_entries()
    yield redis_connection
    _clear_entries()


@pytest.fixture
def mock_s3():
    """Provide a mock S3 environment using moto.

    Note: Tests using this fixture may generate warnings due to s3fs/moto async
    incompatibility. These are suppressed via @pytest.mark.filterwarnings on
    individual tests that use this fixture.
    """
    with mock_aws():
        # Create S3 credentials dict (no endpoint_url for moto)
        creds = {
            'AWS_ACCESS_KEY_ID': 'testing',
            'AWS_SECRET_ACCESS_KEY': 'testing'
        }

        # Create S3 client and bucket
        import boto3
        s3_client = boto3.client(
            's3',
            aws_access_key_id=creds['AWS_ACCESS_KEY_ID'],
            aws_secret_access_key=creds['AWS_SECRET_ACCESS_KEY'],
            region_name='us-east-1'
        )

        bucket_name = 'test-bucket'
        s3_client.create_bucket(Bucket=bucket_name)

        yield {
            'credentials': creds,
            'bucket': bucket_name,
            'hive_path': f'{bucket_name}/datasets',
            's3_client': s3_client
        }


@pytest.fixture
def sample_dataset(tmp_path):
    """Create a sample WebDataset for testing."""
    # Create a temporary WebDataset
    dataset_path = tmp_path / "test-dataset-000000.tar"

    with wds.writer.TarWriter(str(dataset_path)) as sink:
        for i in range(10):
            sample = SimpleTestSample(name=f"sample_{i}", value=i * 10)
            sink.write(sample.as_wds)

    ds = atdata.Dataset[SimpleTestSample](url=str(dataset_path))
    return ds


@dataclass
class SimpleTestSample(atdata.PackableSample):
    """Simple test sample for repository tests."""
    name: str
    value: int


@dataclass
class ArrayTestSample(atdata.PackableSample):
    """Test sample with numpy array for repository tests."""
    label: str
    data: NDArray


def make_simple_dataset(tmp_path: Path, num_samples: int = 10, name: str = "test") -> atdata.Dataset:
    """Create a SimpleTestSample dataset for testing."""
    dataset_path = tmp_path / f"{name}-dataset-000000.tar"
    with wds.writer.TarWriter(str(dataset_path)) as sink:
        for i in range(num_samples):
            sample = SimpleTestSample(name=f"sample_{i}", value=i * 10)
            sink.write(sample.as_wds)
    return atdata.Dataset[SimpleTestSample](url=str(dataset_path))


def make_array_dataset(tmp_path: Path, num_samples: int = 3, array_shape: tuple = (10, 10)) -> atdata.Dataset:
    """Create an ArrayTestSample dataset for testing."""
    dataset_path = tmp_path / "array-dataset-000000.tar"
    with wds.writer.TarWriter(str(dataset_path)) as sink:
        for i in range(num_samples):
            arr = np.random.randn(*array_shape)
            sample = ArrayTestSample(label=f"array_{i}", data=arr)
            sink.write(sample.as_wds)
    return atdata.Dataset[ArrayTestSample](url=str(dataset_path))


##
# Helper function tests

def test_kind_str_for_sample_type():
    """Test that sample types are converted to correct fully-qualified string identifiers.

    Should produce strings in format 'module.name' that uniquely identify the sample type.
    """
    result = atlocal._kind_str_for_sample_type(SimpleTestSample)
    assert result == f"{SimpleTestSample.__module__}.SimpleTestSample"

    result2 = atlocal._kind_str_for_sample_type(ArrayTestSample)
    assert result2 == f"{ArrayTestSample.__module__}.ArrayTestSample"


def test_decode_bytes_dict():
    """Test that byte dictionaries from Redis are correctly decoded to strings.

    Should handle UTF-8 decoding of both keys and values from Redis response format.
    """
    bytes_dict = {
        b'wds_url': b's3://bucket/dataset.tar',
        b'sample_kind': b'module.Sample',
        b'metadata_url': b's3://bucket/metadata.msgpack',
        b'uuid': b'12345678-1234-1234-1234-123456789abc'
    }

    result = atlocal._decode_bytes_dict(bytes_dict)

    assert result == {
        'wds_url': 's3://bucket/dataset.tar',
        'sample_kind': 'module.Sample',
        'metadata_url': 's3://bucket/metadata.msgpack',
        'uuid': '12345678-1234-1234-1234-123456789abc'
    }
    assert all(isinstance(k, str) for k in result.keys())
    assert all(isinstance(v, str) for v in result.values())


def test_s3_env_valid_credentials(tmp_path):
    """Test loading S3 credentials from a valid .env file.

    Should successfully parse AWS_ENDPOINT, AWS_ACCESS_KEY_ID, and AWS_SECRET_ACCESS_KEY
    from a properly formatted .env file.
    """
    env_file = tmp_path / ".env"
    env_file.write_text(
        "AWS_ENDPOINT=http://localhost:9000\n"
        "AWS_ACCESS_KEY_ID=minioadmin\n"
        "AWS_SECRET_ACCESS_KEY=minioadmin\n"
    )

    result = atlocal._s3_env(env_file)

    assert result == {
        'AWS_ENDPOINT': 'http://localhost:9000',
        'AWS_ACCESS_KEY_ID': 'minioadmin',
        'AWS_SECRET_ACCESS_KEY': 'minioadmin'
    }


@pytest.mark.parametrize("missing_field,env_content", [
    ("AWS_ENDPOINT", "AWS_ACCESS_KEY_ID=minioadmin\nAWS_SECRET_ACCESS_KEY=minioadmin\n"),
    ("AWS_ACCESS_KEY_ID", "AWS_ENDPOINT=http://localhost:9000\nAWS_SECRET_ACCESS_KEY=minioadmin\n"),
    ("AWS_SECRET_ACCESS_KEY", "AWS_ENDPOINT=http://localhost:9000\nAWS_ACCESS_KEY_ID=minioadmin\n"),
])
def test_s3_env_missing_required_field(tmp_path, missing_field, env_content):
    """Test that loading S3 credentials fails when a required field is missing.

    Should raise AssertionError when .env file lacks any of the required fields:
    AWS_ENDPOINT, AWS_ACCESS_KEY_ID, or AWS_SECRET_ACCESS_KEY.
    """
    env_file = tmp_path / ".env"
    env_file.write_text(env_content)

    with pytest.raises(AssertionError):
        atlocal._s3_env(env_file)


def test_s3_from_credentials_with_dict():
    """Test creating S3FileSystem from a credentials dictionary.

    Should create a properly configured S3FileSystem instance using dict credentials.
    """
    creds = {
        'AWS_ENDPOINT': 'http://localhost:9000',
        'AWS_ACCESS_KEY_ID': 'minioadmin',
        'AWS_SECRET_ACCESS_KEY': 'minioadmin'
    }

    fs = atlocal._s3_from_credentials(creds)

    assert isinstance(fs, atlocal.S3FileSystem)
    assert fs.endpoint_url == 'http://localhost:9000'
    assert fs.key == 'minioadmin'
    assert fs.secret == 'minioadmin'


def test_s3_from_credentials_with_path(tmp_path):
    """Test creating S3FileSystem from a .env file path.

    Should load credentials from file and create S3FileSystem instance.
    """
    env_file = tmp_path / ".env"
    env_file.write_text(
        "AWS_ENDPOINT=http://localhost:9000\n"
        "AWS_ACCESS_KEY_ID=minioadmin\n"
        "AWS_SECRET_ACCESS_KEY=minioadmin\n"
    )

    fs = atlocal._s3_from_credentials(env_file)

    assert isinstance(fs, atlocal.S3FileSystem)
    assert fs.endpoint_url == 'http://localhost:9000'
    assert fs.key == 'minioadmin'
    assert fs.secret == 'minioadmin'


##
# BasicIndexEntry tests

def test_basic_index_entry_creation():
    """Test creating a BasicIndexEntry with explicit values.

    Should create an entry with provided wds_url, sample_kind, metadata_url, and uuid.
    """
    entry = atlocal.BasicIndexEntry(
        wds_url="s3://bucket/dataset.tar",
        sample_kind="test_module.TestSample",
        metadata_url="s3://bucket/metadata.msgpack",
        uuid="12345678-1234-1234-1234-123456789abc"
    )

    assert entry.wds_url == "s3://bucket/dataset.tar"
    assert entry.sample_kind == "test_module.TestSample"
    assert entry.metadata_url == "s3://bucket/metadata.msgpack"
    assert entry.uuid == "12345678-1234-1234-1234-123456789abc"


def test_basic_index_entry_default_uuid():
    """Test that BasicIndexEntry generates a valid UUID by default.

    Should auto-generate a unique UUID when none is provided, and it should be
    parsable as a valid UUID.
    """
    entry = atlocal.BasicIndexEntry(
        wds_url="s3://bucket/dataset.tar",
        sample_kind="test_module.TestSample",
        metadata_url="s3://bucket/metadata.msgpack"
    )

    assert entry.uuid is not None
    # Verify it's a valid UUID by parsing it
    parsed_uuid = UUID(entry.uuid)
    assert str(parsed_uuid) == entry.uuid


def test_basic_index_entry_write_to_redis(clean_redis):
    """Test persisting a BasicIndexEntry to Redis.

    Should write the entry to Redis as a hash with key 'BasicIndexEntry:{uuid}'
    and all fields should be retrievable with correct values.
    """
    test_uuid = "12345678-1234-1234-1234-123456789abc"

    entry = atlocal.BasicIndexEntry(
        wds_url="s3://bucket/dataset.tar",
        sample_kind="test_module.TestSample",
        metadata_url="s3://bucket/metadata.msgpack",
        uuid=test_uuid
    )

    entry.write_to(clean_redis)

    # Retrieve and verify actual stored values
    stored_data = atlocal._decode_bytes_dict(clean_redis.hgetall(f"BasicIndexEntry:{test_uuid}"))
    assert stored_data['wds_url'] == "s3://bucket/dataset.tar"
    assert stored_data['sample_kind'] == "test_module.TestSample"
    assert stored_data['metadata_url'] == "s3://bucket/metadata.msgpack"
    assert stored_data['uuid'] == test_uuid


def test_basic_index_entry_round_trip_redis(clean_redis):
    """Test writing and reading a BasicIndexEntry from Redis.

    Should be able to write an entry to Redis and read it back with all fields
    intact and matching the original values.
    """
    test_uuid = "12345678-1234-1234-1234-123456789abc"

    original_entry = atlocal.BasicIndexEntry(
        wds_url="s3://bucket/dataset.tar",
        sample_kind="test_module.TestSample",
        metadata_url="s3://bucket/metadata.msgpack",
        uuid=test_uuid
    )

    original_entry.write_to(clean_redis)

    # Read back from Redis
    stored_data = atlocal._decode_bytes_dict(clean_redis.hgetall(f"BasicIndexEntry:{test_uuid}"))
    retrieved_entry = atlocal.BasicIndexEntry(**stored_data)

    assert retrieved_entry.wds_url == original_entry.wds_url
    assert retrieved_entry.sample_kind == original_entry.sample_kind
    assert retrieved_entry.metadata_url == original_entry.metadata_url
    assert retrieved_entry.uuid == original_entry.uuid


##
# Index tests

def test_index_init_default_redis():
    """Test creating an Index with default Redis connection.

    Should create a new Redis connection using default parameters when no
    redis argument is provided.
    """
    index = atlocal.Index()

    assert index._redis is not None
    assert isinstance(index._redis, Redis)


def test_index_init_with_redis_connection():
    """Test creating an Index with an existing Redis connection.

    Should use the provided Redis connection instead of creating a new one.
    """
    redis = Redis()
    index = atlocal.Index(redis=redis)

    assert index._redis is redis


def test_index_init_with_redis_kwargs():
    """Test creating an Index with Redis connection kwargs.

    Should pass custom kwargs to Redis constructor when creating a new connection.
    """
    index = atlocal.Index(host='localhost', port=6379, db=0)

    assert index._redis is not None
    assert isinstance(index._redis, Redis)


def test_index_add_entry_without_uuid(clean_redis):
    """Test adding a dataset entry to the index without specifying UUID.

    Should create a BasicIndexEntry with auto-generated UUID and persist it to Redis.
    """
    index = atlocal.Index(redis=clean_redis)

    ds = atdata.Dataset[SimpleTestSample](
        url="s3://bucket/dataset.tar",
        metadata_url="s3://bucket/metadata.msgpack"
    )

    entry = index.add_entry(ds)

    assert entry.uuid is not None
    assert entry.wds_url == ds.url
    assert entry.sample_kind == f"{SimpleTestSample.__module__}.SimpleTestSample"
    assert entry.metadata_url == ds.metadata_url

    # Verify it was persisted to Redis
    stored_data = clean_redis.hgetall(f"BasicIndexEntry:{entry.uuid}")
    assert len(stored_data) > 0


def test_index_add_entry_with_uuid(clean_redis):
    """Test adding a dataset entry to the index with a specified UUID.

    Should create a BasicIndexEntry with the provided UUID and persist it to Redis.
    """
    index = atlocal.Index(redis=clean_redis)
    test_uuid = "12345678-1234-1234-1234-123456789abc"

    ds = atdata.Dataset[SimpleTestSample](
        url="s3://bucket/dataset.tar",
        metadata_url="s3://bucket/metadata.msgpack"
    )

    entry = index.add_entry(ds, uuid=test_uuid)

    assert entry.uuid == test_uuid
    assert entry.wds_url == ds.url
    assert entry.sample_kind == f"{SimpleTestSample.__module__}.SimpleTestSample"
    assert entry.metadata_url == ds.metadata_url


def test_index_entries_generator_empty(clean_redis):
    """Test iterating over entries in an empty index.

    Should yield no entries when the index is empty.
    """
    index = atlocal.Index(redis=clean_redis)

    entries = list(index.entries)
    assert len(entries) == 0


def test_index_entries_generator_multiple(clean_redis):
    """Test iterating over multiple entries in the index.

    Should yield all BasicIndexEntry objects that have been added to the index.
    """
    index = atlocal.Index(redis=clean_redis)

    ds1 = atdata.Dataset[SimpleTestSample](url="s3://bucket/dataset1.tar")
    ds2 = atdata.Dataset[ArrayTestSample](url="s3://bucket/dataset2.tar")

    entry1 = index.add_entry(ds1)
    entry2 = index.add_entry(ds2)

    entries = list(index.entries)
    assert len(entries) == 2

    uuids = {entry.uuid for entry in entries}
    assert entry1.uuid in uuids
    assert entry2.uuid in uuids


def test_index_all_entries_empty(clean_redis):
    """Test getting all entries as a list from an empty index.

    Should return an empty list when no entries exist.
    """
    index = atlocal.Index(redis=clean_redis)

    entries = index.all_entries
    assert isinstance(entries, list)
    assert len(entries) == 0


def test_index_all_entries_multiple(clean_redis):
    """Test getting all entries as a list with multiple entries.

    Should return a list containing all BasicIndexEntry objects in the index.
    """
    index = atlocal.Index(redis=clean_redis)

    ds1 = atdata.Dataset[SimpleTestSample](url="s3://bucket/dataset1.tar")
    ds2 = atdata.Dataset[ArrayTestSample](url="s3://bucket/dataset2.tar")

    entry1 = index.add_entry(ds1)
    entry2 = index.add_entry(ds2)

    entries = index.all_entries
    assert isinstance(entries, list)
    assert len(entries) == 2


def test_index_entries_filtering(clean_redis):
    """Test that index only returns BasicIndexEntry objects.

    Should only iterate over keys matching 'BasicIndexEntry:*' pattern and
    ignore any other Redis keys.
    """
    index = atlocal.Index(redis=clean_redis)

    # Add a BasicIndexEntry
    ds = atdata.Dataset[SimpleTestSample](url="s3://bucket/dataset.tar")
    entry = index.add_entry(ds)

    # Add some other Redis keys that should be ignored
    clean_redis.set("other_key", "value")
    clean_redis.hset("other_hash", "field", "value")

    entries = list(index.entries)
    assert len(entries) == 1
    assert entries[0].uuid == entry.uuid

    # Clean up non-BasicIndexEntry keys (fixture only cleans BasicIndexEntry:*)
    clean_redis.delete("other_key")
    clean_redis.delete("other_hash")


##
# Repo tests - Initialization

def test_repo_init_no_s3():
    """Test creating a Repo without S3 credentials.

    Should create a Repo with s3_credentials=None, bucket_fs=None, and working index.
    """
    repo = atlocal.Repo()

    assert repo.s3_credentials is None
    assert repo.bucket_fs is None
    assert repo.hive_path is None
    assert repo.hive_bucket is None
    assert repo.index is not None
    assert isinstance(repo.index, atlocal.Index)


def test_repo_init_with_s3_dict():
    """Test creating a Repo with S3 credentials as a dictionary.

    Should create a Repo with S3FileSystem and set hive_path and hive_bucket.
    """
    creds = {
        'AWS_ENDPOINT': 'http://localhost:9000',
        'AWS_ACCESS_KEY_ID': 'minioadmin',
        'AWS_SECRET_ACCESS_KEY': 'minioadmin'
    }

    repo = atlocal.Repo(s3_credentials=creds, hive_path="test-bucket/datasets")

    assert repo.s3_credentials == creds
    assert repo.bucket_fs is not None
    assert isinstance(repo.bucket_fs, atlocal.S3FileSystem)
    assert repo.hive_path == Path("test-bucket/datasets")
    assert repo.hive_bucket == "test-bucket"


def test_repo_init_with_s3_path(tmp_path):
    """Test creating a Repo with S3 credentials from a .env file.

    Should load credentials from file and create S3FileSystem with hive configuration.
    """
    env_file = tmp_path / ".env"
    env_file.write_text(
        "AWS_ENDPOINT=http://localhost:9000\n"
        "AWS_ACCESS_KEY_ID=minioadmin\n"
        "AWS_SECRET_ACCESS_KEY=minioadmin\n"
    )

    repo = atlocal.Repo(s3_credentials=env_file, hive_path="test-bucket/datasets")

    assert repo.s3_credentials is not None
    assert repo.bucket_fs is not None
    assert isinstance(repo.bucket_fs, atlocal.S3FileSystem)
    assert repo.hive_path == Path("test-bucket/datasets")
    assert repo.hive_bucket == "test-bucket"


def test_repo_init_s3_without_hive_path():
    """Test that creating a Repo with S3 but no hive_path raises ValueError.

    Should raise ValueError when s3_credentials is provided but hive_path is None.
    """
    creds = {
        'AWS_ENDPOINT': 'http://localhost:9000',
        'AWS_ACCESS_KEY_ID': 'minioadmin',
        'AWS_SECRET_ACCESS_KEY': 'minioadmin'
    }

    with pytest.raises(ValueError, match="Must specify hive path"):
        atlocal.Repo(s3_credentials=creds)


def test_repo_init_hive_path_parsing():
    """Test that hive_path is correctly parsed to extract bucket name.

    Should set hive_bucket to the first component of hive_path.
    """
    creds = {
        'AWS_ENDPOINT': 'http://localhost:9000',
        'AWS_ACCESS_KEY_ID': 'minioadmin',
        'AWS_SECRET_ACCESS_KEY': 'minioadmin'
    }

    repo = atlocal.Repo(s3_credentials=creds, hive_path="my-bucket/path/to/datasets")

    assert repo.hive_bucket == "my-bucket"
    assert repo.hive_path == Path("my-bucket/path/to/datasets")


def test_repo_init_with_custom_redis():
    """Test creating a Repo with a custom Redis connection.

    Should pass the Redis connection to the Index instance.
    """
    custom_redis = Redis()
    repo = atlocal.Repo(redis=custom_redis)

    assert repo.index._redis is custom_redis


##
# Repo tests - Insert functionality

def test_repo_insert_without_s3():
    """Test that inserting a dataset without S3 configured raises AssertionError.

    Should fail with assertion error when trying to insert without S3 credentials.
    """
    repo = atlocal.Repo()
    ds = atdata.Dataset[SimpleTestSample](url="s3://bucket/dataset.tar")

    with pytest.raises(AssertionError):
        repo.insert(ds)


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
@pytest.mark.filterwarnings("ignore:coroutine.*was never awaited:RuntimeWarning")
def test_repo_insert_single_shard(mock_s3, clean_redis, sample_dataset):
    """Test inserting a small dataset that fits in a single shard.

    Should write the dataset to S3, create metadata, add index entry, and return
    a new Dataset pointing to the stored copy with correct URL format.
    """
    repo = atlocal.Repo(
        s3_credentials=mock_s3['credentials'],
        hive_path=mock_s3['hive_path'],
        redis=clean_redis
    )

    entry, new_ds = repo.insert(sample_dataset, maxcount=100)

    assert entry.uuid is not None
    assert entry.wds_url is not None
    assert entry.sample_kind == f"{SimpleTestSample.__module__}.SimpleTestSample"
    assert len(repo.index.all_entries) == 1
    assert '.tar' in new_ds.url
    assert new_ds.url.startswith(mock_s3['hive_path'])


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
@pytest.mark.filterwarnings("ignore:coroutine.*was never awaited:RuntimeWarning")
def test_repo_insert_multiple_shards(mock_s3, clean_redis, tmp_path):
    """Test inserting a large dataset that spans multiple shards.

    Should write multiple tar files to S3, use brace notation in returned URL,
    and correctly format the shard range.
    """
    ds = make_simple_dataset(tmp_path, num_samples=50, name="large")
    repo = atlocal.Repo(
        s3_credentials=mock_s3['credentials'],
        hive_path=mock_s3['hive_path'],
        redis=clean_redis
    )

    entry, new_ds = repo.insert(ds, maxcount=10)

    assert entry.uuid is not None
    assert entry.wds_url is not None
    assert '{' in new_ds.url and '}' in new_ds.url


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
@pytest.mark.filterwarnings("ignore:coroutine.*was never awaited:RuntimeWarning")
def test_repo_insert_with_metadata(mock_s3, clean_redis, tmp_path):
    """Test inserting a dataset with metadata.

    Should write metadata as msgpack to S3 and include metadata_url in the
    returned Dataset and BasicIndexEntry.
    """
    ds = make_simple_dataset(tmp_path, num_samples=5)
    ds._metadata = {"description": "test dataset", "version": "1.0"}

    repo = atlocal.Repo(
        s3_credentials=mock_s3['credentials'],
        hive_path=mock_s3['hive_path'],
        redis=clean_redis
    )

    entry, new_ds = repo.insert(ds, maxcount=100)

    assert entry.metadata_url is not None
    assert new_ds.metadata_url is not None
    assert 'metadata' in entry.metadata_url


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
@pytest.mark.filterwarnings("ignore:coroutine.*was never awaited:RuntimeWarning")
def test_repo_insert_without_metadata(mock_s3, clean_redis, tmp_path):
    """Test inserting a dataset without metadata.

    Should handle None metadata gracefully and not write a metadata file.
    """
    ds = make_simple_dataset(tmp_path, num_samples=5)
    repo = atlocal.Repo(
        s3_credentials=mock_s3['credentials'],
        hive_path=mock_s3['hive_path'],
        redis=clean_redis
    )

    entry, new_ds = repo.insert(ds, maxcount=100)

    assert entry.uuid is not None
    assert len(repo.index.all_entries) == 1


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
@pytest.mark.filterwarnings("ignore:coroutine.*was never awaited:RuntimeWarning")
def test_repo_insert_cache_local_false(mock_s3, clean_redis, sample_dataset):
    """Test inserting with cache_local=False (direct S3 write).

    Should write tar shards directly to S3 without local caching.
    """
    repo = atlocal.Repo(
        s3_credentials=mock_s3['credentials'],
        hive_path=mock_s3['hive_path'],
        redis=clean_redis
    )

    entry, new_ds = repo.insert(sample_dataset, cache_local=False, maxcount=100)

    assert entry.uuid is not None
    assert entry.wds_url is not None


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
@pytest.mark.filterwarnings("ignore:coroutine.*was never awaited:RuntimeWarning")
def test_repo_insert_cache_local_true(mock_s3, clean_redis, sample_dataset):
    """Test inserting with cache_local=True (local cache then copy).

    Should write to temporary local storage first, then copy to S3, and clean up
    local cache files after copying.
    """
    repo = atlocal.Repo(
        s3_credentials=mock_s3['credentials'],
        hive_path=mock_s3['hive_path'],
        redis=clean_redis
    )

    entry, new_ds = repo.insert(sample_dataset, cache_local=True, maxcount=100)

    assert entry.uuid is not None
    assert entry.wds_url is not None


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
@pytest.mark.filterwarnings("ignore:coroutine.*was never awaited:RuntimeWarning")
def test_repo_insert_creates_index_entry(mock_s3, clean_redis, sample_dataset):
    """Test that insert() creates a valid index entry.

    Should add a BasicIndexEntry to the index with correct wds_url, sample_kind,
    metadata_url, and UUID.
    """
    repo = atlocal.Repo(
        s3_credentials=mock_s3['credentials'],
        hive_path=mock_s3['hive_path'],
        redis=clean_redis
    )

    entry, new_ds = repo.insert(sample_dataset, maxcount=100)

    assert entry.uuid is not None
    assert entry.wds_url == new_ds.url
    assert entry.sample_kind == f"{SimpleTestSample.__module__}.SimpleTestSample"

    all_entries = repo.index.all_entries
    assert len(all_entries) == 1
    assert all_entries[0].uuid == entry.uuid


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
@pytest.mark.filterwarnings("ignore:coroutine.*was never awaited:RuntimeWarning")
def test_repo_insert_uuid_generation(mock_s3, clean_redis, sample_dataset):
    """Test that insert() generates a unique UUID for each dataset.

    Should create a new UUID for the dataset and use it consistently in filenames,
    index entry, and returned Dataset.
    """
    repo = atlocal.Repo(
        s3_credentials=mock_s3['credentials'],
        hive_path=mock_s3['hive_path'],
        redis=clean_redis
    )

    entry1, new_ds1 = repo.insert(sample_dataset, maxcount=100)
    entry2, new_ds2 = repo.insert(sample_dataset, maxcount=100)

    assert entry1.uuid != entry2.uuid
    assert entry1.uuid in new_ds1.url
    assert entry2.uuid in new_ds2.url
    assert len(repo.index.all_entries) == 2


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
@pytest.mark.filterwarnings("ignore:coroutine.*was never awaited:RuntimeWarning")
def test_repo_insert_empty_dataset(mock_s3, clean_redis, tmp_path):
    """Test inserting an empty dataset.

    WebDataset's ShardWriter creates a shard file even with no samples,
    so empty datasets succeed (creating an empty shard) rather than raising
    RuntimeError.
    """
    dataset_path = tmp_path / "empty-dataset-000000.tar"
    with wds.writer.TarWriter(str(dataset_path)) as sink:
        pass  # Write no samples

    ds = atdata.Dataset[SimpleTestSample](url=str(dataset_path))
    repo = atlocal.Repo(
        s3_credentials=mock_s3['credentials'],
        hive_path=mock_s3['hive_path'],
        redis=clean_redis
    )

    # Empty datasets succeed because WebDataset creates a shard file regardless
    entry, new_ds = repo.insert(ds, maxcount=100)
    assert entry.uuid is not None
    assert '.tar' in new_ds.url


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
@pytest.mark.filterwarnings("ignore:coroutine.*was never awaited:RuntimeWarning")
def test_repo_insert_preserves_sample_type(mock_s3, clean_redis, sample_dataset):
    """Test that the returned Dataset preserves the original sample type.

    Should return a Dataset[T] with the same sample type as the input dataset.
    """
    repo = atlocal.Repo(
        s3_credentials=mock_s3['credentials'],
        hive_path=mock_s3['hive_path'],
        redis=clean_redis
    )

    entry, new_ds = repo.insert(sample_dataset, maxcount=100)

    assert new_ds.sample_type == SimpleTestSample
    assert entry.sample_kind == f"{SimpleTestSample.__module__}.SimpleTestSample"


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
@pytest.mark.filterwarnings("ignore:coroutine.*was never awaited:RuntimeWarning")
def test_repo_insert_round_trip(mock_s3, clean_redis, tmp_path):
    """Test full round-trip: insert dataset, then load and compare samples.

    Should be able to insert a dataset and then load it back from the returned
    URL with all samples intact and matching the original.
    """
    pytest.skip("Reading from moto-mocked S3 requires additional s3fs/WebDataset configuration")


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
@pytest.mark.filterwarnings("ignore:coroutine.*was never awaited:RuntimeWarning")
def test_repo_insert_with_shard_writer_kwargs(mock_s3, clean_redis, tmp_path):
    """Test that insert() passes additional kwargs to ShardWriter.

    Should forward kwargs like maxcount, maxsize to the underlying ShardWriter.
    """
    ds = make_simple_dataset(tmp_path, num_samples=30, name="large")
    repo = atlocal.Repo(
        s3_credentials=mock_s3['credentials'],
        hive_path=mock_s3['hive_path'],
        redis=clean_redis
    )

    entry, new_ds = repo.insert(ds, maxcount=5)

    assert '{' in new_ds.url and '}' in new_ds.url


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
@pytest.mark.filterwarnings("ignore:coroutine.*was never awaited:RuntimeWarning")
def test_repo_insert_numpy_arrays(mock_s3, clean_redis, tmp_path):
    """Test inserting a dataset containing samples with numpy arrays.

    Should correctly serialize and store numpy arrays.
    """
    ds = make_array_dataset(tmp_path, num_samples=3, array_shape=(10, 10))
    repo = atlocal.Repo(
        s3_credentials=mock_s3['credentials'],
        hive_path=mock_s3['hive_path'],
        redis=clean_redis
    )

    entry, new_ds = repo.insert(ds, maxcount=100)

    assert entry.uuid is not None
    assert entry.sample_kind == f"{ArrayTestSample.__module__}.ArrayTestSample"


##
# Integration tests

@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
@pytest.mark.filterwarnings("ignore:coroutine.*was never awaited:RuntimeWarning")
def test_repo_index_integration(mock_s3, clean_redis, sample_dataset):
    """Test that Repo and Index work together correctly.

    Should be able to insert datasets into Repo and retrieve their entries
    from the Index.
    """
    repo = atlocal.Repo(
        s3_credentials=mock_s3['credentials'],
        hive_path=mock_s3['hive_path'],
        redis=clean_redis
    )

    entry, new_ds = repo.insert(sample_dataset, maxcount=100)

    all_entries = repo.index.all_entries
    assert len(all_entries) == 1
    assert all_entries[0].uuid == entry.uuid
    assert all_entries[0].wds_url == entry.wds_url


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
@pytest.mark.filterwarnings("ignore:coroutine.*was never awaited:RuntimeWarning")
def test_multiple_datasets_same_type(mock_s3, clean_redis, sample_dataset):
    """Test inserting multiple datasets of the same sample type.

    Should create separate entries with different UUIDs and all should be
    retrievable from the index.
    """
    repo = atlocal.Repo(
        s3_credentials=mock_s3['credentials'],
        hive_path=mock_s3['hive_path'],
        redis=clean_redis
    )

    entry1, _ = repo.insert(sample_dataset, maxcount=100)
    entry2, _ = repo.insert(sample_dataset, maxcount=100)
    entry3, _ = repo.insert(sample_dataset, maxcount=100)

    uuids = {entry1.uuid, entry2.uuid, entry3.uuid}
    assert len(uuids) == 3

    all_entries = repo.index.all_entries
    assert len(all_entries) == 3

    for entry in all_entries:
        assert entry.sample_kind == f"{SimpleTestSample.__module__}.SimpleTestSample"


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
@pytest.mark.filterwarnings("ignore:coroutine.*was never awaited:RuntimeWarning")
def test_multiple_datasets_different_types(mock_s3, clean_redis, tmp_path):
    """Test inserting datasets with different sample types.

    Should correctly track sample_kind for each dataset and create distinct
    index entries.
    """
    simple_ds = make_simple_dataset(tmp_path, num_samples=3, name="simple")
    array_ds = make_array_dataset(tmp_path, num_samples=3, array_shape=(5, 5))

    repo = atlocal.Repo(
        s3_credentials=mock_s3['credentials'],
        hive_path=mock_s3['hive_path'],
        redis=clean_redis
    )

    entry1, _ = repo.insert(simple_ds, maxcount=100)
    entry2, _ = repo.insert(array_ds, maxcount=100)

    assert entry1.sample_kind == f"{SimpleTestSample.__module__}.SimpleTestSample"
    assert entry2.sample_kind == f"{ArrayTestSample.__module__}.ArrayTestSample"
    assert entry1.sample_kind != entry2.sample_kind
    assert len(repo.index.all_entries) == 2


def test_index_persistence_across_instances(clean_redis):
    """Test that index entries persist across Index instance recreations.

    Should be able to create an Index, add entries, create a new Index instance
    with the same Redis connection, and retrieve the same entries.
    """
    index1 = atlocal.Index(redis=clean_redis)
    ds = atdata.Dataset[SimpleTestSample](url="s3://bucket/dataset.tar")
    entry1 = index1.add_entry(ds)

    index2 = atlocal.Index(redis=clean_redis)
    entries = index2.all_entries

    assert len(entries) == 1
    assert entries[0].uuid == entry1.uuid
    assert entries[0].wds_url == entry1.wds_url


def test_concurrent_index_access(clean_redis):
    """Test that multiple Index instances can access the same Redis store.

    Should handle concurrent access to the same Redis index from multiple
    Index instances.
    """
    index1 = atlocal.Index(redis=clean_redis)
    index2 = atlocal.Index(redis=clean_redis)

    ds1 = atdata.Dataset[SimpleTestSample](url="s3://bucket/dataset1.tar")
    ds2 = atdata.Dataset[ArrayTestSample](url="s3://bucket/dataset2.tar")

    entry1 = index1.add_entry(ds1)
    entry2 = index2.add_entry(ds2)

    entries1 = index1.all_entries
    entries2 = index2.all_entries

    assert len(entries1) == 2
    assert len(entries2) == 2

    uuids1 = {e.uuid for e in entries1}
    uuids2 = {e.uuid for e in entries2}

    assert entry1.uuid in uuids1 and entry2.uuid in uuids1
    assert entry1.uuid in uuids2 and entry2.uuid in uuids2
