"""Test local repository storage functionality."""

##
# Imports

import pytest

# System
from dataclasses import dataclass
from pathlib import Path

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


##
# Test fixtures (redis_connection and clean_redis are in conftest.py)


@pytest.fixture
def mock_s3():
    """Provide a mock S3 environment using moto.

    Note: Tests using this fixture may generate warnings due to s3fs/moto async
    incompatibility. These are suppressed via @pytest.mark.filterwarnings on
    individual tests. See tests/EXPECTED_WARNINGS.md for details.
    """
    with mock_aws():
        # Create S3 credentials dict (no endpoint_url for moto)
        creds = {"AWS_ACCESS_KEY_ID": "testing", "AWS_SECRET_ACCESS_KEY": "testing"}

        # Create S3 client and bucket
        import boto3

        s3_client = boto3.client(
            "s3",
            aws_access_key_id=creds["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=creds["AWS_SECRET_ACCESS_KEY"],
            region_name="us-east-1",
        )

        bucket_name = "test-bucket"
        s3_client.create_bucket(Bucket=bucket_name)

        yield {
            "credentials": creds,
            "bucket": bucket_name,
            "hive_path": f"{bucket_name}/datasets",
            "s3_client": s3_client,
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
    """Simple test sample for repository tests.

    Note: This matches SharedBasicSample in conftest.py but is kept local
    because tests verify class name behavior.
    """

    name: str
    value: int


@dataclass
class ArrayTestSample(atdata.PackableSample):
    """Test sample with numpy array for repository tests.

    Note: Similar to SharedNumpySample but kept local for test isolation.
    """

    label: str
    data: NDArray


def make_simple_dataset(
    tmp_path: Path, num_samples: int = 10, name: str = "test"
) -> atdata.Dataset:
    """Create a SimpleTestSample dataset for testing."""
    dataset_path = tmp_path / f"{name}-dataset-000000.tar"
    with wds.writer.TarWriter(str(dataset_path)) as sink:
        for i in range(num_samples):
            sample = SimpleTestSample(name=f"sample_{i}", value=i * 10)
            sink.write(sample.as_wds)
    return atdata.Dataset[SimpleTestSample](url=str(dataset_path))


def make_array_dataset(
    tmp_path: Path, num_samples: int = 3, array_shape: tuple = (10, 10)
) -> atdata.Dataset:
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
        "AWS_ENDPOINT": "http://localhost:9000",
        "AWS_ACCESS_KEY_ID": "minioadmin",
        "AWS_SECRET_ACCESS_KEY": "minioadmin",
    }


@pytest.mark.parametrize(
    "missing_field,env_content",
    [
        (
            "AWS_ENDPOINT",
            "AWS_ACCESS_KEY_ID=minioadmin\nAWS_SECRET_ACCESS_KEY=minioadmin\n",
        ),
        (
            "AWS_ACCESS_KEY_ID",
            "AWS_ENDPOINT=http://localhost:9000\nAWS_SECRET_ACCESS_KEY=minioadmin\n",
        ),
        (
            "AWS_SECRET_ACCESS_KEY",
            "AWS_ENDPOINT=http://localhost:9000\nAWS_ACCESS_KEY_ID=minioadmin\n",
        ),
    ],
)
def test_s3_env_missing_required_field(tmp_path, missing_field, env_content):
    """Test that loading S3 credentials fails when a required field is missing.

    Should raise ValueError when .env file lacks any of the required fields:
    AWS_ENDPOINT, AWS_ACCESS_KEY_ID, or AWS_SECRET_ACCESS_KEY.
    """
    env_file = tmp_path / ".env"
    env_file.write_text(env_content)

    with pytest.raises(ValueError, match=missing_field):
        atlocal._s3_env(env_file)


def test_s3_from_credentials_with_dict():
    """Test creating S3FileSystem from a credentials dictionary.

    Should create a properly configured S3FileSystem instance using dict credentials.
    """
    creds = {
        "AWS_ENDPOINT": "http://localhost:9000",
        "AWS_ACCESS_KEY_ID": "minioadmin",
        "AWS_SECRET_ACCESS_KEY": "minioadmin",
    }

    fs = atlocal._s3_from_credentials(creds)

    assert isinstance(fs, atlocal.S3FileSystem)
    assert fs.endpoint_url == "http://localhost:9000"
    assert fs.key == "minioadmin"
    assert fs.secret == "minioadmin"


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
    assert fs.endpoint_url == "http://localhost:9000"
    assert fs.key == "minioadmin"
    assert fs.secret == "minioadmin"


##
# LocalDatasetEntry tests


def test_local_dataset_entry_creation():
    """Test creating a LocalDatasetEntry with explicit values.

    Should create an entry with provided name, schema_ref, data_urls, and generate CID.
    """
    entry = atlocal.LocalDatasetEntry(
        name="test-dataset",
        schema_ref="local://schemas/test_module.TestSample@1.0.0",
        data_urls=["s3://bucket/dataset.tar"],
        metadata={"description": "test"},
    )

    assert entry.name == "test-dataset"
    assert entry.schema_ref == "local://schemas/test_module.TestSample@1.0.0"
    assert entry.data_urls == ["s3://bucket/dataset.tar"]
    assert entry.metadata == {"description": "test"}
    # CID should be auto-generated
    assert entry.cid is not None
    assert entry.cid.startswith("bafy")


def test_local_dataset_entry_cid_generation():
    """Test that LocalDatasetEntry generates deterministic CIDs.

    Same content should produce the same CID.
    """
    entry1 = atlocal.LocalDatasetEntry(
        name="test-dataset",
        schema_ref="local://schemas/test_module.TestSample@1.0.0",
        data_urls=["s3://bucket/dataset.tar"],
    )
    entry2 = atlocal.LocalDatasetEntry(
        name="test-dataset",  # Name doesn't affect CID
        schema_ref="local://schemas/test_module.TestSample@1.0.0",
        data_urls=["s3://bucket/dataset.tar"],
    )

    # Same schema_ref and data_urls = same CID
    assert entry1.cid == entry2.cid


def test_local_dataset_entry_different_content_different_cid():
    """Test that different content produces different CIDs."""
    entry1 = atlocal.LocalDatasetEntry(
        name="dataset1",
        schema_ref="local://schemas/test_module.TestSample@1.0.0",
        data_urls=["s3://bucket/dataset1.tar"],
    )
    entry2 = atlocal.LocalDatasetEntry(
        name="dataset2",
        schema_ref="local://schemas/test_module.TestSample@1.0.0",
        data_urls=["s3://bucket/dataset2.tar"],  # Different URL
    )

    assert entry1.cid != entry2.cid


def test_local_dataset_entry_write_to_redis(clean_redis):
    """Test persisting a LocalDatasetEntry to Redis.

    Should write the entry to Redis as a hash with key 'LocalDatasetEntry:{cid}'
    and all fields should be retrievable with correct values.
    """
    entry = atlocal.LocalDatasetEntry(
        name="test-dataset",
        schema_ref="local://schemas/test_module.TestSample@1.0.0",
        data_urls=["s3://bucket/dataset.tar"],
        metadata={"version": "1.0"},
    )

    entry.write_to(clean_redis)

    # Verify key exists
    assert clean_redis.exists(f"LocalDatasetEntry:{entry.cid}")

    # Load back and verify
    loaded = atlocal.LocalDatasetEntry.from_redis(clean_redis, entry.cid)
    assert loaded.name == entry.name
    assert loaded.schema_ref == entry.schema_ref
    assert loaded.data_urls == entry.data_urls
    assert loaded.metadata == entry.metadata
    assert loaded.cid == entry.cid


def test_local_dataset_entry_round_trip_redis(clean_redis):
    """Test writing and reading a LocalDatasetEntry from Redis.

    Should be able to write an entry to Redis and read it back with all fields
    intact and matching the original values.
    """
    original_entry = atlocal.LocalDatasetEntry(
        name="my-dataset",
        schema_ref="local://schemas/module.Sample@2.0.0",
        data_urls=["s3://bucket/data-{000000..000009}.tar"],
        metadata={"author": "test", "tags": ["a", "b"]},
    )

    original_entry.write_to(clean_redis)

    # Read back from Redis
    retrieved_entry = atlocal.LocalDatasetEntry.from_redis(
        clean_redis, original_entry.cid
    )

    assert retrieved_entry.name == original_entry.name
    assert retrieved_entry.schema_ref == original_entry.schema_ref
    assert retrieved_entry.data_urls == original_entry.data_urls
    assert retrieved_entry.metadata == original_entry.metadata
    assert retrieved_entry.cid == original_entry.cid


def test_local_dataset_entry_legacy_properties():
    """Test that legacy properties work for backwards compatibility."""
    entry = atlocal.LocalDatasetEntry(
        name="test-dataset",
        schema_ref="local://schemas/test_module.TestSample@1.0.0",
        data_urls=["s3://bucket/dataset.tar"],
    )

    # Legacy properties should work
    assert entry.wds_url == "s3://bucket/dataset.tar"
    assert entry.sample_kind == "local://schemas/test_module.TestSample@1.0.0"


def test_local_dataset_entry_implements_index_entry_protocol():
    """Test that LocalDatasetEntry implements the IndexEntry protocol."""
    from atdata._protocols import IndexEntry

    entry = atlocal.LocalDatasetEntry(
        name="test-dataset",
        schema_ref="local://schemas/test_module.TestSample@1.0.0",
        data_urls=["s3://bucket/dataset.tar"],
    )

    # Should satisfy the protocol
    assert isinstance(entry, IndexEntry)


def test_index_implements_abstract_index_protocol():
    """Test that Index has all AbstractIndex protocol methods."""
    index = atlocal.Index()

    # Check protocol methods exist
    assert hasattr(index, "insert_dataset")
    assert hasattr(index, "get_dataset")
    assert hasattr(index, "list_datasets")
    assert hasattr(index, "publish_schema")
    assert hasattr(index, "get_schema")
    assert hasattr(index, "list_schemas")
    assert hasattr(index, "decode_schema")

    # Check they are callable
    assert callable(index.insert_dataset)
    assert callable(index.get_dataset)
    assert callable(index.list_datasets)


##
# Index tests


def test_index_init_default_sqlite():
    """Test creating an Index with default SQLite provider.

    When no provider or redis argument is given, the Index should use
    SQLite as the zero-dependency default.
    """
    from atdata.providers._sqlite import SqliteProvider

    index = atlocal.Index()

    assert isinstance(index.provider, SqliteProvider)


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
    index = atlocal.Index(host="localhost", port=6379, db=0)

    assert index._redis is not None
    assert isinstance(index._redis, Redis)


def test_index_add_entry(clean_redis):
    """Test adding a dataset entry to the index.

    Should create a LocalDatasetEntry with auto-generated CID and persist it to Redis.
    """
    index = atlocal.Index(redis=clean_redis)

    ds = atdata.Dataset[SimpleTestSample](
        url="s3://bucket/dataset.tar", metadata_url="s3://bucket/metadata.msgpack"
    )

    entry = index.add_entry(ds, name="test-dataset")

    assert entry.cid is not None
    assert entry.cid.startswith("bafy")
    assert entry.name == "test-dataset"
    assert entry.data_urls == ["s3://bucket/dataset.tar"]
    assert "SimpleTestSample" in entry.schema_ref

    # Verify it was persisted to Redis
    stored_data = clean_redis.hgetall(f"LocalDatasetEntry:{entry.cid}")
    assert len(stored_data) > 0


def test_index_add_entry_with_schema_ref(clean_redis):
    """Test adding a dataset entry with explicit schema_ref.

    Should use the provided schema_ref instead of auto-generating.
    """
    index = atlocal.Index(redis=clean_redis)

    ds = atdata.Dataset[SimpleTestSample](url="s3://bucket/dataset.tar")

    entry = index.add_entry(
        ds, name="test-dataset", schema_ref="local://schemas/custom.Schema@2.0.0"
    )

    assert entry.schema_ref == "local://schemas/custom.Schema@2.0.0"


def test_index_add_entry_with_metadata(clean_redis):
    """Test adding a dataset entry with metadata.

    Should store the provided metadata.
    """
    index = atlocal.Index(redis=clean_redis)

    ds = atdata.Dataset[SimpleTestSample](url="s3://bucket/dataset.tar")

    entry = index.add_entry(
        ds, name="test-dataset", metadata={"version": "1.0", "author": "test"}
    )

    assert entry.metadata == {"version": "1.0", "author": "test"}


def test_index_entries_generator_empty(clean_redis):
    """Test iterating over entries in an empty index.

    Should yield no entries when the index is empty.
    """
    index = atlocal.Index(redis=clean_redis)

    entries = list(index.entries)
    assert len(entries) == 0


def test_index_entries_generator_multiple(clean_redis):
    """Test iterating over multiple entries in the index.

    Should yield all LocalDatasetEntry objects that have been added to the index.
    """
    index = atlocal.Index(redis=clean_redis)

    ds1 = atdata.Dataset[SimpleTestSample](url="s3://bucket/dataset1.tar")
    ds2 = atdata.Dataset[ArrayTestSample](url="s3://bucket/dataset2.tar")

    entry1 = index.add_entry(ds1, name="dataset1")
    entry2 = index.add_entry(ds2, name="dataset2")

    entries = list(index.entries)
    assert len(entries) == 2

    cids = {entry.cid for entry in entries}
    assert entry1.cid in cids
    assert entry2.cid in cids


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

    Should return a list containing all LocalDatasetEntry objects in the index.
    """
    index = atlocal.Index(redis=clean_redis)

    ds1 = atdata.Dataset[SimpleTestSample](url="s3://bucket/dataset1.tar")
    ds2 = atdata.Dataset[ArrayTestSample](url="s3://bucket/dataset2.tar")

    index.add_entry(ds1, name="dataset1")
    index.add_entry(ds2, name="dataset2")

    entries = index.all_entries
    assert isinstance(entries, list)
    assert len(entries) == 2


def test_index_entries_filtering(clean_redis):
    """Test that index only returns LocalDatasetEntry objects.

    Should only iterate over keys matching 'LocalDatasetEntry:*' pattern and
    ignore any other Redis keys.
    """
    index = atlocal.Index(redis=clean_redis)

    # Add a LocalDatasetEntry
    ds = atdata.Dataset[SimpleTestSample](url="s3://bucket/dataset.tar")
    entry = index.add_entry(ds, name="test-dataset")

    # Add some other Redis keys that should be ignored
    clean_redis.set("other_key", "value")
    clean_redis.hset("other_hash", "field", "value")

    entries = list(index.entries)
    assert len(entries) == 1
    assert entries[0].cid == entry.cid

    # Clean up non-LocalDatasetEntry keys (fixture only cleans LocalDatasetEntry:*)
    clean_redis.delete("other_key")
    clean_redis.delete("other_hash")


def test_index_get_entry_by_cid(clean_redis):
    """Test retrieving an entry by its CID."""
    index = atlocal.Index(redis=clean_redis)

    ds = atdata.Dataset[SimpleTestSample](url="s3://bucket/dataset.tar")
    entry = index.add_entry(ds, name="test-dataset")

    retrieved = index.get_entry(entry.cid)

    assert retrieved.cid == entry.cid
    assert retrieved.name == entry.name
    assert retrieved.data_urls == entry.data_urls


def test_index_get_entry_by_name(clean_redis):
    """Test retrieving an entry by its name."""
    index = atlocal.Index(redis=clean_redis)

    ds = atdata.Dataset[SimpleTestSample](url="s3://bucket/dataset.tar")
    entry = index.add_entry(ds, name="my-special-dataset")

    retrieved = index.get_entry_by_name("my-special-dataset")

    assert retrieved.cid == entry.cid
    assert retrieved.name == "my-special-dataset"


def test_index_get_entry_by_name_not_found(clean_redis):
    """Test that get_entry_by_name raises KeyError for unknown name."""
    index = atlocal.Index(redis=clean_redis)

    with pytest.raises(KeyError, match="No entry with name"):
        index.get_entry_by_name("nonexistent")


##
# AbstractIndex protocol method tests


def test_index_insert_dataset(clean_redis):
    """Test insert_dataset protocol method."""
    index = atlocal.Index(redis=clean_redis)
    ds = atdata.Dataset[SimpleTestSample](url="s3://bucket/dataset.tar")

    entry = index.insert_dataset(ds, name="protocol-test")

    assert entry.name == "protocol-test"
    assert entry.cid is not None


def test_index_get_dataset(clean_redis):
    """Test get_dataset protocol method."""
    index = atlocal.Index(redis=clean_redis)
    ds = atdata.Dataset[SimpleTestSample](url="s3://bucket/dataset.tar")
    index.insert_dataset(ds, name="my-dataset")

    entry = index.get_dataset("my-dataset")

    assert entry.name == "my-dataset"


def test_index_get_dataset_not_found(clean_redis):
    """Test get_dataset raises KeyError for unknown name."""
    index = atlocal.Index(redis=clean_redis)

    with pytest.raises(KeyError):
        index.get_dataset("nonexistent")


def test_index_list_datasets(clean_redis):
    """Test list_datasets protocol method."""
    index = atlocal.Index(redis=clean_redis)
    ds1 = atdata.Dataset[SimpleTestSample](url="s3://bucket/ds1.tar")
    ds2 = atdata.Dataset[SimpleTestSample](url="s3://bucket/ds2.tar")

    index.insert_dataset(ds1, name="dataset-1")
    index.insert_dataset(ds2, name="dataset-2")

    datasets = list(index.list_datasets())

    assert len(datasets) == 2
    names = {d.name for d in datasets}
    assert names == {"dataset-1", "dataset-2"}


##
# Repo tests - Initialization
# Note: Repo is deprecated; these tests verify backwards compatibility


@pytest.mark.filterwarnings("ignore:Repo is deprecated:DeprecationWarning")
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


@pytest.mark.filterwarnings("ignore:Repo is deprecated:DeprecationWarning")
def test_repo_init_with_s3_dict():
    """Test creating a Repo with S3 credentials as a dictionary.

    Should create a Repo with S3FileSystem and set hive_path and hive_bucket.
    """
    creds = {
        "AWS_ENDPOINT": "http://localhost:9000",
        "AWS_ACCESS_KEY_ID": "minioadmin",
        "AWS_SECRET_ACCESS_KEY": "minioadmin",
    }

    repo = atlocal.Repo(s3_credentials=creds, hive_path="test-bucket/datasets")

    assert repo.s3_credentials == creds
    assert repo.bucket_fs is not None
    assert isinstance(repo.bucket_fs, atlocal.S3FileSystem)
    assert repo.hive_path == Path("test-bucket/datasets")
    assert repo.hive_bucket == "test-bucket"


@pytest.mark.filterwarnings("ignore:Repo is deprecated:DeprecationWarning")
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


@pytest.mark.filterwarnings("ignore:Repo is deprecated:DeprecationWarning")
def test_repo_init_s3_without_hive_path():
    """Test that creating a Repo with S3 but no hive_path raises ValueError.

    Should raise ValueError when s3_credentials is provided but hive_path is None.
    """
    creds = {
        "AWS_ENDPOINT": "http://localhost:9000",
        "AWS_ACCESS_KEY_ID": "minioadmin",
        "AWS_SECRET_ACCESS_KEY": "minioadmin",
    }

    with pytest.raises(ValueError, match="Must specify hive path"):
        atlocal.Repo(s3_credentials=creds)


@pytest.mark.filterwarnings("ignore:Repo is deprecated:DeprecationWarning")
def test_repo_init_hive_path_parsing():
    """Test that hive_path is correctly parsed to extract bucket name.

    Should set hive_bucket to the first component of hive_path.
    """
    creds = {
        "AWS_ENDPOINT": "http://localhost:9000",
        "AWS_ACCESS_KEY_ID": "minioadmin",
        "AWS_SECRET_ACCESS_KEY": "minioadmin",
    }

    repo = atlocal.Repo(s3_credentials=creds, hive_path="my-bucket/path/to/datasets")

    assert repo.hive_bucket == "my-bucket"
    assert repo.hive_path == Path("my-bucket/path/to/datasets")


@pytest.mark.filterwarnings("ignore:Repo is deprecated:DeprecationWarning")
def test_repo_init_with_custom_redis():
    """Test creating a Repo with a custom Redis connection.

    Should pass the Redis connection to the Index instance.
    """
    custom_redis = Redis()
    repo = atlocal.Repo(redis=custom_redis)

    assert repo.index._redis is custom_redis


##
# Repo tests - Insert functionality


@pytest.mark.filterwarnings("ignore:Repo is deprecated:DeprecationWarning")
def test_repo_insert_without_s3():
    """Test that inserting a dataset without S3 configured raises ValueError.

    Should fail with ValueError when trying to insert without S3 credentials.
    """
    repo = atlocal.Repo()
    ds = atdata.Dataset[SimpleTestSample](url="s3://bucket/dataset.tar")

    with pytest.raises(ValueError, match="S3 credentials required"):
        repo.insert(ds, name="test-dataset")


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
@pytest.mark.filterwarnings("ignore:coroutine.*was never awaited:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:Repo is deprecated:DeprecationWarning")
def test_repo_insert_single_shard(mock_s3, clean_redis, sample_dataset):
    """Test inserting a small dataset that fits in a single shard.

    Should write the dataset to S3, create metadata, add index entry, and return
    a new Dataset pointing to the stored copy with correct URL format.
    """
    repo = atlocal.Repo(
        s3_credentials=mock_s3["credentials"],
        hive_path=mock_s3["hive_path"],
        redis=clean_redis,
    )

    entry, new_ds = repo.insert(
        sample_dataset, name="single-shard-dataset", maxcount=100
    )

    assert entry.cid is not None
    assert entry.cid.startswith("bafy")
    assert entry.name == "single-shard-dataset"
    assert len(entry.data_urls) > 0
    assert "SimpleTestSample" in entry.schema_ref
    assert len(repo.index.all_entries) == 1
    assert ".tar" in new_ds.url
    assert new_ds.url.startswith(mock_s3["hive_path"])


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
@pytest.mark.filterwarnings("ignore:coroutine.*was never awaited:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:Repo is deprecated:DeprecationWarning")
def test_repo_insert_multiple_shards(mock_s3, clean_redis, tmp_path):
    """Test inserting a large dataset that spans multiple shards.

    Should write multiple tar files to S3, use brace notation in returned URL,
    and correctly format the shard range.
    """
    ds = make_simple_dataset(tmp_path, num_samples=50, name="large")
    repo = atlocal.Repo(
        s3_credentials=mock_s3["credentials"],
        hive_path=mock_s3["hive_path"],
        redis=clean_redis,
    )

    entry, new_ds = repo.insert(ds, name="multi-shard-dataset", maxcount=10)

    assert entry.cid is not None
    assert len(entry.data_urls) > 0
    assert "{" in new_ds.url and "}" in new_ds.url


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
@pytest.mark.filterwarnings("ignore:coroutine.*was never awaited:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:Repo is deprecated:DeprecationWarning")
def test_repo_insert_with_metadata(mock_s3, clean_redis, tmp_path):
    """Test inserting a dataset with metadata.

    Should write metadata as msgpack to S3 and store metadata in the entry.
    """
    ds = make_simple_dataset(tmp_path, num_samples=5)
    ds._metadata = {"description": "test dataset", "version": "1.0"}

    repo = atlocal.Repo(
        s3_credentials=mock_s3["credentials"],
        hive_path=mock_s3["hive_path"],
        redis=clean_redis,
    )

    entry, new_ds = repo.insert(ds, name="metadata-dataset", maxcount=100)

    assert entry.metadata is not None
    assert entry.metadata.get("description") == "test dataset"
    assert new_ds.metadata_url is not None


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
@pytest.mark.filterwarnings("ignore:coroutine.*was never awaited:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:Repo is deprecated:DeprecationWarning")
def test_repo_insert_without_metadata(mock_s3, clean_redis, tmp_path):
    """Test inserting a dataset without metadata.

    Should handle None metadata gracefully and not write a metadata file.
    """
    ds = make_simple_dataset(tmp_path, num_samples=5)
    repo = atlocal.Repo(
        s3_credentials=mock_s3["credentials"],
        hive_path=mock_s3["hive_path"],
        redis=clean_redis,
    )

    entry, new_ds = repo.insert(ds, name="no-metadata-dataset", maxcount=100)

    assert entry.cid is not None
    assert len(repo.index.all_entries) == 1


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
@pytest.mark.filterwarnings("ignore:coroutine.*was never awaited:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:Repo is deprecated:DeprecationWarning")
def test_repo_insert_cache_local_false(mock_s3, clean_redis, sample_dataset):
    """Test inserting with cache_local=False (direct S3 write).

    Should write tar shards directly to S3 without local caching.
    """
    repo = atlocal.Repo(
        s3_credentials=mock_s3["credentials"],
        hive_path=mock_s3["hive_path"],
        redis=clean_redis,
    )

    entry, new_ds = repo.insert(
        sample_dataset, name="direct-write", cache_local=False, maxcount=100
    )

    assert entry.cid is not None
    assert len(entry.data_urls) > 0


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
@pytest.mark.filterwarnings("ignore:coroutine.*was never awaited:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:Repo is deprecated:DeprecationWarning")
def test_repo_insert_cache_local_true(mock_s3, clean_redis, sample_dataset):
    """Test inserting with cache_local=True (local cache then copy).

    Should write to temporary local storage first, then copy to S3, and clean up
    local cache files after copying.
    """
    repo = atlocal.Repo(
        s3_credentials=mock_s3["credentials"],
        hive_path=mock_s3["hive_path"],
        redis=clean_redis,
    )

    entry, new_ds = repo.insert(
        sample_dataset, name="cached-write", cache_local=True, maxcount=100
    )

    assert entry.cid is not None
    assert len(entry.data_urls) > 0


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
@pytest.mark.filterwarnings("ignore:coroutine.*was never awaited:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:Repo is deprecated:DeprecationWarning")
def test_repo_insert_creates_index_entry(mock_s3, clean_redis, sample_dataset):
    """Test that insert() creates a valid index entry.

    Should add a LocalDatasetEntry to the index with correct data_urls, schema_ref,
    and CID.
    """
    repo = atlocal.Repo(
        s3_credentials=mock_s3["credentials"],
        hive_path=mock_s3["hive_path"],
        redis=clean_redis,
    )

    entry, new_ds = repo.insert(sample_dataset, name="indexed-dataset", maxcount=100)

    assert entry.cid is not None
    assert entry.data_urls == [new_ds.url]
    assert "SimpleTestSample" in entry.schema_ref

    all_entries = repo.index.all_entries
    assert len(all_entries) == 1
    assert all_entries[0].cid == entry.cid


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
@pytest.mark.filterwarnings("ignore:coroutine.*was never awaited:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:Repo is deprecated:DeprecationWarning")
def test_repo_insert_cid_generation(mock_s3, clean_redis, sample_dataset):
    """Test that insert() generates unique CIDs for each dataset.

    Should create different CIDs for datasets with different URLs.
    """
    repo = atlocal.Repo(
        s3_credentials=mock_s3["credentials"],
        hive_path=mock_s3["hive_path"],
        redis=clean_redis,
    )

    entry1, new_ds1 = repo.insert(sample_dataset, name="dataset1", maxcount=100)
    entry2, new_ds2 = repo.insert(sample_dataset, name="dataset2", maxcount=100)

    # Different URLs should produce different CIDs
    assert entry1.cid != entry2.cid
    assert len(repo.index.all_entries) == 2


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
@pytest.mark.filterwarnings("ignore:coroutine.*was never awaited:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:Repo is deprecated:DeprecationWarning")
def test_repo_insert_empty_dataset(mock_s3, clean_redis, tmp_path):
    """Test inserting an empty dataset.

    WebDataset's ShardWriter creates a shard file even with no samples,
    so empty datasets succeed (creating an empty shard) rather than raising
    RuntimeError.
    """
    dataset_path = tmp_path / "empty-dataset-000000.tar"
    with wds.writer.TarWriter(str(dataset_path)):
        pass  # Write no samples

    ds = atdata.Dataset[SimpleTestSample](url=str(dataset_path))
    repo = atlocal.Repo(
        s3_credentials=mock_s3["credentials"],
        hive_path=mock_s3["hive_path"],
        redis=clean_redis,
    )

    # Empty datasets succeed because WebDataset creates a shard file regardless
    entry, new_ds = repo.insert(ds, name="empty-dataset", maxcount=100)
    assert entry.cid is not None
    assert ".tar" in new_ds.url


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
@pytest.mark.filterwarnings("ignore:coroutine.*was never awaited:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:Repo is deprecated:DeprecationWarning")
def test_repo_insert_preserves_sample_type(mock_s3, clean_redis, sample_dataset):
    """Test that the returned Dataset preserves the original sample type.

    Should return a Dataset[T] with the same sample type as the input dataset.
    """
    repo = atlocal.Repo(
        s3_credentials=mock_s3["credentials"],
        hive_path=mock_s3["hive_path"],
        redis=clean_redis,
    )

    entry, new_ds = repo.insert(sample_dataset, name="typed-dataset", maxcount=100)

    assert new_ds.sample_type == SimpleTestSample
    assert "SimpleTestSample" in entry.schema_ref


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
@pytest.mark.filterwarnings("ignore:coroutine.*was never awaited:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:Repo is deprecated:DeprecationWarning")
def test_repo_insert_with_shard_writer_kwargs(mock_s3, clean_redis, tmp_path):
    """Test that insert() passes additional kwargs to ShardWriter.

    Should forward kwargs like maxcount, maxsize to the underlying ShardWriter.
    """
    ds = make_simple_dataset(tmp_path, num_samples=30, name="large")
    repo = atlocal.Repo(
        s3_credentials=mock_s3["credentials"],
        hive_path=mock_s3["hive_path"],
        redis=clean_redis,
    )

    entry, new_ds = repo.insert(ds, name="sharded-dataset", maxcount=5)

    assert "{" in new_ds.url and "}" in new_ds.url


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
@pytest.mark.filterwarnings("ignore:coroutine.*was never awaited:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:Repo is deprecated:DeprecationWarning")
def test_repo_insert_numpy_arrays(mock_s3, clean_redis, tmp_path):
    """Test inserting a dataset containing samples with numpy arrays.

    Should correctly serialize and store numpy arrays.
    """
    ds = make_array_dataset(tmp_path, num_samples=3, array_shape=(10, 10))
    repo = atlocal.Repo(
        s3_credentials=mock_s3["credentials"],
        hive_path=mock_s3["hive_path"],
        redis=clean_redis,
    )

    entry, new_ds = repo.insert(ds, name="array-dataset", maxcount=100)

    assert entry.cid is not None
    assert "ArrayTestSample" in entry.schema_ref


##
# Integration tests


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
@pytest.mark.filterwarnings("ignore:coroutine.*was never awaited:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:Repo is deprecated:DeprecationWarning")
def test_repo_index_integration(mock_s3, clean_redis, sample_dataset):
    """Test that Repo and Index work together correctly.

    Should be able to insert datasets into Repo and retrieve their entries
    from the Index.
    """
    repo = atlocal.Repo(
        s3_credentials=mock_s3["credentials"],
        hive_path=mock_s3["hive_path"],
        redis=clean_redis,
    )

    entry, new_ds = repo.insert(sample_dataset, name="integrated-dataset", maxcount=100)

    all_entries = repo.index.all_entries
    assert len(all_entries) == 1
    assert all_entries[0].cid == entry.cid
    assert all_entries[0].data_urls == entry.data_urls


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
@pytest.mark.filterwarnings("ignore:coroutine.*was never awaited:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:Repo is deprecated:DeprecationWarning")
def test_multiple_datasets_same_type(mock_s3, clean_redis, sample_dataset):
    """Test inserting multiple datasets of the same sample type.

    Should create separate entries with different CIDs and all should be
    retrievable from the index.
    """
    repo = atlocal.Repo(
        s3_credentials=mock_s3["credentials"],
        hive_path=mock_s3["hive_path"],
        redis=clean_redis,
    )

    entry1, _ = repo.insert(sample_dataset, name="dataset-a", maxcount=100)
    entry2, _ = repo.insert(sample_dataset, name="dataset-b", maxcount=100)
    entry3, _ = repo.insert(sample_dataset, name="dataset-c", maxcount=100)

    cids = {entry1.cid, entry2.cid, entry3.cid}
    assert len(cids) == 3

    all_entries = repo.index.all_entries
    assert len(all_entries) == 3

    for entry in all_entries:
        assert "SimpleTestSample" in entry.schema_ref


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
@pytest.mark.filterwarnings("ignore:coroutine.*was never awaited:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:Repo is deprecated:DeprecationWarning")
def test_multiple_datasets_different_types(mock_s3, clean_redis, tmp_path):
    """Test inserting datasets with different sample types.

    Should correctly track schema_ref for each dataset and create distinct
    index entries.
    """
    simple_ds = make_simple_dataset(tmp_path, num_samples=3, name="simple")
    array_ds = make_array_dataset(tmp_path, num_samples=3, array_shape=(5, 5))

    repo = atlocal.Repo(
        s3_credentials=mock_s3["credentials"],
        hive_path=mock_s3["hive_path"],
        redis=clean_redis,
    )

    entry1, _ = repo.insert(simple_ds, name="simple-dataset", maxcount=100)
    entry2, _ = repo.insert(array_ds, name="array-dataset", maxcount=100)

    assert "SimpleTestSample" in entry1.schema_ref
    assert "ArrayTestSample" in entry2.schema_ref
    assert entry1.schema_ref != entry2.schema_ref
    assert len(repo.index.all_entries) == 2


def test_index_persistence_across_instances(clean_redis):
    """Test that index entries persist across Index instance recreations.

    Should be able to create an Index, add entries, create a new Index instance
    with the same Redis connection, and retrieve the same entries.
    """
    index1 = atlocal.Index(redis=clean_redis)
    ds = atdata.Dataset[SimpleTestSample](url="s3://bucket/dataset.tar")
    entry1 = index1.add_entry(ds, name="persistent-dataset")

    index2 = atlocal.Index(redis=clean_redis)
    entries = index2.all_entries

    assert len(entries) == 1
    assert entries[0].cid == entry1.cid
    assert entries[0].data_urls == entry1.data_urls


def test_concurrent_index_access(clean_redis):
    """Test that multiple Index instances can access the same Redis store.

    Should handle concurrent access to the same Redis index from multiple
    Index instances.
    """
    index1 = atlocal.Index(redis=clean_redis)
    index2 = atlocal.Index(redis=clean_redis)

    ds1 = atdata.Dataset[SimpleTestSample](url="s3://bucket/dataset1.tar")
    ds2 = atdata.Dataset[ArrayTestSample](url="s3://bucket/dataset2.tar")

    entry1 = index1.add_entry(ds1, name="dataset1")
    entry2 = index2.add_entry(ds2, name="dataset2")

    entries1 = index1.all_entries
    entries2 = index2.all_entries

    assert len(entries1) == 2
    assert len(entries2) == 2

    cids1 = {e.cid for e in entries1}
    cids2 = {e.cid for e in entries2}

    assert entry1.cid in cids1 and entry2.cid in cids1
    assert entry1.cid in cids2 and entry2.cid in cids2


##
# S3DataStore tests


def test_s3_datastore_init():
    """Test creating an S3DataStore."""
    creds = {
        "AWS_ENDPOINT": "http://localhost:9000",
        "AWS_ACCESS_KEY_ID": "minioadmin",
        "AWS_SECRET_ACCESS_KEY": "minioadmin",
    }

    store = atlocal.S3DataStore(credentials=creds, bucket="test-bucket")

    assert store.bucket == "test-bucket"
    assert store.credentials == creds
    assert store._fs is not None


def test_s3_datastore_supports_streaming():
    """Test that S3DataStore reports streaming support."""
    creds = {"AWS_ACCESS_KEY_ID": "test", "AWS_SECRET_ACCESS_KEY": "test"}

    store = atlocal.S3DataStore(credentials=creds, bucket="test")

    assert store.supports_streaming() is True


def test_s3_datastore_read_url():
    """Test that read_url returns URL unchanged without custom endpoint."""
    creds = {"AWS_ACCESS_KEY_ID": "test", "AWS_SECRET_ACCESS_KEY": "test"}

    store = atlocal.S3DataStore(credentials=creds, bucket="test")

    url = "s3://bucket/path/to/data.tar"
    assert store.read_url(url) == url


def test_s3_datastore_read_url_with_custom_endpoint():
    """Test that read_url transforms s3:// to https:// with custom endpoint."""
    creds = {
        "AWS_ACCESS_KEY_ID": "test",
        "AWS_SECRET_ACCESS_KEY": "test",
        "AWS_ENDPOINT": "https://abc123.r2.cloudflarestorage.com",
    }

    store = atlocal.S3DataStore(credentials=creds, bucket="test")

    # s3:// URL should be transformed to https:// using the endpoint
    url = "s3://my-bucket/path/to/data.tar"
    expected = "https://abc123.r2.cloudflarestorage.com/my-bucket/path/to/data.tar"
    assert store.read_url(url) == expected

    # Trailing slash on endpoint should be handled
    creds["AWS_ENDPOINT"] = "https://endpoint.example.com/"
    store2 = atlocal.S3DataStore(credentials=creds, bucket="test")
    assert (
        store2.read_url(url)
        == "https://endpoint.example.com/my-bucket/path/to/data.tar"
    )

    # Non-s3 URLs should be passed through unchanged
    https_url = "https://example.com/data.tar"
    assert store.read_url(https_url) == https_url


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
@pytest.mark.filterwarnings("ignore:coroutine.*was never awaited:RuntimeWarning")
def test_s3_datastore_write_shards(mock_s3, tmp_path):
    """Test writing shards with S3DataStore."""
    ds = make_simple_dataset(tmp_path, num_samples=5)

    store = atlocal.S3DataStore(
        credentials=mock_s3["credentials"], bucket=mock_s3["bucket"]
    )

    urls = store.write_shards(ds, prefix="test/data", maxcount=100)

    assert len(urls) >= 1
    assert all(url.startswith("s3://") for url in urls)
    assert all(mock_s3["bucket"] in url for url in urls)


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
@pytest.mark.filterwarnings("ignore:coroutine.*was never awaited:RuntimeWarning")
def test_s3_datastore_write_shards_cache_local(mock_s3, tmp_path):
    """Test writing shards with cache_local=True."""
    ds = make_simple_dataset(tmp_path, num_samples=5)

    store = atlocal.S3DataStore(
        credentials=mock_s3["credentials"], bucket=mock_s3["bucket"]
    )

    urls = store.write_shards(ds, prefix="cached/data", cache_local=True, maxcount=100)

    assert len(urls) >= 1
    assert all(url.startswith("s3://") for url in urls)


##
# Index with DataStore tests


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
@pytest.mark.filterwarnings("ignore:coroutine.*was never awaited:RuntimeWarning")
def test_index_with_datastore_insert(mock_s3, clean_redis, tmp_path):
    """Test Index.insert_dataset with a data_store writes shards and indexes."""
    ds = make_simple_dataset(tmp_path, num_samples=5)

    store = atlocal.S3DataStore(
        credentials=mock_s3["credentials"], bucket=mock_s3["bucket"]
    )
    index = atlocal.Index(redis=clean_redis, data_store=store)

    entry = index.insert_dataset(ds, name="stored-dataset", maxcount=100)

    assert entry.name == "stored-dataset"
    assert len(entry.data_urls) >= 1
    assert all(url.startswith("s3://") for url in entry.data_urls)
    assert entry.schema_ref.startswith("atdata://local/sampleSchema/")

    # Verify it's in the index
    retrieved = index.get_dataset("stored-dataset")
    assert retrieved.cid == entry.cid


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
@pytest.mark.filterwarnings("ignore:coroutine.*was never awaited:RuntimeWarning")
def test_index_with_datastore_custom_prefix(mock_s3, clean_redis, tmp_path):
    """Test Index.insert_dataset with custom prefix."""
    ds = make_simple_dataset(tmp_path, num_samples=3)

    store = atlocal.S3DataStore(
        credentials=mock_s3["credentials"], bucket=mock_s3["bucket"]
    )
    index = atlocal.Index(redis=clean_redis, data_store=store)

    entry = index.insert_dataset(
        ds, name="my-dataset", prefix="custom/path/v1", maxcount=100
    )

    assert "custom/path/v1" in entry.data_urls[0]


def test_index_without_datastore_indexes_existing_url(clean_redis, tmp_path):
    """Test Index.insert_dataset without data_store just indexes the URL."""
    ds = make_simple_dataset(tmp_path, num_samples=3)

    index = atlocal.Index(redis=clean_redis)  # No data_store

    entry = index.insert_dataset(ds, name="indexed-only")

    # Should use the original dataset URL
    assert entry.data_urls == [ds.url]
    assert entry.name == "indexed-only"


def test_index_data_store_property(mock_s3, clean_redis):
    """Test that Index.data_store property returns the data store."""
    store = atlocal.S3DataStore(
        credentials=mock_s3["credentials"], bucket=mock_s3["bucket"]
    )
    index = atlocal.Index(redis=clean_redis, data_store=store)

    assert index.data_store is store


def test_index_data_store_property_none(clean_redis):
    """Test that Index.data_store property returns None when not set."""
    index = atlocal.Index(redis=clean_redis)

    assert index.data_store is None


##
# Schema storage tests


def test_publish_schema(clean_redis):
    """Test publishing a schema to Redis."""
    index = atlocal.Index(redis=clean_redis)

    schema_ref = index.publish_schema(SimpleTestSample, version="1.0.0")

    assert schema_ref.startswith("atdata://local/sampleSchema/")
    assert "SimpleTestSample" in schema_ref
    assert "@1.0.0" in schema_ref


def test_publish_schema_with_description(clean_redis):
    """Test publishing a schema with a description."""
    index = atlocal.Index(redis=clean_redis)

    schema_ref = index.publish_schema(
        SimpleTestSample, version="2.0.0", description="A simple test sample type"
    )

    schema = index.get_schema(schema_ref)
    assert schema.get("description") == "A simple test sample type"


def test_publish_schema_auto_increment(clean_redis):
    """Test that publish_schema auto-increments version when not specified."""
    index = atlocal.Index(redis=clean_redis)

    # First publish should default to 1.0.0
    ref1 = index.publish_schema(SimpleTestSample)
    assert "@1.0.0" in ref1

    # Second publish should auto-increment to 1.0.1
    ref2 = index.publish_schema(SimpleTestSample)
    assert "@1.0.1" in ref2

    # Third publish should auto-increment to 1.0.2
    ref3 = index.publish_schema(SimpleTestSample)
    assert "@1.0.2" in ref3

    # Explicit version should override
    ref4 = index.publish_schema(SimpleTestSample, version="2.0.0")
    assert "@2.0.0" in ref4

    # Next auto-increment should be from 2.0.0
    ref5 = index.publish_schema(SimpleTestSample)
    assert "@2.0.1" in ref5


def test_publish_schema_docstring_fallback(clean_redis):
    """Test that publish_schema uses class docstring as description fallback."""
    index = atlocal.Index(redis=clean_redis)

    # SimpleTestSample has a docstring defined
    schema_ref = index.publish_schema(SimpleTestSample, version="1.0.0")
    schema = index.get_schema(schema_ref)

    # Should use the class docstring
    assert schema.get("description") == SimpleTestSample.__doc__


def test_get_schema(clean_redis):
    """Test retrieving a published schema."""
    index = atlocal.Index(redis=clean_redis)

    schema_ref = index.publish_schema(SimpleTestSample, version="1.0.0")
    schema = index.get_schema(schema_ref)

    assert schema["name"] == "SimpleTestSample"
    assert schema["version"] == "1.0.0"
    assert len(schema["fields"]) == 2  # name and value fields
    assert schema["$ref"] == schema_ref


def test_get_schema_not_found(clean_redis):
    """Test that get_schema raises KeyError for missing schema."""
    index = atlocal.Index(redis=clean_redis)

    with pytest.raises(KeyError, match="Schema not found"):
        index.get_schema("atdata://local/sampleSchema/NonexistentSample@1.0.0")


def test_get_schema_invalid_ref(clean_redis):
    """Test that get_schema raises ValueError for invalid reference."""
    index = atlocal.Index(redis=clean_redis)

    with pytest.raises(ValueError, match="Invalid schema reference"):
        index.get_schema("invalid://schemas/Sample@1.0.0")


def test_list_schemas_empty(clean_redis):
    """Test listing schemas when none exist."""
    index = atlocal.Index(redis=clean_redis)

    schemas = list(index.list_schemas())
    assert len(schemas) == 0


def test_list_schemas_multiple(clean_redis):
    """Test listing multiple schemas."""
    index = atlocal.Index(redis=clean_redis)

    index.publish_schema(SimpleTestSample, version="1.0.0")
    index.publish_schema(ArrayTestSample, version="1.0.0")

    schemas = list(index.list_schemas())
    assert len(schemas) == 2

    names = {s["name"] for s in schemas}
    assert "SimpleTestSample" in names
    assert "ArrayTestSample" in names


def test_schema_field_types(clean_redis):
    """Test that schema correctly captures field types."""
    index = atlocal.Index(redis=clean_redis)

    schema_ref = index.publish_schema(SimpleTestSample, version="1.0.0")
    schema = index.get_schema(schema_ref)

    # Find name field (should be str)
    name_field = next(f for f in schema["fields"] if f["name"] == "name")
    assert name_field["fieldType"]["primitive"] == "str"
    assert name_field["optional"] is False

    # Find value field (should be int)
    value_field = next(f for f in schema["fields"] if f["name"] == "value")
    assert value_field["fieldType"]["primitive"] == "int"


def test_schema_ndarray_field(clean_redis):
    """Test that schema correctly captures NDArray fields."""
    index = atlocal.Index(redis=clean_redis)

    schema_ref = index.publish_schema(ArrayTestSample, version="1.0.0")
    schema = index.get_schema(schema_ref)

    # Find data field (should be ndarray)
    data_field = next(f for f in schema["fields"] if f["name"] == "data")
    assert data_field["fieldType"]["$type"] == "local#ndarray"
    assert data_field["fieldType"]["dtype"] == "float32"


def test_decode_schema(clean_redis):
    """Test reconstructing a Python type from a schema."""
    index = atlocal.Index(redis=clean_redis)

    schema_ref = index.publish_schema(SimpleTestSample, version="1.0.0")
    ReconstructedType = index.decode_schema(schema_ref)

    # Should be able to create instances
    instance = ReconstructedType(name="test", value=42)
    assert instance.name == "test"
    assert instance.value == 42


def test_decode_schema_preserves_structure(clean_redis):
    """Test that decoded schema matches original type structure."""
    index = atlocal.Index(redis=clean_redis)

    schema_ref = index.publish_schema(ArrayTestSample, version="1.0.0")
    ReconstructedType = index.decode_schema(schema_ref)

    # Check fields exist
    import numpy as np

    instance = ReconstructedType(label="test", data=np.zeros((3, 3)))
    assert instance.label == "test"
    assert instance.data.shape == (3, 3)


def test_decode_schema_as_typed_helper(clean_redis):
    """Test decode_schema_as returns properly typed result."""
    index = atlocal.Index(redis=clean_redis)

    schema_ref = index.publish_schema(SimpleTestSample, version="1.0.0")

    # decode_schema_as should work like decode_schema but with type hint
    DecodedType = index.decode_schema_as(schema_ref, SimpleTestSample)

    # Should be able to create instances
    instance = DecodedType(name="test", value=42)
    assert instance.name == "test"
    assert instance.value == 42

    # The returned type should be the actual decoded type (not the hint)
    assert DecodedType.__name__ == "SimpleTestSample"


def test_schema_version_handling(clean_redis):
    """Test publishing multiple versions of the same schema."""
    index = atlocal.Index(redis=clean_redis)

    ref_v1 = index.publish_schema(SimpleTestSample, version="1.0.0")
    ref_v2 = index.publish_schema(SimpleTestSample, version="2.0.0")

    assert ref_v1 != ref_v2
    assert "@1.0.0" in ref_v1
    assert "@2.0.0" in ref_v2

    # Both should be retrievable
    schema_v1 = index.get_schema(ref_v1)
    schema_v2 = index.get_schema(ref_v2)

    assert schema_v1["version"] == "1.0.0"
    assert schema_v2["version"] == "2.0.0"


##
# Schema codec tests


def test_schema_codec_type_caching():
    """Test that schema_to_type caches generated types."""
    from atdata._schema_codec import schema_to_type, clear_type_cache, get_cached_types

    clear_type_cache()
    assert len(get_cached_types()) == 0

    schema = {
        "name": "CacheTestSample",
        "version": "1.0.0",
        "fields": [
            {
                "name": "value",
                "fieldType": {"$type": "local#primitive", "primitive": "int"},
                "optional": False,
            }
        ],
    }

    # First call creates and caches type
    Type1 = schema_to_type(schema)
    cached = get_cached_types()
    assert len(cached) == 1

    # Second call returns cached type
    Type2 = schema_to_type(schema)
    assert Type1 is Type2

    clear_type_cache()
    assert len(get_cached_types()) == 0


def test_schema_to_type_missing_name():
    """Test schema_to_type raises on schema without name."""
    from atdata._schema_codec import schema_to_type, clear_type_cache

    clear_type_cache()
    schema = {
        "version": "1.0.0",
        "fields": [
            {
                "name": "value",
                "fieldType": {"$type": "#primitive", "primitive": "int"},
                "optional": False,
            }
        ],
    }

    with pytest.raises(ValueError, match="must have a 'name' field"):
        schema_to_type(schema)


def test_schema_to_type_empty_fields():
    """Test schema_to_type raises on schema with no fields."""
    from atdata._schema_codec import schema_to_type, clear_type_cache

    clear_type_cache()
    schema = {
        "name": "EmptySample",
        "version": "1.0.0",
        "fields": [],
    }

    with pytest.raises(ValueError, match="must have at least one field"):
        schema_to_type(schema)


def test_schema_to_type_field_missing_name():
    """Test schema_to_type raises on field without name."""
    from atdata._schema_codec import schema_to_type, clear_type_cache

    clear_type_cache()
    schema = {
        "name": "BadFieldSample",
        "version": "1.0.0",
        "fields": [
            {
                "fieldType": {"$type": "#primitive", "primitive": "int"},
                "optional": False,
            }
        ],
    }

    # Raises KeyError from cache key generation (accesses f['name']) or
    # ValueError from validation - both indicate invalid schema is rejected
    with pytest.raises((KeyError, ValueError)):
        schema_to_type(schema)


def test_schema_to_type_unknown_primitive():
    """Test schema_to_type raises on unknown primitive type."""
    from atdata._schema_codec import schema_to_type, clear_type_cache

    clear_type_cache()
    schema = {
        "name": "UnknownPrimitiveSample",
        "version": "1.0.0",
        "fields": [
            {
                "name": "value",
                "fieldType": {"$type": "#primitive", "primitive": "unknown_type"},
                "optional": False,
            }
        ],
    }

    with pytest.raises(ValueError, match="Unknown primitive type"):
        schema_to_type(schema)


def test_schema_to_type_unknown_field_kind():
    """Test schema_to_type raises on unknown field type kind."""
    from atdata._schema_codec import schema_to_type, clear_type_cache

    clear_type_cache()
    schema = {
        "name": "UnknownKindSample",
        "version": "1.0.0",
        "fields": [
            {
                "name": "value",
                "fieldType": {"$type": "#unknown_kind"},
                "optional": False,
            }
        ],
    }

    with pytest.raises(ValueError, match="Unknown field type kind"):
        schema_to_type(schema)


def test_schema_to_type_ref_not_supported():
    """Test schema_to_type raises on ref field types (not yet supported)."""
    from atdata._schema_codec import schema_to_type, clear_type_cache

    clear_type_cache()
    schema = {
        "name": "RefSample",
        "version": "1.0.0",
        "fields": [
            {
                "name": "other",
                "fieldType": {"$type": "#ref", "ref": "other.Schema"},
                "optional": False,
            }
        ],
    }

    with pytest.raises(ValueError, match="Schema references.*not yet supported"):
        schema_to_type(schema)


def test_schema_to_type_all_primitives():
    """Test schema_to_type handles all primitive types correctly."""
    from atdata._schema_codec import schema_to_type, clear_type_cache

    clear_type_cache()
    schema = {
        "name": "AllPrimitivesSample",
        "version": "1.0.0",
        "fields": [
            {
                "name": "s",
                "fieldType": {"$type": "#primitive", "primitive": "str"},
                "optional": False,
            },
            {
                "name": "i",
                "fieldType": {"$type": "#primitive", "primitive": "int"},
                "optional": False,
            },
            {
                "name": "f",
                "fieldType": {"$type": "#primitive", "primitive": "float"},
                "optional": False,
            },
            {
                "name": "b",
                "fieldType": {"$type": "#primitive", "primitive": "bool"},
                "optional": False,
            },
            {
                "name": "by",
                "fieldType": {"$type": "#primitive", "primitive": "bytes"},
                "optional": False,
            },
        ],
    }

    SampleType = schema_to_type(schema)
    instance = SampleType(s="hello", i=42, f=3.14, b=True, by=b"data")

    assert instance.s == "hello"
    assert instance.i == 42
    assert instance.f == 3.14
    assert instance.b is True
    assert instance.by == b"data"


def test_schema_to_type_optional_fields():
    """Test schema_to_type handles optional fields with None defaults."""
    from atdata._schema_codec import schema_to_type, clear_type_cache

    clear_type_cache()
    schema = {
        "name": "OptionalSample",
        "version": "1.0.0",
        "fields": [
            {
                "name": "required",
                "fieldType": {"$type": "#primitive", "primitive": "str"},
                "optional": False,
            },
            {
                "name": "optional_str",
                "fieldType": {"$type": "#primitive", "primitive": "str"},
                "optional": True,
            },
        ],
    }

    SampleType = schema_to_type(schema)

    # Can create with only required field
    instance1 = SampleType(required="test")
    assert instance1.required == "test"
    assert instance1.optional_str is None

    # Can provide optional field
    instance2 = SampleType(required="test", optional_str="value")
    assert instance2.optional_str == "value"


def test_schema_to_type_ndarray_field():
    """Test schema_to_type handles NDArray fields."""
    from atdata._schema_codec import schema_to_type, clear_type_cache

    clear_type_cache()
    schema = {
        "name": "ArraySample",
        "version": "1.0.0",
        "fields": [
            {
                "name": "data",
                "fieldType": {"$type": "#ndarray", "dtype": "float32"},
                "optional": False,
            },
        ],
    }

    SampleType = schema_to_type(schema)
    arr = np.zeros((3, 3), dtype=np.float32)
    instance = SampleType(data=arr)

    assert instance.data.shape == (3, 3)


def test_schema_to_type_array_field():
    """Test schema_to_type handles array (list) fields."""
    from atdata._schema_codec import schema_to_type, clear_type_cache

    clear_type_cache()
    schema = {
        "name": "ListSample",
        "version": "1.0.0",
        "fields": [
            {
                "name": "tags",
                "fieldType": {
                    "$type": "#array",
                    "items": {"$type": "#primitive", "primitive": "str"},
                },
                "optional": False,
            },
        ],
    }

    SampleType = schema_to_type(schema)
    instance = SampleType(tags=["a", "b", "c"])

    assert instance.tags == ["a", "b", "c"]


def test_schema_to_type_use_cache_false():
    """Test schema_to_type with use_cache=False creates new types."""
    from atdata._schema_codec import schema_to_type, clear_type_cache

    clear_type_cache()
    schema = {
        "name": "NoCacheSample",
        "version": "1.0.0",
        "fields": [
            {
                "name": "value",
                "fieldType": {"$type": "#primitive", "primitive": "int"},
                "optional": False,
            }
        ],
    }

    Type1 = schema_to_type(schema, use_cache=False)
    Type2 = schema_to_type(schema, use_cache=False)

    # Different instances since caching is disabled
    assert Type1 is not Type2


##
# Auto-Stub Tests


class TestAutoStubs:
    """Tests for automatic stub file generation on schema access."""

    def test_auto_stubs_disabled_by_default(self, clean_redis):
        """Index should not generate stubs unless explicitly enabled."""
        index = atlocal.Index(redis=clean_redis)
        assert index.stub_dir is None

    def test_auto_stubs_enabled_with_flag(self, clean_redis, tmp_path):
        """Index should generate stubs when auto_stubs=True."""
        stub_dir = tmp_path / "stubs"
        index = atlocal.Index(redis=clean_redis, stub_dir=stub_dir)

        assert index.stub_dir == stub_dir

    def test_stub_generated_on_get_schema(self, clean_redis, tmp_path):
        """Stub should be generated when get_schema is called."""
        stub_dir = tmp_path / "stubs"
        index = atlocal.Index(redis=clean_redis, stub_dir=stub_dir)

        # Publish a schema
        ref = index.publish_schema(SimpleTestSample, version="1.0.0")

        # Get schema should trigger stub generation
        _schema = index.get_schema(ref)

        # Check stub was created (in local/ subdirectory for namespacing)
        stub_path = stub_dir / "local" / "SimpleTestSample_1_0_0.py"
        assert stub_path.exists()

        # Verify content
        content = stub_path.read_text()
        assert "class SimpleTestSample(PackableSample):" in content
        assert "name: str" in content

    def test_stub_generated_on_decode_schema(self, clean_redis, tmp_path):
        """Stub should be generated when decode_schema is called."""
        stub_dir = tmp_path / "stubs"
        index = atlocal.Index(redis=clean_redis, stub_dir=stub_dir)

        # Publish a schema
        ref = index.publish_schema(SimpleTestSample, version="2.0.0")

        # Decode schema should trigger stub generation
        DecodedType = index.decode_schema(ref)

        # Check stub was created (in local/ subdirectory for namespacing)
        stub_path = stub_dir / "local" / "SimpleTestSample_2_0_0.py"
        assert stub_path.exists()
        assert DecodedType.__name__ == "SimpleTestSample"

    def test_stub_not_regenerated_if_current(self, clean_redis, tmp_path):
        """Stub should not be regenerated if already current."""
        stub_dir = tmp_path / "stubs"
        index = atlocal.Index(redis=clean_redis, stub_dir=stub_dir)

        ref = index.publish_schema(SimpleTestSample, version="1.0.0")

        # First call generates stub
        index.get_schema(ref)
        stub_path = stub_dir / "local" / "SimpleTestSample_1_0_0.py"
        mtime1 = stub_path.stat().st_mtime

        # Small delay to ensure different mtime if regenerated
        import time

        time.sleep(0.01)

        # Second call should not regenerate
        index.get_schema(ref)
        mtime2 = stub_path.stat().st_mtime

        assert mtime1 == mtime2

    def test_clear_stubs(self, clean_redis, tmp_path):
        """clear_stubs should remove generated stub files."""
        stub_dir = tmp_path / "stubs"
        index = atlocal.Index(redis=clean_redis, stub_dir=stub_dir)

        # Generate some stubs
        ref1 = index.publish_schema(SimpleTestSample, version="1.0.0")
        ref2 = index.publish_schema(SimpleTestSample, version="2.0.0")
        index.get_schema(ref1)
        index.get_schema(ref2)

        # Verify stubs exist (in local/ subdirectory)
        assert (stub_dir / "local" / "SimpleTestSample_1_0_0.py").exists()
        assert (stub_dir / "local" / "SimpleTestSample_2_0_0.py").exists()

        # Clear stubs
        removed = index.clear_stubs()
        assert removed == 2

        # Verify stubs removed
        assert not (stub_dir / "local" / "SimpleTestSample_1_0_0.py").exists()
        assert not (stub_dir / "local" / "SimpleTestSample_2_0_0.py").exists()

    def test_clear_stubs_disabled_returns_zero(self, clean_redis):
        """clear_stubs should return 0 when auto_stubs is disabled."""
        index = atlocal.Index(redis=clean_redis)
        assert index.clear_stubs() == 0

    def test_stub_dir_implies_auto_stubs(self, clean_redis, tmp_path):
        """Providing stub_dir should enable auto_stubs implicitly."""
        stub_dir = tmp_path / "stubs"
        # Only provide stub_dir, not auto_stubs=True
        index = atlocal.Index(redis=clean_redis, stub_dir=stub_dir)

        ref = index.publish_schema(SimpleTestSample, version="1.0.0")
        index.get_schema(ref)

        # Stub should still be generated (in local/ subdirectory)
        stub_path = stub_dir / "local" / "SimpleTestSample_1_0_0.py"
        assert stub_path.exists()

    def test_decode_schema_returns_importable_class(self, clean_redis, tmp_path):
        """decode_schema with auto_stubs returns a class from the generated module."""
        stub_dir = tmp_path / "stubs"
        index = atlocal.Index(redis=clean_redis, auto_stubs=True, stub_dir=stub_dir)

        ref = index.publish_schema(SimpleTestSample, version="1.0.0")
        DecodedType = index.decode_schema(ref)

        # The decoded type should be usable
        assert DecodedType.__name__ == "SimpleTestSample"

        # Should be able to instantiate it (SimpleTestSample has 'name' and 'value')
        sample = DecodedType(name="hello", value=42)
        assert sample.name == "hello"
        assert sample.value == 42

        # The class should satisfy the Packable protocol
        assert isinstance(sample, atdata.Packable)

        # Verify the module was generated
        module_path = stub_dir / "local" / "SimpleTestSample_1_0_0.py"
        assert module_path.exists()

        # Verify __init__.py files were created
        assert (stub_dir / "__init__.py").exists()
        assert (stub_dir / "local" / "__init__.py").exists()

    def test_decode_schema_class_caching(self, clean_redis, tmp_path):
        """decode_schema returns the same class on subsequent calls."""
        stub_dir = tmp_path / "stubs"
        index = atlocal.Index(redis=clean_redis, auto_stubs=True, stub_dir=stub_dir)

        ref = index.publish_schema(SimpleTestSample, version="1.0.0")

        # Decode twice
        Type1 = index.decode_schema(ref)
        Type2 = index.decode_schema(ref)

        # Should return the same class (from cache)
        assert Type1 is Type2


class TestSchemaNamespace:
    """Tests for load_schema() and schemas namespace API."""

    def test_load_schema_returns_class(self, clean_redis):
        """load_schema returns the decoded class."""
        index = atlocal.Index(redis=clean_redis)
        ref = index.publish_schema(SimpleTestSample, version="1.0.0")

        cls = index.load_schema(ref)

        assert cls.__name__ == "SimpleTestSample"
        sample = cls(name="test", value=123)
        assert sample.name == "test"
        assert sample.value == 123

    def test_schemas_namespace_access(self, clean_redis):
        """After load_schema, type is accessible via schemas namespace."""
        index = atlocal.Index(redis=clean_redis)
        ref = index.publish_schema(SimpleTestSample, version="1.0.0")

        index.load_schema(ref)

        # Access via namespace
        MyType = index.types.SimpleTestSample
        assert MyType.__name__ == "SimpleTestSample"

        # Create instance
        sample = MyType(name="hello", value=42)
        assert sample.name == "hello"
        assert sample.value == 42

    def test_schemas_namespace_not_loaded_error(self, clean_redis):
        """Accessing unloaded schema raises AttributeError."""
        index = atlocal.Index(redis=clean_redis)

        with pytest.raises(AttributeError) as exc_info:
            _ = index.types.NotLoadedType

        assert "not loaded" in str(exc_info.value)
        assert "load_schema" in str(exc_info.value)

    def test_schemas_namespace_contains(self, clean_redis):
        """schemas namespace supports 'in' operator."""
        index = atlocal.Index(redis=clean_redis)
        ref = index.publish_schema(SimpleTestSample, version="1.0.0")

        assert "SimpleTestSample" not in index.types
        index.load_schema(ref)
        assert "SimpleTestSample" in index.types

    def test_schemas_namespace_iteration(self, clean_redis):
        """schemas namespace supports iteration."""
        index = atlocal.Index(redis=clean_redis)
        ref1 = index.publish_schema(SimpleTestSample, version="1.0.0")

        index.load_schema(ref1)

        names = list(index.types)
        assert "SimpleTestSample" in names

    def test_schemas_namespace_len(self, clean_redis):
        """schemas namespace supports len()."""
        index = atlocal.Index(redis=clean_redis)
        ref = index.publish_schema(SimpleTestSample, version="1.0.0")

        assert len(index.types) == 0
        index.load_schema(ref)
        assert len(index.types) == 1

    def test_schemas_namespace_repr(self, clean_redis):
        """schemas namespace has useful repr."""
        index = atlocal.Index(redis=clean_redis)
        ref = index.publish_schema(SimpleTestSample, version="1.0.0")

        assert "empty" in repr(index.types)
        index.load_schema(ref)
        assert "SimpleTestSample" in repr(index.types)

    def test_load_schema_with_auto_stubs(self, clean_redis, tmp_path):
        """load_schema works with auto_stubs enabled."""
        stub_dir = tmp_path / "stubs"
        index = atlocal.Index(redis=clean_redis, auto_stubs=True, stub_dir=stub_dir)

        ref = index.publish_schema(SimpleTestSample, version="1.0.0")
        cls = index.load_schema(ref)

        # Class works
        sample = cls(name="test", value=99)
        assert sample.name == "test"

        # Module was generated
        module_path = stub_dir / "local" / "SimpleTestSample_1_0_0.py"
        assert module_path.exists()

        # Accessible via namespace
        assert index.types.SimpleTestSample is cls

    def test_get_import_path(self, clean_redis, tmp_path):
        """get_import_path returns the module import path."""
        stub_dir = tmp_path / "stubs"
        index = atlocal.Index(redis=clean_redis, auto_stubs=True, stub_dir=stub_dir)

        ref = index.publish_schema(SimpleTestSample, version="1.0.0")
        index.load_schema(ref)

        import_path = index.get_import_path(ref)
        assert import_path == "local.SimpleTestSample_1_0_0"

    def test_get_import_path_disabled(self, clean_redis):
        """get_import_path returns None when auto_stubs is disabled."""
        index = atlocal.Index(redis=clean_redis)

        ref = index.publish_schema(SimpleTestSample, version="1.0.0")

        assert index.get_import_path(ref) is None
