"""Integration tests for local storage complete workflow.

Tests end-to-end local storage workflows including:
- Full Repo workflow: Init → publish_schema → insert → query → load
- Schema versioning and CID consistency
- Dataset discovery and querying
- Metadata persistence through full cycle
- cache_local mode comparison
"""

import pytest
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from moto import mock_aws
import webdataset as wds

import atdata
import atdata.local as atlocal


##
# Test sample types


@dataclass
class WorkflowSample(atdata.PackableSample):
    """Sample for workflow tests."""

    name: str
    value: int
    score: float


@dataclass
class ArrayWorkflowSample(atdata.PackableSample):
    """Sample with array for workflow tests."""

    label: str
    data: NDArray


@dataclass
class MetadataSample(atdata.PackableSample):
    """Sample for metadata workflow tests."""

    id: int
    content: str


##
# Fixtures


@pytest.fixture
def mock_s3():
    """Provide mock S3 environment using moto.

    Note: Tests using this fixture may generate warnings due to s3fs/moto async
    incompatibility. These are suppressed via @pytest.mark.filterwarnings on
    individual tests. See tests/EXPECTED_WARNINGS.md for details.
    """
    with mock_aws():
        import boto3

        creds = {"AWS_ACCESS_KEY_ID": "testing", "AWS_SECRET_ACCESS_KEY": "testing"}
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=creds["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=creds["AWS_SECRET_ACCESS_KEY"],
            region_name="us-east-1",
        )
        bucket_name = "integration-test-bucket"
        s3_client.create_bucket(Bucket=bucket_name)
        yield {
            "credentials": creds,
            "bucket": bucket_name,
            "hive_path": f"{bucket_name}/datasets",
            "s3_client": s3_client,
        }


def create_workflow_dataset(tmp_path: Path, n_samples: int = 10) -> atdata.Dataset:
    """Create a WorkflowSample dataset."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    tar_path = tmp_path / "workflow-000000.tar"
    with wds.writer.TarWriter(str(tar_path)) as sink:
        for i in range(n_samples):
            sample = WorkflowSample(
                name=f"item_{i}",
                value=i * 100,
                score=float(i) * 0.5,
            )
            sink.write(sample.as_wds)
    return atdata.Dataset[WorkflowSample](url=str(tar_path))


def create_array_dataset(tmp_path: Path, n_samples: int = 5) -> atdata.Dataset:
    """Create an ArrayWorkflowSample dataset."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    tar_path = tmp_path / "array-000000.tar"
    with wds.writer.TarWriter(str(tar_path)) as sink:
        for i in range(n_samples):
            sample = ArrayWorkflowSample(
                label=f"array_{i}",
                data=np.random.randn(32, 32).astype(np.float32),
            )
            sink.write(sample.as_wds)
    return atdata.Dataset[ArrayWorkflowSample](url=str(tar_path))


##
# Full Workflow Tests


class TestFullRepoWorkflow:
    """End-to-end tests for complete Repo workflow."""

    @pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
    @pytest.mark.filterwarnings("ignore:coroutine.*was never awaited:RuntimeWarning")
    @pytest.mark.filterwarnings("ignore:Repo is deprecated:DeprecationWarning")
    def test_init_publish_schema_insert_query(self, mock_s3, clean_redis, tmp_path):
        """Full workflow: init repo → publish schema → insert → query entry."""
        # Initialize repo
        repo = atlocal.Repo(
            s3_credentials=mock_s3["credentials"],
            hive_path=mock_s3["hive_path"],
            redis=clean_redis,
        )

        # Publish schema first
        schema_ref = repo.index.publish_schema(WorkflowSample)
        assert schema_ref is not None
        assert "WorkflowSample" in schema_ref

        # Create and insert dataset
        ds = create_workflow_dataset(tmp_path, n_samples=15)
        entry, new_ds = repo.insert(ds, name="workflow-test", maxcount=100)

        # Query back
        assert entry.cid is not None
        assert entry.name == "workflow-test"
        assert len(entry.data_urls) > 0

        # Verify in index
        all_entries = repo.index.all_entries
        assert len(all_entries) == 1
        assert all_entries[0].cid == entry.cid

    @pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
    @pytest.mark.filterwarnings("ignore:coroutine.*was never awaited:RuntimeWarning")
    @pytest.mark.filterwarnings("ignore:Repo is deprecated:DeprecationWarning")
    def test_multiple_datasets_same_schema(self, mock_s3, clean_redis, tmp_path):
        """Insert multiple datasets with same schema type."""
        repo = atlocal.Repo(
            s3_credentials=mock_s3["credentials"],
            hive_path=mock_s3["hive_path"],
            redis=clean_redis,
        )

        # Create multiple datasets
        ds1 = create_workflow_dataset(tmp_path / "ds1", n_samples=10)
        ds2 = create_workflow_dataset(tmp_path / "ds2", n_samples=20)
        ds3 = create_workflow_dataset(tmp_path / "ds3", n_samples=5)

        entry1, _ = repo.insert(ds1, name="dataset-1", maxcount=100)
        entry2, _ = repo.insert(ds2, name="dataset-2", maxcount=100)
        entry3, _ = repo.insert(ds3, name="dataset-3", maxcount=100)

        # All should have same schema_ref pattern
        assert "WorkflowSample" in entry1.schema_ref
        assert "WorkflowSample" in entry2.schema_ref
        assert "WorkflowSample" in entry3.schema_ref

        # But different CIDs (different URLs)
        assert entry1.cid != entry2.cid
        assert entry2.cid != entry3.cid

        # All should be in index
        all_entries = repo.index.all_entries
        assert len(all_entries) == 3

    @pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
    @pytest.mark.filterwarnings("ignore:coroutine.*was never awaited:RuntimeWarning")
    @pytest.mark.filterwarnings("ignore:Repo is deprecated:DeprecationWarning")
    def test_different_schema_types(self, mock_s3, clean_redis, tmp_path):
        """Insert datasets with different schema types."""
        repo = atlocal.Repo(
            s3_credentials=mock_s3["credentials"],
            hive_path=mock_s3["hive_path"],
            redis=clean_redis,
        )

        # Different sample types
        ds1 = create_workflow_dataset(tmp_path / "simple", n_samples=5)
        ds2 = create_array_dataset(tmp_path / "array", n_samples=3)

        entry1, _ = repo.insert(ds1, name="simple-ds", maxcount=100)
        entry2, _ = repo.insert(ds2, name="array-ds", maxcount=100)

        # Different schema refs
        assert "WorkflowSample" in entry1.schema_ref
        assert "ArrayWorkflowSample" in entry2.schema_ref

        assert len(repo.index.all_entries) == 2


class TestSchemaManagement:
    """Tests for schema publishing and retrieval."""

    def test_publish_schema_creates_record(self, clean_redis):
        """Publishing schema should create a retrievable record."""
        index = atlocal.Index(redis=clean_redis)

        schema_ref = index.publish_schema(WorkflowSample)
        assert schema_ref is not None

        # Should be able to get schema back
        schema = index.get_schema(schema_ref)
        assert schema is not None
        # Schema name may or may not include module prefix
        assert "WorkflowSample" in schema["name"]

        # Should have correct fields
        field_names = {f["name"] for f in schema["fields"]}
        assert "name" in field_names
        assert "value" in field_names
        assert "score" in field_names

    def test_publish_schema_with_version(self, clean_redis):
        """Publishing schema with version should include version."""
        index = atlocal.Index(redis=clean_redis)

        schema_ref = index.publish_schema(WorkflowSample, version="2.0.0")
        assert "2.0.0" in schema_ref

        schema = index.get_schema(schema_ref)
        assert schema["version"] == "2.0.0"

    def test_publish_schema_with_ndarray(self, clean_redis):
        """Schema with NDArray field should publish correctly."""
        index = atlocal.Index(redis=clean_redis)

        schema_ref = index.publish_schema(ArrayWorkflowSample)
        schema = index.get_schema(schema_ref)

        # Find the data field
        data_field = next(f for f in schema["fields"] if f["name"] == "data")
        assert data_field["fieldType"]["$type"] == "local#ndarray"

    def test_list_schemas(self, clean_redis):
        """Should list all published schemas."""
        index = atlocal.Index(redis=clean_redis)

        # Publish multiple schemas
        index.publish_schema(WorkflowSample, version="1.0.0")
        index.publish_schema(ArrayWorkflowSample, version="1.0.0")

        schemas = list(index.list_schemas())
        assert len(schemas) >= 2

    def test_decode_schema_creates_type(self, clean_redis):
        """decode_schema should reconstruct a usable type."""
        index = atlocal.Index(redis=clean_redis)

        schema_ref = index.publish_schema(WorkflowSample)
        reconstructed = index.decode_schema(schema_ref)

        assert reconstructed is not None
        # Should be able to create instances
        instance = reconstructed(name="test", value=42, score=0.5)
        assert instance.name == "test"
        assert instance.value == 42


class TestCIDDeterminism:
    """Tests for CID generation consistency."""

    def test_same_content_same_cid(self):
        """Identical content should produce identical CIDs."""
        entry1 = atlocal.LocalDatasetEntry(
            name="test",
            schema_ref="local://schemas/Test@1.0.0",
            data_urls=["s3://bucket/data.tar"],
            metadata={"key": "value"},
        )
        entry2 = atlocal.LocalDatasetEntry(
            name="test",
            schema_ref="local://schemas/Test@1.0.0",
            data_urls=["s3://bucket/data.tar"],
            metadata={"key": "value"},
        )

        assert entry1.cid == entry2.cid

    def test_different_urls_different_cid(self):
        """Different data URLs should produce different CIDs."""
        entry1 = atlocal.LocalDatasetEntry(
            name="test",
            schema_ref="local://schemas/Test@1.0.0",
            data_urls=["s3://bucket/data-v1.tar"],
        )
        entry2 = atlocal.LocalDatasetEntry(
            name="test",
            schema_ref="local://schemas/Test@1.0.0",
            data_urls=["s3://bucket/data-v2.tar"],
        )

        assert entry1.cid != entry2.cid

    def test_different_schema_different_cid(self):
        """Different schema refs should produce different CIDs."""
        entry1 = atlocal.LocalDatasetEntry(
            name="test",
            schema_ref="local://schemas/TypeA@1.0.0",
            data_urls=["s3://bucket/data.tar"],
        )
        entry2 = atlocal.LocalDatasetEntry(
            name="test",
            schema_ref="local://schemas/TypeB@1.0.0",
            data_urls=["s3://bucket/data.tar"],
        )

        assert entry1.cid != entry2.cid

    def test_name_does_not_affect_cid(self):
        """Dataset name should not affect CID (only content matters)."""
        entry1 = atlocal.LocalDatasetEntry(
            name="name-one",
            schema_ref="local://schemas/Test@1.0.0",
            data_urls=["s3://bucket/data.tar"],
        )
        entry2 = atlocal.LocalDatasetEntry(
            name="name-two",
            schema_ref="local://schemas/Test@1.0.0",
            data_urls=["s3://bucket/data.tar"],
        )

        # CID based on schema_ref and data_urls, not name
        assert entry1.cid == entry2.cid

    def test_cid_format_is_valid(self):
        """CIDs should have valid ATProto-compatible format."""
        entry = atlocal.LocalDatasetEntry(
            name="test",
            schema_ref="local://schemas/Test@1.0.0",
            data_urls=["s3://bucket/data.tar"],
        )

        # CIDv1 with dag-cbor starts with 'bafy'
        assert entry.cid.startswith("bafy")
        # Should be base32 encoded (alphanumeric lowercase)
        assert entry.cid.isalnum()
        assert entry.cid.islower()


class TestDatasetDiscovery:
    """Tests for querying and discovering datasets."""

    def test_get_entry_by_name(self, clean_redis):
        """Should retrieve entry by name."""
        index = atlocal.Index(redis=clean_redis)

        # Add entries
        entry1 = atlocal.LocalDatasetEntry(
            name="findme",
            schema_ref="local://schemas/Test@1.0.0",
            data_urls=["s3://bucket/findme.tar"],
        )
        entry1.write_to(clean_redis)

        # Query by name
        found = index.get_entry_by_name("findme")
        assert found is not None
        assert found.name == "findme"

    def test_get_entry_by_cid(self, clean_redis):
        """Should retrieve entry by CID."""
        index = atlocal.Index(redis=clean_redis)

        entry = atlocal.LocalDatasetEntry(
            name="bycid",
            schema_ref="local://schemas/Test@1.0.0",
            data_urls=["s3://bucket/bycid.tar"],
        )
        entry.write_to(clean_redis)

        # Query by CID
        found = index.get_entry(cid=entry.cid)
        assert found is not None
        assert found.cid == entry.cid
        assert found.name == "bycid"

    def test_list_all_datasets(self, clean_redis):
        """Should list all datasets in index."""
        index = atlocal.Index(redis=clean_redis)

        # Add multiple entries
        for i in range(5):
            entry = atlocal.LocalDatasetEntry(
                name=f"dataset-{i}",
                schema_ref="local://schemas/Test@1.0.0",
                data_urls=[f"s3://bucket/dataset-{i}.tar"],
            )
            entry.write_to(clean_redis)

        # List all
        all_entries = list(index.entries)
        assert len(all_entries) == 5

        names = {e.name for e in all_entries}
        for i in range(5):
            assert f"dataset-{i}" in names

    def test_entries_generator_is_lazy(self, clean_redis):
        """entries property should be a generator, not load all at once."""
        index = atlocal.Index(redis=clean_redis)

        # Add entries
        for i in range(10):
            entry = atlocal.LocalDatasetEntry(
                name=f"lazy-{i}",
                schema_ref="local://schemas/Test@1.0.0",
                data_urls=[f"s3://bucket/lazy-{i}.tar"],
            )
            entry.write_to(clean_redis)

        # Should be a generator
        entries = index.entries
        import types

        assert isinstance(entries, types.GeneratorType)

        # Can iterate partially
        first_three = []
        for i, entry in enumerate(entries):
            first_three.append(entry)
            if i >= 2:
                break
        assert len(first_three) == 3


class TestMetadataPersistence:
    """Tests for metadata preservation through storage cycle."""

    @pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
    @pytest.mark.filterwarnings("ignore:coroutine.*was never awaited:RuntimeWarning")
    @pytest.mark.filterwarnings("ignore:Repo is deprecated:DeprecationWarning")
    def test_metadata_preserved_through_insert(self, mock_s3, clean_redis, tmp_path):
        """Metadata should be preserved when inserting dataset."""
        repo = atlocal.Repo(
            s3_credentials=mock_s3["credentials"],
            hive_path=mock_s3["hive_path"],
            redis=clean_redis,
        )

        ds = create_workflow_dataset(tmp_path, n_samples=5)
        ds._metadata = {
            "version": "1.0.0",
            "author": "test",
            "created": "2024-01-01",
            "nested": {"key": "value", "count": 42},
        }

        entry, new_ds = repo.insert(ds, name="with-metadata", maxcount=100)

        # Metadata should be in entry
        assert entry.metadata is not None
        assert entry.metadata["version"] == "1.0.0"
        assert entry.metadata["author"] == "test"
        assert entry.metadata["nested"]["key"] == "value"
        assert entry.metadata["nested"]["count"] == 42

    def test_metadata_round_trip_redis(self, clean_redis):
        """Metadata should round-trip through Redis correctly."""
        original = atlocal.LocalDatasetEntry(
            name="meta-test",
            schema_ref="local://schemas/Test@1.0.0",
            data_urls=["s3://bucket/data.tar"],
            metadata={
                "string": "hello",
                "number": 123,
                "float": 3.14,
                "bool": True,
                "list": [1, 2, 3],
                "nested": {"a": 1, "b": 2},
            },
        )

        original.write_to(clean_redis)
        loaded = atlocal.LocalDatasetEntry.from_redis(clean_redis, original.cid)

        assert loaded.metadata == original.metadata
        assert loaded.metadata["string"] == "hello"
        assert loaded.metadata["number"] == 123
        assert loaded.metadata["list"] == [1, 2, 3]
        assert loaded.metadata["nested"]["a"] == 1

    def test_none_metadata_handled(self, clean_redis):
        """None metadata should be handled gracefully."""
        entry = atlocal.LocalDatasetEntry(
            name="no-meta",
            schema_ref="local://schemas/Test@1.0.0",
            data_urls=["s3://bucket/data.tar"],
            metadata=None,
        )

        entry.write_to(clean_redis)
        loaded = atlocal.LocalDatasetEntry.from_redis(clean_redis, entry.cid)

        # Should be None or empty, not error
        assert loaded.metadata is None or loaded.metadata == {}


class TestCacheLocalModes:
    """Tests comparing cache_local=True vs cache_local=False modes."""

    @pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
    @pytest.mark.filterwarnings("ignore:coroutine.*was never awaited:RuntimeWarning")
    @pytest.mark.filterwarnings("ignore:Repo is deprecated:DeprecationWarning")
    def test_cache_local_true_produces_valid_entry(
        self, mock_s3, clean_redis, tmp_path
    ):
        """cache_local=True should produce valid index entry."""
        repo = atlocal.Repo(
            s3_credentials=mock_s3["credentials"],
            hive_path=mock_s3["hive_path"],
            redis=clean_redis,
        )

        ds = create_workflow_dataset(tmp_path, n_samples=10)
        entry, new_ds = repo.insert(ds, name="cached", cache_local=True, maxcount=100)

        assert entry.cid is not None
        assert len(entry.data_urls) > 0
        assert ".tar" in entry.data_urls[0]

    @pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
    @pytest.mark.filterwarnings("ignore:coroutine.*was never awaited:RuntimeWarning")
    @pytest.mark.filterwarnings("ignore:Repo is deprecated:DeprecationWarning")
    def test_cache_local_false_produces_valid_entry(
        self, mock_s3, clean_redis, tmp_path
    ):
        """cache_local=False should produce valid index entry."""
        repo = atlocal.Repo(
            s3_credentials=mock_s3["credentials"],
            hive_path=mock_s3["hive_path"],
            redis=clean_redis,
        )

        ds = create_workflow_dataset(tmp_path, n_samples=10)
        entry, new_ds = repo.insert(ds, name="direct", cache_local=False, maxcount=100)

        assert entry.cid is not None
        assert len(entry.data_urls) > 0
        assert ".tar" in entry.data_urls[0]

    @pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
    @pytest.mark.filterwarnings("ignore:coroutine.*was never awaited:RuntimeWarning")
    @pytest.mark.filterwarnings("ignore:Repo is deprecated:DeprecationWarning")
    def test_both_modes_produce_same_structure(self, mock_s3, clean_redis, tmp_path):
        """Both cache modes should produce entries with same structure."""
        repo = atlocal.Repo(
            s3_credentials=mock_s3["credentials"],
            hive_path=mock_s3["hive_path"],
            redis=clean_redis,
        )

        ds1 = create_workflow_dataset(tmp_path / "cached", n_samples=10)
        ds2 = create_workflow_dataset(tmp_path / "direct", n_samples=10)

        entry1, _ = repo.insert(ds1, name="cached-mode", cache_local=True, maxcount=100)
        entry2, _ = repo.insert(
            ds2, name="direct-mode", cache_local=False, maxcount=100
        )

        # Both should have valid structure
        assert entry1.schema_ref == entry2.schema_ref  # Same type
        assert len(entry1.data_urls) == len(entry2.data_urls)  # Same shard count


class TestIndexEntryProtocol:
    """Tests for IndexEntry protocol compliance."""

    def test_local_entry_implements_protocol(self):
        """LocalDatasetEntry should implement IndexEntry protocol."""
        from atdata._protocols import IndexEntry

        entry = atlocal.LocalDatasetEntry(
            name="protocol-test",
            schema_ref="local://schemas/Test@1.0.0",
            data_urls=["s3://bucket/data.tar"],
        )

        assert isinstance(entry, IndexEntry)

    def test_entry_has_required_properties(self):
        """Entry should have all required IndexEntry properties."""
        entry = atlocal.LocalDatasetEntry(
            name="props-test",
            schema_ref="local://schemas/Test@1.0.0",
            data_urls=["s3://bucket/data.tar"],
            metadata={"key": "value"},
        )

        # Required properties
        assert hasattr(entry, "name")
        assert hasattr(entry, "schema_ref")
        assert hasattr(entry, "data_urls")
        assert hasattr(entry, "metadata")
        assert hasattr(entry, "cid")

        # Values accessible
        assert entry.name == "props-test"
        assert entry.schema_ref == "local://schemas/Test@1.0.0"
        assert entry.data_urls == ["s3://bucket/data.tar"]
        assert entry.metadata == {"key": "value"}

    def test_legacy_properties_work(self):
        """Legacy properties should still work for backwards compatibility."""
        entry = atlocal.LocalDatasetEntry(
            name="legacy-test",
            schema_ref="local://schemas/Test@1.0.0",
            data_urls=["s3://bucket/legacy.tar"],
        )

        # Legacy aliases
        assert entry.wds_url == "s3://bucket/legacy.tar"
        assert entry.sample_kind == "local://schemas/Test@1.0.0"


class TestMultiShardStorage:
    """Tests for multi-shard dataset storage."""

    @pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
    @pytest.mark.filterwarnings("ignore:coroutine.*was never awaited:RuntimeWarning")
    @pytest.mark.filterwarnings("ignore:Repo is deprecated:DeprecationWarning")
    def test_large_dataset_creates_multiple_shards(
        self, mock_s3, clean_redis, tmp_path
    ):
        """Large dataset should create multiple shard files."""
        repo = atlocal.Repo(
            s3_credentials=mock_s3["credentials"],
            hive_path=mock_s3["hive_path"],
            redis=clean_redis,
        )

        # Create dataset with many samples
        tar_path = tmp_path / "large-000000.tar"
        with wds.writer.TarWriter(str(tar_path)) as sink:
            for i in range(100):
                sample = WorkflowSample(
                    name=f"item_{i}",
                    value=i,
                    score=float(i),
                )
                sink.write(sample.as_wds)

        ds = atdata.Dataset[WorkflowSample](url=str(tar_path))

        # Insert with small maxcount to force sharding
        entry, new_ds = repo.insert(ds, name="sharded", maxcount=10)

        # Should have multiple shards (URL with brace notation)
        assert "{" in new_ds.url and "}" in new_ds.url

    @pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
    @pytest.mark.filterwarnings("ignore:coroutine.*was never awaited:RuntimeWarning")
    @pytest.mark.filterwarnings("ignore:Repo is deprecated:DeprecationWarning")
    def test_single_shard_no_brace_notation(self, mock_s3, clean_redis, tmp_path):
        """Small dataset should result in single shard without brace notation."""
        repo = atlocal.Repo(
            s3_credentials=mock_s3["credentials"],
            hive_path=mock_s3["hive_path"],
            redis=clean_redis,
        )

        ds = create_workflow_dataset(tmp_path, n_samples=5)

        # Large maxcount ensures single shard
        entry, new_ds = repo.insert(ds, name="single", maxcount=1000)

        # Should be single file, no brace notation
        assert "{" not in new_ds.url
        assert ".tar" in new_ds.url
