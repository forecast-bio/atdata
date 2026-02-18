"""Tests for the HuggingFace Datasets-style API (_hf_api.py)."""

##
# Imports

import pytest

import numpy as np
import webdataset as wds

import atdata
from atdata._hf_api import (
    load_dataset,
    DatasetDict,
    _is_brace_pattern,
    _is_glob_pattern,
    _is_remote_url,
    _detect_split_from_path,
    _shards_to_wds_url,
    _expand_local_glob,
    _resolve_shards,
    _resolve_data_files,
    _group_shards_by_split,
    _is_indexed_path,
    _is_at_uri,
    _resolve_at_uri,
    _parse_indexed_path,
)
from unittest.mock import Mock, patch, MagicMock

from numpy.typing import NDArray


##
# Test sample types


@atdata.packable
class SimpleTestSample:
    """Simple sample type for testing."""

    text: str
    label: int


@atdata.packable
class NumpyTestSample:
    """Sample type with numpy arrays for testing."""

    embedding: NDArray
    label: int


##
# Helper function tests


class TestIsBracePattern:
    """Tests for _is_brace_pattern()."""

    def test_range_pattern(self):
        assert _is_brace_pattern("data-{000000..000099}.tar") is True

    def test_list_pattern(self):
        assert _is_brace_pattern("data-{train,test,val}.tar") is True

    def test_no_pattern(self):
        assert _is_brace_pattern("data-000000.tar") is False

    def test_empty_braces(self):
        # Empty braces are not valid WebDataset brace notation
        assert _is_brace_pattern("data-{}.tar") is False

    def test_nested_path_with_pattern(self):
        assert _is_brace_pattern("path/to/data-{000..099}.tar") is True


class TestIsGlobPattern:
    """Tests for _is_glob_pattern()."""

    def test_asterisk(self):
        assert _is_glob_pattern("data-*.tar") is True

    def test_question_mark(self):
        assert _is_glob_pattern("data-00000?.tar") is True

    def test_no_pattern(self):
        assert _is_glob_pattern("data-000000.tar") is False

    def test_path_with_glob(self):
        assert _is_glob_pattern("path/to/*.tar") is True


class TestIsRemoteUrl:
    """Tests for _is_remote_url()."""

    def test_s3_url(self):
        assert _is_remote_url("s3://bucket/path/data.tar") is True

    def test_https_url(self):
        assert _is_remote_url("https://example.com/data.tar") is True

    def test_http_url(self):
        assert _is_remote_url("http://example.com/data.tar") is True

    def test_gs_url(self):
        assert _is_remote_url("gs://bucket/path/data.tar") is True

    def test_az_url(self):
        assert _is_remote_url("az://container/path/data.tar") is True

    def test_local_absolute_path(self):
        assert _is_remote_url("/local/path/data.tar") is False

    def test_local_relative_path(self):
        assert _is_remote_url("./data/data.tar") is False

    def test_windows_path(self):
        assert _is_remote_url("C:\\data\\data.tar") is False


class TestDetectSplitFromPath:
    """Tests for _detect_split_from_path()."""

    def test_train_in_filename(self):
        assert _detect_split_from_path("dataset-train-000000.tar") == "train"

    def test_test_in_filename(self):
        assert _detect_split_from_path("dataset-test-000000.tar") == "test"

    def test_validation_in_filename(self):
        assert _detect_split_from_path("dataset-validation-000000.tar") == "validation"

    def test_val_in_filename(self):
        assert _detect_split_from_path("dataset-val-000000.tar") == "validation"

    def test_dev_in_filename(self):
        assert _detect_split_from_path("dataset-dev-000000.tar") == "validation"

    def test_train_directory(self):
        assert _detect_split_from_path("train/shard-000000.tar") == "train"

    def test_test_directory(self):
        assert _detect_split_from_path("test/shard-000000.tar") == "test"

    def test_no_split_detected(self):
        assert _detect_split_from_path("dataset-000000.tar") is None

    def test_case_insensitive(self):
        assert _detect_split_from_path("dataset-TRAIN-000000.tar") == "train"
        assert _detect_split_from_path("dataset-Train-000000.tar") == "train"

    def test_training_variant(self):
        assert _detect_split_from_path("dataset-training-000000.tar") == "train"

    def test_testing_variant(self):
        assert _detect_split_from_path("dataset-testing-000000.tar") == "test"


class TestShardsToWdsUrl:
    """Tests for _shards_to_wds_url()."""

    def test_single_shard(self):
        assert _shards_to_wds_url(["data.tar"]) == "data.tar"

    def test_multiple_shards_common_pattern(self):
        shards = ["data-000.tar", "data-001.tar", "data-002.tar"]
        result = _shards_to_wds_url(shards)
        assert result == "data-00{0,1,2}.tar"

    def test_multiple_shards_different_lengths(self):
        shards = ["data-0.tar", "data-1.tar", "data-10.tar"]
        result = _shards_to_wds_url(shards)
        assert result == "data-{0,1,10}.tar"

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="empty shard list"):
            _shards_to_wds_url([])

    def test_no_common_pattern(self):
        shards = ["train.tar", "test.tar", "val.tar"]
        result = _shards_to_wds_url(shards)
        assert result == "{train,test,val}.tar"


class TestExpandLocalGlob:
    """Tests for _expand_local_glob()."""

    def test_no_matches(self, tmp_path):
        pattern = str(tmp_path / "*.tar")
        assert _expand_local_glob(pattern) == []

    def test_matches_files(self, tmp_path):
        # Create test files
        (tmp_path / "data-000.tar").touch()
        (tmp_path / "data-001.tar").touch()
        (tmp_path / "data-002.tar").touch()

        pattern = str(tmp_path / "*.tar")
        result = _expand_local_glob(pattern)

        assert len(result) == 3
        assert all(".tar" in p for p in result)

    def test_ignores_directories(self, tmp_path):
        # Create a file and a directory
        (tmp_path / "data.tar").touch()
        (tmp_path / "subdir.tar").mkdir()

        pattern = str(tmp_path / "*.tar")
        result = _expand_local_glob(pattern)

        assert len(result) == 1

    def test_nonexistent_directory(self):
        result = _expand_local_glob("/nonexistent/path/*.tar")
        assert result == []


class TestGroupShardsBySplit:
    """Tests for _group_shards_by_split()."""

    def test_single_split(self):
        shards = [
            "train-000.tar",
            "train-001.tar",
            "train-002.tar",
        ]
        result = _group_shards_by_split(shards)
        assert "train" in result
        assert len(result["train"]) == 3

    def test_multiple_splits(self):
        shards = [
            "data-train-000.tar",
            "data-train-001.tar",
            "data-test-000.tar",
            "data-val-000.tar",
        ]
        result = _group_shards_by_split(shards)
        assert "train" in result
        assert "test" in result
        assert "validation" in result
        assert len(result["train"]) == 2
        assert len(result["test"]) == 1
        assert len(result["validation"]) == 1

    def test_no_detected_split_defaults_to_train(self):
        shards = ["shard-000.tar", "shard-001.tar"]
        result = _group_shards_by_split(shards)
        assert "train" in result
        assert len(result["train"]) == 2


class TestResolveDataFiles:
    """Tests for _resolve_data_files()."""

    def test_string_input(self, tmp_path):
        result = _resolve_data_files(str(tmp_path), "data.tar")
        assert "train" in result
        assert len(result["train"]) == 1

    def test_list_input(self, tmp_path):
        result = _resolve_data_files(str(tmp_path), ["a.tar", "b.tar"])
        assert "train" in result
        assert len(result["train"]) == 2

    def test_dict_input(self, tmp_path):
        data_files = {
            "train": ["train-000.tar", "train-001.tar"],
            "test": "test-000.tar",
        }
        result = _resolve_data_files(str(tmp_path), data_files)
        assert "train" in result
        assert "test" in result
        assert len(result["train"]) == 2
        assert len(result["test"]) == 1

    def test_resolves_relative_paths(self, tmp_path):
        result = _resolve_data_files(str(tmp_path), "subdir/data.tar")
        assert str(tmp_path) in result["train"][0]


class TestResolveShards:
    """Tests for _resolve_shards()."""

    def test_brace_pattern_passthrough(self):
        path = "data-{000000..000099}.tar"
        result = _resolve_shards(path)
        assert "train" in result
        assert path in result["train"]

    def test_brace_pattern_with_split_name(self):
        path = "data-train-{000..099}.tar"
        result = _resolve_shards(path)
        assert "train" in result

    def test_single_file(self):
        path = "data.tar"
        result = _resolve_shards(path)
        assert "train" in result
        assert result["train"] == [path]

    def test_with_data_files_override(self, tmp_path):
        data_files = {"train": "train.tar", "test": "test.tar"}
        result = _resolve_shards(str(tmp_path), data_files)
        assert "train" in result
        assert "test" in result

    def test_local_directory(self, tmp_path):
        # Create test tar files
        (tmp_path / "train-000.tar").touch()
        (tmp_path / "train-001.tar").touch()
        (tmp_path / "test-000.tar").touch()

        result = _resolve_shards(str(tmp_path))
        assert "train" in result
        assert "test" in result

    def test_glob_pattern(self, tmp_path):
        # Create test files
        (tmp_path / "data-000.tar").touch()
        (tmp_path / "data-001.tar").touch()

        pattern = str(tmp_path / "*.tar")
        result = _resolve_shards(pattern)
        assert "train" in result  # defaults to train when no split detected


##
# DatasetDict tests


class TestDatasetDict:
    """Tests for DatasetDict class."""

    def test_empty_init(self):
        dd = DatasetDict()
        assert len(dd) == 0

    def test_init_with_splits(self, tmp_path):
        # Create a minimal tar file for Dataset
        tar_path = tmp_path / "data.tar"
        with wds.writer.TarWriter(str(tar_path)) as sink:
            sample = SimpleTestSample(text="hello", label=1)
            sink.write(sample.as_wds)

        train_ds = atdata.Dataset[SimpleTestSample](str(tar_path))
        test_ds = atdata.Dataset[SimpleTestSample](str(tar_path))

        dd = DatasetDict({"train": train_ds, "test": test_ds})

        assert len(dd) == 2
        assert "train" in dd
        assert "test" in dd

    def test_getitem(self, tmp_path):
        tar_path = tmp_path / "data.tar"
        with wds.writer.TarWriter(str(tar_path)) as sink:
            sample = SimpleTestSample(text="hello", label=1)
            sink.write(sample.as_wds)

        train_ds = atdata.Dataset[SimpleTestSample](str(tar_path))
        dd = DatasetDict({"train": train_ds})

        assert dd["train"] is train_ds

    def test_setitem(self, tmp_path):
        tar_path = tmp_path / "data.tar"
        with wds.writer.TarWriter(str(tar_path)) as sink:
            sample = SimpleTestSample(text="hello", label=1)
            sink.write(sample.as_wds)

        dd = DatasetDict()
        train_ds = atdata.Dataset[SimpleTestSample](str(tar_path))
        dd["train"] = train_ds

        assert "train" in dd
        assert dd["train"] is train_ds

    def test_keys_values_items(self, tmp_path):
        tar_path = tmp_path / "data.tar"
        with wds.writer.TarWriter(str(tar_path)) as sink:
            sample = SimpleTestSample(text="hello", label=1)
            sink.write(sample.as_wds)

        train_ds = atdata.Dataset[SimpleTestSample](str(tar_path))
        test_ds = atdata.Dataset[SimpleTestSample](str(tar_path))

        dd = DatasetDict({"train": train_ds, "test": test_ds})

        assert set(dd.keys()) == {"train", "test"}
        assert len(list(dd.values())) == 2
        assert len(list(dd.items())) == 2

    def test_streaming_property(self):
        dd = DatasetDict(streaming=True)
        assert dd.streaming is True

        dd2 = DatasetDict(streaming=False)
        assert dd2.streaming is False

    def test_sample_type_explicit(self):
        dd = DatasetDict(sample_type=SimpleTestSample)
        assert dd.sample_type is SimpleTestSample

    def test_num_shards(self, tmp_path):
        # Create two tar files for train split
        train_path = tmp_path / "train.tar"
        with wds.writer.TarWriter(str(train_path)) as sink:
            sample = SimpleTestSample(text="hello", label=1)
            sink.write(sample.as_wds)

        train_ds = atdata.Dataset[SimpleTestSample](str(train_path))
        dd = DatasetDict({"train": train_ds})

        num_shards = dd.num_shards
        assert "train" in num_shards
        assert num_shards["train"] == 1

    def test_single_split_proxy_ordered(self, tmp_path):
        """Single-split DatasetDict should proxy .ordered() to the Dataset."""
        tar_path = tmp_path / "data.tar"
        with wds.writer.TarWriter(str(tar_path)) as sink:
            for i in range(5):
                sample = SimpleTestSample(text=f"s{i}", label=i)
                sink.write(sample.as_wds)

        dd = DatasetDict({"train": atdata.Dataset[SimpleTestSample](str(tar_path))})
        results = list(dd.ordered(batch_size=None))
        assert len(results) == 5

    def test_single_split_proxy_list_shards(self, tmp_path):
        """Single-split DatasetDict should proxy .list_shards()."""
        tar_path = tmp_path / "data.tar"
        with wds.writer.TarWriter(str(tar_path)) as sink:
            sink.write(SimpleTestSample(text="x", label=0).as_wds)

        dd = DatasetDict({"train": atdata.Dataset[SimpleTestSample](str(tar_path))})
        shards = dd.list_shards()
        assert len(shards) == 1

    def test_multi_split_no_proxy(self, tmp_path):
        """Multi-split DatasetDict should NOT proxy Dataset methods."""
        tar_path = tmp_path / "data.tar"
        with wds.writer.TarWriter(str(tar_path)) as sink:
            sink.write(SimpleTestSample(text="x", label=0).as_wds)

        ds = atdata.Dataset[SimpleTestSample](str(tar_path))
        dd = DatasetDict({"train": ds, "test": ds})

        with pytest.raises(AttributeError, match="2 splits"):
            dd.ordered()

    def test_unknown_attr_raises(self):
        """Non-Dataset attributes should raise normal AttributeError."""
        dd = DatasetDict()
        with pytest.raises(AttributeError, match="has no attribute"):
            dd.nonexistent_method()


##
# load_dataset tests


class TestLoadDataset:
    """Tests for load_dataset() function."""

    def test_load_single_file_with_split(self, tmp_path):
        """Load a single tar file specifying a split."""
        tar_path = tmp_path / "data.tar"
        with wds.writer.TarWriter(str(tar_path)) as sink:
            for i in range(10):
                sample = SimpleTestSample(text=f"sample_{i}", label=i)
                sink.write(sample.as_wds)

        ds = load_dataset(str(tar_path), SimpleTestSample, split="train")

        assert isinstance(ds, atdata.Dataset)
        # Verify we can iterate
        samples = list(ds.ordered(batch_size=None))
        assert len(samples) == 10

    def test_load_returns_dataset_dict_without_split(self, tmp_path):
        """Without split parameter, returns DatasetDict."""
        tar_path = tmp_path / "data.tar"
        with wds.writer.TarWriter(str(tar_path)) as sink:
            sample = SimpleTestSample(text="hello", label=1)
            sink.write(sample.as_wds)

        result = load_dataset(str(tar_path), SimpleTestSample)

        assert isinstance(result, DatasetDict)
        assert "train" in result

    def test_load_with_data_files_dict(self, tmp_path):
        """Load with explicit data_files mapping."""
        # Create train and test files
        train_path = tmp_path / "train.tar"
        test_path = tmp_path / "test.tar"

        with wds.writer.TarWriter(str(train_path)) as sink:
            for i in range(5):
                sample = SimpleTestSample(text=f"train_{i}", label=i)
                sink.write(sample.as_wds)

        with wds.writer.TarWriter(str(test_path)) as sink:
            for i in range(3):
                sample = SimpleTestSample(text=f"test_{i}", label=i)
                sink.write(sample.as_wds)

        result = load_dataset(
            str(tmp_path),
            SimpleTestSample,
            data_files={"train": "train.tar", "test": "test.tar"},
        )

        assert isinstance(result, DatasetDict)
        assert "train" in result
        assert "test" in result

    def test_load_nonexistent_split_raises(self, tmp_path):
        """Requesting a split that doesn't exist raises ValueError."""
        tar_path = tmp_path / "train.tar"
        with wds.writer.TarWriter(str(tar_path)) as sink:
            sample = SimpleTestSample(text="hello", label=1)
            sink.write(sample.as_wds)

        with pytest.raises(ValueError, match="Split 'test' not found"):
            load_dataset(str(tar_path), SimpleTestSample, split="test")

    def test_load_directory_with_split_detection(self, tmp_path):
        """Load from directory auto-detecting splits from filenames."""
        # Create files with split names
        train_path = tmp_path / "data-train-000.tar"
        test_path = tmp_path / "data-test-000.tar"

        with wds.writer.TarWriter(str(train_path)) as sink:
            for i in range(5):
                sample = SimpleTestSample(text=f"train_{i}", label=i)
                sink.write(sample.as_wds)

        with wds.writer.TarWriter(str(test_path)) as sink:
            for i in range(3):
                sample = SimpleTestSample(text=f"test_{i}", label=i)
                sink.write(sample.as_wds)

        result = load_dataset(str(tmp_path), SimpleTestSample)

        assert isinstance(result, DatasetDict)
        assert "train" in result
        assert "test" in result

    def test_load_with_streaming_flag(self, tmp_path):
        """streaming=True sets the streaming property."""
        tar_path = tmp_path / "data.tar"
        with wds.writer.TarWriter(str(tar_path)) as sink:
            sample = SimpleTestSample(text="hello", label=1)
            sink.write(sample.as_wds)

        result = load_dataset(str(tar_path), SimpleTestSample, streaming=True)

        assert isinstance(result, DatasetDict)
        assert result.streaming is True

    def test_load_with_numpy_sample_type(self, tmp_path):
        """Load dataset with numpy arrays in samples."""
        tar_path = tmp_path / "data.tar"
        with wds.writer.TarWriter(str(tar_path)) as sink:
            for i in range(5):
                sample = NumpyTestSample(
                    embedding=np.random.randn(128).astype(np.float32), label=i
                )
                sink.write(sample.as_wds)

        ds = load_dataset(str(tar_path), NumpyTestSample, split="train")
        samples = list(ds.ordered(batch_size=None))

        assert len(samples) == 5
        assert isinstance(samples[0].embedding, np.ndarray)
        assert samples[0].embedding.shape == (128,)

    def test_load_glob_pattern(self, tmp_path):
        """Load using glob pattern."""
        # Create multiple shard files
        for i in range(3):
            shard_path = tmp_path / f"data-{i:03d}.tar"
            with wds.writer.TarWriter(str(shard_path)) as sink:
                sample = SimpleTestSample(text=f"shard_{i}", label=i)
                sink.write(sample.as_wds)

        pattern = str(tmp_path / "*.tar")
        result = load_dataset(pattern, SimpleTestSample)

        assert isinstance(result, DatasetDict)
        assert "train" in result

    def test_load_brace_notation(self, tmp_path):
        """Load using WebDataset brace notation."""
        # Create sharded files
        for i in range(3):
            shard_path = tmp_path / f"data-{i:06d}.tar"
            with wds.writer.TarWriter(str(shard_path)) as sink:
                for j in range(2):
                    sample = SimpleTestSample(text=f"shard_{i}_sample_{j}", label=j)
                    sink.write(sample.as_wds)

        # Use brace notation
        pattern = str(tmp_path / "data-{000000..000002}.tar")
        ds = load_dataset(pattern, SimpleTestSample, split="train")

        assert isinstance(ds, atdata.Dataset)
        samples = list(ds.ordered(batch_size=None))
        assert len(samples) == 6  # 3 shards * 2 samples each

    def test_load_empty_directory_raises(self, tmp_path):
        """Loading from empty directory raises FileNotFoundError."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with pytest.raises(FileNotFoundError):
            load_dataset(str(empty_dir), SimpleTestSample)


##
# Integration tests


class TestLoadDatasetIntegration:
    """Integration tests combining multiple features."""

    def test_full_workflow_train_test_split(self, tmp_path):
        """Full workflow: create sharded dataset, load with splits, iterate."""
        # Create train shards
        for i in range(2):
            shard_path = tmp_path / f"train-{i:03d}.tar"
            with wds.writer.TarWriter(str(shard_path)) as sink:
                for j in range(5):
                    sample = SimpleTestSample(text=f"train_{i}_{j}", label=j)
                    sink.write(sample.as_wds)

        # Create test shard
        test_path = tmp_path / "test-000.tar"
        with wds.writer.TarWriter(str(test_path)) as sink:
            for j in range(3):
                sample = SimpleTestSample(text=f"test_{j}", label=j)
                sink.write(sample.as_wds)

        # Load dataset
        ds = load_dataset(str(tmp_path), SimpleTestSample)

        # Verify structure
        assert "train" in ds
        assert "test" in ds

        # Iterate train
        train_samples = list(ds["train"].ordered(batch_size=None))
        assert len(train_samples) == 10  # 2 shards * 5 samples

        # Iterate test
        test_samples = list(ds["test"].ordered(batch_size=None))
        assert len(test_samples) == 3

    def test_batched_iteration(self, tmp_path):
        """Test batched iteration through loaded dataset."""
        tar_path = tmp_path / "data.tar"
        with wds.writer.TarWriter(str(tar_path)) as sink:
            for i in range(20):
                sample = SimpleTestSample(text=f"sample_{i}", label=i % 5)
                sink.write(sample.as_wds)

        ds = load_dataset(str(tar_path), SimpleTestSample, split="train")

        batches = list(ds.ordered(batch_size=4))
        assert len(batches) == 5  # 20 samples / 4 per batch

        # Check batch structure
        first_batch = batches[0]
        assert len(first_batch.samples) == 4
        # Aggregated attributes
        labels = first_batch.label
        assert len(labels) == 4


##
# Indexed path tests


class TestIsIndexedPath:
    """Tests for _is_indexed_path function."""

    def test_at_handle_path(self):
        """@handle/dataset is indexed."""
        assert _is_indexed_path("@maxine.science/mnist") is True

    def test_at_did_path(self):
        """@did:plc:abc/dataset is indexed."""
        assert _is_indexed_path("@did:plc:abc123/my-dataset") is True

    def test_local_path(self):
        """Local paths are not indexed."""
        assert _is_indexed_path("/path/to/data.tar") is False

    def test_s3_path(self):
        """S3 URLs are not indexed."""
        assert _is_indexed_path("s3://bucket/data.tar") is False

    def test_relative_path(self):
        """Relative paths are not indexed."""
        assert _is_indexed_path("./data/train.tar") is False


class TestParseIndexedPath:
    """Tests for _parse_indexed_path function."""

    def test_parse_handle_dataset(self):
        """Parse @handle/dataset format."""
        handle, name, version = _parse_indexed_path("@maxine.science/mnist")
        assert handle == "maxine.science"
        assert name == "mnist"
        assert version is None

    def test_parse_did_dataset(self):
        """Parse @did:plc:xxx/dataset format."""
        handle, name, version = _parse_indexed_path("@did:plc:abc123/my-dataset")
        assert handle == "did:plc:abc123"
        assert name == "my-dataset"
        assert version is None

    def test_parse_with_version(self):
        """Parse @handle/dataset@version format."""
        handle, name, version = _parse_indexed_path("@maxine.science/mnist@1.0.0")
        assert handle == "maxine.science"
        assert name == "mnist"
        assert version == "1.0.0"

    def test_parse_with_freeform_version(self):
        """Parse @handle/dataset@freeform-version format."""
        handle, name, version = _parse_indexed_path(
            "@alice.bsky.social/cifar10@v2-beta"
        )
        assert handle == "alice.bsky.social"
        assert name == "cifar10"
        assert version == "v2-beta"

    def test_parse_invalid_no_slash(self):
        """Invalid path without slash raises ValueError."""
        with pytest.raises(ValueError, match="Invalid indexed path format"):
            _parse_indexed_path("@handle-only")

    def test_parse_invalid_no_at(self):
        """Path without @ raises ValueError."""
        with pytest.raises(ValueError, match="Not an indexed path"):
            _parse_indexed_path("handle/dataset")

    def test_parse_invalid_empty_parts(self):
        """Empty handle or dataset raises ValueError."""
        with pytest.raises(ValueError, match="Invalid indexed path"):
            _parse_indexed_path("@/dataset")


class TestLoadDatasetWithIndex:
    """Tests for load_dataset with index parameter."""

    def test_indexed_path_uses_default_index(self):
        """@handle/dataset without explicit index uses the default Index."""
        # The default index will attempt to resolve "dataset" against the
        # atmosphere backend, which raises because the anonymous client
        # cannot resolve an unknown handle.  The key point is that it no
        # longer raises ValueError("Index required") -- it actually tries.
        with pytest.raises((KeyError, ValueError)):
            load_dataset("@handle/dataset", SimpleTestSample)

    def test_none_sample_type_defaults_to_dictsample(self, tmp_path):
        """sample_type=None returns Dataset[DictSample]."""
        from atdata import DictSample

        # Create a test tar file
        tar_path = tmp_path / "data.tar"
        sample = SimpleTestSample(text="hello", label=42)
        with wds.writer.TarWriter(str(tar_path)) as writer:
            writer.write(sample.as_wds)

        # Load without specifying sample_type
        ds = load_dataset(str(tar_path), split="train")

        # Should return Dataset[DictSample]
        assert ds.sample_type == DictSample

        # Should be able to iterate and access fields
        for sample in ds.ordered():
            assert sample["text"] == "hello"
            assert sample.label == 42
            break

    def test_indexed_path_with_mock_index(self):
        """load_dataset with indexed path uses index lookup."""
        mock_index = Mock()
        mock_index.data_store = None  # No data store, so no URL transformation
        mock_index.get_label.side_effect = KeyError("no label")
        mock_entry = Mock()
        mock_entry.data_urls = ["s3://bucket/data.tar"]
        mock_entry.schema_ref = "local://schemas/test@1.0.0"
        mock_index.get_dataset.return_value = mock_entry

        # sample_type is provided so get_schema_type won't be called
        ds = load_dataset(
            "@local/my-dataset",
            SimpleTestSample,
            index=mock_index,
            split="train",
        )

        mock_index.get_dataset.assert_called_once_with("@local/my-dataset")
        assert ds.url == "s3://bucket/data.tar"

    def test_indexed_path_auto_type_resolution(self):
        """load_dataset with sample_type=None uses get_schema_type."""
        mock_index = Mock()
        mock_index.data_store = None  # No data store, so no URL transformation
        mock_index.get_label.side_effect = KeyError("no label")
        mock_entry = Mock()
        mock_entry.data_urls = ["s3://bucket/data.tar"]
        mock_entry.schema_ref = "local://schemas/test@1.0.0"
        mock_index.get_dataset.return_value = mock_entry
        mock_index.get_schema_type.return_value = SimpleTestSample

        ds = load_dataset(
            "@local/my-dataset",
            None,
            index=mock_index,
            split="train",
        )

        mock_index.get_schema_type.assert_called_once_with("local://schemas/test@1.0.0")
        assert ds.sample_type == SimpleTestSample

    def test_indexed_path_returns_datasetdict_without_split(self):
        """load_dataset with indexed path returns DatasetDict when split=None."""
        mock_index = Mock()
        mock_index.data_store = None  # No data store, so no URL transformation
        mock_index.get_label.side_effect = KeyError("no label")
        mock_entry = Mock()
        mock_entry.data_urls = ["s3://bucket/data.tar"]
        mock_entry.schema_ref = "local://schemas/test@1.0.0"
        mock_index.get_dataset.return_value = mock_entry

        result = load_dataset(
            "@local/my-dataset",
            SimpleTestSample,
            index=mock_index,
        )

        assert isinstance(result, DatasetDict)
        assert "train" in result

    def test_indexed_path_transforms_urls_via_data_store(self):
        """load_dataset transforms URLs through data_store.read_url() if available."""
        mock_data_store = Mock()
        mock_data_store.read_url.return_value = "https://r2.example.com/bucket/data.tar"

        mock_index = Mock()
        mock_index.data_store = mock_data_store
        mock_index.get_label.side_effect = KeyError("no label")
        mock_entry = Mock()
        mock_entry.data_urls = ["s3://bucket/data.tar"]
        mock_entry.schema_ref = "local://schemas/test@1.0.0"
        mock_index.get_dataset.return_value = mock_entry

        ds = load_dataset(
            "@local/my-dataset",
            SimpleTestSample,
            index=mock_index,
            split="train",
        )

        # Verify read_url was called to transform the URL
        mock_data_store.read_url.assert_called_once_with("s3://bucket/data.tar")
        # Verify the transformed URL is used
        assert ds.url == "https://r2.example.com/bucket/data.tar"

    def test_indexed_path_no_transform_without_data_store(self):
        """load_dataset uses URLs unchanged when index has no data_store."""
        mock_index = Mock()
        mock_index.data_store = None
        mock_index.get_label.side_effect = KeyError("no label")
        mock_entry = Mock()
        mock_entry.data_urls = ["s3://bucket/data.tar"]
        mock_entry.schema_ref = "local://schemas/test@1.0.0"
        mock_index.get_dataset.return_value = mock_entry

        ds = load_dataset(
            "@local/my-dataset",
            SimpleTestSample,
            index=mock_index,
            split="train",
        )

        # URL should be unchanged
        assert ds.url == "s3://bucket/data.tar"

    def test_indexed_path_creates_s3source_with_credentials(self):
        """load_dataset creates S3Source with credentials when S3DataStore is available."""
        from atdata.local import S3DataStore
        from atdata._sources import S3Source

        # Create a real S3DataStore with mock credentials
        mock_credentials = {
            "AWS_ACCESS_KEY_ID": "test-access-key",
            "AWS_SECRET_ACCESS_KEY": "test-secret-key",
            "AWS_ENDPOINT": "https://r2.example.com",
        }

        # Mock the S3DataStore
        mock_store = Mock(spec=S3DataStore)
        mock_store.credentials = mock_credentials

        mock_index = Mock()
        mock_index.data_store = mock_store
        mock_index.get_label.side_effect = KeyError("no label")
        mock_entry = Mock()
        mock_entry.data_urls = [
            "s3://my-bucket/train-000.tar",
            "s3://my-bucket/train-001.tar",
        ]
        mock_entry.schema_ref = "local://schemas/test@1.0.0"
        mock_index.get_dataset.return_value = mock_entry

        ds = load_dataset(
            "@local/my-dataset",
            SimpleTestSample,
            index=mock_index,
            split="train",
        )

        # Verify the dataset source is an S3Source with credentials
        assert isinstance(ds.source, S3Source)
        assert ds.source.bucket == "my-bucket"
        assert ds.source.keys == ["train-000.tar", "train-001.tar"]
        assert ds.source.endpoint == "https://r2.example.com"
        assert ds.source.access_key == "test-access-key"
        assert ds.source.secret_key == "test-secret-key"


class TestIndexedPathAtmosphereRouting:
    """Tests that @handle/dataset routes to atmosphere via Index."""

    def test_atmosphere_handle_routes_to_get_dataset(self):
        """@handle.domain/name passes full @handle/name to index methods."""
        mock_index = Mock()
        mock_index.data_store = None
        mock_index.get_label.side_effect = KeyError("no label")
        mock_entry = Mock()
        mock_entry.data_urls = ["https://cdn.example.com/data.tar"]
        mock_entry.schema_ref = "local://schemas/test@1.0.0"
        mock_index.get_dataset.return_value = mock_entry

        ds = load_dataset(
            "@maxine.science/test-mnist",
            SimpleTestSample,
            index=mock_index,
            split="train",
        )

        mock_index.get_label.assert_called_once_with("@maxine.science/test-mnist", None)
        mock_index.get_dataset.assert_called_once_with("@maxine.science/test-mnist")
        assert ds.url == "https://cdn.example.com/data.tar"


##
# AT URI tests


class TestIsAtUri:
    """Tests for _is_at_uri detection."""

    def test_valid_at_uri(self):
        assert (
            _is_at_uri("at://did:plc:abc123/ac.foundation.dataset.entry/rkey") is True
        )

    def test_at_uri_with_handle(self):
        assert (
            _is_at_uri("at://alice.bsky.social/ac.foundation.dataset.entry/rkey")
            is True
        )

    def test_not_at_uri_indexed_path(self):
        assert _is_at_uri("@local/my-dataset") is False

    def test_not_at_uri_s3(self):
        assert _is_at_uri("s3://bucket/data.tar") is False

    def test_not_at_uri_local(self):
        assert _is_at_uri("/path/to/data.tar") is False

    def test_not_at_uri_http(self):
        assert _is_at_uri("https://example.com/data.tar") is False


class TestResolveAtUri:
    """Tests for _resolve_at_uri with mocked Atmosphere client."""

    def _make_mock_client(
        self, storage, schema_ref="at://did:plc:abc/ac.foundation.dataset.schema/s1"
    ):
        """Build a mock Atmosphere client that returns a dataset record with the given storage."""
        client = MagicMock()
        record = {
            "$type": "ac.foundation.dataset.entry",
            "name": "test-dataset",
            "schemaRef": schema_ref,
            "storage": storage,
            "createdAt": "2026-01-01T00:00:00Z",
        }
        client.get_record.return_value = record
        client._resolve_pds_endpoint.return_value = "https://pds.example.com"
        return client

    def test_resolve_http_storage_with_explicit_type(self):
        """AT URI with storageHttp resolves to URLSource Dataset."""
        storage = {
            "$type": "ac.foundation.dataset.storageHttp",
            "shards": [
                {"url": "https://cdn.example.com/shard-000.tar"},
                {"url": "https://cdn.example.com/shard-001.tar"},
            ],
        }
        client = self._make_mock_client(storage)

        ds, resolved_type = _resolve_at_uri(
            "at://did:plc:abc/ac.foundation.dataset.entry/my-ds",
            sample_type=SimpleTestSample,
            client=client,
        )

        assert resolved_type is SimpleTestSample
        assert isinstance(ds.source, atdata._sources.URLSource)

    def test_resolve_http_storage_single_shard(self):
        """Single HTTP shard produces a valid URL."""
        storage = {
            "$type": "ac.foundation.dataset.storageHttp",
            "shards": [{"url": "https://cdn.example.com/data.tar"}],
        }
        client = self._make_mock_client(storage)

        ds, _ = _resolve_at_uri(
            "at://did:plc:abc/ac.foundation.dataset.entry/my-ds",
            sample_type=SimpleTestSample,
            client=client,
        )

        assert ds.url == "https://cdn.example.com/data.tar"

    def test_resolve_s3_storage(self):
        """AT URI with storageS3 resolves to s3:// URLs."""
        storage = {
            "$type": "ac.foundation.dataset.storageS3",
            "bucket": "my-bucket",
            "shards": [{"key": "datasets/shard-000.tar"}],
        }
        client = self._make_mock_client(storage)

        ds, _ = _resolve_at_uri(
            "at://did:plc:abc/ac.foundation.dataset.entry/my-ds",
            sample_type=SimpleTestSample,
            client=client,
        )

        assert "s3://my-bucket/datasets/shard-000.tar" in ds.url

    def test_resolve_s3_with_endpoint(self):
        """storageS3 with endpoint uses endpoint-based URLs."""
        storage = {
            "$type": "ac.foundation.dataset.storageS3",
            "bucket": "my-bucket",
            "endpoint": "https://r2.example.com",
            "shards": [{"key": "data.tar"}],
        }
        client = self._make_mock_client(storage)

        ds, _ = _resolve_at_uri(
            "at://did:plc:abc/ac.foundation.dataset.entry/my-ds",
            sample_type=SimpleTestSample,
            client=client,
        )

        assert ds.url == "https://r2.example.com/my-bucket/data.tar"

    def test_resolve_blob_storage(self):
        """AT URI with storageBlobs resolves to BlobSource."""
        from atdata._sources import BlobSource

        storage = {
            "$type": "ac.foundation.dataset.storageBlobs",
            "blobs": [
                {
                    "blob": {
                        "$type": "blob",
                        "ref": {"$link": "bafkreiabc123"},
                        "mimeType": "application/x-tar",
                        "size": 1024,
                    },
                    "checksum": {"algo": "sha256", "hash": "abc123"},
                },
            ],
        }
        client = self._make_mock_client(storage)

        ds, _ = _resolve_at_uri(
            "at://did:plc:abc/ac.foundation.dataset.entry/my-ds",
            sample_type=SimpleTestSample,
            client=client,
        )

        assert isinstance(ds.source, BlobSource)
        assert len(ds.source.blob_refs) == 1
        assert ds.source.blob_refs[0]["did"] == "did:plc:abc"
        assert ds.source.blob_refs[0]["cid"] == "bafkreiabc123"
        assert ds.source.pds_endpoint == "https://pds.example.com"

    def test_resolve_legacy_external_storage(self):
        """AT URI with legacy storageExternal resolves to URLSource."""
        storage = {
            "$type": "ac.foundation.dataset.storageExternal",
            "urls": ["https://example.com/data.tar"],
        }
        client = self._make_mock_client(storage)

        ds, _ = _resolve_at_uri(
            "at://did:plc:abc/ac.foundation.dataset.entry/my-ds",
            sample_type=SimpleTestSample,
            client=client,
        )

        assert ds.url == "https://example.com/data.tar"

    def test_resolve_no_sample_type_decodes_schema(self):
        """When sample_type=None, schema is decoded from the record's schemaRef."""
        storage = {
            "$type": "ac.foundation.dataset.storageHttp",
            "shards": [{"url": "https://cdn.example.com/data.tar"}],
        }
        client = self._make_mock_client(storage)

        # Mock the schema loader to return a schema record
        schema_record = {
            "$type": "ac.foundation.dataset.schema",
            "name": "TestSample",
            "version": "1.0.0",
            "fields": [
                {
                    "name": "text",
                    "fieldType": {"$type": "local#primitive", "primitive": "str"},
                    "optional": False,
                },
            ],
        }

        with patch("atdata.atmosphere.schema.SchemaLoader") as MockSchemaLoader:
            mock_loader_instance = MockSchemaLoader.return_value
            mock_loader_instance.get.return_value = schema_record

            ds, resolved_type = _resolve_at_uri(
                "at://did:plc:abc/ac.foundation.dataset.entry/my-ds",
                sample_type=None,
                client=client,
            )

            MockSchemaLoader.assert_called_once_with(client)
            mock_loader_instance.get.assert_called_once_with(
                "at://did:plc:abc/ac.foundation.dataset.schema/s1"
            )
            assert resolved_type.__name__ == "TestSample"

    def test_resolve_no_sample_type_no_schema_ref_defaults_to_dictsample(self):
        """When sample_type=None and no schemaRef, falls back to DictSample."""
        storage = {
            "$type": "ac.foundation.dataset.storageHttp",
            "shards": [{"url": "https://cdn.example.com/data.tar"}],
        }
        client = self._make_mock_client(storage, schema_ref=None)
        # Override to return record without schemaRef
        record = {
            "$type": "ac.foundation.dataset.entry",
            "name": "test-dataset",
            "storage": storage,
            "createdAt": "2026-01-01T00:00:00Z",
        }
        client.get_record.return_value = record

        ds, resolved_type = _resolve_at_uri(
            "at://did:plc:abc/ac.foundation.dataset.entry/my-ds",
            sample_type=None,
            client=client,
        )

        from atdata import DictSample

        assert resolved_type is DictSample

    def test_resolve_empty_urls_raises(self):
        """Empty storage URLs raise ValueError."""
        storage = {
            "$type": "ac.foundation.dataset.storageHttp",
            "shards": [],
        }
        client = self._make_mock_client(storage)

        with pytest.raises(ValueError, match="has no storage URLs"):
            _resolve_at_uri(
                "at://did:plc:abc/ac.foundation.dataset.entry/my-ds",
                sample_type=SimpleTestSample,
                client=client,
            )

    def test_resolve_wrong_record_type_raises(self):
        """AT URI pointing to a schema record (not dataset) raises ValueError."""
        client = MagicMock()
        client.get_record.return_value = {
            "$type": "ac.foundation.dataset.schema",
            "name": "SomeSchema",
            "version": "1.0.0",
        }

        with pytest.raises(ValueError, match="is not a dataset record"):
            _resolve_at_uri(
                "at://did:plc:abc/ac.foundation.dataset.schema/s1",
                sample_type=SimpleTestSample,
                client=client,
            )

    def test_resolve_unknown_storage_type_raises(self):
        """Unknown storage $type raises ValueError."""
        client = MagicMock()
        client.get_record.return_value = {
            "$type": "ac.foundation.dataset.entry",
            "name": "test",
            "storage": {"$type": "ac.foundation.dataset.storageFuture"},
            "createdAt": "2026-01-01T00:00:00Z",
        }

        with pytest.raises(ValueError, match="Unknown storage type"):
            _resolve_at_uri(
                "at://did:plc:abc/ac.foundation.dataset.entry/my-ds",
                sample_type=SimpleTestSample,
                client=client,
            )

    def test_resolve_blob_storage_multiple_blobs(self):
        """Multiple blob entries are all resolved into BlobSource refs."""
        from atdata._sources import BlobSource

        storage = {
            "$type": "ac.foundation.dataset.storageBlobs",
            "blobs": [
                {
                    "blob": {
                        "$type": "blob",
                        "ref": {"$link": "bafkreiabc"},
                        "mimeType": "application/x-tar",
                        "size": 1024,
                    },
                },
                {
                    "blob": {
                        "$type": "blob",
                        "ref": {"$link": "bafkreidef"},
                        "mimeType": "application/x-tar",
                        "size": 2048,
                    },
                },
                {
                    "blob": {
                        "$type": "blob",
                        "ref": {"$link": "bafkreighi"},
                        "mimeType": "application/x-tar",
                        "size": 512,
                    },
                },
            ],
        }
        client = self._make_mock_client(storage)

        ds, _ = _resolve_at_uri(
            "at://did:plc:abc/ac.foundation.dataset.entry/my-ds",
            sample_type=SimpleTestSample,
            client=client,
        )

        assert isinstance(ds.source, BlobSource)
        assert len(ds.source.blob_refs) == 3
        cids = [r["cid"] for r in ds.source.blob_refs]
        assert cids == ["bafkreiabc", "bafkreidef", "bafkreighi"]
        assert all(r["did"] == "did:plc:abc" for r in ds.source.blob_refs)

    def test_resolve_single_get_record_call(self):
        """Verify get_record is called exactly once (no redundant fetches)."""
        storage = {
            "$type": "ac.foundation.dataset.storageHttp",
            "shards": [{"url": "https://cdn.example.com/data.tar"}],
        }
        client = self._make_mock_client(storage)

        _resolve_at_uri(
            "at://did:plc:abc/ac.foundation.dataset.entry/my-ds",
            sample_type=SimpleTestSample,
            client=client,
        )

        client.get_record.assert_called_once()

    def test_resolve_creates_anonymous_client_when_none(self):
        """When no client provided, creates unauthenticated Atmosphere."""
        with patch("atdata.atmosphere.client.Atmosphere") as MockAtmo:
            mock_client = MagicMock()
            MockAtmo.return_value = mock_client

            record = {
                "$type": "ac.foundation.dataset.entry",
                "name": "test",
                "schemaRef": None,
                "storage": {
                    "$type": "ac.foundation.dataset.storageHttp",
                    "shards": [{"url": "https://cdn.example.com/data.tar"}],
                },
                "createdAt": "2026-01-01T00:00:00Z",
            }
            mock_client.get_record.return_value = record

            ds, _ = _resolve_at_uri(
                "at://did:plc:abc/ac.foundation.dataset.entry/my-ds",
                sample_type=SimpleTestSample,
            )

            MockAtmo.assert_called_once_with()


class TestLoadDatasetWithAtUri:
    """Integration tests for load_dataset() with AT URIs."""

    def _make_mock_client(self, storage):
        """Build a mock Atmosphere client for AT URI resolution."""
        client = MagicMock()
        record = {
            "$type": "ac.foundation.dataset.entry",
            "name": "test-dataset",
            "schemaRef": None,
            "storage": storage,
            "createdAt": "2026-01-01T00:00:00Z",
        }
        client.get_record.return_value = record
        return client

    @patch("atdata.atmosphere.client.Atmosphere")
    def test_load_dataset_at_uri_with_split(self, MockAtmo):
        """load_dataset with at:// and split returns Dataset."""
        mock_client = MagicMock()
        MockAtmo.return_value = mock_client
        mock_client.get_record.return_value = {
            "$type": "ac.foundation.dataset.entry",
            "name": "test",
            "schemaRef": None,
            "storage": {
                "$type": "ac.foundation.dataset.storageHttp",
                "shards": [{"url": "https://cdn.example.com/data.tar"}],
            },
            "createdAt": "2026-01-01T00:00:00Z",
        }

        ds = load_dataset(
            "at://did:plc:abc/ac.foundation.dataset.entry/my-ds",
            SimpleTestSample,
            split="train",
        )

        assert isinstance(ds, atdata.Dataset)
        assert ds.url == "https://cdn.example.com/data.tar"

    @patch("atdata.atmosphere.client.Atmosphere")
    def test_load_dataset_at_uri_without_split(self, MockAtmo):
        """load_dataset with at:// and no split returns DatasetDict."""
        mock_client = MagicMock()
        MockAtmo.return_value = mock_client
        mock_client.get_record.return_value = {
            "$type": "ac.foundation.dataset.entry",
            "name": "test",
            "schemaRef": None,
            "storage": {
                "$type": "ac.foundation.dataset.storageHttp",
                "shards": [{"url": "https://cdn.example.com/data.tar"}],
            },
            "createdAt": "2026-01-01T00:00:00Z",
        }

        result = load_dataset(
            "at://did:plc:abc/ac.foundation.dataset.entry/my-ds",
            SimpleTestSample,
        )

        assert isinstance(result, DatasetDict)
        assert "train" in result

    @patch("atdata.atmosphere.client.Atmosphere")
    def test_load_dataset_at_uri_precedes_indexed_path(self, MockAtmo):
        """at:// URIs are resolved before @handle/dataset paths."""
        mock_client = MagicMock()
        MockAtmo.return_value = mock_client
        mock_client.get_record.return_value = {
            "$type": "ac.foundation.dataset.entry",
            "name": "test",
            "schemaRef": None,
            "storage": {
                "$type": "ac.foundation.dataset.storageHttp",
                "shards": [{"url": "https://cdn.example.com/data.tar"}],
            },
            "createdAt": "2026-01-01T00:00:00Z",
        }

        # at:// should be handled before checking _is_indexed_path
        ds = load_dataset(
            "at://did:plc:abc/ac.foundation.dataset.entry/rkey",
            SimpleTestSample,
            split="train",
        )

        assert isinstance(ds, atdata.Dataset)
        MockAtmo.assert_called_once()
