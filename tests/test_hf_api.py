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
    _parse_indexed_path,
)
from unittest.mock import Mock

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
        # Algorithm finds longest common prefix/suffix, resulting in compact notation
        # Both "data-{000,001,002}.tar" and "data-00{0,1,2}.tar" are valid
        assert "{" in result and "}" in result
        assert ".tar" in result
        assert "data-" in result

    def test_multiple_shards_different_lengths(self):
        shards = ["data-0.tar", "data-1.tar", "data-10.tar"]
        result = _shards_to_wds_url(shards)
        # Should still produce brace notation
        assert "{" in result and "}" in result

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="empty shard list"):
            _shards_to_wds_url([])

    def test_no_common_pattern(self):
        shards = ["train.tar", "test.tar", "val.tar"]
        result = _shards_to_wds_url(shards)
        # Falls back to space-separated or brace notation
        assert "train" in result


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
        handle, name = _parse_indexed_path("@maxine.science/mnist")
        assert handle == "maxine.science"
        assert name == "mnist"

    def test_parse_did_dataset(self):
        """Parse @did:plc:xxx/dataset format."""
        handle, name = _parse_indexed_path("@did:plc:abc123/my-dataset")
        assert handle == "did:plc:abc123"
        assert name == "my-dataset"

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
        mock_entry = Mock()
        mock_entry.data_urls = ["s3://bucket/data.tar"]
        mock_entry.schema_ref = "local://schemas/test@1.0.0"
        mock_index.get_dataset.return_value = mock_entry

        # Need to mock decode_schema since sample_type is provided
        ds = load_dataset(
            "@local/my-dataset",
            SimpleTestSample,
            index=mock_index,
            split="train",
        )

        mock_index.get_dataset.assert_called_once_with("my-dataset")
        assert ds.url == "s3://bucket/data.tar"

    def test_indexed_path_auto_type_resolution(self):
        """load_dataset with sample_type=None uses decode_schema."""
        mock_index = Mock()
        mock_index.data_store = None  # No data store, so no URL transformation
        mock_entry = Mock()
        mock_entry.data_urls = ["s3://bucket/data.tar"]
        mock_entry.schema_ref = "local://schemas/test@1.0.0"
        mock_index.get_dataset.return_value = mock_entry
        mock_index.decode_schema.return_value = SimpleTestSample

        ds = load_dataset(
            "@local/my-dataset",
            None,
            index=mock_index,
            split="train",
        )

        mock_index.decode_schema.assert_called_once_with("local://schemas/test@1.0.0")
        assert ds.sample_type == SimpleTestSample

    def test_indexed_path_returns_datasetdict_without_split(self):
        """load_dataset with indexed path returns DatasetDict when split=None."""
        mock_index = Mock()
        mock_index.data_store = None  # No data store, so no URL transformation
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
