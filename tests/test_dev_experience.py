"""Tests for GH#38 developer experience features.

Tests cover:
- Dataset convenience methods: head, __iter__, __len__, get, schema,
  column_names, describe, filter, map, select, to_pandas, to_dict
- Custom exception hierarchy
- CLI commands (inspect, schema, preview)
"""

import pytest
import numpy as np
import webdataset as wds

from atdata import (
    Dataset,
    DictSample,
    LensNetwork,
    packable,
)
from atdata._exceptions import (
    AtdataError,
    LensNotFoundError,
    SchemaError,
    SampleKeyError,
    ShardError,
)
from numpy.typing import NDArray


##
# Fixtures


@packable
class DevSample:
    name: str
    value: int


@packable
class ImageSample:
    label: int
    pixels: NDArray


def _write_tar(path, samples):
    """Write a list of PackableSample instances to a single tar file."""
    with wds.writer.TarWriter(str(path)) as sink:
        for s in samples:
            sink.write(s.as_wds)


@pytest.fixture
def dev_tar(tmp_path):
    """Create a tar with 20 DevSample instances."""
    samples = [DevSample(name=f"s{i:03d}", value=i) for i in range(20)]
    tar_path = tmp_path / "dev.tar"
    _write_tar(tar_path, samples)
    return str(tar_path), samples


@pytest.fixture
def image_tar(tmp_path):
    """Create a tar with 5 ImageSample instances."""
    samples = [
        ImageSample(label=i, pixels=np.random.rand(4, 4).astype(np.float32))
        for i in range(5)
    ]
    tar_path = tmp_path / "images.tar"
    _write_tar(tar_path, samples)
    return str(tar_path), samples


##
# Exception hierarchy tests


class TestExceptions:
    def test_atdata_error_is_base(self):
        assert issubclass(LensNotFoundError, AtdataError)
        assert issubclass(SchemaError, AtdataError)
        assert issubclass(SampleKeyError, AtdataError)
        assert issubclass(ShardError, AtdataError)

    def test_lens_not_found_error_is_value_error(self):
        """Backward compatibility: LensNotFoundError is a ValueError."""
        assert issubclass(LensNotFoundError, ValueError)

    def test_sample_key_error_is_key_error(self):
        assert issubclass(SampleKeyError, KeyError)

    def test_lens_not_found_error_message(self):
        err = LensNotFoundError(DevSample, ImageSample)
        msg = str(err)
        assert "DevSample" in msg
        assert "ImageSample" in msg
        assert "\u2192" in msg
        assert "@lens" in msg

    def test_lens_not_found_error_with_available(self):
        err = LensNotFoundError(
            DevSample,
            ImageSample,
            available_targets=[(DictSample, "dev_to_dict")],
        )
        msg = str(err)
        assert "DictSample" in msg
        assert "dev_to_dict" in msg

    def test_schema_error_message(self):
        err = SchemaError("MySample", ["a", "b", "c"], ["a", "d"])
        msg = str(err)
        assert "MySample" in msg
        assert "Missing fields" in msg
        assert "b" in msg
        assert "c" in msg
        assert "Unexpected fields" in msg
        assert "d" in msg

    def test_sample_key_error_message(self):
        err = SampleKeyError("abc123")
        assert "abc123" in str(err)

    def test_shard_error_message(self):
        err = ShardError("data-000001.tar", "file not found")
        assert "data-000001.tar" in str(err)
        assert "file not found" in str(err)

    def test_lens_network_raises_lens_not_found(self):
        """LensNetwork.transform() raises LensNotFoundError."""

        @packable
        class IsolatedA:
            x: int

        @packable
        class IsolatedB:
            y: int

        network = LensNetwork()
        with pytest.raises(LensNotFoundError):
            network.transform(IsolatedA, IsolatedB)


##
# Dataset convenience method tests


class TestDatasetHead:
    def test_head_default(self, dev_tar):
        url, samples = dev_tar
        ds = Dataset[DevSample](url)
        result = ds.head()
        assert len(result) == 5
        assert all(isinstance(s, DevSample) for s in result)

    def test_head_custom_n(self, dev_tar):
        url, samples = dev_tar
        ds = Dataset[DevSample](url)
        result = ds.head(3)
        assert len(result) == 3

    def test_head_more_than_available(self, dev_tar):
        url, samples = dev_tar
        ds = Dataset[DevSample](url)
        result = ds.head(1000)
        assert len(result) == 20

    def test_head_zero(self, dev_tar):
        url, _ = dev_tar
        ds = Dataset[DevSample](url)
        result = ds.head(0)
        assert result == []


class TestDatasetIter:
    def test_iter(self, dev_tar):
        url, samples = dev_tar
        ds = Dataset[DevSample](url)
        collected = []
        for s in ds:
            collected.append(s)
        assert len(collected) == 20
        assert all(isinstance(s, DevSample) for s in collected)


class TestDatasetLen:
    def test_len(self, dev_tar):
        url, _ = dev_tar
        ds = Dataset[DevSample](url)
        assert len(ds) == 20

    def test_len_cached(self, dev_tar):
        url, _ = dev_tar
        ds = Dataset[DevSample](url)
        first = len(ds)
        second = len(ds)
        assert first == second == 20


class TestDatasetSchema:
    def test_schema_typed(self, dev_tar):
        url, _ = dev_tar
        ds = Dataset[DevSample](url)
        schema = ds.schema
        assert "name" in schema
        assert "value" in schema

    def test_schema_dict_sample(self, dev_tar):
        url, _ = dev_tar
        ds = Dataset[DictSample](url)
        schema = ds.schema
        assert "_data" in schema

    def test_column_names_typed(self, dev_tar):
        url, _ = dev_tar
        ds = Dataset[DevSample](url)
        cols = ds.column_names
        assert "name" in cols
        assert "value" in cols

    def test_column_names_dict_sample(self, dev_tar):
        url, _ = dev_tar
        ds = Dataset[DictSample](url)
        cols = ds.column_names
        assert cols == []


class TestDatasetDescribe:
    def test_describe(self, dev_tar):
        url, _ = dev_tar
        ds = Dataset[DevSample](url)
        info = ds.describe()
        assert info["sample_type"] == "DevSample"
        assert "name" in info["fields"]
        assert info["num_shards"] == 1
        assert info["url"] == url


class TestDatasetGet:
    def test_get_existing(self, dev_tar):
        url, _ = dev_tar
        ds = Dataset[DevSample](url)
        # Extract a key from the raw pipeline using the dataset's source
        from atdata.dataset import _ShardListStage, _StreamOpenerStage

        pipeline = wds.pipeline.DataPipeline(
            _ShardListStage(ds.source),
            _StreamOpenerStage(ds.source),
            wds.tariterators.tar_file_expander,
            wds.tariterators.group_by_keys,
        )
        raw = next(iter(pipeline))
        key = raw["__key__"]
        found = ds.get(key)
        assert isinstance(found, DevSample)

    def test_get_missing(self, dev_tar):
        url, _ = dev_tar
        ds = Dataset[DevSample](url)
        with pytest.raises(SampleKeyError):
            ds.get("nonexistent-key-12345")


class TestDatasetFilter:
    def test_filter(self, dev_tar):
        url, _ = dev_tar
        ds = Dataset[DevSample](url)
        filtered = ds.filter(lambda s: s.value >= 15)
        result = list(filtered.ordered())
        assert len(result) == 5
        assert all(s.value >= 15 for s in result)

    def test_filter_chain(self, dev_tar):
        url, _ = dev_tar
        ds = Dataset[DevSample](url)
        filtered = ds.filter(lambda s: s.value >= 10).filter(lambda s: s.value < 15)
        result = list(filtered.ordered())
        assert len(result) == 5
        assert all(10 <= s.value < 15 for s in result)


class TestDatasetMap:
    def test_map(self, dev_tar):
        url, _ = dev_tar
        ds = Dataset[DevSample](url)
        mapped = ds.map(lambda s: s.name)
        result = list(mapped.ordered())
        assert len(result) == 20
        assert all(isinstance(r, str) for r in result)
        assert result[0] == "s000"


class TestDatasetSelect:
    def test_select(self, dev_tar):
        url, _ = dev_tar
        ds = Dataset[DevSample](url)
        result = ds.select([0, 5, 10])
        assert len(result) == 3
        assert result[0].value == 0
        assert result[1].value == 5
        assert result[2].value == 10

    def test_select_empty(self, dev_tar):
        url, _ = dev_tar
        ds = Dataset[DevSample](url)
        assert ds.select([]) == []


class TestDatasetToPandas:
    def test_to_pandas(self, dev_tar):
        url, _ = dev_tar
        ds = Dataset[DevSample](url)
        df = ds.to_pandas(limit=5)
        assert len(df) == 5
        assert "name" in df.columns
        assert "value" in df.columns

    def test_to_pandas_full(self, dev_tar):
        url, _ = dev_tar
        ds = Dataset[DevSample](url)
        df = ds.to_pandas()
        assert len(df) == 20


class TestDatasetToDict:
    def test_to_dict(self, dev_tar):
        url, _ = dev_tar
        ds = Dataset[DevSample](url)
        d = ds.to_dict(limit=5)
        assert "name" in d
        assert "value" in d
        assert len(d["name"]) == 5

    def test_to_dict_full(self, dev_tar):
        url, _ = dev_tar
        ds = Dataset[DevSample](url)
        d = ds.to_dict()
        assert len(d["name"]) == 20

    def test_to_dict_empty(self, tmp_path):
        tar_path = tmp_path / "empty.tar"
        _write_tar(tar_path, [])
        ds = Dataset[DevSample](str(tar_path))
        d = ds.to_dict()
        assert d == {}


class TestDatasetWithNDArray:
    def test_head_with_ndarray(self, image_tar):
        url, _ = image_tar
        ds = Dataset[ImageSample](url)
        result = ds.head(2)
        assert len(result) == 2
        assert isinstance(result[0].pixels, np.ndarray)
        assert result[0].pixels.shape == (4, 4)

    def test_to_pandas_with_ndarray(self, image_tar):
        url, _ = image_tar
        ds = Dataset[ImageSample](url)
        df = ds.to_pandas(limit=3)
        assert len(df) == 3
        assert "pixels" in df.columns


##
# CLI tests


class TestCLIInspect:
    def test_inspect_returns_zero(self, dev_tar):
        url, _ = dev_tar
        from atdata.cli.inspect import inspect_dataset

        ret = inspect_dataset(url)
        assert ret == 0

    def test_inspect_bad_url(self):
        from atdata.cli.inspect import inspect_dataset

        ret = inspect_dataset("/nonexistent/path.tar")
        assert ret == 1


class TestCLIPreview:
    def test_preview_returns_zero(self, dev_tar):
        url, _ = dev_tar
        from atdata.cli.preview import preview_dataset

        ret = preview_dataset(url, limit=3)
        assert ret == 0


class TestCLISchema:
    def test_schema_show_returns_zero(self, dev_tar):
        url, _ = dev_tar
        from atdata.cli.schema import schema_show

        ret = schema_show(url)
        assert ret == 0

    def test_schema_diff_identical(self, dev_tar):
        url, _ = dev_tar
        from atdata.cli.schema import schema_diff

        ret = schema_diff(url, url)
        assert ret == 0

    def test_schema_diff_different(self, dev_tar, image_tar):
        url_a, _ = dev_tar
        url_b, _ = image_tar
        from atdata.cli.schema import schema_diff

        ret = schema_diff(url_a, url_b)
        assert ret == 1


class TestCLIArgparse:
    def test_inspect_command_parsed(self):
        from atdata.cli import main

        # --help would print and exit; just verify the import works
        assert callable(main)
