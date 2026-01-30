"""Test dataaset functionality."""

##
# Imports

# Tests
import pytest

# System
from dataclasses import dataclass

# External
import numpy as np
import webdataset as wds

# Local
import atdata
import atdata.dataset as atds

# Typing
from numpy.typing import NDArray
from typing import (
    Type,
)


##
# Sample test cases


@dataclass
class BasicTestSample(atdata.PackableSample):
    name: str
    position: int
    value: float


@dataclass
class NumpyTestSample(atdata.PackableSample):
    label: int
    image: NDArray


@atdata.packable
class BasicTestSampleDecorated:
    name: str
    position: int
    value: float


@atdata.packable
class NumpyTestSampleDecorated:
    label: int
    image: NDArray


@atdata.packable
class NumpyOptionalSampleDecorated:
    label: int
    image: NDArray
    embeddings: NDArray | None = None


test_cases = [
    {
        "SampleType": BasicTestSample,
        "sample_data": {
            "name": "Hello, world!",
            "position": 42,
            "value": 1024.768,
        },
        "sample_wds_stem": "basic_test",
        "test_parquet": True,
    },
    {
        "SampleType": NumpyTestSample,
        "sample_data": {
            "label": 9_001,
            "image": np.random.randn(1024, 1024),
        },
        "sample_wds_stem": "numpy_test",
        "test_parquet": False,
    },
    {
        "SampleType": BasicTestSampleDecorated,
        "sample_data": {
            "name": "Hello, world!",
            "position": 42,
            "value": 1024.768,
        },
        "sample_wds_stem": "basic_test_decorated",
        "test_parquet": True,
    },
    {
        "SampleType": NumpyTestSampleDecorated,
        "sample_data": {
            "label": 9_001,
            "image": np.random.randn(1024, 1024),
        },
        "sample_wds_stem": "numpy_test_decorated",
        "test_parquet": False,
    },
    {
        "SampleType": NumpyOptionalSampleDecorated,
        "sample_data": {
            "label": 9_001,
            "image": np.random.randn(1024, 1024),
            "embeddings": np.random.randn(512),
        },
        "sample_wds_stem": "numpy_optional_decorated",
        "test_parquet": False,
    },
    {
        "SampleType": NumpyOptionalSampleDecorated,
        "sample_data": {
            "label": 9_001,
            "image": np.random.randn(1024, 1024),
            "embeddings": None,
        },
        "sample_wds_stem": "numpy_optional_decorated_none",
        "test_parquet": False,
    },
]


## Tests


@pytest.mark.parametrize(
    ("SampleType", "sample_data"),
    [(case["SampleType"], case["sample_data"]) for case in test_cases],
)
def test_create_sample(
    SampleType: Type[atdata.Packable],
    sample_data: atds.WDSRawSample,
):
    """Test our ability to create samples from semi-structured data"""

    sample = SampleType.from_data(sample_data)
    assert isinstance(sample, SampleType), (
        f"Did not properly form sample for test type {SampleType}"
    )

    for k, v in sample_data.items():
        cur_assertion: bool
        if isinstance(v, np.ndarray):
            cur_assertion = np.all(getattr(sample, k) == v)
        else:
            cur_assertion = getattr(sample, k) == v
        assert cur_assertion, (
            f"Did not properly incorporate property {k} of test type {SampleType}"
        )


@pytest.mark.parametrize(
    ("SampleType", "sample_data", "sample_wds_stem"),
    [
        (case["SampleType"], case["sample_data"], case["sample_wds_stem"])
        for case in test_cases
    ],
)
def test_wds(
    SampleType: Type[atdata.Packable],
    sample_data: atds.WDSRawSample,
    sample_wds_stem: str,
    tmp_path,
):
    """Test our ability to write samples as `WebDatasets` to disk"""

    ## Testing hyperparameters

    n_copies = 100
    shard_maxcount = 10
    batch_size = 4
    n_iterate = 10

    ## Write sharded dataset

    file_pattern = (tmp_path / (f"{sample_wds_stem}" + "-{shard_id}.tar")).as_posix()
    file_wds_pattern = file_pattern.format(shard_id="%06d")

    with wds.writer.ShardWriter(
        pattern=file_wds_pattern,
        maxcount=shard_maxcount,
    ) as sink:
        for i_sample in range(n_copies):
            new_sample = SampleType.from_data(sample_data)
            assert isinstance(new_sample, SampleType), (
                f"Did not properly form sample for test type {SampleType}"
            )

            sink.write(new_sample.as_wds)

    ## Ordered

    # Read first shard, no batches

    first_filename = file_pattern.format(shard_id=f"{0:06d}")
    dataset = atdata.Dataset[SampleType](first_filename)

    iterations_run = 0
    for i_iterate, cur_sample in enumerate(dataset.ordered(batch_size=None)):
        assert isinstance(cur_sample, SampleType), (
            f"Single sample for {SampleType} written to `wds` is of wrong type"
        )

        # Check sample values

        for k, v in sample_data.items():
            if isinstance(v, np.ndarray):
                is_correct = np.all(getattr(cur_sample, k) == v)
            else:
                is_correct = getattr(cur_sample, k) == v
            assert is_correct, (
                f"{SampleType}: Incorrect sample value found for {k} - {type(getattr(cur_sample, k))}"
            )

        iterations_run += 1
        if iterations_run >= n_iterate:
            break

    assert iterations_run == n_iterate, (
        f"Only found {iterations_run} samples, not {n_iterate}"
    )

    # Read all shards, batches

    start_id = f"{0:06d}"
    end_id = f"{9:06d}"
    first_filename = file_pattern.format(shard_id="{" + start_id + ".." + end_id + "}")
    dataset = atdata.Dataset[SampleType](first_filename)

    iterations_run = 0
    for i_iterate, cur_batch in enumerate(dataset.ordered(batch_size=batch_size)):
        assert isinstance(cur_batch, atdata.SampleBatch), (
            f"{SampleType}: Batch sample is not correctly a batch"
        )

        assert cur_batch.sample_type == SampleType, (
            f"{SampleType}: Batch `sample_type` is incorrect type"
        )

        if i_iterate == 0:
            cur_n = len(cur_batch.samples)
            assert cur_n == batch_size, (
                f"{SampleType}: Batch has {cur_n} samples, not {batch_size}"
            )

        assert isinstance(cur_batch.samples[0], SampleType), (
            f"{SampleType}: Batch sample of wrong type ({type(cur_batch.samples[0])})"
        )

        # Check batch values
        for k, v in sample_data.items():
            cur_batch_data = getattr(cur_batch, k)

            if isinstance(v, np.ndarray):
                assert isinstance(cur_batch_data, np.ndarray), (
                    f"{SampleType}: `NDArray` not carried through to batch"
                )

                is_correct = all(
                    [
                        np.all(cur_batch_data[i] == v)
                        for i in range(cur_batch_data.shape[0])
                    ]
                )

            else:
                is_correct = all(
                    [cur_batch_data[i] == v for i in range(len(cur_batch_data))]
                )

            assert is_correct, f"{SampleType}: Incorrect sample value found for {k}"

        iterations_run += 1
        if iterations_run >= n_iterate:
            break

    assert iterations_run == n_iterate, (
        f"Only found {iterations_run} samples, not {n_iterate}"
    )

    ## Shuffled

    # Read first shard, no batches

    first_filename = file_pattern.format(shard_id=f"{0:06d}")
    dataset = atdata.Dataset[SampleType](first_filename)

    iterations_run = 0
    for i_iterate, cur_sample in enumerate(dataset.shuffled(batch_size=None)):
        assert isinstance(cur_sample, SampleType), (
            f"Single sample for {SampleType} written to `wds` is of wrong type"
        )

        iterations_run += 1
        if iterations_run >= n_iterate:
            break

    assert iterations_run == n_iterate, (
        f"Only found {iterations_run} samples, not {n_iterate}"
    )

    # Read all shards, batches

    start_id = f"{0:06d}"
    end_id = f"{9:06d}"
    first_filename = file_pattern.format(shard_id="{" + start_id + ".." + end_id + "}")
    dataset = atdata.Dataset[SampleType](first_filename)

    iterations_run = 0
    for i_iterate, cur_sample in enumerate(dataset.shuffled(batch_size=batch_size)):
        assert isinstance(cur_sample, atdata.SampleBatch), (
            f"{SampleType}: Batch sample is not correctly a batch"
        )

        assert cur_sample.sample_type == SampleType, (
            f"{SampleType}: Batch `sample_type` is incorrect type"
        )

        if i_iterate == 0:
            cur_n = len(cur_sample.samples)
            assert cur_n == batch_size, (
                f"{SampleType}: Batch has {cur_n} samples, not {batch_size}"
            )

        assert isinstance(cur_sample.samples[0], SampleType), (
            f"{SampleType}: Batch sample of wrong type ({type(cur_sample.samples[0])})"
        )

        iterations_run += 1
        if iterations_run >= n_iterate:
            break

    assert iterations_run == n_iterate, (
        f"Only found {iterations_run} samples, not {n_iterate}"
    )


#


@pytest.mark.parametrize(
    ("SampleType", "sample_data", "sample_wds_stem", "test_parquet"),
    [
        (
            case["SampleType"],
            case["sample_data"],
            case["sample_wds_stem"],
            case["test_parquet"],
        )
        for case in test_cases
    ],
)
def test_parquet_export(
    SampleType: Type[atdata.Packable],
    sample_data: atds.WDSRawSample,
    sample_wds_stem: str,
    test_parquet: bool,
    tmp_path,
):
    """Test our ability to export a dataset to `parquet` format"""

    # Skip irrelevant test cases
    if not test_parquet:
        return

    ## Testing hyperparameters

    n_copies_dataset = 1_000
    n_per_file = 100

    ## Start out by writing tar dataset

    wds_filename = (tmp_path / f"{sample_wds_stem}.tar").as_posix()
    with wds.writer.TarWriter(wds_filename) as sink:
        for _ in range(n_copies_dataset):
            new_sample = SampleType.from_data(sample_data)
            sink.write(new_sample.as_wds)

    ## Now export to `parquet`

    dataset = atdata.Dataset[SampleType](wds_filename)
    parquet_filename = tmp_path / f"{sample_wds_stem}.parquet"
    dataset.to_parquet(parquet_filename)

    parquet_filename = tmp_path / f"{sample_wds_stem}-segments.parquet"
    dataset.to_parquet(parquet_filename, maxcount=n_per_file)


##
# Edge case tests for coverage


def test_batch_aggregate_empty():
    """Test _batch_aggregate with empty list returns empty list."""
    result = atds._batch_aggregate([])
    assert result == [], "Empty input should return empty list"


def test_sample_batch_attribute_error():
    """Test SampleBatch raises AttributeError for non-existent attributes."""

    @atdata.packable
    class SimpleSample:
        name: str
        value: int

    samples = [SimpleSample(name="test", value=1)]
    batch = atdata.SampleBatch[SimpleSample](samples)

    with pytest.raises(AttributeError, match="No sample attribute named"):
        _ = batch.nonexistent_attribute


def test_sample_batch_type_property():
    """Test SampleBatch.sample_type property."""

    @atdata.packable
    class TypedSample:
        data: str

    samples = [TypedSample(data="hello")]
    batch = atdata.SampleBatch[TypedSample](samples)

    assert batch.sample_type == TypedSample


def test_dataset_batch_type_property(tmp_path):
    """Test Dataset.batch_type property."""

    @atdata.packable
    class BatchTypeSample:
        value: int

    # Create a simple dataset
    wds_filename = (tmp_path / "batch_type_test.tar").as_posix()
    with wds.writer.TarWriter(wds_filename) as sink:
        sample = BatchTypeSample(value=42)
        sink.write(sample.as_wds)

    dataset = atdata.Dataset[BatchTypeSample](wds_filename)
    batch_type = dataset.batch_type

    # batch_type should be SampleBatch parameterized with the sample type
    assert batch_type.__origin__ == atdata.SampleBatch


def test_dataset_shard_list_property(tmp_path):
    """Test Dataset.shard_list property returns list of shard URLs."""

    @atdata.packable
    class ShardListSample:
        value: int

    # Create multiple shards
    file_pattern = (tmp_path / "shards_test-%06d.tar").as_posix()
    with wds.writer.ShardWriter(pattern=file_pattern, maxcount=5) as sink:
        for i in range(15):
            sample = ShardListSample(value=i)
            sink.write(sample.as_wds)

    # Read with brace pattern
    brace_pattern = (tmp_path / "shards_test-{000000..000002}.tar").as_posix()
    dataset = atdata.Dataset[ShardListSample](brace_pattern)

    shard_list = dataset.shard_list
    assert isinstance(shard_list, list)
    assert len(shard_list) == 3


def test_dataset_metadata_property(tmp_path):
    """Test Dataset.metadata property fetches and caches metadata from URL."""
    from unittest.mock import patch, Mock
    import msgpack

    @atdata.packable
    class MetadataSample:
        value: int

    # Create a simple dataset
    wds_filename = (tmp_path / "metadata_test.tar").as_posix()
    with wds.writer.TarWriter(wds_filename) as sink:
        sample = MetadataSample(value=42)
        sink.write(sample.as_wds)

    # Mock the requests.get call
    mock_metadata = {"key": "value", "count": 100}
    mock_response = Mock()
    mock_response.content = msgpack.packb(mock_metadata)
    mock_response.raise_for_status = Mock()
    mock_response.__enter__ = Mock(return_value=mock_response)
    mock_response.__exit__ = Mock(return_value=False)

    with patch("atdata.dataset.requests.get", return_value=mock_response) as mock_get:
        dataset = atdata.Dataset[MetadataSample](
            wds_filename, metadata_url="http://example.com/metadata.msgpack"
        )

        # First call should fetch
        metadata = dataset.metadata
        assert metadata == mock_metadata
        mock_get.assert_called_once_with(
            "http://example.com/metadata.msgpack", stream=True
        )

        # Second call should use cache
        metadata2 = dataset.metadata
        assert metadata2 == mock_metadata
        assert mock_get.call_count == 1  # Still only one call


def test_dataset_metadata_property_none(tmp_path):
    """Test Dataset.metadata returns None when no metadata_url is set."""

    @atdata.packable
    class NoMetadataSample:
        value: int

    wds_filename = (tmp_path / "no_metadata_test.tar").as_posix()
    with wds.writer.TarWriter(wds_filename) as sink:
        sample = NoMetadataSample(value=42)
        sink.write(sample.as_wds)

    dataset = atdata.Dataset[NoMetadataSample](wds_filename)
    assert dataset.metadata is None


def test_parquet_export_with_remainder(tmp_path):
    """Test parquet export with maxcount that doesn't divide evenly."""

    @atdata.packable
    class RemainderSample:
        name: str
        value: int

    # Create dataset with 25 samples
    n_samples = 25
    maxcount = 10  # Will create 3 segments: 10, 10, 5

    wds_filename = (tmp_path / "remainder_test.tar").as_posix()
    with wds.writer.TarWriter(wds_filename) as sink:
        for i in range(n_samples):
            sample = RemainderSample(name=f"sample_{i}", value=i)
            sink.write(sample.as_wds)

    dataset = atdata.Dataset[RemainderSample](wds_filename)
    parquet_path = tmp_path / "remainder_output.parquet"
    dataset.to_parquet(parquet_path, maxcount=maxcount)

    # Should have created 3 segment files
    import pandas as pd

    segment_files = list(tmp_path.glob("remainder_output-*.parquet"))
    assert len(segment_files) == 3

    # Check total row count
    total_rows = sum(len(pd.read_parquet(f)) for f in segment_files)
    assert total_rows == n_samples


def test_dataset_with_lens_batched(tmp_path):
    """Test dataset iteration with lens transformation in batch mode."""
    from dataclasses import dataclass

    @dataclass
    class SourceSample(atdata.PackableSample):
        name: str
        age: int
        score: float

    @dataclass
    class ViewSample(atdata.PackableSample):
        name: str
        score: float

    @atdata.lens
    def extract_view(s: SourceSample) -> ViewSample:
        return ViewSample(name=s.name, score=s.score)

    # Create dataset
    n_samples = 20
    batch_size = 4
    wds_filename = (tmp_path / "lens_batch_test.tar").as_posix()

    with wds.writer.TarWriter(wds_filename) as sink:
        for i in range(n_samples):
            sample = SourceSample(name=f"person_{i}", age=20 + i, score=float(i) * 1.5)
            sink.write(sample.as_wds)

    # Read with lens transformation in batch mode
    dataset = atdata.Dataset[SourceSample](wds_filename).as_type(ViewSample)

    batches_seen = 0
    for batch in dataset.ordered(batch_size=batch_size):
        assert isinstance(batch, atdata.SampleBatch)
        assert batch.sample_type == ViewSample

        # Check that samples are ViewSample type (not SourceSample)
        for sample in batch.samples:
            assert isinstance(sample, ViewSample)
            assert hasattr(sample, "name")
            assert hasattr(sample, "score")
            assert not hasattr(sample, "age")  # age is not in ViewSample

        batches_seen += 1

    assert batches_seen == n_samples // batch_size


def test_from_bytes_invalid_msgpack():
    """Test from_bytes raises on invalid msgpack data."""

    @atdata.packable
    class SimpleSample:
        value: int

    with pytest.raises(TypeError):  # unpacked data is not a mapping
        SimpleSample.from_bytes(b"not valid msgpack data")


def test_from_bytes_missing_field():
    """Test from_bytes raises when required field is missing."""

    @atdata.packable
    class RequiredFieldSample:
        name: str
        count: int

    import ormsgpack

    # Only provide 'name', missing 'count'
    incomplete_data = ormsgpack.packb({"name": "test"})

    with pytest.raises(TypeError):  # Missing required argument
        RequiredFieldSample.from_bytes(incomplete_data)


def test_wrap_missing_msgpack_key(tmp_path):
    """Test wrap raises ValueError on sample missing msgpack key."""

    @atdata.packable
    class WrapTestSample:
        value: int

    wds_filename = (tmp_path / "wrap_test.tar").as_posix()
    with wds.writer.TarWriter(wds_filename) as sink:
        sample = WrapTestSample(value=42)
        sink.write(sample.as_wds)

    dataset = atdata.Dataset[WrapTestSample](wds_filename)

    # Directly call wrap with missing key
    with pytest.raises(ValueError, match="missing 'msgpack' key"):
        dataset.wrap({"__key__": "test"})  # Missing 'msgpack' key


def test_wrap_wrong_msgpack_type(tmp_path):
    """Test wrap raises ValueError when msgpack value is not bytes."""

    @atdata.packable
    class WrapTypeSample:
        value: int

    wds_filename = (tmp_path / "wrap_type_test.tar").as_posix()
    with wds.writer.TarWriter(wds_filename) as sink:
        sample = WrapTypeSample(value=42)
        sink.write(sample.as_wds)

    dataset = atdata.Dataset[WrapTypeSample](wds_filename)

    # Directly call wrap with wrong type
    with pytest.raises(ValueError, match="to be bytes"):
        dataset.wrap({"__key__": "test", "msgpack": "not bytes"})


def test_wrap_corrupted_msgpack(tmp_path):
    """Test wrap raises on corrupted msgpack bytes."""

    @atdata.packable
    class CorruptedSample:
        value: int

    wds_filename = (tmp_path / "corrupted_test.tar").as_posix()
    with wds.writer.TarWriter(wds_filename) as sink:
        sample = CorruptedSample(value=42)
        sink.write(sample.as_wds)

    dataset = atdata.Dataset[CorruptedSample](wds_filename)

    # Corrupted msgpack bytes should raise during deserialization
    with pytest.raises(TypeError):  # unpacked data is not a mapping
        dataset.wrap({"__key__": "test", "msgpack": b"\xff\xfe\x00\x01invalid"})


def test_dataset_nonexistent_file():
    """Test Dataset raises on nonexistent tar file during iteration."""

    @atdata.packable
    class NonexistentSample:
        value: int

    dataset = atdata.Dataset[NonexistentSample]("/nonexistent/path/data.tar")

    # Dataset creation succeeds (lazy loading) — verify type is correct
    assert dataset.sample_type is NonexistentSample

    # Iteration fails when file doesn't exist
    with pytest.raises(FileNotFoundError):
        list(dataset.ordered(batch_size=None))


def test_dataset_invalid_batch_size(tmp_path):
    """Test Dataset raises on invalid batch_size values."""

    @atdata.packable
    class BatchSizeSample:
        value: int

    wds_filename = (tmp_path / "batch_test.tar").as_posix()
    with wds.writer.TarWriter(wds_filename) as sink:
        sample = BatchSizeSample(value=42)
        sink.write(sample.as_wds)

    dataset = atdata.Dataset[BatchSizeSample](wds_filename)

    # batch_size=0 produces empty batches, causing IndexError in webdataset's
    # batched() when it tries to inspect the first element of an empty group
    with pytest.raises(IndexError):
        list(dataset.ordered(batch_size=0))


##
# DictSample tests


def test_dictsample_creation():
    """Test DictSample can be created with keyword args or dict."""
    # From keyword args
    ds1 = atdata.DictSample(name="test", value=42)
    assert ds1.name == "test"
    assert ds1.value == 42

    # From _data dict
    ds2 = atdata.DictSample(_data={"name": "test2", "value": 100})
    assert ds2.name == "test2"
    assert ds2.value == 100


def test_dictsample_getattr():
    """Test DictSample attribute access."""
    sample = atdata.DictSample(text="hello", label=1)

    assert sample.text == "hello"
    assert sample.label == 1

    # Non-existent attribute raises AttributeError
    with pytest.raises(AttributeError, match="has no field"):
        _ = sample.nonexistent


def test_dictsample_getitem():
    """Test DictSample dict-style access."""
    sample = atdata.DictSample(text="hello", label=1)

    assert sample["text"] == "hello"
    assert sample["label"] == 1

    # Non-existent key raises KeyError
    with pytest.raises(KeyError):
        _ = sample["nonexistent"]


def test_dictsample_dict_methods():
    """Test DictSample dict-like methods."""
    sample = atdata.DictSample(a=1, b=2, c=3)

    assert set(sample.keys()) == {"a", "b", "c"}
    assert set(sample.values()) == {1, 2, 3}
    assert set(sample.items()) == {("a", 1), ("b", 2), ("c", 3)}
    assert "a" in sample
    assert "x" not in sample
    assert sample.get("a") == 1
    assert sample.get("x", "default") == "default"


def test_dictsample_to_dict():
    """Test DictSample.to_dict returns a copy."""
    sample = atdata.DictSample(name="test", value=42)
    d = sample.to_dict()

    assert d == {"name": "test", "value": 42}
    # Should be a copy
    d["name"] = "modified"
    assert sample.name == "test"


def test_dictsample_serialization():
    """Test DictSample can be serialized and deserialized."""
    original = atdata.DictSample(text="hello", count=42)

    # Serialize
    packed = original.packed

    # Deserialize
    restored = atdata.DictSample.from_bytes(packed)

    assert restored.text == "hello"
    assert restored.count == 42


def test_dictsample_as_wds():
    """Test DictSample.as_wds produces valid WebDataset format."""
    sample = atdata.DictSample(name="test", value=123)
    wds_dict = sample.as_wds

    assert "__key__" in wds_dict
    assert "msgpack" in wds_dict
    assert isinstance(wds_dict["msgpack"], bytes)


def test_dictsample_repr():
    """Test DictSample has a useful repr."""
    sample = atdata.DictSample(name="test", value=42)
    repr_str = repr(sample)

    assert "DictSample" in repr_str
    assert "name" in repr_str
    assert "value" in repr_str


def test_dictsample_dataset_iteration(tmp_path):
    """Test Dataset[DictSample] can iterate over data."""

    # Create typed sample data
    @atdata.packable
    class SourceSample:
        text: str
        label: int

    wds_filename = (tmp_path / "dictsample_test.tar").as_posix()
    with wds.writer.TarWriter(wds_filename) as sink:
        for i in range(5):
            sample = SourceSample(text=f"item_{i}", label=i)
            sink.write(sample.as_wds)

    # Read as DictSample
    dataset = atdata.Dataset[atdata.DictSample](wds_filename)

    samples = list(dataset.ordered())
    assert len(samples) == 5

    for i, sample in enumerate(samples):
        assert isinstance(sample, atdata.DictSample)
        assert sample.text == f"item_{i}"
        assert sample["label"] == i


def test_dictsample_to_typed_via_as_type(tmp_path):
    """Test converting DictSample dataset to typed via as_type."""

    @atdata.packable
    class TypedSample:
        text: str
        label: int

    # Create data using typed sample
    wds_filename = (tmp_path / "astype_test.tar").as_posix()
    with wds.writer.TarWriter(wds_filename) as sink:
        for i in range(5):
            sample = TypedSample(text=f"item_{i}", label=i)
            sink.write(sample.as_wds)

    # Load as DictSample first
    ds_dict = atdata.Dataset[atdata.DictSample](wds_filename)

    # Convert to typed
    ds_typed = ds_dict.as_type(TypedSample)

    # Verify typed iteration works
    samples = list(ds_typed.ordered())
    assert len(samples) == 5

    for i, sample in enumerate(samples):
        assert isinstance(sample, TypedSample)
        assert sample.text == f"item_{i}"
        assert sample.label == i


def test_packable_auto_registers_dictsample_lens():
    """Test @packable decorator auto-registers lens from DictSample."""

    @atdata.packable
    class AutoLensSample:
        name: str
        value: int

    # The lens should be registered automatically
    network = atdata.LensNetwork()
    lens = network.transform(atdata.DictSample, AutoLensSample)

    # Test the lens works
    dict_sample = atdata.DictSample(name="test", value=42)
    typed_sample = lens(dict_sample)

    assert isinstance(typed_sample, AutoLensSample)
    assert typed_sample.name == "test"
    assert typed_sample.value == 42


def test_dictsample_batched_iteration(tmp_path):
    """Test Dataset[DictSample] works with batched iteration."""

    @atdata.packable
    class BatchSource:
        text: str
        value: int

    wds_filename = (tmp_path / "batch_dictsample_test.tar").as_posix()
    with wds.writer.TarWriter(wds_filename) as sink:
        for i in range(10):
            sample = BatchSource(text=f"item_{i}", value=i)
            sink.write(sample.as_wds)

    # Read as DictSample with batching
    dataset = atdata.Dataset[atdata.DictSample](wds_filename)

    batch_count = 0
    for batch in dataset.ordered(batch_size=4):
        assert isinstance(batch, atdata.SampleBatch)
        assert len(batch.samples) <= 4
        for sample in batch.samples:
            assert isinstance(sample, atdata.DictSample)
        batch_count += 1

    assert batch_count == 3  # 10 samples / 4 per batch = 2 full + 1 partial


##
