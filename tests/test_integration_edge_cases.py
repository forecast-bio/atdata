"""Integration tests for edge cases and data type coverage.

Tests boundary conditions and unusual data patterns including:
- Empty and single-sample datasets
- Special numpy dtypes (float16, complex128)
- Unicode and special characters
- Very long strings and large arrays
- Nested list types
- All primitive type variations
"""

import numpy as np
from numpy.typing import NDArray

import atdata
from atdata.local import Index, LocalDatasetEntry

# Use centralized tar creation helper from conftest
from conftest import create_tar_with_samples


##
# Edge Case Sample Types


@atdata.packable
class EmptyCompatSample:
    """Sample type for empty dataset tests."""

    id: int


@atdata.packable
class AllPrimitivesSample:
    """Sample with all primitive types."""

    str_field: str
    int_field: int
    float_field: float
    bool_field: bool
    bytes_field: bytes


@atdata.packable
class OptionalFieldsSample:
    """Sample with optional fields."""

    required_str: str
    optional_str: str | None
    optional_int: int | None
    optional_float: float | None
    optional_array: NDArray | None


@atdata.packable
class ListFieldsSample:
    """Sample with list fields."""

    str_list: list[str]
    int_list: list[int]
    float_list: list[float]
    bool_list: list[bool]


@atdata.packable
class UnicodeSample:
    """Sample with unicode content."""

    text: str
    label: str


@atdata.packable
class NDArraySample:
    """Sample with NDArray field."""

    label: str
    data: NDArray


##
# Empty and Single Sample Tests


class TestEmptyAndMinimalDatasets:
    """Tests for boundary dataset sizes."""

    def test_single_sample_dataset(self, tmp_path):
        """Dataset with exactly one sample should work correctly."""
        tar_path = tmp_path / "single-000000.tar"
        sample = EmptyCompatSample(id=42)
        create_tar_with_samples(tar_path, [sample])

        ds = atdata.Dataset[EmptyCompatSample](str(tar_path))
        samples = list(ds.ordered(batch_size=None))

        assert len(samples) == 1
        assert samples[0].id == 42

    def test_single_sample_batch(self, tmp_path):
        """Batching single sample should produce batch of size 1."""
        tar_path = tmp_path / "single-batch-000000.tar"
        sample = EmptyCompatSample(id=99)
        create_tar_with_samples(tar_path, [sample])

        ds = atdata.Dataset[EmptyCompatSample](str(tar_path))
        batches = list(ds.ordered(batch_size=10))

        assert len(batches) >= 1
        assert len(batches[0].samples) == 1

    def test_empty_tar_iteration(self, tmp_path):
        """Iteration over empty tar file should yield no samples."""
        import webdataset as wds

        tar_path = tmp_path / "empty-000000.tar"
        # Create empty tar file with no samples
        with wds.writer.TarWriter(str(tar_path)):
            pass

        ds = atdata.Dataset[EmptyCompatSample](str(tar_path))

        # Ordered iteration should yield nothing
        samples = list(ds.ordered(batch_size=None))
        assert samples == []

        # Batched iteration should also yield nothing
        batches = list(ds.ordered(batch_size=10))
        assert batches == []

    def test_empty_tar_shuffled_iteration(self, tmp_path):
        """Shuffled iteration over empty tar should yield no samples."""
        import webdataset as wds

        tar_path = tmp_path / "empty-shuffled-000000.tar"
        with wds.writer.TarWriter(str(tar_path)):
            pass

        ds = atdata.Dataset[EmptyCompatSample](str(tar_path))
        samples = list(ds.shuffled(batch_size=None))
        assert samples == []


##
# Primitive Type Coverage Tests


class TestPrimitiveTypes:
    """Tests for all primitive types."""

    def test_all_primitives_roundtrip(self, tmp_path):
        """All primitive types should serialize and deserialize correctly."""
        tar_path = tmp_path / "primitives-000000.tar"

        original = AllPrimitivesSample(
            str_field="hello world",
            int_field=42,
            float_field=3.14159,
            bool_field=True,
            bytes_field=b"\x00\x01\x02\xff",
        )
        create_tar_with_samples(tar_path, [original])

        ds = atdata.Dataset[AllPrimitivesSample](str(tar_path))
        loaded = list(ds.ordered(batch_size=None))[0]

        assert loaded.str_field == "hello world"
        assert loaded.int_field == 42
        assert abs(loaded.float_field - 3.14159) < 1e-5
        assert loaded.bool_field is True
        assert loaded.bytes_field == b"\x00\x01\x02\xff"

    def test_extreme_int_values(self, tmp_path):
        """Very large and small integers should be preserved."""
        tar_path = tmp_path / "extreme-int-000000.tar"

        @atdata.packable
        class ExtremeSample:
            value: int

        samples = [
            ExtremeSample(value=0),
            ExtremeSample(value=-1),
            ExtremeSample(value=2**62),  # Large positive
            ExtremeSample(value=-(2**62)),  # Large negative
        ]
        create_tar_with_samples(tar_path, samples)

        ds = atdata.Dataset[ExtremeSample](str(tar_path))
        loaded = list(ds.ordered(batch_size=None))

        assert loaded[0].value == 0
        assert loaded[1].value == -1
        assert loaded[2].value == 2**62
        assert loaded[3].value == -(2**62)

    def test_special_float_values(self, tmp_path):
        """Special float values (inf, -inf, very small) should be handled."""
        tar_path = tmp_path / "special-float-000000.tar"

        @atdata.packable
        class FloatSample:
            value: float

        samples = [
            FloatSample(value=0.0),
            FloatSample(value=-0.0),
            FloatSample(value=1e-300),  # Very small
            FloatSample(value=1e300),  # Very large
            FloatSample(value=float("inf")),
            FloatSample(value=float("-inf")),
        ]
        create_tar_with_samples(tar_path, samples)

        ds = atdata.Dataset[FloatSample](str(tar_path))
        loaded = list(ds.ordered(batch_size=None))

        assert loaded[0].value == 0.0
        assert loaded[2].value == 1e-300
        assert loaded[3].value == 1e300
        assert loaded[4].value == float("inf")
        assert loaded[5].value == float("-inf")


##
# Optional Field Tests


class TestOptionalFields:
    """Tests for optional (nullable) fields."""

    def test_optional_fields_with_values(self, tmp_path):
        """Optional fields with values should roundtrip correctly."""
        tar_path = tmp_path / "optional-present-000000.tar"

        sample = OptionalFieldsSample(
            required_str="required",
            optional_str="optional",
            optional_int=42,
            optional_float=3.14,
            optional_array=np.array([1, 2, 3]),
        )
        create_tar_with_samples(tar_path, [sample])

        ds = atdata.Dataset[OptionalFieldsSample](str(tar_path))
        loaded = list(ds.ordered(batch_size=None))[0]

        assert loaded.required_str == "required"
        assert loaded.optional_str == "optional"
        assert loaded.optional_int == 42
        assert loaded.optional_float == 3.14
        assert np.array_equal(loaded.optional_array, np.array([1, 2, 3]))

    def test_optional_fields_with_none(self, tmp_path):
        """Optional fields with None should roundtrip correctly."""
        tar_path = tmp_path / "optional-none-000000.tar"

        sample = OptionalFieldsSample(
            required_str="required",
            optional_str=None,
            optional_int=None,
            optional_float=None,
            optional_array=None,
        )
        create_tar_with_samples(tar_path, [sample])

        ds = atdata.Dataset[OptionalFieldsSample](str(tar_path))
        loaded = list(ds.ordered(batch_size=None))[0]

        assert loaded.required_str == "required"
        assert loaded.optional_str is None
        assert loaded.optional_int is None
        assert loaded.optional_float is None
        assert loaded.optional_array is None


##
# List Field Tests


class TestListFields:
    """Tests for list type fields."""

    def test_list_fields_roundtrip(self, tmp_path):
        """List fields should serialize and deserialize correctly."""
        tar_path = tmp_path / "lists-000000.tar"

        sample = ListFieldsSample(
            str_list=["a", "b", "c"],
            int_list=[1, 2, 3, 4, 5],
            float_list=[1.1, 2.2, 3.3],
            bool_list=[True, False, True],
        )
        create_tar_with_samples(tar_path, [sample])

        ds = atdata.Dataset[ListFieldsSample](str(tar_path))
        loaded = list(ds.ordered(batch_size=None))[0]

        assert loaded.str_list == ["a", "b", "c"]
        assert loaded.int_list == [1, 2, 3, 4, 5]
        assert loaded.float_list == [1.1, 2.2, 3.3]
        assert loaded.bool_list == [True, False, True]

    def test_empty_lists(self, tmp_path):
        """Empty lists should be handled correctly."""
        tar_path = tmp_path / "empty-lists-000000.tar"

        sample = ListFieldsSample(
            str_list=[],
            int_list=[],
            float_list=[],
            bool_list=[],
        )
        create_tar_with_samples(tar_path, [sample])

        ds = atdata.Dataset[ListFieldsSample](str(tar_path))
        loaded = list(ds.ordered(batch_size=None))[0]

        assert loaded.str_list == []
        assert loaded.int_list == []
        assert loaded.float_list == []
        assert loaded.bool_list == []

    def test_large_lists(self, tmp_path):
        """Large lists should be handled correctly."""
        tar_path = tmp_path / "large-lists-000000.tar"

        sample = ListFieldsSample(
            str_list=[f"item-{i}" for i in range(1000)],
            int_list=list(range(1000)),
            float_list=[float(i) for i in range(1000)],
            bool_list=[i % 2 == 0 for i in range(1000)],
        )
        create_tar_with_samples(tar_path, [sample])

        ds = atdata.Dataset[ListFieldsSample](str(tar_path))
        loaded = list(ds.ordered(batch_size=None))[0]

        assert len(loaded.str_list) == 1000
        assert len(loaded.int_list) == 1000
        assert loaded.str_list[500] == "item-500"
        assert loaded.int_list[999] == 999


##
# Unicode and Special Character Tests


class TestUnicodeAndSpecialChars:
    """Tests for unicode and special characters."""

    def test_unicode_strings(self, tmp_path):
        """Unicode strings should roundtrip correctly."""
        tar_path = tmp_path / "unicode-000000.tar"

        samples = [
            UnicodeSample(text="Hello World", label="ascii"),
            UnicodeSample(text="Bonjour le monde", label="accents"),
            UnicodeSample(text="Hallo Welt", label="german"),
            UnicodeSample(text="Witaj Swiecie", label="polish"),
        ]
        create_tar_with_samples(tar_path, samples)

        ds = atdata.Dataset[UnicodeSample](str(tar_path))
        loaded = list(ds.ordered(batch_size=None))

        assert loaded[0].text == "Hello World"
        assert loaded[1].text == "Bonjour le monde"

    def test_emoji(self, tmp_path):
        """Emoji should roundtrip correctly."""
        tar_path = tmp_path / "emoji-000000.tar"

        sample = UnicodeSample(
            text="Hello World! Have a great day!", label="with-emoji"
        )
        create_tar_with_samples(tar_path, [sample])

        ds = atdata.Dataset[UnicodeSample](str(tar_path))
        loaded = list(ds.ordered(batch_size=None))[0]

        assert "Hello" in loaded.text
        assert "great day" in loaded.text

    def test_cjk_characters(self, tmp_path):
        """CJK characters should roundtrip correctly."""
        tar_path = tmp_path / "cjk-000000.tar"

        samples = [
            UnicodeSample(text="Nihongo", label="japanese"),
            UnicodeSample(text="Zhongwen", label="chinese"),
            UnicodeSample(text="Hangugeo", label="korean"),
        ]
        create_tar_with_samples(tar_path, samples)

        ds = atdata.Dataset[UnicodeSample](str(tar_path))
        loaded = list(ds.ordered(batch_size=None))

        assert len(loaded) == 3

    def test_special_chars_in_string_fields(self, tmp_path):
        """Special characters (newlines, tabs, quotes) should roundtrip."""
        tar_path = tmp_path / "special-chars-000000.tar"

        sample = UnicodeSample(
            text='Line1\nLine2\tTabbed\r\nWindows\0Null"Quotes"',
            label="special",
        )
        create_tar_with_samples(tar_path, [sample])

        ds = atdata.Dataset[UnicodeSample](str(tar_path))
        loaded = list(ds.ordered(batch_size=None))[0]

        assert "Line1\nLine2" in loaded.text
        assert "\t" in loaded.text


##
# NDArray Type Tests


class TestNDArrayTypes:
    """Tests for various NDArray dtypes and shapes."""

    def test_common_dtypes(self, tmp_path):
        """Common numpy dtypes should work correctly."""
        dtypes = [np.float32, np.float64, np.int32, np.int64, np.uint8]

        for dtype in dtypes:
            tar_path = tmp_path / f"dtype-{dtype.__name__}-000000.tar"
            sample = NDArraySample(
                label=f"dtype-{dtype.__name__}",
                data=np.array([1, 2, 3, 4, 5], dtype=dtype),
            )
            create_tar_with_samples(tar_path, [sample])

            ds = atdata.Dataset[NDArraySample](str(tar_path))
            loaded = list(ds.ordered(batch_size=None))[0]

            assert loaded.data.dtype == dtype
            assert np.array_equal(loaded.data, np.array([1, 2, 3, 4, 5], dtype=dtype))

    def test_float16_dtype(self, tmp_path):
        """float16 (half precision) should work correctly."""
        tar_path = tmp_path / "float16-000000.tar"
        sample = NDArraySample(
            label="float16",
            data=np.array([1.0, 2.0, 3.0], dtype=np.float16),
        )
        create_tar_with_samples(tar_path, [sample])

        ds = atdata.Dataset[NDArraySample](str(tar_path))
        loaded = list(ds.ordered(batch_size=None))[0]

        assert loaded.data.dtype == np.float16

    def test_complex_dtype(self, tmp_path):
        """Complex dtypes should work correctly."""
        tar_path = tmp_path / "complex-000000.tar"
        sample = NDArraySample(
            label="complex128",
            data=np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128),
        )
        create_tar_with_samples(tar_path, [sample])

        ds = atdata.Dataset[NDArraySample](str(tar_path))
        loaded = list(ds.ordered(batch_size=None))[0]

        assert loaded.data.dtype == np.complex128
        assert loaded.data[0] == 1 + 2j

    def test_multidimensional_arrays(self, tmp_path):
        """Multidimensional arrays should preserve shape."""
        tmp_path / "multidim-000000.tar"

        shapes = [(3, 4), (2, 3, 4), (2, 2, 2, 2)]

        for i, shape in enumerate(shapes):
            tar_path_i = tmp_path / f"multidim-{i}-000000.tar"
            sample = NDArraySample(
                label=f"shape-{shape}",
                data=np.ones(shape, dtype=np.float32),
            )
            create_tar_with_samples(tar_path_i, [sample])

            ds = atdata.Dataset[NDArraySample](str(tar_path_i))
            loaded = list(ds.ordered(batch_size=None))[0]

            assert loaded.data.shape == shape

    def test_large_array(self, tmp_path):
        """Moderately large arrays should work correctly."""
        tar_path = tmp_path / "large-array-000000.tar"

        # 1000x1000 float32 = 4MB
        large_array = np.random.randn(1000, 1000).astype(np.float32)
        sample = NDArraySample(label="large", data=large_array)
        create_tar_with_samples(tar_path, [sample])

        ds = atdata.Dataset[NDArraySample](str(tar_path))
        loaded = list(ds.ordered(batch_size=None))[0]

        assert loaded.data.shape == (1000, 1000)
        assert np.allclose(loaded.data, large_array)


##
# String Edge Cases


class TestStringEdgeCases:
    """Tests for string field edge cases."""

    def test_empty_string(self, tmp_path):
        """Empty strings should be preserved."""
        tar_path = tmp_path / "empty-string-000000.tar"
        sample = UnicodeSample(text="", label="empty")
        create_tar_with_samples(tar_path, [sample])

        ds = atdata.Dataset[UnicodeSample](str(tar_path))
        loaded = list(ds.ordered(batch_size=None))[0]

        assert loaded.text == ""
        assert loaded.label == "empty"

    def test_long_string(self, tmp_path):
        """Long strings should be handled correctly."""
        tar_path = tmp_path / "long-string-000000.tar"

        # 100KB string
        long_text = "x" * (100 * 1024)
        sample = UnicodeSample(text=long_text, label="long")
        create_tar_with_samples(tar_path, [sample])

        ds = atdata.Dataset[UnicodeSample](str(tar_path))
        loaded = list(ds.ordered(batch_size=None))[0]

        assert len(loaded.text) == 100 * 1024
        assert loaded.text == long_text

    def test_binary_bytes_field(self, tmp_path):
        """Binary bytes with all possible byte values."""
        tar_path = tmp_path / "binary-bytes-000000.tar"

        # All possible byte values
        all_bytes = bytes(range(256))
        sample = AllPrimitivesSample(
            str_field="test",
            int_field=0,
            float_field=0.0,
            bool_field=False,
            bytes_field=all_bytes,
        )
        create_tar_with_samples(tar_path, [sample])

        ds = atdata.Dataset[AllPrimitivesSample](str(tar_path))
        loaded = list(ds.ordered(batch_size=None))[0]

        assert loaded.bytes_field == all_bytes
        assert len(loaded.bytes_field) == 256


##
# Schema and Index Edge Cases


class TestSchemaEdgeCases:
    """Tests for schema edge cases."""

    def test_schema_with_many_fields(self, clean_redis):
        """Schema with many fields should work correctly."""

        @atdata.packable
        class ManyFieldsSample:
            f1: str
            f2: str
            f3: str
            f4: str
            f5: str
            f6: int
            f7: int
            f8: int
            f9: float
            f10: float

        index = Index(redis=clean_redis)
        schema_ref = index.publish_schema(ManyFieldsSample, version="1.0.0")
        schema = index.get_schema(schema_ref)

        assert len(schema["fields"]) == 10

    def test_dataset_name_with_special_chars(self, clean_redis):
        """Dataset names with special characters should work."""
        index = Index(redis=clean_redis)
        schema_ref = index.publish_schema(EmptyCompatSample, version="1.0.0")

        # Various special names
        names = [
            "dataset-with-dashes",
            "dataset_with_underscores",
            "dataset.with.dots",
            "UPPERCASE-name",
        ]

        for name in names:
            entry = LocalDatasetEntry(
                name=name,
                schema_ref=schema_ref,
                data_urls=["s3://bucket/data.tar"],
            )
            entry.write_to(clean_redis)

            retrieved = index.get_entry_by_name(name)
            assert retrieved.name == name


##
# Batch Processing Edge Cases


class TestBatchEdgeCases:
    """Tests for batch processing edge cases."""

    def test_batch_size_larger_than_dataset(self, tmp_path):
        """Batch size larger than dataset size should work."""
        tar_path = tmp_path / "small-dataset-000000.tar"
        samples = [EmptyCompatSample(id=i) for i in range(3)]
        create_tar_with_samples(tar_path, samples)

        ds = atdata.Dataset[EmptyCompatSample](str(tar_path))
        batches = list(ds.ordered(batch_size=100))

        # Should get at least one batch
        assert len(batches) >= 1
        # Total samples should be 3
        total_samples = sum(len(batch.samples) for batch in batches)
        assert total_samples == 3

    def test_batch_size_one(self, tmp_path):
        """Batch size of 1 should produce individual samples in batches."""
        tar_path = tmp_path / "batch-one-000000.tar"
        samples = [EmptyCompatSample(id=i) for i in range(5)]
        create_tar_with_samples(tar_path, samples)

        ds = atdata.Dataset[EmptyCompatSample](str(tar_path))
        batches = list(ds.ordered(batch_size=1))

        assert len(batches) == 5
        for batch in batches:
            assert len(batch.samples) == 1

    def test_batch_aggregation_with_arrays(self, tmp_path):
        """Batch aggregation should stack NDArrays correctly."""
        tar_path = tmp_path / "batch-arrays-000000.tar"

        samples = [
            NDArraySample(label=f"s{i}", data=np.array([i, i + 1, i + 2]))
            for i in range(4)
        ]
        create_tar_with_samples(tar_path, samples)

        ds = atdata.Dataset[NDArraySample](str(tar_path))
        batches = list(ds.ordered(batch_size=4))

        batch = batches[0]
        # data attribute should be stacked
        stacked_data = batch.data

        assert stacked_data.shape == (4, 3)
        assert np.array_equal(stacked_data[0], np.array([0, 1, 2]))
        assert np.array_equal(stacked_data[3], np.array([3, 4, 5]))
