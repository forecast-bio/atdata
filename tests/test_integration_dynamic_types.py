"""Integration tests for dynamic type loading from schemas.

Tests the schema_to_type() functionality for:
- Schema → Type reconstruction
- Reconstructed types working with Dataset
- Complex field types (NDArray, optional, lists)
- Type caching behavior
- Schema from different sources (local, atmosphere)
"""

import pytest
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
import webdataset as wds

import atdata
from atdata._schema_codec import (
    schema_to_type,
    generate_stub,
    clear_type_cache,
    get_cached_types,
)
import atdata.local as atlocal


##
# Test sample types for comparison


@dataclass
class SimpleSample(atdata.PackableSample):
    """Simple sample for testing."""

    name: str
    value: int
    score: float


@dataclass
class ArraySample(atdata.PackableSample):
    """Sample with NDArray field."""

    label: str
    image: NDArray


@dataclass
class OptionalSample(atdata.PackableSample):
    """Sample with optional fields."""

    name: str
    value: int
    extra: str | None = None
    embedding: NDArray | None = None


@dataclass
class ListSample(atdata.PackableSample):
    """Sample with list fields."""

    tags: list[str]
    scores: list[float]


##
# Fixtures


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear type cache before each test."""
    clear_type_cache()
    yield
    clear_type_cache()


##
# Basic Schema to Type Tests


class TestSchemaToType:
    """Tests for basic schema_to_type functionality."""

    def test_simple_primitive_schema(self):
        """Schema with primitive fields should produce usable type."""
        schema = {
            "name": "SimpleSample",
            "version": "1.0.0",
            "fields": [
                {
                    "name": "name",
                    "fieldType": {"$type": "local#primitive", "primitive": "str"},
                    "optional": False,
                },
                {
                    "name": "value",
                    "fieldType": {"$type": "local#primitive", "primitive": "int"},
                    "optional": False,
                },
                {
                    "name": "score",
                    "fieldType": {"$type": "local#primitive", "primitive": "float"},
                    "optional": False,
                },
            ],
        }

        SampleType = schema_to_type(schema)

        # Should be able to create instances
        instance = SampleType(name="test", value=42, score=0.5)
        assert instance.name == "test"
        assert instance.value == 42
        assert instance.score == 0.5

    def test_ndarray_field_schema(self):
        """Schema with NDArray field should produce working type."""
        schema = {
            "name": "ArraySample",
            "version": "1.0.0",
            "fields": [
                {
                    "name": "label",
                    "fieldType": {"$type": "local#primitive", "primitive": "str"},
                    "optional": False,
                },
                {
                    "name": "image",
                    "fieldType": {"$type": "local#ndarray", "dtype": "float32"},
                    "optional": False,
                },
            ],
        }

        SampleType = schema_to_type(schema)

        # Should work with numpy arrays
        arr = np.random.randn(32, 32).astype(np.float32)
        instance = SampleType(label="test", image=arr)
        assert instance.label == "test"
        np.testing.assert_array_equal(instance.image, arr)

    def test_optional_field_schema(self):
        """Schema with optional fields should use None as default."""
        schema = {
            "name": "OptionalSample",
            "version": "1.0.0",
            "fields": [
                {
                    "name": "name",
                    "fieldType": {"$type": "local#primitive", "primitive": "str"},
                    "optional": False,
                },
                {
                    "name": "extra",
                    "fieldType": {"$type": "local#primitive", "primitive": "str"},
                    "optional": True,
                },
            ],
        }

        SampleType = schema_to_type(schema)

        # Optional field should default to None
        instance = SampleType(name="test")
        assert instance.name == "test"
        assert instance.extra is None

        # Can also provide value
        instance2 = SampleType(name="test", extra="optional")
        assert instance2.extra == "optional"

    def test_list_field_schema(self):
        """Schema with list fields should produce working type."""
        schema = {
            "name": "ListSample",
            "version": "1.0.0",
            "fields": [
                {
                    "name": "tags",
                    "fieldType": {
                        "$type": "local#array",
                        "items": {"$type": "local#primitive", "primitive": "str"},
                    },
                    "optional": False,
                },
                {
                    "name": "scores",
                    "fieldType": {
                        "$type": "local#array",
                        "items": {"$type": "local#primitive", "primitive": "float"},
                    },
                    "optional": False,
                },
            ],
        }

        SampleType = schema_to_type(schema)

        instance = SampleType(tags=["a", "b", "c"], scores=[1.0, 2.0, 3.0])
        assert instance.tags == ["a", "b", "c"]
        assert instance.scores == [1.0, 2.0, 3.0]

    def test_all_primitive_types(self):
        """All primitive types should be supported."""
        schema = {
            "name": "AllPrimitives",
            "version": "1.0.0",
            "fields": [
                {
                    "name": "s",
                    "fieldType": {"$type": "local#primitive", "primitive": "str"},
                    "optional": False,
                },
                {
                    "name": "i",
                    "fieldType": {"$type": "local#primitive", "primitive": "int"},
                    "optional": False,
                },
                {
                    "name": "f",
                    "fieldType": {"$type": "local#primitive", "primitive": "float"},
                    "optional": False,
                },
                {
                    "name": "b",
                    "fieldType": {"$type": "local#primitive", "primitive": "bool"},
                    "optional": False,
                },
                {
                    "name": "raw",
                    "fieldType": {"$type": "local#primitive", "primitive": "bytes"},
                    "optional": False,
                },
            ],
        }

        SampleType = schema_to_type(schema)

        instance = SampleType(s="hello", i=42, f=3.14, b=True, raw=b"bytes")
        assert instance.s == "hello"
        assert instance.i == 42
        assert instance.f == 3.14
        assert instance.b is True
        assert instance.raw == b"bytes"


class TestDynamicTypeWithDataset:
    """Tests for using dynamically generated types with Dataset."""

    def test_load_dataset_with_dynamic_type(self, tmp_path):
        """Dynamic type should work with Dataset loading."""
        # First, create a dataset with a known type
        tar_path = tmp_path / "data.tar"
        original_samples = [
            SimpleSample(name=f"item_{i}", value=i * 10, score=float(i) * 0.5)
            for i in range(10)
        ]

        with wds.writer.TarWriter(str(tar_path)) as sink:
            for sample in original_samples:
                sink.write(sample.as_wds)

        # Now create the type dynamically
        schema = {
            "name": "SimpleSample",
            "version": "1.0.0",
            "fields": [
                {
                    "name": "name",
                    "fieldType": {"$type": "local#primitive", "primitive": "str"},
                    "optional": False,
                },
                {
                    "name": "value",
                    "fieldType": {"$type": "local#primitive", "primitive": "int"},
                    "optional": False,
                },
                {
                    "name": "score",
                    "fieldType": {"$type": "local#primitive", "primitive": "float"},
                    "optional": False,
                },
            ],
        }

        DynamicType = schema_to_type(schema)

        # Load with dynamic type
        dataset = atdata.Dataset[DynamicType](str(tar_path))
        loaded = list(dataset.ordered(batch_size=None))

        assert len(loaded) == 10
        for i, sample in enumerate(loaded):
            assert sample.name == f"item_{i}"
            assert sample.value == i * 10
            assert sample.score == float(i) * 0.5

    def test_load_dataset_with_ndarray_dynamic_type(self, tmp_path):
        """Dynamic type with NDArray should deserialize correctly."""
        # Create dataset
        tar_path = tmp_path / "array_data.tar"
        original_arrays = [np.random.randn(16, 16).astype(np.float32) for _ in range(5)]

        with wds.writer.TarWriter(str(tar_path)) as sink:
            for i, arr in enumerate(original_arrays):
                sample = ArraySample(label=f"arr_{i}", image=arr)
                sink.write(sample.as_wds)

        # Dynamic type
        schema = {
            "name": "ArraySample",
            "version": "1.0.0",
            "fields": [
                {
                    "name": "label",
                    "fieldType": {"$type": "local#primitive", "primitive": "str"},
                    "optional": False,
                },
                {
                    "name": "image",
                    "fieldType": {"$type": "local#ndarray", "dtype": "float32"},
                    "optional": False,
                },
            ],
        }

        DynamicType = schema_to_type(schema)
        dataset = atdata.Dataset[DynamicType](str(tar_path))
        loaded = list(dataset.ordered(batch_size=None))

        assert len(loaded) == 5
        for i, sample in enumerate(loaded):
            assert sample.label == f"arr_{i}"
            np.testing.assert_array_almost_equal(sample.image, original_arrays[i])

    def test_batch_iteration_with_dynamic_type(self, tmp_path):
        """Batching should work with dynamic types."""
        tar_path = tmp_path / "batch_data.tar"

        with wds.writer.TarWriter(str(tar_path)) as sink:
            for i in range(20):
                sample = SimpleSample(name=f"item_{i}", value=i, score=float(i))
                sink.write(sample.as_wds)

        schema = {
            "name": "SimpleSample",
            "version": "1.0.0",
            "fields": [
                {
                    "name": "name",
                    "fieldType": {"$type": "local#primitive", "primitive": "str"},
                    "optional": False,
                },
                {
                    "name": "value",
                    "fieldType": {"$type": "local#primitive", "primitive": "int"},
                    "optional": False,
                },
                {
                    "name": "score",
                    "fieldType": {"$type": "local#primitive", "primitive": "float"},
                    "optional": False,
                },
            ],
        }

        DynamicType = schema_to_type(schema)
        dataset = atdata.Dataset[DynamicType](str(tar_path))

        batches = list(dataset.ordered(batch_size=5))
        assert len(batches) == 4

        for batch in batches:
            assert isinstance(batch, atdata.SampleBatch)
            assert len(batch.samples) == 5


class TestTypeCaching:
    """Tests for type caching behavior."""

    def test_same_schema_returns_cached_type(self):
        """Identical schemas should return same cached type."""
        schema = {
            "name": "CachedSample",
            "version": "1.0.0",
            "fields": [
                {
                    "name": "value",
                    "fieldType": {"$type": "local#primitive", "primitive": "int"},
                    "optional": False,
                },
            ],
        }

        Type1 = schema_to_type(schema)
        Type2 = schema_to_type(schema)

        assert Type1 is Type2

    def test_different_version_different_type(self):
        """Different version should produce different type."""
        schema1 = {
            "name": "VersionedSample",
            "version": "1.0.0",
            "fields": [
                {
                    "name": "value",
                    "fieldType": {"$type": "local#primitive", "primitive": "int"},
                    "optional": False,
                },
            ],
        }
        schema2 = {
            "name": "VersionedSample",
            "version": "2.0.0",
            "fields": [
                {
                    "name": "value",
                    "fieldType": {"$type": "local#primitive", "primitive": "int"},
                    "optional": False,
                },
            ],
        }

        Type1 = schema_to_type(schema1)
        Type2 = schema_to_type(schema2)

        # Different versions = different types
        assert Type1 is not Type2

    def test_different_fields_different_type(self):
        """Different fields should produce different type."""
        schema1 = {
            "name": "FieldSample",
            "version": "1.0.0",
            "fields": [
                {
                    "name": "a",
                    "fieldType": {"$type": "local#primitive", "primitive": "int"},
                    "optional": False,
                },
            ],
        }
        schema2 = {
            "name": "FieldSample",
            "version": "1.0.0",
            "fields": [
                {
                    "name": "b",
                    "fieldType": {"$type": "local#primitive", "primitive": "int"},
                    "optional": False,
                },
            ],
        }

        Type1 = schema_to_type(schema1)
        Type2 = schema_to_type(schema2)

        assert Type1 is not Type2

    def test_use_cache_false_bypasses_cache(self):
        """use_cache=False should always create new type."""
        schema = {
            "name": "NoCacheSample",
            "version": "1.0.0",
            "fields": [
                {
                    "name": "value",
                    "fieldType": {"$type": "local#primitive", "primitive": "int"},
                    "optional": False,
                },
            ],
        }

        Type1 = schema_to_type(schema, use_cache=False)
        Type2 = schema_to_type(schema, use_cache=False)

        # Without cache, each call creates new type
        assert Type1 is not Type2

    def test_clear_cache_removes_types(self):
        """clear_type_cache should remove all cached types."""
        schema = {
            "name": "ClearableSample",
            "version": "1.0.0",
            "fields": [
                {
                    "name": "value",
                    "fieldType": {"$type": "local#primitive", "primitive": "int"},
                    "optional": False,
                },
            ],
        }

        Type1 = schema_to_type(schema)
        clear_type_cache()
        Type2 = schema_to_type(schema)

        # After clear, should get new type
        assert Type1 is not Type2

    def test_get_cached_types_returns_cache_copy(self):
        """get_cached_types should return cache contents."""
        schema = {
            "name": "TrackedSample",
            "version": "1.0.0",
            "fields": [
                {
                    "name": "value",
                    "fieldType": {"$type": "local#primitive", "primitive": "int"},
                    "optional": False,
                },
            ],
        }

        schema_to_type(schema)
        cache = get_cached_types()

        assert len(cache) == 1
        assert "TrackedSample" in list(cache.keys())[0]


class TestSchemaFromIndex:
    """Tests for loading types from Index schemas."""

    def test_publish_then_decode_schema(self, clean_redis):
        """Published schema should be decodable to usable type."""
        index = atlocal.Index(redis=clean_redis)

        # Publish a schema
        schema_ref = index.publish_schema(SimpleSample)

        # Decode it back
        ReconstructedType = index.decode_schema(schema_ref)

        # Should be usable
        instance = ReconstructedType(name="test", value=42, score=0.5)
        assert instance.name == "test"
        assert instance.value == 42

    def test_publish_ndarray_then_decode(self, clean_redis):
        """NDArray schema should decode correctly."""
        index = atlocal.Index(redis=clean_redis)

        schema_ref = index.publish_schema(ArraySample)
        ReconstructedType = index.decode_schema(schema_ref)

        arr = np.random.randn(8, 8).astype(np.float32)
        instance = ReconstructedType(label="test", image=arr)
        np.testing.assert_array_equal(instance.image, arr)

    def test_decoded_type_works_with_dataset(self, clean_redis, tmp_path):
        """Decoded type should work with Dataset iteration."""
        index = atlocal.Index(redis=clean_redis)

        # Create dataset with original type
        tar_path = tmp_path / "original.tar"
        with wds.writer.TarWriter(str(tar_path)) as sink:
            for i in range(5):
                sample = SimpleSample(name=f"s_{i}", value=i, score=float(i))
                sink.write(sample.as_wds)

        # Publish and decode schema
        schema_ref = index.publish_schema(SimpleSample)
        DecodedType = index.decode_schema(schema_ref)

        # Load with decoded type
        dataset = atdata.Dataset[DecodedType](str(tar_path))
        loaded = list(dataset.ordered(batch_size=None))

        assert len(loaded) == 5
        for i, sample in enumerate(loaded):
            assert sample.name == f"s_{i}"
            assert sample.value == i


class TestSchemaValidation:
    """Tests for schema validation and error handling."""

    def test_schema_without_name_raises(self):
        """Schema without name should raise ValueError."""
        schema = {
            "version": "1.0.0",
            "fields": [
                {
                    "name": "value",
                    "fieldType": {"$type": "local#primitive", "primitive": "int"},
                    "optional": False,
                },
            ],
        }

        with pytest.raises(ValueError, match="must have a 'name'"):
            schema_to_type(schema)

    def test_schema_without_fields_raises(self):
        """Schema without fields should raise ValueError."""
        schema = {"name": "EmptySample", "version": "1.0.0", "fields": []}

        with pytest.raises(ValueError, match="must have at least one field"):
            schema_to_type(schema)

    def test_field_without_name_raises(self):
        """Field without name should raise an error."""
        schema = {
            "name": "BadFieldSample",
            "version": "1.0.0",
            "fields": [
                {
                    "fieldType": {"$type": "local#primitive", "primitive": "int"},
                    "optional": False,
                },
            ],
        }

        # Raises KeyError during cache key generation or ValueError during field processing
        with pytest.raises((KeyError, ValueError)):
            schema_to_type(schema)

    def test_unknown_primitive_raises(self):
        """Unknown primitive type should raise ValueError."""
        schema = {
            "name": "UnknownPrimitive",
            "version": "1.0.0",
            "fields": [
                {
                    "name": "value",
                    "fieldType": {
                        "$type": "local#primitive",
                        "primitive": "complex128",
                    },
                    "optional": False,
                },
            ],
        }

        with pytest.raises(ValueError, match="Unknown primitive type"):
            schema_to_type(schema)

    def test_unknown_field_type_raises(self):
        """Unknown field type kind should raise ValueError."""
        schema = {
            "name": "UnknownType",
            "version": "1.0.0",
            "fields": [
                {
                    "name": "value",
                    "fieldType": {"$type": "local#custom"},
                    "optional": False,
                },
            ],
        }

        with pytest.raises(ValueError, match="Unknown field type kind"):
            schema_to_type(schema)


class TestComplexSchemaScenarios:
    """Complex integration scenarios with dynamic types."""

    def test_optional_ndarray_schema(self, tmp_path):
        """Optional NDArray field should handle None correctly."""
        schema = {
            "name": "OptionalArraySample",
            "version": "1.0.0",
            "fields": [
                {
                    "name": "name",
                    "fieldType": {"$type": "local#primitive", "primitive": "str"},
                    "optional": False,
                },
                {
                    "name": "embedding",
                    "fieldType": {"$type": "local#ndarray", "dtype": "float32"},
                    "optional": True,
                },
            ],
        }

        DynamicType = schema_to_type(schema)

        # Create dataset with some None values
        tar_path = tmp_path / "optional_array.tar"
        with wds.writer.TarWriter(str(tar_path)) as sink:
            for i in range(6):
                if i % 2 == 0:
                    sample = OptionalSample(
                        name=f"s_{i}", value=i, embedding=np.zeros(4, dtype=np.float32)
                    )
                else:
                    sample = OptionalSample(name=f"s_{i}", value=i, embedding=None)
                sink.write(sample.as_wds)

        # Note: The OptionalSample has different fields than DynamicType
        # This test verifies the dynamic type can be created, not cross-compatibility

        instance_with = DynamicType(name="test", embedding=np.zeros(4))
        instance_without = DynamicType(name="test")

        assert instance_with.embedding is not None
        assert instance_without.embedding is None

    def test_nested_list_schema(self):
        """Nested list types should work."""
        schema = {
            "name": "NestedListSample",
            "version": "1.0.0",
            "fields": [
                {
                    "name": "matrix",
                    "fieldType": {
                        "$type": "local#array",
                        "items": {
                            "$type": "local#array",
                            "items": {"$type": "local#primitive", "primitive": "int"},
                        },
                    },
                    "optional": False,
                },
            ],
        }

        DynamicType = schema_to_type(schema)

        instance = DynamicType(matrix=[[1, 2], [3, 4], [5, 6]])
        assert instance.matrix == [[1, 2], [3, 4], [5, 6]]

    def test_multiple_schemas_same_session(self, clean_redis):
        """Multiple different schemas should coexist."""
        index = atlocal.Index(redis=clean_redis)

        # Publish multiple schemas
        ref1 = index.publish_schema(SimpleSample, version="1.0.0")
        ref2 = index.publish_schema(ArraySample, version="1.0.0")
        ref3 = index.publish_schema(ListSample, version="1.0.0")

        # Decode all
        Type1 = index.decode_schema(ref1)
        Type2 = index.decode_schema(ref2)
        Type3 = index.decode_schema(ref3)

        # All should be usable
        assert Type1(name="a", value=1, score=0.5).name == "a"
        assert Type2(label="b", image=np.zeros(4)).label == "b"
        assert Type3(tags=["x"], scores=[1.0]).tags == ["x"]


class TestGenerateStub:
    """Tests for generate_stub() function for IDE support."""

    def test_basic_stub_generation(self):
        """Should generate valid stub content for simple schema."""
        schema = {
            "name": "SimpleSample",
            "version": "1.0.0",
            "fields": [
                {
                    "name": "name",
                    "fieldType": {"$type": "local#primitive", "primitive": "str"},
                    "optional": False,
                },
                {
                    "name": "value",
                    "fieldType": {"$type": "local#primitive", "primitive": "int"},
                    "optional": False,
                },
            ],
        }

        stub = generate_stub(schema)

        assert "class SimpleSample(PackableSample):" in stub
        assert "name: str" in stub
        assert "value: int" in stub
        assert "def __init__(self, name: str, value: int) -> None: ..." in stub

    def test_stub_with_ndarray_field(self):
        """Should generate NDArray type hint in stub."""
        schema = {
            "name": "ArraySample",
            "version": "1.0.0",
            "fields": [
                {
                    "name": "image",
                    "fieldType": {"$type": "local#ndarray", "dtype": "float32"},
                    "optional": False,
                },
            ],
        }

        stub = generate_stub(schema)

        assert "image: NDArray[Any]" in stub
        assert "from numpy.typing import NDArray" in stub

    def test_stub_with_optional_field(self):
        """Should generate optional type hint with default None."""
        schema = {
            "name": "OptionalSample",
            "version": "1.0.0",
            "fields": [
                {
                    "name": "name",
                    "fieldType": {"$type": "local#primitive", "primitive": "str"},
                    "optional": False,
                },
                {
                    "name": "extra",
                    "fieldType": {"$type": "local#primitive", "primitive": "str"},
                    "optional": True,
                },
            ],
        }

        stub = generate_stub(schema)

        assert "name: str" in stub
        assert "extra: str | None" in stub
        assert "extra: str | None = None" in stub

    def test_stub_with_list_field(self):
        """Should generate list type hint in stub."""
        schema = {
            "name": "ListSample",
            "version": "1.0.0",
            "fields": [
                {
                    "name": "tags",
                    "fieldType": {
                        "$type": "local#array",
                        "items": {"$type": "local#primitive", "primitive": "str"},
                    },
                    "optional": False,
                },
            ],
        }

        stub = generate_stub(schema)

        assert "tags: list[str]" in stub

    def test_stub_includes_header_comments(self):
        """Stub should include helpful header comments."""
        schema = {
            "name": "MySample",
            "version": "2.1.0",
            "fields": [
                {
                    "name": "value",
                    "fieldType": {"$type": "local#primitive", "primitive": "int"},
                    "optional": False,
                },
            ],
        }

        stub = generate_stub(schema)

        assert "# Auto-generated stub" in stub
        assert "# Schema: MySample@2.1.0" in stub
        assert "VS Code/Pylance" in stub or "PyCharm" in stub

    def test_stub_includes_imports(self):
        """Stub should include necessary imports."""
        schema = {
            "name": "ImportSample",
            "version": "1.0.0",
            "fields": [
                {
                    "name": "value",
                    "fieldType": {"$type": "local#primitive", "primitive": "int"},
                    "optional": False,
                },
            ],
        }

        stub = generate_stub(schema)

        assert "from typing import Any" in stub
        assert "from atdata import PackableSample" in stub

    def test_stub_all_primitive_types(self):
        """Should handle all primitive types correctly."""
        schema = {
            "name": "AllPrimitives",
            "version": "1.0.0",
            "fields": [
                {
                    "name": "s",
                    "fieldType": {"$type": "local#primitive", "primitive": "str"},
                    "optional": False,
                },
                {
                    "name": "i",
                    "fieldType": {"$type": "local#primitive", "primitive": "int"},
                    "optional": False,
                },
                {
                    "name": "f",
                    "fieldType": {"$type": "local#primitive", "primitive": "float"},
                    "optional": False,
                },
                {
                    "name": "b",
                    "fieldType": {"$type": "local#primitive", "primitive": "bool"},
                    "optional": False,
                },
                {
                    "name": "raw",
                    "fieldType": {"$type": "local#primitive", "primitive": "bytes"},
                    "optional": False,
                },
            ],
        }

        stub = generate_stub(schema)

        assert "s: str" in stub
        assert "i: int" in stub
        assert "f: float" in stub
        assert "b: bool" in stub
        assert "raw: bytes" in stub

    def test_stub_with_nested_list(self):
        """Should handle nested list types."""
        schema = {
            "name": "NestedSample",
            "version": "1.0.0",
            "fields": [
                {
                    "name": "matrix",
                    "fieldType": {
                        "$type": "local#array",
                        "items": {
                            "$type": "local#array",
                            "items": {"$type": "local#primitive", "primitive": "int"},
                        },
                    },
                    "optional": False,
                },
            ],
        }

        stub = generate_stub(schema)

        assert "matrix: list[list[int]]" in stub

    def test_stub_with_ref_field_uses_any(self):
        """Schema ref fields should fall back to Any in stubs."""
        schema = {
            "name": "RefSample",
            "version": "1.0.0",
            "fields": [
                {
                    "name": "nested",
                    "fieldType": {
                        "$type": "local#ref",
                        "ref": "local://schemas/Other@1.0.0",
                    },
                    "optional": False,
                },
            ],
        }

        stub = generate_stub(schema)

        # Ref types can't be resolved statically, so use Any
        assert "nested: Any" in stub
