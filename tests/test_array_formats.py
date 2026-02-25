"""Tests for new array format types and NDArray v1.1.0 annotations.

Covers:
- Round-trip serialization for each new format (sparse, structured,
  arrow_tensor, safetensors, dataframe)
- Missing-dependency error messages
- SchemaFieldType extensions (new kinds, v1.1 annotation fields)
- Codegen: _json_schema_prop_to_field_type recognises new $ref patterns
- Codegen: _schema_to_type generates __field_formats__ and
  __ndarray_annotations__ metadata
- Pipeline: _make_packable serialises new types, _ensure_good deserialises
  via __field_formats__
"""

import numpy as np
import pandas as pd
import pytest

from atdata._helpers import (
    array_to_bytes,
    bytes_to_array,
    structured_to_bytes,
    bytes_to_structured,
    dataframe_to_bytes,
    bytes_to_dataframe,
    FORMAT_SERIALIZERS,
)
from atdata._schema_codec import (
    _json_schema_prop_to_field_type,
    _field_type_to_python,
    _schema_to_type,
    _field_type_to_stub_str,
    clear_type_cache,
)
from atdata.index._schema import SchemaFieldType


# ---------------------------------------------------------------------------
# Structured array serialization
# ---------------------------------------------------------------------------


class TestStructuredArraySerialization:
    """Round-trip tests for structured_to_bytes / bytes_to_structured."""

    def test_basic_round_trip(self):
        dt = np.dtype([("x", "f4"), ("y", "i4")])
        arr = np.array([(1.0, 2), (3.0, 4)], dtype=dt)
        restored = bytes_to_structured(structured_to_bytes(arr))
        np.testing.assert_array_equal(restored, arr)

    def test_preserves_compound_dtype(self):
        dt = np.dtype([("name", "U10"), ("score", "f8")])
        arr = np.array([("alice", 95.5), ("bob", 87.3)], dtype=dt)
        restored = bytes_to_structured(structured_to_bytes(arr))
        assert restored.dtype == dt

    def test_empty_structured_array(self):
        dt = np.dtype([("a", "i4"), ("b", "f4")])
        arr = np.array([], dtype=dt)
        restored = bytes_to_structured(structured_to_bytes(arr))
        assert restored.dtype == dt
        assert len(restored) == 0

    def test_rejects_plain_ndarray(self):
        arr = np.array([1, 2, 3])
        with pytest.raises(TypeError, match="compound dtype"):
            structured_to_bytes(arr)

    def test_rejects_non_ndarray(self):
        with pytest.raises(TypeError, match="Expected numpy ndarray"):
            structured_to_bytes([1, 2, 3])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# DataFrame serialization
# ---------------------------------------------------------------------------


class TestDataFrameSerialization:
    """Round-trip tests for dataframe_to_bytes / bytes_to_dataframe."""

    def test_basic_round_trip(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        restored = bytes_to_dataframe(dataframe_to_bytes(df))
        pd.testing.assert_frame_equal(restored, df)

    def test_string_columns(self):
        df = pd.DataFrame({"name": ["alice", "bob"], "score": [95, 87]})
        restored = bytes_to_dataframe(dataframe_to_bytes(df))
        pd.testing.assert_frame_equal(restored, df)

    def test_empty_dataframe(self):
        df = pd.DataFrame({"a": pd.array([], dtype="int64")})
        restored = bytes_to_dataframe(dataframe_to_bytes(df))
        assert len(restored) == 0
        assert list(restored.columns) == ["a"]

    def test_rejects_non_dataframe(self):
        with pytest.raises(TypeError, match="Expected pandas DataFrame"):
            dataframe_to_bytes({"a": [1, 2]})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Sparse matrix serialization (requires scipy)
# ---------------------------------------------------------------------------


class TestSparseSerialization:
    """Round-trip tests for sparse_to_bytes / bytes_to_sparse."""

    @pytest.fixture(autouse=True)
    def _skip_without_scipy(self):
        pytest.importorskip("scipy")

    def test_csr_round_trip(self):
        import scipy.sparse as sp
        from atdata._helpers import sparse_to_bytes, bytes_to_sparse

        mat = sp.csr_matrix([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
        restored = bytes_to_sparse(sparse_to_bytes(mat))
        assert sp.issparse(restored)
        np.testing.assert_array_equal(restored.toarray(), mat.toarray())

    def test_coo_round_trip(self):
        import scipy.sparse as sp
        from atdata._helpers import sparse_to_bytes, bytes_to_sparse

        row = np.array([0, 1, 2])
        col = np.array([0, 1, 2])
        data = np.array([10.0, 20.0, 30.0])
        mat = sp.coo_matrix((data, (row, col)), shape=(3, 3))
        restored = bytes_to_sparse(sparse_to_bytes(mat))
        np.testing.assert_array_almost_equal(restored.toarray(), mat.toarray())

    def test_empty_sparse(self):
        import scipy.sparse as sp
        from atdata._helpers import sparse_to_bytes, bytes_to_sparse

        mat = sp.csr_matrix((3, 3))
        restored = bytes_to_sparse(sparse_to_bytes(mat))
        assert restored.nnz == 0
        assert restored.shape == (3, 3)

    def test_rejects_non_sparse(self):
        from atdata._helpers import sparse_to_bytes

        with pytest.raises(TypeError, match="Expected scipy sparse matrix"):
            sparse_to_bytes(np.array([1, 2, 3]))


# ---------------------------------------------------------------------------
# Arrow tensor serialization (requires pyarrow)
# ---------------------------------------------------------------------------


class TestArrowTensorSerialization:
    """Round-trip tests for arrow_tensor_to_bytes / bytes_to_arrow_tensor."""

    @pytest.fixture(autouse=True)
    def _skip_without_pyarrow(self):
        pytest.importorskip("pyarrow")

    def test_basic_round_trip(self):
        import pyarrow as pa
        from atdata._helpers import arrow_tensor_to_bytes, bytes_to_arrow_tensor

        arr = np.array([[1, 2], [3, 4]], dtype=np.int64)
        tensor = pa.Tensor.from_numpy(arr)
        restored = bytes_to_arrow_tensor(arrow_tensor_to_bytes(tensor))
        np.testing.assert_array_equal(restored.to_numpy(), arr)

    def test_float_tensor(self):
        import pyarrow as pa
        from atdata._helpers import arrow_tensor_to_bytes, bytes_to_arrow_tensor

        arr = np.random.rand(3, 4, 5).astype(np.float32)
        tensor = pa.Tensor.from_numpy(arr)
        restored = bytes_to_arrow_tensor(arrow_tensor_to_bytes(tensor))
        np.testing.assert_array_almost_equal(restored.to_numpy(), arr)

    def test_rejects_non_tensor(self):
        from atdata._helpers import arrow_tensor_to_bytes

        with pytest.raises(TypeError, match="Expected pyarrow.Tensor"):
            arrow_tensor_to_bytes(np.array([1, 2, 3]))


# ---------------------------------------------------------------------------
# Safetensors serialization (requires safetensors)
# ---------------------------------------------------------------------------


class TestSafetensorsSerialization:
    """Round-trip tests for safetensors_to_bytes / bytes_to_safetensors."""

    @pytest.fixture(autouse=True)
    def _skip_without_safetensors(self):
        pytest.importorskip("safetensors")

    def test_basic_round_trip(self):
        from atdata._helpers import safetensors_to_bytes, bytes_to_safetensors

        tensors = {
            "weight": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "bias": np.array([0.5], dtype=np.float32),
        }
        restored = bytes_to_safetensors(safetensors_to_bytes(tensors))
        for key in tensors:
            np.testing.assert_array_equal(restored[key], tensors[key])

    def test_multiple_dtypes(self):
        from atdata._helpers import safetensors_to_bytes, bytes_to_safetensors

        tensors = {
            "ints": np.array([1, 2, 3], dtype=np.int64),
            "floats": np.array([1.0, 2.0], dtype=np.float64),
        }
        restored = bytes_to_safetensors(safetensors_to_bytes(tensors))
        for key in tensors:
            np.testing.assert_array_equal(restored[key], tensors[key])

    def test_rejects_non_dict(self):
        from atdata._helpers import safetensors_to_bytes

        with pytest.raises(TypeError, match="Expected dict"):
            safetensors_to_bytes(np.array([1, 2, 3]))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# FORMAT_SERIALIZERS dispatch table
# ---------------------------------------------------------------------------


class TestFormatSerializers:
    """Verify the dispatch table has entries for all format kinds."""

    def test_all_kinds_present(self):
        expected = {
            "ndarray",
            "structured",
            "sparse",
            "arrow_tensor",
            "safetensors",
            "dataframe",
        }
        assert set(FORMAT_SERIALIZERS.keys()) == expected

    def test_ndarray_round_trip_via_dispatch(self):
        serialize, deserialize = FORMAT_SERIALIZERS["ndarray"]
        arr = np.array([1.0, 2.0], dtype=np.float32)
        restored = deserialize(serialize(arr))
        np.testing.assert_array_equal(restored, arr)

    def test_structured_round_trip_via_dispatch(self):
        serialize, deserialize = FORMAT_SERIALIZERS["structured"]
        dt = np.dtype([("a", "i4"), ("b", "f4")])
        arr = np.array([(1, 2.0), (3, 4.0)], dtype=dt)
        restored = deserialize(serialize(arr))
        np.testing.assert_array_equal(restored, arr)


# ---------------------------------------------------------------------------
# SchemaFieldType extensions
# ---------------------------------------------------------------------------


class TestSchemaFieldTypeExtensions:
    """Test new kind values and v1.1 annotation fields on SchemaFieldType."""

    @pytest.mark.parametrize(
        "kind", ["sparse", "structured", "arrow_tensor", "safetensors", "dataframe"]
    )
    def test_new_kinds_round_trip(self, kind):
        ft = SchemaFieldType(kind=kind)  # type: ignore[arg-type]
        d = ft.to_dict()
        assert d["$type"] == f"local#{kind}"
        restored = SchemaFieldType.from_dict(d)
        assert restored.kind == kind

    def test_ndarray_with_v11_annotations(self):
        ft = SchemaFieldType(
            kind="ndarray",
            dtype="float32",
            shape=[None, 224, 224, 3],
            dimension_names=["batch", "height", "width", "channels"],
        )
        d = ft.to_dict()
        assert d["dtype"] == "float32"
        assert d["shape"] == [None, 224, 224, 3]
        assert d["dimensionNames"] == ["batch", "height", "width", "channels"]

        restored = SchemaFieldType.from_dict(d)
        assert restored.shape == [None, 224, 224, 3]
        assert restored.dimension_names == ["batch", "height", "width", "channels"]
        assert restored.dtype == "float32"

    def test_ndarray_without_annotations(self):
        """Existing ndarray fields still work without v1.1 annotations."""
        ft = SchemaFieldType(kind="ndarray", dtype="int64")
        d = ft.to_dict()
        assert "shape" not in d
        assert "dimensionNames" not in d
        restored = SchemaFieldType.from_dict(d)
        assert restored.shape is None
        assert restored.dimension_names is None


# ---------------------------------------------------------------------------
# Codegen: _json_schema_prop_to_field_type
# ---------------------------------------------------------------------------


class TestJsonSchemaPropNewRefs:
    """Test that _json_schema_prop_to_field_type recognises new shim $refs."""

    def test_sparse_ref(self):
        prop = {
            "$ref": "https://alt.science/schemas/atdata-sparse-bytes/1.0.0#/$defs/sparse"
        }
        result = _json_schema_prop_to_field_type(prop)
        assert result["$type"] == "local#sparse"

    def test_structured_ref(self):
        prop = {
            "$ref": "https://alt.science/schemas/atdata-structured-bytes/1.0.0#/$defs/structured"
        }
        result = _json_schema_prop_to_field_type(prop)
        assert result["$type"] == "local#structured"

    def test_arrow_tensor_ref(self):
        prop = {
            "$ref": "https://alt.science/schemas/atdata-arrow-tensor/1.0.0#/$defs/tensor"
        }
        result = _json_schema_prop_to_field_type(prop)
        assert result["$type"] == "local#arrow_tensor"

    def test_safetensors_ref(self):
        prop = {
            "$ref": "https://alt.science/schemas/atdata-safetensors/1.0.0#/$defs/safetensors"
        }
        result = _json_schema_prop_to_field_type(prop)
        assert result["$type"] == "local#safetensors"

    def test_dataframe_ref(self):
        prop = {
            "$ref": "https://alt.science/schemas/atdata-dataframe/1.0.0#/$defs/dataframe"
        }
        result = _json_schema_prop_to_field_type(prop)
        assert result["$type"] == "local#dataframe"

    def test_ndarray_ref_still_works(self):
        prop = {
            "$ref": "https://alt.science/schemas/atdata-ndarray-bytes/1.0.0#/$defs/ndarray"
        }
        result = _json_schema_prop_to_field_type(prop)
        assert result["$type"] == "local#ndarray"

    def test_ndarray_v11_ref_with_annotations(self):
        prop = {
            "$ref": "https://alt.science/schemas/atdata-ndarray-bytes/1.1.0#/$defs/ndarray",
            "dtype": "float32",
            "shape": [None, 64],
            "dimensionNames": ["batch", "features"],
        }
        result = _json_schema_prop_to_field_type(prop)
        assert result["$type"] == "local#ndarray"
        assert result["dtype"] == "float32"
        assert result["shape"] == [None, 64]
        assert result["dimensionNames"] == ["batch", "features"]


# ---------------------------------------------------------------------------
# Codegen: _field_type_to_python for new kinds
# ---------------------------------------------------------------------------


class TestFieldTypeToPythonNewKinds:
    """Test _field_type_to_python maps new kinds to correct Python types."""

    def test_structured_maps_to_ndarray(self):
        from numpy.typing import NDArray

        result = _field_type_to_python({"$type": "local#structured"})
        assert result is NDArray

    @pytest.mark.parametrize(
        "kind", ["sparse", "arrow_tensor", "safetensors", "dataframe"]
    )
    def test_binary_formats_map_to_bytes(self, kind):
        result = _field_type_to_python({"$type": f"local#{kind}"})
        assert result is bytes


# ---------------------------------------------------------------------------
# Codegen: _field_type_to_stub_str for new kinds
# ---------------------------------------------------------------------------


class TestFieldTypeToStubStrNewKinds:
    """Test stub string generation for new kinds."""

    def test_structured_stub(self):
        assert _field_type_to_stub_str({"$type": "local#structured"}) == "NDArray[Any]"

    @pytest.mark.parametrize(
        "kind", ["sparse", "arrow_tensor", "safetensors", "dataframe"]
    )
    def test_binary_format_stubs(self, kind):
        assert _field_type_to_stub_str({"$type": f"local#{kind}"}) == "bytes"

    def test_optional_binary_format(self):
        assert (
            _field_type_to_stub_str({"$type": "local#sparse"}, optional=True)
            == "bytes | None"
        )


# ---------------------------------------------------------------------------
# Codegen: _schema_to_type metadata
# ---------------------------------------------------------------------------


class TestSchemaToTypeFormats:
    """Test that _schema_to_type sets __field_formats__ and __ndarray_annotations__."""

    def setup_method(self):
        clear_type_cache()

    def test_field_formats_set_for_new_kinds(self):
        schema = {
            "name": "FormatTestSample",
            "version": "1.0.0",
            "fields": [
                {
                    "name": "label",
                    "fieldType": {"$type": "local#primitive", "primitive": "str"},
                    "optional": False,
                },
                {
                    "name": "sparse_data",
                    "fieldType": {"$type": "local#sparse"},
                    "optional": False,
                },
                {
                    "name": "table",
                    "fieldType": {"$type": "local#dataframe"},
                    "optional": False,
                },
            ],
        }
        cls = _schema_to_type(schema, use_cache=False)
        assert cls.__field_formats__ == {"sparse_data": "sparse", "table": "dataframe"}

    def test_ndarray_annotations_set(self):
        schema = {
            "name": "AnnotatedSample",
            "version": "1.0.0",
            "fields": [
                {
                    "name": "image",
                    "fieldType": {
                        "$type": "local#ndarray",
                        "dtype": "float32",
                        "shape": [None, 224, 224, 3],
                        "dimensionNames": ["batch", "height", "width", "channels"],
                    },
                    "optional": False,
                },
                {
                    "name": "label",
                    "fieldType": {"$type": "local#primitive", "primitive": "str"},
                    "optional": False,
                },
            ],
        }
        cls = _schema_to_type(schema, use_cache=False)
        ann = cls.__ndarray_annotations__
        assert "image" in ann
        assert ann["image"]["dtype"] == "float32"
        assert ann["image"]["shape"] == [None, 224, 224, 3]
        assert ann["image"]["dimensionNames"] == [
            "batch",
            "height",
            "width",
            "channels",
        ]
        assert "label" not in ann

    def test_no_annotations_when_absent(self):
        schema = {
            "name": "PlainNDArraySample",
            "version": "1.0.0",
            "fields": [
                {
                    "name": "data",
                    "fieldType": {"$type": "local#ndarray"},
                    "optional": False,
                },
            ],
        }
        cls = _schema_to_type(schema, use_cache=False)
        assert cls.__ndarray_annotations__ == {}


# ---------------------------------------------------------------------------
# Pipeline: _make_packable for new types
# ---------------------------------------------------------------------------


class TestMakePackableNewTypes:
    """Test that _make_packable serialises new types correctly."""

    def test_structured_array(self):
        from atdata.dataset import _make_packable

        dt = np.dtype([("x", "f4"), ("y", "i4")])
        arr = np.array([(1.0, 2)], dtype=dt)
        result = _make_packable(arr)
        assert isinstance(result, bytes)
        restored = bytes_to_structured(result)
        np.testing.assert_array_equal(restored, arr)

    def test_dataframe(self):
        from atdata.dataset import _make_packable

        df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
        result = _make_packable(df)
        assert isinstance(result, bytes)
        restored = bytes_to_dataframe(result)
        pd.testing.assert_frame_equal(restored, df)

    def test_plain_ndarray_unchanged(self):
        from atdata.dataset import _make_packable

        arr = np.array([1.0, 2.0], dtype=np.float32)
        result = _make_packable(arr)
        assert isinstance(result, bytes)
        restored = bytes_to_array(result)
        np.testing.assert_array_equal(restored, arr)

    def test_sparse_matrix(self):
        sp = pytest.importorskip("scipy.sparse")
        from atdata.dataset import _make_packable
        from atdata._helpers import bytes_to_sparse

        mat = sp.csr_matrix([[1, 0], [0, 2]])
        result = _make_packable(mat)
        assert isinstance(result, bytes)
        restored = bytes_to_sparse(result)
        np.testing.assert_array_equal(restored.toarray(), mat.toarray())

    def test_arrow_tensor(self):
        pa = pytest.importorskip("pyarrow")
        from atdata.dataset import _make_packable
        from atdata._helpers import bytes_to_arrow_tensor

        arr = np.array([[1, 2], [3, 4]], dtype=np.int64)
        tensor = pa.Tensor.from_numpy(arr)
        result = _make_packable(tensor)
        assert isinstance(result, bytes)
        restored = bytes_to_arrow_tensor(result)
        np.testing.assert_array_equal(restored.to_numpy(), arr)


# ---------------------------------------------------------------------------
# Pipeline: _ensure_good with __field_formats__
# ---------------------------------------------------------------------------


class TestEnsureGoodFieldFormats:
    """Test that _ensure_good deserialises bytes for __field_formats__ fields."""

    def setup_method(self):
        clear_type_cache()

    def test_structured_field_deserialized(self):
        schema = {
            "name": "StructuredEnsureSample",
            "version": "1.0.0",
            "fields": [
                {
                    "name": "data",
                    "fieldType": {"$type": "local#structured"},
                    "optional": False,
                },
            ],
        }
        cls = _schema_to_type(schema, use_cache=False)

        dt = np.dtype([("x", "f4"), ("y", "i4")])
        arr = np.array([(1.0, 2), (3.0, 4)], dtype=dt)
        raw_bytes = structured_to_bytes(arr)

        # Construct the sample from raw bytes (as would happen after msgpack decode)
        sample = cls(data=raw_bytes)
        # _ensure_good should have deserialized via __field_formats__
        assert isinstance(sample.data, np.ndarray)
        assert sample.data.dtype == dt
        np.testing.assert_array_equal(sample.data, arr)

    def test_dataframe_field_deserialized(self):
        schema = {
            "name": "DataFrameEnsureSample",
            "version": "1.0.0",
            "fields": [
                {
                    "name": "table",
                    "fieldType": {"$type": "local#dataframe"},
                    "optional": False,
                },
            ],
        }
        cls = _schema_to_type(schema, use_cache=False)

        df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
        raw_bytes = dataframe_to_bytes(df)

        sample = cls(table=raw_bytes)
        assert isinstance(sample.table, pd.DataFrame)
        pd.testing.assert_frame_equal(sample.table, df)

    def test_ndarray_field_still_works(self):
        """Existing NDArray deserialization is unaffected."""
        schema = {
            "name": "NDArrayEnsureSample",
            "version": "1.0.0",
            "fields": [
                {
                    "name": "embedding",
                    "fieldType": {"$type": "local#ndarray"},
                    "optional": False,
                },
            ],
        }
        cls = _schema_to_type(schema, use_cache=False)

        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        raw_bytes = array_to_bytes(arr)

        sample = cls(embedding=raw_bytes)
        assert isinstance(sample.embedding, np.ndarray)
        np.testing.assert_array_equal(sample.embedding, arr)


# ---------------------------------------------------------------------------
# Backward compatibility: existing ndarray tests still pass
# ---------------------------------------------------------------------------


class TestExistingNDArrayBackwardCompat:
    """Ensure existing NDArray serialization is untouched."""

    @pytest.mark.parametrize("dtype", [np.float32, np.int64, np.uint8])
    def test_array_to_bytes_round_trip(self, dtype):
        arr = np.array([1, 2, 3, 4], dtype=dtype)
        restored = bytes_to_array(array_to_bytes(arr))
        np.testing.assert_array_equal(restored, arr)
        assert restored.dtype == arr.dtype

    def test_multidim_array(self):
        arr = np.random.rand(5, 10).astype(np.float32)
        restored = bytes_to_array(array_to_bytes(arr))
        np.testing.assert_array_almost_equal(restored, arr)
