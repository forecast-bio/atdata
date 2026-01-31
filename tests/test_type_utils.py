"""Tests for _type_utils module — shared type conversion utilities."""

from typing import Optional, Union

import numpy as np
import pytest
from numpy.typing import NDArray

from atdata._type_utils import (
    NUMPY_DTYPE_MAP,
    PRIMITIVE_TYPE_MAP,
    extract_ndarray_dtype,
    is_ndarray_type,
    numpy_dtype_to_string,
    parse_semver,
    unwrap_optional,
)


##
# numpy_dtype_to_string


class TestNumpyDtypeToString:
    """Tests for numpy_dtype_to_string()."""

    @pytest.mark.parametrize("key", list(NUMPY_DTYPE_MAP.keys()))
    def test_all_known_dtypes(self, key: str):
        """Every key in NUMPY_DTYPE_MAP should round-trip."""
        assert numpy_dtype_to_string(key) == NUMPY_DTYPE_MAP[key]

    def test_numpy_dtype_objects(self):
        """Should handle actual numpy dtype objects."""
        assert numpy_dtype_to_string(np.dtype("float32")) == "float32"
        assert numpy_dtype_to_string(np.dtype("int64")) == "int64"
        assert numpy_dtype_to_string(np.dtype("uint8")) == "uint8"

    def test_unknown_dtype_defaults_to_float32(self):
        """Unknown dtypes should default to float32."""
        assert numpy_dtype_to_string("mystery_type") == "float32"
        assert numpy_dtype_to_string(object) == "float32"

    def test_partial_match(self):
        """String containing a known dtype substring should match."""
        assert numpy_dtype_to_string("numpy.float64") == "float64"


##
# unwrap_optional


class TestUnwrapOptional:
    """Tests for unwrap_optional()."""

    def test_plain_type(self):
        """Non-optional types should return (type, False)."""
        inner, is_opt = unwrap_optional(str)
        assert inner is str
        assert is_opt is False

    def test_optional_typing(self):
        """Optional[T] from typing module should unwrap."""
        inner, is_opt = unwrap_optional(Optional[int])
        assert inner is int
        assert is_opt is True

    def test_optional_pipe_syntax(self):
        """T | None syntax should unwrap."""
        annotation = eval("int | None")  # noqa: S307 — safe literal
        inner, is_opt = unwrap_optional(annotation)
        assert inner is int
        assert is_opt is True

    def test_union_raises_on_complex(self):
        """Union[A, B] where both are non-None should raise TypeError."""
        with pytest.raises(TypeError, match="Complex union"):
            unwrap_optional(Union[int, str])

    def test_union_with_none(self):
        """Union[T, None] is equivalent to Optional[T]."""
        inner, is_opt = unwrap_optional(Union[float, None])
        assert inner is float
        assert is_opt is True


##
# is_ndarray_type


class TestIsNdarrayType:
    """Tests for is_ndarray_type()."""

    def test_ndarray_annotation(self):
        assert is_ndarray_type(NDArray) is True

    def test_ndarray_typed(self):
        assert is_ndarray_type(NDArray[np.float32]) is True

    def test_plain_ndarray_class(self):
        assert is_ndarray_type(np.ndarray) is True

    def test_non_ndarray(self):
        assert is_ndarray_type(str) is False
        assert is_ndarray_type(int) is False
        assert is_ndarray_type(list) is False


##
# extract_ndarray_dtype


class TestExtractNdarrayDtype:
    """Tests for extract_ndarray_dtype()."""

    def test_typed_ndarray(self):
        assert extract_ndarray_dtype(NDArray[np.float64]) == "float64"

    def test_untyped_ndarray_defaults_float32(self):
        """Bare NDArray without dtype args should default to float32."""
        assert extract_ndarray_dtype(NDArray) == "float32"

    def test_plain_type_defaults_float32(self):
        """Non-NDArray type should default to float32."""
        assert extract_ndarray_dtype(str) == "float32"


##
# parse_semver


class TestParseSemver:
    """Tests for parse_semver()."""

    def test_basic(self):
        assert parse_semver("1.2.3") == (1, 2, 3)

    def test_zeros(self):
        assert parse_semver("0.0.0") == (0, 0, 0)

    def test_large_numbers(self):
        assert parse_semver("100.200.300") == (100, 200, 300)

    def test_ordering(self):
        """Parsed tuples should compare correctly."""
        assert parse_semver("1.0.0") < parse_semver("2.0.0")
        assert parse_semver("1.1.0") < parse_semver("1.2.0")
        assert parse_semver("1.1.1") < parse_semver("1.1.2")

    def test_invalid_too_few_parts(self):
        with pytest.raises(ValueError, match="Invalid semver"):
            parse_semver("1.2")

    def test_invalid_too_many_parts(self):
        with pytest.raises(ValueError, match="Invalid semver"):
            parse_semver("1.2.3.4")

    def test_invalid_non_numeric(self):
        with pytest.raises(ValueError):
            parse_semver("a.b.c")

    def test_empty_string(self):
        with pytest.raises(ValueError, match="Invalid semver"):
            parse_semver("")


##
# Constants


class TestConstants:
    """Verify constant maps are populated."""

    def test_numpy_dtype_map_populated(self):
        assert len(NUMPY_DTYPE_MAP) >= 14

    def test_primitive_type_map_populated(self):
        assert PRIMITIVE_TYPE_MAP[str] == "str"
        assert PRIMITIVE_TYPE_MAP[int] == "int"
        assert PRIMITIVE_TYPE_MAP[float] == "float"
        assert PRIMITIVE_TYPE_MAP[bool] == "bool"
        assert PRIMITIVE_TYPE_MAP[bytes] == "bytes"
