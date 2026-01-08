"""Tests for atdata._helpers module."""

import numpy as np
import pytest

from atdata._helpers import array_to_bytes, bytes_to_array


class TestArraySerialization:
    """Test array_to_bytes and bytes_to_array round-trip serialization."""

    @pytest.mark.parametrize("dtype", [
        np.float32,
        np.float64,
        np.int32,
        np.int64,
        np.uint8,
        np.bool_,
        np.complex64,
    ])
    def test_dtype_preservation(self, dtype):
        """Verify dtype is preserved through serialization."""
        original = np.array([1, 2, 3], dtype=dtype)
        serialized = array_to_bytes(original)
        restored = bytes_to_array(serialized)

        assert restored.dtype == original.dtype
        np.testing.assert_array_equal(restored, original)

    @pytest.mark.parametrize("shape", [
        (10,),
        (3, 4),
        (2, 3, 4),
        (1, 1, 1, 1),
    ])
    def test_shape_preservation(self, shape):
        """Verify shape is preserved through serialization."""
        original = np.random.rand(*shape).astype(np.float32)
        serialized = array_to_bytes(original)
        restored = bytes_to_array(serialized)

        assert restored.shape == original.shape
        np.testing.assert_array_almost_equal(restored, original)

    def test_empty_array(self):
        """Verify empty arrays serialize correctly."""
        original = np.array([], dtype=np.float32)
        serialized = array_to_bytes(original)
        restored = bytes_to_array(serialized)

        assert restored.shape == (0,)
        assert restored.dtype == np.float32

    def test_scalar_array(self):
        """Verify 0-dimensional arrays serialize correctly."""
        original = np.array(42.0)
        serialized = array_to_bytes(original)
        restored = bytes_to_array(serialized)

        assert restored.shape == ()
        assert restored == 42.0

    def test_large_array(self):
        """Verify large arrays serialize correctly."""
        original = np.random.rand(100, 100).astype(np.float32)
        serialized = array_to_bytes(original)
        restored = bytes_to_array(serialized)

        np.testing.assert_array_almost_equal(restored, original)

    def test_contiguous_and_noncontiguous(self):
        """Verify non-contiguous arrays serialize correctly."""
        original = np.random.rand(10, 10).astype(np.float32)
        non_contiguous = original[::2, ::2]  # Strided view

        assert not non_contiguous.flags['C_CONTIGUOUS']

        serialized = array_to_bytes(non_contiguous)
        restored = bytes_to_array(serialized)

        np.testing.assert_array_almost_equal(restored, non_contiguous)

    def test_bytes_output_type(self):
        """Verify array_to_bytes returns bytes."""
        arr = np.array([1, 2, 3])
        result = array_to_bytes(arr)
        assert isinstance(result, bytes)

    def test_ndarray_output_type(self):
        """Verify bytes_to_array returns ndarray."""
        arr = np.array([1, 2, 3])
        serialized = array_to_bytes(arr)
        result = bytes_to_array(serialized)
        assert isinstance(result, np.ndarray)
