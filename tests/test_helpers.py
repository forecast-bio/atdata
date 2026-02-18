"""Tests for atdata._helpers module."""

import numpy as np
import pytest

from atdata._helpers import array_to_bytes, bytes_to_array


class TestArraySerialization:
    """Test array_to_bytes and bytes_to_array round-trip serialization."""

    @pytest.mark.parametrize(
        "dtype",
        [
            np.float32,
            np.float64,
            np.int32,
            np.int64,
            np.uint8,
            np.bool_,
            np.complex64,
        ],
    )
    def test_dtype_preservation(self, dtype):
        """Verify dtype is preserved through serialization."""
        original = np.array([1, 2, 3], dtype=dtype)
        serialized = array_to_bytes(original)
        restored = bytes_to_array(serialized)

        assert restored.dtype == original.dtype
        np.testing.assert_array_equal(restored, original)

    @pytest.mark.parametrize(
        "shape",
        [
            (10,),
            (3, 4),
            (2, 3, 4),
            (1, 1, 1, 1),
        ],
    )
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

        assert not non_contiguous.flags["C_CONTIGUOUS"]

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

    def test_object_dtype_rejected(self):
        """Object dtype arrays are rejected to prevent pickle serialization."""
        original = np.array([{"a": 1}, {"b": 2}], dtype=object)
        with pytest.raises(ValueError, match="object-dtype"):
            array_to_bytes(original)

    def test_legacy_npy_format_deserialization(self):
        """bytes_to_array can read legacy .npy-serialized arrays."""
        from io import BytesIO

        original = np.array([10, 20, 30], dtype=np.int32)
        buf = BytesIO()
        np.save(buf, original)
        legacy_bytes = buf.getvalue()

        restored = bytes_to_array(legacy_bytes)
        np.testing.assert_array_equal(restored, original)


class TestBytesToArrayBoundsChecking:
    """Verify bytes_to_array rejects truncated/corrupted buffers."""

    def test_empty_buffer_raises(self):
        with pytest.raises(ValueError, match="too short"):
            bytes_to_array(b"")

    def test_single_byte_raises(self):
        with pytest.raises(ValueError, match="too short"):
            bytes_to_array(b"\x03")

    def test_truncated_dtype_raises(self):
        # dtype_len says 3 but only 1 byte of dtype follows
        with pytest.raises(ValueError, match="too short"):
            bytes_to_array(b"\x03<f")

    def test_truncated_shape_raises(self):
        # Valid dtype header but shape data is missing
        arr = np.array([1.0], dtype=np.float32)
        full = array_to_bytes(arr)
        # Chop off shape and data bytes
        dlen = full[0]
        truncated = full[: 2 + dlen]  # dtype_len + dtype + ndim only
        with pytest.raises(ValueError, match="too short"):
            bytes_to_array(truncated)
