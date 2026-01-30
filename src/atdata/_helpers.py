"""Helper utilities for numpy array serialization.

This module provides utility functions for converting numpy arrays to and from
bytes for msgpack serialization.

Functions:
    - ``array_to_bytes()``: Serialize numpy array to bytes
    - ``bytes_to_array()``: Deserialize bytes to numpy array

These helpers are used internally by ``PackableSample`` to enable transparent
handling of NDArray fields during msgpack packing/unpacking.
"""

##
# Imports

import struct
from io import BytesIO

import numpy as np

# .npy format magic prefix (used for backward-compatible deserialization)
_NPY_MAGIC = b"\x93NUMPY"


##


def array_to_bytes(x: np.ndarray) -> bytes:
    """Convert a numpy array to bytes for msgpack serialization.

    Uses a compact binary format: a short header (dtype + shape) followed by
    raw array bytes via ``ndarray.tobytes()``. Falls back to numpy's ``.npy``
    format for object dtypes that cannot be represented as raw bytes.

    Args:
        x: A numpy array to serialize.

    Returns:
        Raw bytes representing the serialized array.
    """
    if x.dtype == object:
        buf = BytesIO()
        np.save(buf, x, allow_pickle=True)
        return buf.getvalue()

    dtype_str = x.dtype.str.encode()  # e.g. b'<f4'
    header = struct.pack(f"<B{len(x.shape)}q", len(x.shape), *x.shape)
    return struct.pack("<B", len(dtype_str)) + dtype_str + header + x.tobytes()


def bytes_to_array(b: bytes) -> np.ndarray:
    """Convert serialized bytes back to a numpy array.

    Transparently handles both the compact format produced by the current
    ``array_to_bytes()`` and the legacy ``.npy`` format.

    Args:
        b: Raw bytes from a serialized numpy array.

    Returns:
        The deserialized numpy array with original dtype and shape.
    """
    if b[:6] == _NPY_MAGIC:
        return np.load(BytesIO(b), allow_pickle=True)

    # Compact format: dtype_len(1B) + dtype_str + ndim(1B) + shape(ndim×8B) + data
    dlen = b[0]
    dtype = np.dtype(b[1 : 1 + dlen].decode())
    ndim = b[1 + dlen]
    offset = 2 + dlen
    shape = struct.unpack_from(f"<{ndim}q", b, offset)
    offset += ndim * 8
    return np.frombuffer(b, dtype=dtype, offset=offset).reshape(shape).copy()
