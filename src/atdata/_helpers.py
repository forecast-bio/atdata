"""Helper utilities for numpy array serialization.

This module provides utility functions for converting numpy arrays to and from
bytes for msgpack serialization. The functions use numpy's native save/load
format to preserve array dtype and shape information.

Functions:
    - ``array_to_bytes()``: Serialize numpy array to bytes
    - ``bytes_to_array()``: Deserialize bytes to numpy array

These helpers are used internally by ``PackableSample`` to enable transparent
handling of NDArray fields during msgpack packing/unpacking.
"""

##
# Imports

from io import BytesIO

import numpy as np


##

def array_to_bytes( x: np.ndarray ) -> bytes:
    """Convert a numpy array to bytes for msgpack serialization.

    Uses numpy's native ``save()`` format to preserve array dtype and shape.

    Args:
        x: A numpy array to serialize.

    Returns:
        Raw bytes representing the serialized array.

    Note:
        Uses ``allow_pickle=True`` to support object dtypes.
    """
    np_bytes = BytesIO()
    np.save( np_bytes, x, allow_pickle = True )
    return np_bytes.getvalue()

def bytes_to_array( b: bytes ) -> np.ndarray:
    """Convert serialized bytes back to a numpy array.

    Reverses the serialization performed by ``array_to_bytes()``.

    Args:
        b: Raw bytes from a serialized numpy array.

    Returns:
        The deserialized numpy array with original dtype and shape.

    Note:
        Uses ``allow_pickle=True`` to support object dtypes.
    """
    np_bytes = BytesIO( b )
    return np.load( np_bytes, allow_pickle = True )