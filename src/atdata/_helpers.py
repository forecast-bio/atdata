"""Assorted helper methods for `atdata`"""

##
# Imports

from io import BytesIO

import numpy as np


##

def array_to_bytes( x: np.ndarray ) -> bytes:
    """Convert `numpy` array to a format suitable for packing"""
    np_bytes = BytesIO()
    np.save( np_bytes, x, allow_pickle = True )
    return np_bytes.getvalue()

def bytes_to_array( b: bytes ) -> np.ndarray:
    """Convert packed bytes back to a `numpy` array"""
    np_bytes = BytesIO( b )
    return np.load( np_bytes, allow_pickle = True )