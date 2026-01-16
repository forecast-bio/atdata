"""Shared type conversion utilities for schema handling.

This module provides common type mapping functions used by both local.py
and atmosphere/schema.py to avoid code duplication.
"""

from typing import Any

# Mapping from numpy dtype strings to schema dtype names
NUMPY_DTYPE_MAP = {
    "float16": "float16", "float32": "float32", "float64": "float64",
    "int8": "int8", "int16": "int16", "int32": "int32", "int64": "int64",
    "uint8": "uint8", "uint16": "uint16", "uint32": "uint32", "uint64": "uint64",
    "bool": "bool", "complex64": "complex64", "complex128": "complex128",
}

# Mapping from Python primitive types to schema type names
PRIMITIVE_TYPE_MAP = {
    str: "str", int: "int", float: "float", bool: "bool", bytes: "bytes",
}


def numpy_dtype_to_string(dtype: Any) -> str:
    """Convert a numpy dtype annotation to a schema dtype string.

    Args:
        dtype: A numpy dtype or type annotation containing dtype info.

    Returns:
        Schema dtype string (e.g., "float32", "int64"). Defaults to "float32".
    """
    dtype_str = str(dtype)
    for key, value in NUMPY_DTYPE_MAP.items():
        if key in dtype_str:
            return value
    return "float32"
