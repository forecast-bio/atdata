"""Shared type conversion utilities for schema handling.

This module provides common type mapping functions used by both local.py
and atmosphere/schema.py to avoid code duplication.
"""

import types
from typing import Any, get_origin, get_args, Union

# Mapping from numpy dtype strings to schema dtype names
NUMPY_DTYPE_MAP = {
    "float16": "float16",
    "float32": "float32",
    "float64": "float64",
    "int8": "int8",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
    "uint8": "uint8",
    "uint16": "uint16",
    "uint32": "uint32",
    "uint64": "uint64",
    "bool": "bool",
    "complex64": "complex64",
    "complex128": "complex128",
}

# Mapping from Python primitive types to schema type names
PRIMITIVE_TYPE_MAP = {
    str: "str",
    int: "int",
    float: "float",
    bool: "bool",
    bytes: "bytes",
}


def numpy_dtype_to_string(dtype: Any) -> str:
    """Convert a numpy dtype annotation to a schema dtype string.

    Args:
        dtype: A numpy dtype or type annotation containing dtype info.

    Returns:
        Schema dtype string (e.g., "float32", "int64"). Defaults to "float32".
    """
    dtype_str = str(dtype)
    # Exact match first (handles "float32", "int64", etc.)
    if dtype_str in NUMPY_DTYPE_MAP:
        return NUMPY_DTYPE_MAP[dtype_str]
    # Substring match, longest keys first to avoid "int8" matching "uint8"
    for key in sorted(NUMPY_DTYPE_MAP, key=len, reverse=True):
        if key in dtype_str:
            return NUMPY_DTYPE_MAP[key]
    return "float32"


def unwrap_optional(python_type: Any) -> tuple[Any, bool]:
    """Extract the inner type from Optional/Union types.

    Handles both `Optional[T]` (Union[T, None]) and `T | None` syntax.

    Args:
        python_type: A Python type annotation.

    Returns:
        Tuple of (inner_type, is_optional). If type is not Optional,
        returns (python_type, False).

    Raises:
        TypeError: If complex union types (Union[A, B] where both are non-None).
    """
    origin = get_origin(python_type)

    if origin is Union or isinstance(python_type, types.UnionType):
        args = get_args(python_type)
        non_none_args = [a for a in args if a is not type(None)]
        is_optional = type(None) in args or len(non_none_args) < len(args)

        if len(non_none_args) == 1:
            return non_none_args[0], is_optional
        elif len(non_none_args) > 1:
            raise TypeError(f"Complex union types not supported: {python_type}")

    return python_type, False


def is_ndarray_type(python_type: Any) -> bool:
    """Check if a type annotation represents an NDArray."""
    type_str = str(python_type)
    return "NDArray" in type_str or "ndarray" in type_str.lower()


def extract_ndarray_dtype(python_type: Any) -> str:
    """Extract dtype from NDArray type annotation.

    Args:
        python_type: NDArray type annotation (e.g., NDArray[np.float32]).

    Returns:
        Dtype string (e.g., "float32"). Defaults to "float32".
    """
    args = get_args(python_type)
    if args:
        dtype_arg = args[-1]
        if dtype_arg is not None:
            return numpy_dtype_to_string(dtype_arg)
    return "float32"


def parse_semver(version: str) -> tuple[int, int, int]:
    """Parse a semantic version string into a comparable tuple.

    Args:
        version: A ``"major.minor.patch"`` version string.

    Returns:
        Tuple of (major, minor, patch) integers.

    Raises:
        ValueError: If the version string is not valid semver.

    Examples:
        >>> parse_semver("1.2.3")
        (1, 2, 3)
    """
    parts = version.split(".")
    if len(parts) != 3:
        raise ValueError(f"Invalid semver: {version}")
    return int(parts[0]), int(parts[1]), int(parts[2])
