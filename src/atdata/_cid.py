"""CID (Content Identifier) utilities for atdata.

This module provides utilities for generating ATProto-compatible CIDs from
data. CIDs are content-addressable identifiers that can be used to uniquely
identify schemas, datasets, and other records.

The CIDs generated here use:
- CIDv1 format
- dag-cbor codec (0x71)
- SHA-256 hash (0x12)

This ensures compatibility with ATProto's CID requirements and enables
seamless promotion from local storage to atmosphere (ATProto network).

Examples:
    >>> schema = {"name": "ImageSample", "version": "1.0.0", "fields": [...]}
    >>> cid = generate_cid(schema)
    >>> print(cid)
    bafyreihffx5a2e7k6r5zqgp5iwpjqr2gfyheqhzqtlxagvqjqyxzqpzqaa
"""

import hashlib
from typing import Any

import libipld


# CID constants
CID_VERSION_1 = 0x01
CODEC_DAG_CBOR = 0x71
HASH_SHA256 = 0x12
SHA256_SIZE = 0x20


def generate_cid(data: Any) -> str:
    """Generate an ATProto-compatible CID from arbitrary data.

    The data is first encoded as DAG-CBOR, then hashed with SHA-256,
    and finally formatted as a CIDv1 string (base32 multibase).

    Args:
        data: Any data structure that can be encoded as DAG-CBOR.
            This includes dicts, lists, strings, numbers, bytes, etc.

    Returns:
        CIDv1 string in base32 multibase format (starts with 'bafy').

    Raises:
        ValueError: If the data cannot be encoded as DAG-CBOR.

    Examples:
        >>> generate_cid({"name": "test", "value": 42})
        'bafyrei...'
    """
    # Encode data as DAG-CBOR
    try:
        cbor_bytes = libipld.encode_dag_cbor(data)
    except (TypeError, ValueError, OverflowError) as e:
        raise ValueError(f"Failed to encode data as DAG-CBOR: {e}") from e

    # Hash with SHA-256
    sha256_hash = hashlib.sha256(cbor_bytes).digest()

    # Build raw CID bytes:
    # CIDv1 = version(1) + codec(dag-cbor) + multihash
    # Multihash = code(sha256) + size(32) + digest
    raw_cid_bytes = (
        bytes([CID_VERSION_1, CODEC_DAG_CBOR, HASH_SHA256, SHA256_SIZE]) + sha256_hash
    )

    # Encode to base32 multibase string
    return libipld.encode_cid(raw_cid_bytes)


def generate_cid_from_bytes(data_bytes: bytes) -> str:
    """Generate a CID from raw bytes (already encoded data).

    Use this when you have pre-encoded data (e.g., DAG-CBOR bytes from
    another source) and want to generate its CID without re-encoding.

    Args:
        data_bytes: Raw bytes to hash (treated as opaque blob).

    Returns:
        CIDv1 string in base32 multibase format.

    Examples:
        >>> cbor_bytes = libipld.encode_dag_cbor({"key": "value"})
        >>> cid = generate_cid_from_bytes(cbor_bytes)
    """
    sha256_hash = hashlib.sha256(data_bytes).digest()
    raw_cid_bytes = (
        bytes([CID_VERSION_1, CODEC_DAG_CBOR, HASH_SHA256, SHA256_SIZE]) + sha256_hash
    )
    return libipld.encode_cid(raw_cid_bytes)


def verify_cid(cid: str, data: Any) -> bool:
    """Verify that a CID matches the given data.

    Args:
        cid: CID string to verify.
        data: Data that should correspond to the CID.

    Returns:
        True if the CID matches the data, False otherwise.

    Examples:
        >>> cid = generate_cid({"name": "test"})
        >>> verify_cid(cid, {"name": "test"})
        True
        >>> verify_cid(cid, {"name": "different"})
        False
    """
    expected_cid = generate_cid(data)
    return cid == expected_cid


def parse_cid(cid: str) -> dict:
    """Parse a CID string into its components.

    Args:
        cid: CID string to parse.

    Returns:
        Dictionary with 'version', 'codec', and 'hash' keys.
        The 'hash' value is itself a dict with 'code', 'size', and 'digest'.

    Examples:
        >>> info = parse_cid('bafyrei...')
        >>> info['version']
        1
        >>> info['codec']
        113  # 0x71 = dag-cbor
    """
    return libipld.decode_cid(cid)


__all__ = [
    "generate_cid",
    "generate_cid_from_bytes",
    "verify_cid",
    "parse_cid",
]
