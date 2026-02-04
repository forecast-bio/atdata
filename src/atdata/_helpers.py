"""Helper utilities for numpy array serialization and content checksums.

This module provides utility functions for converting numpy arrays to and from
bytes for msgpack serialization, as well as SHA-256 checksum utilities for
verifying dataset shard integrity.

Functions:
    - ``array_to_bytes()``: Serialize numpy array to bytes
    - ``bytes_to_array()``: Deserialize bytes to numpy array
    - ``sha256_file()``: Compute SHA-256 hex digest of a file
    - ``sha256_bytes()``: Compute SHA-256 hex digest of in-memory bytes
    - ``verify_checksums()``: Verify stored checksums against shard data

Classes:
    - ``ShardWriteResult``: ``list[str]`` subclass carrying per-shard checksums
"""

from __future__ import annotations

##
# Imports

import hashlib
import struct
from io import BytesIO
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from atdata._protocols import IndexEntry

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
    if len(b) < 2:
        raise ValueError(f"Array buffer too short ({len(b)} bytes): need at least 2")
    dlen = b[0]
    min_header = 2 + dlen  # dtype_len + dtype_str + ndim
    if len(b) < min_header:
        raise ValueError(
            f"Array buffer too short ({len(b)} bytes): need at least {min_header} for header"
        )
    dtype = np.dtype(b[1 : 1 + dlen].decode())
    ndim = b[1 + dlen]
    offset = 2 + dlen
    min_with_shape = offset + ndim * 8
    if len(b) < min_with_shape:
        raise ValueError(
            f"Array buffer too short ({len(b)} bytes): need at least {min_with_shape} for shape"
        )
    shape = struct.unpack_from(f"<{ndim}q", b, offset)
    offset += ndim * 8
    return np.frombuffer(b, dtype=dtype, offset=offset).reshape(shape).copy()


##
# Checksum utilities


def sha256_file(path: str | Path, *, chunk_size: int = 8192) -> str:
    """Compute SHA-256 hex digest of a file.

    Reads the file in chunks to support large files without loading
    everything into memory.

    Args:
        path: Path to the file.
        chunk_size: Read buffer size in bytes.

    Returns:
        Hex-encoded SHA-256 digest string (64 characters).

    Examples:
        >>> digest = sha256_file("/path/to/shard.tar")
        >>> len(digest)
        64
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Compute SHA-256 hex digest of in-memory bytes.

    Args:
        data: Raw bytes to hash.

    Returns:
        Hex-encoded SHA-256 digest string (64 characters).

    Examples:
        >>> sha256_bytes(b"hello")
        '2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824'
    """
    return hashlib.sha256(data).hexdigest()


class ShardWriteResult(list):
    """Return type carrying shard URLs and per-shard checksums.

    Extends ``list[str]`` so it satisfies the ``AbstractDataStore.write_shards()``
    return type (``list[str]``), while also carrying SHA-256 checksum metadata.

    Attributes:
        checksums: Dict mapping each shard URL to its SHA-256 hex digest.

    Examples:
        >>> result = ShardWriteResult(["shard-0.tar"], {"shard-0.tar": "abcd..."})
        >>> result[0]
        'shard-0.tar'
        >>> result.checksums["shard-0.tar"]
        'abcd...'
    """

    checksums: dict[str, str]

    def __init__(self, urls: list[str], checksums: dict[str, str]) -> None:
        super().__init__(urls)
        self.checksums = checksums


def verify_checksums(entry: "IndexEntry") -> dict[str, str]:
    """Verify SHA-256 checksums for all shards in an index entry.

    Compares stored checksums (from ``entry.metadata["checksums"]``) against
    freshly computed digests. Shards without stored checksums are reported
    as ``"skipped"``.

    Currently supports local file paths only. S3 and AT URIs are reported
    as ``"skipped"`` unless a corresponding checksum is absent.

    Args:
        entry: An IndexEntry with ``data_urls`` and optional metadata checksums.

    Returns:
        Dict mapping each shard URL to one of:
        ``"ok"``, ``"mismatch"``, ``"skipped"``, or ``"error:<message>"``.

    Examples:
        >>> results = verify_checksums(entry)
        >>> assert all(v == "ok" for v in results.values())
    """
    stored: dict[str, str] = {}
    if entry.metadata and "checksums" in entry.metadata:
        stored = entry.metadata["checksums"]

    results: dict[str, str] = {}
    for url in entry.data_urls:
        if url not in stored:
            results[url] = "skipped"
            continue
        # Only local file paths can be verified; skip remote URLs
        if url.startswith(("s3://", "at://", "http://", "https://")):
            results[url] = "skipped"
            continue
        try:
            actual = sha256_file(url)
            results[url] = "ok" if actual == stored[url] else "mismatch"
        except Exception as e:
            results[url] = f"error:{e}"
    return results
