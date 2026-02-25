"""Helper utilities for array serialization and content checksums.

This module provides utility functions for converting numpy arrays (and other
array-like types) to and from bytes for msgpack serialization, as well as
SHA-256 checksum utilities for verifying dataset shard integrity.

Supported array formats:
    - **ndarray** (``array_to_bytes`` / ``bytes_to_array``): Dense numpy arrays
    - **sparse** (``sparse_to_bytes`` / ``bytes_to_sparse``): Scipy sparse matrices
    - **structured** (``structured_to_bytes`` / ``bytes_to_structured``): Numpy structured arrays
    - **arrow_tensor** (``arrow_tensor_to_bytes`` / ``bytes_to_arrow_tensor``): Arrow IPC tensors
    - **safetensors** (``safetensors_to_bytes`` / ``bytes_to_safetensors``): HuggingFace safetensors
    - **dataframe** (``dataframe_to_bytes`` / ``bytes_to_dataframe``): Pandas DataFrames (Parquet)

The sparse, arrow_tensor, and safetensors formats require optional dependencies
(``scipy``, ``pyarrow``, ``safetensors``). Install via extras::

    pip install atdata[sparse]       # scipy
    pip install atdata[arrow]        # pyarrow
    pip install atdata[safetensors]  # safetensors
    pip install atdata[all-formats]  # all of the above

Classes:
    - ``ShardWriteResult``: ``list[str]`` subclass carrying per-shard checksums
"""

from __future__ import annotations

##
# Imports

import hashlib
import struct
from io import BytesIO
from typing import Any, TYPE_CHECKING

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
    raw array bytes via ``ndarray.tobytes()``. Object-dtype arrays are
    rejected to prevent pickle-based serialization (security risk).

    Args:
        x: A numpy array to serialize.

    Returns:
        Raw bytes representing the serialized array.
    """
    if x.dtype == object:
        raise ValueError(
            "Cannot serialize object-dtype arrays. "
            "Convert to a concrete dtype before serializing."
        )

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
        return np.load(BytesIO(b), allow_pickle=False)

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
# Sparse matrix serialization (requires scipy)


def sparse_to_bytes(x: "Any") -> bytes:
    """Serialize a scipy sparse matrix to bytes.

    Uses ``scipy.sparse.save_npz`` to write the matrix in compressed NPZ
    format to an in-memory buffer.

    Args:
        x: A scipy sparse matrix (CSR, CSC, COO, etc.).

    Returns:
        Raw bytes of the serialized sparse matrix.

    Raises:
        ImportError: If scipy is not installed.
        TypeError: If *x* is not a scipy sparse matrix.

    Examples:
        >>> import scipy.sparse as sp
        >>> mat = sp.csr_matrix([[1, 0], [0, 2]])
        >>> data = sparse_to_bytes(mat)
        >>> roundtrip = bytes_to_sparse(data)
        >>> (roundtrip != mat).nnz == 0
        True
    """
    try:
        import scipy.sparse as sp
    except ImportError:
        raise ImportError(
            "scipy is required for sparse matrix serialization. "
            "Install it with: pip install atdata[sparse]"
        ) from None

    if not sp.issparse(x):
        raise TypeError(f"Expected scipy sparse matrix, got {type(x).__name__}")

    buf = BytesIO()
    sp.save_npz(buf, x)
    return buf.getvalue()


def bytes_to_sparse(b: bytes) -> "Any":
    """Deserialize bytes to a scipy sparse matrix.

    Args:
        b: Raw bytes from ``sparse_to_bytes``.

    Returns:
        The deserialized scipy sparse matrix.

    Raises:
        ImportError: If scipy is not installed.

    Examples:
        >>> import scipy.sparse as sp
        >>> mat = sp.csr_matrix([[1, 0], [0, 2]])
        >>> roundtrip = bytes_to_sparse(sparse_to_bytes(mat))
        >>> (roundtrip != mat).nnz == 0
        True
    """
    try:
        import scipy.sparse as sp
    except ImportError:
        raise ImportError(
            "scipy is required for sparse matrix deserialization. "
            "Install it with: pip install atdata[sparse]"
        ) from None

    return sp.load_npz(BytesIO(b))


##
# Structured array serialization


def structured_to_bytes(x: np.ndarray) -> bytes:
    """Serialize a numpy structured array to bytes.

    Uses numpy ``.npy`` format via ``np.save`` which preserves compound dtype
    information in the header. This is the same underlying format as dense
    arrays but is called out separately so that schemas can distinguish
    structured (compound dtype) arrays from plain dense arrays.

    Args:
        x: A numpy structured array (compound dtype).

    Returns:
        Raw bytes of the serialized structured array.

    Raises:
        TypeError: If *x* does not have a compound (structured) dtype.

    Examples:
        >>> dt = np.dtype([("x", "f4"), ("y", "i4")])
        >>> arr = np.array([(1.0, 2), (3.0, 4)], dtype=dt)
        >>> data = structured_to_bytes(arr)
        >>> roundtrip = bytes_to_structured(data)
        >>> np.array_equal(roundtrip, arr)
        True
    """
    if not isinstance(x, np.ndarray):
        raise TypeError(f"Expected numpy ndarray, got {type(x).__name__}")
    if x.dtype.names is None:
        raise TypeError(
            f"Expected structured array with compound dtype, got dtype={x.dtype}"
        )

    buf = BytesIO()
    np.save(buf, x, allow_pickle=False)
    return buf.getvalue()


def bytes_to_structured(b: bytes) -> np.ndarray:
    """Deserialize bytes to a numpy structured array.

    Args:
        b: Raw bytes from ``structured_to_bytes``.

    Returns:
        The deserialized numpy structured array with compound dtype.

    Examples:
        >>> dt = np.dtype([("x", "f4"), ("y", "i4")])
        >>> arr = np.array([(1.0, 2), (3.0, 4)], dtype=dt)
        >>> roundtrip = bytes_to_structured(structured_to_bytes(arr))
        >>> np.array_equal(roundtrip, arr)
        True
    """
    return np.load(BytesIO(b), allow_pickle=False)


##
# Arrow tensor serialization (requires pyarrow)


def arrow_tensor_to_bytes(x: "Any") -> bytes:
    """Serialize a PyArrow Tensor to bytes using Arrow IPC format.

    Args:
        x: A ``pyarrow.Tensor`` instance.

    Returns:
        Raw bytes of the serialized tensor.

    Raises:
        ImportError: If pyarrow is not installed.
        TypeError: If *x* is not a ``pyarrow.Tensor``.

    Examples:
        >>> import pyarrow as pa
        >>> tensor = pa.Tensor.from_numpy(np.array([[1, 2], [3, 4]]))
        >>> data = arrow_tensor_to_bytes(tensor)
        >>> roundtrip = bytes_to_arrow_tensor(data)
        >>> roundtrip.to_numpy().tolist()
        [[1, 2], [3, 4]]
    """
    try:
        import pyarrow as pa
    except ImportError:
        raise ImportError(
            "pyarrow is required for Arrow tensor serialization. "
            "Install it with: pip install atdata[arrow]"
        ) from None

    if not isinstance(x, pa.Tensor):
        raise TypeError(f"Expected pyarrow.Tensor, got {type(x).__name__}")

    buf = pa.BufferOutputStream()
    pa.ipc.write_tensor(x, buf)
    return buf.getvalue().to_pybytes()


def bytes_to_arrow_tensor(b: bytes) -> "Any":
    """Deserialize bytes to a PyArrow Tensor.

    Args:
        b: Raw bytes from ``arrow_tensor_to_bytes``.

    Returns:
        The deserialized ``pyarrow.Tensor``.

    Raises:
        ImportError: If pyarrow is not installed.

    Examples:
        >>> import pyarrow as pa
        >>> tensor = pa.Tensor.from_numpy(np.array([[1, 2], [3, 4]]))
        >>> roundtrip = bytes_to_arrow_tensor(arrow_tensor_to_bytes(tensor))
        >>> roundtrip.to_numpy().tolist()
        [[1, 2], [3, 4]]
    """
    try:
        import pyarrow as pa
    except ImportError:
        raise ImportError(
            "pyarrow is required for Arrow tensor deserialization. "
            "Install it with: pip install atdata[arrow]"
        ) from None

    reader = pa.BufferReader(b)
    return pa.ipc.read_tensor(reader)


##
# Safetensors serialization (requires safetensors)


def safetensors_to_bytes(tensors: dict[str, np.ndarray]) -> bytes:
    """Serialize a dict of numpy arrays to safetensors format.

    Args:
        tensors: Mapping of tensor names to numpy arrays.

    Returns:
        Raw bytes in safetensors format.

    Raises:
        ImportError: If the safetensors package is not installed.
        TypeError: If *tensors* is not a dict.

    Examples:
        >>> tensors = {"weight": np.array([1.0, 2.0]), "bias": np.array([0.5])}
        >>> data = safetensors_to_bytes(tensors)
        >>> roundtrip = bytes_to_safetensors(data)
        >>> list(roundtrip.keys())
        ['weight', 'bias']
    """
    try:
        from safetensors.numpy import save
    except ImportError:
        raise ImportError(
            "safetensors is required for safetensors serialization. "
            "Install it with: pip install atdata[safetensors]"
        ) from None

    if not isinstance(tensors, dict):
        raise TypeError(f"Expected dict[str, ndarray], got {type(tensors).__name__}")

    return save(tensors)


def bytes_to_safetensors(b: bytes) -> dict[str, np.ndarray]:
    """Deserialize safetensors bytes to a dict of numpy arrays.

    Args:
        b: Raw bytes from ``safetensors_to_bytes``.

    Returns:
        Dict mapping tensor names to numpy arrays.

    Raises:
        ImportError: If the safetensors package is not installed.

    Examples:
        >>> tensors = {"weight": np.array([1.0, 2.0]), "bias": np.array([0.5])}
        >>> roundtrip = bytes_to_safetensors(safetensors_to_bytes(tensors))
        >>> np.array_equal(roundtrip["weight"], tensors["weight"])
        True
    """
    try:
        from safetensors.numpy import load
    except ImportError:
        raise ImportError(
            "safetensors is required for safetensors deserialization. "
            "Install it with: pip install atdata[safetensors]"
        ) from None

    return load(b)


##
# DataFrame serialization (uses pandas + parquet)


def dataframe_to_bytes(df: "Any") -> bytes:
    """Serialize a pandas DataFrame to Parquet bytes.

    Args:
        df: A ``pandas.DataFrame`` instance.

    Returns:
        Raw bytes in Parquet format.

    Raises:
        TypeError: If *df* is not a pandas DataFrame.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
        >>> data = dataframe_to_bytes(df)
        >>> roundtrip = bytes_to_dataframe(data)
        >>> roundtrip.equals(df)
        True
    """
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")

    buf = BytesIO()
    df.to_parquet(buf, engine="fastparquet")
    return buf.getvalue()


def bytes_to_dataframe(b: bytes) -> "Any":
    """Deserialize Parquet bytes to a pandas DataFrame.

    Args:
        b: Raw bytes from ``dataframe_to_bytes``.

    Returns:
        The deserialized ``pandas.DataFrame``.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
        >>> roundtrip = bytes_to_dataframe(dataframe_to_bytes(df))
        >>> roundtrip.equals(df)
        True
    """
    import pandas as pd

    return pd.read_parquet(BytesIO(b), engine="fastparquet")


# Dispatch table mapping format kind strings to (serialize, deserialize) pairs.
# Used by dataset.py pipeline integration.
FORMAT_SERIALIZERS: dict[str, tuple["Any", "Any"]] = {
    "ndarray": (array_to_bytes, bytes_to_array),
    "structured": (structured_to_bytes, bytes_to_structured),
    "sparse": (sparse_to_bytes, bytes_to_sparse),
    "arrow_tensor": (arrow_tensor_to_bytes, bytes_to_arrow_tensor),
    "safetensors": (safetensors_to_bytes, bytes_to_safetensors),
    "dataframe": (dataframe_to_bytes, bytes_to_dataframe),
}


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
