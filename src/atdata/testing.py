"""Testing utilities for atdata.

Provides mock clients, dataset factories, and pytest fixtures for writing
tests against atdata without requiring external services (Redis, S3, ATProto PDS).

Usage::

    import atdata.testing as at_test

    # Create a dataset from samples
    ds = at_test.make_dataset(tmp_path, [sample1, sample2])

    # Generate random samples
    samples = at_test.make_samples(MyType, n=100)

    # Use mock atmosphere client
    client = at_test.MockAtmosphereClient()

    # Use in-memory index (SQLite backed, temporary)
    index = at_test.mock_index(tmp_path)

Pytest fixtures (available when ``atdata`` is installed)::

    def test_something(mock_atmosphere):
        client = mock_atmosphere
        client.login("user", "pass")
        ...
"""

from __future__ import annotations

import tempfile
import uuid
from dataclasses import fields as dc_fields
from pathlib import Path
from typing import Any, Sequence, Type, TypeVar

import numpy as np
import webdataset as wds

import atdata
from atdata import Dataset, PackableSample
from atdata.local._index import Index
from atdata.providers._sqlite import SqliteProvider

ST = TypeVar("ST")


# ---------------------------------------------------------------------------
# Mock Atmosphere Client
# ---------------------------------------------------------------------------


class MockAtmosphereClient:
    """In-memory mock of ``AtmosphereClient`` for testing.

    Simulates login, schema publishing, dataset publishing, and record
    retrieval without requiring a live ATProto PDS.

    Examples:
        >>> client = MockAtmosphereClient()
        >>> client.login("alice.test", "password")
        >>> client.did
        'did:plc:mock000000000000'
    """

    def __init__(
        self,
        did: str = "did:plc:mock000000000000",
        handle: str = "test.mock.social",
    ) -> None:
        self.did = did
        self.handle = handle
        self._logged_in = False
        self._records: dict[str, dict[str, Any]] = {}
        self._schemas: dict[str, dict[str, Any]] = {}
        self._datasets: dict[str, dict[str, Any]] = {}
        self._blobs: dict[str, bytes] = {}
        self._session_string = "mock-session-string"
        self._call_log: list[tuple[str, dict[str, Any]]] = []

    def login(self, handle: str, password: str) -> dict[str, Any]:
        """Simulate login. Always succeeds."""
        self._logged_in = True
        self.handle = handle
        self._call_log.append(("login", {"handle": handle}))
        return {"did": self.did, "handle": self.handle}

    @property
    def is_authenticated(self) -> bool:
        return self._logged_in

    def export_session_string(self) -> str:
        return self._session_string

    def create_record(
        self,
        collection: str,
        record: dict[str, Any],
        rkey: str | None = None,
    ) -> str:
        """Simulate creating a record. Returns a mock AT URI."""
        key = rkey or uuid.uuid4().hex[:12]
        uri = f"at://{self.did}/{collection}/{key}"
        self._records[uri] = record
        self._call_log.append(
            ("create_record", {"collection": collection, "rkey": key, "uri": uri})
        )
        return uri

    def get_record(self, uri: str) -> dict[str, Any]:
        """Retrieve a previously created record by URI."""
        if uri not in self._records:
            raise KeyError(f"Record not found: {uri}")
        return self._records[uri]

    def list_records(self, collection: str) -> list[dict[str, Any]]:
        """List records for a collection."""
        return [
            {"uri": uri, "value": rec}
            for uri, rec in self._records.items()
            if collection in uri
        ]

    def upload_blob(self, data: bytes) -> dict[str, Any]:
        """Simulate uploading a blob. Returns a mock blob ref."""
        ref = f"blob:{uuid.uuid4().hex[:16]}"
        self._blobs[ref] = data
        self._call_log.append(("upload_blob", {"ref": ref, "size": len(data)}))
        return {"ref": {"$link": ref}, "mimeType": "application/octet-stream"}

    def get_blob(self, did: str, cid: str) -> bytes:
        """Retrieve a previously uploaded blob."""
        if cid not in self._blobs:
            raise KeyError(f"Blob not found: {cid}")
        return self._blobs[cid]

    def reset(self) -> None:
        """Clear all stored state."""
        self._records.clear()
        self._schemas.clear()
        self._datasets.clear()
        self._blobs.clear()
        self._call_log.clear()
        self._logged_in = False


# ---------------------------------------------------------------------------
# Dataset Factory
# ---------------------------------------------------------------------------


def make_dataset(
    path: Path,
    samples: Sequence[PackableSample],
    *,
    name: str = "test",
    sample_type: type | None = None,
) -> Dataset:
    """Create a ``Dataset`` from a list of samples.

    Writes the samples to a WebDataset tar file in *path* and returns a
    ``Dataset`` configured to read them back.

    Args:
        path: Directory where the tar file will be created.
        samples: Sequence of ``PackableSample`` (or ``@packable``) instances.
        name: Filename prefix for the tar file.
        sample_type: Explicit sample type for the Dataset generic parameter.
            If ``None``, inferred from the first sample.

    Returns:
        A ``Dataset`` ready for iteration.

    Examples:
        >>> ds = make_dataset(tmp_path, [MySample(x=1), MySample(x=2)])
        >>> assert len(list(ds.ordered())) == 2
    """
    if not samples:
        raise ValueError("samples must be non-empty")

    tar_path = path / f"{name}-000000.tar"
    tar_path.parent.mkdir(parents=True, exist_ok=True)

    with wds.writer.TarWriter(str(tar_path)) as writer:
        for sample in samples:
            writer.write(sample.as_wds)

    st = sample_type or type(samples[0])
    return Dataset[st](url=str(tar_path))


def make_samples(
    sample_type: Type[ST], n: int = 10, seed: int | None = None
) -> list[ST]:
    """Generate *n* random instances of a ``@packable`` sample type.

    Inspects the dataclass fields and generates appropriate random data:
    - ``str`` fields get ``"field_name_0"``, ``"field_name_1"``, etc.
    - ``int`` fields get sequential integers
    - ``float`` fields get random floats in [0, 1)
    - ``bool`` fields alternate True/False
    - ``bytes`` fields get random 16 bytes
    - NDArray fields get random ``(4, 4)`` float32 arrays

    Args:
        sample_type: A ``@packable``-decorated class or ``PackableSample`` subclass.
        n: Number of samples to generate.
        seed: Optional random seed for reproducibility.

    Returns:
        List of *n* sample instances.

    Examples:
        >>> @atdata.packable
        ... class Point:
        ...     x: float
        ...     y: float
        ...     label: str
        >>> points = make_samples(Point, n=5, seed=42)
        >>> len(points)
        5
    """
    rng = np.random.default_rng(seed)
    result: list[ST] = []

    for i in range(n):
        kwargs: dict[str, Any] = {}
        for field in dc_fields(sample_type):
            type_str = str(field.type)
            fname = field.name

            if field.type is str or type_str == "str":
                kwargs[fname] = f"{fname}_{i}"
            elif field.type is int or type_str == "int":
                kwargs[fname] = i
            elif field.type is float or type_str == "float":
                kwargs[fname] = float(rng.random())
            elif field.type is bool or type_str == "bool":
                kwargs[fname] = i % 2 == 0
            elif field.type is bytes or type_str == "bytes":
                kwargs[fname] = rng.bytes(16)
            elif "NDArray" in type_str or "ndarray" in type_str.lower():
                kwargs[fname] = rng.standard_normal((4, 4)).astype(np.float32)
            elif "list" in type_str.lower():
                kwargs[fname] = [f"{fname}_{i}_{j}" for j in range(3)]
            elif "None" in type_str:
                # Optional field — leave at default
                if field.default is not field.default_factory:  # type: ignore[attr-defined]
                    continue
            else:
                kwargs[fname] = f"{fname}_{i}"

        result.append(sample_type(**kwargs))

    return result


# ---------------------------------------------------------------------------
# Mock Index
# ---------------------------------------------------------------------------


def mock_index(path: Path | None = None, **kwargs: Any) -> Index:
    """Create an in-memory SQLite-backed ``Index`` for testing.

    No Redis or external services required.

    Args:
        path: Directory for the SQLite database file. If ``None``, uses
            a temporary directory.
        **kwargs: Additional keyword arguments passed to ``Index()``.

    Returns:
        An ``Index`` instance backed by a temporary SQLite database.

    Examples:
        >>> index = mock_index(tmp_path)
        >>> ref = index.publish_schema(MyType, version="1.0.0")
    """
    if path is None:
        path = Path(tempfile.mkdtemp())
    db_path = path / "test_index.db"
    provider = SqliteProvider(str(db_path))
    return Index(provider=provider, atmosphere=None, **kwargs)


# ---------------------------------------------------------------------------
# Pytest plugin (fixtures auto-discovered when atdata is installed)
# ---------------------------------------------------------------------------

try:
    import pytest

    @pytest.fixture
    def mock_atmosphere():
        """Provide a fresh ``MockAtmosphereClient`` for each test."""
        client = MockAtmosphereClient()
        client.login("test.mock.social", "test-password")
        yield client
        client.reset()

    @pytest.fixture
    def tmp_dataset(tmp_path: Path):
        """Provide a small ``Dataset[SharedBasicSample]`` with 10 samples.

        Uses ``SharedBasicSample`` (name: str, value: int) from the test suite.
        """

        @atdata.packable
        class _TmpSample:
            name: str
            value: int

        samples = [_TmpSample(name=f"s{i}", value=i) for i in range(10)]
        return make_dataset(tmp_path, samples, sample_type=_TmpSample)

    @pytest.fixture
    def tmp_index(tmp_path: Path):
        """Provide a fresh SQLite-backed ``Index`` for each test."""
        return mock_index(tmp_path)

except ImportError:
    # pytest not installed — skip fixture registration
    _no_pytest = True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "MockAtmosphereClient",
    "make_dataset",
    "make_samples",
    "mock_index",
]
