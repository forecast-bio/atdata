"""Tests for atdata.testing module."""

import numpy as np
import pytest
from numpy.typing import NDArray

import atdata
from atdata.testing import (
    MockAtmosphereClient,
    make_dataset,
    make_samples,
    mock_index,
)


# ---------------------------------------------------------------------------
# MockAtmosphereClient
# ---------------------------------------------------------------------------


class TestMockAtmosphereClient:
    def test_login(self):
        client = MockAtmosphereClient()
        assert not client.is_authenticated
        result = client.login("alice.test", "pw")
        assert client.is_authenticated
        assert result["did"] == "did:plc:mock000000000000"
        assert result["handle"] == "alice.test"

    def test_session_string(self):
        client = MockAtmosphereClient()
        assert client.export_session_string() == "mock-session-string"

    def test_create_and_get_record(self):
        client = MockAtmosphereClient()
        uri = client.create_record("app.bsky.feed.post", {"text": "hello"})
        assert uri.startswith("at://did:plc:mock")
        record = client.get_record(uri)
        assert record["text"] == "hello"

    def test_get_record_missing_raises(self):
        client = MockAtmosphereClient()
        with pytest.raises(KeyError, match="Record not found"):
            client.get_record("at://did:plc:fake/col/missing")

    def test_list_records(self):
        client = MockAtmosphereClient()
        client.create_record("app.bsky.feed.post", {"text": "a"})
        client.create_record("app.bsky.feed.post", {"text": "b"})
        records = client.list_records("app.bsky.feed.post")
        assert len(records) == 2

    def test_upload_and_get_blob(self):
        client = MockAtmosphereClient()
        ref = client.upload_blob(b"binary data")
        cid = ref["ref"]["$link"]
        data = client.get_blob(client.did, cid)
        assert data == b"binary data"

    def test_get_blob_missing_raises(self):
        client = MockAtmosphereClient()
        with pytest.raises(KeyError, match="Blob not found"):
            client.get_blob("did:plc:x", "nonexistent")

    def test_reset(self):
        client = MockAtmosphereClient()
        client.login("u", "p")
        client.create_record("col", {"k": "v"})
        client.upload_blob(b"data")
        client.reset()
        assert not client.is_authenticated
        assert len(client._records) == 0
        assert len(client._blobs) == 0
        assert len(client._call_log) == 0

    def test_call_log(self):
        client = MockAtmosphereClient()
        client.login("u", "p")
        client.create_record("col", {"x": 1})
        assert len(client._call_log) == 2
        assert client._call_log[0][0] == "login"
        assert client._call_log[1][0] == "create_record"

    def test_custom_did_and_handle(self):
        client = MockAtmosphereClient(did="did:plc:custom", handle="custom.test")
        assert client.did == "did:plc:custom"
        assert client.handle == "custom.test"


# ---------------------------------------------------------------------------
# make_dataset
# ---------------------------------------------------------------------------


class TestMakeDataset:
    def test_basic(self, tmp_path):
        @atdata.packable
        class Pt:
            x: int
            label: str

        samples = [Pt(x=i, label=f"l{i}") for i in range(5)]
        ds = make_dataset(tmp_path, samples)
        loaded = list(ds.ordered())
        assert len(loaded) == 5
        assert loaded[0].x == 0
        assert loaded[4].label == "l4"

    def test_empty_raises(self, tmp_path):
        with pytest.raises(ValueError, match="non-empty"):
            make_dataset(tmp_path, [])

    def test_explicit_sample_type(self, tmp_path):
        @atdata.packable
        class A:
            v: int

        samples = [A(v=1)]
        ds = make_dataset(tmp_path, samples, sample_type=A)
        assert ds.sample_type is A


# ---------------------------------------------------------------------------
# make_samples
# ---------------------------------------------------------------------------


class TestMakeSamples:
    def test_basic_fields(self):
        @atdata.packable
        class Multi:
            name: str
            count: int
            score: float
            flag: bool

        samples = make_samples(Multi, n=5, seed=42)
        assert len(samples) == 5
        assert samples[0].name == "name_0"
        assert samples[0].count == 0
        assert isinstance(samples[0].score, float)
        assert samples[0].flag is True
        assert samples[1].flag is False

    def test_ndarray_field(self):
        @atdata.packable
        class Arr:
            data: NDArray
            label: str

        samples = make_samples(Arr, n=3, seed=0)
        assert len(samples) == 3
        assert isinstance(samples[0].data, np.ndarray)
        assert samples[0].data.shape == (4, 4)

    def test_seed_reproducibility(self):
        @atdata.packable
        class S:
            v: float

        a = make_samples(S, n=3, seed=99)
        b = make_samples(S, n=3, seed=99)
        assert [s.v for s in a] == [s.v for s in b]

    def test_bytes_field(self):
        @atdata.packable
        class B:
            payload: bytes

        samples = make_samples(B, n=2, seed=0)
        assert isinstance(samples[0].payload, bytes)
        assert len(samples[0].payload) == 16


# ---------------------------------------------------------------------------
# mock_index
# ---------------------------------------------------------------------------


class TestMockIndex:
    def test_publish_and_get_schema(self, tmp_path):
        @atdata.packable
        class Idx:
            val: int

        index = mock_index(tmp_path)
        ref = index.publish_schema(Idx, version="1.0.0")
        schema = index.get_schema(ref)
        assert schema["name"] == "Idx"

    def test_insert_and_get_dataset(self, tmp_path):
        @atdata.packable
        class Ds:
            x: int

        index = mock_index(tmp_path)
        samples = [Ds(x=i) for i in range(3)]
        ds = make_dataset(tmp_path, samples, sample_type=Ds)
        index.insert_dataset(ds, name="test-ds")
        entry = index.get_dataset("test-ds")
        assert entry.name == "test-ds"

    def test_no_path_uses_tempdir(self):
        index = mock_index()
        assert index is not None
