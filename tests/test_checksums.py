"""Tests for content checksum computation and verification."""

from __future__ import annotations

import atdata
from atdata._helpers import (
    ShardWriteResult,
    sha256_bytes,
    sha256_file,
    verify_checksums,
)
from atdata.index._entry import LocalDatasetEntry
from atdata.stores._disk import LocalDiskStore
from atdata.testing import make_dataset, make_samples, mock_index


@atdata.packable
class ChecksumSample:
    name: str
    value: int


# ---------------------------------------------------------------------------
# sha256 helpers
# ---------------------------------------------------------------------------


class TestSha256Helpers:
    def test_sha256_bytes_deterministic(self):
        assert sha256_bytes(b"hello") == sha256_bytes(b"hello")

    def test_sha256_bytes_known_value(self):
        expected = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        assert sha256_bytes(b"hello") == expected

    def test_sha256_bytes_different_input(self):
        assert sha256_bytes(b"hello") != sha256_bytes(b"world")

    def test_sha256_file(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"hello")
        assert sha256_file(str(f)) == sha256_bytes(b"hello")

    def test_sha256_file_accepts_path_object(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"hello")
        assert sha256_file(f) == sha256_bytes(b"hello")

    def test_sha256_file_large(self, tmp_path):
        """Chunked reading produces same result as full read."""
        data = b"x" * 100_000
        f = tmp_path / "big.bin"
        f.write_bytes(data)
        assert sha256_file(str(f)) == sha256_bytes(data)

    def test_sha256_file_empty(self, tmp_path):
        f = tmp_path / "empty.bin"
        f.write_bytes(b"")
        assert sha256_file(str(f)) == sha256_bytes(b"")

    def test_sha256_hex_length(self):
        assert len(sha256_bytes(b"test")) == 64


# ---------------------------------------------------------------------------
# ShardWriteResult
# ---------------------------------------------------------------------------


class TestShardWriteResult:
    def test_is_list(self):
        r = ShardWriteResult(["a", "b"], {"a": "abc", "b": "def"})
        assert isinstance(r, list)
        assert len(r) == 2
        assert r[0] == "a"

    def test_checksums_attribute(self):
        r = ShardWriteResult(["a"], {"a": "abc"})
        assert r.checksums == {"a": "abc"}

    def test_empty(self):
        r = ShardWriteResult([], {})
        assert len(r) == 0
        assert r.checksums == {}

    def test_list_operations(self):
        r = ShardWriteResult(["a", "b"], {"a": "x", "b": "y"})
        assert list(r) == ["a", "b"]
        assert "a" in r


# ---------------------------------------------------------------------------
# LocalDiskStore integration
# ---------------------------------------------------------------------------


class TestLocalDiskStoreChecksums:
    def test_write_returns_checksums(self, tmp_path):
        store = LocalDiskStore(root=tmp_path / "data")
        samples = make_samples(ChecksumSample, n=5)
        ds = make_dataset(tmp_path / "src", samples, sample_type=ChecksumSample)
        result = store.write_shards(ds, prefix="test")

        assert hasattr(result, "checksums")
        assert len(result.checksums) == len(result)
        for url, digest in result.checksums.items():
            assert len(digest) == 64
            assert sha256_file(url) == digest

    def test_checksums_in_metadata_via_index(self, tmp_path):
        store = LocalDiskStore(root=tmp_path / "data")
        index = mock_index(tmp_path / "idx", data_store=store)
        samples = make_samples(ChecksumSample, n=5)
        entry = index.write_samples(samples, name="test-cs")

        assert entry.metadata is not None
        assert "checksums" in entry.metadata
        assert len(entry.metadata["checksums"]) == 1  # 5 samples → 1 shard

    def test_checksums_preserved_with_existing_metadata(self, tmp_path):
        store = LocalDiskStore(root=tmp_path / "data")
        index = mock_index(tmp_path / "idx", data_store=store)
        samples = make_samples(ChecksumSample, n=5)
        entry = index.write_samples(
            samples, name="test-meta", metadata={"author": "test"}
        )

        assert entry.metadata is not None
        assert entry.metadata["author"] == "test"
        assert "checksums" in entry.metadata


# ---------------------------------------------------------------------------
# verify_checksums
# ---------------------------------------------------------------------------


class TestVerifyChecksums:
    def test_verify_pass(self, tmp_path):
        store = LocalDiskStore(root=tmp_path / "data")
        index = mock_index(tmp_path / "idx", data_store=store)
        samples = make_samples(ChecksumSample, n=5)
        entry = index.write_samples(samples, name="verify-ok")

        results = verify_checksums(entry)
        assert all(v == "ok" for v in results.values())

    def test_verify_detects_corruption(self, tmp_path):
        store = LocalDiskStore(root=tmp_path / "data")
        index = mock_index(tmp_path / "idx", data_store=store)
        samples = make_samples(ChecksumSample, n=5)
        entry = index.write_samples(samples, name="verify-bad")

        # Corrupt the first shard
        shard_path = entry.data_urls[0]
        with open(shard_path, "ab") as f:
            f.write(b"CORRUPTED")

        results = verify_checksums(entry)
        assert results[shard_path] == "mismatch"

    def test_verify_no_checksums_skips(self):
        entry = LocalDatasetEntry(
            name="legacy",
            schema_ref="local://schemas/Foo@1.0.0",
            data_urls=["/nonexistent/shard.tar"],
            metadata=None,
        )
        results = verify_checksums(entry)
        assert results["/nonexistent/shard.tar"] == "skipped"

    def test_verify_empty_metadata_skips(self):
        entry = LocalDatasetEntry(
            name="legacy",
            schema_ref="local://schemas/Foo@1.0.0",
            data_urls=["/nonexistent/shard.tar"],
            metadata={},
        )
        results = verify_checksums(entry)
        assert results["/nonexistent/shard.tar"] == "skipped"

    def test_verify_skips_remote_urls(self):
        """Remote URLs with stored checksums should be skipped, not error."""
        entry = LocalDatasetEntry(
            name="remote",
            schema_ref="local://schemas/Foo@1.0.0",
            data_urls=["s3://bucket/shard.tar", "at://did:plc:abc/blob/cid123"],
            metadata={
                "checksums": {
                    "s3://bucket/shard.tar": "abc",
                    "at://did:plc:abc/blob/cid123": "def",
                }
            },
        )
        results = verify_checksums(entry)
        assert all(v == "skipped" for v in results.values())

    def test_verify_missing_file_reports_error(self):
        entry = LocalDatasetEntry(
            name="gone",
            schema_ref="local://schemas/Foo@1.0.0",
            data_urls=["/nonexistent/shard.tar"],
            metadata={"checksums": {"/nonexistent/shard.tar": "abc123"}},
        )
        results = verify_checksums(entry)
        assert results["/nonexistent/shard.tar"].startswith("error:")

    def test_verify_multiple_shards_mixed(self, tmp_path):
        store = LocalDiskStore(root=tmp_path / "data")
        index = mock_index(tmp_path / "idx", data_store=store)
        samples = make_samples(ChecksumSample, n=20)
        entry = index.write_samples(samples, name="multi", maxcount=5)

        # Corrupt one shard, leave others intact
        first_url = entry.data_urls[0]
        with open(first_url, "ab") as f:
            f.write(b"BAD")

        results = verify_checksums(entry)
        assert results[first_url] == "mismatch"
        other_results = {u: s for u, s in results.items() if u != first_url}
        assert all(v == "ok" for v in other_results.values())
