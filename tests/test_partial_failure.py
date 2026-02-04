"""Tests for PartialFailureError and Dataset.process_shards."""

from pathlib import Path

import pytest
import webdataset as wds

import atdata
from atdata import Dataset, PartialFailureError
from atdata.testing import make_dataset


@atdata.packable
class PFSample:
    name: str
    value: int


def _make_multi_shard_dataset(tmp_path: Path, n_shards: int = 3, per_shard: int = 5):
    """Create a dataset with multiple shards."""
    tar_paths = []
    for s in range(n_shards):
        tar_path = tmp_path / f"data-{s:06d}.tar"
        with wds.writer.TarWriter(str(tar_path)) as writer:
            for i in range(per_shard):
                sample = PFSample(name=f"s{s}_{i}", value=s * 100 + i)
                writer.write(sample.as_wds)
        tar_paths.append(str(tar_path))

    brace = str(tmp_path / ("data-{000000..%06d}.tar" % (n_shards - 1)))
    return Dataset[PFSample](brace)


# ---------------------------------------------------------------------------
# PartialFailureError
# ---------------------------------------------------------------------------


class TestPartialFailureError:
    def test_attributes(self):
        err = PartialFailureError(
            succeeded_shards=["a.tar", "b.tar"],
            failed_shards=["c.tar"],
            errors={"c.tar": ValueError("bad")},
            results={"a.tar": 10, "b.tar": 20},
        )
        assert err.succeeded_shards == ["a.tar", "b.tar"]
        assert err.failed_shards == ["c.tar"]
        assert "c.tar" in err.errors
        assert err.results["a.tar"] == 10

    def test_message_format(self):
        err = PartialFailureError(
            succeeded_shards=["a.tar"],
            failed_shards=["b.tar", "c.tar"],
            errors={"b.tar": RuntimeError("oops"), "c.tar": IOError("gone")},
            results={"a.tar": 1},
        )
        msg = str(err)
        assert "2/3 shards failed" in msg
        assert "b.tar" in msg
        assert ".succeeded_shards" in msg

    def test_truncation_beyond_5(self):
        failed = [f"shard-{i}.tar" for i in range(8)]
        errors = {s: ValueError("err") for s in failed}
        err = PartialFailureError(
            succeeded_shards=[],
            failed_shards=failed,
            errors=errors,
            results={},
        )
        msg = str(err)
        assert "and 3 more" in msg

    def test_is_atdata_error(self):
        err = PartialFailureError([], ["x.tar"], {"x.tar": ValueError()}, {})
        assert isinstance(err, atdata.AtdataError)


# ---------------------------------------------------------------------------
# Dataset.process_shards
# ---------------------------------------------------------------------------


class TestProcessShards:
    def test_all_succeed(self, tmp_path):
        ds = _make_multi_shard_dataset(tmp_path, n_shards=2, per_shard=3)
        results = ds.process_shards(lambda samples: len(samples))
        assert len(results) == 2
        assert all(v == 3 for v in results.values())

    def test_partial_failure(self, tmp_path):
        ds = _make_multi_shard_dataset(tmp_path, n_shards=3, per_shard=2)

        call_count = 0

        def failing_fn(samples):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("shard 2 failed")
            return len(samples)

        with pytest.raises(PartialFailureError) as exc_info:
            ds.process_shards(failing_fn)

        err = exc_info.value
        assert len(err.succeeded_shards) == 2
        assert len(err.failed_shards) == 1
        assert isinstance(err.errors[err.failed_shards[0]], RuntimeError)
        assert all(v == 2 for v in err.results.values())

    def test_retry_failed_shards(self, tmp_path):
        ds = _make_multi_shard_dataset(tmp_path, n_shards=3, per_shard=2)

        first_call = True

        def sometimes_fails(samples):
            nonlocal first_call
            shard_name = samples[0].name.split("_")[0]
            if first_call and shard_name == "s1":
                first_call = False
                raise RuntimeError("transient")
            return len(samples)

        with pytest.raises(PartialFailureError) as exc_info:
            ds.process_shards(sometimes_fails)

        # Retry just the failed shards
        retry_results = ds.process_shards(
            sometimes_fails, shards=exc_info.value.failed_shards
        )
        assert len(retry_results) == 1
        assert all(v == 2 for v in retry_results.values())

    def test_explicit_shard_list(self, tmp_path):
        ds = _make_multi_shard_dataset(tmp_path, n_shards=3, per_shard=2)
        all_shards = ds.list_shards()
        # Process only first shard
        results = ds.process_shards(len, shards=all_shards[:1])
        assert len(results) == 1

    def test_single_shard(self, tmp_path):
        @atdata.packable
        class S:
            v: int

        ds = make_dataset(tmp_path, [S(v=i) for i in range(4)])
        results = ds.process_shards(lambda samples: sum(s.v for s in samples))
        assert len(results) == 1
        assert list(results.values())[0] == 0 + 1 + 2 + 3


# ---------------------------------------------------------------------------
# Checkpoint / Resume
# ---------------------------------------------------------------------------


class TestCheckpointResume:
    def test_checkpoint_persists_on_partial_failure(self, tmp_path):
        ds = _make_multi_shard_dataset(tmp_path, n_shards=3, per_shard=2)
        ckpt = tmp_path / "checkpoint.txt"
        call_count = 0

        def failing_fn(samples):
            nonlocal call_count
            call_count += 1
            if call_count == 3:
                raise RuntimeError("fail")
            return len(samples)

        with pytest.raises(PartialFailureError):
            ds.process_shards(failing_fn, checkpoint=ckpt)

        assert ckpt.exists()
        lines = ckpt.read_text().splitlines()
        assert len(lines) == 2  # 2 succeeded, 1 failed

    def test_resume_skips_checkpointed_shards(self, tmp_path):
        ds = _make_multi_shard_dataset(tmp_path, n_shards=3, per_shard=2)
        ckpt = tmp_path / "checkpoint.txt"
        all_shards = ds.list_shards()
        # Pre-populate checkpoint with first two shards
        ckpt.write_text(all_shards[0] + "\n" + all_shards[1] + "\n")

        processed_shards: list[str] = []

        def track_fn(samples):
            processed_shards.append(samples[0].name.split("_")[0])
            return len(samples)

        ds.process_shards(track_fn, checkpoint=ckpt)
        # Only the third shard should have been processed
        assert len(processed_shards) == 1

    def test_checkpoint_deleted_on_full_success(self, tmp_path):
        ds = _make_multi_shard_dataset(tmp_path, n_shards=2, per_shard=2)
        ckpt = tmp_path / "checkpoint.txt"
        ds.process_shards(len, checkpoint=ckpt)
        assert not ckpt.exists()

    def test_checkpoint_none_is_default_behavior(self, tmp_path):
        ds = _make_multi_shard_dataset(tmp_path, n_shards=2, per_shard=2)
        results = ds.process_shards(len)
        assert len(results) == 2

    def test_all_checkpointed_returns_empty(self, tmp_path):
        ds = _make_multi_shard_dataset(tmp_path, n_shards=2, per_shard=2)
        ckpt = tmp_path / "checkpoint.txt"
        all_shards = ds.list_shards()
        ckpt.write_text("\n".join(all_shards) + "\n")
        results = ds.process_shards(len, checkpoint=ckpt)
        assert results == {}


# ---------------------------------------------------------------------------
# on_shard_error callback
# ---------------------------------------------------------------------------


class TestOnShardError:
    def test_callback_invoked_on_failure(self, tmp_path):
        ds = _make_multi_shard_dataset(tmp_path, n_shards=3, per_shard=2)
        errors_seen: list[tuple[str, str]] = []

        def error_cb(shard_id: str, exc: Exception) -> None:
            errors_seen.append((shard_id, str(exc)))

        call_count = 0

        def failing_fn(samples):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("shard 2 failed")
            return len(samples)

        with pytest.raises(PartialFailureError):
            ds.process_shards(failing_fn, on_shard_error=error_cb)

        assert len(errors_seen) == 1
        assert "shard 2 failed" in errors_seen[0][1]

    def test_callback_not_invoked_on_success(self, tmp_path):
        ds = _make_multi_shard_dataset(tmp_path, n_shards=2, per_shard=2)
        errors_seen: list[str] = []
        ds.process_shards(len, on_shard_error=lambda s, e: errors_seen.append(s))
        assert len(errors_seen) == 0
