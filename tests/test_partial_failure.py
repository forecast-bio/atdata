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
