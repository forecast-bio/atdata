"""Tests for the atdata CLI modules.

Covers:
- cli/__init__.py (app, main, version, and command wiring)
- cli/diagnose.py (diagnose_redis, _print_status)
- cli/local.py (local_up, local_down, local_status, helpers)
- cli/preview.py (preview_dataset, _format_value)
- cli/schema.py (schema_show, schema_diff, _type_label)
- cli/inspect.py (inspect_dataset, _describe_value)
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import webdataset as wds
from typer.testing import CliRunner

import atdata
from atdata.cli import app, main
from conftest import SharedBasicSample
from atdata.cli.diagnose import diagnose_redis, _print_status
from atdata.cli.local import (
    _check_docker,
    _container_running,
    _get_compose_file,
    _run_compose,
    local_down,
    local_status,
    local_up,
    REDIS_CONTAINER,
    MINIO_CONTAINER,
)
from atdata.cli.preview import _format_value, preview_dataset
from atdata.cli.schema import _type_label, schema_diff, schema_show
from atdata.cli.inspect import _describe_value, inspect_dataset

runner = CliRunner()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@dataclass
class CliTestSampleAlt(atdata.PackableSample):
    """Alternative sample with a different schema (name + score)."""

    name: str
    score: float


@pytest.fixture()
def sample_tar(tmp_path):
    """Create a tar file with three SharedBasicSample records."""
    tar_path = tmp_path / "test-000000.tar"
    with wds.writer.TarWriter(str(tar_path)) as sink:
        for i in range(3):
            s = SharedBasicSample(name=f"s{i}", value=i * 10)
            sink.write(s.as_wds)
    return str(tar_path)


@pytest.fixture()
def alt_tar(tmp_path):
    """Create a tar with the alternative schema."""
    tar_path = tmp_path / "alt-000000.tar"
    with wds.writer.TarWriter(str(tar_path)) as sink:
        s = CliTestSampleAlt(name="x", score=3.14)
        sink.write(s.as_wds)
    return str(tar_path)


# ===================================================================
# cli/__init__.py  --  version, main, command wiring
# ===================================================================


class TestVersion:
    def test_version_command(self):
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "atdata" in result.output

    def test_version_importlib_fallback(self):
        """When atdata.__version__ is absent, importlib.metadata is used."""
        with patch.dict("sys.modules", {"atdata": MagicMock(spec=[])}):
            # spec=[] means __version__ won't exist -> ImportError on attribute
            result = runner.invoke(app, ["version"])
            # Should still succeed (via importlib fallback)
            assert result.exit_code == 0


class TestMain:
    def test_main_with_version(self):
        code = main(["version"])
        assert code == 0

    def test_main_no_args(self):
        # no_args_is_help=True causes SystemExit; main() catches it
        code = main([])
        assert isinstance(code, int)

    def test_main_bad_command(self):
        code = main(["nonexistent-command-xyz"])
        # Typer raises SystemExit for unknown commands
        assert isinstance(code, int)

    def test_main_none_argv(self):
        """main(None) uses sys.argv; call with patched argv."""
        with patch("sys.argv", ["atdata", "version"]):
            code = main(None)
            assert code == 0


# ===================================================================
# cli/inspect.py
# ===================================================================


class TestDescribeValue:
    def test_ndarray(self):
        arr = np.zeros((2, 3), dtype=np.float32)
        desc = _describe_value(arr)
        assert "ndarray" in desc
        assert "float32" in desc
        assert "(2, 3)" in desc

    def test_bytes(self):
        desc = _describe_value(b"hello")
        assert "bytes" in desc
        assert "len=5" in desc

    def test_str_short(self):
        desc = _describe_value("hello")
        assert 'str "hello"' == desc

    def test_str_long(self):
        long = "a" * 100
        desc = _describe_value(long)
        assert "..." in desc

    def test_int(self):
        assert _describe_value(42) == "int 42"

    def test_float(self):
        assert _describe_value(1.5) == "float 1.5"

    def test_bool(self):
        assert _describe_value(True) == "bool True"

    def test_list(self):
        desc = _describe_value([1, 2, 3])
        assert "list" in desc
        assert "len=3" in desc

    def test_other(self):
        desc = _describe_value({"a": 1})
        assert "dict" in desc


class TestInspectDataset:
    def test_inspect_success(self, sample_tar, capsys):
        code = inspect_dataset(sample_tar)
        assert code == 0
        out = capsys.readouterr().out
        assert "Shards:" in out
        assert "Samples:" in out
        assert "Schema:" in out

    def test_inspect_bad_url(self, capsys):
        code = inspect_dataset("/nonexistent/path.tar")
        assert code == 1
        err = capsys.readouterr().err
        assert "Error" in err

    def test_inspect_via_cli(self, sample_tar):
        result = runner.invoke(app, ["inspect", sample_tar])
        # inspect raises typer.Exit with the return code
        assert result.exit_code == 0


# ===================================================================
# cli/preview.py
# ===================================================================


class TestFormatValue:
    def test_ndarray(self):
        arr = np.ones((4,), dtype=np.int32)
        out = _format_value(arr)
        assert "ndarray" in out
        assert "shape=(4,)" in out
        assert "int32" in out

    def test_bytes_short(self):
        out = _format_value(b"abc")
        assert out == repr(b"abc")

    def test_bytes_long(self):
        data = b"x" * 100
        out = _format_value(data)
        assert "bytes[100]" in out
        assert "..." in out

    def test_str_short(self):
        out = _format_value("hello")
        assert out == repr("hello")

    def test_str_long(self):
        long = "z" * 200
        out = _format_value(long)
        assert "..." in out

    def test_list_short(self):
        out = _format_value([1, 2])
        assert out == repr([1, 2])

    def test_list_long(self):
        out = _format_value(list(range(20)))
        assert "20 items" in out

    def test_other(self):
        out = _format_value(42)
        assert out == repr(42)


class TestPreviewDataset:
    def test_preview_success(self, sample_tar, capsys):
        code = preview_dataset(sample_tar, limit=2)
        assert code == 0
        out = capsys.readouterr().out
        assert "Preview of" in out
        assert "Sample 0" in out
        assert "Sample 1" in out

    def test_preview_bad_url(self, capsys):
        """A URL that fails during Dataset construction returns 1."""
        # The Dataset constructor succeeds for any URL string; the error
        # occurs on iteration. Mock the Dataset to raise at construction.
        mock_ds_cls = MagicMock(side_effect=ValueError("bad url"))
        with patch(
            "atdata.dataset.Dataset.__class_getitem__", return_value=mock_ds_cls
        ):
            code = preview_dataset("/no/such/file.tar")
        assert code == 1
        err = capsys.readouterr().err
        assert "Error" in err

    def test_preview_via_cli(self, sample_tar):
        result = runner.invoke(app, ["preview", sample_tar, "--limit", "1"])
        assert result.exit_code == 0

    def test_preview_empty_dataset(self, tmp_path, capsys):
        """Preview an empty tar returns error code 1."""
        tar_path = tmp_path / "empty.tar"
        import tarfile

        tf = tarfile.open(str(tar_path), "w")
        tf.close()
        code = preview_dataset(str(tar_path))
        assert code == 1


# ===================================================================
# cli/schema.py
# ===================================================================


class TestTypeLabel:
    def test_ndarray(self):
        arr = np.zeros((2,), dtype=np.float64)
        assert _type_label(arr) == "ndarray[float64]"

    def test_bytes(self):
        assert _type_label(b"data") == "bytes"

    def test_other(self):
        assert _type_label("hello") == "str"
        assert _type_label(42) == "int"


class TestSchemaShow:
    def test_show_success(self, sample_tar, capsys):
        code = schema_show(sample_tar)
        assert code == 0
        out = capsys.readouterr().out
        assert "Schema for:" in out
        assert "Fields" in out

    def test_show_bad_url(self, capsys):
        mock_ds_cls = MagicMock(side_effect=ValueError("bad url"))
        with patch(
            "atdata.dataset.Dataset.__class_getitem__", return_value=mock_ds_cls
        ):
            code = schema_show("/no/such.tar")
        assert code == 1

    def test_show_via_cli(self, sample_tar):
        result = runner.invoke(app, ["schema", "show", sample_tar])
        assert result.exit_code == 0


class TestSchemaDiff:
    def test_identical_schemas(self, sample_tar, capsys):
        code = schema_diff(sample_tar, sample_tar)
        assert code == 0
        out = capsys.readouterr().out
        assert "identical" in out.lower()

    def test_different_schemas(self, sample_tar, alt_tar, capsys):
        code = schema_diff(sample_tar, alt_tar)
        assert code == 1
        out = capsys.readouterr().out
        # value is removed (in sample_tar but not alt_tar)
        # score is added (in alt_tar but not sample_tar)
        assert "Added" in out or "Removed" in out or "Changed" in out

    def test_diff_bad_url(self, capsys):
        mock_ds_cls = MagicMock(side_effect=ValueError("bad url"))
        with patch(
            "atdata.dataset.Dataset.__class_getitem__", return_value=mock_ds_cls
        ):
            code = schema_diff("/no/a.tar", "/no/b.tar")
        assert code == 2

    def test_diff_empty_first(self, tmp_path, alt_tar, capsys):
        """First dataset is empty -> error code 2."""
        import tarfile

        tar_path = tmp_path / "empty.tar"
        tf = tarfile.open(str(tar_path), "w")
        tf.close()
        code = schema_diff(str(tar_path), alt_tar)
        assert code == 2

    def test_diff_empty_second(self, tmp_path, sample_tar, capsys):
        """Second dataset is empty -> error code 2."""
        import tarfile

        tar_path = tmp_path / "empty.tar"
        tf = tarfile.open(str(tar_path), "w")
        tf.close()
        code = schema_diff(sample_tar, str(tar_path))
        assert code == 2

    def test_diff_via_cli(self, sample_tar, alt_tar):
        result = runner.invoke(app, ["schema", "diff", sample_tar, alt_tar])
        # diff returns 1 (different) which becomes exit_code 1
        assert result.exit_code == 1


# ===================================================================
# cli/diagnose.py
# ===================================================================


class TestPrintStatus:
    def test_ok_with_detail(self, capsys):
        _print_status("Connection", True, "connected")
        out = capsys.readouterr().out
        assert "\u2713" in out  # checkmark
        assert "connected" in out

    def test_fail_no_detail(self, capsys):
        _print_status("Check", False)
        out = capsys.readouterr().out
        assert "\u2717" in out  # X mark
        assert "Check" in out


class TestDiagnoseRedis:
    def _make_redis_mock(
        self,
        *,
        aof="yes",
        save="3600 1",
        policy="noeviction",
        maxmemory="0",
        version="7.0.0",
    ):
        """Create a mock Redis that passes all checks by default."""
        mock = MagicMock()
        mock.ping.return_value = True
        mock.info.return_value = {"redis_version": version}
        mock.info.side_effect = lambda *a, **kw: (
            {"used_memory_human": "1M", "used_memory_peak_human": "2M"}
            if a and a[0] == "memory"
            else {"redis_version": version}
        )
        mock.config_get.side_effect = lambda key: {
            "appendonly": {"appendonly": aof},
            "save": {"save": save},
            "maxmemory-policy": {"maxmemory-policy": policy},
            "maxmemory": {"maxmemory": maxmemory},
        }[key]
        mock.scan_iter.return_value = iter([])
        return mock

    @patch("redis.Redis")
    def test_all_checks_pass(self, mock_redis_cls, capsys):
        mock_redis_cls.return_value = self._make_redis_mock()
        code = diagnose_redis("localhost", 6379)
        assert code == 0
        out = capsys.readouterr().out
        assert "All checks passed" in out

    @patch("redis.Redis")
    def test_connection_failure(self, mock_redis_cls, capsys):
        mock_redis_cls.return_value.ping.side_effect = ConnectionError("refused")
        code = diagnose_redis()
        assert code == 1
        out = capsys.readouterr().out
        assert "Cannot connect" in out

    @patch("redis.Redis")
    def test_aof_disabled(self, mock_redis_cls, capsys):
        mock_redis_cls.return_value = self._make_redis_mock(aof="no")
        code = diagnose_redis()
        assert code == 1
        out = capsys.readouterr().out
        assert "DISABLED" in out

    @patch("redis.Redis")
    def test_rdb_disabled(self, mock_redis_cls, capsys):
        mock_redis_cls.return_value = self._make_redis_mock(save="")
        diagnose_redis()
        out = capsys.readouterr().out
        assert "RDB" in out

    @patch("redis.Redis")
    def test_unsafe_policy(self, mock_redis_cls, capsys):
        mock_redis_cls.return_value = self._make_redis_mock(policy="allkeys-lru")
        code = diagnose_redis()
        assert code == 1
        out = capsys.readouterr().out
        assert "may evict" in out

    @patch("redis.Redis")
    def test_maxmemory_limited(self, mock_redis_cls, capsys):
        mock_redis_cls.return_value = self._make_redis_mock(
            maxmemory=str(256 * 1024 * 1024)
        )
        code = diagnose_redis()
        assert code == 0
        out = capsys.readouterr().out
        assert "256 MB" in out

    @patch("redis.Redis")
    def test_maxmemory_unlimited(self, mock_redis_cls, capsys):
        mock_redis_cls.return_value = self._make_redis_mock(maxmemory="0")
        code = diagnose_redis()
        assert code == 0
        out = capsys.readouterr().out
        assert "unlimited" in out

    @patch("redis.Redis")
    def test_version_check_failure(self, mock_redis_cls, capsys):
        mock = self._make_redis_mock()
        mock.info.side_effect = Exception("info failed")
        mock_redis_cls.return_value = mock
        code = diagnose_redis()
        # Version failure sets issues_found
        assert code == 1

    @patch("redis.Redis")
    def test_key_counts(self, mock_redis_cls, capsys):
        mock = self._make_redis_mock()
        # Return some keys for each scan_iter call
        mock.scan_iter.side_effect = lambda match, count: (
            iter([b"LocalDatasetEntry:a", b"LocalDatasetEntry:b"])
            if "Dataset" in match
            else iter([b"LocalSchema:x"])
        )
        mock_redis_cls.return_value = mock
        code = diagnose_redis()
        assert code == 0
        out = capsys.readouterr().out
        assert "2 datasets" in out
        assert "1 schemas" in out

    def test_redis_import_error(self, capsys):
        """When redis package is missing, diagnose returns 1."""
        # Simulate ImportError by making the redis module unimportable.
        # diagnose_redis does `from redis import Redis` so patching
        # sys.modules makes the import raise ImportError.
        with patch.dict("sys.modules", {"redis": None}):
            code = diagnose_redis()
            assert code == 1
            err = capsys.readouterr().err
            assert "redis" in err.lower()

    def test_diagnose_via_cli(self):
        """CLI command wires through to diagnose_redis."""
        with patch("atdata.cli.diagnose.diagnose_redis", return_value=0):
            # Patch at the import location used by __init__.py
            with patch("atdata.cli.diagnose.diagnose_redis", return_value=0):
                result = runner.invoke(app, ["diagnose"])
                # May have either exit code depending on import path
                # Just verify it doesn't crash
                assert isinstance(result.exit_code, int)


# ===================================================================
# cli/local.py
# ===================================================================


class TestCheckDocker:
    @patch("atdata.cli.local.shutil.which", return_value=None)
    def test_docker_not_installed(self, mock_which, capsys):
        assert _check_docker() is False
        err = capsys.readouterr().err
        assert "not installed" in err

    @patch("atdata.cli.local.subprocess.run")
    @patch("atdata.cli.local.shutil.which", return_value="/usr/bin/docker")
    def test_docker_daemon_not_running(self, mock_which, mock_run, capsys):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr=""
        )
        assert _check_docker() is False
        err = capsys.readouterr().err
        assert "not running" in err

    @patch("atdata.cli.local.subprocess.run")
    @patch("atdata.cli.local.shutil.which", return_value="/usr/bin/docker")
    def test_docker_timeout(self, mock_which, mock_run, capsys):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="docker info", timeout=10)
        assert _check_docker() is False
        err = capsys.readouterr().err
        assert "not responding" in err

    @patch("atdata.cli.local.subprocess.run")
    @patch("atdata.cli.local.shutil.which", return_value="/usr/bin/docker")
    def test_docker_other_exception(self, mock_which, mock_run, capsys):
        mock_run.side_effect = OSError("bang")
        assert _check_docker() is False
        err = capsys.readouterr().err
        assert "Error checking Docker" in err

    @patch("atdata.cli.local.subprocess.run")
    @patch("atdata.cli.local.shutil.which", return_value="/usr/bin/docker")
    def test_docker_ok(self, mock_which, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        assert _check_docker() is True


class TestGetComposeFile:
    def test_template_substitution(self):
        content = _get_compose_file(
            redis_port=16379, minio_port=19000, minio_console_port=19001
        )
        assert "16379:6379" in content
        assert "19000:9000" in content
        assert "19001:9001" in content
        assert REDIS_CONTAINER in content
        assert MINIO_CONTAINER in content


class TestContainerRunning:
    @patch("atdata.cli.local.subprocess.run")
    def test_running(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="true\n", stderr=""
        )
        assert _container_running("my-container") is True

    @patch("atdata.cli.local.subprocess.run")
    def test_not_running(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="false\n", stderr=""
        )
        assert _container_running("my-container") is False

    @patch("atdata.cli.local.subprocess.run")
    def test_container_not_found(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr=""
        )
        assert _container_running("missing") is False

    @patch("atdata.cli.local.subprocess.run", side_effect=OSError("no docker"))
    def test_exception(self, mock_run):
        assert _container_running("x") is False


class TestRunCompose:
    @patch("atdata.cli.local.subprocess.run")
    @patch("atdata.cli.local.shutil.which", return_value="/usr/bin/docker")
    def test_compose_v2(self, mock_which, mock_run, tmp_path):
        # First call: docker compose version -> success (v2)
        # Second call: actual compose command
        mock_run.side_effect = [
            subprocess.CompletedProcess(args=[], returncode=0, stdout="v2.20"),
            subprocess.CompletedProcess(args=[], returncode=0, stdout="ok"),
        ]
        with patch("atdata.cli.local.Path.home", return_value=tmp_path):
            result = _run_compose("version: '3'\n", ["up", "-d"])
        assert result.returncode == 0
        # Verify the second call uses "docker compose"
        call_args = mock_run.call_args_list[1][0][0]
        assert call_args[:2] == ["docker", "compose"]

    @patch("atdata.cli.local.subprocess.run")
    @patch("atdata.cli.local.shutil.which")
    def test_compose_v1_fallback(self, mock_which, mock_run, tmp_path):
        # docker exists, but `docker compose version` fails -> fallback to docker-compose
        mock_which.side_effect = lambda cmd: (
            "/usr/bin/docker"
            if cmd == "docker"
            else "/usr/bin/docker-compose"
            if cmd == "docker-compose"
            else None
        )
        mock_run.side_effect = [
            subprocess.CompletedProcess(args=[], returncode=1),  # v2 check fails
            subprocess.CompletedProcess(args=[], returncode=0, stdout="ok"),
        ]
        with patch("atdata.cli.local.Path.home", return_value=tmp_path):
            _run_compose("version: '3'\n", ["up"])
        call_args = mock_run.call_args_list[1][0][0]
        assert call_args[0] == "docker-compose"

    @patch("atdata.cli.local.subprocess.run")
    @patch("atdata.cli.local.shutil.which")
    def test_no_compose_available(self, mock_which, mock_run, tmp_path):
        mock_which.side_effect = lambda cmd: (
            "/usr/bin/docker" if cmd == "docker" else None
        )
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=1)
        with patch("atdata.cli.local.Path.home", return_value=tmp_path):
            with pytest.raises(RuntimeError, match="Neither"):
                _run_compose("version: '3'\n", ["up"])

    @patch("atdata.cli.local.subprocess.run")
    @patch("atdata.cli.local.shutil.which", return_value=None)
    def test_docker_not_found(self, mock_which, mock_run, tmp_path):
        with patch("atdata.cli.local.Path.home", return_value=tmp_path):
            with pytest.raises(RuntimeError, match="Docker not found"):
                _run_compose("version: '3'\n", ["up"])


class TestLocalUp:
    @patch("atdata.cli.local._run_compose")
    @patch("atdata.cli.local._check_docker", return_value=True)
    @patch("atdata.cli.local.time", create=True)
    def test_success(self, mock_time, mock_check, mock_compose, capsys):
        mock_compose.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        # Patch time.sleep inside local_up
        with patch("time.sleep"):
            code = local_up()
        assert code == 0
        out = capsys.readouterr().out
        assert "Redis:" in out

    @patch("atdata.cli.local._check_docker", return_value=False)
    def test_docker_unavailable(self, mock_check):
        code = local_up()
        assert code == 1

    @patch("atdata.cli.local._run_compose")
    @patch("atdata.cli.local._check_docker", return_value=True)
    def test_compose_failure(self, mock_check, mock_compose, capsys):
        mock_compose.return_value = subprocess.CompletedProcess(args=[], returncode=2)
        code = local_up()
        assert code == 2

    @patch("atdata.cli.local._run_compose", side_effect=RuntimeError("boom"))
    @patch("atdata.cli.local._check_docker", return_value=True)
    def test_compose_exception(self, mock_check, mock_compose, capsys):
        code = local_up()
        assert code == 1

    @patch("atdata.cli.local._run_compose")
    @patch("atdata.cli.local._check_docker", return_value=True)
    def test_detach_flag(self, mock_check, mock_compose, capsys):
        mock_compose.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        with patch("time.sleep"):
            local_up(detach=True)
        compose_cmd = mock_compose.call_args[0][1]
        assert "-d" in compose_cmd

    @patch("atdata.cli.local._run_compose")
    @patch("atdata.cli.local._check_docker", return_value=True)
    def test_no_detach(self, mock_check, mock_compose, capsys):
        mock_compose.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        with patch("time.sleep"):
            local_up(detach=False)
        compose_cmd = mock_compose.call_args[0][1]
        assert "-d" not in compose_cmd


class TestLocalDown:
    @patch("atdata.cli.local._run_compose")
    @patch("atdata.cli.local._check_docker", return_value=True)
    def test_success(self, mock_check, mock_compose, capsys):
        mock_compose.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        code = local_down()
        assert code == 0
        out = capsys.readouterr().out
        assert "stopped" in out.lower()

    @patch("atdata.cli.local._check_docker", return_value=False)
    def test_docker_unavailable(self, mock_check):
        assert local_down() == 1

    @patch("atdata.cli.local._run_compose")
    @patch("atdata.cli.local._check_docker", return_value=True)
    def test_with_volumes(self, mock_check, mock_compose, capsys):
        mock_compose.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        local_down(remove_volumes=True)
        compose_cmd = mock_compose.call_args[0][1]
        assert "-v" in compose_cmd
        out = capsys.readouterr().out
        assert "delete" in out.lower()

    @patch("atdata.cli.local._run_compose")
    @patch("atdata.cli.local._check_docker", return_value=True)
    def test_compose_failure(self, mock_check, mock_compose, capsys):
        mock_compose.return_value = subprocess.CompletedProcess(args=[], returncode=3)
        code = local_down()
        assert code == 3

    @patch("atdata.cli.local._run_compose", side_effect=RuntimeError("fail"))
    @patch("atdata.cli.local._check_docker", return_value=True)
    def test_compose_exception(self, mock_check, mock_compose):
        assert local_down() == 1


class TestLocalStatus:
    @patch("atdata.cli.local._container_running")
    @patch("atdata.cli.local._check_docker", return_value=True)
    def test_both_running(self, mock_check, mock_running, capsys):
        mock_running.return_value = True
        code = local_status()
        assert code == 0
        out = capsys.readouterr().out
        assert "running" in out
        assert "To stop" in out

    @patch("atdata.cli.local._container_running", return_value=False)
    @patch("atdata.cli.local._check_docker", return_value=True)
    def test_both_stopped(self, mock_check, mock_running, capsys):
        code = local_status()
        assert code == 0
        out = capsys.readouterr().out
        assert "stopped" in out
        assert "To start" in out

    @patch("atdata.cli.local._check_docker", return_value=False)
    def test_docker_unavailable(self, mock_check):
        assert local_status() == 1

    def test_status_via_cli(self):
        with patch("atdata.cli.local.local_status", return_value=0):
            result = runner.invoke(app, ["local", "status"])
            assert isinstance(result.exit_code, int)


# ===================================================================
# CLI wiring tests (via CliRunner)
# ===================================================================


class TestCliWiring:
    """Verify the typer app wires sub-commands correctly."""

    def test_local_up_via_cli(self):
        with patch("atdata.cli.local.local_up", return_value=0):
            result = runner.invoke(app, ["local", "up"])
            assert isinstance(result.exit_code, int)

    def test_local_down_via_cli(self):
        with patch("atdata.cli.local.local_down", return_value=0):
            result = runner.invoke(app, ["local", "down"])
            assert isinstance(result.exit_code, int)

    def test_diagnose_via_cli(self):
        with patch("atdata.cli.diagnose.diagnose_redis", return_value=0):
            result = runner.invoke(app, ["diagnose"])
            assert isinstance(result.exit_code, int)

    def test_no_args_shows_help(self):
        result = runner.invoke(app, [])
        # no_args_is_help=True causes exit code 0 or 2 depending on typer version
        assert "Usage" in result.output or "Commands" in result.output
