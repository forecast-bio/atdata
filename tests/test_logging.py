"""Tests for atdata._logging module."""

import logging

import pytest

import atdata
from atdata._logging import LoggerProtocol, configure_logging, get_logger, log_operation


class TestGetLogger:
    def test_default_is_stdlib(self):
        log = get_logger()
        assert isinstance(log, logging.Logger)
        assert log.name == "atdata"

    def test_satisfies_protocol(self):
        log = get_logger()
        assert isinstance(log, LoggerProtocol)


class TestConfigureLogging:
    def test_custom_logger(self):
        calls: list[tuple[str, str]] = []

        class CustomLogger:
            def debug(self, msg, *a, **kw):
                calls.append(("debug", msg % a if a else msg))

            def info(self, msg, *a, **kw):
                calls.append(("info", msg % a if a else msg))

            def warning(self, msg, *a, **kw):
                calls.append(("warning", msg % a if a else msg))

            def error(self, msg, *a, **kw):
                calls.append(("error", msg % a if a else msg))

        custom = CustomLogger()
        configure_logging(custom)
        try:
            log = get_logger()
            assert log is custom
            log.info("hello %s", "world")
            assert calls[-1] == ("info", "hello world")
        finally:
            # Restore default
            configure_logging(logging.getLogger("atdata"))

    def test_restore_default(self):
        """Ensure default logger is stdlib after test cleanup."""
        log = get_logger()
        assert isinstance(log, logging.Logger)


class TestLogOperation:
    def _capture_logger(self):
        """Return (custom_logger, calls_list) for capturing log output."""
        calls: list[tuple[str, str]] = []

        class CapturingLogger:
            def debug(self, msg, *a, **kw):
                calls.append(("debug", msg % a if a else msg))

            def info(self, msg, *a, **kw):
                calls.append(("info", msg % a if a else msg))

            def warning(self, msg, *a, **kw):
                calls.append(("warning", msg % a if a else msg))

            def error(self, msg, *a, **kw):
                calls.append(("error", msg % a if a else msg))

        return CapturingLogger(), calls

    def test_logs_start_and_complete_with_context(self):
        logger, calls = self._capture_logger()
        configure_logging(logger)
        try:
            with log_operation("test_op", x=42):
                pass
            assert any("test_op: started" in msg and "x=42" in msg for _, msg in calls)
            assert any(
                "test_op: completed in" in msg and "x=42" in msg for _, msg in calls
            )
        finally:
            configure_logging(logging.getLogger("atdata"))

    def test_logs_error_on_exception(self):
        logger, calls = self._capture_logger()
        configure_logging(logger)
        try:
            with pytest.raises(ValueError, match="boom"):
                with log_operation("fail_op"):
                    raise ValueError("boom")
            assert any(
                level == "error" and "fail_op: failed after" in msg
                for level, msg in calls
            )
        finally:
            configure_logging(logging.getLogger("atdata"))

    def test_no_context(self):
        logger, calls = self._capture_logger()
        configure_logging(logger)
        try:
            with log_operation("bare"):
                pass
            start_msgs = [msg for _, msg in calls if "bare: started" in msg]
            assert len(start_msgs) == 1
            # No parenthesized context
            assert "(" not in start_msgs[0]
        finally:
            configure_logging(logging.getLogger("atdata"))

    def test_elapsed_time_is_positive(self):
        logger, calls = self._capture_logger()
        configure_logging(logger)
        try:
            with log_operation("timed"):
                pass
            completed = [msg for _, msg in calls if "timed: completed in" in msg]
            assert len(completed) == 1
            # Extract the float from "completed in X.XXs"
            elapsed_str = completed[0].split("completed in ")[1].split("s")[0]
            assert float(elapsed_str) >= 0.0
        finally:
            configure_logging(logging.getLogger("atdata"))


class TestConfigureLoggingViaPublicApi:
    def test_atdata_configure_logging(self):
        """configure_logging is accessible from atdata top-level."""
        assert atdata.configure_logging is configure_logging

    def test_atdata_get_logger(self):
        assert atdata.get_logger is get_logger

    def test_atdata_log_operation(self):
        assert atdata.log_operation is log_operation
