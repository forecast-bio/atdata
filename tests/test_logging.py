"""Tests for atdata._logging module."""

import logging

import atdata
from atdata._logging import LoggerProtocol, configure_logging, get_logger


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


class TestConfigureLoggingViaPublicApi:
    def test_atdata_configure_logging(self):
        """configure_logging is accessible from atdata top-level."""
        assert atdata.configure_logging is configure_logging

    def test_atdata_get_logger(self):
        assert atdata.get_logger is get_logger
