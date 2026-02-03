"""Pluggable logging for atdata.

Provides a thin abstraction over Python's stdlib ``logging`` module that can
be replaced with ``structlog`` or any other logger implementing the standard
``debug``/``info``/``warning``/``error`` interface.

Usage::

    # Default: stdlib logging (no config needed)
    from atdata._logging import get_logger
    log = get_logger()
    log.info("processing shard", extra={"shard": "data-000.tar"})

    # Plug in structlog (or any compatible logger):
    import structlog
    import atdata
    atdata.configure_logging(structlog.get_logger())

The module also exports a lightweight ``LoggerProtocol`` for type checking
custom logger implementations.
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Generator
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LoggerProtocol(Protocol):
    """Minimal interface that a pluggable logger must satisfy."""

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None: ...
    def info(self, msg: str, *args: Any, **kwargs: Any) -> None: ...
    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None: ...
    def error(self, msg: str, *args: Any, **kwargs: Any) -> None: ...


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_logger: LoggerProtocol = logging.getLogger("atdata")


def configure_logging(logger: LoggerProtocol) -> None:
    """Replace the default logger with a custom implementation.

    The provided logger must implement ``debug``, ``info``, ``warning``, and
    ``error`` methods. Both ``structlog`` bound loggers and stdlib
    ``logging.Logger`` instances satisfy this interface.

    Args:
        logger: A logger instance implementing :class:`LoggerProtocol`.

    Examples:
        >>> import structlog
        >>> atdata.configure_logging(structlog.get_logger())
    """
    global _logger
    _logger = logger


def get_logger() -> LoggerProtocol:
    """Return the currently configured logger.

    Returns the stdlib ``logging.getLogger("atdata")`` by default, or
    whatever was last set via :func:`configure_logging`.
    """
    return _logger


@contextlib.contextmanager
def log_operation(op_name: str, **context: Any) -> Generator[None, None, None]:
    """Log the start, completion, and duration of an operation.

    Emits an ``info`` message on entry and on successful completion
    (with elapsed time), or an ``error`` message if an exception
    propagates out.

    Args:
        op_name: Short label for the operation (e.g. ``"write_samples"``).
        **context: Arbitrary key-value pairs included in every log message.

    Examples:
        >>> with log_operation("write_samples", shard_count=10):
        ...     do_work()
    """
    log = get_logger()
    ctx_str = ", ".join(f"{k}={v}" for k, v in context.items())
    if ctx_str:
        log.info("%s: started (%s)", op_name, ctx_str)
    else:
        log.info("%s: started", op_name)
    t0 = time.monotonic()
    try:
        yield
    except Exception:
        elapsed = time.monotonic() - t0
        if ctx_str:
            log.error("%s: failed after %.2fs (%s)", op_name, elapsed, ctx_str)
        else:
            log.error("%s: failed after %.2fs", op_name, elapsed)
        raise
    else:
        elapsed = time.monotonic() - t0
        if ctx_str:
            log.info("%s: completed in %.2fs (%s)", op_name, elapsed, ctx_str)
        else:
            log.info("%s: completed in %.2fs", op_name, elapsed)
