"""AppView fallback helper for atmosphere operations.

Provides a decorator that wraps ATProto AppView calls with automatic
fallback to client-side resolution when the AppView is unreachable or
returns an error.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

import httpx

from .._exceptions import AppViewError

T = TypeVar("T")

_APPVIEW_ERRORS = (
    httpx.HTTPStatusError,
    httpx.ConnectError,
    httpx.TimeoutException,
    AppViewError,
)


def with_appview_fallback(
    appview_fn: Callable[..., T],
    fallback_fn: Callable[..., T],
    *,
    client: Any,
    operation: str,
) -> T:
    """Try an AppView operation, falling back on network/HTTP errors.

    Args:
        appview_fn: Callable that performs the AppView request.
        fallback_fn: Callable that performs the client-side fallback.
        client: The Atmosphere client (checked for ``has_appview``).
        operation: Human-readable operation name for log messages.

    Returns:
        Result from *appview_fn* if successful, otherwise from *fallback_fn*.
    """
    if getattr(client, "has_appview", False) is True:
        try:
            return appview_fn()
        except _APPVIEW_ERRORS:
            from .._logging import get_logger

            get_logger().warning(
                "AppView %s failed, falling back to client-side",
                operation,
                exc_info=True,
            )

    return fallback_fn()
