"""E2E tests for ATProto session management flows.

Exercises login, session export/import round-trips, unauthenticated access,
and error handling for bad credentials and malformed session strings against
a real PDS.

Requires ``ATPROTO_TEST_HANDLE`` and ``ATPROTO_TEST_PASSWORD`` environment
variables.  Tests that need credentials skip when they are unavailable.
Tests with intentionally bad credentials run unconditionally since they
don't need a valid account.
"""

from __future__ import annotations

import pytest
from atproto.exceptions import AtProtocolError

from atdata.atmosphere import Atmosphere


# ── Happy path: login and session properties ─────────────────────


class TestSessionLogin:
    """Login with valid credentials and verify session properties.

    Uses one fresh ``Atmosphere.login()`` call per test because the
    purpose of this class is to exercise the login path itself.
    """

    def test_login_with_credentials(self, atproto_credentials: tuple[str, str]):
        """Login with handle/password returns authenticated client."""
        handle, password = atproto_credentials
        atmo = Atmosphere.login(handle, password)
        assert atmo.is_authenticated
        assert atmo.did.startswith("did:plc:")
        assert atmo.handle == handle

    def test_login_populates_did_and_handle(self, atproto_credentials: tuple[str, str]):
        """Login populates both did and handle with non-empty strings."""
        handle, password = atproto_credentials
        atmo = Atmosphere.login(handle, password)
        assert len(atmo.did) > len("did:plc:")
        assert isinstance(atmo.handle, str)
        assert len(atmo.handle) > 0


# ── Session export / import round-trip ───────────────────────────


class TestSessionExportImport:
    """Session export/import round-trip tests.

    Reuses the session-scoped ``atproto_client`` fixture where possible
    to avoid redundant logins and PDS rate-limit pressure.
    """

    def test_export_returns_nonempty_string(self, atproto_client: Atmosphere):
        """export_session() returns a non-empty string."""
        session_str = atproto_client.export_session()
        assert isinstance(session_str, str)
        assert len(session_str) > 0

    def test_session_roundtrip_preserves_identity(self, atproto_client: Atmosphere):
        """Exported session can be used to create a new client with same identity."""
        session_str = atproto_client.export_session()

        atmo2 = Atmosphere.from_session(session_str)
        assert atmo2.is_authenticated
        assert atmo2.did == atproto_client.did
        assert atmo2.handle == atproto_client.handle

    def test_reimported_client_can_export_again(self, atproto_client: Atmosphere):
        """A client created from an imported session can re-export."""
        session1 = atproto_client.export_session()

        atmo2 = Atmosphere.from_session(session1)
        session2 = atmo2.export_session()
        assert isinstance(session2, str)
        assert len(session2) > 0

    def test_export_requires_authentication(self):
        """export_session() raises ValueError when not authenticated."""
        atmo = Atmosphere()
        with pytest.raises(ValueError, match="Not authenticated"):
            atmo.export_session()


# ── Unauthenticated access ───────────────────────────────────────


class TestUnauthenticatedAccess:
    """Unauthenticated client behavior."""

    def test_unauthenticated_is_not_authenticated(self):
        """Unauthenticated client reports is_authenticated=False."""
        atmo = Atmosphere()
        assert not atmo.is_authenticated

    def test_unauthenticated_did_raises(self):
        """Accessing did on unauthenticated client raises ValueError."""
        atmo = Atmosphere()
        with pytest.raises(ValueError, match="Not authenticated"):
            _ = atmo.did

    def test_unauthenticated_handle_raises(self):
        """Accessing handle on unauthenticated client raises ValueError."""
        atmo = Atmosphere()
        with pytest.raises(ValueError, match="Not authenticated"):
            _ = atmo.handle


# ── Error cases ──────────────────────────────────────────────────


class TestSessionErrors:
    """Error handling for bad credentials and malformed sessions."""

    def test_bad_credentials_raises(self):
        """Bad credentials raise an AtProtocolError from the SDK."""
        with pytest.raises(AtProtocolError):
            Atmosphere.login("invalid-handle-does-not-exist.test", "wrong-password")

    def test_empty_handle_raises(self):
        """Empty handle raises a ValueError from the SDK client."""
        with pytest.raises(ValueError):
            Atmosphere.login("", "some-password")

    def test_malformed_session_string_raises(self):
        """Completely invalid session string raises an error."""
        with pytest.raises(Exception):
            Atmosphere.from_session("not-a-valid-session-string-at-all")

    def test_empty_session_string_raises(self):
        """Empty session string raises an error."""
        with pytest.raises(Exception):
            Atmosphere.from_session("")

    def test_truncated_session_string_is_nonfunctional(
        self, atproto_client: Atmosphere
    ):
        """A truncated (corrupted) session string produces a broken client.

        The atproto SDK silently accepts truncated session strings without
        raising during ``from_session()``.  However, the resulting client
        should fail on any authenticated operation.

        See: https://github.com/MarshalX/atproto/issues/656
        """
        session_str = atproto_client.export_session()

        # Corrupt the session by truncating it
        corrupted = session_str[: len(session_str) // 2]

        # SDK may or may not raise during session import itself
        try:
            broken = Atmosphere.from_session(corrupted)
        except Exception:
            return  # raised eagerly — acceptable behavior

        # If it didn't raise, the client should be non-functional:
        # either not authenticated or unable to make API calls.
        if not broken.is_authenticated:
            return  # correctly reports as unauthenticated

        # If it claims to be authenticated, any real API call should fail
        with pytest.raises(Exception):
            broken.export_session()


# ── Cross-session consistency ────────────────────────────────────


class TestSessionConsistency:
    """Cross-session consistency checks.

    Uses the session-scoped ``atproto_client`` plus one fresh login
    to verify identity stability without excessive PDS calls.
    """

    def test_did_stable_across_sessions(
        self, atproto_client: Atmosphere, atproto_credentials: tuple[str, str]
    ):
        """A fresh login returns the same DID as the session-scoped client."""
        handle, password = atproto_credentials
        atmo_fresh = Atmosphere.login(handle, password)
        assert atmo_fresh.did == atproto_client.did

    def test_handle_stable_across_sessions(
        self, atproto_client: Atmosphere, atproto_credentials: tuple[str, str]
    ):
        """A fresh login returns the same handle as the session-scoped client."""
        handle, password = atproto_credentials
        atmo_fresh = Atmosphere.login(handle, password)
        assert atmo_fresh.handle == atproto_client.handle

    def test_session_reuse_matches_fresh_login(
        self, atproto_client: Atmosphere, atproto_credentials: tuple[str, str]
    ):
        """Session from export matches a fresh login's identity."""
        session_str = atproto_client.export_session()
        atmo_reused = Atmosphere.from_session(session_str)

        handle, password = atproto_credentials
        atmo_fresh = Atmosphere.login(handle, password)

        assert atmo_reused.did == atmo_fresh.did
        assert atmo_reused.handle == atmo_fresh.handle
