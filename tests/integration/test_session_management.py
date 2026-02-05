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

from atdata.atmosphere import Atmosphere


# ── Happy path: login and session properties ─────────────────────


class TestSessionLogin:
    """Login with valid credentials and verify session properties."""

    def test_login_with_credentials(self, atproto_credentials: tuple[str, str]):
        """Login with handle/password returns authenticated client."""
        handle, password = atproto_credentials
        atmo = Atmosphere.login(handle, password)
        assert atmo.is_authenticated
        assert atmo.did.startswith("did:plc:")
        assert atmo.handle == handle

    def test_login_populates_did_and_handle(
        self, atproto_credentials: tuple[str, str]
    ):
        """Login populates both did and handle with non-empty strings."""
        handle, password = atproto_credentials
        atmo = Atmosphere.login(handle, password)
        assert len(atmo.did) > len("did:plc:")
        assert isinstance(atmo.handle, str)
        assert len(atmo.handle) > 0


# ── Session export / import round-trip ───────────────────────────


class TestSessionExportImport:
    """Session export/import round-trip tests."""

    def test_export_returns_nonempty_string(
        self, atproto_credentials: tuple[str, str]
    ):
        """export_session() returns a non-empty string."""
        handle, password = atproto_credentials
        atmo = Atmosphere.login(handle, password)
        session_str = atmo.export_session()
        assert isinstance(session_str, str)
        assert len(session_str) > 0

    def test_session_roundtrip_preserves_identity(
        self, atproto_credentials: tuple[str, str]
    ):
        """Exported session can be used to create a new client with same identity."""
        handle, password = atproto_credentials
        atmo1 = Atmosphere.login(handle, password)
        session_str = atmo1.export_session()

        atmo2 = Atmosphere.from_session(session_str)
        assert atmo2.is_authenticated
        assert atmo2.did == atmo1.did
        assert atmo2.handle == atmo1.handle

    def test_reimported_client_can_export_again(
        self, atproto_credentials: tuple[str, str]
    ):
        """A client created from an imported session can re-export."""
        handle, password = atproto_credentials
        atmo1 = Atmosphere.login(handle, password)
        session1 = atmo1.export_session()

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

    def test_unauthenticated_ensure_authenticated_raises(self):
        """_ensure_authenticated raises ValueError for unauthenticated client."""
        atmo = Atmosphere()
        with pytest.raises(ValueError, match="authenticated"):
            atmo._ensure_authenticated()


# ── Error cases ──────────────────────────────────────────────────


class TestSessionErrors:
    """Error handling for bad credentials and malformed sessions."""

    def test_bad_credentials_raises(self):
        """Bad credentials raise an error from the atproto SDK.

        The exact exception type depends on the PDS response (typically
        ``UnauthorizedError`` or ``BadRequestError``, both subclasses of
        ``AtProtocolError``).
        """
        with pytest.raises(Exception):
            Atmosphere.login("invalid-handle-does-not-exist.test", "wrong-password")

    def test_empty_handle_raises(self):
        """Empty handle raises an error."""
        with pytest.raises(Exception):
            Atmosphere.login("", "some-password")

    def test_malformed_session_string_raises(self):
        """Completely invalid session string raises an error."""
        with pytest.raises(Exception):
            Atmosphere.from_session("not-a-valid-session-string-at-all")

    def test_empty_session_string_raises(self):
        """Empty session string raises an error."""
        with pytest.raises(Exception):
            Atmosphere.from_session("")

    def test_truncated_session_string_raises(
        self, atproto_credentials: tuple[str, str]
    ):
        """A truncated (corrupted) session string raises an error."""
        handle, password = atproto_credentials
        atmo = Atmosphere.login(handle, password)
        session_str = atmo.export_session()

        # Corrupt the session by truncating it
        corrupted = session_str[: len(session_str) // 2]
        with pytest.raises(Exception):
            Atmosphere.from_session(corrupted)


# ── Cross-session consistency ────────────────────────────────────


class TestSessionConsistency:
    """Cross-session consistency checks."""

    def test_did_stable_across_logins(self, atproto_credentials: tuple[str, str]):
        """Two logins for the same account return the same DID."""
        handle, password = atproto_credentials
        atmo1 = Atmosphere.login(handle, password)
        atmo2 = Atmosphere.login(handle, password)
        assert atmo1.did == atmo2.did

    def test_handle_stable_across_logins(self, atproto_credentials: tuple[str, str]):
        """Two logins for the same account return the same handle."""
        handle, password = atproto_credentials
        atmo1 = Atmosphere.login(handle, password)
        atmo2 = Atmosphere.login(handle, password)
        assert atmo1.handle == atmo2.handle

    def test_session_reuse_matches_fresh_login(
        self, atproto_credentials: tuple[str, str]
    ):
        """Session from export matches a fresh login's identity."""
        handle, password = atproto_credentials
        atmo_fresh = Atmosphere.login(handle, password)
        session_str = atmo_fresh.export_session()

        atmo_reused = Atmosphere.from_session(session_str)
        atmo_fresh2 = Atmosphere.login(handle, password)

        assert atmo_reused.did == atmo_fresh2.did
        assert atmo_reused.handle == atmo_fresh2.handle
