"""ATProto SDK signature compatibility tests.

These tests instantiate a real atproto Client (with mocked network) to verify
that our Atmosphere wrapper calls SDK methods with compatible signatures.
This catches TypeErrors like passing unsupported kwargs that unit tests with
unspecced Mocks would silently accept.

The key technique: we patch ``ClientRaw._invoke`` (the lowest-level dispatch
point for all ATProto XRPC calls) so no network I/O occurs, while keeping the
full SDK method-signature validation chain intact.  We also set a far-future
JWT expiry so ``Client._invoke``'s session-refresh guard passes through to the
real ``super()._invoke`` call without attempting a token refresh.
"""

from __future__ import annotations

import time
from unittest.mock import Mock, patch

import pytest

from atproto_client.client.raw import ClientRaw

from atdata.atmosphere.client import Atmosphere


def _make_invoke_mock(json_response: dict):
    """Return a mock _invoke that returns a canned JSON response."""
    response = Mock()
    response.content = json_response
    return Mock(return_value=response)


@pytest.fixture
def atmosphere_real_sdk():
    """Create an Atmosphere wrapper backed by a real atproto Client.

    The Client is real (so method signature validation happens in the SDK
    namespace layer), but ``ClientRaw._invoke`` is patched so no HTTP calls
    are made.  Authentication is simulated by injecting a mock session with
    a far-future JWT expiry so the refresh guard in ``Client._invoke`` is a
    no-op pass-through.
    """
    from atproto import Client

    client = Client()

    # Build a mock session that satisfies the SDK's refresh-check guard.
    # Client._invoke checks: self._session.access_jwt_payload.exp (int)
    session = Mock()
    session.access_jwt = "fake-jwt"
    session.refresh_jwt = "fake-refresh"
    session.did = "did:plc:compat-test"
    session.handle = "compat.test"
    # Far-future expiry so _should_refresh_session() returns False
    session.access_jwt_payload = Mock()
    session.access_jwt_payload.exp = int(time.time()) + 3600
    client._session = session
    client.me = Mock(did="did:plc:compat-test", handle="compat.test")

    atmo = Atmosphere(_client=client)
    # Set the Atmosphere wrapper's own session dict so is_authenticated is True
    atmo._session = {"did": "did:plc:compat-test", "handle": "compat.test"}

    return atmo


class TestUploadBlobSignature:
    """Verify upload_blob calls the SDK with a compatible signature."""

    def test_timeout_kwarg_reaches_sdk(self, atmosphere_real_sdk):
        """The timeout kwarg must pass through the SDK namespace layer."""
        invoke_mock = _make_invoke_mock(
            {
                "blob": {
                    "ref": {"$link": "bafkreitest"},
                    "mimeType": "application/x-tar",
                    "size": 100,
                }
            }
        )
        with patch.object(ClientRaw, "_invoke", invoke_mock):
            result = atmosphere_real_sdk.upload_blob(b"test data", timeout=120.0)

        assert result["$type"] == "blob"
        assert result["ref"]["$link"] == "bafkreitest"
        # Verify timeout was passed through to _invoke
        call_kwargs = invoke_mock.call_args.kwargs
        assert call_kwargs.get("timeout") == 120.0

    def test_default_timeout_heuristic(self, atmosphere_real_sdk):
        """Default timeout=None computes a heuristic without TypeError."""
        invoke_mock = _make_invoke_mock(
            {
                "blob": {
                    "ref": {"$link": "bafkrei123"},
                    "mimeType": "application/octet-stream",
                    "size": 50,
                }
            }
        )
        with patch.object(ClientRaw, "_invoke", invoke_mock):
            result = atmosphere_real_sdk.upload_blob(b"x" * 100)

        assert result["$type"] == "blob"
        call_kwargs = invoke_mock.call_args.kwargs
        assert call_kwargs.get("timeout") == 60.0  # max(60, 30 + 0.0001)


class TestCreateRecordSignature:
    """Verify create_record calls the SDK with a compatible signature."""

    def test_basic_create(self, atmosphere_real_sdk):
        invoke_mock = _make_invoke_mock(
            {
                "uri": "at://did:plc:compat-test/col/abc",
                "cid": "bafytest",
            }
        )
        with patch.object(ClientRaw, "_invoke", invoke_mock):
            result = atmosphere_real_sdk.create_record(
                collection="app.bsky.feed.post",
                record={"text": "hello"},
            )

        assert result.authority == "did:plc:compat-test"


class TestListRecordsSignature:
    """Verify list_records calls the SDK with a compatible signature."""

    def test_basic_list(self, atmosphere_real_sdk):
        invoke_mock = _make_invoke_mock(
            {
                "records": [],
                "cursor": None,
            }
        )
        with patch.object(ClientRaw, "_invoke", invoke_mock):
            records, cursor = atmosphere_real_sdk.list_records(
                collection="app.bsky.feed.post",
                repo="did:plc:compat-test",
            )

        assert records == []


class TestGetRecordSignature:
    """Verify get_record calls the SDK with a compatible signature."""

    def test_basic_get(self, atmosphere_real_sdk):
        invoke_mock = _make_invoke_mock(
            {
                "uri": "at://did:plc:compat-test/col/key",
                "cid": "bafytest",
                "value": {"field": "value"},
            }
        )
        with patch.object(ClientRaw, "_invoke", invoke_mock):
            result = atmosphere_real_sdk.get_record("at://did:plc:compat-test/col/key")

        assert result["field"] == "value"


class TestDeleteRecordSignature:
    """Verify delete_record calls the SDK with a compatible signature."""

    def test_basic_delete(self, atmosphere_real_sdk):
        invoke_mock = _make_invoke_mock({})
        with patch.object(ClientRaw, "_invoke", invoke_mock):
            atmosphere_real_sdk.delete_record("at://did:plc:compat-test/col/key")

        invoke_mock.assert_called_once()


class TestExportSessionSignature:
    """Verify export_session calls the SDK with a compatible signature."""

    def test_export(self, atmosphere_real_sdk):
        # export_session_string is a real SDK method; with a mock session
        # it returns a serialized session string
        with patch.object(
            type(atmosphere_real_sdk._client),
            "export_session_string",
            return_value="mock-exported-session",
        ):
            result = atmosphere_real_sdk.export_session()
        assert isinstance(result, str)
