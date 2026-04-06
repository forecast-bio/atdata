"""ATProto client wrapper for atdata.

This module provides the ``Atmosphere`` class which wraps the atproto SDK
client with atdata-specific helpers for publishing and querying records.
"""

from __future__ import annotations

from typing import Optional, Any

import threading

from ._types import AtUri, LEXICON_NAMESPACE

# Lazy import to avoid requiring atproto if not using atmosphere features.
# Protected by double-checked locking for thread safety.
_atproto_client_class: Optional[type] = None
_atproto_lock = threading.Lock()


def _get_atproto_client_class():
    """Lazily import the atproto Client class (thread-safe)."""
    global _atproto_client_class
    if _atproto_client_class is None:
        with _atproto_lock:
            if _atproto_client_class is None:
                try:
                    from atproto import Client

                    _atproto_client_class = Client
                except ImportError as e:
                    raise ImportError(
                        "The 'atproto' package is required for ATProto integration. "
                        "Install it with: pip install atproto"
                    ) from e
    return _atproto_client_class


def _value_to_dict(value: Any) -> dict | Any:
    """Convert an ATProto model value to a plain dict.

    Handles DotDict (to_dict), pydantic (model_dump), plain dict, and
    generic objects with __dict__.
    """
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return value.model_dump()
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return value


def _did_web_to_url(did: str) -> str:
    """Convert a ``did:web:`` DID to an HTTPS URL.

    Args:
        did: A ``did:web:`` DID string (e.g., ``did:web:datasets.atdata.blue``
            or ``did:web:localhost%3A8000``).

    Returns:
        The resolved HTTPS URL.
    """
    from urllib.parse import unquote

    if not did.startswith("did:web:"):
        raise ValueError(f"Not a did:web DID: {did}")
    host_part = did[len("did:web:") :]
    host_part = unquote(host_part)
    return f"https://{host_part}"


def _url_to_did_web(url: str) -> str:
    """Convert an HTTPS URL to a ``did:web:`` DID.

    Args:
        url: An HTTPS URL (e.g., ``https://datasets.atdata.blue``).

    Returns:
        The ``did:web:`` DID string.
    """
    from urllib.parse import urlparse, quote

    parsed = urlparse(url)
    host = parsed.hostname or ""
    port = parsed.port
    if port:
        host = f"{host}%3A{port}"
    else:
        host = quote(host, safe=".")
    return f"did:web:{host}"


class Atmosphere:
    """ATProto client wrapper for atdata operations.

    This class wraps the atproto SDK client and provides higher-level methods
    for working with atdata records (schemas, datasets, lenses).

    Examples:
        >>> atmo = Atmosphere.login("alice.bsky.social", "app-password")
        >>> print(atmo.did)
        'did:plc:...'

    Note:
        The password should be an app-specific password, not your main account
        password. Create app passwords in your Bluesky account settings.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        *,
        appview: Optional[str] = None,
        _client: Optional[Any] = None,
    ):
        """Initialize the ATProto client.

        Args:
            base_url: Optional PDS base URL. Defaults to bsky.social.
            appview: Optional AppView URL or DID. When set, XRPC queries
                are sent directly to the AppView and procedures are proxied
                through the PDS. Accepts an HTTPS URL
                (e.g., ``"https://datasets.atdata.blue"``) or a ``did:web``
                DID (e.g., ``"did:web:datasets.atdata.blue"``).
            _client: Optional pre-configured atproto Client for testing.
        """
        if _client is not None:
            self._client = _client
        else:
            Client = _get_atproto_client_class()
            self._client = Client(base_url=base_url) if base_url else Client()

        self._session: Optional[dict] = None
        self._appview_url: Optional[str] = None
        self._appview_did: Optional[str] = None
        self._httpx_client: Any = None  # lazily created httpx.Client

        if appview is not None:
            self._configure_appview(appview)

    @classmethod
    def login(
        cls,
        handle: str,
        password: str,
        *,
        base_url: Optional[str] = None,
        appview: Optional[str] = None,
    ) -> Atmosphere:
        """Create an authenticated Atmosphere client.

        Args:
            handle: Your Bluesky handle (e.g., 'alice.bsky.social').
            password: App-specific password (not your main password).
            base_url: Optional PDS base URL. Defaults to bsky.social.
            appview: Optional AppView URL or DID for XRPC queries/procedures.

        Returns:
            An authenticated Atmosphere instance.

        Raises:
            atproto.exceptions.AtProtocolError: If authentication fails.

        Examples:
            >>> atmo = Atmosphere.login("alice.bsky.social", "app-password")
            >>> index = Index(atmosphere=atmo)

            >>> atmo = Atmosphere.login(
            ...     "alice.bsky.social", "app-password",
            ...     appview="https://datasets.atdata.blue",
            ... )
        """
        instance = cls(base_url=base_url, appview=appview)
        instance._login(handle, password)
        return instance

    @classmethod
    def from_session(
        cls,
        session_string: str,
        *,
        base_url: Optional[str] = None,
        appview: Optional[str] = None,
    ) -> Atmosphere:
        """Create an Atmosphere client from an exported session string.

        This allows reusing a session without re-authenticating, which helps
        avoid rate limits on session creation.

        Args:
            session_string: Session string from ``export_session()``.
            base_url: Optional PDS base URL. Defaults to bsky.social.
            appview: Optional AppView URL or DID for XRPC queries/procedures.

        Returns:
            An authenticated Atmosphere instance.

        Examples:
            >>> session = atmo.export_session()
            >>> atmo2 = Atmosphere.from_session(session)
        """
        instance = cls(base_url=base_url, appview=appview)
        instance._login_with_session(session_string)
        return instance

    @classmethod
    def from_env(
        cls,
        *,
        base_url: Optional[str] = None,
    ) -> Atmosphere:
        """Create an Atmosphere client from environment variables.

        Reads credentials and optional AppView URL from the environment:

        - ``ATDATA_HANDLE``: Bluesky handle (required)
        - ``ATDATA_PASSWORD``: App-specific password (required)
        - ``ATDATA_APPVIEW``: AppView URL or DID (optional)
        - ``ATDATA_PDS_URL``: PDS base URL (optional, overrides *base_url*)

        Args:
            base_url: Fallback PDS base URL if ``ATDATA_PDS_URL`` is not set.

        Returns:
            An authenticated Atmosphere instance.

        Raises:
            EnvironmentError: If required environment variables are missing.
        """
        import os

        handle = os.environ.get("ATDATA_HANDLE")
        password = os.environ.get("ATDATA_PASSWORD")
        if not handle or not password:
            raise EnvironmentError(
                "ATDATA_HANDLE and ATDATA_PASSWORD environment variables are required"
            )

        pds_url = os.environ.get("ATDATA_PDS_URL", base_url)
        appview = os.environ.get("ATDATA_APPVIEW")

        return cls.login(handle, password, base_url=pds_url, appview=appview)

    def _login(self, handle: str, password: str) -> None:
        """Authenticate with the ATProto PDS.

        Args:
            handle: Your Bluesky handle (e.g., 'alice.bsky.social').
            password: App-specific password (not your main password).

        Raises:
            atproto.exceptions.AtProtocolError: If authentication fails.
        """
        profile = self._client.login(handle, password)
        self._session = {
            "did": profile.did,
            "handle": profile.handle,
        }

    def _login_with_session(self, session_string: str) -> None:
        """Authenticate using an exported session string.

        Args:
            session_string: Session string from ``export_session()``.
        """
        self._client.login(session_string=session_string)
        self._session = {
            "did": self._client.me.did,
            "handle": self._client.me.handle,
        }

    def export_session(self) -> str:
        """Export the current session for later reuse.

        Returns:
            Session string that can be passed to ``login_with_session()``.

        Raises:
            ValueError: If not authenticated.
        """
        if not self.is_authenticated:
            raise ValueError("Not authenticated")
        return self._client.export_session_string()

    @property
    def is_authenticated(self) -> bool:
        """Check if the client has a valid session."""
        return self._session is not None

    @property
    def did(self) -> str:
        """Get the DID of the authenticated user.

        Returns:
            The DID string (e.g., 'did:plc:...').

        Raises:
            ValueError: If not authenticated.
        """
        if not self._session:
            raise ValueError("Not authenticated")
        return self._session["did"]

    @property
    def handle(self) -> str:
        """Get the handle of the authenticated user.

        Returns:
            The handle string (e.g., 'alice.bsky.social').

        Raises:
            ValueError: If not authenticated.
        """
        if not self._session:
            raise ValueError("Not authenticated")
        return self._session["handle"]

    def _ensure_authenticated(self) -> None:
        """Raise if not authenticated."""
        if not self.is_authenticated:
            raise ValueError("Client must be authenticated to perform this operation")

    # ------------------------------------------------------------------ #
    # AppView configuration and XRPC transport
    # ------------------------------------------------------------------ #

    def _configure_appview(self, appview: str) -> None:
        """Parse and store AppView URL/DID.

        Args:
            appview: HTTPS URL or ``did:web:`` DID string.
        """
        if appview.startswith("did:web:"):
            self._appview_did = appview
            self._appview_url = _did_web_to_url(appview)
        elif appview.startswith("https://") or appview.startswith("http://"):
            self._appview_url = appview.rstrip("/")
            self._appview_did = _url_to_did_web(self._appview_url)
        else:
            raise ValueError(
                f"Invalid appview value: {appview!r}. "
                "Expected an HTTPS URL or did:web: DID."
            )

    @property
    def has_appview(self) -> bool:
        """Whether an AppView is configured."""
        return self._appview_url is not None

    @property
    def appview_url(self) -> str | None:
        """The resolved AppView base URL, or ``None``."""
        return self._appview_url

    @property
    def appview_did(self) -> str | None:
        """The AppView DID (``did:web:...``), or ``None``."""
        return self._appview_did

    def _get_httpx_client(self) -> Any:
        """Return a shared httpx.Client for AppView requests."""
        if self._httpx_client is None:
            import httpx

            self._httpx_client = httpx.Client(timeout=30.0)
        return self._httpx_client

    def xrpc_query(
        self,
        nsid: str,
        params: dict | None = None,
    ) -> dict:
        """Call an XRPC query (GET) on the AppView.

        Queries are public (no auth required) and sent directly to the
        AppView URL.

        Args:
            nsid: The XRPC method NSID
                (e.g., ``"science.alt.dataset.listEntries"``).
            params: Optional query parameters.

        Returns:
            The JSON response body as a dict.

        Raises:
            AppViewRequiredError: If no AppView is configured.
            AppViewUnavailableError: If the AppView is unreachable.
        """
        from .._exceptions import AppViewRequiredError, AppViewUnavailableError

        if not self.has_appview:
            raise AppViewRequiredError(nsid)

        import httpx

        url = f"{self._appview_url}/xrpc/{nsid}"
        client = self._get_httpx_client()

        try:
            response = client.get(url, params=params or {})
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code >= 500:
                raise AppViewUnavailableError(
                    self._appview_url, f"HTTP {exc.response.status_code}"
                ) from exc
            raise
        except httpx.ConnectError as exc:
            raise AppViewUnavailableError(self._appview_url, str(exc)) from exc

    def xrpc_procedure(
        self,
        nsid: str,
        input: dict | None = None,
    ) -> dict:
        """Call an XRPC procedure (POST) via PDS proxy to the AppView.

        Procedures require authentication. The request is sent to the PDS
        with an ``atproto-proxy`` header so the PDS forwards it to the
        AppView for validation before writing.

        Args:
            nsid: The XRPC method NSID
                (e.g., ``"science.alt.dataset.publishSchema"``).
            input: Optional JSON request body.

        Returns:
            The JSON response body as a dict.

        Raises:
            AppViewRequiredError: If no AppView is configured.
            AppViewUnavailableError: If the AppView is unreachable.
            ValueError: If not authenticated.
        """
        from .._exceptions import AppViewRequiredError, AppViewUnavailableError

        if not self.has_appview:
            raise AppViewRequiredError(nsid)
        self._ensure_authenticated()

        import httpx

        # POST to the PDS with atproto-proxy header
        pds_url = str(self._client._base_url).rstrip("/")

        url = f"{pds_url}/xrpc/{nsid}"
        headers = {
            "atproto-proxy": f"{self._appview_did}#atdata_appview",
        }

        # Get the current access token from the SDK client session
        if hasattr(self._client, "_session") and self._client._session:
            access_jwt = getattr(self._client._session, "access_jwt", None)
            if access_jwt:
                headers["Authorization"] = f"Bearer {access_jwt}"

        client = self._get_httpx_client()

        try:
            response = client.post(url, json=input or {}, headers=headers)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code >= 500:
                raise AppViewUnavailableError(
                    self._appview_url, f"HTTP {exc.response.status_code}"
                ) from exc
            raise
        except httpx.ConnectError as exc:
            raise AppViewUnavailableError(self._appview_url, str(exc)) from exc

    # ------------------------------------------------------------------ #
    # Cross-account reads via generic AppView (Tier 1)
    # ------------------------------------------------------------------ #

    _GENERIC_APPVIEW_URL: str | None = None

    @classmethod
    def _get_generic_appview_url(cls) -> str:
        """Return the generic AppView URL for unauthenticated cross-account reads.

        Reads from the ``ATDATA_GENERIC_APPVIEW`` environment variable,
        falling back to ``https://bsky.social``.
        """
        if cls._GENERIC_APPVIEW_URL is None:
            import os

            cls._GENERIC_APPVIEW_URL = os.environ.get(
                "ATDATA_GENERIC_APPVIEW", "https://bsky.social"
            )
        return cls._GENERIC_APPVIEW_URL

    def _get_appview_client(self) -> Any:
        """Return a shared, unauthenticated client pointed at the public AppView.

        Used for cross-account reads where the authenticated PDS doesn't
        host the target repository.
        """
        if not hasattr(self, "_appview_client") or self._appview_client is None:
            Client = _get_atproto_client_class()
            self._appview_client = Client(
                base_url=self._get_generic_appview_url()
            )
        return self._appview_client

    # Low-level record operations

    def create_record(
        self,
        collection: str,
        record: dict,
        *,
        rkey: Optional[str] = None,
        validate: bool = False,
    ) -> AtUri:
        """Create a record in the user's repository.

        Args:
            collection: The NSID of the record collection
                (e.g., 'science.alt.dataset.schema').
            record: The record data. Must include a '$type' field.
            rkey: Optional explicit record key. If not provided, a TID is generated.
            validate: Whether to validate against the Lexicon schema. Set to False
                for custom lexicons that the PDS doesn't know about.

        Returns:
            The AT URI of the created record.

        Raises:
            ValueError: If not authenticated.
            atproto.exceptions.AtProtocolError: If record creation fails.
        """
        self._ensure_authenticated()

        response = self._client.com.atproto.repo.create_record(
            data={
                "repo": self.did,
                "collection": collection,
                "record": record,
                "rkey": rkey,
                "validate": validate,
            }
        )

        return AtUri.parse(response.uri)

    def put_record(
        self,
        collection: str,
        rkey: str,
        record: dict,
        *,
        validate: bool = False,
        swap_commit: Optional[str] = None,
    ) -> AtUri:
        """Create or update a record at a specific key.

        Args:
            collection: The NSID of the record collection.
            rkey: The record key.
            record: The record data. Must include a '$type' field.
            validate: Whether to validate against the Lexicon schema.
            swap_commit: Optional CID for compare-and-swap update.

        Returns:
            The AT URI of the record.

        Raises:
            ValueError: If not authenticated.
            atproto.exceptions.AtProtocolError: If operation fails.
        """
        self._ensure_authenticated()

        data: dict[str, Any] = {
            "repo": self.did,
            "collection": collection,
            "rkey": rkey,
            "record": record,
            "validate": validate,
        }
        if swap_commit:
            data["swapCommit"] = swap_commit

        response = self._client.com.atproto.repo.put_record(data=data)

        return AtUri.parse(response.uri)

    def get_record(
        self,
        uri: str | AtUri,
    ) -> dict:
        """Fetch a record by AT URI.

        When the target record belongs to a different user than the
        authenticated client, the request is routed through the public
        AppView (``bsky.social``) instead of the authenticated PDS.
        This avoids ``RecordNotFound`` errors caused by querying a PDS
        that doesn't host the target repository.

        Args:
            uri: The AT URI of the record.

        Returns:
            The record data as a dictionary.

        Raises:
            atproto.exceptions.AtProtocolError: If record not found.
        """
        if isinstance(uri, str):
            uri = AtUri.parse(uri)

        # If the target DID differs from our authenticated user, route
        # through the public AppView.  getRecord is an unauthenticated
        # query so this always works.
        is_foreign = self.is_authenticated and uri.authority != self.did
        if is_foreign:
            client = self._get_appview_client()
        else:
            client = self._client

        response = client.com.atproto.repo.get_record(
            params={
                "repo": uri.authority,
                "collection": uri.collection,
                "rkey": uri.rkey,
            }
        )

        return _value_to_dict(response.value)

    def delete_record(
        self,
        uri: str | AtUri,
        *,
        swap_commit: Optional[str] = None,
    ) -> None:
        """Delete a record.

        Args:
            uri: The AT URI of the record to delete.
            swap_commit: Optional CID for compare-and-swap delete.

        Raises:
            ValueError: If not authenticated.
            atproto.exceptions.AtProtocolError: If deletion fails.
        """
        self._ensure_authenticated()

        if isinstance(uri, str):
            uri = AtUri.parse(uri)

        data: dict[str, Any] = {
            "repo": self.did,
            "collection": uri.collection,
            "rkey": uri.rkey,
        }
        if swap_commit:
            data["swapCommit"] = swap_commit

        self._client.com.atproto.repo.delete_record(data=data)

    def upload_blob(
        self,
        data: bytes,
        mime_type: str = "application/octet-stream",
        *,
        timeout: float | None = None,
    ) -> dict:
        """Upload binary data as a blob to the PDS.

        Args:
            data: Binary data to upload.
            mime_type: MIME type of the data (for reference, not enforced by PDS).
            timeout: HTTP timeout in seconds. If ``None`` (default), uses a
                heuristic based on data size: 30 s base + 1 s per MB, with
                a minimum of 60 s. This overrides the httpx client default
                (5 s), which is too short for large blob uploads.

        Returns:
            A blob reference dict with keys: '$type', 'ref', 'mimeType', 'size'.
            This can be embedded directly in record fields.

        Raises:
            ValueError: If not authenticated.
            atproto.exceptions.AtProtocolError: If upload fails.
        """
        self._ensure_authenticated()

        if timeout is None:
            # 30 s base + 1 s per MB, minimum 60 s
            timeout = max(60.0, 30.0 + len(data) / 1_000_000)

        response = self._client.com.atproto.repo.upload_blob(data, timeout=timeout)
        blob_ref = response.blob

        # Convert to dict format suitable for embedding in records
        return {
            "$type": "blob",
            "ref": {
                "$link": blob_ref.ref.link
                if hasattr(blob_ref.ref, "link")
                else str(blob_ref.ref)
            },
            "mimeType": blob_ref.mime_type,
            "size": blob_ref.size,
        }

    def get_blob(
        self,
        did: str,
        cid: str,
    ) -> bytes:
        """Download a blob from a PDS.

        This resolves the PDS endpoint from the DID document and fetches
        the blob directly from the PDS.

        Args:
            did: The DID of the repository containing the blob.
            cid: The CID of the blob.

        Returns:
            The blob data as bytes.

        Raises:
            ValueError: If PDS endpoint cannot be resolved.
            requests.HTTPError: If blob fetch fails.
        """
        import requests

        from . import _resolve_pds_endpoint

        pds_endpoint = _resolve_pds_endpoint(did)
        url = f"{pds_endpoint}/xrpc/com.atproto.sync.getBlob"
        response = requests.get(url, params={"did": did, "cid": cid})
        response.raise_for_status()
        return response.content

    def get_blob_url(self, did: str, cid: str) -> str:
        """Get the direct URL for fetching a blob.

        This is useful for passing to WebDataset or other HTTP clients.

        Args:
            did: The DID of the repository containing the blob.
            cid: The CID of the blob.

        Returns:
            The full URL for fetching the blob.

        Raises:
            ValueError: If PDS endpoint cannot be resolved.
        """
        from . import _resolve_pds_endpoint

        pds_endpoint = _resolve_pds_endpoint(did)
        return f"{pds_endpoint}/xrpc/com.atproto.sync.getBlob?did={did}&cid={cid}"

    def resolve_did(self, handle_or_did: str) -> str:
        """Resolve a handle to a DID, or return a DID string unchanged.

        Args:
            handle_or_did: An AT Protocol handle (e.g. ``alice.bsky.social``)
                or a DID string (e.g. ``did:plc:abc123``).

        Returns:
            The DID string.
        """
        if handle_or_did.startswith("did:"):
            return handle_or_did

        response = self._client.com.atproto.identity.resolve_handle(
            params={"handle": handle_or_did}
        )
        return response.did

    def list_records(
        self,
        collection: str,
        *,
        repo: Optional[str] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> tuple[list[dict], Optional[str]]:
        """List records in a collection.

        Args:
            collection: The NSID of the record collection.
            repo: The DID of the repository to query. Defaults to the
                authenticated user's repository.
            limit: Maximum number of records to return (default 100).
            cursor: Pagination cursor from a previous call.

        Returns:
            A tuple of (records, next_cursor). The cursor is None if there
            are no more records.

        Raises:
            ValueError: If repo is None and not authenticated.
        """
        if repo is None:
            self._ensure_authenticated()
            repo = self.did

        # Route through AppView for foreign repos.
        is_foreign = self.is_authenticated and repo != self.did
        client = self._get_appview_client() if is_foreign else self._client

        response = client.com.atproto.repo.list_records(
            params={
                "repo": repo,
                "collection": collection,
                "limit": limit,
                "cursor": cursor,
            }
        )

        records = [_value_to_dict(r.value) for r in response.records]
        return records, response.cursor

    # Convenience methods for atdata collections
    #
    # These thin wrappers call list_records() which maps to
    # com.atproto.repo.listRecords — a generic PDS endpoint.
    # They return at most `limit` records with no automatic pagination.
    # When a dedicated AppView with collection-specific listing endpoints
    # is available, these should be updated to use those endpoints for
    # proper server-side pagination and filtering.

    def _list_collection(
        self,
        collection_suffix: str,
        repo: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """List records from a collection, warning on truncation."""
        collection = f"{LEXICON_NAMESPACE}.{collection_suffix}"
        records, cursor = self.list_records(
            collection,
            repo=repo,
            limit=limit,
        )
        if cursor:
            import warnings

            warnings.warn(
                f"list_{collection_suffix}s() returned {len(records)} records "
                f"but more exist (cursor={cursor!r}). Use list_records() with "
                f"pagination to retrieve all records.",
                stacklevel=3,
            )
        return records

    def list_schemas(
        self,
        repo: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """List schema records.

        Args:
            repo: The DID to query. Defaults to authenticated user.
            limit: Maximum number to return.  Warns if more records exist
                beyond the limit.

        Returns:
            List of schema records.
        """
        return self._list_collection("schema", repo=repo, limit=limit)

    def list_datasets(
        self,
        repo: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """List dataset records.

        Args:
            repo: The DID to query. Defaults to authenticated user.
            limit: Maximum number to return.  Warns if more records exist
                beyond the limit.

        Returns:
            List of dataset records.
        """
        return self._list_collection("entry", repo=repo, limit=limit)

    def list_lenses(
        self,
        repo: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """List lens records.

        Args:
            repo: The DID to query. Defaults to authenticated user.
            limit: Maximum number to return.  Warns if more records exist
                beyond the limit.

        Returns:
            List of lens records.
        """
        return self._list_collection("lens", repo=repo, limit=limit)

    def list_labels(
        self,
        repo: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """List label records.

        Args:
            repo: The DID to query. Defaults to authenticated user.
            limit: Maximum number to return.  Warns if more records exist
                beyond the limit.

        Returns:
            List of label records.
        """
        return self._list_collection("label", repo=repo, limit=limit)

    # ------------------------------------------------------------------ #
    # AppView-only capabilities
    # ------------------------------------------------------------------ #

    def search_datasets(
        self,
        q: str,
        *,
        tags: list[str] | None = None,
        schema_ref: str | None = None,
        repo: str | None = None,
        limit: int = 25,
    ) -> list[dict]:
        """Search datasets via the AppView full-text search.

        Requires an AppView to be configured.

        Args:
            q: Search query string.
            tags: Optional tag filter (array containment).
            schema_ref: Optional schema AT-URI filter.
            repo: Optional DID filter.
            limit: Maximum results (1-100, default 25).

        Returns:
            List of matching dataset entry dicts.

        Raises:
            AppViewRequiredError: If no AppView is configured.
        """
        params: dict[str, Any] = {"q": q, "limit": min(limit, 100)}
        if tags is not None:
            params["tags"] = tags
        if schema_ref is not None:
            params["schemaRef"] = schema_ref
        if repo is not None:
            params["repo"] = repo

        result = self.xrpc_query(f"{LEXICON_NAMESPACE}.searchDatasets", params=params)
        return result.get("entries", [])

    def search_lenses(
        self,
        *,
        source_schema: str | None = None,
        target_schema: str | None = None,
        limit: int = 25,
    ) -> list[dict]:
        """Search lenses by source and/or target schema via the AppView.

        Requires an AppView to be configured.

        Args:
            source_schema: Optional source schema AT-URI.
            target_schema: Optional target schema AT-URI.
            limit: Maximum results (1-100, default 25).

        Returns:
            List of matching lens record dicts.

        Raises:
            AppViewRequiredError: If no AppView is configured.
        """
        params: dict[str, Any] = {"limit": min(limit, 100)}
        if source_schema is not None:
            params["sourceSchema"] = source_schema
        if target_schema is not None:
            params["targetSchema"] = target_schema

        result = self.xrpc_query(f"{LEXICON_NAMESPACE}.searchLenses", params=params)
        return result.get("lenses", [])

    def describe_service(self) -> dict:
        """Get AppView service description including identity, collections, and analytics.

        Requires an AppView to be configured.

        Returns:
            Service description dict with keys: ``did``,
            ``availableCollections``, ``recordCount``, ``analytics``.

        Raises:
            AppViewRequiredError: If no AppView is configured.
        """
        return self.xrpc_query(f"{LEXICON_NAMESPACE}.describeService")

    def get_entries(self, uris: list[str]) -> list[dict]:
        """Batch-fetch multiple dataset entries via the AppView.

        Requires an AppView to be configured.

        Args:
            uris: List of AT-URIs (max 25 per call).

        Returns:
            List of dataset entry dicts.

        Raises:
            AppViewRequiredError: If no AppView is configured.
            ValueError: If more than 25 URIs are provided.
        """
        if len(uris) > 25:
            raise ValueError("getEntries supports at most 25 URIs per call")

        result = self.xrpc_query(
            f"{LEXICON_NAMESPACE}.getEntries", params={"uris": uris}
        )
        return result.get("entries", [])

    def get_entry_stats(
        self,
        uri: str,
        period: str = "week",
    ) -> dict:
        """Get view/search statistics for a dataset entry.

        Requires an AppView to be configured.

        Args:
            uri: AT-URI of the dataset entry.
            period: Time period — ``"day"``, ``"week"``, or ``"month"``.

        Returns:
            Stats dict with keys: ``views``, ``searchAppearances``,
            ``period``.

        Raises:
            AppViewRequiredError: If no AppView is configured.
        """
        return self.xrpc_query(
            f"{LEXICON_NAMESPACE}.getEntryStats",
            params={"uri": uri, "period": period},
        )

    # ------------------------------------------------------------------ #
    # Namespaced access — power-user operations grouped by concern
    # ------------------------------------------------------------------ #

    @property
    def records(self) -> RecordOps:
        """Namespaced record operations (create, get, put, delete, list)."""
        if not hasattr(self, "_records_ops"):
            self._records_ops = RecordOps(self)
        return self._records_ops

    @property
    def blobs(self) -> BlobOps:
        """Namespaced blob operations (upload, get, get_url)."""
        if not hasattr(self, "_blobs_ops"):
            self._blobs_ops = BlobOps(self)
        return self._blobs_ops

    @property
    def xrpc(self) -> XrpcClient:
        """Namespaced XRPC transport (query, procedure)."""
        if not hasattr(self, "_xrpc_client"):
            self._xrpc_client = XrpcClient(self)
        return self._xrpc_client


class RecordOps:
    """Namespaced ATProto record operations.

    Accessed via ``atmo.records``. Provides create, get, put, delete,
    and list operations on ATProto records.
    """

    __slots__ = ("_atmo",)

    def __init__(self, atmo: Atmosphere) -> None:
        self._atmo = atmo

    def create(
        self,
        collection: str,
        record: dict,
        *,
        rkey: Optional[str] = None,
        validate: bool = False,
    ) -> AtUri:
        """Create a record. See :meth:`Atmosphere.create_record`."""
        return self._atmo.create_record(
            collection, record, rkey=rkey, validate=validate
        )

    def put(
        self,
        collection: str,
        rkey: str,
        record: dict,
        *,
        validate: bool = False,
        swap_commit: Optional[str] = None,
    ) -> AtUri:
        """Create or update a record. See :meth:`Atmosphere.put_record`."""
        return self._atmo.put_record(
            collection, rkey, record, validate=validate, swap_commit=swap_commit
        )

    def get(self, uri: str | AtUri) -> dict:
        """Fetch a record by AT URI. See :meth:`Atmosphere.get_record`."""
        return self._atmo.get_record(uri)

    def delete(
        self,
        uri: str | AtUri,
        *,
        swap_commit: Optional[str] = None,
    ) -> None:
        """Delete a record. See :meth:`Atmosphere.delete_record`."""
        self._atmo.delete_record(uri, swap_commit=swap_commit)

    def list(
        self,
        collection: str,
        *,
        repo: Optional[str] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> tuple[list[dict], Optional[str]]:
        """List records in a collection. See :meth:`Atmosphere.list_records`."""
        return self._atmo.list_records(
            collection, repo=repo, limit=limit, cursor=cursor
        )


class BlobOps:
    """Namespaced ATProto blob operations.

    Accessed via ``atmo.blobs``. Provides upload, download, and URL
    generation for PDS blobs.
    """

    __slots__ = ("_atmo",)

    def __init__(self, atmo: Atmosphere) -> None:
        self._atmo = atmo

    def upload(
        self,
        data: bytes,
        mime_type: str = "application/octet-stream",
        *,
        timeout: float | None = None,
    ) -> dict:
        """Upload a blob. See :meth:`Atmosphere.upload_blob`."""
        return self._atmo.upload_blob(data, mime_type, timeout=timeout)

    def get(self, did: str, cid: str) -> bytes:
        """Download a blob. See :meth:`Atmosphere.get_blob`."""
        return self._atmo.get_blob(did, cid)

    def get_url(self, did: str, cid: str) -> str:
        """Get direct URL for a blob. See :meth:`Atmosphere.get_blob_url`."""
        return self._atmo.get_blob_url(did, cid)


class XrpcClient:
    """Namespaced XRPC transport operations.

    Accessed via ``atmo.xrpc``. Provides raw query (GET) and procedure
    (POST) calls to the AppView.
    """

    __slots__ = ("_atmo",)

    def __init__(self, atmo: Atmosphere) -> None:
        self._atmo = atmo

    def query(self, nsid: str, params: dict | None = None) -> dict:
        """Call an XRPC query (GET). See :meth:`Atmosphere.xrpc_query`."""
        return self._atmo.xrpc_query(nsid, params=params)

    def procedure(self, nsid: str, input: dict | None = None) -> dict:
        """Call an XRPC procedure (POST). See :meth:`Atmosphere.xrpc_procedure`."""
        return self._atmo.xrpc_procedure(nsid, input=input)
