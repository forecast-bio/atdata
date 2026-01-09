"""ATProto client wrapper for atdata.

This module provides the ``AtmosphereClient`` class which wraps the atproto SDK
client with atdata-specific helpers for publishing and querying records.
"""

from typing import Optional, Any

from ._types import AtUri, LEXICON_NAMESPACE

# Lazy import to avoid requiring atproto if not using atmosphere features
_atproto_client_class: Optional[type] = None


def _get_atproto_client_class():
    """Lazily import the atproto Client class."""
    global _atproto_client_class
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


class AtmosphereClient:
    """ATProto client wrapper for atdata operations.

    This class wraps the atproto SDK client and provides higher-level methods
    for working with atdata records (schemas, datasets, lenses).

    Example:
        >>> client = AtmosphereClient()
        >>> client.login("alice.bsky.social", "app-password")
        >>> print(client.did)
        'did:plc:...'

    Note:
        The password should be an app-specific password, not your main account
        password. Create app passwords in your Bluesky account settings.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        *,
        _client: Optional[Any] = None,
    ):
        """Initialize the ATProto client.

        Args:
            base_url: Optional PDS base URL. Defaults to bsky.social.
            _client: Optional pre-configured atproto Client for testing.
        """
        if _client is not None:
            self._client = _client
        else:
            Client = _get_atproto_client_class()
            self._client = Client(base_url=base_url) if base_url else Client()

        self._session: Optional[dict] = None

    def login(self, handle: str, password: str) -> None:
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

    def login_with_session(self, session_string: str) -> None:
        """Authenticate using an exported session string.

        This allows reusing a session without re-authenticating, which helps
        avoid rate limits on session creation.

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
                (e.g., 'ac.foundation.dataset.sampleSchema').
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

        Args:
            uri: The AT URI of the record.

        Returns:
            The record data as a dictionary.

        Raises:
            atproto.exceptions.AtProtocolError: If record not found.
        """
        if isinstance(uri, str):
            uri = AtUri.parse(uri)

        response = self._client.com.atproto.repo.get_record(
            params={
                "repo": uri.authority,
                "collection": uri.collection,
                "rkey": uri.rkey,
            }
        )

        return response.value

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

        response = self._client.com.atproto.repo.list_records(
            params={
                "repo": repo,
                "collection": collection,
                "limit": limit,
                "cursor": cursor,
            }
        )

        records = [r.value for r in response.records]
        return records, response.cursor

    # Convenience methods for atdata collections

    def list_schemas(
        self,
        repo: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """List schema records.

        Args:
            repo: The DID to query. Defaults to authenticated user.
            limit: Maximum number to return.

        Returns:
            List of schema records.
        """
        records, _ = self.list_records(
            f"{LEXICON_NAMESPACE}.sampleSchema",
            repo=repo,
            limit=limit,
        )
        return records

    def list_datasets(
        self,
        repo: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """List dataset records.

        Args:
            repo: The DID to query. Defaults to authenticated user.
            limit: Maximum number to return.

        Returns:
            List of dataset records.
        """
        records, _ = self.list_records(
            f"{LEXICON_NAMESPACE}.record",
            repo=repo,
            limit=limit,
        )
        return records

    def list_lenses(
        self,
        repo: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """List lens records.

        Args:
            repo: The DID to query. Defaults to authenticated user.
            limit: Maximum number to return.

        Returns:
            List of lens records.
        """
        records, _ = self.list_records(
            f"{LEXICON_NAMESPACE}.lens",
            repo=repo,
            limit=limit,
        )
        return records
