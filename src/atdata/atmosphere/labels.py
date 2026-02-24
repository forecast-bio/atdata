"""Dataset label publishing and loading for ATProto.

This module provides classes for publishing named label records to ATProto
and resolving them back to dataset records. Labels are published as
``science.alt.dataset.label`` records.

Labels separate dataset identity (CID-addressed records) from naming
(mutable label records), enabling versioning, renaming, and flexible routing.
"""

from __future__ import annotations

from .client import Atmosphere
from ._types import AtUri, LEXICON_NAMESPACE
from ._lexicon_types import LexLabelRecord


class LabelPublisher:
    """Publishes dataset label records to ATProto.

    Labels are named pointers to dataset records. Multiple labels with the
    same name but different versions can coexist, each pointing to a
    different dataset record.

    Examples:
        >>> atmo = Atmosphere.login("handle", "password")
        >>> publisher = LabelPublisher(atmo)
        >>> uri = publisher.publish(
        ...     name="mnist",
        ...     dataset_uri="at://did:plc:abc/science.alt.dataset.entry/xyz",
        ...     version="1.0.0",
        ...     description="Initial release",
        ... )
    """

    def __init__(self, client: Atmosphere):
        """Initialize the label publisher.

        Args:
            client: Authenticated Atmosphere instance.
        """
        self.client = client

    def publish(
        self,
        name: str,
        dataset_uri: str,
        *,
        version: str | None = None,
        description: str | None = None,
        rkey: str | None = None,
    ) -> AtUri:
        """Publish a label record to ATProto.

        When an AppView is configured, uses
        ``science.alt.dataset.publishLabel`` for server-side validation
        (verifies that ``dataset_uri`` exists). Falls back to direct
        ``createRecord`` otherwise.

        Args:
            name: User-facing label name (e.g. 'mnist').
            dataset_uri: AT URI of the dataset record to label.
            version: Optional version string (e.g. '1.0.0').
            description: Optional description of this labeled version.
            rkey: Optional explicit record key.

        Returns:
            The AT URI of the created label record.
        """
        label_record = LexLabelRecord(
            name=name,
            dataset_uri=dataset_uri,
            version=version,
            description=description,
        )

        if getattr(self.client, "has_appview", False) is True:
            return self._publish_via_appview(label_record, rkey=rkey)

        return self.client.create_record(
            collection=f"{LEXICON_NAMESPACE}.label",
            record=label_record.to_record(),
            rkey=rkey,
            validate=False,
        )

    def _publish_via_appview(
        self,
        label_record: LexLabelRecord,
        *,
        rkey: str | None = None,
    ) -> AtUri:
        """Publish via AppView procedure for server-side validation."""
        body: dict = {"record": label_record.to_record()}
        if rkey is not None:
            body["rkey"] = rkey

        result = self.client.xrpc_procedure(
            f"{LEXICON_NAMESPACE}.publishLabel",
            input=body,
        )
        return AtUri.parse(result["uri"])


class LabelLoader:
    """Loads and resolves dataset label records from ATProto.

    When an AppView is configured, queries are routed through it for
    efficient server-side resolution. Otherwise falls back to client-side
    workarounds using ``com.atproto.repo.listRecords``.

    Examples:
        >>> atmo = Atmosphere.login("handle", "password")
        >>> loader = LabelLoader(atmo)
        >>> dataset_uri = loader.resolve("did:plc:abc", "mnist")
        >>> dataset_uri = loader.resolve("did:plc:abc", "mnist", version="1.0.0")
    """

    def __init__(self, client: Atmosphere):
        """Initialize the label loader.

        Args:
            client: Atmosphere instance.
        """
        self.client = client

    def get(self, uri: str | AtUri) -> dict:
        """Fetch a label record by AT URI.

        Args:
            uri: The AT URI of the label record.

        Returns:
            The label record as a dictionary.

        Raises:
            ValueError: If the record is not a label record.
        """
        record = self.client.get_record(uri)

        expected_type = f"{LEXICON_NAMESPACE}.label"
        if record.get("$type") != expected_type:
            raise ValueError(
                f"Record at {uri} is not a label record. "
                f"Expected $type='{expected_type}', got '{record.get('$type')}'"
            )

        return record

    def get_typed(self, uri: str | AtUri) -> LexLabelRecord:
        """Fetch a label record and return as a typed object.

        Args:
            uri: The AT URI of the label record.

        Returns:
            LexLabelRecord instance.
        """
        record = self.get(uri)
        return LexLabelRecord.from_record(record)

    def list_all(
        self,
        repo: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """List label records from a repository.

        This delegates to ``com.atproto.repo.listRecords`` which returns at
        most ``limit`` records with no automatic pagination.  Repositories
        with more labels than ``limit`` will return a truncated result.

        Args:
            repo: The DID of the repository. Defaults to authenticated user.
            limit: Maximum number of records to return.

        Returns:
            List of label records.
        """
        return self.client.list_labels(repo=repo, limit=limit)

    def resolve(
        self,
        handle_or_did: str,
        name: str,
        version: str | None = None,
    ) -> str:
        """Resolve a named label to its dataset AT URI.

        When an AppView is configured, uses
        ``science.alt.dataset.resolveLabel`` for efficient server-side
        resolution. Falls back to client-side filtering otherwise.

        Args:
            handle_or_did: DID or handle of the dataset owner.
            name: Label name (e.g. 'mnist').
            version: Specific version to resolve. If ``None``, resolves
                to the most recently created label.

        Returns:
            AT URI of the referenced dataset record.

        Raises:
            KeyError: If no matching label is found.
        """
        if getattr(self.client, "has_appview", False) is True:
            try:
                return self._resolve_via_appview(handle_or_did, name, version)
            except Exception:
                from .._logging import get_logger

                get_logger().warning(
                    "AppView label resolution failed, falling back to client-side",
                    exc_info=True,
                )

        return self._resolve_client_side(handle_or_did, name, version)

    def _resolve_via_appview(
        self,
        handle_or_did: str,
        name: str,
        version: str | None = None,
    ) -> str:
        """Resolve a label via AppView XRPC query."""
        params: dict[str, str] = {"handle": handle_or_did, "name": name}
        if version is not None:
            params["version"] = version

        result = self.client.xrpc_query(
            f"{LEXICON_NAMESPACE}.resolveLabel",
            params=params,
        )
        return result["uri"]

    def _resolve_client_side(
        self,
        handle_or_did: str,
        name: str,
        version: str | None = None,
    ) -> str:
        """Resolve a label by fetching all records and filtering in Python.

        This is the client-side workaround used when no AppView is available.
        Known limitations:
        - Hard-coded limit of 100 records
        - No pagination
        - O(n) per resolve
        """
        did = self._resolve_did(handle_or_did)

        records, _ = self.client.list_records(
            f"{LEXICON_NAMESPACE}.label",
            repo=did,
            limit=100,
        )

        matches: list[dict] = []
        for record in records:
            if record.get("name") != name:
                continue
            if version is not None and record.get("version") != version:
                continue
            matches.append(record)

        if not matches:
            raise KeyError(
                f"No label {name!r}"
                + (f" version {version!r}" if version else "")
                + f" found for {handle_or_did!r}"
            )

        best = max(matches, key=lambda r: r.get("createdAt", ""))
        return best["datasetUri"]

    def _resolve_did(self, handle_or_did: str) -> str:
        """Resolve a handle to a DID, or return the DID directly."""
        return self.client.resolve_did(handle_or_did)
