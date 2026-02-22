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

        return self.client.create_record(
            collection=f"{LEXICON_NAMESPACE}.label",
            record=label_record.to_record(),
            rkey=rkey,
            validate=False,
        )


class LabelLoader:
    """Loads and resolves dataset label records from ATProto.

    Note:
        The ``resolve()`` method is a **client-side workaround** for the
        ``science.alt.dataset.resolveLabel`` query lexicon, which is
        defined but has no AppView to serve it yet. See ``resolve()`` for
        details and known limitations.

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

        Finds label records matching the given name (and optionally version)
        in the specified repository. When no version is given, returns the
        most recently created matching label.

        .. note:: **Client-side workaround (no AppView)**

           This method emulates the ``science.alt.dataset.resolveLabel``
           query lexicon by fetching all label records via
           ``com.atproto.repo.listRecords`` and filtering in Python.
           When an AppView is available, replace this with a direct
           ``GET /xrpc/science.alt.dataset.resolveLabel`` call.

           Known limitations of the client-side approach:

           - Hard-coded limit of 100 records (repos with >100 labels
             silently lose results)
           - No pagination (would need a cursor loop for correctness)
           - O(n) per resolve (fetches all labels every time)
           - No server-side filtering (defeats the purpose of the query
             lexicon)

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
        did = self._resolve_did(handle_or_did)

        # WORKAROUND: Client-side query (no AppView)
        # Lexicon: science.alt.dataset.resolveLabel
        # This fetches all records via list_records() and filters in Python.
        # Replace with GET /xrpc/science.alt.dataset.resolveLabel when
        # AppView is available.
        # Known limitations:
        #   - Hard-coded limit of 100 records (repos with >100 silently lose results)
        #   - No pagination (would need cursor loop)
        #   - O(n) per resolve (fetches all records every time)
        records, _ = self.client.list_records(
            f"{LEXICON_NAMESPACE}.label",
            repo=did,
            limit=100,
        )

        # Filter by name and optionally version
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

        # Pick latest by createdAt
        best = max(matches, key=lambda r: r.get("createdAt", ""))
        return best["datasetUri"]

    def _resolve_did(self, handle_or_did: str) -> str:
        """Resolve a handle to a DID, or return the DID directly."""
        return self.client.resolve_did(handle_or_did)
