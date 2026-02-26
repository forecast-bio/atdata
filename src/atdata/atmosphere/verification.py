"""Lens verification publishing and loading for ATProto.

This module provides classes for publishing and retrieving lens verification
records on ATProto. Verification records are published as
``science.alt.dataset.lensVerification`` records.

Verification records attest that a lens transformation has been reviewed,
tested, or formally verified. The verifier's identity is implicit -- the DID
of the repo owner who writes the record.
"""

from __future__ import annotations

from .client import Atmosphere
from ._types import AtUri, LEXICON_NAMESPACE
from ._lexicon_types import (
    LexLensVerification,
    LexCodeHash,
    LexCodeReference,
)

_COLLECTION = f"{LEXICON_NAMESPACE}.lensVerification"


class VerificationPublisher:
    """Publishes lens verification records to ATProto.

    Examples:
        >>> atmo = Atmosphere.login("handle", "password")
        >>> publisher = VerificationPublisher(atmo)
        >>> uri = publisher.publish(
        ...     lens_uri="at://did:plc:abc/science.alt.dataset.lens/xyz",
        ...     lens_commit="bafy...",
        ...     verification_method="codeReview",
        ...     description="Reviewed getter/putter for correctness",
        ... )
    """

    def __init__(self, client: Atmosphere) -> None:
        """Initialize the verification publisher.

        Args:
            client: Authenticated Atmosphere instance.
        """
        self.client = client

    def publish(
        self,
        *,
        lens_uri: str,
        lens_commit: str,
        verification_method: str,
        code_hash: LexCodeHash | None = None,
        proof_ref: LexCodeReference | None = None,
        description: str | None = None,
        rkey: str | None = None,
    ) -> AtUri:
        """Publish a lens verification record to ATProto.

        When an AppView is configured, uses
        ``science.alt.dataset.publishLensVerification`` for server-side
        validation. Falls back to direct ``createRecord`` otherwise.

        Args:
            lens_uri: AT-URI of the lens record being verified.
            lens_commit: CID of the specific lens record version.
            verification_method: Verification method identifier
                (e.g., 'codeReview', 'formalProof', 'signedHash',
                'automatedTest').
            code_hash: Hash of the code at the referenced commit.
                Required for ``signedHash`` method.
            proof_ref: Link to proof artifact (Coq/Lean proof, test
                suite, etc.). Used with ``formalProof`` or
                ``automatedTest`` methods.
            description: Human-readable description of what was verified.
            rkey: Optional explicit record key.

        Returns:
            The AT URI of the created verification record.
        """
        record = LexLensVerification(
            lens=lens_uri,
            lens_commit=lens_commit,
            verification_method=verification_method,
            code_hash=code_hash,
            proof_ref=proof_ref,
            description=description,
        )

        if getattr(self.client, "has_appview", False) is True:
            return self._publish_via_appview(record, rkey=rkey)

        return self.client.create_record(
            collection=_COLLECTION,
            record=record.to_record(),
            rkey=rkey,
            validate=False,
        )

    def _publish_via_appview(
        self,
        record: LexLensVerification,
        *,
        rkey: str | None = None,
    ) -> AtUri:
        """Publish via AppView procedure for server-side validation."""
        body: dict = {"record": record.to_record()}
        if rkey is not None:
            body["rkey"] = rkey

        result = self.client.xrpc_procedure(
            f"{LEXICON_NAMESPACE}.publishLensVerification",
            input=body,
        )
        return AtUri.parse(result["uri"])


class VerificationLoader:
    """Loads lens verification records from ATProto.

    Examples:
        >>> atmo = Atmosphere.login("handle", "password")
        >>> loader = VerificationLoader(atmo)
        >>> record = loader.get("at://did:plc:abc/science.alt.dataset.lensVerification/xyz")
        >>> print(record["verificationMethod"])
    """

    def __init__(self, client: Atmosphere) -> None:
        """Initialize the verification loader.

        Args:
            client: Atmosphere instance.
        """
        self.client = client

    def get(self, uri: str | AtUri) -> dict:
        """Fetch a verification record by AT URI.

        Args:
            uri: The AT URI of the verification record.

        Returns:
            The verification record as a dictionary.

        Raises:
            ValueError: If the record is not a verification record.
        """
        record = self.client.get_record(uri)

        if record.get("$type") != _COLLECTION:
            raise ValueError(
                f"Record at {uri} is not a lensVerification record. "
                f"Expected $type='{_COLLECTION}', got '{record.get('$type')}'"
            )

        return record

    def get_typed(self, uri: str | AtUri) -> LexLensVerification:
        """Fetch a verification record and return as a typed object.

        Args:
            uri: The AT URI of the verification record.

        Returns:
            LexLensVerification instance.
        """
        record = self.get(uri)
        return LexLensVerification.from_record(record)

    def list_for_lens(
        self,
        lens_uri: str,
        repo: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """List verification records for a specific lens.

        Paginates through all ``lensVerification`` records in the repo
        and filters by the target lens URI.

        Args:
            lens_uri: AT-URI of the lens to find verifications for.
            repo: DID of the repository. Defaults to authenticated user.
            limit: Maximum number of records to return.

        Returns:
            List of matching verification records.
        """
        if repo is None:
            self.client._ensure_authenticated()
            repo = self.client.did

        matches: list[dict] = []
        cursor: str | None = None
        while len(matches) < limit:
            records, cursor = self.client.list_records(
                _COLLECTION,
                repo=repo,
                limit=100,
                cursor=cursor,
            )
            for rec in records:
                if rec.get("lens") == lens_uri:
                    matches.append(rec)
                    if len(matches) >= limit:
                        break
            if not cursor:
                break

        return matches

    def find_by_method(
        self,
        method: str,
        repo: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Find verification records by verification method.

        Args:
            method: Verification method identifier (e.g., 'codeReview').
            repo: DID of the repository. Defaults to authenticated user.
            limit: Maximum number of records to return.

        Returns:
            List of matching verification records.
        """
        if repo is None:
            self.client._ensure_authenticated()
            repo = self.client.did

        matches: list[dict] = []
        cursor: str | None = None
        while len(matches) < limit:
            records, cursor = self.client.list_records(
                _COLLECTION,
                repo=repo,
                limit=100,
                cursor=cursor,
            )
            for rec in records:
                if rec.get("verificationMethod") == method:
                    matches.append(rec)
                    if len(matches) >= limit:
                        break
            if not cursor:
                break

        return matches
