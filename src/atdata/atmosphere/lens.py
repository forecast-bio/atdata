"""Lens transformation publishing for ATProto.

This module provides classes for publishing Lens transformation records to
ATProto. Lenses are published as ``ac.foundation.dataset.lens`` records.

Note:
    For security reasons, lens code is stored as references to git repositories
    rather than inline code. Users must manually install and import lens
    implementations.
"""

from typing import Optional, Callable

from .client import AtmosphereClient
from ._types import (
    AtUri,
    LensRecord,
    CodeReference,
    LEXICON_NAMESPACE,
)

# Import for type checking only
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..lens import Lens


class LensPublisher:
    """Publishes Lens transformation records to ATProto.

    This class creates lens records that reference source and target schemas
    and point to the transformation code in a git repository.

    Example:
        >>> @atdata.lens
        ... def my_lens(source: SourceType) -> TargetType:
        ...     return TargetType(field=source.other_field)
        >>>
        >>> client = AtmosphereClient()
        >>> client.login("handle", "password")
        >>>
        >>> publisher = LensPublisher(client)
        >>> uri = publisher.publish(
        ...     name="my_lens",
        ...     source_schema_uri="at://did:plc:abc/ac.foundation.dataset.sampleSchema/source",
        ...     target_schema_uri="at://did:plc:abc/ac.foundation.dataset.sampleSchema/target",
        ...     code_repository="https://github.com/user/repo",
        ...     code_commit="abc123def456",
        ...     getter_path="mymodule.lenses:my_lens",
        ...     putter_path="mymodule.lenses:my_lens_putter",
        ... )

    Security Note:
        Lens code is stored as references to git repositories rather than
        inline code. This prevents arbitrary code execution from ATProto
        records. Users must manually install and trust lens implementations.
    """

    def __init__(self, client: AtmosphereClient):
        """Initialize the lens publisher.

        Args:
            client: Authenticated AtmosphereClient instance.
        """
        self.client = client

    def publish(
        self,
        *,
        name: str,
        source_schema_uri: str,
        target_schema_uri: str,
        description: Optional[str] = None,
        code_repository: Optional[str] = None,
        code_commit: Optional[str] = None,
        getter_path: Optional[str] = None,
        putter_path: Optional[str] = None,
        rkey: Optional[str] = None,
    ) -> AtUri:
        """Publish a lens transformation record to ATProto.

        Args:
            name: Human-readable lens name.
            source_schema_uri: AT URI of the source schema.
            target_schema_uri: AT URI of the target schema.
            description: What this transformation does.
            code_repository: Git repository URL containing the lens code.
            code_commit: Git commit hash for reproducibility.
            getter_path: Module path to the getter function
                (e.g., 'mymodule.lenses:my_getter').
            putter_path: Module path to the putter function
                (e.g., 'mymodule.lenses:my_putter').
            rkey: Optional explicit record key.

        Returns:
            The AT URI of the created lens record.

        Raises:
            ValueError: If code references are incomplete.
        """
        # Build code references if provided
        getter_code: Optional[CodeReference] = None
        putter_code: Optional[CodeReference] = None

        if code_repository and code_commit:
            if getter_path:
                getter_code = CodeReference(
                    repository=code_repository,
                    commit=code_commit,
                    path=getter_path,
                )
            if putter_path:
                putter_code = CodeReference(
                    repository=code_repository,
                    commit=code_commit,
                    path=putter_path,
                )

        lens_record = LensRecord(
            name=name,
            source_schema=source_schema_uri,
            target_schema=target_schema_uri,
            description=description,
            getter_code=getter_code,
            putter_code=putter_code,
        )

        return self.client.create_record(
            collection=f"{LEXICON_NAMESPACE}.lens",
            record=lens_record.to_record(),
            rkey=rkey,
            validate=False,
        )

    def publish_from_lens(
        self,
        lens_obj: "Lens",
        *,
        name: str,
        source_schema_uri: str,
        target_schema_uri: str,
        code_repository: str,
        code_commit: str,
        description: Optional[str] = None,
        rkey: Optional[str] = None,
    ) -> AtUri:
        """Publish a lens record from an existing Lens object.

        This method extracts the getter and putter function names from
        the Lens object and publishes a record referencing them.

        Args:
            lens_obj: The Lens object to publish.
            name: Human-readable lens name.
            source_schema_uri: AT URI of the source schema.
            target_schema_uri: AT URI of the target schema.
            code_repository: Git repository URL.
            code_commit: Git commit hash.
            description: What this transformation does.
            rkey: Optional explicit record key.

        Returns:
            The AT URI of the created lens record.
        """
        # Extract function names from the lens
        getter_name = lens_obj._getter.__name__
        putter_name = lens_obj._putter.__name__

        # Get module info if available
        getter_module = getattr(lens_obj._getter, "__module__", "")
        putter_module = getattr(lens_obj._putter, "__module__", "")

        getter_path = f"{getter_module}:{getter_name}" if getter_module else getter_name
        putter_path = f"{putter_module}:{putter_name}" if putter_module else putter_name

        return self.publish(
            name=name,
            source_schema_uri=source_schema_uri,
            target_schema_uri=target_schema_uri,
            description=description,
            code_repository=code_repository,
            code_commit=code_commit,
            getter_path=getter_path,
            putter_path=putter_path,
            rkey=rkey,
        )


class LensLoader:
    """Loads lens records from ATProto.

    This class fetches lens transformation records. Note that actually
    using a lens requires installing the referenced code and importing
    it manually.

    Example:
        >>> client = AtmosphereClient()
        >>> loader = LensLoader(client)
        >>>
        >>> record = loader.get("at://did:plc:abc/ac.foundation.dataset.lens/xyz")
        >>> print(record["name"])
        >>> print(record["sourceSchema"])
        >>> print(record.get("getterCode", {}).get("repository"))
    """

    def __init__(self, client: AtmosphereClient):
        """Initialize the lens loader.

        Args:
            client: AtmosphereClient instance.
        """
        self.client = client

    def get(self, uri: str | AtUri) -> dict:
        """Fetch a lens record by AT URI.

        Args:
            uri: The AT URI of the lens record.

        Returns:
            The lens record as a dictionary.

        Raises:
            ValueError: If the record is not a lens record.
        """
        record = self.client.get_record(uri)

        expected_type = f"{LEXICON_NAMESPACE}.lens"
        if record.get("$type") != expected_type:
            raise ValueError(
                f"Record at {uri} is not a lens record. "
                f"Expected $type='{expected_type}', got '{record.get('$type')}'"
            )

        return record

    def list_all(
        self,
        repo: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """List lens records from a repository.

        Args:
            repo: The DID of the repository. Defaults to authenticated user.
            limit: Maximum number of records to return.

        Returns:
            List of lens records.
        """
        return self.client.list_lenses(repo=repo, limit=limit)

    def find_by_schemas(
        self,
        source_schema_uri: str,
        target_schema_uri: Optional[str] = None,
        repo: Optional[str] = None,
    ) -> list[dict]:
        """Find lenses that transform between specific schemas.

        Args:
            source_schema_uri: AT URI of the source schema.
            target_schema_uri: Optional AT URI of the target schema.
                If not provided, returns all lenses from the source.
            repo: The DID of the repository to search.

        Returns:
            List of matching lens records.
        """
        all_lenses = self.list_all(repo=repo, limit=1000)

        matches = []
        for lens_record in all_lenses:
            if lens_record.get("sourceSchema") == source_schema_uri:
                if target_schema_uri is None:
                    matches.append(lens_record)
                elif lens_record.get("targetSchema") == target_schema_uri:
                    matches.append(lens_record)

        return matches
