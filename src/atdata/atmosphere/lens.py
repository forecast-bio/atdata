"""Lens transformation publishing for ATProto.

This module provides classes for publishing Lens transformation records to
ATProto. Lenses are published as ``science.alt.dataset.lens`` records.

Note:
    For security reasons, lens code is stored as references to git repositories
    rather than inline code. Users must manually install and import lens
    implementations.
"""

from typing import Optional, TYPE_CHECKING

from .client import Atmosphere
from ._types import AtUri, LEXICON_NAMESPACE
from ._lexicon_types import LexLensRecord, LexCodeReference

if TYPE_CHECKING:
    from ..lens import Lens


class LensPublisher:
    """Publishes Lens transformation records to ATProto.

    This class creates lens records that reference source and target schemas
    and point to the transformation code in a git repository.

    Examples:
        >>> @atdata.lens
        ... def my_lens(source: SourceType) -> TargetType:
        ...     return TargetType(field=source.other_field)
        >>>
        >>> atmo = Atmosphere.login("handle", "password")
        >>>
        >>> publisher = LensPublisher(atmo)
        >>> uri = publisher.publish(
        ...     name="my_lens",
        ...     source_schema_uri="at://did:plc:abc/science.alt.dataset.schema/source",
        ...     target_schema_uri="at://did:plc:abc/science.alt.dataset.schema/target",
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

    def __init__(self, client: Atmosphere):
        """Initialize the lens publisher.

        Args:
            client: Authenticated Atmosphere instance.
        """
        self.client = client

    def publish(
        self,
        *,
        name: str,
        source_schema_uri: str,
        target_schema_uri: str,
        code_repository: str,
        code_commit: str,
        getter_path: str,
        putter_path: str,
        description: Optional[str] = None,
        language: Optional[str] = None,
        metadata: Optional[dict] = None,
        source_schema_version: Optional[str] = None,
        target_schema_version: Optional[str] = None,
        rkey: Optional[str] = None,
    ) -> AtUri:
        """Publish a lens transformation record to ATProto.

        When an AppView is configured, uses
        ``science.alt.dataset.publishLens`` for server-side validation
        (verifies both schema URIs exist). Falls back to direct
        ``createRecord`` otherwise.

        Args:
            name: Human-readable lens name.
            source_schema_uri: AT URI of the source schema.
            target_schema_uri: AT URI of the target schema.
            code_repository: Git repository URL containing the lens code.
            code_commit: Git commit hash for reproducibility.
            getter_path: Module path to the getter function
                (e.g., 'mymodule.lenses:my_getter').
            putter_path: Module path to the putter function
                (e.g., 'mymodule.lenses:my_putter').
            description: What this transformation does.
            language: Programming language (e.g., 'python').
            metadata: Arbitrary metadata dictionary.
            source_schema_version: Semver version or range for source
                schema compatibility (e.g., '1.0.0', '>=1.0.0 <2.0.0').
            target_schema_version: Semver version or range for target
                schema compatibility.
            rkey: Optional explicit record key.

        Returns:
            The AT URI of the created lens record.
        """
        getter_code = LexCodeReference(
            repository=code_repository,
            commit=code_commit,
            path=getter_path,
        )
        putter_code = LexCodeReference(
            repository=code_repository,
            commit=code_commit,
            path=putter_path,
        )

        lens_record = LexLensRecord(
            name=name,
            source_schema=source_schema_uri,
            target_schema=target_schema_uri,
            getter_code=getter_code,
            putter_code=putter_code,
            description=description,
            language=language,
            metadata=metadata,
            source_schema_version=source_schema_version,
            target_schema_version=target_schema_version,
        )

        if getattr(self.client, "has_appview", False) is True:
            return self._publish_via_appview(lens_record, rkey=rkey)

        return self.client.create_record(
            collection=f"{LEXICON_NAMESPACE}.lens",
            record=lens_record.to_record(),
            rkey=rkey,
            validate=False,
        )

    def _publish_via_appview(
        self,
        lens_record: LexLensRecord,
        *,
        rkey: Optional[str] = None,
    ) -> AtUri:
        """Publish via AppView procedure for server-side validation."""
        body: dict = {"record": lens_record.to_record()}
        if rkey is not None:
            body["rkey"] = rkey

        result = self.client.xrpc_procedure(
            f"{LEXICON_NAMESPACE}.publishLens",
            input=body,
        )
        return AtUri.parse(result["uri"])

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
        language: Optional[str] = None,
        metadata: Optional[dict] = None,
        source_schema_version: Optional[str] = None,
        target_schema_version: Optional[str] = None,
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
            language: Programming language (e.g., 'python').
            metadata: Arbitrary metadata dictionary.
            source_schema_version: Semver version or range for source
                schema compatibility.
            target_schema_version: Semver version or range for target
                schema compatibility.
            rkey: Optional explicit record key.

        Returns:
            The AT URI of the created lens record.
        """
        getter_name = lens_obj._getter.__name__
        putter_name = lens_obj._putter.__name__

        getter_module = getattr(lens_obj._getter, "__module__", "")
        putter_module = getattr(lens_obj._putter, "__module__", "")

        getter_path = f"{getter_module}:{getter_name}" if getter_module else getter_name
        putter_path = f"{putter_module}:{putter_name}" if putter_module else putter_name

        return self.publish(
            name=name,
            source_schema_uri=source_schema_uri,
            target_schema_uri=target_schema_uri,
            code_repository=code_repository,
            code_commit=code_commit,
            getter_path=getter_path,
            putter_path=putter_path,
            description=description,
            language=language,
            metadata=metadata,
            source_schema_version=source_schema_version,
            target_schema_version=target_schema_version,
            rkey=rkey,
        )


class LensLoader:
    """Loads lens records from ATProto.

    When an AppView is configured, queries are routed through it for
    paginated listing and efficient schema-based search. Otherwise falls
    back to client-side workarounds.

    Examples:
        >>> atmo = Atmosphere.login("handle", "password")
        >>> loader = LensLoader(atmo)
        >>>
        >>> record = loader.get("at://did:plc:abc/science.alt.dataset.lens/xyz")
        >>> print(record["name"])
        >>> print(record["sourceSchema"])
        >>> print(record.get("getterCode", {}).get("repository"))
    """

    def __init__(self, client: Atmosphere):
        """Initialize the lens loader.

        Args:
            client: Atmosphere instance.
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

    def get_typed(self, uri: str | AtUri) -> LexLensRecord:
        """Fetch a lens record and return as a typed object.

        Args:
            uri: The AT URI of the lens record.

        Returns:
            LexLensRecord instance.
        """
        record = self.get(uri)
        return LexLensRecord.from_record(record)

    def list_all(
        self,
        repo: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """List lens records from a repository.

        When an AppView is configured, uses
        ``science.alt.dataset.listLenses`` with cursor-based pagination.
        Falls back to ``com.atproto.repo.listRecords`` otherwise.

        Args:
            repo: The DID of the repository. Defaults to authenticated user.
            limit: Maximum number of records to return.

        Returns:
            List of lens records.
        """
        if getattr(self.client, "has_appview", False) is True:
            try:
                return self._list_via_appview(repo=repo, limit=limit)
            except Exception:
                from .._logging import get_logger

                get_logger().warning(
                    "AppView listLenses failed, falling back to client-side",
                    exc_info=True,
                )

        return self.client.list_lenses(repo=repo, limit=limit)

    def _list_via_appview(
        self,
        repo: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """List lenses via AppView with cursor pagination."""
        results: list[dict] = []
        cursor: Optional[str] = None
        page_size = min(limit, 100)

        while len(results) < limit:
            params: dict[str, str | int] = {"limit": page_size}
            if repo is not None:
                params["repo"] = repo
            if cursor is not None:
                params["cursor"] = cursor

            response = self.client.xrpc_query(
                f"{LEXICON_NAMESPACE}.listLenses",
                params=params,
            )

            lenses = response.get("lenses", [])
            results.extend(lenses)
            cursor = response.get("cursor")
            if not cursor or not lenses:
                break

        return results[:limit]

    def find_by_schemas(
        self,
        source_schema_uri: str,
        target_schema_uri: Optional[str] = None,
        repo: Optional[str] = None,
    ) -> list[dict]:
        """Find lenses that transform between specific schemas.

        When an AppView is configured, uses
        ``science.alt.dataset.searchLenses`` for efficient server-side
        filtering. Falls back to client-side pagination and filtering
        otherwise.

        Args:
            source_schema_uri: AT URI of the source schema.
            target_schema_uri: Optional AT URI of the target schema.
                If not provided, returns all lenses from the source.
            repo: The DID of the repository to search.

        Returns:
            List of matching lens records.
        """
        if getattr(self.client, "has_appview", False) is True:
            try:
                return self._find_by_schemas_via_appview(
                    source_schema_uri, target_schema_uri
                )
            except Exception:
                from .._logging import get_logger

                get_logger().warning(
                    "AppView searchLenses failed, falling back to client-side",
                    exc_info=True,
                )

        return self._find_by_schemas_client_side(
            source_schema_uri, target_schema_uri, repo
        )

    def _find_by_schemas_via_appview(
        self,
        source_schema_uri: str,
        target_schema_uri: Optional[str] = None,
    ) -> list[dict]:
        """Search lenses via AppView XRPC query."""
        params: dict[str, str | int] = {"sourceSchema": source_schema_uri, "limit": 100}
        if target_schema_uri is not None:
            params["targetSchema"] = target_schema_uri

        results: list[dict] = []
        cursor: Optional[str] = None

        while True:
            if cursor is not None:
                params["cursor"] = cursor

            response = self.client.xrpc_query(
                f"{LEXICON_NAMESPACE}.searchLenses",
                params=params,
            )

            lenses = response.get("lenses", [])
            results.extend(lenses)
            cursor = response.get("cursor")
            if not cursor or not lenses:
                break

        return results

    def _find_by_schemas_client_side(
        self,
        source_schema_uri: str,
        target_schema_uri: Optional[str] = None,
        repo: Optional[str] = None,
    ) -> list[dict]:
        """Find lenses by paginating through all records and filtering."""
        collection = f"{LEXICON_NAMESPACE}.lens"
        if repo is None:
            self.client._ensure_authenticated()
            repo = self.client.did

        matches = []
        cursor: Optional[str] = None
        while True:
            records, cursor = self.client.list_records(
                collection,
                repo=repo,
                limit=100,
                cursor=cursor,
            )
            for lens_record in records:
                if lens_record.get("sourceSchema") == source_schema_uri:
                    if target_schema_uri is None:
                        matches.append(lens_record)
                    elif lens_record.get("targetSchema") == target_schema_uri:
                        matches.append(lens_record)
            if not cursor:
                break

        return matches
