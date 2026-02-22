"""ATProto integration for distributed dataset federation.

This module provides ATProto publishing and discovery capabilities for atdata,
enabling a loose federation of distributed, typed datasets on the AT Protocol
network.

Key components:

- ``Atmosphere``: Authentication and session management for ATProto
- ``SchemaPublisher``: Publish PackableSample schemas as ATProto records
- ``DatasetPublisher``: Publish dataset index records with WebDataset URLs
- ``LensPublisher``: Publish lens transformation records

The ATProto integration is additive - existing atdata functionality continues
to work unchanged. These features are opt-in for users who want to publish
or discover datasets on the ATProto network.

Examples:
    >>> from atdata.atmosphere import Atmosphere
    >>>
    >>> atmo = Atmosphere.login("handle.bsky.social", "app-password")
    >>> index = Index(atmosphere=atmo)

Note:
    This module requires the ``atproto`` package to be installed::

        pip install atproto
"""

from typing import Iterator, Optional, Type, TYPE_CHECKING

from .client import Atmosphere
from .schema import SchemaPublisher, SchemaLoader
from .records import DatasetPublisher, DatasetLoader
from .lens import LensPublisher, LensLoader
from .labels import LabelPublisher, LabelLoader
from .store import PDSBlobStore
from ._types import AtUri, LEXICON_NAMESPACE
from ._lexicon_types import (
    DatasetMetadata,
    LexSchemaRecord,
    LexDatasetEntry,
    LexLensRecord,
    LexLabelRecord,
    LexCodeReference,
    JsonSchemaFormat,
    StorageHttp,
    StorageS3,
    StorageBlobs,
    ShardChecksum,
    HttpShardEntry,
    S3ShardEntry,
    BlobEntry,
    DatasetSize,
    ShardManifestRef,
    StorageUnion,
    storage_from_record,
)

if TYPE_CHECKING:
    from ..dataset import Dataset
    from .._protocols import Packable


_pds_endpoint_cache: dict[str, str] = {}


def _resolve_pds_endpoint(did: str) -> str:
    """Resolve the PDS endpoint for a DID via plc.directory.

    Results are cached in a module-level dict to avoid repeated lookups.

    Args:
        did: The DID to resolve (e.g. ``did:plc:abc123``).

    Returns:
        PDS service endpoint URL.

    Raises:
        ValueError: If PDS endpoint cannot be resolved.
    """
    if did in _pds_endpoint_cache:
        return _pds_endpoint_cache[did]

    import requests

    try:
        if did.startswith("did:plc:"):
            response = requests.get(f"https://plc.directory/{did}", timeout=10)
            response.raise_for_status()
            doc = response.json()

            for service in doc.get("service", []):
                if service.get("type") == "AtprotoPersonalDataServer":
                    endpoint = service.get("serviceEndpoint", "")
                    _pds_endpoint_cache[did] = endpoint
                    return endpoint
    except requests.RequestException as exc:
        raise ValueError(f"Could not resolve PDS endpoint for {did}") from exc

    raise ValueError(f"Could not resolve PDS endpoint for {did}")


class AtmosphereIndexEntry:
    """Entry wrapper for ATProto dataset records implementing IndexEntry protocol.

    Attributes:
        _uri: AT URI of the record.
        _record: Raw record dictionary.
    """

    def __init__(self, uri: str, record: dict):
        self._uri = uri
        self._record = record

    @property
    def name(self) -> str:
        """Human-readable dataset name."""
        return self._record.get("name", "")

    @property
    def schema_ref(self) -> str:
        """AT URI of the schema record."""
        return self._record.get("schemaRef", "")

    @property
    def data_urls(self) -> list[str]:
        """WebDataset URLs from storage.

        Handles storageHttp (shard URLs), storageS3 (s3:// URLs),
        storageExternal (legacy), and storageBlobs (PDS blob URLs).
        """
        storage = self._record.get("storage", {})
        storage_type = storage.get("$type", "")
        if "storageHttp" in storage_type:
            return [s["url"] for s in storage.get("shards", [])]
        if "storageS3" in storage_type:
            bucket = storage.get("bucket", "")
            return [f"s3://{bucket}/{s['key']}" for s in storage.get("shards", [])]
        if "storageExternal" in storage_type:
            return storage.get("urls", [])
        if "storageBlobs" in storage_type:
            parsed = AtUri.parse(self._uri)
            did = parsed.authority
            pds_endpoint = _resolve_pds_endpoint(did)
            urls = []
            for entry in storage.get("blobs", []):
                blob = entry.get("blob", entry)
                ref = blob.get("ref", {})
                cid = ref.get("$link") if isinstance(ref, dict) else str(ref)
                if cid:
                    urls.append(
                        f"{pds_endpoint}/xrpc/com.atproto.sync.getBlob"
                        f"?did={did}&cid={cid}"
                    )
            return urls
        return []

    @property
    def metadata(self) -> Optional[dict]:
        """Metadata from the record, if any.

        Returns a flat dict for backward compatibility. Handles both the new
        structured JSON metadata and legacy msgpack-encoded bytes.
        """
        from ._lexicon_types import decode_metadata_raw

        return decode_metadata_raw(self._record.get("metadata"))

    @property
    def uri(self) -> str:
        """AT URI of this record."""
        return self._uri


class AtmosphereIndex:
    """ATProto index implementing AbstractIndex protocol.

    .. deprecated::
        Use ``atdata.Index(atmosphere=client)`` instead.  ``AtmosphereIndex``
        is retained for backwards compatibility and will be removed in a
        future release.

    Wraps SchemaPublisher/Loader and DatasetPublisher/Loader to provide
    a unified interface compatible with Index.

    Optionally accepts a ``PDSBlobStore`` for writing dataset shards as
    ATProto blobs, enabling fully decentralized dataset storage.

    Examples:
        >>> # Preferred: use unified Index
        >>> from atdata.local import Index
        >>> from atdata.atmosphere import AtmosphereClient
        >>> index = Index(atmosphere=client)
        >>>
        >>> # Legacy (deprecated)
        >>> index = AtmosphereIndex(client)
    """

    def __init__(
        self,
        client: Atmosphere,
        *,
        data_store: Optional[PDSBlobStore] = None,
    ):
        """Initialize the atmosphere index.

        Args:
            client: Authenticated Atmosphere instance.
            data_store: Optional PDSBlobStore for writing shards as blobs.
                If provided, insert_dataset will upload shards to PDS.
        """
        import warnings

        warnings.warn(
            "AtmosphereIndex is deprecated. Use atdata.Index(atmosphere=client) "
            "instead for unified index access.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.client = client
        self._schema_publisher = SchemaPublisher(client)
        self._schema_loader = SchemaLoader(client)
        self._dataset_publisher = DatasetPublisher(client)
        self._dataset_loader = DatasetLoader(client)
        self._data_store = data_store

    @property
    def data_store(self) -> Optional[PDSBlobStore]:
        """The PDS blob store for writing shards, or None if not configured."""
        return self._data_store

    # Dataset operations

    def insert_dataset(
        self,
        ds: "Dataset",
        *,
        name: str,
        schema_ref: Optional[str] = None,
        **kwargs,
    ) -> AtmosphereIndexEntry:
        """Insert a dataset into ATProto.

        Args:
            ds: The Dataset to publish.
            name: Human-readable name.
            schema_ref: Optional schema AT URI. If None, auto-publishes schema.
            **kwargs: Additional options (description, tags, license).

        Returns:
            AtmosphereIndexEntry for the inserted dataset.
        """
        uri = self._dataset_publisher.publish(
            ds,
            name=name,
            schema_uri=schema_ref,
            description=kwargs.get("description"),
            tags=kwargs.get("tags"),
            license=kwargs.get("license"),
            auto_publish_schema=(schema_ref is None),
        )
        record = self._dataset_loader.get(uri)
        return AtmosphereIndexEntry(str(uri), record)

    def get_dataset(self, ref: str) -> AtmosphereIndexEntry:
        """Get a dataset by AT URI.

        Args:
            ref: AT URI of the dataset record.

        Returns:
            AtmosphereIndexEntry for the dataset.

        Raises:
            ValueError: If record is not a dataset.
        """
        record = self._dataset_loader.get(ref)
        return AtmosphereIndexEntry(ref, record)

    @property
    def datasets(self) -> Iterator[AtmosphereIndexEntry]:
        """Lazily iterate over all dataset entries (AbstractIndex protocol).

        Uses the authenticated user's repository.

        Yields:
            AtmosphereIndexEntry for each dataset.
        """
        records = self._dataset_loader.list_all()
        for rec in records:
            uri = rec.get("uri", "")
            yield AtmosphereIndexEntry(uri, rec.get("value", rec))

    def list_datasets(self, repo: Optional[str] = None) -> list[AtmosphereIndexEntry]:
        """Get all dataset entries as a materialized list (AbstractIndex protocol).

        Args:
            repo: DID of repository. Defaults to authenticated user.

        Returns:
            List of AtmosphereIndexEntry for each dataset.
        """
        records = self._dataset_loader.list_all(repo=repo)
        return [
            AtmosphereIndexEntry(rec.get("uri", ""), rec.get("value", rec))
            for rec in records
        ]

    # Schema operations

    def publish_schema(
        self,
        sample_type: "Type[Packable]",
        *,
        version: str = "1.0.0",
        **kwargs,
    ) -> str:
        """Publish a schema to ATProto.

        Args:
            sample_type: A Packable type (PackableSample subclass or @packable-decorated).
            version: Semantic version string.
            **kwargs: Additional options (description, metadata).

        Returns:
            AT URI of the schema record.
        """
        uri = self._schema_publisher.publish(
            sample_type,
            version=version,
            description=kwargs.get("description"),
            metadata=kwargs.get("metadata"),
        )
        return str(uri)

    def get_schema(self, ref: str) -> dict:
        """Get a schema record by AT URI or handle reference.

        Args:
            ref: AT URI of the schema record, or a handle reference
                in ``@handle/TypeName@version`` format.

        Returns:
            Schema record dictionary.

        Raises:
            ValueError: If record is not a schema or format is invalid.
            KeyError: If no matching schema found for a handle reference.
        """
        return self._schema_loader.get(ref)

    @property
    def schemas(self) -> Iterator[dict]:
        """Lazily iterate over all schema records (AbstractIndex protocol).

        Uses the authenticated user's repository.

        Yields:
            Schema records as dictionaries.
        """
        records = self._schema_loader.list_all()
        for rec in records:
            yield rec.get("value", rec)

    def list_schemas(self, repo: Optional[str] = None) -> list[dict]:
        """Get all schema records as a materialized list (AbstractIndex protocol).

        Args:
            repo: DID of repository. Defaults to authenticated user.

        Returns:
            List of schema records as dictionaries.
        """
        records = self._schema_loader.list_all(repo=repo)
        return [rec.get("value", rec) for rec in records]

    def get_schema_type(self, ref: str) -> "Type[Packable]":
        """Reconstruct a Python type from a schema record.

        Args:
            ref: AT URI of the schema record.

        Returns:
            Dynamically generated Packable type.

        Raises:
            ValueError: If schema cannot be decoded.
        """
        from .._schema_codec import _schema_to_type

        schema = self.get_schema(ref)
        return _schema_to_type(schema)

    def decode_schema(self, ref: str) -> "Type[Packable]":
        """Reconstruct a Python type from a schema record.

        .. deprecated::
            Use :meth:`get_schema_type` instead.
        """
        import warnings

        warnings.warn(
            "Atmosphere.decode_schema() is deprecated, use Atmosphere.get_schema_type() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_schema_type(ref)


# Deprecated alias for backward compatibility
AtmosphereClient = Atmosphere

__all__ = [
    # Client
    "Atmosphere",
    "AtmosphereClient",  # deprecated alias
    # Storage
    "PDSBlobStore",
    # Unified index (AbstractIndex protocol)
    "AtmosphereIndex",
    "AtmosphereIndexEntry",
    # Schema operations
    "SchemaPublisher",
    "SchemaLoader",
    # Dataset operations
    "DatasetPublisher",
    "DatasetLoader",
    # Lens operations
    "LensPublisher",
    "LensLoader",
    # Label operations
    "LabelPublisher",
    "LabelLoader",
    # Core types
    "AtUri",
    "LEXICON_NAMESPACE",
    # Lexicon-mirror types (Tier 1)
    "DatasetMetadata",
    "LexSchemaRecord",
    "LexDatasetEntry",
    "LexLensRecord",
    "LexLabelRecord",
    "LexCodeReference",
    "JsonSchemaFormat",
    "StorageHttp",
    "StorageS3",
    "StorageBlobs",
    "StorageUnion",
    "storage_from_record",
    "ShardChecksum",
    "HttpShardEntry",
    "S3ShardEntry",
    "BlobEntry",
    "DatasetSize",
    "ShardManifestRef",
]
