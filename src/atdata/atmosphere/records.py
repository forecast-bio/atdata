"""Dataset record publishing and loading for ATProto.

This module provides classes for publishing dataset index records to ATProto
and loading them back. Dataset records are published as
``ac.foundation.dataset.record`` records.
"""

from typing import Type, TypeVar, Optional
import msgpack

from .client import Atmosphere
from .schema import SchemaPublisher
from ._types import AtUri, LEXICON_NAMESPACE
from ._lexicon_types import (
    DatasetMetadata,
    LexDatasetRecord,
    StorageHttp,
    StorageS3,
    StorageBlobs,
    HttpShardEntry,
    S3ShardEntry,
    BlobEntry,
    ShardChecksum,
)

# Import for type checking only to avoid circular imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..dataset import Dataset
    from .._protocols import Packable

ST = TypeVar("ST", bound="Packable")


def _placeholder_checksum() -> ShardChecksum:
    """Return an empty checksum placeholder for shards without pre-computed digests."""
    return ShardChecksum(algorithm="none", digest="")


class DatasetPublisher:
    """Publishes dataset index records to ATProto.

    This class creates dataset records that reference a schema and point to
    HTTP storage, S3 storage, or ATProto blobs.

    Examples:
        >>> dataset = atdata.Dataset[MySample]("https://example.com/data-000000.tar")
        >>>
        >>> atmo = Atmosphere.login("handle", "password")
        >>>
        >>> publisher = DatasetPublisher(atmo)
        >>> uri = publisher.publish(
        ...     dataset,
        ...     name="My Training Data",
        ...     description="Training data for my model",
        ...     tags=["computer-vision", "training"],
        ... )
    """

    def __init__(self, client: Atmosphere):
        """Initialize the dataset publisher.

        Args:
            client: Authenticated Atmosphere instance.
        """
        self.client = client
        self._schema_publisher = SchemaPublisher(client)

    def _create_record(
        self,
        storage: "StorageHttp | StorageS3 | StorageBlobs",
        *,
        name: str,
        schema_uri: str,
        description: Optional[str] = None,
        tags: Optional[list[str]] = None,
        license: Optional[str] = None,
        metadata: Optional[DatasetMetadata | dict] = None,
        rkey: Optional[str] = None,
    ) -> AtUri:
        """Build a LexDatasetRecord and publish it to ATProto."""
        typed_metadata: Optional[DatasetMetadata] = None
        if isinstance(metadata, DatasetMetadata):
            typed_metadata = metadata
        elif isinstance(metadata, dict):
            typed_metadata = DatasetMetadata.from_dict(metadata)

        dataset_record = LexDatasetRecord(
            name=name,
            schema_ref=schema_uri,
            storage=storage,
            description=description,
            tags=tags or [],
            license=license,
            metadata=typed_metadata,
        )

        return self.client.create_record(
            collection=f"{LEXICON_NAMESPACE}.record",
            record=dataset_record.to_record(),
            rkey=rkey,
            validate=False,
        )

    def publish(
        self,
        dataset: "Dataset[ST]",
        *,
        name: str,
        schema_uri: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[list[str]] = None,
        license: Optional[str] = None,
        auto_publish_schema: bool = True,
        schema_version: str = "1.0.0",
        rkey: Optional[str] = None,
    ) -> AtUri:
        """Publish a dataset index record to ATProto.

        Args:
            dataset: The Dataset to publish.
            name: Human-readable dataset name.
            schema_uri: AT URI of the schema record. If not provided and
                auto_publish_schema is True, the schema will be published.
            description: Human-readable description.
            tags: Searchable tags for discovery.
            license: SPDX license identifier (e.g., 'MIT', 'Apache-2.0').
            auto_publish_schema: If True and schema_uri not provided,
                automatically publish the schema first.
            schema_version: Version for auto-published schema.
            rkey: Optional explicit record key.

        Returns:
            The AT URI of the created dataset record.

        Raises:
            ValueError: If schema_uri is not provided and auto_publish_schema is False.
        """
        if schema_uri is None:
            if not auto_publish_schema:
                raise ValueError(
                    "schema_uri is required when auto_publish_schema=False"
                )
            schema_uri_obj = self._schema_publisher.publish(
                dataset.sample_type,
                version=schema_version,
            )
            schema_uri = str(schema_uri_obj)

        shard_urls = dataset.list_shards()
        storage = StorageHttp(
            shards=[
                HttpShardEntry(url=url, checksum=_placeholder_checksum())
                for url in shard_urls
            ]
        )

        return self._create_record(
            storage,
            name=name,
            schema_uri=schema_uri,
            description=description,
            tags=tags,
            license=license,
            metadata=dataset.metadata,
            rkey=rkey,
        )

    def publish_with_urls(
        self,
        urls: list[str],
        schema_uri: str,
        *,
        name: str,
        description: Optional[str] = None,
        tags: Optional[list[str]] = None,
        license: Optional[str] = None,
        metadata: Optional[dict] = None,
        checksums: Optional[list[ShardChecksum]] = None,
        rkey: Optional[str] = None,
    ) -> AtUri:
        """Publish a dataset record with explicit HTTP URLs.

        This method allows publishing a dataset record without having a
        Dataset object, useful for registering existing WebDataset files.
        Each URL should be an individual shard (no brace notation).

        Args:
            urls: List of individual shard URLs.
            schema_uri: AT URI of the schema record.
            name: Human-readable dataset name.
            description: Human-readable description.
            tags: Searchable tags for discovery.
            license: SPDX license identifier.
            metadata: Arbitrary metadata dictionary.
            checksums: Per-shard checksums. If not provided, empty checksums
                are used.
            rkey: Optional explicit record key.

        Returns:
            The AT URI of the created dataset record.
        """
        if checksums and len(checksums) != len(urls):
            raise ValueError(
                f"checksums length ({len(checksums)}) must match "
                f"urls length ({len(urls)})"
            )

        shards = [
            HttpShardEntry(
                url=url,
                checksum=checksums[i] if checksums else _placeholder_checksum(),
            )
            for i, url in enumerate(urls)
        ]

        return self._create_record(
            StorageHttp(shards=shards),
            name=name,
            schema_uri=schema_uri,
            description=description,
            tags=tags,
            license=license,
            metadata=metadata,
            rkey=rkey,
        )

    def publish_with_s3(
        self,
        bucket: str,
        keys: list[str],
        schema_uri: str,
        *,
        name: str,
        region: Optional[str] = None,
        endpoint: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[list[str]] = None,
        license: Optional[str] = None,
        metadata: Optional[dict] = None,
        checksums: Optional[list[ShardChecksum]] = None,
        rkey: Optional[str] = None,
    ) -> AtUri:
        """Publish a dataset record with S3 storage.

        Args:
            bucket: S3 bucket name.
            keys: List of S3 object keys for shard files.
            schema_uri: AT URI of the schema record.
            name: Human-readable dataset name.
            region: AWS region (e.g., 'us-east-1').
            endpoint: Custom S3-compatible endpoint URL.
            description: Human-readable description.
            tags: Searchable tags for discovery.
            license: SPDX license identifier.
            metadata: Arbitrary metadata dictionary.
            checksums: Per-shard checksums.
            rkey: Optional explicit record key.

        Returns:
            The AT URI of the created dataset record.
        """
        if checksums and len(checksums) != len(keys):
            raise ValueError(
                f"checksums length ({len(checksums)}) must match "
                f"keys length ({len(keys)})"
            )

        shards = [
            S3ShardEntry(
                key=key,
                checksum=checksums[i] if checksums else _placeholder_checksum(),
            )
            for i, key in enumerate(keys)
        ]

        return self._create_record(
            StorageS3(bucket=bucket, shards=shards, region=region, endpoint=endpoint),
            name=name,
            schema_uri=schema_uri,
            description=description,
            tags=tags,
            license=license,
            metadata=metadata,
            rkey=rkey,
        )

    def publish_with_blob_refs(
        self,
        blob_refs: list[dict],
        schema_uri: str,
        *,
        name: str,
        description: Optional[str] = None,
        tags: Optional[list[str]] = None,
        license: Optional[str] = None,
        metadata: Optional[dict] = None,
        checksums: Optional[list[ShardChecksum]] = None,
        rkey: Optional[str] = None,
    ) -> AtUri:
        """Publish a dataset record with pre-uploaded blob references.

        Unlike ``publish_with_blobs`` (which takes raw bytes and uploads them),
        this method accepts blob ref dicts that have already been uploaded to
        the PDS.  The refs are embedded directly in the record so the PDS
        retains the blobs.

        Args:
            blob_refs: List of blob reference dicts as returned by
                ``Atmosphere.upload_blob()``.  Each dict must contain
                ``$type``, ``ref`` (with ``$link``), ``mimeType``, and ``size``.
            schema_uri: AT URI of the schema record.
            name: Human-readable dataset name.
            description: Human-readable description.
            tags: Searchable tags for discovery.
            license: SPDX license identifier.
            metadata: Arbitrary metadata dictionary.
            checksums: Per-shard checksums. If not provided, empty checksums
                are used.
            rkey: Optional explicit record key.

        Returns:
            The AT URI of the created dataset record.
        """
        if checksums and len(checksums) != len(blob_refs):
            raise ValueError(
                f"checksums length ({len(checksums)}) must match "
                f"blob_refs length ({len(blob_refs)})"
            )

        blob_entries = [
            BlobEntry(
                blob=ref,
                checksum=checksums[i] if checksums else _placeholder_checksum(),
            )
            for i, ref in enumerate(blob_refs)
        ]

        return self._create_record(
            StorageBlobs(blobs=blob_entries),
            name=name,
            schema_uri=schema_uri,
            description=description,
            tags=tags,
            license=license,
            metadata=metadata,
            rkey=rkey,
        )

    def publish_with_blobs(
        self,
        blobs: list[bytes],
        schema_uri: str,
        *,
        name: str,
        description: Optional[str] = None,
        tags: Optional[list[str]] = None,
        license: Optional[str] = None,
        metadata: Optional[dict] = None,
        mime_type: str = "application/x-tar",
        rkey: Optional[str] = None,
    ) -> AtUri:
        """Publish a dataset with data stored as ATProto blobs.

        This method uploads the provided data as blobs to the PDS and creates
        a dataset record referencing them. Suitable for smaller datasets that
        fit within blob size limits (typically 50MB per blob, configurable).

        Args:
            blobs: List of binary data (e.g., tar shards) to upload as blobs.
            schema_uri: AT URI of the schema record.
            name: Human-readable dataset name.
            description: Human-readable description.
            tags: Searchable tags for discovery.
            license: SPDX license identifier.
            metadata: Arbitrary metadata dictionary.
            mime_type: MIME type for the blobs (default: application/x-tar).
            rkey: Optional explicit record key.

        Returns:
            The AT URI of the created dataset record.

        Note:
            Blobs are only retained by the PDS when referenced in a committed
            record. This method handles that automatically.
        """
        blob_entries = []
        for blob_data in blobs:
            blob_ref = self.client.upload_blob(blob_data, mime_type=mime_type)
            import hashlib

            digest = hashlib.sha256(blob_data).hexdigest()
            blob_entries.append(
                BlobEntry(
                    blob=blob_ref,
                    checksum=ShardChecksum(algorithm="sha256", digest=digest),
                )
            )

        return self._create_record(
            StorageBlobs(blobs=blob_entries),
            name=name,
            schema_uri=schema_uri,
            description=description,
            tags=tags,
            license=license,
            metadata=metadata,
            rkey=rkey,
        )


class DatasetLoader:
    """Loads dataset records from ATProto.

    This class fetches dataset index records and can create Dataset objects
    from them. Note that loading a dataset requires having the corresponding
    Python class for the sample type.

    Examples:
        >>> atmo = Atmosphere.login("handle", "password")
        >>> loader = DatasetLoader(atmo)
        >>>
        >>> # List available datasets
        >>> datasets = loader.list()
        >>> for ds in datasets:
        ...     print(ds["name"], ds["schemaRef"])
        >>>
        >>> # Get a specific dataset record
        >>> record = loader.get("at://did:plc:abc/ac.foundation.dataset.record/xyz")
    """

    def __init__(self, client: Atmosphere):
        """Initialize the dataset loader.

        Args:
            client: Atmosphere instance.
        """
        self.client = client

    def get(self, uri: str | AtUri) -> dict:
        """Fetch a dataset record by AT URI.

        Args:
            uri: The AT URI of the dataset record.

        Returns:
            The dataset record as a dictionary.

        Raises:
            ValueError: If the record is not a dataset record.
        """
        record = self.client.get_record(uri)

        expected_type = f"{LEXICON_NAMESPACE}.record"
        if record.get("$type") != expected_type:
            raise ValueError(
                f"Record at {uri} is not a dataset record. "
                f"Expected $type='{expected_type}', got '{record.get('$type')}'"
            )

        return record

    def get_typed(self, uri: str | AtUri) -> LexDatasetRecord:
        """Fetch a dataset record and return as a typed object.

        Args:
            uri: The AT URI of the dataset record.

        Returns:
            LexDatasetRecord instance.
        """
        record = self.get(uri)
        return LexDatasetRecord.from_record(record)

    def list_all(
        self,
        repo: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """List dataset records from a repository.

        Args:
            repo: The DID of the repository. Defaults to authenticated user.
            limit: Maximum number of records to return.

        Returns:
            List of dataset records.
        """
        return self.client.list_datasets(repo=repo, limit=limit)

    def get_storage_type(self, uri: str | AtUri) -> str:
        """Get the storage type of a dataset record.

        Args:
            uri: The AT URI of the dataset record.

        Returns:
            One of "http", "s3", "blobs", or "external" (legacy).

        Raises:
            ValueError: If storage type is unknown.
        """
        record = self.get(uri)
        storage = record.get("storage", {})
        storage_type = storage.get("$type", "")

        if "storageHttp" in storage_type:
            return "http"
        elif "storageS3" in storage_type:
            return "s3"
        elif "storageBlobs" in storage_type:
            return "blobs"
        elif "storageExternal" in storage_type:
            return "external"
        else:
            raise ValueError(f"Unknown storage type: {storage_type}")

    def get_urls(self, uri: str | AtUri) -> list[str]:
        """Get the WebDataset URLs from a dataset record.

        Supports storageHttp, storageS3, and legacy storageExternal formats.

        Args:
            uri: The AT URI of the dataset record.

        Returns:
            List of WebDataset URLs.

        Raises:
            ValueError: If the storage type is blob-only.
        """
        record = self.get(uri)
        storage = record.get("storage", {})
        storage_type = storage.get("$type", "")

        if "storageHttp" in storage_type:
            return [s["url"] for s in storage.get("shards", [])]
        elif "storageS3" in storage_type:
            bucket = storage.get("bucket", "")
            endpoint = storage.get("endpoint")
            urls = []
            for s in storage.get("shards", []):
                if endpoint:
                    urls.append(f"{endpoint.rstrip('/')}/{bucket}/{s['key']}")
                else:
                    urls.append(f"s3://{bucket}/{s['key']}")
            return urls
        elif "storageExternal" in storage_type:
            return storage.get("urls", [])
        elif "storageBlobs" in storage_type:
            raise ValueError(
                "Dataset uses blob storage, not URLs. Use get_blob_urls() instead."
            )
        else:
            raise ValueError(f"Unknown storage type: {storage_type}")

    def get_s3_info(self, uri: str | AtUri) -> dict:
        """Get S3 storage details from a dataset record.

        Args:
            uri: The AT URI of the dataset record.

        Returns:
            Dict with keys: bucket, keys, region (optional), endpoint (optional).

        Raises:
            ValueError: If the storage type is not S3.
        """
        record = self.get(uri)
        storage = record.get("storage", {})
        storage_type = storage.get("$type", "")

        if "storageS3" not in storage_type:
            raise ValueError(
                f"Dataset does not use S3 storage. Storage type: {storage_type}"
            )

        return {
            "bucket": storage.get("bucket", ""),
            "keys": [s["key"] for s in storage.get("shards", [])],
            "region": storage.get("region"),
            "endpoint": storage.get("endpoint"),
        }

    def get_blobs(self, uri: str | AtUri) -> list[dict]:
        """Get the blob references from a dataset record.

        Args:
            uri: The AT URI of the dataset record.

        Returns:
            List of blob entry dicts.

        Raises:
            ValueError: If the storage type is not blobs.
        """
        record = self.get(uri)
        storage = record.get("storage", {})

        storage_type = storage.get("$type", "")
        if "storageBlobs" in storage_type:
            return storage.get("blobs", [])
        else:
            raise ValueError(
                f"Dataset does not use blob storage. Storage type: {storage_type}. "
                "Use get_urls() instead."
            )

    def get_blob_urls(self, uri: str | AtUri) -> list[str]:
        """Get fetchable URLs for blob-stored dataset shards.

        This resolves the PDS endpoint and constructs URLs that can be
        used to fetch the blob data directly.

        Args:
            uri: The AT URI of the dataset record.

        Returns:
            List of URLs for fetching the blob data.

        Raises:
            ValueError: If storage type is not blobs or PDS cannot be resolved.
        """
        if isinstance(uri, str):
            parsed_uri = AtUri.parse(uri)
        else:
            parsed_uri = uri

        blob_entries = self.get_blobs(uri)
        did = parsed_uri.authority

        urls = []
        for entry in blob_entries:
            # Handle both new blobEntry format and legacy bare blob format
            blob = entry.get("blob", entry)
            ref = blob.get("ref", {})
            cid = ref.get("$link") if isinstance(ref, dict) else str(ref)
            if cid:
                url = self.client.get_blob_url(did, cid)
                urls.append(url)

        return urls

    def get_metadata(self, uri: str | AtUri) -> Optional[dict]:
        """Get the metadata from a dataset record as a plain dict.

        Handles both the new structured metadata format (JSON object) and the
        legacy ``$bytes``-encoded msgpack format for backward compatibility.

        Args:
            uri: The AT URI of the dataset record.

        Returns:
            The metadata as a flat dictionary, or None if no metadata.
        """
        import base64

        record = self.get(uri)
        metadata_raw = record.get("metadata")

        if metadata_raw is None:
            return None

        # Legacy: ATProto $bytes-encoded msgpack.
        if isinstance(metadata_raw, dict) and "$bytes" in metadata_raw:
            metadata_bytes = base64.b64decode(metadata_raw["$bytes"])
            return msgpack.unpackb(metadata_bytes, raw=False)

        # Legacy: raw msgpack bytes (local storage / tests).
        if isinstance(metadata_raw, bytes):
            return msgpack.unpackb(metadata_raw, raw=False)

        # New structured format: plain JSON object.
        if isinstance(metadata_raw, dict):
            return DatasetMetadata.from_record(metadata_raw).to_dict()

        raise ValueError(f"Unexpected metadata format: {type(metadata_raw).__name__}")

    def get_metadata_typed(self, uri: str | AtUri) -> Optional[DatasetMetadata]:
        """Get the metadata from a dataset record as a typed object.

        Handles both the new structured metadata format and the legacy
        ``$bytes``-encoded msgpack format.

        Args:
            uri: The AT URI of the dataset record.

        Returns:
            DatasetMetadata instance, or None if no metadata.
        """
        record = self.get(uri)
        raw = record.get("metadata")
        if raw is None:
            return None
        # Delegate to LexDatasetRecord.from_record which handles all formats.
        typed_record = LexDatasetRecord.from_record(record)
        return typed_record.metadata

    def to_dataset(
        self,
        uri: str | AtUri,
        sample_type: Type[ST],
    ) -> "Dataset[ST]":
        """Create a Dataset object from an ATProto record.

        This method creates a Dataset instance from a published record.
        You must provide the sample type class, which should match the
        schema referenced by the record.

        Supports HTTP, S3, blob, and legacy external storage.

        Args:
            uri: The AT URI of the dataset record.
            sample_type: The Python class for the sample type.

        Returns:
            A Dataset instance configured from the record.

        Raises:
            ValueError: If no storage URLs can be resolved.

        Examples:
            >>> loader = DatasetLoader(client)
            >>> dataset = loader.to_dataset(uri, MySampleType)
            >>> for batch in dataset.shuffled(batch_size=32):
            ...     process(batch)
        """
        # Import here to avoid circular import
        from ..dataset import Dataset

        storage_type = self.get_storage_type(uri)

        if storage_type == "blobs":
            urls = self.get_blob_urls(uri)
        else:
            urls = self.get_urls(uri)

        if not urls:
            raise ValueError("Dataset record has no storage URLs")

        # Use the first URL (multi-URL support could be added later)
        url = urls[0]

        # Get metadata URL if available
        record = self.get(uri)
        metadata_url = record.get("metadataUrl")

        return Dataset[sample_type](url, metadata_url=metadata_url)
