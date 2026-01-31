"""Dataset record publishing and loading for ATProto.

This module provides classes for publishing dataset index records to ATProto
and loading them back. Dataset records are published as
``ac.foundation.dataset.record`` records.
"""

from typing import Type, TypeVar, Optional
import msgpack

from .client import AtmosphereClient
from .schema import SchemaPublisher
from ._types import (
    AtUri,
    DatasetRecord,
    StorageLocation,
    LEXICON_NAMESPACE,
)

# Import for type checking only to avoid circular imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..dataset import Dataset
    from .._protocols import Packable

ST = TypeVar("ST", bound="Packable")


class DatasetPublisher:
    """Publishes dataset index records to ATProto.

    This class creates dataset records that reference a schema and point to
    external storage (WebDataset URLs) or ATProto blobs.

    Examples:
        >>> dataset = atdata.Dataset[MySample]("s3://bucket/data-{000000..000009}.tar")
        >>>
        >>> client = AtmosphereClient()
        >>> client.login("handle", "password")
        >>>
        >>> publisher = DatasetPublisher(client)
        >>> uri = publisher.publish(
        ...     dataset,
        ...     name="My Training Data",
        ...     description="Training data for my model",
        ...     tags=["computer-vision", "training"],
        ... )
    """

    def __init__(self, client: AtmosphereClient):
        """Initialize the dataset publisher.

        Args:
            client: Authenticated AtmosphereClient instance.
        """
        self.client = client
        self._schema_publisher = SchemaPublisher(client)

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
        # Ensure we have a schema reference
        if schema_uri is None:
            if not auto_publish_schema:
                raise ValueError(
                    "schema_uri is required when auto_publish_schema=False"
                )
            # Auto-publish the schema
            schema_uri_obj = self._schema_publisher.publish(
                dataset.sample_type,
                version=schema_version,
            )
            schema_uri = str(schema_uri_obj)

        # Build the storage location
        storage = StorageLocation(
            kind="external",
            urls=[dataset.url],
        )

        # Build dataset record
        metadata_bytes: Optional[bytes] = None
        if dataset.metadata is not None:
            metadata_bytes = msgpack.packb(dataset.metadata)

        dataset_record = DatasetRecord(
            name=name,
            schema_ref=schema_uri,
            storage=storage,
            description=description,
            tags=tags or [],
            license=license,
            metadata=metadata_bytes,
        )

        # Publish to ATProto
        return self.client.create_record(
            collection=f"{LEXICON_NAMESPACE}.record",
            record=dataset_record.to_record(),
            rkey=rkey,
            validate=False,
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
        rkey: Optional[str] = None,
    ) -> AtUri:
        """Publish a dataset record with explicit URLs.

        This method allows publishing a dataset record without having a
        Dataset object, useful for registering existing WebDataset files.

        Args:
            urls: List of WebDataset URLs with brace notation.
            schema_uri: AT URI of the schema record.
            name: Human-readable dataset name.
            description: Human-readable description.
            tags: Searchable tags for discovery.
            license: SPDX license identifier.
            metadata: Arbitrary metadata dictionary.
            rkey: Optional explicit record key.

        Returns:
            The AT URI of the created dataset record.
        """
        storage = StorageLocation(
            kind="external",
            urls=urls,
        )

        metadata_bytes: Optional[bytes] = None
        if metadata is not None:
            metadata_bytes = msgpack.packb(metadata)

        dataset_record = DatasetRecord(
            name=name,
            schema_ref=schema_uri,
            storage=storage,
            description=description,
            tags=tags or [],
            license=license,
            metadata=metadata_bytes,
        )

        return self.client.create_record(
            collection=f"{LEXICON_NAMESPACE}.record",
            record=dataset_record.to_record(),
            rkey=rkey,
            validate=False,
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
        # Upload all blobs
        blob_refs = []
        for blob_data in blobs:
            blob_ref = self.client.upload_blob(blob_data, mime_type=mime_type)
            blob_refs.append(blob_ref)

        # Create storage location with blob references
        storage = StorageLocation(
            kind="blobs",
            blob_refs=blob_refs,
        )

        metadata_bytes: Optional[bytes] = None
        if metadata is not None:
            metadata_bytes = msgpack.packb(metadata)

        dataset_record = DatasetRecord(
            name=name,
            schema_ref=schema_uri,
            storage=storage,
            description=description,
            tags=tags or [],
            license=license,
            metadata=metadata_bytes,
        )

        return self.client.create_record(
            collection=f"{LEXICON_NAMESPACE}.record",
            record=dataset_record.to_record(),
            rkey=rkey,
            validate=False,
        )


class DatasetLoader:
    """Loads dataset records from ATProto.

    This class fetches dataset index records and can create Dataset objects
    from them. Note that loading a dataset requires having the corresponding
    Python class for the sample type.

    Examples:
        >>> client = AtmosphereClient()
        >>> loader = DatasetLoader(client)
        >>>
        >>> # List available datasets
        >>> datasets = loader.list()
        >>> for ds in datasets:
        ...     print(ds["name"], ds["schemaRef"])
        >>>
        >>> # Get a specific dataset record
        >>> record = loader.get("at://did:plc:abc/ac.foundation.dataset.record/xyz")
    """

    def __init__(self, client: AtmosphereClient):
        """Initialize the dataset loader.

        Args:
            client: AtmosphereClient instance.
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
            Either "external" or "blobs".

        Raises:
            ValueError: If storage type is unknown.
        """
        record = self.get(uri)
        storage = record.get("storage", {})
        storage_type = storage.get("$type", "")

        if "storageExternal" in storage_type:
            return "external"
        elif "storageBlobs" in storage_type:
            return "blobs"
        else:
            raise ValueError(f"Unknown storage type: {storage_type}")

    def get_urls(self, uri: str | AtUri) -> list[str]:
        """Get the WebDataset URLs from a dataset record.

        Args:
            uri: The AT URI of the dataset record.

        Returns:
            List of WebDataset URLs.

        Raises:
            ValueError: If the storage type is not external URLs.
        """
        record = self.get(uri)
        storage = record.get("storage", {})

        storage_type = storage.get("$type", "")
        if "storageExternal" in storage_type:
            return storage.get("urls", [])
        elif "storageBlobs" in storage_type:
            raise ValueError(
                "Dataset uses blob storage, not external URLs. "
                "Use get_blob_urls() instead."
            )
        else:
            raise ValueError(f"Unknown storage type: {storage_type}")

    def get_blobs(self, uri: str | AtUri) -> list[dict]:
        """Get the blob references from a dataset record.

        Args:
            uri: The AT URI of the dataset record.

        Returns:
            List of blob reference dicts with keys: $type, ref, mimeType, size.

        Raises:
            ValueError: If the storage type is not blobs.
        """
        record = self.get(uri)
        storage = record.get("storage", {})

        storage_type = storage.get("$type", "")
        if "storageBlobs" in storage_type:
            return storage.get("blobs", [])
        elif "storageExternal" in storage_type:
            raise ValueError(
                "Dataset uses external URL storage, not blobs. Use get_urls() instead."
            )
        else:
            raise ValueError(f"Unknown storage type: {storage_type}")

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

        blobs = self.get_blobs(uri)
        did = parsed_uri.authority

        urls = []
        for blob in blobs:
            # Extract CID from blob reference
            ref = blob.get("ref", {})
            cid = ref.get("$link") if isinstance(ref, dict) else str(ref)
            if cid:
                url = self.client.get_blob_url(did, cid)
                urls.append(url)

        return urls

    def get_metadata(self, uri: str | AtUri) -> Optional[dict]:
        """Get the metadata from a dataset record.

        Args:
            uri: The AT URI of the dataset record.

        Returns:
            The metadata dictionary, or None if no metadata.
        """
        record = self.get(uri)
        metadata_bytes = record.get("metadata")

        if metadata_bytes is None:
            return None

        return msgpack.unpackb(metadata_bytes, raw=False)

    def to_dataset(
        self,
        uri: str | AtUri,
        sample_type: Type[ST],
    ) -> "Dataset[ST]":
        """Create a Dataset object from an ATProto record.

        This method creates a Dataset instance from a published record.
        You must provide the sample type class, which should match the
        schema referenced by the record.

        Supports both external URL storage and ATProto blob storage.

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

        if storage_type == "external":
            urls = self.get_urls(uri)
        else:
            urls = self.get_blob_urls(uri)

        if not urls:
            raise ValueError("Dataset record has no storage URLs")

        # Use the first URL (multi-URL support could be added later)
        url = urls[0]

        # Get metadata URL if available
        record = self.get(uri)
        metadata_url = record.get("metadataUrl")

        return Dataset[sample_type](url, metadata_url=metadata_url)
