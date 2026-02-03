"""PDS blob storage for dataset shards.

This module provides ``PDSBlobStore``, an implementation of the AbstractDataStore
protocol that stores dataset shards as ATProto blobs in a Personal Data Server.

This enables fully decentralized dataset storage where both metadata (records)
and data (blobs) live on the AT Protocol network.

Examples:
    >>> from atdata.atmosphere import Atmosphere, PDSBlobStore
    >>>
    >>> atmo = Atmosphere.login("handle.bsky.social", "app-password")
    >>>
    >>> store = PDSBlobStore(atmo)
    >>> urls = store.write_shards(dataset, prefix="mnist/v1")
    >>> print(urls)
    ['at://did:plc:.../blob/bafyrei...', ...]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

#: Maximum size in bytes for a single PDS blob upload (50 MB).
PDS_BLOB_LIMIT_BYTES: int = 50_000_000

#: Maximum total dataset size in bytes for atmosphere uploads (1 GB).
PDS_TOTAL_DATASET_LIMIT_BYTES: int = 1_000_000_000

if TYPE_CHECKING:
    from ..dataset import Dataset
    from .._sources import BlobSource
    from .client import Atmosphere


@dataclass
class PDSBlobStore:
    """PDS blob store implementing AbstractDataStore protocol.

    Stores dataset shards as ATProto blobs, enabling decentralized dataset
    storage on the AT Protocol network.

    Each shard is written to a temporary tar file, then uploaded as a blob
    to the user's PDS. The returned URLs are AT URIs that can be resolved
    to HTTP URLs for streaming.

    Attributes:
        client: Authenticated Atmosphere instance.

    Examples:
        >>> store = PDSBlobStore(client)
        >>> urls = store.write_shards(dataset, prefix="training/v1")
        >>> # Returns AT URIs like:
        >>> # ['at://did:plc:abc/blob/bafyrei...', ...]
    """

    client: "Atmosphere"

    def write_shards(
        self,
        ds: "Dataset",
        *,
        prefix: str,
        **kwargs: Any,
    ) -> list[str]:
        """Upload existing dataset shards as PDS blobs.

        Reads the tar archives already written to disk by the caller and
        uploads each as a blob to the authenticated user's PDS. This
        avoids re-serializing samples that have already been written.

        Args:
            ds: The Dataset whose shards to upload.
            prefix: Logical path prefix (unused, kept for protocol compat).
            **kwargs: Unused, kept for protocol compatibility.

        Returns:
            List of AT URIs for the uploaded blobs, in format:
            ``at://{did}/blob/{cid}``

        Raises:
            ValueError: If not authenticated.
            RuntimeError: If no shards are found on the dataset.
        """
        if not self.client.did:
            raise ValueError("Client must be authenticated to upload blobs")

        did = self.client.did
        blob_urls: list[str] = []

        shard_paths = ds.list_shards()
        if not shard_paths:
            raise RuntimeError("No shards to upload")

        for shard_url in shard_paths:
            with open(shard_url, "rb") as f:
                shard_data = f.read()

            blob_ref = self.client.upload_blob(
                shard_data,
                mime_type="application/x-tar",
            )

            cid = blob_ref["ref"]["$link"]
            at_uri = f"at://{did}/blob/{cid}"
            blob_urls.append(at_uri)

        return blob_urls

    def read_url(self, url: str) -> str:
        """Resolve an AT URI blob reference to an HTTP URL.

        Transforms ``at://did/blob/cid`` URIs to HTTP URLs that can be
        streamed by WebDataset.

        Args:
            url: AT URI in format ``at://{did}/blob/{cid}``.

        Returns:
            HTTP URL for fetching the blob via PDS API.

        Raises:
            ValueError: If URL format is invalid or PDS cannot be resolved.
        """
        if not url.startswith("at://"):
            # Not an AT URI, return unchanged
            return url

        # Parse at://did/blob/cid
        parts = url[5:].split("/")  # Remove 'at://'
        if len(parts) != 3 or parts[1] != "blob":
            raise ValueError(f"Invalid blob AT URI format: {url}")

        did, _, cid = parts
        return self.client.get_blob_url(did, cid)

    def supports_streaming(self) -> bool:
        """PDS blobs support streaming via HTTP.

        Returns:
            True.
        """
        return True

    def create_source(self, urls: list[str]) -> "BlobSource":
        """Create a BlobSource for reading these AT URIs.

        This is a convenience method for creating a DataSource that can
        stream the blobs written by this store.

        Args:
            urls: List of AT URIs from write_shards().

        Returns:
            BlobSource configured for the given URLs.

        Raises:
            ValueError: If URLs are not valid AT URIs.
        """
        from .._sources import BlobSource

        blob_refs: list[dict[str, str]] = []

        for url in urls:
            if not url.startswith("at://"):
                raise ValueError(f"Not an AT URI: {url}")

            parts = url[5:].split("/")
            if len(parts) != 3 or parts[1] != "blob":
                raise ValueError(f"Invalid blob AT URI: {url}")

            did, _, cid = parts
            blob_refs.append({"did": did, "cid": cid})

        return BlobSource(blob_refs=blob_refs)


__all__ = ["PDS_BLOB_LIMIT_BYTES", "PDS_TOTAL_DATASET_LIMIT_BYTES", "PDSBlobStore"]
