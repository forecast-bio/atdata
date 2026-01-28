"""PDS blob storage for dataset shards.

This module provides ``PDSBlobStore``, an implementation of the AbstractDataStore
protocol that stores dataset shards as ATProto blobs in a Personal Data Server.

This enables fully decentralized dataset storage where both metadata (records)
and data (blobs) live on the AT Protocol network.

Examples:
    >>> from atdata.atmosphere import AtmosphereClient, PDSBlobStore
    >>>
    >>> client = AtmosphereClient()
    >>> client.login("handle.bsky.social", "app-password")
    >>>
    >>> store = PDSBlobStore(client)
    >>> urls = store.write_shards(dataset, prefix="mnist/v1")
    >>> print(urls)
    ['at://did:plc:.../blob/bafyrei...', ...]
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import webdataset as wds

if TYPE_CHECKING:
    from ..dataset import Dataset
    from .._sources import BlobSource
    from .client import AtmosphereClient


@dataclass
class PDSBlobStore:
    """PDS blob store implementing AbstractDataStore protocol.

    Stores dataset shards as ATProto blobs, enabling decentralized dataset
    storage on the AT Protocol network.

    Each shard is written to a temporary tar file, then uploaded as a blob
    to the user's PDS. The returned URLs are AT URIs that can be resolved
    to HTTP URLs for streaming.

    Attributes:
        client: Authenticated AtmosphereClient instance.

    Examples:
        >>> store = PDSBlobStore(client)
        >>> urls = store.write_shards(dataset, prefix="training/v1")
        >>> # Returns AT URIs like:
        >>> # ['at://did:plc:abc/blob/bafyrei...', ...]
    """

    client: "AtmosphereClient"

    def write_shards(
        self,
        ds: "Dataset",
        *,
        prefix: str,
        maxcount: int = 10000,
        maxsize: float = 3e9,
        **kwargs: Any,
    ) -> list[str]:
        """Write dataset shards as PDS blobs.

        Creates tar archives from the dataset and uploads each as a blob
        to the authenticated user's PDS.

        Args:
            ds: The Dataset to write.
            prefix: Logical path prefix for naming (used in shard names only).
            maxcount: Maximum samples per shard (default: 10000).
            maxsize: Maximum shard size in bytes (default: 3GB, PDS limit).
            **kwargs: Additional args passed to wds.ShardWriter.

        Returns:
            List of AT URIs for the written blobs, in format:
            ``at://{did}/blob/{cid}``

        Raises:
            ValueError: If not authenticated.
            RuntimeError: If no shards were written.

        Note:
            PDS blobs have size limits (typically 50MB-5GB depending on PDS).
            Adjust maxcount/maxsize to stay within limits.
        """
        if not self.client.did:
            raise ValueError("Client must be authenticated to upload blobs")

        did = self.client.did
        blob_urls: list[str] = []

        # Write shards to temp files, upload each as blob
        with tempfile.TemporaryDirectory() as temp_dir:
            shard_pattern = f"{temp_dir}/shard-%06d.tar"
            written_files: list[str] = []

            # Track written files via custom post callback
            def track_file(fname: str) -> None:
                written_files.append(fname)

            with wds.writer.ShardWriter(
                shard_pattern,
                maxcount=maxcount,
                maxsize=maxsize,
                post=track_file,
                **kwargs,
            ) as sink:
                for sample in ds.ordered(batch_size=None):
                    sink.write(sample.as_wds)

            if not written_files:
                raise RuntimeError("No shards written")

            # Upload each shard as a blob
            for shard_path in written_files:
                with open(shard_path, "rb") as f:
                    shard_data = f.read()

                blob_ref = self.client.upload_blob(
                    shard_data,
                    mime_type="application/x-tar",
                )

                # Extract CID from blob reference
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


__all__ = ["PDSBlobStore"]
