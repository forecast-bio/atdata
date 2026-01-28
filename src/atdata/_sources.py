"""Data source implementations for streaming dataset shards.

This module provides concrete implementations of the DataSource protocol,
enabling Dataset to work with various data backends without URL transformation
hacks.

Classes:
    URLSource: WebDataset-compatible URLs (http, https, pipe, gs, etc.)
    S3Source: S3-compatible storage with explicit credentials

The key insight is that WebDataset's tar_file_expander only needs
{url: str, stream: IO} dicts - it doesn't care how streams are created.
By providing streams directly, we can support private repos, custom
endpoints, and future backends like ATProto blobs.

Examples:
    >>> # Standard URL (uses WebDataset's gopen)
    >>> source = URLSource("https://example.com/data-{000..009}.tar")
    >>> ds = Dataset[MySample](source)
    >>>
    >>> # Private S3 with credentials
    >>> source = S3Source(
    ...     bucket="my-bucket",
    ...     keys=["train/shard-000.tar", "train/shard-001.tar"],
    ...     endpoint="https://my-r2.cloudflarestorage.com",
    ...     access_key="...",
    ...     secret_key="...",
    ... )
    >>> ds = Dataset[MySample](source)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import IO, Iterator, Any

import braceexpand
import webdataset as wds


@dataclass
class URLSource:
    """Data source for WebDataset-compatible URLs.

    Wraps WebDataset's gopen to open URLs using built-in handlers for
    http, https, pipe, gs, hf, sftp, etc. Supports brace expansion
    for shard patterns like "data-{000..099}.tar".

    This is the default source type when a string URL is passed to Dataset.

    Attributes:
        url: URL or brace pattern for the shards.

    Examples:
        >>> source = URLSource("https://example.com/train-{000..009}.tar")
        >>> for shard_id, stream in source.shards:
        ...     print(f"Streaming {shard_id}")
    """

    url: str

    def list_shards(self) -> list[str]:
        """Expand brace pattern and return list of shard URLs."""
        return list(braceexpand.braceexpand(self.url))

    # Legacy alias for backwards compatibility
    @property
    def shard_list(self) -> list[str]:
        """Expand brace pattern and return list of shard URLs (deprecated, use list_shards())."""
        return self.list_shards()

    @property
    def shards(self) -> Iterator[tuple[str, IO[bytes]]]:
        """Lazily yield (url, stream) pairs for each shard.

        Uses WebDataset's gopen to open URLs, which handles various schemes:
        - http/https: via curl
        - pipe: shell command streaming
        - gs: Google Cloud Storage via gsutil
        - hf: HuggingFace Hub
        - file or no scheme: local filesystem

        Yields:
            Tuple of (url, file-like stream).
        """
        for url in self.list_shards():
            stream = wds.gopen(url, mode="rb")
            yield url, stream

    def open_shard(self, shard_id: str) -> IO[bytes]:
        """Open a single shard by URL.

        Args:
            shard_id: URL of the shard to open.

        Returns:
            File-like stream from gopen.

        Raises:
            KeyError: If shard_id is not in list_shards().
        """
        if shard_id not in self.list_shards():
            raise KeyError(f"Shard not found: {shard_id}")
        return wds.gopen(shard_id, mode="rb")


@dataclass
class S3Source:
    """Data source for S3-compatible storage with explicit credentials.

    Uses boto3 to stream directly from S3, supporting:
    - Standard AWS S3
    - S3-compatible endpoints (Cloudflare R2, MinIO, etc.)
    - Private buckets with credentials
    - IAM role authentication (when keys not provided)

    Unlike URL-based approaches, this doesn't require URL transformation
    or global gopen_schemes registration. Credentials are scoped to the
    source instance.

    Attributes:
        bucket: S3 bucket name.
        keys: List of object keys (paths within bucket).
        endpoint: Optional custom endpoint URL for S3-compatible services.
        access_key: Optional AWS access key ID.
        secret_key: Optional AWS secret access key.
        region: Optional AWS region (defaults to us-east-1).

    Examples:
        >>> source = S3Source(
        ...     bucket="my-datasets",
        ...     keys=["train/shard-000.tar", "train/shard-001.tar"],
        ...     endpoint="https://abc123.r2.cloudflarestorage.com",
        ...     access_key="AKIAIOSFODNN7EXAMPLE",
        ...     secret_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        ... )
        >>> for shard_id, stream in source.shards:
        ...     process(stream)
    """

    bucket: str
    keys: list[str]
    endpoint: str | None = None
    access_key: str | None = None
    secret_key: str | None = None
    region: str | None = None
    _client: Any = field(default=None, repr=False, compare=False)

    def _get_client(self) -> Any:
        """Get or create boto3 S3 client."""
        if self._client is not None:
            return self._client

        import boto3

        client_kwargs: dict[str, Any] = {}

        if self.endpoint:
            client_kwargs["endpoint_url"] = self.endpoint

        if self.access_key and self.secret_key:
            client_kwargs["aws_access_key_id"] = self.access_key
            client_kwargs["aws_secret_access_key"] = self.secret_key

        if self.region:
            client_kwargs["region_name"] = self.region
        elif not self.endpoint:
            # Default region for AWS S3
            client_kwargs["region_name"] = os.environ.get(
                "AWS_DEFAULT_REGION", "us-east-1"
            )

        self._client = boto3.client("s3", **client_kwargs)
        return self._client

    def list_shards(self) -> list[str]:
        """Return list of S3 URIs for the shards."""
        return [f"s3://{self.bucket}/{key}" for key in self.keys]

    # Legacy alias for backwards compatibility
    @property
    def shard_list(self) -> list[str]:
        """Return list of S3 URIs for the shards (deprecated, use list_shards())."""
        return self.list_shards()

    @property
    def shards(self) -> Iterator[tuple[str, IO[bytes]]]:
        """Lazily yield (s3_uri, stream) pairs for each shard.

        Uses boto3 to get streaming response bodies, which are file-like
        objects that can be read directly by tarfile.

        Yields:
            Tuple of (s3://bucket/key URI, StreamingBody).
        """
        client = self._get_client()

        for key in self.keys:
            response = client.get_object(Bucket=self.bucket, Key=key)
            stream = response["Body"]
            uri = f"s3://{self.bucket}/{key}"
            yield uri, stream

    def open_shard(self, shard_id: str) -> IO[bytes]:
        """Open a single shard by S3 URI.

        Args:
            shard_id: S3 URI of the shard (s3://bucket/key).

        Returns:
            StreamingBody for reading the object.

        Raises:
            KeyError: If shard_id is not in list_shards().
        """
        if shard_id not in self.list_shards():
            raise KeyError(f"Shard not found: {shard_id}")

        # Parse s3://bucket/key -> key
        if not shard_id.startswith(f"s3://{self.bucket}/"):
            raise KeyError(f"Shard not in this bucket: {shard_id}")

        key = shard_id[len(f"s3://{self.bucket}/") :]
        client = self._get_client()
        response = client.get_object(Bucket=self.bucket, Key=key)
        return response["Body"]

    @classmethod
    def from_urls(
        cls,
        urls: list[str],
        *,
        endpoint: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        region: str | None = None,
    ) -> "S3Source":
        """Create S3Source from s3:// URLs.

        Parses s3://bucket/key URLs and extracts bucket and keys.
        All URLs must be in the same bucket.

        Args:
            urls: List of s3:// URLs.
            endpoint: Optional custom endpoint.
            access_key: Optional access key.
            secret_key: Optional secret key.
            region: Optional region.

        Returns:
            S3Source configured for the given URLs.

        Raises:
            ValueError: If URLs are not valid s3:// URLs or span multiple buckets.

        Examples:
            >>> source = S3Source.from_urls(
            ...     ["s3://my-bucket/train-000.tar", "s3://my-bucket/train-001.tar"],
            ...     endpoint="https://r2.example.com",
            ... )
        """
        if not urls:
            raise ValueError("urls cannot be empty")

        buckets: set[str] = set()
        keys: list[str] = []

        for url in urls:
            if not url.startswith("s3://"):
                raise ValueError(f"Not an S3 URL: {url}")

            # s3://bucket/path/to/key -> bucket, path/to/key
            path = url[5:]  # Remove 's3://'
            if "/" not in path:
                raise ValueError(f"Invalid S3 URL (no key): {url}")

            bucket, key = path.split("/", 1)
            buckets.add(bucket)
            keys.append(key)

        if len(buckets) > 1:
            raise ValueError(f"All URLs must be in the same bucket, got: {buckets}")

        return cls(
            bucket=buckets.pop(),
            keys=keys,
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            region=region,
        )

    @classmethod
    def from_credentials(
        cls,
        credentials: dict[str, str],
        bucket: str,
        keys: list[str],
    ) -> "S3Source":
        """Create S3Source from a credentials dictionary.

        Accepts the same credential format used by S3DataStore.

        Args:
            credentials: Dict with AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
                and optionally AWS_ENDPOINT.
            bucket: S3 bucket name.
            keys: List of object keys.

        Returns:
            Configured S3Source.

        Examples:
            >>> creds = {
            ...     "AWS_ACCESS_KEY_ID": "...",
            ...     "AWS_SECRET_ACCESS_KEY": "...",
            ...     "AWS_ENDPOINT": "https://r2.example.com",
            ... }
            >>> source = S3Source.from_credentials(creds, "my-bucket", ["data.tar"])
        """
        return cls(
            bucket=bucket,
            keys=keys,
            endpoint=credentials.get("AWS_ENDPOINT"),
            access_key=credentials.get("AWS_ACCESS_KEY_ID"),
            secret_key=credentials.get("AWS_SECRET_ACCESS_KEY"),
            region=credentials.get("AWS_REGION"),
        )


@dataclass
class BlobSource:
    """Data source for ATProto PDS blob storage.

    Streams dataset shards stored as blobs on an ATProto Personal Data Server.
    Each shard is identified by a blob reference containing the DID and CID.

    This source resolves blob references to HTTP URLs and streams the content
    directly, supporting efficient iteration over shards without downloading
    everything upfront.

    Attributes:
        blob_refs: List of blob reference dicts with 'did' and 'cid' keys.
        pds_endpoint: Optional PDS endpoint URL. If not provided, resolved from DID.

    Examples:
        >>> source = BlobSource(
        ...     blob_refs=[
        ...         {"did": "did:plc:abc123", "cid": "bafyrei..."},
        ...         {"did": "did:plc:abc123", "cid": "bafyrei..."},
        ...     ],
        ... )
        >>> for shard_id, stream in source.shards:
        ...     process(stream)
    """

    blob_refs: list[dict[str, str]]
    pds_endpoint: str | None = None
    _endpoint_cache: dict[str, str] = field(
        default_factory=dict, repr=False, compare=False
    )

    def _resolve_pds_endpoint(self, did: str) -> str:
        """Resolve PDS endpoint for a DID, with caching."""
        if did in self._endpoint_cache:
            return self._endpoint_cache[did]

        if self.pds_endpoint:
            self._endpoint_cache[did] = self.pds_endpoint
            return self.pds_endpoint

        import requests

        # Resolve via plc.directory
        if did.startswith("did:plc:"):
            plc_url = f"https://plc.directory/{did}"
            response = requests.get(plc_url, timeout=10)
            response.raise_for_status()
            doc = response.json()

            for service in doc.get("service", []):
                if service.get("type") == "AtprotoPersonalDataServer":
                    endpoint = service.get("serviceEndpoint", "")
                    self._endpoint_cache[did] = endpoint
                    return endpoint

        raise ValueError(f"Could not resolve PDS endpoint for {did}")

    def _get_blob_url(self, did: str, cid: str) -> str:
        """Get HTTP URL for fetching a blob."""
        endpoint = self._resolve_pds_endpoint(did)
        return f"{endpoint}/xrpc/com.atproto.sync.getBlob?did={did}&cid={cid}"

    def _make_shard_id(self, ref: dict[str, str]) -> str:
        """Create shard identifier from blob reference."""
        return f"at://{ref['did']}/blob/{ref['cid']}"

    def list_shards(self) -> list[str]:
        """Return list of AT URI-style shard identifiers."""
        return [self._make_shard_id(ref) for ref in self.blob_refs]

    @property
    def shards(self) -> Iterator[tuple[str, IO[bytes]]]:
        """Lazily yield (at_uri, stream) pairs for each shard.

        Fetches blobs via HTTP from the PDS and yields streaming responses.

        Yields:
            Tuple of (at://did/blob/cid URI, streaming response body).
        """
        import requests

        for ref in self.blob_refs:
            did = ref["did"]
            cid = ref["cid"]
            url = self._get_blob_url(did, cid)

            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            shard_id = self._make_shard_id(ref)
            # Wrap response in a file-like object
            yield shard_id, response.raw

    def open_shard(self, shard_id: str) -> IO[bytes]:
        """Open a single shard by its AT URI.

        Args:
            shard_id: AT URI of the shard (at://did/blob/cid).

        Returns:
            Streaming response body for reading the blob.

        Raises:
            KeyError: If shard_id is not in list_shards().
            ValueError: If shard_id format is invalid.
        """
        if shard_id not in self.list_shards():
            raise KeyError(f"Shard not found: {shard_id}")

        # Parse at://did/blob/cid
        if not shard_id.startswith("at://"):
            raise ValueError(f"Invalid shard ID format: {shard_id}")

        parts = shard_id[5:].split("/")  # Remove 'at://'
        if len(parts) != 3 or parts[1] != "blob":
            raise ValueError(f"Invalid blob URI format: {shard_id}")

        did, _, cid = parts
        url = self._get_blob_url(did, cid)

        import requests

        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        return response.raw

    @classmethod
    def from_refs(
        cls,
        refs: list[dict],
        *,
        pds_endpoint: str | None = None,
    ) -> "BlobSource":
        """Create BlobSource from blob reference dicts.

        Accepts blob references in the format returned by upload_blob:
        ``{"$type": "blob", "ref": {"$link": "cid"}, ...}``

        Also accepts simplified format: ``{"did": "...", "cid": "..."}``

        Args:
            refs: List of blob reference dicts.
            pds_endpoint: Optional PDS endpoint to use for all blobs.

        Returns:
            Configured BlobSource.

        Raises:
            ValueError: If refs is empty or format is invalid.
        """
        if not refs:
            raise ValueError("refs cannot be empty")

        blob_refs: list[dict[str, str]] = []

        for ref in refs:
            if "did" in ref and "cid" in ref:
                # Simple format
                blob_refs.append({"did": ref["did"], "cid": ref["cid"]})
            elif "ref" in ref and "$link" in ref.get("ref", {}):
                # ATProto blob format - need DID from elsewhere
                raise ValueError(
                    "ATProto blob format requires 'did' field. "
                    "Use from_record_storage() for records with storage.blobs."
                )
            else:
                raise ValueError(f"Invalid blob reference format: {ref}")

        return cls(blob_refs=blob_refs, pds_endpoint=pds_endpoint)


__all__ = [
    "URLSource",
    "S3Source",
    "BlobSource",
]
