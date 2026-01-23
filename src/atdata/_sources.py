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

Example:
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

    Example:
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

    Example:
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
            client_kwargs["region_name"] = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

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

        key = shard_id[len(f"s3://{self.bucket}/"):]
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

        Example:
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

        Example:
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


__all__ = [
    "URLSource",
    "S3Source",
]
