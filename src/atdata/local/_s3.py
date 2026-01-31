"""S3-compatible data store and helper functions."""

from atdata import Dataset

from pathlib import Path
from uuid import uuid4
from tempfile import TemporaryDirectory
from dotenv import dotenv_values
from typing import Any, BinaryIO, cast

from s3fs import S3FileSystem
import webdataset as wds


def _s3_env(credentials_path: str | Path) -> dict[str, Any]:
    """Load S3 credentials from .env file.

    Args:
        credentials_path: Path to .env file containing AWS_ENDPOINT,
            AWS_ACCESS_KEY_ID, and AWS_SECRET_ACCESS_KEY.

    Returns:
        Dict with the three required credential keys.

    Raises:
        ValueError: If any required key is missing from the .env file.
    """
    credentials_path = Path(credentials_path)
    env_values = dotenv_values(credentials_path)

    required_keys = ("AWS_ENDPOINT", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
    missing = [k for k in required_keys if k not in env_values]
    if missing:
        raise ValueError(
            f"Missing required keys in {credentials_path}: {', '.join(missing)}"
        )

    return {k: env_values[k] for k in required_keys}


def _s3_from_credentials(creds: str | Path | dict) -> S3FileSystem:
    """Create S3FileSystem from credentials dict or .env file path."""
    if not isinstance(creds, dict):
        creds = _s3_env(creds)

    # Build kwargs, making endpoint_url optional
    kwargs = {
        "key": creds["AWS_ACCESS_KEY_ID"],
        "secret": creds["AWS_SECRET_ACCESS_KEY"],
    }
    if "AWS_ENDPOINT" in creds:
        kwargs["endpoint_url"] = creds["AWS_ENDPOINT"]

    return S3FileSystem(**kwargs)


def _create_s3_write_callbacks(
    credentials: dict[str, Any],
    temp_dir: str,
    written_shards: list[str],
    fs: S3FileSystem | None,
    cache_local: bool,
    add_s3_prefix: bool = False,
) -> tuple:
    """Create opener and post callbacks for ShardWriter with S3 upload.

    Args:
        credentials: S3 credentials dict.
        temp_dir: Temporary directory for local caching.
        written_shards: List to append written shard paths to.
        fs: S3FileSystem for direct writes (used when cache_local=False).
        cache_local: If True, write locally then copy to S3.
        add_s3_prefix: If True, prepend 's3://' to shard paths.

    Returns:
        Tuple of (writer_opener, writer_post) callbacks.
    """
    if cache_local:
        import boto3

        s3_client_kwargs = {
            "aws_access_key_id": credentials["AWS_ACCESS_KEY_ID"],
            "aws_secret_access_key": credentials["AWS_SECRET_ACCESS_KEY"],
        }
        if "AWS_ENDPOINT" in credentials:
            s3_client_kwargs["endpoint_url"] = credentials["AWS_ENDPOINT"]
        s3_client = boto3.client("s3", **s3_client_kwargs)

        def _writer_opener(p: str):
            local_path = Path(temp_dir) / p
            local_path.parent.mkdir(parents=True, exist_ok=True)
            return open(local_path, "wb")

        def _writer_post(p: str):
            local_path = Path(temp_dir) / p
            path_parts = Path(p).parts
            bucket = path_parts[0]
            key = str(Path(*path_parts[1:]))

            with open(local_path, "rb") as f_in:
                s3_client.put_object(Bucket=bucket, Key=key, Body=f_in.read())

            local_path.unlink()
            if add_s3_prefix:
                written_shards.append(f"s3://{p}")
            else:
                written_shards.append(p)

        return _writer_opener, _writer_post
    else:
        if fs is None:
            raise ValueError("S3FileSystem required when cache_local=False")

        def _direct_opener(s: str):
            return cast(BinaryIO, fs.open(f"s3://{s}", "wb"))

        def _direct_post(s: str):
            if add_s3_prefix:
                written_shards.append(f"s3://{s}")
            else:
                written_shards.append(s)

        return _direct_opener, _direct_post


class S3DataStore:
    """S3-compatible data store implementing AbstractDataStore protocol.

    Handles writing dataset shards to S3-compatible object storage and
    resolving URLs for reading.

    Attributes:
        credentials: S3 credentials dictionary.
        bucket: Target bucket name.
        _fs: S3FileSystem instance.
    """

    def __init__(
        self,
        credentials: str | Path | dict[str, Any],
        *,
        bucket: str,
    ) -> None:
        """Initialize an S3 data store.

        Args:
            credentials: Path to .env file or dict with AWS_ACCESS_KEY_ID,
                AWS_SECRET_ACCESS_KEY, and optionally AWS_ENDPOINT.
            bucket: Name of the S3 bucket for storage.
        """
        if isinstance(credentials, dict):
            self.credentials = credentials
        else:
            self.credentials = _s3_env(credentials)

        self.bucket = bucket
        self._fs = _s3_from_credentials(self.credentials)

    def write_shards(
        self,
        ds: Dataset,
        *,
        prefix: str,
        cache_local: bool = False,
        manifest: bool = False,
        schema_version: str = "1.0.0",
        source_job_id: str | None = None,
        parent_shards: list[str] | None = None,
        pipeline_version: str | None = None,
        **kwargs,
    ) -> list[str]:
        """Write dataset shards to S3.

        Args:
            ds: The Dataset to write.
            prefix: Path prefix within bucket (e.g., 'datasets/mnist/v1').
            cache_local: If True, write locally first then copy to S3.
            manifest: If True, generate per-shard manifest files alongside
                each tar shard (``.manifest.json`` + ``.manifest.parquet``).
            schema_version: Schema version for manifest headers.
            source_job_id: Optional provenance job identifier for manifests.
            parent_shards: Optional list of input shard identifiers for provenance.
            pipeline_version: Optional pipeline version string for provenance.
            **kwargs: Additional args passed to wds.ShardWriter (e.g., maxcount).

        Returns:
            List of S3 URLs for the written shards.

        Raises:
            RuntimeError: If no shards were written.
        """
        new_uuid = str(uuid4())
        shard_pattern = f"{self.bucket}/{prefix}/data--{new_uuid}--%06d.tar"

        written_shards: list[str] = []

        # Manifest tracking state shared with the post callback
        manifest_builders: list = []
        current_builder: list = [None]  # mutable ref for closure
        shard_counter: list[int] = [0]

        if manifest:
            from atdata.manifest import ManifestBuilder, ManifestWriter

            def _make_builder(shard_idx: int) -> ManifestBuilder:
                shard_id = f"{self.bucket}/{prefix}/data--{new_uuid}--{shard_idx:06d}"
                return ManifestBuilder(
                    sample_type=ds.sample_type,
                    shard_id=shard_id,
                    schema_version=schema_version,
                    source_job_id=source_job_id,
                    parent_shards=parent_shards,
                    pipeline_version=pipeline_version,
                )

            current_builder[0] = _make_builder(0)

        with TemporaryDirectory() as temp_dir:
            writer_opener, writer_post_orig = _create_s3_write_callbacks(
                credentials=self.credentials,
                temp_dir=temp_dir,
                written_shards=written_shards,
                fs=self._fs,
                cache_local=cache_local,
                add_s3_prefix=True,
            )

            if manifest:

                def writer_post(p: str):
                    # Finalize the current manifest builder when a shard completes
                    builder = current_builder[0]
                    if builder is not None:
                        manifest_builders.append(builder)
                    # Advance to the next shard's builder
                    shard_counter[0] += 1
                    current_builder[0] = _make_builder(shard_counter[0])
                    # Call original post callback
                    writer_post_orig(p)
            else:
                writer_post = writer_post_orig

            offset = 0
            with wds.writer.ShardWriter(
                shard_pattern,
                opener=writer_opener,
                post=writer_post,
                **kwargs,
            ) as sink:
                for sample in ds.ordered(batch_size=None):
                    wds_dict = sample.as_wds
                    sink.write(wds_dict)

                    if manifest and current_builder[0] is not None:
                        packed_size = len(wds_dict.get("msgpack", b""))
                        current_builder[0].add_sample(
                            key=wds_dict["__key__"],
                            offset=offset,
                            size=packed_size,
                            sample=sample,
                        )
                        # Approximate tar entry: 512-byte header + data rounded to 512
                        offset += 512 + packed_size + (512 - packed_size % 512) % 512

            # Finalize the last shard's builder (post isn't called for the last shard
            # until ShardWriter closes, but we handle it here for safety)
            if manifest and current_builder[0] is not None:
                builder = current_builder[0]
                if builder._rows:  # Only if samples were added
                    manifest_builders.append(builder)

            # Write all manifest files
            if manifest:
                for builder in manifest_builders:
                    built = builder.build()
                    writer = ManifestWriter(Path(temp_dir) / Path(built.shard_id))
                    json_path, parquet_path = writer.write(built)

                    # Upload manifest files to S3 alongside shards
                    shard_id = built.shard_id
                    json_key = f"{shard_id}.manifest.json"
                    parquet_key = f"{shard_id}.manifest.parquet"

                    if cache_local:
                        import boto3

                        s3_kwargs = {
                            "aws_access_key_id": self.credentials["AWS_ACCESS_KEY_ID"],
                            "aws_secret_access_key": self.credentials[
                                "AWS_SECRET_ACCESS_KEY"
                            ],
                        }
                        if "AWS_ENDPOINT" in self.credentials:
                            s3_kwargs["endpoint_url"] = self.credentials["AWS_ENDPOINT"]
                        s3_client = boto3.client("s3", **s3_kwargs)

                        bucket_name = Path(shard_id).parts[0]
                        json_s3_key = str(Path(*Path(json_key).parts[1:]))
                        parquet_s3_key = str(Path(*Path(parquet_key).parts[1:]))

                        with open(json_path, "rb") as f:
                            s3_client.put_object(
                                Bucket=bucket_name, Key=json_s3_key, Body=f.read()
                            )
                        with open(parquet_path, "rb") as f:
                            s3_client.put_object(
                                Bucket=bucket_name, Key=parquet_s3_key, Body=f.read()
                            )
                    else:
                        self._fs.put(str(json_path), f"s3://{json_key}")
                        self._fs.put(str(parquet_path), f"s3://{parquet_key}")

        if len(written_shards) == 0:
            raise RuntimeError("No shards written")

        return written_shards

    def read_url(self, url: str) -> str:
        """Resolve an S3 URL for reading/streaming.

        For S3-compatible stores with custom endpoints (like Cloudflare R2,
        MinIO, etc.), converts s3:// URLs to HTTPS URLs that WebDataset can
        stream directly.

        For standard AWS S3 (no custom endpoint), URLs are returned unchanged
        since WebDataset's built-in s3fs integration handles them.

        Args:
            url: S3 URL to resolve (e.g., 's3://bucket/path/file.tar').

        Returns:
            HTTPS URL if custom endpoint is configured, otherwise unchanged.
            Example: 's3://bucket/path' -> 'https://endpoint.com/bucket/path'
        """
        endpoint = self.credentials.get("AWS_ENDPOINT")
        if endpoint and url.startswith("s3://"):
            # s3://bucket/path -> https://endpoint/bucket/path
            path = url[5:]  # Remove 's3://' prefix
            endpoint = endpoint.rstrip("/")
            return f"{endpoint}/{path}"
        return url

    def supports_streaming(self) -> bool:
        """S3 supports streaming reads.

        Returns:
            True.
        """
        return True
