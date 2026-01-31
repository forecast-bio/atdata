"""Deprecated Repo class for legacy S3 repository operations."""

from atdata import Dataset

from atdata.local._entry import LocalDatasetEntry
from atdata.local._s3 import _s3_env, _s3_from_credentials, _create_s3_write_callbacks

from pathlib import Path
from uuid import uuid4
from tempfile import TemporaryDirectory
from typing import Any, BinaryIO, TypeVar, cast

from redis import Redis
import msgpack
import webdataset as wds
import warnings

from atdata._protocols import Packable

T = TypeVar("T", bound=Packable)


class Repo:
    """Repository for storing and managing atdata datasets.

    .. deprecated::
        Use :class:`Index` with :class:`S3DataStore` instead::

            store = S3DataStore(credentials, bucket="my-bucket")
            index = Index(redis=redis, data_store=store)
            entry = index.insert_dataset(ds, name="my-dataset")

    Provides storage of datasets in S3-compatible object storage with Redis-based
    indexing. Datasets are stored as WebDataset tar files with optional metadata.

    Attributes:
        s3_credentials: S3 credentials dictionary or None.
        bucket_fs: S3FileSystem instance or None.
        hive_path: Path within S3 bucket for storing datasets.
        hive_bucket: Name of the S3 bucket.
        index: Index instance for tracking datasets.
    """

    ##

    def __init__(
        self,
        s3_credentials: str | Path | dict[str, Any] | None = None,
        hive_path: str | Path | None = None,
        redis: Redis | None = None,
    ) -> None:
        """Initialize a repository.

        .. deprecated::
            Use Index with S3DataStore instead.

        Args:
            s3_credentials: Path to .env file with S3 credentials, or dict with
                AWS_ENDPOINT, AWS_ACCESS_KEY_ID, and AWS_SECRET_ACCESS_KEY.
                If None, S3 functionality will be disabled.
            hive_path: Path within the S3 bucket to store datasets.
                Required if s3_credentials is provided.
            redis: Redis connection for indexing. If None, creates a new connection.

        Raises:
            ValueError: If hive_path is not provided when s3_credentials is set.
        """
        warnings.warn(
            "Repo is deprecated. Use Index with S3DataStore instead:\n"
            "  store = S3DataStore(credentials, bucket='my-bucket')\n"
            "  index = Index(redis=redis, data_store=store)\n"
            "  entry = index.insert_dataset(ds, name='my-dataset')",
            DeprecationWarning,
            stacklevel=2,
        )

        if s3_credentials is None:
            self.s3_credentials = None
        elif isinstance(s3_credentials, dict):
            self.s3_credentials = s3_credentials
        else:
            self.s3_credentials = _s3_env(s3_credentials)

        if self.s3_credentials is None:
            self.bucket_fs = None
        else:
            self.bucket_fs = _s3_from_credentials(self.s3_credentials)

        if self.bucket_fs is not None:
            if hive_path is None:
                raise ValueError("Must specify hive path within bucket")
            self.hive_path = Path(hive_path)
            self.hive_bucket = self.hive_path.parts[0]
        else:
            self.hive_path = None
            self.hive_bucket = None

        #

        from atdata.local._index import Index

        self.index = Index(redis=redis)

    ##

    def insert(
        self,
        ds: Dataset[T],
        *,
        name: str,
        cache_local: bool = False,
        schema_ref: str | None = None,
        **kwargs,
    ) -> tuple[LocalDatasetEntry, Dataset[T]]:
        """Insert a dataset into the repository.

        Writes the dataset to S3 as WebDataset tar files, stores metadata,
        and creates an index entry in Redis.

        Args:
            ds: The dataset to insert.
            name: Human-readable name for the dataset.
            cache_local: If True, write to local temporary storage first, then
                copy to S3. This can be faster for some workloads.
            schema_ref: Optional schema reference. If None, generates from sample type.
            **kwargs: Additional arguments passed to wds.ShardWriter.

        Returns:
            A tuple of (index_entry, new_dataset) where:
                - index_entry: LocalDatasetEntry for the stored dataset
                - new_dataset: Dataset object pointing to the stored copy

        Raises:
            ValueError: If S3 credentials or hive_path are not configured.
            RuntimeError: If no shards were written.
        """
        if self.s3_credentials is None:
            raise ValueError(
                "S3 credentials required for insert(). Initialize Repo with s3_credentials."
            )
        if self.hive_bucket is None or self.hive_path is None:
            raise ValueError(
                "hive_path required for insert(). Initialize Repo with hive_path."
            )

        new_uuid = str(uuid4())

        hive_fs = _s3_from_credentials(self.s3_credentials)

        # Write metadata
        metadata_path = (
            self.hive_path / "metadata" / f"atdata-metadata--{new_uuid}.msgpack"
        )
        # Note: S3 doesn't need directories created beforehand - s3fs handles this

        if ds.metadata is not None:
            # Use s3:// prefix to ensure s3fs treats this as an S3 path
            with cast(
                BinaryIO, hive_fs.open(f"s3://{metadata_path.as_posix()}", "wb")
            ) as f:
                meta_packed = msgpack.packb(ds.metadata)
                f.write(cast(bytes, meta_packed))

        # Write data
        shard_pattern = (self.hive_path / f"atdata--{new_uuid}--%06d.tar").as_posix()

        written_shards: list[str] = []
        with TemporaryDirectory() as temp_dir:
            writer_opener, writer_post = _create_s3_write_callbacks(
                credentials=self.s3_credentials,
                temp_dir=temp_dir,
                written_shards=written_shards,
                fs=hive_fs,
                cache_local=cache_local,
                add_s3_prefix=False,
            )

            with wds.writer.ShardWriter(
                shard_pattern,
                opener=writer_opener,
                post=writer_post,
                **kwargs,
            ) as sink:
                for sample in ds.ordered(batch_size=None):
                    sink.write(sample.as_wds)

        # Make a new Dataset object for the written dataset copy
        if len(written_shards) == 0:
            raise RuntimeError(
                "Cannot form new dataset entry -- did not write any shards"
            )

        elif len(written_shards) < 2:
            new_dataset_url = (
                self.hive_path / (Path(written_shards[0]).name)
            ).as_posix()

        else:
            shard_s3_format = (
                (self.hive_path / f"atdata--{new_uuid}").as_posix()
            ) + "--{shard_id}.tar"
            shard_id_braced = "{" + f"{0:06d}..{len(written_shards) - 1:06d}" + "}"
            new_dataset_url = shard_s3_format.format(shard_id=shard_id_braced)

        new_dataset = Dataset[ds.sample_type](
            url=new_dataset_url,
            metadata_url=metadata_path.as_posix(),
        )

        # Add to index (use ds._metadata to avoid network requests)
        new_entry = self.index.add_entry(
            new_dataset,
            name=name,
            schema_ref=schema_ref,
            metadata=ds._metadata,
        )

        return new_entry, new_dataset
