"""Local repository storage for atdata datasets.

This module provides a local storage backend for atdata datasets using:
- S3-compatible object storage for dataset tar files and metadata
- Redis for indexing and tracking datasets

The main classes are:
- Repo: Manages dataset storage in S3 with Redis indexing
- Index: Redis-backed index for tracking dataset metadata
- BasicIndexEntry: Index entry representing a stored dataset

This is intended for development and small-scale deployment before
migrating to the full atproto PDS infrastructure.
"""

##
# Imports

from atdata import (
    PackableSample,
    Dataset,
)

import os
from pathlib import Path
from uuid import uuid4
from tempfile import TemporaryDirectory
from dotenv import dotenv_values
import msgpack

from redis import Redis

from s3fs import (
    S3FileSystem,
)

import webdataset as wds

from dataclasses import (
    dataclass,
    asdict,
    field,
)
from typing import (
    Any,
    Optional,
    Dict,
    Type,
    TypeVar,
    Generator,
    BinaryIO,
    cast,
)

T = TypeVar( 'T', bound = PackableSample )


##
# Helpers

def _kind_str_for_sample_type( st: Type[PackableSample] ) -> str:
    """Convert a sample type to a fully-qualified string identifier.

    Args:
        st: The sample type class.

    Returns:
        A string in the format 'module.name' identifying the sample type.
    """
    return f'{st.__module__}.{st.__name__}'

def _decode_bytes_dict( d: dict[bytes, bytes] ) -> dict[str, str]:
    """Decode a dictionary with byte keys and values to strings.

    Redis returns dictionaries with bytes keys/values, this converts them to strings.

    Args:
        d: Dictionary with bytes keys and values.

    Returns:
        Dictionary with UTF-8 decoded string keys and values.
    """
    return {
        k.decode('utf-8'): v.decode('utf-8')
        for k, v in d.items()
    }


##
# Redis object model

@dataclass
class BasicIndexEntry:
    """Index entry for a dataset stored in the repository.

    Tracks metadata about a dataset stored in S3, including its location,
    type, and unique identifier.
    """
    ##

    wds_url: str
    """WebDataset URL for the dataset tar files, for use with atdata.Dataset."""

    sample_kind: str
    """Fully-qualified sample type name (e.g., 'module.ClassName')."""

    metadata_url: str | None
    """S3 URL to the dataset's metadata msgpack file, if any."""

    uuid: str = field( default_factory = lambda: str( uuid4() ) )
    """Unique identifier for this dataset entry. Defaults to a new UUID if not provided."""

    def write_to( self, redis: Redis ):
        """Persist this index entry to Redis.

        Stores the entry as a Redis hash with key 'BasicIndexEntry:{uuid}'.

        Args:
            redis: Redis connection to write to.
        """
        save_key = f'BasicIndexEntry:{self.uuid}'
        # Filter out None values - Redis doesn't accept None
        data = {k: v for k, v in asdict(self).items() if v is not None}
        # redis-py typing uses untyped dict, so type checker complains about dict[str, Any]
        redis.hset( save_key, mapping = data )  # type: ignore[arg-type]

def _s3_env( credentials_path: str | Path ) -> dict[str, Any]:
    """Load S3 credentials from a .env file.

    Args:
        credentials_path: Path to .env file containing S3 credentials.

    Returns:
        Dictionary with AWS_ENDPOINT, AWS_ACCESS_KEY_ID, and AWS_SECRET_ACCESS_KEY.

    Raises:
        AssertionError: If required credentials are missing from the file.
    """
    ##
    credentials_path = Path( credentials_path )
    env_values = dotenv_values( credentials_path )
    assert 'AWS_ENDPOINT' in env_values
    assert 'AWS_ACCESS_KEY_ID' in env_values
    assert 'AWS_SECRET_ACCESS_KEY' in env_values

    return {
        k: env_values[k]
        for k in (
            'AWS_ENDPOINT',
            'AWS_ACCESS_KEY_ID',
            'AWS_SECRET_ACCESS_KEY',
        )
    }

def _s3_from_credentials( creds: str | Path | dict ) -> S3FileSystem:
    """Create an S3FileSystem from credentials.

    Args:
        creds: Either a path to a .env file with credentials, or a dict
            containing AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and optionally
            AWS_ENDPOINT.

    Returns:
        Configured S3FileSystem instance.
    """
    ##
    if not isinstance( creds, dict ):
        creds = _s3_env( creds )

    # Build kwargs, making endpoint_url optional
    kwargs = {
        'key': creds['AWS_ACCESS_KEY_ID'],
        'secret': creds['AWS_SECRET_ACCESS_KEY']
    }
    if 'AWS_ENDPOINT' in creds:
        kwargs['endpoint_url'] = creds['AWS_ENDPOINT']

    return S3FileSystem(**kwargs)


##
# Classes

class Repo:
    """Repository for storing and managing atdata datasets.

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

    def __init__( self,
                #
                s3_credentials: str | Path | dict[str, Any] | None = None,
                hive_path: str | Path | None = None,
                redis: Redis | None = None,
                #
                #
                **kwargs
            ) -> None:
        """Initialize a repository.

        Args:
            s3_credentials: Path to .env file with S3 credentials, or dict with
                AWS_ENDPOINT, AWS_ACCESS_KEY_ID, and AWS_SECRET_ACCESS_KEY.
                If None, S3 functionality will be disabled.
            hive_path: Path within the S3 bucket to store datasets.
                Required if s3_credentials is provided.
            redis: Redis connection for indexing. If None, creates a new connection.
            **kwargs: Additional arguments (reserved for future use).

        Raises:
            ValueError: If hive_path is not provided when s3_credentials is set.
        """

        if s3_credentials is None:
            self.s3_credentials = None
        elif isinstance( s3_credentials, dict ):
            self.s3_credentials = s3_credentials
        else:
            self.s3_credentials = _s3_env( s3_credentials )

        if self.s3_credentials is None:
            self.bucket_fs = None
        else:
            self.bucket_fs = _s3_from_credentials( self.s3_credentials )

        if self.bucket_fs is not None:
            if hive_path is None:
                raise ValueError( 'Must specify hive path within bucket' )
            self.hive_path = Path( hive_path )
            self.hive_bucket = self.hive_path.parts[0]
        else:
            self.hive_path = None
            self.hive_bucket = None
        
        #

        self.index = Index( redis = redis )

    ##

    def insert( self, ds: Dataset[T],
               #
               cache_local: bool = False,
               #
                **kwargs
            ) -> tuple[BasicIndexEntry, Dataset[T]]:
        """Insert a dataset into the repository.

        Writes the dataset to S3 as WebDataset tar files, stores metadata,
        and creates an index entry in Redis.

        Args:
            ds: The dataset to insert.
            cache_local: If True, write to local temporary storage first, then
                copy to S3. This can be faster for some workloads.
            **kwargs: Additional arguments passed to wds.ShardWriter.

        Returns:
            A tuple of (index_entry, new_dataset) where:
                - index_entry: BasicIndexEntry for the stored dataset
                - new_dataset: Dataset object pointing to the stored copy

        Raises:
            AssertionError: If S3 credentials or hive_path are not configured.
            RuntimeError: If no shards were written.
        """
        
        assert self.s3_credentials is not None
        assert self.hive_bucket is not None
        assert self.hive_path is not None

        new_uuid = str( uuid4() )

        hive_fs = _s3_from_credentials( self.s3_credentials )

        # Write metadata
        metadata_path = (
            self.hive_path
            / 'metadata'
            / f'atdata-metadata--{new_uuid}.msgpack'
        )
        # Note: S3 doesn't need directories created beforehand - s3fs handles this

        if ds.metadata is not None:
            # Use s3:// prefix to ensure s3fs treats this as an S3 path
            with cast( BinaryIO, hive_fs.open( f's3://{metadata_path.as_posix()}', 'wb' ) ) as f:
                meta_packed = msgpack.packb( ds.metadata )
                assert meta_packed is not None
                f.write( cast( bytes, meta_packed ) )


        # Write data
        shard_pattern = (
            self.hive_path
            / f'atdata--{new_uuid}--%06d.tar'
        ).as_posix()

        with TemporaryDirectory() as temp_dir:

            if cache_local:
                # For cache_local, we need to use boto3 directly to avoid s3fs async issues with moto
                import boto3

                # Create boto3 client from credentials
                s3_client_kwargs = {
                    'aws_access_key_id': self.s3_credentials['AWS_ACCESS_KEY_ID'],
                    'aws_secret_access_key': self.s3_credentials['AWS_SECRET_ACCESS_KEY']
                }
                if 'AWS_ENDPOINT' in self.s3_credentials:
                    s3_client_kwargs['endpoint_url'] = self.s3_credentials['AWS_ENDPOINT']
                s3_client = boto3.client('s3', **s3_client_kwargs)

                def _writer_opener( p: str ):
                    local_cache_path = Path( temp_dir ) / p
                    local_cache_path.parent.mkdir( parents = True, exist_ok = True )
                    return open( local_cache_path, 'wb' )
                writer_opener = _writer_opener

                def _writer_post( p: str ):
                    local_cache_path = Path( temp_dir ) / p

                    # Copy to S3 using boto3 client (avoids s3fs async issues)
                    path_parts = Path( p ).parts
                    bucket = path_parts[0]
                    key = str( Path( *path_parts[1:] ) )

                    with open( local_cache_path, 'rb' ) as f_in:
                        s3_client.put_object( Bucket=bucket, Key=key, Body=f_in.read() )

                    # Delete local cache file
                    local_cache_path.unlink()

                    written_shards.append( p )
                writer_post = _writer_post

            else:
                # Use s3:// prefix to ensure s3fs treats paths as S3 paths
                writer_opener = lambda s: cast( BinaryIO, hive_fs.open( f's3://{s}', 'wb' ) )
                writer_post = lambda s: written_shards.append( s )

            written_shards = []
            with wds.writer.ShardWriter(
                shard_pattern,
                opener = writer_opener,
                post = writer_post,
                **kwargs,
            ) as sink:
                for sample in ds.ordered( batch_size = None ):
                    sink.write( sample.as_wds )

        # Make a new Dataset object for the written dataset copy
        if len( written_shards ) == 0:
            raise RuntimeError( 'Cannot form new dataset entry -- did not write any shards' )
        
        elif len( written_shards ) < 2:
            new_dataset_url = (
                self.hive_path
                / ( Path( written_shards[0] ).name )
            ).as_posix()

        else:
            shard_s3_format = (
                (
                    self.hive_path
                    / f'atdata--{new_uuid}'
                ).as_posix()
            ) + '--{shard_id}.tar'
            shard_id_braced = '{' + f'{0:06d}..{len( written_shards ) - 1:06d}' + '}'
            new_dataset_url = shard_s3_format.format( shard_id = shard_id_braced )

        new_dataset = Dataset[ds.sample_type](
            url = new_dataset_url,
            metadata_url = metadata_path.as_posix(),
        )

        # Add to index
        new_entry = self.index.add_entry( new_dataset, uuid = new_uuid )

        return new_entry, new_dataset


class Index:
    """Redis-backed index for tracking datasets in a repository.

    Maintains a registry of BasicIndexEntry objects in Redis, allowing
    enumeration and lookup of stored datasets.

    Attributes:
        _redis: Redis connection for index storage.
    """

    ##

    def __init__( self,
                redis: Redis | None = None,
                **kwargs
            ) -> None:
        """Initialize an index.

        Args:
            redis: Redis connection to use. If None, creates a new connection
                using the provided kwargs.
            **kwargs: Additional arguments passed to Redis() constructor if
                redis is None.
        """
        ##

        if redis is not None:
            self._redis = redis
        else:
            self._redis: Redis = Redis( **kwargs )

    @property
    def all_entries( self ) -> list[BasicIndexEntry]:
        """Get all index entries as a list.

        Returns:
            List of all BasicIndexEntry objects in the index.
        """
        return list( self.entries )

    @property
    def entries( self ) -> Generator[BasicIndexEntry, None, None]:
        """Iterate over all index entries.

        Scans Redis for all BasicIndexEntry keys and yields them one at a time.

        Yields:
            BasicIndexEntry objects from the index.
        """
        ##
        for key in self._redis.scan_iter( match = 'BasicIndexEntry:*' ):
            # hgetall returns dict[bytes, bytes] which we decode to dict[str, str]
            cur_entry_data = _decode_bytes_dict( cast(dict[bytes, bytes], self._redis.hgetall( key )) )
            
            # Provide default None for optional fields that may be missing
            # Type checker complains about None in dict[str, str], but BasicIndexEntry accepts it
            cur_entry_data: dict[str, Any] = dict( **cur_entry_data )
            cur_entry_data.setdefault('metadata_url', None)
            
            cur_entry = BasicIndexEntry( **cur_entry_data )
            yield cur_entry

        return

    def add_entry( self, ds: Dataset,
                uuid: str | None = None,
            ) -> BasicIndexEntry:
        """Add a dataset to the index.

        Creates a BasicIndexEntry for the dataset and persists it to Redis.

        Args:
            ds: The dataset to add to the index.
            uuid: Optional UUID for the entry. If None, a new UUID is generated.

        Returns:
            The created BasicIndexEntry object.
        """
        ##
        temp_sample_kind = _kind_str_for_sample_type( ds.sample_type )

        if uuid is None:
            ret_data = BasicIndexEntry(
                wds_url = ds.url,
                sample_kind = temp_sample_kind,
                metadata_url = ds.metadata_url,
            )
        else:
            ret_data = BasicIndexEntry(
                wds_url = ds.url,
                sample_kind = temp_sample_kind,
                metadata_url = ds.metadata_url,
                uuid = uuid,
            )

        ret_data.write_to( self._redis )

        return ret_data


#