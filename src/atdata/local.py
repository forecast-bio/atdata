"""TODO"""

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
import shutil
import subprocess
from dotenv import dotenv_values
import msgpack

# from redis_om import (
#     EmbeddedJsonModel,
#     JsonModel,
#     Field,
#     Migrator,
#     get_redis_connection,
# )
from redis import (
    Redis,
)

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
)

T = TypeVar( 'T', bound = PackableSample )


##
# Helpers

def _kind_str_for_sample_type( st: Type[PackableSample] ) -> str:
    """TODO"""
    return f'{st.__module__}.{st.__name__}'

def _decode_bytes_dict( d: dict[bytes, bytes] ) -> dict[str, str]:
    """TODO"""
    return {
        k.decode('utf-8'): v.decode('utf-8')
        for k, v in d.items()
    }


##
# Redis object model

@dataclass
class BasicIndexEntry:
    """TODO"""
    ##

    wds_url: str
    """TODO"""
    sample_kind: str
    """TODO"""

    metadata_url: str | None
    """TODO"""

    uuid: str | None = field( default_factory = lambda: str( uuid4() ) )
    """TODO"""

    def write_to( self, redis: Redis ):
        """TODO"""
        save_key = f'BasicIndexEntry:{self.uuid}'
        # TODO figure out how to get linting to work correctly here
        redis.hset( save_key, mapping = asdict( self ) )

def _s3_env( credentials_path: str | Path ) -> dict[str, Any]:
    """TODO"""
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
    """TODO"""
    ##
    if not isinstance( creds, dict ):
        creds = _s3_env( creds )
    
    return S3FileSystem(
        endpoint_url = creds['AWS_ENDPOINT'],
        key = creds['AWS_ACCESS_KEY_ID'],
        secret = creds['AWS_SECRET_ACCESS_KEY']
    )


##
# Classes

class Repo:
    """TODO"""

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
        """TODO"""

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
        """TODO"""
        
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
        metadata_path.parent.mkdir( parents = True, exist_ok = True )

        if ds.metadata is not None:
            with hive_fs.open( metadata_path, 'wb' ) as f:
                # TODO Figure out how to make linting work better here
                f.write( msgpack.packb( ds.metadata ) )


        # Write data
        shard_pattern = (
            self.hive_path
            / f'atdata--{new_uuid}--%06d.tar'
        ).as_posix()

        with TemporaryDirectory() as temp_dir:

            if cache_local:
                def _writer_opener( p: str ):
                    local_cache_path = Path( temp_dir ) / p
                    local_cache_path.parent.mkdir( parents = True, exist_ok = True )
                    return open( local_cache_path, 'wb' )
                writer_opener = _writer_opener

                def _writer_post( p: str ):
                    local_cache_path = Path( temp_dir ) / p

                    # Copy to S3
                    print( 'Copying file to s3 ...', end = '' )
                    with open( local_cache_path, 'rb' ) as f_in:
                        with hive_fs.open( p, 'wb' ) as f_out:
                            # TODO Linting issues
                            f_out.write( f_in.read() )
                    print( ' done.' )

                    # Delete local cache file
                    print( 'Deleting local cache file ...', end = '' )
                    os.remove( local_cache_path )
                    print( ' done.' )

                    written_shards.append( s )
                writer_post = _writer_post

            else:
                writer_opener = lambda s: hive_fs.open( s, 'wb' )
                writer_post = lambda s: written_shards.append( s )

            written_shards = []
            with wds.writer.ShardWriter( shard_pattern,
                # opener = lambda s: hive_fs.open( s, 'wb' ),
                # post = lambda s: written_shards.append( s ),
                opener = writer_opener,
                post = writer_post,
                **kwargs
            ) as sink:
                for sample in ds.ordered( batch_size = None ):
                    sink.write( sample.as_wds )

        # with TemporaryDirectory() as tmpdir:

        #     # Mount S3 filesystem
        #     mount_path = Path( tmpdir ) / 'atdata-s3' / self.hive_bucket
        #     mount_path.mkdir( parents = True, exist_ok = True )
        #     s3fs_cmd = shutil.which( 's3fs' )
        #     mount_cmd = [
        #         s3fs_cmd,
        #         self.hive_bucket,
        #         mount_path.as_posix()
        #     ]
        #     result = subprocess.run( mount_cmd, env = self.s3_credentials )
        #     print( result )

        #     new_uuid = str( uuid4() )

        #     # Write metadata
        #     metadata_path = (
        #         mount_path
        #         / 'metadata'
        #         / f'atdata-metadata--{new_uuid}.msgpack'
        #     )
        #     metadata_path.parent.mkdir( parents = True, exist_ok = True )
        #     with open( metadata_path, 'wb' ) as f:
        #         if ds.metadata is not None:
        #             # TODO Figure out how to make linting work better here
        #             f.write( msgpack.packb( ds.metadata ) )

        #     # Write data
        #     shard_pattern = (Path( tmpdir ) / 'atdata-cache' / f'atdata--{new_uuid}--%06d.tar').as_posix()
        #     written_shards = []
        #     with wds.writer.ShardWriter( shard_pattern,
        #         opener = lambda s: 
        #         post = lambda s: written_shards.append( s ),
        #         **kwargs
        #     ) as sink:
        #         for sample in ds.ordered( batch_size = None ):
        #             sink.write( sample.as_wds )

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
    """TODO"""

    ##

    def __init__( self,
                redis: Redis | None = None,
                **kwargs
            ) -> None:
        """TODO"""
        ##

        if redis is not None:
            self._redis = redis
        else:
            self._redis = Redis( **kwargs )

        # needed before we can do anything with `redis`
        # TODO this only works / is necessary for `redis_om``
        # Migrator().run()

    @property
    def all_entries( self ) -> list[BasicIndexEntry]:
        """TODO"""
        return list( self.entries )

    @property
    def entries( self ) -> Generator[BasicIndexEntry, None, None]:
        """TODO"""
        ##
        for key in self._redis.scan_iter( match = 'BasicIndexEntry:*' ):
            # TODO typing issue for `redis`
            cur_entry_data = _decode_bytes_dict( self._redis.hgetall( key ) )
            cur_entry = BasicIndexEntry( **cur_entry_data )
            yield cur_entry
        
        return

    def add_entry( self, ds: Dataset,
                uuid: str | None = None,
            ) -> BasicIndexEntry:
        """TODO"""
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