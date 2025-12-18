"""TODO"""

##
# Imports

from atdata import (
    PackableSample,
    Dataset,
)

from pathlib import Path
from uuid import uuid4
from tempfile import TemporaryDirectory
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
    TypeVar
)

T = TypeVar( 'T', bound = PackableSample )


##
# Helpers

def _kind_str_for_sample_type( st: Type[PackableSample] ) -> str:
    """TODO"""
    return f'{st.__module__}.{st.__name__}'


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
                **kwargs
            ) -> Dataset[T]:
        """TODO"""
        
        assert self.hive_bucket is not None
        assert self.hive_path is not None

        with TemporaryDirectory() as tmpdir:

            # Mount S3 filesystem
            mount_path = Path( tmpdir ) / 'atdata-local' / self.hive_bucket
            mount_cmd = [
                's3fs',
                self.hive_bucket,
                mount_path.as_posix()
            ]
            subprocess.run( mount_cmd, env = self.s3_credentials )

            new_uuid = uuid4()

            # Write metadata
            metadata_path = (
                mount_path
                / 'metadata'
                / f'atdata-metadata--{new_uuid}.msgpack'
            )
            with open( metadata_path, 'wb' ) as f:
                if ds.metadata is not None:
                    # TODO Figure out how to make linting work better here
                    f.write( msgpack.packb( ds.metadata ) )

            # Write data
            shard_pattern = (mount_path / f'atdata--{new_uuid}--%06d.tar').as_posix()
            written_shards = []
            with wds.writer.ShardWriter( shard_pattern,
                post = lambda s: written_shards.append( s ),
                **kwargs
            ) as sink:
                for sample in ds.ordered( batch_size = None ):
                    sink.write( sample.as_wds )
        
        # Return created dataset
        shard_s3_format = (
            (
                self.hive_path
                / f'atdata--{new_uuid}'
            ).as_posix()
        ) + '--{shard_id}.tar'
        metadata_s3_path = (
            self.hive_path
            / 'metadata'
            / f'atdata-metadata--{new_uuid}.msgpack'
        )
        shard_id_braced = '{' + f'{0:06d}..{len( written_shards ) - 1:06d}' + '}'
        return Dataset(
            url = shard_s3_format.format( shard_id = shard_id_braced ),
            metadata_url = metadata_path.as_posix(),
        )


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

    def list( self ):
        """TODO"""
        ##
        ret = []
        for key in self._redis.scan_iter( match = 'BasicIndexEntry:*' ):
            ret.append( self._redis.hgetall( key ) )
        return ret

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