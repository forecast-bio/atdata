"""TODO"""

##
# Imports

from atdata import (
    PackableSample,
    Dataset,
)

from uuid import uuid4

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
)


##
# Heplers

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

    def save_to( self, redis: Redis ):
        """TODO"""
        save_key = f'BasicIndexEntry:{self.uuid}'
        # TODO figure out how to get linting to work correctly here
        redis.hset( save_key, mapping = asdict( self ) )
        

##
# Classes

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
            self._redis = Redis()

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

    def add( self, ds: Dataset ) -> BasicIndexEntry:
        """TODO"""
        ##
        temp_sample_kind = _kind_str_for_sample_type( ds.sample_type )

        ret_data = BasicIndexEntry(
            wds_url = ds.url,
            sample_kind = temp_sample_kind,
            metadata_url = ds.metadata_url,
        )
        ret_data.save_to( self._redis )

        return ret_data