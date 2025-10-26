"""Schematized WebDatasets"""

##
# Imports

import webdataset as wds

from dataclasses import dataclass
import uuid

import numpy as np

from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    Any,
    Optional,
    Dict,
    Sequence,
    #
    Self,
    Generic,
    Type,
    TypeVar,
    TypeAlias,
)
# from typing_inspect import get_bound, get_parameters
from numpy.typing import (
    NDArray,
    ArrayLike,
)

#

# import ekumen.atmosphere as eat

import msgpack
import ormsgpack
from . import _helpers as eh


##
# Typing help

WDSRawSample: TypeAlias = Dict[str, Any]
WDSRawBatch: TypeAlias = Dict[str, Any]


##
# Main base classes

# TODO Check for best way to ensure this typevar is used as a dataclass type
# DT = TypeVar( 'DT', bound = dataclass.__class__ )
DT = TypeVar( 'DT' )

MsgpackRawSample: TypeAlias = Dict[str, Any]

# @dataclass
# class ArrayBytes:
#     """Annotates bytes that should be interpreted as the raw contents of a
#     numpy NDArray"""
    
#     raw_bytes: bytes
#     """The raw bytes of the corresponding NDArray"""

#     def __init__( self,
#             array: Optional[ArrayLike] = None,  
#             raw: Optional[bytes] = None,
#         ):
#         """TODO"""

#         if array is not None:
#             array = np.array( array )
#             self.raw_bytes = eh.array_to_bytes( array )
        
#         elif raw is not None:
#             self.raw_bytes = raw
        
#         else:
#             raise ValueError( 'Must provide either `array` or `raw` bytes' )

#     @property
#     def to_numpy( self ) -> NDArray:
#         """Return the `raw_bytes` data as an NDArray"""
#         return eh.bytes_to_array( self.raw_bytes )

def _make_packable( x ):
    # if isinstance( x, ArrayBytes ):
    #     return x.raw_bytes
    if isinstance( x, np.ndarray ):
        return eh.array_to_bytes( x )
    return x

class PackableSample( ABC ):
    """A sample that can be packed and unpacked with msgpack"""

    def __post_init__( self ):

        # Auto-convert known types when annotated
        for var_name, var_type in vars( self.__class__ )['__annotations__'].items():

            # Annotation for this variable is to be an NDArray
            if var_type == NDArray:
                # ... so, we'll always auto-convert to numpy

                var_cur_value = getattr( self, var_name )

                # Execute the appropriate conversion for intermediate data
                # based on what is provided

                if isinstance( var_cur_value, np.ndarray ):
                    # we're good!
                    pass

                # elif isinstance( var_cur_value, ArrayBytes ):
                #     setattr( self, var_name, var_cur_value.to_numpy )

                elif isinstance( var_cur_value, bytes ):
                    setattr( self, var_name, eh.bytes_to_array( var_cur_value ) )

    ##

    @classmethod
    def from_data( cls, data: MsgpackRawSample ) -> Self:
        """Create a sample instance from unpacked msgpack data"""
        return cls( **data )
    
    @classmethod
    def from_bytes( cls, bs: bytes ) -> Self:
        """Create a sample instance from raw msgpack bytes"""
        return cls.from_data( ormsgpack.unpackb( bs ) )

    @property
    def packed( self ) -> bytes:
        """Pack this sample's data into msgpack bytes"""

        # Make sure that all of our (possibly unpackable) data is in a packable
        # format
        o = {
            k: _make_packable( v )
            for k, v in vars( self ).items()
        }

        ret = msgpack.packb( o )

        if ret is None:
            raise RuntimeError( f'Failed to pack sample to bytes: {o}' )

        return ret
    
    # TODO Expand to allow for specifying explicit __key__
    @property
    def as_wds( self ) -> WDSRawSample:
        """Pack this sample's data for writing to webdataset"""
        return {
            # Generates a UUID that is timelike-sortable
            '__key__': str( uuid.uuid1( 0, 0 ) ),
            'msgpack': self.packed,
        }

def _batch_aggregate( xs: Sequence ):

    if not xs:
        # Empty sequence
        return []

    # Aggregate 
    if isinstance( xs[0], np.ndarray ):
        return np.array( list( xs ) )

    return list( xs )

class SampleBatch( Generic[DT] ):

    def __init__( self, samples: Sequence[DT] ):
        """TODO"""
        self.samples = list( samples )
        self._aggregate_cache = dict()

    @property
    def sample_type( self ) -> Type:
        """The type of each sample in this batch"""
        return self.__orig_class__.__args__[0]

    def __getattr__( self, name ):
        # Aggregate named params of sample type
        if name in vars( self.sample_type )['__annotations__']:
            if name not in self._aggregate_cache:
                self._aggregate_cache[name] = _batch_aggregate(
                    [ getattr( x, name )
                      for x in self.samples ]
                )
            
            return self._aggregate_cache[name]
        
        raise AttributeError( f'No sample attribute named {name}' )


# class AnySample( BaseModel ):
#     """A sample that can hold anything"""
#     value: Any

# class AnyBatch( BaseModel ):
#     """A batch of `AnySample`s"""
#     values: list[AnySample]


ST = TypeVar( 'ST', bound = PackableSample )
# BT = TypeVar( 'BT' )

# TODO For python 3.13
# BT = TypeVar( 'BT', default = None )
# IT = TypeVar( 'IT', default = Any )

class Dataset( Generic[ST] ):
    """A dataset that ingests and formats raw samples from a WebDataset
    
    (Abstract base for subclassing)
    """

    # sample_class: Type = get_parameters( )
    # """The type of each returned sample from this `Dataset`'s iterator"""
    # batch_class: Type = get_bound( BT )
    # """The type of a batch built from `sample_class`"""

    @property
    def sample_type( self ) -> Type:
        """The type of each returned sample from this `Dataset`'s iterator"""
        return self.__orig_class__.__args__[0]
    @property
    def batch_type( self ) -> Type:
        """The type of a batch built from `sample_class`"""
        # return self.__orig_class__.__args__[1]
        return SampleBatch[self.sample_type]


    # _schema_registry_sample: dict[str, Type]
    # _schema_registry_batch: dict[str, Type | None]

    #

    def __init__( self, url: str ) -> None:
        """TODO"""
        super().__init__()
        self.url = url

    # @classmethod
    # def register( cls, uri: str,
    #             sample_class: Type,
    #             batch_class: Optional[Type] = None,
    #         ):
    #     """Register an `ekumen` schema to use a particular dataset sample class"""
    #     cls._schema_registry_sample[uri] = sample_class
    #     cls._schema_registry_batch[uri] = batch_class

    # @classmethod
    # def at( cls, uri: str ) -> 'Dataset':
    #     """Create a Dataset for the `ekumen` index entry at `uri`"""
    #     client = eat.Client()
    #     return cls( )
    
    # Common functionality

    @property
    def shard_list( self ) -> list[str]:
        """List of individual dataset shards
        
        Returns:
            A full (non-lazy) list of the individual ``tar`` files within the
            source WebDataset.
        """
        pipe = wds.DataPipeline(
            wds.SimpleShardList( self.url ),
            wds.map( lambda x: x['url'] )
        )
        return list( pipe )
    
    def ordered( self,
                batch_size: int | None = 1,
            ) -> wds.DataPipeline:
        """Iterate over the dataset in order
        
        Args:
            batch_size (:obj:`int`, optional): The size of iterated batches.
                Default: 1. If ``None``, iterates over one sample at a time
                with no batch dimension.
        
        Returns:
            :obj:`webdataset.DataPipeline` A data pipeline that iterates over
            the dataset in its original sample order
        
        """

        if batch_size is None:
            # TODO Duplication here
            return wds.DataPipeline(
                wds.SimpleShardList( self.url ),
                wds.split_by_worker,
                #
                wds.tarfile_to_samples(),
                # wds.map( self.preprocess ),
                wds.map( self.wrap ),
            )

        return wds.DataPipeline(
            wds.SimpleShardList( self.url ),
            wds.split_by_worker,
            #
            wds.tarfile_to_samples(),
            # wds.map( self.preprocess ),
            wds.batched( batch_size ),
            wds.map( self.wrap_batch ),
        )

    def shuffled( self,
                buffer_shards: int = 100,
                buffer_samples: int = 10_000,
                batch_size: int | None = 1,
            ) -> wds.DataPipeline:
        """Iterate over the dataset in random order
        
        Args:
            buffer_shards (int): Asdf
            batch_size (:obj:`int`, optional) The size of iterated batches.
                Default: 1. If ``None``, iterates over one sample at a time
                with no batch dimension.
        
        Returns:
            :obj:`webdataset.DataPipeline` A data pipeline that iterates over
                the dataset in its original sample order
        
        """

        if batch_size is None:
            # TODO Duplication here
            return wds.DataPipeline(
                wds.SimpleShardList( self.url ),
                wds.shuffle( buffer_shards ),
                wds.split_by_worker,
                #
                wds.tarfile_to_samples(),
                # wds.shuffle( buffer_samples ),
                # wds.map( self.preprocess ),
                wds.shuffle( buffer_samples ),
                wds.map( self.wrap ),
            )

        return wds.DataPipeline(
            wds.SimpleShardList( self.url ),
            wds.shuffle( buffer_shards ),
            wds.split_by_worker,
            #
            wds.tarfile_to_samples(),
            # wds.shuffle( buffer_samples ),
            # wds.map( self.preprocess ),
            wds.shuffle( buffer_samples ),
            wds.batched( batch_size ),
            wds.map( self.wrap_batch ),
        )

    # Implemented by specific subclasses

    # @property
    # @abstractmethod
    # def url( self ) -> str:
    #     """str: Brace-notation URL of the underlying full WebDataset"""
    #     pass

    # @classmethod
    # # TODO replace Any with IT
    # def preprocess( cls, sample: WDSRawSample ) -> Any:
    #     """Pre-built preprocessor for a raw `sample` from the given dataset"""
    #     return sample

    # @classmethod
    # TODO replace Any with IT
    def wrap( self, sample: MsgpackRawSample ) -> ST:
        """Wrap a `sample` into the appropriate dataset-specific type"""
        assert 'msgpack' in sample
        assert type( sample['msgpack'] ) == bytes
        
        return self.sample_type.from_bytes( sample['msgpack'] )
    
        try:
            assert type( sample ) == dict
            return cls.sample_class( **{
                k: v
                for k, v in sample.items() if k != '__key__'
            } )
        
        except Exception as e:
            # Sample constructor failed -- revert to default
            return AnySample(
                value = sample,
            )

    def wrap_batch( self, batch: WDSRawBatch ) -> SampleBatch[ST]:
        """Wrap a `batch` of samples into the appropriate dataset-specific type
       
        This default implementation simply creates a list one sample at a time
        """

        assert 'msgpack' in batch
        batch_unpacked = [ self.sample_type.from_bytes( bs )
                           for bs in batch['msgpack'] ]
        return SampleBatch[self.sample_type]( batch_unpacked )


    # # @classmethod
    # def wrap_batch( self, batch: WDSRawBatch ) -> BT:
    #     """Wrap a `batch` of samples into the appropriate dataset-specific type
        
    #     This default implementation simply creates a list one sample at a time
    #     """
    #     assert cls.batch_class is not None, 'No batch class specified'
    #     return cls.batch_class( **batch )