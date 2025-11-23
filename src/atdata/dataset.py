"""Schematized WebDatasets"""

##
# Imports

import webdataset as wds

from pathlib import Path
import uuid
import functools

import dataclasses
import types
from dataclasses import (
    dataclass,
    asdict,
)
from abc import (
    ABC,
    abstractmethod,
)

from tqdm import tqdm
import numpy as np
import pandas as pd

import typing
from typing import (
    Any,
    Optional,
    Dict,
    Sequence,
    Iterable,
    Callable,
    Union,
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
from .lens import Lens, LensNetwork


##
# Typing help

Pathlike = str | Path

WDSRawSample: TypeAlias = Dict[str, Any]
WDSRawBatch: TypeAlias = Dict[str, Any]

SampleExportRow: TypeAlias = Dict[str, Any]
SampleExportMap: TypeAlias = Callable[['PackableSample'], SampleExportRow]


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
    """Convert a value to a msgpack-compatible format.

    Args:
        x: A value to convert. If it's a numpy array, converts to bytes.
            Otherwise returns the value unchanged.

    Returns:
        The value in a format suitable for msgpack serialization.
    """
    # if isinstance( x, ArrayBytes ):
    #     return x.raw_bytes
    if isinstance( x, np.ndarray ):
        return eh.array_to_bytes( x )
    return x

def _is_possibly_ndarray_type( t ):
    """Checks if a type annotation is possibly an NDArray."""

    # Directly an NDArray
    if t == NDArray:
        # print( 'is an NDArray' )
        return True
    
    # Check for Optionals (i.e., NDArray | None)
    if isinstance( t, types.UnionType ):
        t_parts = t.__args__
        if any( x == NDArray
                for x in t_parts ):
            return True
    
    # Not an NDArray
    return False

@dataclass
class PackableSample( ABC ):
    """A sample that can be packed and unpacked with msgpack"""

    def _ensure_good( self ):
        """Auto-convert annotated NDArray fields from bytes to numpy arrays.

        This method scans all dataclass fields and for any field annotated as
        ``NDArray`` or ``NDArray | None``, automatically converts bytes values
        to numpy arrays using the helper deserialization function. This enables
        transparent handling of array serialization in msgpack data.

        Note:
            This is called during ``__post_init__`` to ensure proper type
            conversion after deserialization.
        """

        # Auto-convert known types when annotated
        # for var_name, var_type in vars( self.__class__ )['__annotations__'].items():
        for field in dataclasses.fields( self ):
            var_name = field.name
            var_type = field.type

            # Annotation for this variable is to be an NDArray
            if _is_possibly_ndarray_type( var_type ):
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
                    # TODO This does create a constraint that serialized bytes
                    # in a field that might be an NDArray are always interpreted
                    # as being the NDArray interpretation
                    setattr( self, var_name, eh.bytes_to_array( var_cur_value ) )

    def __post_init__( self ):
        self._ensure_good()

    ##

    @classmethod
    def from_data( cls, data: MsgpackRawSample ) -> Self:
        """Create a sample instance from unpacked msgpack data"""
        ret = cls( **data )
        ret._ensure_good()
        return ret
    
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
        """Pack this sample's data for writing to WebDataset.

        Returns:
            A dictionary with ``__key__`` (UUID v1 for sortable keys) and
            ``msgpack`` (packed sample data) fields suitable for WebDataset.

        Note:
            TODO: Expand to allow specifying explicit ``__key__`` values.
        """
        return {
            # Generates a UUID that is timelike-sortable
            '__key__': str( uuid.uuid1( 0, 0 ) ),
            'msgpack': self.packed,
        }

def _batch_aggregate( xs: Sequence ):
    """Aggregate a sequence of values into a batch-appropriate format.

    Args:
        xs: A sequence of values to aggregate. If the first element is a numpy
            array, all elements are stacked into a single array. Otherwise,
            returns a list.

    Returns:
        A numpy array (if elements are arrays) or a list (otherwise).
    """

    if not xs:
        # Empty sequence
        return []

    # Aggregate
    if isinstance( xs[0], np.ndarray ):
        return np.array( list( xs ) )

    return list( xs )

class SampleBatch( Generic[DT] ):

    def __init__( self, samples: Sequence[DT] ):
        """Create a batch from a sequence of samples.

        Args:
            samples: A sequence of sample instances to aggregate into a batch.
                Each sample must be an instance of a type derived from
                ``PackableSample``.
        """
        self.samples = list( samples )
        self._aggregate_cache = dict()

    @property
    def sample_type( self ) -> Type:
        """The type of each sample in this batch.

        Returns:
            The type parameter ``DT`` used when creating this ``SampleBatch[DT]``.
        """
        return typing.get_args( self.__orig_class__)[0]

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

RT = TypeVar( 'RT', bound = PackableSample )

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
        # TODO Figure out why linting fails here
        return typing.get_args( self.__orig_class__ )[0]
    @property
    def batch_type( self ) -> Type:
        """The type of a batch built from `sample_class`"""
        # return self.__orig_class__.__args__[1]
        return SampleBatch[self.sample_type]


    # _schema_registry_sample: dict[str, Type]
    # _schema_registry_batch: dict[str, Type | None]

    #

    def __init__( self, url: str ) -> None:
        """Create a dataset from a WebDataset URL.

        Args:
            url: WebDataset brace-notation URL pointing to tar files, e.g.,
                ``"path/to/file-{000000..000009}.tar"`` for multiple shards or
                ``"path/to/file-000000.tar"`` for a single shard.
        """
        super().__init__()
        self.url = url

        # Allow addition of automatic transformation of raw underlying data
        self._output_lens: Lens | None = None

    def as_type( self, other: Type[RT] ) -> 'Dataset[RT]':
        """View this dataset through a different sample type using a registered lens.

        Args:
            other: The target sample type to transform into. Must be a type
                derived from ``PackableSample``.

        Returns:
            A new ``Dataset`` instance that yields samples of type ``other``
            by applying the appropriate lens transformation from the global
            ``LensNetwork`` registry.

        Raises:
            ValueError: If no registered lens exists between the current
                sample type and the target type.
        """
        ret = Dataset[other]( self.url )
        # Get the singleton lens registry
        lenses = LensNetwork()
        ret._output_lens = lenses.transform( self.sample_type, ret.sample_type )
        return ret

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
        pipe = wds.pipeline.DataPipeline(
            wds.shardlists.SimpleShardList( self.url ),
            wds.filters.map( lambda x: x['url'] )
        )
        return list( pipe )
    
    def ordered( self,
                batch_size: int | None = 1,
            ) -> Iterable[ST]:
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
            return wds.pipeline.DataPipeline(
                wds.shardlists.SimpleShardList( self.url ),
                wds.shardlists.split_by_worker,
                #
                wds.tariterators.tarfile_to_samples(),
                # wds.map( self.preprocess ),
                wds.filters.map( self.wrap ),
            )

        return wds.pipeline.DataPipeline(
            wds.shardlists.SimpleShardList( self.url ),
            wds.shardlists.split_by_worker,
            #
            wds.tariterators.tarfile_to_samples(),
            # wds.map( self.preprocess ),
            wds.filters.batched( batch_size ),
            wds.filters.map( self.wrap_batch ),
        )

    def shuffled( self,
                buffer_shards: int = 100,
                buffer_samples: int = 10_000,
                batch_size: int | None = 1,
            ) -> Iterable[ST]:
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
            return wds.pipeline.DataPipeline(
                wds.shardlists.SimpleShardList( self.url ),
                wds.filters.shuffle( buffer_shards ),
                wds.shardlists.split_by_worker,
                #
                wds.tariterators.tarfile_to_samples(),
                # wds.shuffle( buffer_samples ),
                # wds.map( self.preprocess ),
                wds.filters.shuffle( buffer_samples ),
                wds.filters.map( self.wrap ),
            )

        return wds.pipeline.DataPipeline(
            wds.shardlists.SimpleShardList( self.url ),
            wds.filters.shuffle( buffer_shards ),
            wds.shardlists.split_by_worker,
            #
            wds.tariterators.tarfile_to_samples(),
            # wds.shuffle( buffer_samples ),
            # wds.map( self.preprocess ),
            wds.filters.shuffle( buffer_samples ),
            wds.filters.batched( batch_size ),
            wds.filters.map( self.wrap_batch ),
        )
    
    # TODO Rewrite to eliminate `pandas` dependency directly calling
    # `fastparquet`
    def to_parquet( self, path: Pathlike,
                sample_map: Optional[SampleExportMap] = None,
                maxcount: Optional[int] = None,
                **kwargs,
            ):
        """Save dataset contents to a `parquet` file at `path`

        `kwargs` sent to `pandas.to_parquet`
        """
        ##

        # Normalize args
        path = Path( path )
        if sample_map is None:
            sample_map = asdict
        
        verbose = kwargs.get( 'verbose', False )

        it = self.ordered( batch_size = None )
        if verbose:
            it = tqdm( it )

        #

        if maxcount is None:
            # Load and save full dataset
            df = pd.DataFrame( [ sample_map( x )
                                 for x in self.ordered( batch_size = None ) ] )
            df.to_parquet( path, **kwargs )
        
        else:
            # Load and save dataset in segments of size `maxcount`

            cur_segment = 0
            cur_buffer = []
            path_template = (path.parent / f'{path.stem}-%06d.{path.suffix}').as_posix()

            for x in self.ordered( batch_size = None ):
                cur_buffer.append( sample_map( x ) )
                
                if len( cur_buffer ) >= maxcount:
                    # Write current segment
                    cur_path = path_template.format( cur_segment )
                    df = pd.DataFrame( cur_buffer )
                    df.to_parquet( cur_path, **kwargs )

                    cur_segment += 1
                    cur_buffer = []
                
            if len( cur_buffer ) > 0:
                # Write one last segment with remainder
                cur_path = path_template.format( cur_segment )
                df = pd.DataFrame( cur_buffer )
                df.to_parquet( cur_path, **kwargs )


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
        """Wrap a raw msgpack sample into the appropriate dataset-specific type.

        Args:
            sample: A dictionary containing at minimum a ``'msgpack'`` key with
                serialized sample bytes.

        Returns:
            A deserialized sample of type ``ST``, optionally transformed through
            a lens if ``as_type()`` was called.
        """
        assert 'msgpack' in sample
        assert type( sample['msgpack'] ) == bytes
        
        if self._output_lens is None:
            return self.sample_type.from_bytes( sample['msgpack'] )

        source_sample = self._output_lens.source_type.from_bytes( sample['msgpack'] )
        return self._output_lens( source_sample )
    
        # try:
        #     assert type( sample ) == dict
        #     return cls.sample_class( **{
        #         k: v
        #         for k, v in sample.items() if k != '__key__'
        #     } )
        
        # except Exception as e:
        #     # Sample constructor failed -- revert to default
        #     return AnySample(
        #         value = sample,
        #     )

    def wrap_batch( self, batch: WDSRawBatch ) -> SampleBatch[ST]:
        """Wrap a batch of raw msgpack samples into a typed SampleBatch.

        Args:
            batch: A dictionary containing a ``'msgpack'`` key with a list of
                serialized sample bytes.

        Returns:
            A ``SampleBatch[ST]`` containing deserialized samples, optionally
            transformed through a lens if ``as_type()`` was called.

        Note:
            This implementation deserializes samples one at a time, then
            aggregates them into a batch.
        """

        assert 'msgpack' in batch

        if self._output_lens is None:
            batch_unpacked = [ self.sample_type.from_bytes( bs )
                               for bs in batch['msgpack'] ]
            return SampleBatch[self.sample_type]( batch_unpacked )

        batch_source = [ self._output_lens.source_type.from_bytes( bs )
                         for bs in batch['msgpack'] ]
        batch_view = [ self._output_lens( s )
                       for s in batch_source ]
        return SampleBatch[self.sample_type]( batch_view )

    # # @classmethod
    # def wrap_batch( self, batch: WDSRawBatch ) -> BT:
    #     """Wrap a `batch` of samples into the appropriate dataset-specific type
        
    #     This default implementation simply creates a list one sample at a time
    #     """
    #     assert cls.batch_class is not None, 'No batch class specified'
    #     return cls.batch_class( **batch )


##
# Shortcut decorators

# def packable( cls ):
#     """TODO"""

#     def decorator( cls ):
#         # Create a new class dynamically
#         # The new class inherits from the new_parent_class first, then the original cls
#         new_bases = (PackableSample,) + cls.__bases__
#         new_cls = type(cls.__name__, new_bases, dict(cls.__dict__))

#         # Optionally, update __module__ and __qualname__ for better introspection
#         new_cls.__module__ = cls.__module__
#         new_cls.__qualname__ = cls.__qualname__

#         return new_cls
#     return decorator

def packable( cls ):
    """Decorator to convert a regular class into a ``PackableSample``.

    This decorator transforms a class into a dataclass that inherits from
    ``PackableSample``, enabling automatic msgpack serialization/deserialization
    with special handling for NDArray fields.

    Args:
        cls: The class to convert. Should have type annotations for its fields.

    Returns:
        A new dataclass that inherits from ``PackableSample`` with the same
        name and annotations as the original class.

    Example:
        >>> @packable
        ... class MyData:
        ...     name: str
        ...     values: NDArray
        ...
        >>> sample = MyData(name="test", values=np.array([1, 2, 3]))
        >>> bytes_data = sample.packed
        >>> restored = MyData.from_bytes(bytes_data)
    """

    ##

    class_name = cls.__name__
    class_annotations = cls.__annotations__

    # Add in dataclass niceness to original class
    as_dataclass = dataclass( cls )

    # This triggers a bunch of behind-the-scenes stuff for the newly annotated class
    @dataclass
    class as_packable( as_dataclass, PackableSample ):
        def __post_init__( self ):
            return PackableSample.__post_init__( self )
    
    # TODO This doesn't properly carry over the original
    as_packable.__name__ = class_name
    as_packable.__annotations__ = class_annotations

    ##

    return as_packable