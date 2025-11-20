"""Lenses between typed datasets"""

##
# Imports

import functools
import inspect

from typing import (
    TypeAlias,
    Type,
    TypeVar,
    Tuple,
    Dict,
    Callable,
    Optional,
    Generic,
    #
    TYPE_CHECKING
)

if TYPE_CHECKING:
    from .dataset import PackableSample


##
# Typing helpers

DatasetType: TypeAlias = Type['PackableSample']
LensSignature: TypeAlias = Tuple[DatasetType, DatasetType]

S = TypeVar( 'S', bound = 'PackableSample' )
V = TypeVar( 'V', bound = 'PackableSample' )
type LensGetter[S, V] = Callable[[S], V]
type LensPutter[S, V] = Callable[[V, S], S]


##
# Shortcut decorators

class Lens( Generic[S, V] ):
    """TODO"""

    # @property
    # def source_type( self ) -> Type[S]:
    #     """The source type (S) for the lens; what is put to"""
    #     # TODO Figure out why linting fails here
    #     return self.__orig_class__.__args__[0]

    # @property
    # def view_type( self ) -> Type[V]:
    #     """The view type (V) for the lens; what is get'd from"""
    #     # TODO FIgure out why linting fails here
    #     return self.__orig_class__.__args__[1]

    def __init__( self, get: LensGetter[S, V],
                put: Optional[LensPutter[S, V]] = None
            ) -> None:
        """TODO"""
        ##

        # Check argument validity

        sig = inspect.signature( get )
        input_types = list( sig.parameters.values() )
        assert len( input_types ) == 1, \
            'Wrong number of input args for lens: should only have one'

        # Update function details for this object as returned by annotation
        functools.update_wrapper( self, get )

        self.source_type: Type[PackableSample] = input_types[0].annotation
        self.view_type = sig.return_annotation

        # Store the getter
        self._getter = get
        
        # Determine and store the putter
        if put is None:
            # Trivial putter does not update the source
            def _trivial_put( v: V, s: S ) -> S:
                return s
            put = _trivial_put
        self._putter = put
    
    #

    def putter( self, put: LensPutter[S, V] ) -> LensPutter[S, V]:
        """TODO"""
        ##
        self._putter = put
        return put
    
    # Methods to actually execute transformations

    def put( self, v: V, s: S ) -> S:
        """TODO"""
        return self._putter( v, s )

    def get( self, s: S ) -> V:
        """TODO"""
        return self( s )

    # Convenience to enable calling the lens as its getter
    
    def __call__( self, s: S ) -> V:
        return self._getter( s )

# TODO Figure out how to properly parameterize this
# def _lens_factory[S, V]( register: bool = True ):
#     """Register the annotated function `f` as the getter of a sample lens"""

#     # The actual lens decorator taking a lens getter function to a lens object
#     def _decorator( f: LensGetter[S, V] ) -> Lens[S, V]:
#         ret = Lens[S, V]( f )
#         if register:
#             _network.register( ret )
#         return ret
    
#     # Return the lens decorator
#     return _decorator

# # For convenience
# lens = _lens_factory

def lens(  f: LensGetter[S, V] ) -> Lens[S, V]:
    ret = Lens[S, V]( f )
    _network.register( ret )
    return ret


##
# Global registry of used lenses

# _registered_lenses: Dict[LensSignature, Lens] = dict()
# """TODO"""

class LensNetwork:
    """TODO"""

    _instance = None
    """The singleton instance"""

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            # If no instance exists, create a new one
            cls._instance = super().__new__(cls)
        return cls._instance  # Return the existing (or newly created) instance

    def __init__(self):
        if not hasattr(self, '_initialized'):  # Check if already initialized
            self._registry: Dict[LensSignature, Lens] = dict()
            self._initialized = True
    
    def register( self, _lens: Lens ):
        """Set `lens` as the canonical view between its source and view types"""
    
        # sig = inspect.signature( _lens.get )
        # input_types = list( sig.parameters.values() )
        # assert len( input_types ) == 1, \
        #     'Wrong number of input args for lens: should only have one'
        
        # input_type = input_types[0].annotation
        # print( input_type )
        # output_type = sig.return_annotation

        # self._registry[input_type, output_type] = _lens
        # print( _lens.source_type )
        self._registry[_lens.source_type, _lens.view_type] = _lens
    
    def transform( self, source: DatasetType, view: DatasetType ) -> Lens:
        """TODO"""

        # TODO Handle compositional closure
        ret = self._registry.get( (source, view), None )
        if ret is None:
            raise ValueError( f'No registered lens from source {source} to view {view}' )
        
        return ret


# Create global singleton registry instance
_network = LensNetwork()

# def lens( f: LensPutter ) -> Lens:
#     """Register the annotated function `f` as a sample lens"""
#     ##
    
#     sig = inspect.signature( f )

#     input_types = list( sig.parameters.values() )
#     output_type = sig.return_annotation
    
#     _registered_lenses[]

#     f.lens = Lens(

#     )

#     return f