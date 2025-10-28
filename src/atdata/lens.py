"""Lenses between typed datasets"""

##
# Imports

from .dataset import PackableSample

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
)


##
# Typing helpers

DatasetType: TypeAlias = Type[PackableSample]
LensSignature: TypeAlias = Tuple[DatasetType, DatasetType]

S = TypeVar( 'S', bound = PackableSample )
V = TypeVar( 'V', bound = PackableSample )
type LensGetter[S, V] = Callable[[S], V]
type LensPutter[S, V] = Callable[[V, S], S]


##
# Shortcut decorators

class Lens( Generic[S, V] ):
    """TODO"""

    def __init__( self, get: LensGetter[S, V],
                put: Optional[LensPutter[S, V]] = None
            ) -> None:
        """TODO"""
        ##

        # Update
        functools.update_wrapper( self, get )

        # Store the getter
        self._getter = get
        
        # Determine and store the putter
        if put is None:
            # Trivial putter does not update the source
            def _trivial_put( v: V, s: S ) -> S:
                return s
            put = _trivial_put
        
        self._putter = put

        # Register this lens for this type signature
    
        sig = inspect.signature( get )
        input_types = list( sig.parameters.values() )
        assert len( input_types ) == 1, \
            'Wrong number of input args for lens: should only have one'
        
        input_type = input_types[0].annotation
        output_type = sig.return_annotation

        _registered_lenses[(input_type, output_type)] = self
        print( _registered_lenses )
    
    #

    def putter( self, put: LensPutter[S, V] ) -> LensPutter[S, V]:
        """TODO"""
        ##
        self._putter = put
        return put
    
    def put( self, v: V, s: S ) -> S:
        """TODO"""
        return self._putter( v, s )

    def get( self, s: S ) -> V:
        """TODO"""
        return self( s )

    #
    
    def __call__( self, s: S ) -> V:
        return self._getter( s )

def lens( f: LensGetter[S, V] ) -> Lens[S, V]:
    """Register the annotated function `f` as the getter of a sample lens"""
    return Lens[S, V]( f )


##
# Global registration of used lenses

_registered_lenses: Dict[LensSignature, Lens] = dict()
"""TODO"""

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