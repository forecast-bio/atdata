"""Lenses between typed datasets"""

##
# Imports

from .dataset import PackableSample

import inspect

from typing import (
    TypeAlias,
    Type,
    Tuple,
    Dict,
    Callable,
)


##
# Typing

DatasetType: TypeAlias = Type[PackableSample]
LensSignature: TypeAlias = Tuple[DatasetType, DatasetType]
Lens: TypeAlias = Callable[[DatasetType], DatasetType]


##
# Shortcut decorators

_registered_lenses: Dict[LensSignature, Lens] = dict()

def lens( f: Lens ) -> Lens:
    """Register the annotated function `f` as a sample lens"""
    ##
    
    sig = inspect.signature( f )

    input_types = list( sig.parameters.values() )
    output_type = sig.return_annotation
    
    _registered_lenses[]
    return f