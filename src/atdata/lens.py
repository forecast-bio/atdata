"""Lens-based type transformations for datasets.

This module implements a lens system for bidirectional transformations between
different sample types. Lenses enable viewing a dataset through different type
schemas without duplicating the underlying data.

Key components:

- ``Lens``: Bidirectional transformation with getter (S -> V) and optional
  putter (V, S -> S)
- ``LensNetwork``: Global singleton registry for lens transformations
- ``@lens``: Decorator to create and register lens transformations

Lenses support the functional programming concept of composable, well-behaved
transformations that satisfy lens laws (GetPut and PutGet).

Example:
    >>> @packable
    ... class FullData:
    ...     name: str
    ...     age: int
    ...     embedding: NDArray
    ...
    >>> @packable
    ... class NameOnly:
    ...     name: str
    ...
    >>> @lens
    ... def name_view(full: FullData) -> NameOnly:
    ...     return NameOnly(name=full.name)
    ...
    >>> @name_view.putter
    ... def name_view_put(view: NameOnly, source: FullData) -> FullData:
    ...     return FullData(name=view.name, age=source.age,
    ...                     embedding=source.embedding)
    ...
    >>> ds = Dataset[FullData]("data.tar")
    >>> ds_names = ds.as_type(NameOnly)  # Uses registered lens
"""

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
    """A bidirectional transformation between two sample types.

    A lens provides a way to view and update data of type ``S`` (source) as if
    it were type ``V`` (view). It consists of a getter that transforms ``S -> V``
    and an optional putter that transforms ``(V, S) -> S``, enabling updates to
    the view to be reflected back in the source.

    Type Parameters:
        S: The source type, must derive from ``PackableSample``.
        V: The view type, must derive from ``PackableSample``.

    Example:
        >>> @lens
        ... def name_lens(full: FullData) -> NameOnly:
        ...     return NameOnly(name=full.name)
        ...
        >>> @name_lens.putter
        ... def name_lens_put(view: NameOnly, source: FullData) -> FullData:
        ...     return FullData(name=view.name, age=source.age)
    """

    def __init__( self, get: LensGetter[S, V],
                put: Optional[LensPutter[S, V]] = None
            ) -> None:
        """Initialize a lens with a getter and optional putter function.

        Args:
            get: A function that transforms from source type ``S`` to view type
                ``V``. Must accept exactly one parameter annotated with the
                source type.
            put: An optional function that updates the source based on a modified
                view. Takes a view of type ``V`` and original source of type ``S``,
                and returns an updated source of type ``S``. If not provided, a
                trivial putter is used that ignores updates to the view.

        Raises:
            AssertionError: If the getter function doesn't have exactly one
                parameter.
        """
        ##

        # Check argument validity

        sig = inspect.signature( get )
        input_types = list( sig.parameters.values() )
        assert len( input_types ) == 1, \
            'Wrong number of input args for lens: should only have one'

        # Update function details for this object as returned by annotation
        functools.update_wrapper( self, get )

        self.source_type: Type[PackableSample] = input_types[0].annotation
        self.view_type: Type[PackableSample] = sig.return_annotation

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
        """Decorator to register a putter function for this lens.

        Args:
            put: A function that takes a view of type ``V`` and source of type
                ``S``, and returns an updated source of type ``S``.

        Returns:
            The putter function, allowing this to be used as a decorator.

        Example:
            >>> @my_lens.putter
            ... def my_lens_put(view: ViewType, source: SourceType) -> SourceType:
            ...     return SourceType(...)
        """
        ##
        self._putter = put
        return put
    
    # Methods to actually execute transformations

    def put( self, v: V, s: S ) -> S:
        """Update the source based on a modified view.

        Args:
            v: The modified view of type ``V``.
            s: The original source of type ``S``.

        Returns:
            An updated source of type ``S`` that reflects changes from the view.
        """
        return self._putter( v, s )

    def get( self, s: S ) -> V:
        """Transform the source into the view type.

        Args:
            s: The source sample of type ``S``.

        Returns:
            A view of the source as type ``V``.
        """
        return self( s )

    # Convenience to enable calling the lens as its getter

    def __call__( self, s: S ) -> V:
        """Apply the lens transformation (same as ``get()``).

        Args:
            s: The source sample of type ``S``.

        Returns:
            A view of the source as type ``V``.
        """
        return self._getter( s )


def lens(  f: LensGetter[S, V] ) -> Lens[S, V]:
    """Decorator to create and register a lens transformation.

    This decorator converts a getter function into a ``Lens`` object and
    automatically registers it in the global ``LensNetwork`` registry.

    Args:
        f: A getter function that transforms from source type ``S`` to view
            type ``V``. Must have exactly one parameter with a type annotation.

    Returns:
        A ``Lens[S, V]`` object that can be called to apply the transformation
        or decorated with ``@lens_name.putter`` to add a putter function.

    Example:
        >>> @lens
        ... def extract_name(full: FullData) -> NameOnly:
        ...     return NameOnly(name=full.name)
        ...
        >>> @extract_name.putter
        ... def extract_name_put(view: NameOnly, source: FullData) -> FullData:
        ...     return FullData(name=view.name, age=source.age)
    """
    ret = Lens[S, V]( f )
    _network.register( ret )
    return ret


class LensNetwork:
    """Global registry for lens transformations between sample types.

    This class implements a singleton pattern to maintain a global registry of
    all lenses decorated with ``@lens``. It enables looking up transformations
    between different ``PackableSample`` types.

    Attributes:
        _instance: The singleton instance of this class.
        _registry: Dictionary mapping ``(source_type, view_type)`` tuples to
            their corresponding ``Lens`` objects.
    """

    _instance = None
    """The singleton instance"""

    def __new__(cls, *args, **kwargs):
        """Ensure only one instance of LensNetwork exists (singleton pattern)."""
        if cls._instance is None:
            # If no instance exists, create a new one
            cls._instance = super().__new__(cls)
        return cls._instance  # Return the existing (or newly created) instance

    def __init__(self):
        """Initialize the lens registry (only on first instantiation)."""
        if not hasattr(self, '_initialized'):  # Check if already initialized
            self._registry: Dict[LensSignature, Lens] = dict()
            self._initialized = True
    
    def register( self, _lens: Lens ):
        """Register a lens as the canonical transformation between two types.

        Args:
            _lens: The lens to register. Will be stored in the registry under
                the key ``(_lens.source_type, _lens.view_type)``.

        Note:
            If a lens already exists for the same type pair, it will be
            overwritten.
        """
        self._registry[_lens.source_type, _lens.view_type] = _lens
    
    def transform( self, source: DatasetType, view: DatasetType ) -> Lens:
        """Look up the lens transformation between two sample types.

        Args:
            source: The source sample type (must derive from ``PackableSample``).
            view: The target view type (must derive from ``PackableSample``).

        Returns:
            The registered ``Lens`` that transforms from ``source`` to ``view``.

        Raises:
            ValueError: If no lens has been registered for the given type pair.

        Note:
            Currently only supports direct transformations. Compositional
            transformations (chaining multiple lenses) are not yet implemented.
        """
        ret = self._registry.get( (source, view), None )
        if ret is None:
            raise ValueError( f'No registered lens from source {source} to view {view}' )

        return ret


# Global singleton registry instance
_network = LensNetwork()