"""Typed proxy DSL for manifest queries.

Provides ``FieldProxy`` and ``Predicate`` classes that build pandas
filter expressions with IDE autocomplete and type safety.

Components:

- ``FieldProxy``: Wraps a field name; comparison operators return ``Predicate``
- ``Predicate``: Composable boolean expression tree; compiles to pandas ops
- ``query_fields()``: Factory that creates a typed proxy from a sample type
- ``F``: Untyped convenience proxy (Django-style F expressions)

Examples:
    >>> Q = query_fields(MySample)
    >>> pred = (Q.confidence > 0.9) & (Q.label == "dog")

    >>> from atdata.manifest import F
    >>> pred = (F.confidence > 0.9)
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from ._fields import resolve_manifest_fields


class Predicate:
    """A composable boolean predicate over manifest fields.

    Constructed by comparison operators on ``FieldProxy`` objects.
    Supports ``&`` (AND), ``|`` (OR), and ``~`` (NOT).

    Call the predicate directly on a DataFrame to evaluate it,
    or pass it as the ``where`` argument to ``QueryExecutor.query()``.

    Examples:
        >>> from atdata.manifest import F
        >>> pred = (F.confidence > 0.9) & (F.label == "dog")
        >>> pred = (F.score >= 0.5) | (F.label.isin(["cat", "dog"]))
    """

    __slots__ = ("_kind", "_field", "_op", "_value", "_children", "_child", "_compiled")
    __hash__ = None  # type: ignore[assignment]

    def __init__(
        self,
        kind: str,
        *,
        field: str | None = None,
        op: str | None = None,
        value: Any = None,
        children: list[Predicate] | None = None,
        child: Predicate | None = None,
    ) -> None:
        self._kind = kind
        self._field = field
        self._op = op
        self._value = value
        self._children = children
        self._child = child
        self._compiled: Callable[[pd.DataFrame], pd.Series] | None = None

    def __and__(self, other: Predicate) -> Predicate:
        if not isinstance(other, Predicate):
            return NotImplemented
        # Flatten nested ANDs for a cleaner tree
        left = self._children if self._kind == "and" else [self]
        right = other._children if other._kind == "and" else [other]
        return Predicate("and", children=[*left, *right])

    def __or__(self, other: Predicate) -> Predicate:
        if not isinstance(other, Predicate):
            return NotImplemented
        left = self._children if self._kind == "or" else [self]
        right = other._children if other._kind == "or" else [other]
        return Predicate("or", children=[*left, *right])

    def __invert__(self) -> Predicate:
        return Predicate("not", child=self)

    def compile(self) -> Callable[[pd.DataFrame], pd.Series]:
        """Compile this predicate tree into a callable DataFrame filter.

        Returns:
            A callable that accepts a ``pd.DataFrame`` and returns a
            boolean ``pd.Series``.
        """
        if self._compiled is not None:
            return self._compiled

        self._compiled = self._build()
        return self._compiled

    def _build(self) -> Callable[[pd.DataFrame], pd.Series]:
        """Recursively build the pandas filter closure."""
        import pandas as pd  # noqa: F811

        if self._kind == "comparison":
            field = self._field
            op = self._op
            value = self._value
            return _make_comparison(field, op, value)

        if self._kind == "and":
            compiled_children = [c.compile() for c in self._children]  # type: ignore[union-attr]
            return _make_and(compiled_children)

        if self._kind == "or":
            compiled_children = [c.compile() for c in self._children]  # type: ignore[union-attr]
            return _make_or(compiled_children)

        if self._kind == "not":
            compiled_child = self._child.compile()  # type: ignore[union-attr]
            return _make_not(compiled_child)

        raise ValueError(f"Unknown predicate kind: {self._kind!r}")

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        """Evaluate this predicate against a DataFrame.

        This makes ``Predicate`` directly usable as a ``where`` argument
        to ``QueryExecutor.query()`` without any adapter code.
        """
        return self.compile()(df)

    def __repr__(self) -> str:
        if self._kind == "comparison":
            return f"Predicate({self._field!r} {self._op} {self._value!r})"
        if self._kind == "not":
            return f"~{self._child!r}"
        sep = " & " if self._kind == "and" else " | "
        parts = sep.join(repr(c) for c in self._children)  # type: ignore[union-attr]
        return f"({parts})"


def _make_comparison(
    field: str | None, op: str | None, value: Any
) -> Callable[[pd.DataFrame], pd.Series]:
    """Create a closure for a single comparison operation."""
    if op == "gt":
        return lambda df: df[field] > value
    if op == "lt":
        return lambda df: df[field] < value
    if op == "ge":
        return lambda df: df[field] >= value
    if op == "le":
        return lambda df: df[field] <= value
    if op == "eq":
        return lambda df: df[field] == value
    if op == "ne":
        return lambda df: df[field] != value
    if op == "isin":
        return lambda df: df[field].isin(value)
    raise ValueError(f"Unknown operator: {op!r}")


def _make_and(
    children: list[Callable[[pd.DataFrame], pd.Series]],
) -> Callable[[pd.DataFrame], pd.Series]:
    """Create a closure that ANDs multiple child predicates."""

    def _and(df: pd.DataFrame) -> pd.Series:
        return functools.reduce(lambda a, b: a & b, (c(df) for c in children))

    return _and


def _make_or(
    children: list[Callable[[pd.DataFrame], pd.Series]],
) -> Callable[[pd.DataFrame], pd.Series]:
    """Create a closure that ORs multiple child predicates."""

    def _or(df: pd.DataFrame) -> pd.Series:
        return functools.reduce(lambda a, b: a | b, (c(df) for c in children))

    return _or


def _make_not(
    child: Callable[[pd.DataFrame], pd.Series],
) -> Callable[[pd.DataFrame], pd.Series]:
    """Create a closure that negates a child predicate."""
    return lambda df: ~child(df)


class FieldProxy:
    """Proxy for a single manifest field.

    Comparison operators return ``Predicate`` objects for composable queries.

    Args:
        name: The manifest field name (column name in the parquet DataFrame).

    Examples:
        >>> from atdata.manifest import F
        >>> pred = F.confidence > 0.9
        >>> pred = F.label.isin(["dog", "cat"])
    """

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        self._name = name

    def __gt__(self, value: Any) -> Predicate:
        return Predicate("comparison", field=self._name, op="gt", value=value)

    def __lt__(self, value: Any) -> Predicate:
        return Predicate("comparison", field=self._name, op="lt", value=value)

    def __ge__(self, value: Any) -> Predicate:
        return Predicate("comparison", field=self._name, op="ge", value=value)

    def __le__(self, value: Any) -> Predicate:
        return Predicate("comparison", field=self._name, op="le", value=value)

    def __eq__(self, value: Any) -> Predicate:  # type: ignore[override]
        return Predicate("comparison", field=self._name, op="eq", value=value)

    def __ne__(self, value: Any) -> Predicate:  # type: ignore[override]
        return Predicate("comparison", field=self._name, op="ne", value=value)

    def isin(self, values: Sequence[Any]) -> Predicate:
        """Check membership in a set of values.

        Args:
            values: Collection of values to test membership against.

        Returns:
            A ``Predicate`` that filters for rows where this field's
            value is in *values*.

        Examples:
            >>> pred = F.label.isin(["dog", "cat", "bird"])
        """
        return Predicate("comparison", field=self._name, op="isin", value=values)

    def between(self, low: Any, high: Any) -> Predicate:
        """Check that the field value is within a closed range.

        Shorthand for ``(field >= low) & (field <= high)``.

        Args:
            low: Lower bound (inclusive).
            high: Upper bound (inclusive).

        Returns:
            A ``Predicate`` that filters for rows where this field's
            value is between *low* and *high* inclusive.

        Examples:
            >>> pred = F.confidence.between(0.5, 0.9)
        """
        return (self >= low) & (self <= high)

    def __repr__(self) -> str:
        return f"FieldProxy({self._name!r})"


def query_fields(sample_type: type) -> Any:
    """Create a typed field proxy for querying a sample type.

    Returns an object whose attributes are ``FieldProxy`` instances for
    each manifest-eligible field of *sample_type*. Provides IDE
    autocomplete when the return type is inferred.

    Args:
        sample_type: A ``@packable`` or ``PackableSample`` subclass.

    Returns:
        A proxy object with one ``FieldProxy`` attribute per manifest field.

    Raises:
        TypeError: If *sample_type* is not a dataclass.

    Examples:
        >>> Q = query_fields(MySample)
        >>> pred = (Q.confidence > 0.9) & (Q.label == "dog")
    """
    fields = resolve_manifest_fields(sample_type)
    attrs: dict[str, Any] = {}
    annotations: dict[str, type] = {}
    for name in fields:
        attrs[name] = FieldProxy(name)
        annotations[name] = FieldProxy
    attrs["__annotations__"] = annotations
    attrs["__slots__"] = ()
    attrs["__repr__"] = lambda self: f"{sample_type.__name__}Fields({', '.join(annotations)})"

    proxy_cls = type(f"{sample_type.__name__}Fields", (), attrs)
    return proxy_cls()


class _UntypedFieldProxy:
    """Untyped convenience proxy for quick field access.

    Attribute access returns a ``FieldProxy`` for any name, without
    requiring a sample type. Useful for ad-hoc queries where IDE
    autocomplete is not needed.

    Examples:
        >>> from atdata.manifest import F
        >>> pred = (F.confidence > 0.9) & (F.label == "dog")
    """

    def __getattr__(self, name: str) -> FieldProxy:
        if name.startswith("_"):
            raise AttributeError(name)
        return FieldProxy(name)

    def __repr__(self) -> str:
        return "F"


F = _UntypedFieldProxy()
