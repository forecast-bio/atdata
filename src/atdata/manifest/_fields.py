"""Manifest field annotation and introspection.

Provides the ``ManifestField`` marker for annotating which sample fields
should appear in per-shard manifests, and ``resolve_manifest_fields()``
for introspecting a sample type to discover its manifest-included fields.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Literal, get_args, get_origin, get_type_hints

from atdata._type_utils import PRIMITIVE_TYPE_MAP, is_ndarray_type, unwrap_optional

AggregateKind = Literal["categorical", "numeric", "set"]


@dataclass(frozen=True)
class ManifestField:
    """Marker for manifest-included fields.

    Use with ``Annotated`` to control which fields appear in per-shard
    manifests and what aggregate statistics to compute.

    Args:
        aggregate: The type of statistical aggregate to compute.
        exclude: If True, explicitly exclude this field from the manifest
            even if it would be auto-inferred.

    Examples:
        >>> from typing import Annotated
        >>> from numpy.typing import NDArray
        >>> @atdata.packable
        ... class ImageSample:
        ...     image: NDArray
        ...     label: Annotated[str, ManifestField("categorical")]
        ...     confidence: Annotated[float, ManifestField("numeric")]
        ...     tags: Annotated[list[str], ManifestField("set")]
    """

    aggregate: AggregateKind
    exclude: bool = False


def _extract_manifest_field(annotation: Any) -> ManifestField | None:
    """Extract a ManifestField from an Annotated type, if present."""
    if get_origin(annotation) is not None:
        # Check for Annotated[T, ManifestField(...)]
        # In Python 3.12+, typing.Annotated has __metadata__
        metadata = getattr(annotation, "__metadata__", None)
        if metadata is not None:
            for item in metadata:
                if isinstance(item, ManifestField):
                    return item
    return None


def _infer_aggregate_kind(python_type: Any) -> AggregateKind | None:
    """Infer the aggregate kind from a Python type annotation.

    Returns:
        The inferred aggregate kind, or None if the type should be excluded.
    """
    # Unwrap Optional
    inner_type, _ = unwrap_optional(python_type)

    # Exclude NDArray and bytes
    if is_ndarray_type(inner_type):
        return None
    if inner_type is bytes:
        return None

    # Check primitives
    if inner_type in PRIMITIVE_TYPE_MAP:
        type_name = PRIMITIVE_TYPE_MAP[inner_type]
        if type_name in ("str", "bool"):
            return "categorical"
        if type_name in ("int", "float"):
            return "numeric"
        return None

    # Check list types -> set aggregate
    origin = get_origin(inner_type)
    if origin is list:
        return "set"

    return None


def _get_base_type(annotation: Any) -> Any:
    """Get the base type from an Annotated type or return as-is."""
    args = get_args(annotation)
    metadata = getattr(annotation, "__metadata__", None)
    if metadata is not None and args:
        return args[0]
    return annotation


def resolve_manifest_fields(sample_type: type) -> dict[str, ManifestField]:
    """Extract manifest field descriptors from a Packable type.

    Inspects type hints for ``Annotated[..., ManifestField(...)]`` markers.
    For fields without explicit markers, applies auto-inference rules:

    - ``str``, ``bool`` -> categorical
    - ``int``, ``float`` -> numeric
    - ``list[T]`` -> set
    - ``NDArray``, ``bytes`` -> excluded

    Args:
        sample_type: A ``@packable`` or ``PackableSample`` subclass.

    Returns:
        Dict mapping field name to ``ManifestField`` descriptor for all
        manifest-included fields.

    Examples:
        >>> from typing import Annotated
        >>> @atdata.packable
        ... class MySample:
        ...     label: Annotated[str, ManifestField("categorical")]
        ...     score: float
        >>> fields = resolve_manifest_fields(MySample)
        >>> fields["label"].aggregate
        'categorical'
        >>> fields["score"].aggregate
        'numeric'
    """
    if not dataclasses.is_dataclass(sample_type):
        raise TypeError(f"{sample_type} is not a dataclass")

    hints = get_type_hints(sample_type, include_extras=True)
    dc_fields = {f.name for f in dataclasses.fields(sample_type)}
    result: dict[str, ManifestField] = {}

    for name, annotation in hints.items():
        if name not in dc_fields:
            continue

        # Check for explicit ManifestField annotation
        explicit = _extract_manifest_field(annotation)
        if explicit is not None:
            if not explicit.exclude:
                result[name] = explicit
            continue

        # Auto-infer from base type
        base_type = _get_base_type(annotation)
        kind = _infer_aggregate_kind(base_type)
        if kind is not None:
            result[name] = ManifestField(aggregate=kind)

    return result
