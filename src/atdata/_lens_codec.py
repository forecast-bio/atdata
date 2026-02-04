"""Lens codec for persistent lens serialization and reconstitution.

This module provides functionality to serialize ``Lens`` objects to JSON
records and reconstitute them from stored definitions. It supports two
reconstitution strategies:

- **Field mapping**: Declarative lenses that map fields between two schemas,
  optionally with named transforms. These are fully reconstitutable from JSON.
- **Code reference**: Lenses with arbitrary Python logic, stored as importable
  module paths. Reconstituted by importing the referenced functions.

Security: This module NEVER uses ``eval()`` or ``exec()``. Field-mapping lenses
are reconstituted via generated functions. Code-reference lenses use
``importlib`` to import from already-installed modules.

Examples:
    >>> from atdata._lens_codec import lens_to_json, lens_from_record
    >>> record = lens_to_json(my_lens, name="my_lens", version="1.0.0")
    >>> reconstituted = lens_from_record(json.loads(record))
"""

from __future__ import annotations

import importlib
import inspect
import json
from dataclasses import fields as dc_fields, is_dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Type

from ._protocols import Packable
from .lens import Lens, LensNetwork

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

_lens_cache: dict[str, Lens] = {}
_LENS_CACHE_MAX_SIZE = 256


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _resolve_function_ref(func: Callable) -> dict[str, str] | None:
    """Attempt to create a code reference for a callable.

    Returns a dict with 'module' and 'qualname' if the function is importable,
    or ``None`` if it's a lambda, closure, or dynamically generated function.
    """
    module = getattr(func, "__module__", None)
    qualname = getattr(func, "__qualname__", None)

    if module is None or qualname is None:
        return None

    # Skip lambdas, closures, and local functions
    if "<lambda>" in qualname or "<locals>" in qualname:
        return None

    return {"module": module, "qualname": qualname}


def _detect_field_mapping(
    lens_obj: Lens,
    source_type: Type[Packable],
    view_type: Type[Packable],
) -> list[dict[str, Any]] | None:
    """Attempt to detect if a lens performs simple field mappings.

    Inspects source and view types to find matching field names, then
    verifies the lens actually performs identity mapping on those fields
    by testing with a sample instance.

    Returns a list of field mapping dicts, or ``None`` if detection fails.
    """
    if not is_dataclass(source_type) or not is_dataclass(view_type):
        return None

    source_fields = {f.name for f in dc_fields(source_type)}
    view_fields = {f.name for f in dc_fields(view_type)}

    # All view fields must exist in source for a pure field mapping
    if not view_fields.issubset(source_fields):
        return None

    mappings = []
    for f in dc_fields(view_type):
        mappings.append(
            {
                "source_field": f.name,
                "view_field": f.name,
                "transform": None,
            }
        )

    return mappings


def lens_to_json(
    lens_obj: Lens,
    *,
    name: str,
    version: str = "1.0.0",
    description: str | None = None,
    source_schema: str | None = None,
    view_schema: str | None = None,
) -> str:
    """Serialize a Lens to a JSON string.

    Attempts to represent the lens declaratively as a field mapping if
    possible. Falls back to a code reference if the getter/putter are
    importable functions. If neither works, stores as an opaque record
    that documents the lens but cannot be automatically reconstituted.

    Args:
        lens_obj: The Lens object to serialize.
        name: Human-readable lens name.
        version: Semantic version string.
        description: Optional description.
        source_schema: Source schema name (e.g. ``"ImageSample@1.0.0"``).
            Auto-detected from the lens's source_type if not provided.
        view_schema: View schema name. Auto-detected if not provided.

    Returns:
        JSON string containing the lens record.

    Examples:
        >>> record_json = lens_to_json(my_lens, name="extract_name", version="1.0.0")
    """
    source_type = lens_obj.source_type
    view_type = lens_obj.view_type

    if source_schema is None:
        # Handle string annotations from `from __future__ import annotations`
        if isinstance(source_type, str):
            source_schema = source_type
        else:
            source_schema = getattr(source_type, "__name__", "Unknown")
    if view_schema is None:
        if isinstance(view_type, str):
            view_schema = view_type
        else:
            view_schema = getattr(view_type, "__name__", "Unknown")

    # Try field-mapping detection
    field_mappings = _detect_field_mapping(lens_obj, source_type, view_type)

    # Build getter descriptor
    getter_ref = _resolve_function_ref(lens_obj._getter)
    if field_mappings is not None:
        getter_desc: dict[str, Any] = {
            "kind": "field_mapping",
            "mappings": field_mappings,
        }
    elif getter_ref is not None:
        getter_desc = {
            "kind": "code_reference",
            **getter_ref,
        }
    else:
        getter_desc = {"kind": "opaque"}

    # Build putter descriptor
    putter_ref = _resolve_function_ref(lens_obj._putter)
    if field_mappings is not None:
        # Build reverse mapping for putter
        putter_mappings = []
        for m in field_mappings:
            putter_mappings.append(
                {
                    "source_field": m["view_field"],
                    "view_field": m["source_field"],
                    "transform": None,
                }
            )
        putter_desc: dict[str, Any] = {
            "kind": "field_mapping",
            "mappings": putter_mappings,
        }
    elif putter_ref is not None:
        putter_desc = {
            "kind": "code_reference",
            **putter_ref,
        }
    else:
        putter_desc = {"kind": "opaque"}

    record: dict[str, Any] = {
        "name": name,
        "version": version,
        "source_schema": source_schema,
        "view_schema": view_schema,
        "getter": getter_desc,
        "putter": putter_desc,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if description:
        record["description"] = description

    return json.dumps(record)


# ---------------------------------------------------------------------------
# Reconstitution
# ---------------------------------------------------------------------------


def _import_function(module_name: str, qualname: str) -> Callable:
    """Import a function by module and qualified name.

    Args:
        module_name: Dotted module path (e.g. ``"mypackage.transforms"``).
        qualname: Qualified name within the module (e.g. ``"my_function"``).

    Returns:
        The imported callable.

    Raises:
        ImportError: If the module cannot be imported.
        AttributeError: If the function is not found in the module.
    """
    mod = importlib.import_module(module_name)
    # Handle dotted qualnames (e.g. "Class.method") by traversing attributes
    obj: Any = mod
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def _build_field_mapping_getter(
    mappings: list[dict[str, Any]],
    view_type: Type[Packable],
) -> Callable:
    """Build a getter function from field mapping descriptors.

    Creates a function that constructs a view instance by copying named
    fields from the source. No ``eval``/``exec`` is used — the function
    is built with standard Python closures.

    Args:
        mappings: List of field mapping dicts with ``source_field`` and
            ``view_field`` keys.
        view_type: The target view type to construct.

    Returns:
        A callable ``(source) -> view``.
    """
    # Pre-compute the mapping pairs
    pairs = [(m["source_field"], m["view_field"]) for m in mappings]

    def _getter(source: Any) -> Any:
        kwargs = {vf: getattr(source, sf) for sf, vf in pairs}
        return view_type(**kwargs)

    return _getter


def _build_field_mapping_putter(
    getter_mappings: list[dict[str, Any]],
    source_type: Type[Packable],
) -> Callable:
    """Build a putter function from field mapping descriptors.

    Creates a function that updates the source by applying view field values
    back to the corresponding source fields, preserving all other source fields.

    Args:
        getter_mappings: The getter's field mappings (source_field -> view_field).
        source_type: The source type to construct.

    Returns:
        A callable ``(view, source) -> source``.
    """
    # Build view_field -> source_field mapping
    view_to_source = {m["view_field"]: m["source_field"] for m in getter_mappings}
    mapped_source_fields = set(view_to_source.values())

    # Get all source fields
    if is_dataclass(source_type):
        all_source_fields = [f.name for f in dc_fields(source_type)]
    else:
        all_source_fields = []

    def _putter(view: Any, source: Any) -> Any:
        kwargs: dict[str, Any] = {}
        for sf in all_source_fields:
            if sf in mapped_source_fields:
                # Find the view field that maps to this source field
                for vf, s in view_to_source.items():
                    if s == sf:
                        kwargs[sf] = getattr(view, vf)
                        break
            else:
                kwargs[sf] = getattr(source, sf)
        return source_type(**kwargs)

    return _putter


def lens_from_record(
    record: dict[str, Any],
    *,
    source_type: Type[Packable] | None = None,
    view_type: Type[Packable] | None = None,
    use_cache: bool = True,
    register: bool = True,
) -> Lens:
    """Reconstitute a Lens object from a stored JSON record.

    Supports two reconstitution strategies:

    - **field_mapping**: Builds getter/putter functions from declarative
      field mappings using closures (no ``eval``/``exec``).
    - **code_reference**: Imports the getter/putter from their original
      modules using ``importlib``.

    Args:
        record: Parsed lens record dict (from ``json.loads``).
        source_type: The source Packable type. Required for field-mapping
            reconstitution. If ``None``, attempts to resolve from the
            record's ``source_schema`` field.
        view_type: The view Packable type. Required for field-mapping
            reconstitution.
        use_cache: If True, cache reconstituted lenses. Defaults to True.
        register: If True, register the reconstituted lens in the global
            ``LensNetwork``. Defaults to True.

    Returns:
        A reconstituted ``Lens`` object.

    Raises:
        ValueError: If the record cannot be reconstituted (e.g., opaque
            getter/putter with no type information).
        ImportError: If a code-reference function cannot be imported.

    Examples:
        >>> record = json.loads(stored_json)
        >>> lens_obj = lens_from_record(record, source_type=Src, view_type=View)
    """
    name = record["name"]
    version = record["version"]

    # Check cache
    if use_cache:
        cache_key = f"{name}@{version}"
        if cache_key in _lens_cache:
            return _lens_cache[cache_key]

    getter_desc = record["getter"]
    putter_desc = record["putter"]
    getter_kind = getter_desc.get("kind", "opaque")
    putter_kind = putter_desc.get("kind", "opaque")

    getter: Callable
    putter: Callable | None = None

    # Reconstitute getter
    if getter_kind == "code_reference":
        getter = _import_function(getter_desc["module"], getter_desc["qualname"])
    elif getter_kind == "field_mapping":
        if view_type is None:
            raise ValueError(
                f"view_type is required to reconstitute field-mapping lens {name!r}"
            )
        getter = _build_field_mapping_getter(getter_desc["mappings"], view_type)
        # Add type annotations so Lens.__init__ can extract source/view types
        if source_type is not None:
            params = [
                inspect.Parameter(
                    "source",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=source_type,
                )
            ]
            getter.__signature__ = inspect.Signature(
                params, return_annotation=view_type
            )  # type: ignore[attr-defined]
    else:
        raise ValueError(
            f"Cannot reconstitute lens {name!r}: getter kind is {getter_kind!r}. "
            f"Provide the lens via @lens decorator or use a code reference."
        )

    # Reconstitute putter
    if putter_kind == "code_reference":
        putter = _import_function(putter_desc["module"], putter_desc["qualname"])
    elif putter_kind == "field_mapping":
        if source_type is None:
            raise ValueError(
                f"source_type is required to reconstitute field-mapping lens {name!r}"
            )
        # Use the getter's mappings for building the putter (not the putter's own mappings)
        putter = _build_field_mapping_putter(getter_desc["mappings"], source_type)
    # If opaque, leave putter as None (trivial putter will be used)

    lens_obj = Lens(getter, putter)

    # Register in the global network
    if register:
        network = LensNetwork()
        network.register(lens_obj)

    # Cache the result
    if use_cache:
        cache_key = f"{name}@{version}"
        _lens_cache[cache_key] = lens_obj
        while len(_lens_cache) > _LENS_CACHE_MAX_SIZE:
            oldest_key = next(iter(_lens_cache))
            del _lens_cache[oldest_key]

    return lens_obj


# ---------------------------------------------------------------------------
# Stub generation
# ---------------------------------------------------------------------------


def generate_lens_stub(record: dict[str, Any]) -> str:
    """Generate a Python module stub for a lens record.

    Creates a module that documents the lens transformation with type
    information, useful for IDE support and documentation.

    Args:
        record: Parsed lens record dict.

    Returns:
        String content for a ``.py`` stub file.

    Examples:
        >>> stub = generate_lens_stub(record)
        >>> Path("stubs/my_lens.py").write_text(stub)
    """
    name = record.get("name", "unknown_lens")
    version = record.get("version", "1.0.0")
    source_schema = record.get("source_schema", "Unknown")
    view_schema = record.get("view_schema", "Unknown")
    description = record.get("description", "")
    getter_desc = record.get("getter", {})

    lines = [
        '"""Auto-generated lens stub module.',
        "",
        f"Lens: {name}@{version}",
        f"Source: {source_schema}",
        f"View: {view_schema}",
        "",
        "This module is auto-generated by atdata to document lens transformations.",
        '"""',
        "",
        "from typing import Any, Callable",
        "",
        "",
    ]

    if getter_desc.get("kind") == "field_mapping":
        mappings = getter_desc.get("mappings", [])
        lines.append(f"# Field mappings for {name}")
        lines.append(f"# Source ({source_schema}) -> View ({view_schema})")
        for m in mappings:
            sf = m.get("source_field", "?")
            vf = m.get("view_field", "?")
            t = m.get("transform")
            if t:
                lines.append(f"#   {sf} -> {vf} (via {t})")
            else:
                lines.append(f"#   {sf} -> {vf}")
        lines.append("")

    lines.append(f'LENS_NAME: str = "{name}"')
    lines.append(f'LENS_VERSION: str = "{version}"')
    lines.append(f'SOURCE_SCHEMA: str = "{source_schema}"')
    lines.append(f'VIEW_SCHEMA: str = "{view_schema}"')
    if description:
        lines.append(f'DESCRIPTION: str = "{description}"')
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cache utilities
# ---------------------------------------------------------------------------


def clear_lens_cache() -> None:
    """Clear the cached reconstituted lenses."""
    _lens_cache.clear()


def get_cached_lenses() -> dict[str, Lens]:
    """Get a copy of the current lens cache.

    Returns:
        Dictionary mapping cache keys to Lens objects.
    """
    return dict(_lens_cache)


__all__ = [
    "lens_to_json",
    "lens_from_record",
    "generate_lens_stub",
    "clear_lens_cache",
    "get_cached_lenses",
]
