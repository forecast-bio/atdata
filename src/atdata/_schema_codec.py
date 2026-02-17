"""Schema codec for dynamic PackableSample type generation.

This module provides functionality to reconstruct Python PackableSample types
from schema records. This enables loading datasets without knowing the sample
type ahead of time - the type can be dynamically generated from stored schema
metadata.

The schema format follows the ATProto record structure defined in
``atmosphere/_types.py``, with field types supporting primitives, ndarrays,
arrays, and schema references.

Examples:
    >>> schema = {
    ...     "name": "ImageSample",
    ...     "version": "1.0.0",
    ...     "fields": [
    ...         {"name": "image", "fieldType": {"$type": "...#ndarray", "dtype": "float32"}, "optional": False},
    ...         {"name": "label", "fieldType": {"$type": "...#primitive", "primitive": "str"}, "optional": False},
    ...     ]
    ... }
    >>> ImageSample = _schema_to_type(schema)
    >>> sample = ImageSample(image=np.zeros((64, 64)), label="cat")
"""

from dataclasses import field, make_dataclass
from typing import Any, Optional, Type
import hashlib

from numpy.typing import NDArray

# Import PackableSample for inheritance in dynamic class generation
from .dataset import PackableSample
from ._protocols import Packable
from ._exceptions import SchemaError

# Maximum atdataSchemaVersion this library can read.
_MAX_SUPPORTED_SCHEMA_VERSION = 1


# Type cache to avoid regenerating identical types
# Uses insertion order (Python 3.7+) for simple FIFO eviction
_type_cache: dict[str, Type[Packable]] = {}
_TYPE_CACHE_MAX_SIZE = 256


def _schema_cache_key(schema: dict) -> str:
    """Generate a cache key for a schema.

    Uses name + version + field signature to identify unique schemas.
    """
    name = schema.get("name", "Unknown")
    version = schema.get("version", "0.0.0")
    fields = schema.get("fields", [])

    # Create a stable string representation of fields
    field_sig = ";".join(
        f"{f['name']}:{f['fieldType'].get('$type', '')}:{f.get('optional', False)}"
        for f in fields
    )

    # Hash for compactness
    sig_hash = hashlib.md5(field_sig.encode()).hexdigest()[:8]
    return f"{name}@{version}#{sig_hash}"


def _field_type_to_python(field_type: dict, optional: bool = False) -> Any:
    """Convert a schema field type to a Python type annotation.

    Args:
        field_type: Field type dict with '$type' and type-specific fields.
        optional: Whether this field is optional (can be None).

    Returns:
        Python type annotation suitable for dataclass field.

    Raises:
        ValueError: If field type is not supported.
    """
    type_str = field_type.get("$type", "")

    # Extract kind from $type (e.g., "ac.foundation.dataset.schemaType#primitive" -> "primitive")
    if "#" in type_str:
        kind = type_str.split("#")[-1]
    else:
        # Fallback for simplified format
        kind = field_type.get("kind", "")

    python_type: Any

    if kind == "primitive":
        primitive = field_type.get("primitive", "str")
        primitive_map = {
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "bytes": bytes,
        }
        python_type = primitive_map.get(primitive)
        if python_type is None:
            raise ValueError(f"Unknown primitive type: {primitive}")

    elif kind == "ndarray":
        # NDArray type - dtype info is available but we use generic NDArray
        # The dtype is handled at runtime by PackableSample serialization
        python_type = NDArray

    elif kind == "array":
        # List type - recursively resolve item type
        items = field_type.get("items")
        if items:
            item_type = _field_type_to_python(items, optional=False)
            python_type = list[item_type]
        else:
            python_type = list

    elif kind == "ref":
        # Reference to another schema - not yet supported for dynamic generation
        raise ValueError(
            f"Schema references ('ref') are not yet supported for dynamic type generation. "
            f"Referenced schema: {field_type.get('ref')}"
        )

    else:
        raise ValueError(f"Unknown field type kind: {kind}")

    # Wrap in Optional if needed
    if optional:
        python_type = Optional[python_type]

    return python_type


def _convert_atmosphere_schema(record: dict) -> dict:
    """Convert atmosphere JSON Schema format to local fields format.

    Handles two variants of the atmosphere schema ``schema`` object:

    * **Nested** (legacy/test format): ``schema.schemaBody.properties``
    * **Flattened** (actual ATProto wire format): ``schema.properties``

    ``JsonSchemaFormat.to_record()`` merges schema-body keys directly into
    the ``schema`` dict, so records fetched from a PDS have ``properties``
    at the top level of ``schema`` with no ``schemaBody`` wrapper.

    Args:
        record: Atmosphere schema record dict.

    Returns:
        Schema dict in local format with ``fields`` array.
    """
    inner_schema = record.get("schema", {})
    # Prefer nested schemaBody if present; fall back to flattened format.
    schema_body = inner_schema.get("schemaBody", inner_schema)
    properties = schema_body.get("properties", {})
    required = set(schema_body.get("required", []))

    fields_list = []
    for field_name, prop in properties.items():
        field_type_dict = _json_schema_prop_to_field_type(prop)
        fields_list.append(
            {
                "name": field_name,
                "fieldType": field_type_dict,
                "optional": field_name not in required,
            }
        )

    return {
        "name": record.get("name", "Unknown"),
        "version": record.get("version", "1.0.0"),
        "fields": fields_list,
        "description": record.get("description"),
    }


def _json_schema_prop_to_field_type(prop: dict) -> dict:
    """Convert a single JSON Schema property to a local fieldType dict.

    Args:
        prop: JSON Schema property definition.

    Returns:
        Local fieldType dict with ``$type`` key.
    """
    # NDArray reference
    if "$ref" in prop and "ndarray" in prop["$ref"]:
        return {"$type": "local#ndarray"}

    json_type = prop.get("type", "")

    if json_type == "string":
        # Check for bytes (base64-encoded)
        if prop.get("format") == "byte" or prop.get("contentEncoding") == "base64":
            return {"$type": "local#primitive", "primitive": "bytes"}
        return {"$type": "local#primitive", "primitive": "str"}

    if json_type == "integer":
        return {"$type": "local#primitive", "primitive": "int"}

    if json_type == "number":
        return {"$type": "local#primitive", "primitive": "float"}

    if json_type == "boolean":
        return {"$type": "local#primitive", "primitive": "bool"}

    if json_type == "array":
        items = prop.get("items")
        if items:
            return {
                "$type": "local#array",
                "items": _json_schema_prop_to_field_type(items),
            }
        return {
            "$type": "local#array",
            "items": {"$type": "local#primitive", "primitive": "str"},
        }

    # Fallback: treat as string
    return {"$type": "local#primitive", "primitive": "str"}


def _check_schema_record_version(schema: dict) -> None:
    """Validate that a schema record's ``atdataSchemaVersion`` is supported.

    Records without the field are treated as version 1 (backward compat).

    Args:
        schema: Schema record dict.

    Raises:
        SchemaError: If the version is higher than this library supports.
    """
    v = schema.get("atdataSchemaVersion", schema.get("$atdataSchemaVersion", 1))
    if v > _MAX_SUPPORTED_SCHEMA_VERSION:
        raise SchemaError(
            f"Unsupported schema record version: {v}. "
            f"This version of atdata supports schema record versions "
            f"up to {_MAX_SUPPORTED_SCHEMA_VERSION}. "
            f"Upgrade atdata to read this schema."
        )


def _is_atmosphere_schema(schema: dict) -> bool:
    """Detect if a schema dict uses atmosphere JSON Schema format."""
    if schema.get("schemaType") == "jsonSchema":
        return True
    inner = schema.get("schema")
    return isinstance(inner, dict) and "schemaBody" in inner


def _schema_to_type(
    schema: dict,
    *,
    use_cache: bool = True,
) -> Type[Packable]:
    """Generate a PackableSample subclass from a schema record.

    This function dynamically creates a dataclass that inherits from PackableSample,
    with fields matching the schema definition. The generated class can be used
    with ``Dataset[T]`` to load and process samples.

    Supports both local schema format (with ``fields`` array) and atmosphere
    JSON Schema format (with ``schemaType: "jsonSchema"``). The latter is
    automatically converted before processing.

    Args:
        schema: Schema record dict with 'name', 'version', 'fields', etc.
            Fields should have 'name', 'fieldType', and 'optional' keys.
        use_cache: If True, cache and reuse generated types for identical schemas.
            Defaults to True.

    Returns:
        A dynamically generated PackableSample subclass.

    Raises:
        ValueError: If schema is malformed or contains unsupported types.

    Examples:
        >>> schema = index.get_schema("local://schemas/MySample@1.0.0")
        >>> MySample = _schema_to_type(schema)
        >>> ds = Dataset[MySample]("data.tar")
        >>> for sample in ds.ordered():
        ...     print(sample)
    """
    # Check atdataSchemaVersion before processing
    _check_schema_record_version(schema)

    # Convert atmosphere JSON Schema format to local format
    if _is_atmosphere_schema(schema):
        schema = _convert_atmosphere_schema(schema)

    # Check cache first
    if use_cache:
        cache_key = _schema_cache_key(schema)
        if cache_key in _type_cache:
            return _type_cache[cache_key]

    # Extract schema metadata
    name = schema.get("name")
    if not name:
        raise ValueError("Schema must have a 'name' field")

    version = schema.get("version", "1.0.0")
    fields_data = schema.get("fields", [])

    if not fields_data:
        raise ValueError("Schema must have at least one field")

    # Build field definitions for make_dataclass
    # Format: (name, type) or (name, type, field())
    dataclass_fields: list[tuple[str, Any] | tuple[str, Any, Any]] = []

    for field_def in fields_data:
        field_name = field_def.get("name")
        if not field_name:
            raise ValueError("Each field must have a 'name'")

        field_type_dict = field_def.get("fieldType", {})
        is_optional = field_def.get("optional", False)

        # Convert to Python type
        python_type = _field_type_to_python(field_type_dict, optional=is_optional)

        # Optional fields need a default value of None
        if is_optional:
            dataclass_fields.append((field_name, python_type, field(default=None)))
        else:
            dataclass_fields.append((field_name, python_type))

    # Create the dataclass dynamically
    # We need to make it inherit from PackableSample and call __post_init__
    generated_class = make_dataclass(
        name,
        dataclass_fields,
        bases=(PackableSample,),
        namespace={
            "__post_init__": lambda self: PackableSample.__post_init__(self),
            "__schema_version__": version,
            "__schema_ref__": schema.get(
                "$ref", None
            ),  # Store original ref if available
        },
    )

    # Cache the generated type with FIFO eviction
    if use_cache:
        cache_key = _schema_cache_key(schema)
        _type_cache[cache_key] = generated_class
        # Evict oldest entries if cache exceeds max size
        while len(_type_cache) > _TYPE_CACHE_MAX_SIZE:
            oldest_key = next(iter(_type_cache))
            del _type_cache[oldest_key]

    return generated_class


def _field_type_to_stub_str(field_type: dict, optional: bool = False) -> str:
    """Convert a schema field type to a Python type string for stub files.

    Args:
        field_type: Field type dict with '$type' and type-specific fields.
        optional: Whether this field is optional (can be None).

    Returns:
        String representation of the Python type for use in .pyi files.
    """
    type_str = field_type.get("$type", "")

    # Extract kind from $type
    if "#" in type_str:
        kind = type_str.split("#")[-1]
    else:
        kind = field_type.get("kind", "")

    if kind == "primitive":
        primitive = field_type.get("primitive", "str")
        py_type = (
            primitive  # str, int, float, bool, bytes are all valid Python type names
        )
    elif kind == "ndarray":
        py_type = "NDArray[Any]"
    elif kind == "array":
        items = field_type.get("items")
        if items:
            item_type = _field_type_to_stub_str(items, optional=False)
            py_type = f"list[{item_type}]"
        else:
            py_type = "list[Any]"
    elif kind == "ref":
        # Reference to another schema - use Any for now
        py_type = "Any"
    else:
        py_type = "Any"

    if optional:
        return f"{py_type} | None"
    return py_type


def generate_stub(schema: dict) -> str:
    """Generate a .pyi stub file content for a schema.

    This function creates type stub content that can be saved to a .pyi file
    to provide IDE autocomplete and type checking support for dynamically
    decoded sample types.

    Note:
        Types created by ``_schema_to_type()`` work correctly at runtime but
        static type checkers cannot analyze dynamically generated classes.
        Stub files bridge this gap by providing static type information.

    Args:
        schema: Schema record dict with 'name', 'version', 'fields', etc.

    Returns:
        String content for a .pyi stub file.

    Examples:
        >>> schema = index.get_schema("atdata://local/schema/MySample@1.0.0")
        >>> stub_content = generate_stub(schema.to_dict())
        >>> # Save to a stubs directory configured in your IDE
        >>> with open("stubs/my_sample.pyi", "w") as f:
        ...     f.write(stub_content)
    """
    name = schema.get("name", "UnknownSample")
    version = schema.get("version", "1.0.0")
    fields = schema.get("fields", [])

    lines = [
        "# Auto-generated stub for dynamically decoded schema",
        f"# Schema: {name}@{version}",
        "#",
        "# Save this file to a stubs directory and configure your IDE to include it.",
        "# For VS Code/Pylance: add to python.analysis.extraPaths in settings.json",
        "# For PyCharm: mark the stubs directory as Sources Root",
        "",
        "from typing import Any",
        "from numpy.typing import NDArray",
        "from atdata import PackableSample",
        "",
        f"class {name}(PackableSample):",
        f'    """Dynamically decoded sample type from schema {name}@{version}."""',
    ]

    # Add field annotations
    if fields:
        for field_def in fields:
            fname = field_def.get("name", "unknown")
            ftype = _field_type_to_stub_str(
                field_def.get("fieldType", {}),
                field_def.get("optional", False),
            )
            lines.append(f"    {fname}: {ftype}")
    else:
        lines.append("    pass")

    # Add __init__ signature
    lines.append("")
    init_params = ["self"]
    for field_def in fields:
        fname = field_def.get("name", "unknown")
        ftype = _field_type_to_stub_str(
            field_def.get("fieldType", {}),
            field_def.get("optional", False),
        )
        if field_def.get("optional", False):
            init_params.append(f"{fname}: {ftype} = None")
        else:
            init_params.append(f"{fname}: {ftype}")

    lines.append(f"    def __init__({', '.join(init_params)}) -> None: ...")
    lines.append("")

    return "\n".join(lines)


def generate_module(schema: dict) -> str:
    """Generate an importable Python module for a schema.

    This function creates a Python module that defines a PackableSample subclass
    matching the schema. Unlike stub files, this module can be imported at runtime,
    allowing ``get_schema_type`` to return properly typed classes.

    The generated class inherits from PackableSample and uses @dataclass decorator
    for proper initialization. This provides both runtime functionality and static
    type checking support.

    Args:
        schema: Schema record dict with 'name', 'version', 'fields', etc.

    Returns:
        String content for a .py module file.

    Examples:
        >>> schema = index.get_schema("atdata://local/schema/MySample@1.0.0")
        >>> module_content = generate_module(schema.to_dict())
        >>> # The module can be imported after being saved
    """
    name = schema.get("name", "UnknownSample")
    version = schema.get("version", "1.0.0")
    fields = schema.get("fields", [])

    lines = [
        '"""Auto-generated module for dynamically decoded schema.',
        "",
        f"Schema: {name}@{version}",
        "",
        "This module is auto-generated by atdata to provide IDE autocomplete",
        "and type checking support for dynamically decoded schema types.",
        '"""',
        "",
        "from dataclasses import dataclass",
        "from typing import Any",
        "from numpy.typing import NDArray",
        "from atdata import PackableSample",
        "",
        "",
        "@dataclass",
        f"class {name}(PackableSample):",
        f'    """Dynamically decoded sample type from schema {name}@{version}."""',
        "",
    ]

    # Add field annotations
    if fields:
        for field_def in fields:
            fname = field_def.get("name", "unknown")
            ftype = _field_type_to_stub_str(
                field_def.get("fieldType", {}),
                field_def.get("optional", False),
            )
            is_optional = field_def.get("optional", False)
            if is_optional:
                lines.append(f"    {fname}: {ftype} = None")
            else:
                lines.append(f"    {fname}: {ftype}")
    else:
        lines.append("    pass")

    lines.append("")
    lines.append("")
    lines.append(f"__all__ = [{name!r}]")
    lines.append("")

    return "\n".join(lines)


def clear_type_cache() -> None:
    """Clear the cached generated types.

    Useful for testing or when schema definitions change.
    """
    _type_cache.clear()


def get_cached_types() -> dict[str, Type[Packable]]:
    """Get a copy of the current type cache.

    Returns:
        Dictionary mapping cache keys to generated types.
    """
    return dict(_type_cache)


def schema_to_type(
    schema: dict,
    *,
    use_cache: bool = True,
) -> Type[Packable]:
    """Generate a PackableSample subclass from a schema record.

    .. deprecated::
        Use ``_schema_to_type`` directly or ``index.get_schema_type()`` instead.
    """
    import warnings

    warnings.warn(
        "schema_to_type() is deprecated, use index.get_schema_type() instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return _schema_to_type(schema, use_cache=use_cache)


__all__ = [
    "schema_to_type",
    "_schema_to_type",
    "generate_stub",
    "generate_module",
    "clear_type_cache",
    "get_cached_types",
]
