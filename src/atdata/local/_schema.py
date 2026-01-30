"""Schema models and helper functions for local storage."""

from atdata._type_utils import (
    PRIMITIVE_TYPE_MAP,
    unwrap_optional,
    is_ndarray_type,
    extract_ndarray_dtype,
    parse_semver,
)
from atdata._protocols import Packable

from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from typing import (
    Any,
    Type,
    TypeVar,
    Iterator,
    Optional,
    Literal,
    get_type_hints,
    get_origin,
    get_args,
)

T = TypeVar("T", bound=Packable)

# URI scheme prefixes
_ATDATA_URI_PREFIX = "atdata://local/sampleSchema/"
_LEGACY_URI_PREFIX = "local://schemas/"


class SchemaNamespace:
    """Namespace for accessing loaded schema types as attributes.

    After ``index.load_schema(uri)``, the type is available as an attribute.
    Supports attribute access, iteration, ``len()``, and ``in`` checks.

    Examples:
        >>> index.load_schema("atdata://local/sampleSchema/MySample@1.0.0")
        >>> MyType = index.types.MySample
        >>> sample = MyType(field1="hello", field2=42)

    Note:
        For full IDE autocomplete, enable ``auto_stubs=True`` and add
        ``index.stub_dir`` to your IDE's extraPaths.
    """

    def __init__(self) -> None:
        self._types: dict[str, Type[Packable]] = {}

    def _register(self, name: str, cls: Type[Packable]) -> None:
        """Register a schema type in the namespace."""
        self._types[name] = cls

    def __getattr__(self, name: str) -> Any:
        # Returns Any to avoid IDE complaints about unknown attributes.
        # For full IDE support, import from the generated module instead.
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
        if name not in self._types:
            raise AttributeError(
                f"Schema '{name}' not loaded. "
                f"Call index.load_schema() first to load the schema."
            )
        return self._types[name]

    def __dir__(self) -> list[str]:
        return list(self._types.keys()) + ["_types", "_register", "get"]

    def __iter__(self) -> Iterator[str]:
        return iter(self._types)

    def __len__(self) -> int:
        return len(self._types)

    def __contains__(self, name: str) -> bool:
        return name in self._types

    def __repr__(self) -> str:
        if not self._types:
            return "SchemaNamespace(empty)"
        names = ", ".join(sorted(self._types.keys()))
        return f"SchemaNamespace({names})"

    def get(self, name: str, default: T | None = None) -> Type[Packable] | T | None:
        """Get a type by name, returning default if not found.

        Args:
            name: The schema class name to look up.
            default: Value to return if not found (default: None).

        Returns:
            The schema class, or default if not loaded.
        """
        return self._types.get(name, default)


##
# Schema types


@dataclass
class SchemaFieldType:
    """Schema field type definition for local storage.

    Represents a type in the schema type system, supporting primitives,
    ndarrays, arrays, and references to other schemas.
    """

    kind: Literal["primitive", "ndarray", "ref", "array"]
    """The category of type."""

    primitive: Optional[str] = None
    """For kind='primitive': one of 'str', 'int', 'float', 'bool', 'bytes'."""

    dtype: Optional[str] = None
    """For kind='ndarray': numpy dtype string (e.g., 'float32')."""

    ref: Optional[str] = None
    """For kind='ref': URI of referenced schema."""

    items: Optional["SchemaFieldType"] = None
    """For kind='array': type of array elements."""

    @classmethod
    def from_dict(cls, data: dict) -> "SchemaFieldType":
        """Create from a dictionary (e.g., from Redis storage)."""
        type_str = data.get("$type", "")
        if "#" in type_str:
            kind = type_str.split("#")[-1]
        else:
            kind = data.get("kind", "primitive")

        items = None
        if "items" in data and data["items"]:
            items = cls.from_dict(data["items"])

        return cls(
            kind=kind,  # type: ignore[arg-type]
            primitive=data.get("primitive"),
            dtype=data.get("dtype"),
            ref=data.get("ref"),
            items=items,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        result: dict[str, Any] = {"$type": f"local#{self.kind}"}
        if self.kind == "primitive":
            result["primitive"] = self.primitive
        elif self.kind == "ndarray":
            result["dtype"] = self.dtype
        elif self.kind == "ref":
            result["ref"] = self.ref
        elif self.kind == "array" and self.items:
            result["items"] = self.items.to_dict()
        return result


@dataclass
class SchemaField:
    """Schema field definition for local storage."""

    name: str
    """Field name."""

    field_type: SchemaFieldType
    """Type of this field."""

    optional: bool = False
    """Whether this field can be None."""

    @classmethod
    def from_dict(cls, data: dict) -> "SchemaField":
        """Create from a dictionary."""
        return cls(
            name=data["name"],
            field_type=SchemaFieldType.from_dict(data["fieldType"]),
            optional=data.get("optional", False),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "name": self.name,
            "fieldType": self.field_type.to_dict(),
            "optional": self.optional,
        }


@dataclass
class LocalSchemaRecord:
    """Schema record for local storage.

    Represents a PackableSample schema stored in the local index.
    Aligns with the atmosphere SchemaRecord structure for seamless promotion.
    """

    name: str
    """Schema name (typically the class name)."""

    version: str
    """Semantic version string (e.g., '1.0.0')."""

    fields: list[SchemaField]
    """List of field definitions."""

    ref: str
    """Schema reference URI (atdata://local/sampleSchema/{name}@{version})."""

    description: Optional[str] = None
    """Human-readable description."""

    created_at: Optional[datetime] = None
    """When this schema was published."""

    @classmethod
    def from_dict(cls, data: dict) -> "LocalSchemaRecord":
        """Create from a dictionary (e.g., from Redis storage)."""
        created_at = None
        if "createdAt" in data:
            try:
                created_at = datetime.fromisoformat(data["createdAt"])
            except (ValueError, TypeError):
                created_at = None  # Invalid datetime format, leave as None

        return cls(
            name=data["name"],
            version=data["version"],
            fields=[SchemaField.from_dict(f) for f in data.get("fields", [])],
            ref=data.get("$ref", ""),
            description=data.get("description"),
            created_at=created_at,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        result: dict[str, Any] = {
            "name": self.name,
            "version": self.version,
            "fields": [f.to_dict() for f in self.fields],
            "$ref": self.ref,
        }
        if self.description:
            result["description"] = self.description
        if self.created_at:
            result["createdAt"] = self.created_at.isoformat()
        return result


##
# Schema helpers


def _kind_str_for_sample_type(st: Type[Packable]) -> str:
    """Return fully-qualified 'module.name' string for a sample type."""
    return f"{st.__module__}.{st.__name__}"


def _schema_ref_from_type(sample_type: Type[Packable], version: str) -> str:
    """Generate 'atdata://local/sampleSchema/{name}@{version}' reference."""
    return _make_schema_ref(sample_type.__name__, version)


def _make_schema_ref(name: str, version: str) -> str:
    """Generate schema reference URI from name and version."""
    return f"{_ATDATA_URI_PREFIX}{name}@{version}"


def _parse_schema_ref(ref: str) -> tuple[str, str]:
    """Parse schema reference into (name, version).

    Supports both new format: 'atdata://local/sampleSchema/{name}@{version}'
    and legacy format: 'local://schemas/{module.Class}@{version}'
    """
    if ref.startswith(_ATDATA_URI_PREFIX):
        path = ref[len(_ATDATA_URI_PREFIX) :]
    elif ref.startswith(_LEGACY_URI_PREFIX):
        path = ref[len(_LEGACY_URI_PREFIX) :]
    else:
        raise ValueError(f"Invalid schema reference: {ref}")

    if "@" not in path:
        raise ValueError(f"Schema reference must include version (@version): {ref}")

    name, version = path.rsplit("@", 1)
    # For legacy format, extract just the class name from module.Class
    if "." in name:
        name = name.rsplit(".", 1)[1]
    return name, version


def _increment_patch(version: str) -> str:
    """Increment patch version: 1.0.0 -> 1.0.1"""
    major, minor, patch = parse_semver(version)
    return f"{major}.{minor}.{patch + 1}"


def _python_type_to_field_type(python_type: Any) -> dict:
    """Convert Python type annotation to schema field type dict."""
    if python_type in PRIMITIVE_TYPE_MAP:
        return {
            "$type": "local#primitive",
            "primitive": PRIMITIVE_TYPE_MAP[python_type],
        }

    if is_ndarray_type(python_type):
        return {"$type": "local#ndarray", "dtype": extract_ndarray_dtype(python_type)}

    origin = get_origin(python_type)
    if origin is list:
        args = get_args(python_type)
        items = (
            _python_type_to_field_type(args[0])
            if args
            else {"$type": "local#primitive", "primitive": "str"}
        )
        return {"$type": "local#array", "items": items}

    if is_dataclass(python_type):
        raise TypeError(
            f"Nested dataclass types not yet supported: {python_type.__name__}. "
            "Publish nested types separately and use references."
        )

    raise TypeError(f"Unsupported type for schema field: {python_type}")


def _build_schema_record(
    sample_type: Type[Packable],
    *,
    version: str,
    description: str | None = None,
) -> dict:
    """Build a schema record dict from a PackableSample type.

    Args:
        sample_type: The PackableSample subclass to introspect.
        version: Semantic version string.
        description: Optional human-readable description. If None, uses the
            class docstring.

    Returns:
        Schema record dict suitable for Redis storage.

    Raises:
        ValueError: If sample_type is not a dataclass.
        TypeError: If a field type is not supported.
    """
    if not is_dataclass(sample_type):
        raise ValueError(f"{sample_type.__name__} must be a dataclass (use @packable)")

    # Use docstring as fallback for description
    if description is None:
        description = sample_type.__doc__

    field_defs = []
    type_hints = get_type_hints(sample_type)

    for f in fields(sample_type):
        field_type = type_hints.get(f.name, f.type)
        field_type, is_optional = unwrap_optional(field_type)
        field_type_dict = _python_type_to_field_type(field_type)

        field_defs.append(
            {
                "name": f.name,
                "fieldType": field_type_dict,
                "optional": is_optional,
            }
        )

    return {
        "name": sample_type.__name__,
        "version": version,
        "fields": field_defs,
        "description": description,
        "createdAt": datetime.now(timezone.utc).isoformat(),
    }
