"""Schema publishing and loading for ATProto.

This module provides classes for publishing PackableSample schemas to ATProto
and loading them back. Schemas are published as ``ac.foundation.dataset.sampleSchema``
records.
"""

from dataclasses import fields, is_dataclass
from typing import Type, TypeVar, Optional, Union, get_type_hints, get_origin, get_args
import types

from .client import AtmosphereClient
from ._types import (
    AtUri,
    SchemaRecord,
    FieldDef,
    FieldType,
    LEXICON_NAMESPACE,
)

# Import for type checking only to avoid circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..dataset import PackableSample

ST = TypeVar("ST", bound="PackableSample")


class SchemaPublisher:
    """Publishes PackableSample schemas to ATProto.

    This class introspects a PackableSample class to extract its field
    definitions and publishes them as an ATProto schema record.

    Example:
        >>> @atdata.packable
        ... class MySample:
        ...     image: NDArray
        ...     label: str
        ...
        >>> client = AtmosphereClient()
        >>> client.login("handle", "password")
        >>>
        >>> publisher = SchemaPublisher(client)
        >>> uri = publisher.publish(MySample, version="1.0.0")
        >>> print(uri)
        at://did:plc:.../ac.foundation.dataset.sampleSchema/...
    """

    def __init__(self, client: AtmosphereClient):
        """Initialize the schema publisher.

        Args:
            client: Authenticated AtmosphereClient instance.
        """
        self.client = client

    def publish(
        self,
        sample_type: Type[ST],
        *,
        name: Optional[str] = None,
        version: str = "1.0.0",
        description: Optional[str] = None,
        metadata: Optional[dict] = None,
        rkey: Optional[str] = None,
    ) -> AtUri:
        """Publish a PackableSample schema to ATProto.

        Args:
            sample_type: The PackableSample class to publish.
            name: Human-readable name. Defaults to the class name.
            version: Semantic version string (e.g., '1.0.0').
            description: Human-readable description.
            metadata: Arbitrary metadata dictionary.
            rkey: Optional explicit record key. If not provided, a TID is generated.

        Returns:
            The AT URI of the created schema record.

        Raises:
            ValueError: If sample_type is not a dataclass or client is not authenticated.
            TypeError: If a field type is not supported.
        """
        if not is_dataclass(sample_type):
            raise ValueError(f"{sample_type.__name__} must be a dataclass (use @packable)")

        # Build the schema record
        schema_record = self._build_schema_record(
            sample_type,
            name=name,
            version=version,
            description=description,
            metadata=metadata,
        )

        # Publish to ATProto
        return self.client.create_record(
            collection=f"{LEXICON_NAMESPACE}.sampleSchema",
            record=schema_record.to_record(),
            rkey=rkey,
            validate=False,  # PDS doesn't know our lexicon
        )

    def _build_schema_record(
        self,
        sample_type: Type[ST],
        *,
        name: Optional[str],
        version: str,
        description: Optional[str],
        metadata: Optional[dict],
    ) -> SchemaRecord:
        """Build a SchemaRecord from a PackableSample class."""
        field_defs = []
        type_hints = get_type_hints(sample_type)

        for f in fields(sample_type):
            field_type = type_hints.get(f.name, f.type)
            field_def = self._field_to_def(f.name, field_type)
            field_defs.append(field_def)

        return SchemaRecord(
            name=name or sample_type.__name__,
            version=version,
            description=description,
            fields=field_defs,
            metadata=metadata,
        )

    def _field_to_def(self, name: str, python_type) -> FieldDef:
        """Convert a Python field to a FieldDef."""
        # Check for Optional types (Union with None)
        is_optional = False
        origin = get_origin(python_type)

        # Handle Union types (including Optional which is Union[T, None])
        if origin is Union or isinstance(python_type, types.UnionType):
            args = get_args(python_type)
            non_none_args = [a for a in args if a is not type(None)]
            if type(None) in args or len(non_none_args) < len(args):
                is_optional = True
            if len(non_none_args) == 1:
                python_type = non_none_args[0]
            elif len(non_none_args) > 1:
                # Complex union type - not fully supported yet
                raise TypeError(f"Complex union types not supported: {python_type}")

        field_type = self._python_type_to_field_type(python_type)

        return FieldDef(
            name=name,
            field_type=field_type,
            optional=is_optional,
        )

    def _python_type_to_field_type(self, python_type) -> FieldType:
        """Map a Python type to a FieldType."""
        # Handle primitives
        if python_type is str:
            return FieldType(kind="primitive", primitive="str")
        elif python_type is int:
            return FieldType(kind="primitive", primitive="int")
        elif python_type is float:
            return FieldType(kind="primitive", primitive="float")
        elif python_type is bool:
            return FieldType(kind="primitive", primitive="bool")
        elif python_type is bytes:
            return FieldType(kind="primitive", primitive="bytes")

        # Check for NDArray
        # NDArray from numpy.typing is a special generic alias
        type_str = str(python_type)
        if "NDArray" in type_str or "ndarray" in type_str.lower():
            # Try to extract dtype info if available
            dtype = "float32"  # Default
            args = get_args(python_type)
            if args:
                # NDArray[np.float64] or similar
                dtype_arg = args[-1] if args else None
                if dtype_arg is not None:
                    dtype = self._numpy_dtype_to_string(dtype_arg)

            return FieldType(kind="ndarray", dtype=dtype, shape=None)

        # Check for list/array types
        origin = get_origin(python_type)
        if origin is list:
            args = get_args(python_type)
            if args:
                items = self._python_type_to_field_type(args[0])
                return FieldType(kind="array", items=items)
            else:
                # Untyped list
                return FieldType(kind="array", items=FieldType(kind="primitive", primitive="str"))

        # Check for nested PackableSample (not yet supported)
        if is_dataclass(python_type):
            raise TypeError(
                f"Nested dataclass types not yet supported: {python_type.__name__}. "
                "Publish nested types separately and use references."
            )

        raise TypeError(f"Unsupported type for schema field: {python_type}")

    def _numpy_dtype_to_string(self, dtype) -> str:
        """Convert a numpy dtype annotation to a string."""
        dtype_str = str(dtype)
        # Handle common numpy dtypes
        dtype_map = {
            "float16": "float16",
            "float32": "float32",
            "float64": "float64",
            "int8": "int8",
            "int16": "int16",
            "int32": "int32",
            "int64": "int64",
            "uint8": "uint8",
            "uint16": "uint16",
            "uint32": "uint32",
            "uint64": "uint64",
            "bool": "bool",
            "complex64": "complex64",
            "complex128": "complex128",
        }

        for key, value in dtype_map.items():
            if key in dtype_str:
                return value

        return "float32"  # Default fallback


class SchemaLoader:
    """Loads PackableSample schemas from ATProto.

    This class fetches schema records from ATProto and can list available
    schemas from a repository.

    Example:
        >>> client = AtmosphereClient()
        >>> client.login("handle", "password")
        >>>
        >>> loader = SchemaLoader(client)
        >>> schema = loader.get("at://did:plc:.../ac.foundation.dataset.sampleSchema/...")
        >>> print(schema["name"])
        'MySample'
    """

    def __init__(self, client: AtmosphereClient):
        """Initialize the schema loader.

        Args:
            client: AtmosphereClient instance (authentication optional for reads).
        """
        self.client = client

    def get(self, uri: str | AtUri) -> dict:
        """Fetch a schema record by AT URI.

        Args:
            uri: The AT URI of the schema record.

        Returns:
            The schema record as a dictionary.

        Raises:
            ValueError: If the record is not a schema record.
            atproto.exceptions.AtProtocolError: If record not found.
        """
        record = self.client.get_record(uri)

        expected_type = f"{LEXICON_NAMESPACE}.sampleSchema"
        if record.get("$type") != expected_type:
            raise ValueError(
                f"Record at {uri} is not a schema record. "
                f"Expected $type='{expected_type}', got '{record.get('$type')}'"
            )

        return record

    def list_all(
        self,
        repo: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """List schema records from a repository.

        Args:
            repo: The DID of the repository. Defaults to authenticated user.
            limit: Maximum number of records to return.

        Returns:
            List of schema records.
        """
        return self.client.list_schemas(repo=repo, limit=limit)
