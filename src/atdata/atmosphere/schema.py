"""Schema publishing and loading for ATProto.

This module provides classes for publishing PackableSample schemas to ATProto
and loading them back. Schemas are published as ``ac.foundation.dataset.schema``
records.
"""

from dataclasses import fields, is_dataclass
from typing import Type, TypeVar, Optional, get_type_hints, get_origin, get_args

from .client import Atmosphere
from ._types import AtUri, LEXICON_NAMESPACE
from ._lexicon_types import LexSchemaRecord, JsonSchemaFormat
from .._type_utils import (
    unwrap_optional,
    is_ndarray_type,
)

# Import for type checking only to avoid circular imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .._protocols import Packable

ST = TypeVar("ST", bound="Packable")


class SchemaPublisher:
    """Publishes PackableSample schemas to ATProto.

    This class introspects a PackableSample class to extract its field
    definitions and publishes them as an ATProto schema record.

    Examples:
        >>> @atdata.packable
        ... class MySample:
        ...     image: NDArray
        ...     label: str
        ...
        >>> atmo = Atmosphere.login("handle", "password")
        >>>
        >>> publisher = SchemaPublisher(atmo)
        >>> uri = publisher.publish(MySample, version="1.0.0")
        >>> print(uri)
        at://did:plc:.../ac.foundation.dataset.schema/...
    """

    def __init__(self, client: Atmosphere):
        """Initialize the schema publisher.

        Args:
            client: Authenticated Atmosphere instance.
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
        from atdata._logging import log_operation

        if not is_dataclass(sample_type):
            raise ValueError(
                f"{sample_type.__name__} must be a dataclass (use @packable)"
            )

        with log_operation(
            "SchemaPublisher.publish", schema=sample_type.__name__, version=version
        ):
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
                collection=f"{LEXICON_NAMESPACE}.schema",
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
    ) -> LexSchemaRecord:
        """Build a LexSchemaRecord from a PackableSample class."""
        type_hints = get_type_hints(sample_type)
        properties: dict[str, dict] = {}
        required_fields: list[str] = []
        has_ndarray = False

        for f in fields(sample_type):
            field_type = type_hints.get(f.name, f.type)
            field_type, is_optional = unwrap_optional(field_type)
            prop = self._python_type_to_json_schema(field_type)
            properties[f.name] = prop
            if not is_optional:
                required_fields.append(f.name)
            if is_ndarray_type(field_type):
                has_ndarray = True

        schema_body = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": properties,
        }
        if required_fields:
            schema_body["required"] = required_fields

        array_format_versions = None
        if has_ndarray:
            array_format_versions = {"ndarrayBytes": "1.0.0"}

        return LexSchemaRecord(
            name=name or sample_type.__name__,
            version=version,
            schema_type="jsonSchema",
            schema=JsonSchemaFormat(
                schema_body=schema_body,
                array_format_versions=array_format_versions,
            ),
            description=description,
            metadata=metadata,
        )

    def _python_type_to_json_schema(self, python_type) -> dict:
        """Map a Python type to a JSON Schema property definition."""
        if python_type is str:
            return {"type": "string"}
        if python_type is int:
            return {"type": "integer"}
        if python_type is float:
            return {"type": "number"}
        if python_type is bool:
            return {"type": "boolean"}
        if python_type is bytes:
            return {"type": "string", "format": "byte", "contentEncoding": "base64"}

        if is_ndarray_type(python_type):
            return {
                "$ref": "https://foundation.ac/schemas/atdata-ndarray-bytes/1.0.0#/$defs/ndarray"
            }

        origin = get_origin(python_type)
        if origin is list:
            args = get_args(python_type)
            items = (
                self._python_type_to_json_schema(args[0])
                if args
                else {"type": "string"}
            )
            return {"type": "array", "items": items}

        if is_dataclass(python_type):
            raise TypeError(
                f"Nested dataclass types not yet supported: {python_type.__name__}. "
                "Publish nested types separately and use references."
            )

        raise TypeError(f"Unsupported type for schema field: {python_type}")


class SchemaLoader:
    """Loads PackableSample schemas from ATProto.

    This class fetches schema records from ATProto and can list available
    schemas from a repository.

    Examples:
        >>> atmo = Atmosphere.login("handle", "password")
        >>>
        >>> loader = SchemaLoader(atmo)
        >>> schema = loader.get("at://did:plc:.../ac.foundation.dataset.schema/...")
        >>> print(schema["name"])
        'MySample'
    """

    def __init__(self, client: Atmosphere):
        """Initialize the schema loader.

        Args:
            client: Atmosphere instance (authentication optional for reads).
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

        expected_type = f"{LEXICON_NAMESPACE}.schema"
        if record.get("$type") != expected_type:
            raise ValueError(
                f"Record at {uri} is not a schema record. "
                f"Expected $type='{expected_type}', got '{record.get('$type')}'"
            )

        return record

    def get_typed(self, uri: str | AtUri) -> LexSchemaRecord:
        """Fetch a schema record and return as a typed object.

        Args:
            uri: The AT URI of the schema record.

        Returns:
            LexSchemaRecord instance.
        """
        record = self.get(uri)
        return LexSchemaRecord.from_record(record)

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
