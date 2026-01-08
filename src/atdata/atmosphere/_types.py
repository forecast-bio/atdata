"""Type definitions for ATProto record structures.

This module defines the data structures used to represent ATProto records
for schemas, datasets, and lenses. These types map to the Lexicon definitions
in the ``ac.foundation.dataset.*`` namespace.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Literal, Any

# Lexicon namespace for atdata records
LEXICON_NAMESPACE = "ac.foundation.dataset"


@dataclass
class AtUri:
    """Parsed AT Protocol URI.

    AT URIs follow the format: at://<authority>/<collection>/<rkey>

    Example:
        >>> uri = AtUri.parse("at://did:plc:abc123/ac.foundation.dataset.sampleSchema/xyz")
        >>> uri.authority
        'did:plc:abc123'
        >>> uri.collection
        'ac.foundation.dataset.sampleSchema'
        >>> uri.rkey
        'xyz'
    """

    authority: str
    """The DID or handle of the repository owner."""

    collection: str
    """The NSID of the record collection."""

    rkey: str
    """The record key within the collection."""

    @classmethod
    def parse(cls, uri: str) -> "AtUri":
        """Parse an AT URI string into components.

        Args:
            uri: AT URI string in format ``at://<authority>/<collection>/<rkey>``

        Returns:
            Parsed AtUri instance.

        Raises:
            ValueError: If the URI format is invalid.
        """
        if not uri.startswith("at://"):
            raise ValueError(f"Invalid AT URI: must start with 'at://': {uri}")

        parts = uri[5:].split("/")
        if len(parts) < 3:
            raise ValueError(f"Invalid AT URI: expected authority/collection/rkey: {uri}")

        return cls(
            authority=parts[0],
            collection=parts[1],
            rkey="/".join(parts[2:]),  # rkey may contain slashes
        )

    def __str__(self) -> str:
        """Format as AT URI string."""
        return f"at://{self.authority}/{self.collection}/{self.rkey}"


@dataclass
class FieldType:
    """Schema field type definition.

    Represents a type in the schema type system, supporting primitives,
    ndarrays, and references to other schemas.
    """

    kind: Literal["primitive", "ndarray", "ref", "array"]
    """The category of type."""

    primitive: Optional[str] = None
    """For kind='primitive': one of 'str', 'int', 'float', 'bool', 'bytes'."""

    dtype: Optional[str] = None
    """For kind='ndarray': numpy dtype string (e.g., 'float32')."""

    shape: Optional[list[int | None]] = None
    """For kind='ndarray': shape constraints (None for any dimension)."""

    ref: Optional[str] = None
    """For kind='ref': AT URI of referenced schema."""

    items: Optional["FieldType"] = None
    """For kind='array': type of array elements."""


@dataclass
class FieldDef:
    """Schema field definition."""

    name: str
    """Field name."""

    field_type: FieldType
    """Type of this field."""

    optional: bool = False
    """Whether this field can be None."""

    description: Optional[str] = None
    """Human-readable description."""


@dataclass
class SchemaRecord:
    """ATProto record for a PackableSample schema.

    Maps to the ``ac.foundation.dataset.sampleSchema`` Lexicon.
    """

    name: str
    """Human-readable schema name."""

    version: str
    """Semantic version string (e.g., '1.0.0')."""

    fields: list[FieldDef]
    """List of field definitions."""

    description: Optional[str] = None
    """Human-readable description."""

    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """When this record was created."""

    metadata: Optional[dict] = None
    """Arbitrary metadata as msgpack-encoded bytes."""

    def to_record(self) -> dict:
        """Convert to ATProto record dict for publishing."""
        record = {
            "$type": f"{LEXICON_NAMESPACE}.sampleSchema",
            "name": self.name,
            "version": self.version,
            "fields": [self._field_to_dict(f) for f in self.fields],
            "createdAt": self.created_at.isoformat(),
        }
        if self.description:
            record["description"] = self.description
        if self.metadata:
            record["metadata"] = self.metadata
        return record

    def _field_to_dict(self, field_def: FieldDef) -> dict:
        """Convert a field definition to dict."""
        result = {
            "name": field_def.name,
            "fieldType": self._type_to_dict(field_def.field_type),
            "optional": field_def.optional,
        }
        if field_def.description:
            result["description"] = field_def.description
        return result

    def _type_to_dict(self, field_type: FieldType) -> dict:
        """Convert a field type to dict."""
        result: dict = {"$type": f"{LEXICON_NAMESPACE}.schemaType#{field_type.kind}"}

        if field_type.kind == "primitive":
            result["primitive"] = field_type.primitive
        elif field_type.kind == "ndarray":
            result["dtype"] = field_type.dtype
            if field_type.shape:
                result["shape"] = field_type.shape
        elif field_type.kind == "ref":
            result["ref"] = field_type.ref
        elif field_type.kind == "array":
            if field_type.items:
                result["items"] = self._type_to_dict(field_type.items)

        return result


@dataclass
class StorageLocation:
    """Dataset storage location specification."""

    kind: Literal["external", "blobs"]
    """Storage type: external URLs or ATProto blobs."""

    urls: Optional[list[str]] = None
    """For kind='external': WebDataset URLs with brace notation."""

    blob_refs: Optional[list[dict]] = None
    """For kind='blobs': ATProto blob references."""


@dataclass
class DatasetRecord:
    """ATProto record for a dataset index.

    Maps to the ``ac.foundation.dataset.record`` Lexicon.
    """

    name: str
    """Human-readable dataset name."""

    schema_ref: str
    """AT URI of the schema record."""

    storage: StorageLocation
    """Where the dataset data is stored."""

    description: Optional[str] = None
    """Human-readable description."""

    tags: list[str] = field(default_factory=list)
    """Searchable tags."""

    license: Optional[str] = None
    """SPDX license identifier."""

    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """When this record was created."""

    metadata: Optional[bytes] = None
    """Arbitrary metadata as msgpack-encoded bytes."""

    def to_record(self) -> dict:
        """Convert to ATProto record dict for publishing."""
        record = {
            "$type": f"{LEXICON_NAMESPACE}.record",
            "name": self.name,
            "schemaRef": self.schema_ref,
            "storage": self._storage_to_dict(),
            "createdAt": self.created_at.isoformat(),
        }
        if self.description:
            record["description"] = self.description
        if self.tags:
            record["tags"] = self.tags
        if self.license:
            record["license"] = self.license
        if self.metadata:
            record["metadata"] = self.metadata
        return record

    def _storage_to_dict(self) -> dict:
        """Convert storage location to dict."""
        if self.storage.kind == "external":
            return {
                "$type": f"{LEXICON_NAMESPACE}.storageExternal",
                "urls": self.storage.urls or [],
            }
        else:
            return {
                "$type": f"{LEXICON_NAMESPACE}.storageBlobs",
                "blobs": self.storage.blob_refs or [],
            }


@dataclass
class CodeReference:
    """Reference to lens code in a git repository."""

    repository: str
    """Git repository URL."""

    commit: str
    """Git commit hash."""

    path: str
    """Path to the code file/function."""


@dataclass
class LensRecord:
    """ATProto record for a lens transformation.

    Maps to the ``ac.foundation.dataset.lens`` Lexicon.
    """

    name: str
    """Human-readable lens name."""

    source_schema: str
    """AT URI of the source schema."""

    target_schema: str
    """AT URI of the target schema."""

    description: Optional[str] = None
    """What this transformation does."""

    getter_code: Optional[CodeReference] = None
    """Reference to getter function code."""

    putter_code: Optional[CodeReference] = None
    """Reference to putter function code."""

    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """When this record was created."""

    def to_record(self) -> dict:
        """Convert to ATProto record dict for publishing."""
        record: dict[str, Any] = {
            "$type": f"{LEXICON_NAMESPACE}.lens",
            "name": self.name,
            "sourceSchema": self.source_schema,
            "targetSchema": self.target_schema,
            "createdAt": self.created_at.isoformat(),
        }
        if self.description:
            record["description"] = self.description
        if self.getter_code:
            record["getterCode"] = {
                "repository": self.getter_code.repository,
                "commit": self.getter_code.commit,
                "path": self.getter_code.path,
            }
        if self.putter_code:
            record["putterCode"] = {
                "repository": self.putter_code.repository,
                "commit": self.putter_code.commit,
                "path": self.putter_code.path,
            }
        return record
