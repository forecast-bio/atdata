"""Type definitions for ATProto record structures.

This module provides the ``AtUri`` utility class and the ``LEXICON_NAMESPACE``
constant. Lexicon-mirror record types (``LexSchemaRecord``, ``LexDatasetRecord``,
``LexLensRecord``, etc.) have moved to ``atdata.atmosphere._lexicon_types``.

The old type names (``SchemaRecord``, ``DatasetRecord``, ``LensRecord``,
``StorageLocation``, ``FieldType``, ``FieldDef``, ``CodeReference``) are
re-exported here as deprecated aliases for backward compatibility.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Literal, Optional

# Canonical constant — also defined in _lexicon_types but kept here as the
# historically authoritative location so existing imports continue to work.
LEXICON_NAMESPACE = "ac.foundation.dataset"


@dataclass
class AtUri:
    """Parsed AT Protocol URI.

    AT URIs follow the format: at://<authority>/<collection>/<rkey>

    Examples:
        >>> uri = AtUri.parse("at://did:plc:abc123/ac.foundation.dataset.schema/xyz")
        >>> uri.authority
        'did:plc:abc123'
        >>> uri.collection
        'ac.foundation.dataset.schema'
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
    def parse(cls, uri: str) -> AtUri:
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
            raise ValueError(
                f"Invalid AT URI: expected authority/collection/rkey: {uri}"
            )

        return cls(
            authority=parts[0],
            collection=parts[1],
            rkey="/".join(parts[2:]),  # rkey may contain slashes
        )

    def __str__(self) -> str:
        """Format as AT URI string."""
        return f"at://{self.authority}/{self.collection}/{self.rkey}"


# ---------------------------------------------------------------------------
# Deprecated re-exports (will be removed in a future version)
# ---------------------------------------------------------------------------
# These names existed in this module before the lexicon-mirror types were
# split into _lexicon_types.py.  They are re-exported here so that existing
# imports like ``from atdata.atmosphere._types import SchemaRecord`` continue
# to work during the migration period.


def __getattr__(name: str) -> Any:
    _DEPRECATED_ALIASES: dict[str, tuple[str, str]] = {
        # old name → (new module attribute, import path in _lexicon_types)
        "FieldType": ("FieldType", "atdata.atmosphere._lexicon_types"),
        "FieldDef": ("FieldDef", "atdata.atmosphere._lexicon_types"),
        "SchemaRecord": ("LexSchemaRecord", "atdata.atmosphere._lexicon_types"),
        "DatasetRecord": ("LexDatasetRecord", "atdata.atmosphere._lexicon_types"),
        "LensRecord": ("LexLensRecord", "atdata.atmosphere._lexicon_types"),
        "StorageLocation": ("StorageLocation", "atdata.atmosphere._lexicon_types"),
        "CodeReference": ("LexCodeReference", "atdata.atmosphere._lexicon_types"),
    }
    if name in _DEPRECATED_ALIASES:
        new_name, mod_path = _DEPRECATED_ALIASES[name]
        warnings.warn(
            f"{name} has been moved. Import {new_name} from {mod_path} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from . import _lexicon_types

        # For StorageLocation, provide a lightweight shim
        if name == "StorageLocation":
            return _StorageLocationShim
        # FieldType / FieldDef don't exist in _lexicon_types; they were
        # internal-only types used by the old SchemaRecord.  Return them
        # from the shim definitions below.
        if name in ("FieldType", "FieldDef"):
            return _FIELD_SHIMS[name]
        return getattr(_lexicon_types, new_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Lightweight shims for types that have no direct equivalent in _lexicon_types


@dataclass
class _FieldTypeShim:
    """Deprecated: schema field type used by the old SchemaRecord."""

    kind: Literal["primitive", "ndarray", "ref", "array"]
    primitive: Optional[str] = None
    dtype: Optional[str] = None
    shape: Optional[list[int | None]] = None
    ref: Optional[str] = None
    items: Optional[_FieldTypeShim] = None


@dataclass
class _FieldDefShim:
    """Deprecated: schema field definition used by the old SchemaRecord."""

    name: str
    field_type: _FieldTypeShim
    optional: bool = False
    description: Optional[str] = None


@dataclass
class _StorageLocationShim:
    """Deprecated: use StorageHttp / StorageS3 / StorageBlobs instead."""

    kind: Literal["external", "blobs"]
    urls: Optional[list[str]] = None
    blob_refs: Optional[list[dict]] = None


_FIELD_SHIMS: dict[str, type] = {
    "FieldType": _FieldTypeShim,
    "FieldDef": _FieldDefShim,
}
