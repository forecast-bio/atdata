"""Index and entry models for atdata datasets.

Key classes:

- ``Index``: Unified index with pluggable providers (SQLite default),
  named repositories, and optional atmosphere backend.
- ``LocalDatasetEntry``: Index entry with ATProto-compatible CIDs.
"""

from atdata.index._entry import (
    LocalDatasetEntry,
    BasicIndexEntry,
    REDIS_KEY_DATASET_ENTRY,
    REDIS_KEY_SCHEMA,
)
from atdata.index._schema import (
    SchemaNamespace,
    SchemaFieldType,
    SchemaField,
    LocalSchemaRecord,
    _ATDATA_URI_PREFIX,
    _LEGACY_URI_PREFIX,
    _kind_str_for_sample_type,
    _schema_ref_from_type,
    _make_schema_ref,
    _parse_schema_ref,
    _increment_patch,
    _python_type_to_field_type,
    _build_schema_record,
)
from atdata.index._index import Index

__all__ = [
    # Public API
    "Index",
    "LocalDatasetEntry",
    "BasicIndexEntry",
    "SchemaNamespace",
    "SchemaFieldType",
    "SchemaField",
    "LocalSchemaRecord",
    "REDIS_KEY_DATASET_ENTRY",
    "REDIS_KEY_SCHEMA",
    # Internal helpers (re-exported for backward compatibility)
    "_ATDATA_URI_PREFIX",
    "_LEGACY_URI_PREFIX",
    "_kind_str_for_sample_type",
    "_schema_ref_from_type",
    "_make_schema_ref",
    "_parse_schema_ref",
    "_increment_patch",
    "_python_type_to_field_type",
    "_build_schema_record",
]
