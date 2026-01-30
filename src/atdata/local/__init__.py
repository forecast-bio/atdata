"""Local storage backend for atdata datasets.

Key classes:

- ``Index``: Unified index with pluggable providers (SQLite default),
  named repositories, and optional atmosphere backend.
- ``LocalDatasetEntry``: Index entry with ATProto-compatible CIDs.
- ``S3DataStore``: S3-compatible shard storage.
"""

from atdata.local._entry import (
    LocalDatasetEntry,
    BasicIndexEntry,
    REDIS_KEY_DATASET_ENTRY,
    REDIS_KEY_SCHEMA,
)
from atdata.local._schema import (
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
from atdata.local._index import Index
from atdata.local._s3 import (
    S3DataStore,
    _s3_env,
    _s3_from_credentials,
    _create_s3_write_callbacks,
)
from atdata.local._repo_legacy import Repo

# Re-export third-party types that were previously importable from the
# monolithic local.py (tests reference atdata.local.S3FileSystem, etc.)
from s3fs import S3FileSystem  # noqa: F401 — re-exported for backward compat

__all__ = [
    # Public API
    "Index",
    "LocalDatasetEntry",
    "BasicIndexEntry",
    "S3DataStore",
    "Repo",
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
    "_s3_env",
    "_s3_from_credentials",
    "_create_s3_write_callbacks",
]
