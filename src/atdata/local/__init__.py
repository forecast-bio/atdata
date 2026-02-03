"""Backward-compatibility shim for atdata.local.

.. deprecated::
    Import from ``atdata.index`` and ``atdata.stores`` instead::

        from atdata.index import Index, LocalDatasetEntry
        from atdata.stores import S3DataStore, LocalDiskStore
"""

from atdata.index import (
    Index,
    LocalDatasetEntry,
    BasicIndexEntry,
    SchemaNamespace,
    SchemaFieldType,
    SchemaField,
    LocalSchemaRecord,
    REDIS_KEY_DATASET_ENTRY,
    REDIS_KEY_SCHEMA,
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
from atdata.stores import (
    LocalDiskStore,
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
    "LocalDiskStore",
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
