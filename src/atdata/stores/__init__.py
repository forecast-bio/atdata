"""Data stores for atdata datasets.

Key classes:

- ``LocalDiskStore``: Local filesystem data store.
- ``S3DataStore``: S3-compatible object storage.
"""

from atdata.stores._disk import LocalDiskStore
from atdata.stores._s3 import (
    S3DataStore,
    _s3_env,
    _s3_from_credentials,
    _create_s3_write_callbacks,
)

__all__ = [
    "LocalDiskStore",
    "S3DataStore",
    "_s3_env",
    "_s3_from_credentials",
    "_create_s3_write_callbacks",
]
