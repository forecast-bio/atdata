"""PostgreSQL-backed index provider.

Stores dataset entries and schema records in PostgreSQL tables.
Requires the ``psycopg`` (v3) package, which is an optional dependency::

    pip install "atdata[postgres]"

The provider lazily imports ``psycopg`` so that ``import atdata`` never
fails when the package is absent.
"""

from __future__ import annotations

from typing import Iterator

import msgpack

from ._base import IndexProvider
from .._type_utils import parse_semver

_CREATE_TABLES = """\
CREATE TABLE IF NOT EXISTS dataset_entries (
    cid         TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    schema_ref  TEXT NOT NULL,
    data_urls   BYTEA NOT NULL,
    metadata    BYTEA,
    legacy_uuid TEXT,
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_entries_name
    ON dataset_entries(name);

CREATE TABLE IF NOT EXISTS schemas (
    name        TEXT NOT NULL,
    version     TEXT NOT NULL,
    schema_json TEXT NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (name, version)
);
"""


class PostgresProvider(IndexProvider):
    """Index provider backed by PostgreSQL.

    Args:
        dsn: PostgreSQL connection string, e.g.
            ``"postgresql://user:pass@host:5432/dbname"``.

    Raises:
        ImportError: If ``psycopg`` is not installed.

    Examples:
        >>> provider = PostgresProvider(dsn="postgresql://localhost/atdata")
        >>> provider.store_schema("MySample", "1.0.0", '{"name":"MySample"}')
    """

    def __init__(self, dsn: str) -> None:
        try:
            import psycopg
        except ImportError as exc:
            raise ImportError(
                "The postgres provider requires the 'psycopg' package. "
                "Install it with: pip install 'atdata[postgres]'"
            ) from exc

        self._conn = psycopg.connect(dsn, autocommit=False)
        with self._conn.cursor() as cur:
            cur.execute(_CREATE_TABLES)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Dataset entry operations
    # ------------------------------------------------------------------

    def store_entry(self, entry: "LocalDatasetEntry") -> None:  # noqa: F821
        with self._conn.cursor() as cur:
            cur.execute(
                """INSERT INTO dataset_entries
                   (cid, name, schema_ref, data_urls, metadata, legacy_uuid)
                   VALUES (%s, %s, %s, %s, %s, %s)
                   ON CONFLICT (cid) DO UPDATE SET
                       name = EXCLUDED.name,
                       schema_ref = EXCLUDED.schema_ref,
                       data_urls = EXCLUDED.data_urls,
                       metadata = EXCLUDED.metadata,
                       legacy_uuid = EXCLUDED.legacy_uuid""",
                (
                    entry.cid,
                    entry.name,
                    entry.schema_ref,
                    msgpack.packb(entry.data_urls),
                    msgpack.packb(entry.metadata)
                    if entry.metadata is not None
                    else None,
                    entry._legacy_uuid,
                ),
            )
        self._conn.commit()

    def get_entry_by_cid(self, cid: str) -> "LocalDatasetEntry":  # noqa: F821
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT cid, name, schema_ref, data_urls, metadata, legacy_uuid "
                "FROM dataset_entries WHERE cid = %s",
                (cid,),
            )
            row = cur.fetchone()
        if row is None:
            raise KeyError(f"LocalDatasetEntry not found: {cid}")
        return _row_to_entry(row)

    def get_entry_by_name(self, name: str) -> "LocalDatasetEntry":  # noqa: F821
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT cid, name, schema_ref, data_urls, metadata, legacy_uuid "
                "FROM dataset_entries WHERE name = %s LIMIT 1",
                (name,),
            )
            row = cur.fetchone()
        if row is None:
            raise KeyError(f"No entry with name: {name}")
        return _row_to_entry(row)

    def iter_entries(self) -> Iterator["LocalDatasetEntry"]:  # noqa: F821
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT cid, name, schema_ref, data_urls, metadata, legacy_uuid "
                "FROM dataset_entries"
            )
            for row in cur:
                yield _row_to_entry(row)

    # ------------------------------------------------------------------
    # Schema operations
    # ------------------------------------------------------------------

    def store_schema(self, name: str, version: str, schema_json: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """INSERT INTO schemas (name, version, schema_json)
                   VALUES (%s, %s, %s)
                   ON CONFLICT (name, version) DO UPDATE SET
                       schema_json = EXCLUDED.schema_json""",
                (name, version, schema_json),
            )
        self._conn.commit()

    def get_schema_json(self, name: str, version: str) -> str | None:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT schema_json FROM schemas WHERE name = %s AND version = %s",
                (name, version),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return row[0]

    def iter_schemas(self) -> Iterator[tuple[str, str, str]]:
        with self._conn.cursor() as cur:
            cur.execute("SELECT name, version, schema_json FROM schemas")
            for row in cur:
                yield row[0], row[1], row[2]

    def find_latest_version(self, name: str) -> str | None:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT version FROM schemas WHERE name = %s",
                (name,),
            )
            latest: tuple[int, int, int] | None = None
            latest_str: str | None = None
            for (version_str,) in cur:
                try:
                    v = parse_semver(version_str)
                    if latest is None or v > latest:
                        latest = v
                        latest_str = version_str
                except ValueError:
                    continue
        return latest_str

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the PostgreSQL connection."""
        self._conn.close()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _row_to_entry(row: tuple) -> "LocalDatasetEntry":  # noqa: F821
    """Convert a database row to a ``LocalDatasetEntry``."""
    from ..local import LocalDatasetEntry

    cid, name, schema_ref, data_urls_blob, metadata_blob, legacy_uuid = row
    return LocalDatasetEntry(
        name=name,
        schema_ref=schema_ref,
        data_urls=msgpack.unpackb(bytes(data_urls_blob)),
        metadata=msgpack.unpackb(bytes(metadata_blob))
        if metadata_blob is not None
        else None,
        _cid=cid,
        _legacy_uuid=legacy_uuid,
    )
