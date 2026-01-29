"""SQLite-backed index provider.

Stores dataset entries and schema records in a local SQLite database file.
Uses WAL journal mode for concurrent read access and ``INSERT OR REPLACE``
for upsert semantics.

No external dependencies — uses Python's built-in ``sqlite3`` module.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterator

import msgpack

from ._base import IndexProvider
from .._type_utils import parse_semver

_CREATE_TABLES = """\
CREATE TABLE IF NOT EXISTS dataset_entries (
    cid         TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    schema_ref  TEXT NOT NULL,
    data_urls   BLOB NOT NULL,
    metadata    BLOB,
    legacy_uuid TEXT,
    created_at  TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_entries_name
    ON dataset_entries(name);

CREATE TABLE IF NOT EXISTS schemas (
    name        TEXT NOT NULL,
    version     TEXT NOT NULL,
    schema_json TEXT NOT NULL,
    created_at  TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (name, version)
);
"""


class SqliteProvider(IndexProvider):
    """Index provider backed by a local SQLite database.

    Args:
        path: Path to the database file.  The parent directory is created
            automatically.  Defaults to ``~/.atdata/index.db``.

    Examples:
        >>> provider = SqliteProvider(path="/tmp/test-index.db")
        >>> provider.store_schema("MySample", "1.0.0", '{"name":"MySample"}')
        >>> provider.get_schema_json("MySample", "1.0.0")
        '{"name":"MySample"}'
    """

    def __init__(self, path: str | Path | None = None) -> None:
        if path is None:
            path = Path.home() / ".atdata" / "index.db"
        self._path = Path(path).expanduser()
        self._path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(self._path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_CREATE_TABLES)
        self._conn.commit()

    @property
    def path(self) -> Path:
        """Path to the SQLite database file."""
        return self._path

    # ------------------------------------------------------------------
    # Dataset entry operations
    # ------------------------------------------------------------------

    def store_entry(self, entry: "LocalDatasetEntry") -> None:  # noqa: F821
        self._conn.execute(
            """INSERT OR REPLACE INTO dataset_entries
               (cid, name, schema_ref, data_urls, metadata, legacy_uuid)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                entry.cid,
                entry.name,
                entry.schema_ref,
                msgpack.packb(entry.data_urls),
                msgpack.packb(entry.metadata) if entry.metadata is not None else None,
                entry._legacy_uuid,
            ),
        )
        self._conn.commit()

    def get_entry_by_cid(self, cid: str) -> "LocalDatasetEntry":  # noqa: F821
        row = self._conn.execute(
            "SELECT cid, name, schema_ref, data_urls, metadata, legacy_uuid "
            "FROM dataset_entries WHERE cid = ?",
            (cid,),
        ).fetchone()
        if row is None:
            raise KeyError(f"LocalDatasetEntry not found: {cid}")
        return _row_to_entry(row)

    def get_entry_by_name(self, name: str) -> "LocalDatasetEntry":  # noqa: F821
        row = self._conn.execute(
            "SELECT cid, name, schema_ref, data_urls, metadata, legacy_uuid "
            "FROM dataset_entries WHERE name = ? LIMIT 1",
            (name,),
        ).fetchone()
        if row is None:
            raise KeyError(f"No entry with name: {name}")
        return _row_to_entry(row)

    def iter_entries(self) -> Iterator["LocalDatasetEntry"]:  # noqa: F821
        cursor = self._conn.execute(
            "SELECT cid, name, schema_ref, data_urls, metadata, legacy_uuid "
            "FROM dataset_entries"
        )
        for row in cursor:
            yield _row_to_entry(row)

    # ------------------------------------------------------------------
    # Schema operations
    # ------------------------------------------------------------------

    def store_schema(self, name: str, version: str, schema_json: str) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO schemas (name, version, schema_json)
               VALUES (?, ?, ?)""",
            (name, version, schema_json),
        )
        self._conn.commit()

    def get_schema_json(self, name: str, version: str) -> str | None:
        row = self._conn.execute(
            "SELECT schema_json FROM schemas WHERE name = ? AND version = ?",
            (name, version),
        ).fetchone()
        if row is None:
            return None
        return row[0]

    def iter_schemas(self) -> Iterator[tuple[str, str, str]]:
        cursor = self._conn.execute("SELECT name, version, schema_json FROM schemas")
        yield from cursor

    def find_latest_version(self, name: str) -> str | None:
        cursor = self._conn.execute(
            "SELECT version FROM schemas WHERE name = ?",
            (name,),
        )
        latest: tuple[int, int, int] | None = None
        latest_str: str | None = None
        for (version_str,) in cursor:
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
        """Close the SQLite connection."""
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
        data_urls=msgpack.unpackb(data_urls_blob),
        metadata=msgpack.unpackb(metadata_blob) if metadata_blob is not None else None,
        _cid=cid,
        _legacy_uuid=legacy_uuid,
    )
