"""Redis-backed index provider.

This module extracts the Redis persistence logic that was previously
inlined in ``atdata.local.Index`` and ``LocalDatasetEntry`` into a
standalone ``IndexProvider`` implementation.
"""

from __future__ import annotations

from typing import Iterator

import msgpack
from redis import Redis

from ._base import IndexProvider
from .._type_utils import parse_semver

# Redis key prefixes — kept in sync with local.py constants
_KEY_DATASET_ENTRY = "LocalDatasetEntry"
_KEY_SCHEMA = "LocalSchema"


class RedisProvider(IndexProvider):
    """Index provider backed by a Redis connection.

    This reproduces the exact storage layout used by the original
    ``Index`` class so that existing Redis data is fully compatible.

    Args:
        redis: An active ``redis.Redis`` connection.
    """

    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    @property
    def redis(self) -> Redis:
        """The underlying Redis connection (for advanced use / migration)."""
        return self._redis

    # ------------------------------------------------------------------
    # Dataset entry operations
    # ------------------------------------------------------------------

    def store_entry(self, entry: "LocalDatasetEntry") -> None:  # noqa: F821
        save_key = f"{_KEY_DATASET_ENTRY}:{entry.cid}"
        data: dict[str, str | bytes] = {
            "name": entry.name,
            "schema_ref": entry.schema_ref,
            "data_urls": msgpack.packb(entry.data_urls),
            "cid": entry.cid,
        }
        if entry.metadata is not None:
            data["metadata"] = msgpack.packb(entry.metadata)
        if entry._legacy_uuid is not None:
            data["legacy_uuid"] = entry._legacy_uuid

        self._redis.hset(save_key, mapping=data)  # type: ignore[arg-type]

    def get_entry_by_cid(self, cid: str) -> "LocalDatasetEntry":  # noqa: F821
        save_key = f"{_KEY_DATASET_ENTRY}:{cid}"
        raw_data = self._redis.hgetall(save_key)
        if not raw_data:
            raise KeyError(f"{_KEY_DATASET_ENTRY} not found: {cid}")

        return _entry_from_redis_hash(raw_data)

    def get_entry_by_name(self, name: str) -> "LocalDatasetEntry":  # noqa: F821
        for entry in self.iter_entries():
            if entry.name == name:
                return entry
        raise KeyError(f"No entry with name: {name}")

    def iter_entries(self) -> Iterator["LocalDatasetEntry"]:  # noqa: F821
        prefix = f"{_KEY_DATASET_ENTRY}:"
        for key in self._redis.scan_iter(match=f"{prefix}*"):
            key_str = key.decode("utf-8") if isinstance(key, bytes) else key
            cid = key_str[len(prefix) :]
            yield self.get_entry_by_cid(cid)

    # ------------------------------------------------------------------
    # Schema operations
    # ------------------------------------------------------------------

    def store_schema(self, name: str, version: str, schema_json: str) -> None:
        redis_key = f"{_KEY_SCHEMA}:{name}@{version}"
        self._redis.set(redis_key, schema_json)

    def get_schema_json(self, name: str, version: str) -> str | None:
        redis_key = f"{_KEY_SCHEMA}:{name}@{version}"
        value = self._redis.get(redis_key)
        if value is None:
            return None
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return value  # type: ignore[return-value]

    def iter_schemas(self) -> Iterator[tuple[str, str, str]]:
        prefix = f"{_KEY_SCHEMA}:"
        for key in self._redis.scan_iter(match=f"{prefix}*"):
            key_str = key.decode("utf-8") if isinstance(key, bytes) else key
            schema_id = key_str[len(prefix) :]

            if "@" not in schema_id:
                continue

            raw_name, version = schema_id.rsplit("@", 1)
            # Handle legacy format: module.Class -> Class
            if "." in raw_name:
                raw_name = raw_name.rsplit(".", 1)[1]

            value = self._redis.get(key)
            if value is None:
                continue
            schema_json = value.decode("utf-8") if isinstance(value, bytes) else value
            yield raw_name, version, schema_json  # type: ignore[misc]

    def find_latest_version(self, name: str) -> str | None:
        latest: tuple[int, int, int] | None = None
        latest_str: str | None = None

        for schema_name, version, _ in self.iter_schemas():
            if schema_name != name:
                continue
            try:
                v = parse_semver(version)
                if latest is None or v > latest:
                    latest = v
                    latest_str = version
            except ValueError:
                continue

        return latest_str

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the Redis connection."""
        self._redis.close()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _entry_from_redis_hash(raw_data: dict) -> "LocalDatasetEntry":  # noqa: F821
    """Reconstruct a ``LocalDatasetEntry`` from a Redis hash mapping."""
    from ..local import LocalDatasetEntry
    from typing import cast

    raw = cast(dict[bytes, bytes], raw_data)
    name = raw[b"name"].decode("utf-8")
    schema_ref = raw[b"schema_ref"].decode("utf-8")
    cid_value = raw.get(b"cid", b"").decode("utf-8") or None
    legacy_uuid = raw.get(b"legacy_uuid", b"").decode("utf-8") or None
    data_urls = msgpack.unpackb(raw[b"data_urls"])
    metadata = None
    if b"metadata" in raw:
        metadata = msgpack.unpackb(raw[b"metadata"])

    return LocalDatasetEntry(
        name=name,
        schema_ref=schema_ref,
        data_urls=data_urls,
        metadata=metadata,
        _cid=cid_value,
        _legacy_uuid=legacy_uuid,
    )
