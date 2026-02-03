"""Dataset entry model and Redis key constants."""

from atdata._cid import generate_cid

from dataclasses import dataclass, field
from typing import Any, cast

import msgpack
from redis import Redis


# Redis key prefixes for index entries and schemas
REDIS_KEY_DATASET_ENTRY = "LocalDatasetEntry"
REDIS_KEY_SCHEMA = "LocalSchema"


@dataclass
class LocalDatasetEntry:
    """Index entry for a dataset stored in the local repository.

    Implements the IndexEntry protocol for compatibility with AbstractIndex.
    Uses dual identity: a content-addressable CID (ATProto-compatible) and
    a human-readable name.

    The CID is generated from the entry's content (schema_ref + data_urls),
    ensuring the same data produces the same CID whether stored locally or
    in the atmosphere. This enables seamless promotion from local to ATProto.

    Attributes:
        name: Human-readable name for this dataset.
        schema_ref: Reference to the schema for this dataset.
        data_urls: WebDataset URLs for the data.
        metadata: Arbitrary metadata dictionary, or None if not set.
    """

    ##

    name: str
    """Human-readable name for this dataset."""

    schema_ref: str
    """Reference to the schema for this dataset."""

    data_urls: list[str]
    """WebDataset URLs for the data."""

    metadata: dict | None = None
    """Arbitrary metadata dictionary, or None if not set."""

    _cid: str | None = field(default=None, repr=False)
    """Content identifier (ATProto-compatible CID). Generated from content if not provided."""

    # Legacy field for backwards compatibility during migration
    _legacy_uuid: str | None = field(default=None, repr=False)
    """Legacy UUID for backwards compatibility with existing Redis entries."""

    def __post_init__(self):
        """Generate CID from content if not provided."""
        if self._cid is None:
            self._cid = self._generate_cid()

    def _generate_cid(self) -> str:
        """Generate ATProto-compatible CID from entry content."""
        # CID is based on schema_ref and data_urls - the identity of the dataset
        content = {
            "schema_ref": self.schema_ref,
            "data_urls": self.data_urls,
        }
        return generate_cid(content)

    @property
    def cid(self) -> str:
        """Content identifier (ATProto-compatible CID)."""
        if self._cid is None:
            raise RuntimeError(
                "CID not initialized; this should not happen after __post_init__"
            )
        return self._cid

    # Legacy compatibility

    @property
    def wds_url(self) -> str:
        """Legacy property: returns first data URL for backwards compatibility."""
        return self.data_urls[0] if self.data_urls else ""

    @property
    def sample_kind(self) -> str:
        """Legacy property: returns schema_ref for backwards compatibility."""
        return self.schema_ref

    def write_to(self, redis: Redis):
        """Persist this index entry to Redis.

        Stores the entry as a Redis hash with key '{REDIS_KEY_DATASET_ENTRY}:{cid}'.

        Args:
            redis: Redis connection to write to.
        """
        save_key = f"{REDIS_KEY_DATASET_ENTRY}:{self.cid}"
        data: dict[str, Any] = {
            "name": self.name,
            "schema_ref": self.schema_ref,
            "data_urls": msgpack.packb(self.data_urls),  # Serialize list
            "cid": self.cid,
        }
        if self.metadata is not None:
            data["metadata"] = msgpack.packb(self.metadata)
        if self._legacy_uuid is not None:
            data["legacy_uuid"] = self._legacy_uuid

        redis.hset(save_key, mapping=data)  # type: ignore[arg-type]

    @classmethod
    def from_redis(cls, redis: Redis, cid: str) -> "LocalDatasetEntry":
        """Load an entry from Redis by CID.

        Args:
            redis: Redis connection to read from.
            cid: Content identifier of the entry to load.

        Returns:
            LocalDatasetEntry loaded from Redis.

        Raises:
            KeyError: If entry not found.
        """
        save_key = f"{REDIS_KEY_DATASET_ENTRY}:{cid}"
        raw_data = redis.hgetall(save_key)
        if not raw_data:
            raise KeyError(f"{REDIS_KEY_DATASET_ENTRY} not found: {cid}")

        # Decode string fields, keep binary fields as bytes for msgpack
        raw_data_typed = cast(dict[bytes, bytes], raw_data)
        name = raw_data_typed[b"name"].decode("utf-8")
        schema_ref = raw_data_typed[b"schema_ref"].decode("utf-8")
        cid_value = raw_data_typed.get(b"cid", b"").decode("utf-8") or None
        legacy_uuid = raw_data_typed.get(b"legacy_uuid", b"").decode("utf-8") or None

        # Deserialize msgpack fields (stored as raw bytes)
        data_urls = msgpack.unpackb(raw_data_typed[b"data_urls"])
        metadata = None
        if b"metadata" in raw_data_typed:
            metadata = msgpack.unpackb(raw_data_typed[b"metadata"])

        return cls(
            name=name,
            schema_ref=schema_ref,
            data_urls=data_urls,
            metadata=metadata,
            _cid=cid_value,
            _legacy_uuid=legacy_uuid,
        )


# Backwards compatibility alias
BasicIndexEntry = LocalDatasetEntry
