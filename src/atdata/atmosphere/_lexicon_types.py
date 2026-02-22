"""Lexicon-mirror types for the ``science.alt.dataset`` namespace.

These dataclasses map 1:1 to the ATProto Lexicon JSON definitions. They are
the canonical Python representation for serializing to and deserializing from
ATProto record dicts. Each class provides ``to_record()`` and ``from_record()``
for round-trip conversion.

Internal/local types (used outside the atmosphere context) live in
``atdata.index._schema`` and ``atdata.index._entry``.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .._schema_codec import _get_schema_version
from ._types import LEXICON_NAMESPACE


def decode_metadata_raw(raw: Any) -> dict | None:
    """Decode legacy or structured metadata to a plain dict.

    Handles three ATProto metadata formats:

    1. ``$bytes`` envelope — base64-encoded msgpack from ATProto wire format
    2. Raw ``bytes`` — msgpack from local storage or tests
    3. Plain ``dict`` — new structured JSON format

    Args:
        raw: The raw metadata value from the record.

    Returns:
        A flat dictionary, or ``None`` if *raw* is ``None``.

    Raises:
        ValueError: If *raw* is an unrecognised type.
    """
    if raw is None:
        return None

    if isinstance(raw, bytes):
        import msgpack

        return msgpack.unpackb(raw, raw=False)

    if isinstance(raw, dict) and "$bytes" in raw:
        import msgpack

        return msgpack.unpackb(base64.b64decode(raw["$bytes"]), raw=False)

    if isinstance(raw, dict):
        return DatasetMetadata.from_record(raw).to_dict()

    raise ValueError(f"Unexpected metadata format: {type(raw).__name__}")


# ---------------------------------------------------------------------------
# Shared definitions
# ---------------------------------------------------------------------------


@dataclass
class ShardChecksum:
    """Content hash for shard integrity verification.

    Mirrors ``science.alt.dataset.entry#shardChecksum``.
    """

    algorithm: str
    """Hash algorithm identifier (e.g., 'sha256', 'blake3')."""

    digest: str
    """Hex-encoded hash digest."""

    def to_record(self) -> dict[str, str]:
        """Serialize to ATProto record dict."""
        return {"algorithm": self.algorithm, "digest": self.digest}

    @classmethod
    def from_record(cls, d: dict[str, Any]) -> ShardChecksum:
        """Deserialize from ATProto record dict."""
        return cls(algorithm=d["algorithm"], digest=d["digest"])


@dataclass
class DatasetSize:
    """Dataset size metadata.

    Mirrors ``science.alt.dataset.entry#datasetSize``.
    """

    samples: int | None = None
    bytes_: int | None = None
    shards: int | None = None

    def to_record(self) -> dict[str, Any]:
        """Serialize to ATProto record dict."""
        d: dict[str, Any] = {}
        if self.samples is not None:
            d["samples"] = self.samples
        if self.bytes_ is not None:
            d["bytes"] = self.bytes_
        if self.shards is not None:
            d["shards"] = self.shards
        return d

    @classmethod
    def from_record(cls, d: dict[str, Any]) -> DatasetSize:
        """Deserialize from ATProto record dict."""
        return cls(
            samples=d.get("samples"),
            bytes_=d.get("bytes"),
            shards=d.get("shards"),
        )


@dataclass
class ShardManifestRef:
    """References to manifest sidecar data for a single shard.

    Mirrors ``science.alt.dataset.entry#shardManifestRef``.
    The header contains schema info, sample count, and per-field aggregates.
    The samples file is a Parquet table with per-sample metadata for
    query-based access.
    """

    header: dict[str, Any]
    """ATProto blob reference for the manifest JSON header."""

    samples: dict[str, Any] | None = None
    """ATProto blob reference for the Parquet samples file."""

    def to_record(self) -> dict[str, Any]:
        """Serialize to ATProto record dict."""
        d: dict[str, Any] = {"header": self.header}
        if self.samples is not None:
            d["samples"] = self.samples
        return d

    @classmethod
    def from_record(cls, d: dict[str, Any]) -> ShardManifestRef:
        """Deserialize from ATProto record dict."""
        return cls(header=d["header"], samples=d.get("samples"))


@dataclass
class DatasetMetadata:
    """Typed metadata for dataset records.

    Mirrors ``science.alt.dataset.entry#datasetMetadata``. Provides
    well-known fields for common dataset metadata, plus a ``custom`` dict
    for domain-specific extensions.

    All fields are optional — only set what applies to your dataset.

    Examples:
        >>> meta = DatasetMetadata(
        ...     source_uri="https://example.com/raw-data",
        ...     created_by="training-pipeline v2",
        ...     processing_steps=["normalize", "augment"],
        ...     custom={"model": "resnet50", "epoch": 10},
        ... )
        >>> record = meta.to_record()
        >>> restored = DatasetMetadata.from_record(record)
    """

    source_uri: str | None = None
    """URI or identifier of the upstream data source."""

    created_by: str | None = None
    """Tool, pipeline, or user that created this dataset."""

    version: str | None = None
    """Dataset version string (e.g., '2.1.0')."""

    processing_steps: list[str] | None = None
    """Ordered list of processing/transformation steps applied."""

    split: str | None = None
    """Dataset split name (e.g., 'train', 'test', 'validation')."""

    custom: dict[str, Any] | None = None
    """Arbitrary domain-specific metadata. Keys should be short strings."""

    def to_record(self) -> dict[str, Any]:
        """Serialize to ATProto record dict."""
        d: dict[str, Any] = {}
        if self.source_uri is not None:
            d["sourceUri"] = self.source_uri
        if self.created_by is not None:
            d["createdBy"] = self.created_by
        if self.version is not None:
            d["version"] = self.version
        if self.processing_steps is not None:
            d["processingSteps"] = self.processing_steps
        if self.split is not None:
            d["split"] = self.split
        if self.custom is not None:
            d["custom"] = self.custom
        return d

    @classmethod
    def from_record(cls, d: dict[str, Any]) -> DatasetMetadata:
        """Deserialize from ATProto record dict."""
        return cls(
            source_uri=d.get("sourceUri"),
            created_by=d.get("createdBy"),
            version=d.get("version"),
            processing_steps=d.get("processingSteps"),
            split=d.get("split"),
            custom=d.get("custom"),
        )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DatasetMetadata:
        """Create from a plain dict, mapping known keys to typed fields.

        Unknown keys are placed in ``custom``. This provides a migration path
        from the old untyped metadata dicts.

        Args:
            d: Plain metadata dictionary.

        Returns:
            DatasetMetadata with known fields extracted and the rest in custom.

        Examples:
            >>> meta = DatasetMetadata.from_dict({"split": "train", "model": "bert"})
            >>> meta.split
            'train'
            >>> meta.custom
            {'model': 'bert'}
        """
        known_keys = {
            "source_uri",
            "sourceUri",
            "created_by",
            "createdBy",
            "version",
            "processing_steps",
            "processingSteps",
            "split",
            "custom",
        }
        custom_extra: dict[str, Any] = {}
        for k, v in d.items():
            if k not in known_keys:
                custom_extra[k] = v

        explicit_custom = d.get("custom")
        if isinstance(explicit_custom, dict):
            custom_extra.update(explicit_custom)

        def _pick(camel: str, snake: str) -> Any:
            """Return the camelCase value if present, else the snake_case value."""
            v = d.get(camel)
            return v if v is not None else d.get(snake)

        return cls(
            source_uri=_pick("sourceUri", "source_uri"),
            created_by=_pick("createdBy", "created_by"),
            version=d.get("version"),
            processing_steps=_pick("processingSteps", "processing_steps"),
            split=d.get("split"),
            custom=custom_extra or None,
        )

    def to_dict(self) -> dict[str, Any]:
        """Flatten to a plain dict for backward-compatible consumption.

        Well-known fields are placed at the top level with their Python
        attribute names. Custom keys are merged in (custom keys do not
        overwrite well-known keys).

        Returns:
            A flat dict suitable for the old ``dict | None`` metadata API.

        Examples:
            >>> meta = DatasetMetadata(split="train", custom={"model": "bert"})
            >>> meta.to_dict()
            {'split': 'train', 'model': 'bert'}
        """
        d: dict[str, Any] = {}
        if self.source_uri is not None:
            d["source_uri"] = self.source_uri
        if self.created_by is not None:
            d["created_by"] = self.created_by
        if self.version is not None:
            d["version"] = self.version
        if self.processing_steps is not None:
            d["processing_steps"] = self.processing_steps
        if self.split is not None:
            d["split"] = self.split
        if self.custom:
            for k, v in self.custom.items():
                if k not in d:
                    d[k] = v
        return d


# ---------------------------------------------------------------------------
# Storage types
# ---------------------------------------------------------------------------


@dataclass
class HttpShardEntry:
    """A single HTTP-accessible shard with integrity checksum.

    Mirrors ``science.alt.dataset.storageHttp#shardEntry``.
    """

    url: str
    checksum: ShardChecksum

    def to_record(self) -> dict[str, Any]:
        """Serialize to ATProto record dict."""
        return {"url": self.url, "checksum": self.checksum.to_record()}

    @classmethod
    def from_record(cls, d: dict[str, Any]) -> HttpShardEntry:
        """Deserialize from ATProto record dict."""
        return cls(
            url=d["url"],
            checksum=ShardChecksum.from_record(d["checksum"]),
        )


@dataclass
class StorageHttp:
    """HTTP/HTTPS storage for WebDataset tar archives.

    Mirrors ``science.alt.dataset.storageHttp``.
    """

    shards: list[HttpShardEntry]

    def to_record(self) -> dict[str, Any]:
        """Serialize to ATProto record dict."""
        return {
            "$type": f"{LEXICON_NAMESPACE}.storageHttp",
            "shards": [s.to_record() for s in self.shards],
        }

    @classmethod
    def from_record(cls, d: dict[str, Any]) -> StorageHttp:
        """Deserialize from ATProto record dict."""
        return cls(
            shards=[HttpShardEntry.from_record(s) for s in d["shards"]],
        )


@dataclass
class S3ShardEntry:
    """A single S3 object shard with integrity checksum.

    Mirrors ``science.alt.dataset.storageS3#shardEntry``.
    """

    key: str
    checksum: ShardChecksum

    def to_record(self) -> dict[str, Any]:
        """Serialize to ATProto record dict."""
        return {"key": self.key, "checksum": self.checksum.to_record()}

    @classmethod
    def from_record(cls, d: dict[str, Any]) -> S3ShardEntry:
        """Deserialize from ATProto record dict."""
        return cls(
            key=d["key"],
            checksum=ShardChecksum.from_record(d["checksum"]),
        )


@dataclass
class StorageS3:
    """S3/S3-compatible storage for WebDataset tar archives.

    Mirrors ``science.alt.dataset.storageS3``.
    """

    bucket: str
    shards: list[S3ShardEntry]
    region: str | None = None
    endpoint: str | None = None

    def to_record(self) -> dict[str, Any]:
        """Serialize to ATProto record dict."""
        d: dict[str, Any] = {
            "$type": f"{LEXICON_NAMESPACE}.storageS3",
            "bucket": self.bucket,
            "shards": [s.to_record() for s in self.shards],
        }
        if self.region is not None:
            d["region"] = self.region
        if self.endpoint is not None:
            d["endpoint"] = self.endpoint
        return d

    @classmethod
    def from_record(cls, d: dict[str, Any]) -> StorageS3:
        """Deserialize from ATProto record dict."""
        return cls(
            bucket=d["bucket"],
            shards=[S3ShardEntry.from_record(s) for s in d["shards"]],
            region=d.get("region"),
            endpoint=d.get("endpoint"),
        )


@dataclass
class BlobEntry:
    """A single PDS blob shard with optional integrity checksum.

    Mirrors ``science.alt.dataset.storageBlobs#blobEntry``.
    """

    blob: dict[str, Any]
    """ATProto blob reference dict."""

    checksum: ShardChecksum | None = None

    def to_record(self) -> dict[str, Any]:
        """Serialize to ATProto record dict."""
        d: dict[str, Any] = {"blob": self.blob}
        if self.checksum is not None:
            d["checksum"] = self.checksum.to_record()
        return d

    @classmethod
    def from_record(cls, d: dict[str, Any]) -> BlobEntry:
        """Deserialize from ATProto record dict."""
        checksum = None
        if "checksum" in d:
            checksum = ShardChecksum.from_record(d["checksum"])
        return cls(blob=d["blob"], checksum=checksum)


@dataclass
class StorageBlobs:
    """ATProto PDS blob storage for WebDataset tar archives.

    Mirrors ``science.alt.dataset.storageBlobs``.
    """

    blobs: list[BlobEntry]

    def to_record(self) -> dict[str, Any]:
        """Serialize to ATProto record dict."""
        return {
            "$type": f"{LEXICON_NAMESPACE}.storageBlobs",
            "blobs": [b.to_record() for b in self.blobs],
        }

    @classmethod
    def from_record(cls, d: dict[str, Any]) -> StorageBlobs:
        """Deserialize from ATProto record dict."""
        return cls(
            blobs=[BlobEntry.from_record(b) for b in d["blobs"]],
        )


StorageUnion = StorageHttp | StorageS3 | StorageBlobs
"""Union of all storage types for dataset records."""


_STORAGE_TYPE_MAP: dict[str, type[StorageHttp | StorageS3 | StorageBlobs]] = {
    f"{LEXICON_NAMESPACE}.storageHttp": StorageHttp,
    f"{LEXICON_NAMESPACE}.storageS3": StorageS3,
    f"{LEXICON_NAMESPACE}.storageBlobs": StorageBlobs,
}


def storage_from_record(d: dict[str, Any]) -> StorageUnion:
    """Deserialize a storage union variant from an ATProto record dict.

    Args:
        d: Storage dict with ``$type`` discriminator.

    Returns:
        The appropriate storage type instance.

    Raises:
        ValueError: If the ``$type`` is not recognized.
    """
    type_id = d.get("$type", "")
    # Exact match first
    if type_id in _STORAGE_TYPE_MAP:
        return _STORAGE_TYPE_MAP[type_id].from_record(d)
    # Legacy: storageExternal → treat as HTTP (without checksums)
    if "storageExternal" in type_id:
        urls = d.get("urls", [])
        shards = [
            HttpShardEntry(url=url, checksum=ShardChecksum("none", "")) for url in urls
        ]
        return StorageHttp(shards=shards)
    raise ValueError(f"Unknown storage type: {type_id!r}")


# ---------------------------------------------------------------------------
# Code references (lens)
# ---------------------------------------------------------------------------


@dataclass
class LexCodeReference:
    """Reference to code in an external repository.

    Mirrors ``science.alt.dataset.lens#codeReference``.
    All fields are required per the lexicon.
    """

    repository: str
    """Repository URL."""

    commit: str
    """Git commit hash (ensures immutability)."""

    path: str
    """Path to function within repository."""

    branch: str | None = None
    """Optional branch name (commit hash is authoritative)."""

    def to_record(self) -> dict[str, str]:
        """Serialize to ATProto record dict."""
        d: dict[str, str] = {
            "repository": self.repository,
            "commit": self.commit,
            "path": self.path,
        }
        if self.branch is not None:
            d["branch"] = self.branch
        return d

    @classmethod
    def from_record(cls, d: dict[str, Any]) -> LexCodeReference:
        """Deserialize from ATProto record dict."""
        return cls(
            repository=d["repository"],
            commit=d["commit"],
            path=d["path"],
            branch=d.get("branch"),
        )


# ---------------------------------------------------------------------------
# Schema record
# ---------------------------------------------------------------------------


@dataclass
class JsonSchemaFormat:
    """JSON Schema Draft 7 format for sample type definitions.

    Mirrors ``science.alt.dataset.schema#jsonSchemaFormat``.
    """

    schema_body: dict[str, Any]
    """The JSON Schema object (with $schema, type, properties keys)."""

    array_format_versions: dict[str, str] | None = None
    """Mapping from array format identifiers to semver strings."""

    def to_record(self) -> dict[str, Any]:
        """Serialize to ATProto record dict."""
        d: dict[str, Any] = {
            "$type": f"{LEXICON_NAMESPACE}.schema#jsonSchemaFormat",
        }
        # Merge the schema body keys directly into the record
        d.update(self.schema_body)
        if self.array_format_versions:
            d["arrayFormatVersions"] = self.array_format_versions
        return d

    @classmethod
    def from_record(cls, d: dict[str, Any]) -> JsonSchemaFormat:
        """Deserialize from ATProto record dict."""
        afv = d.get("arrayFormatVersions")
        # Extract schema body: everything except $type and arrayFormatVersions
        body = {k: v for k, v in d.items() if k not in ("$type", "arrayFormatVersions")}
        return cls(schema_body=body, array_format_versions=afv)


@dataclass
class LexSchemaRecord:
    """Versioned sample type definition.

    Mirrors ``science.alt.dataset.schema`` (main record).
    """

    name: str
    """Human-readable display name."""

    version: str
    """Semantic version string (e.g., '1.0.0')."""

    schema_type: str
    """Schema format identifier (e.g., 'jsonSchema')."""

    schema: JsonSchemaFormat
    """Schema definition (currently only jsonSchemaFormat)."""

    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """Timestamp when this schema version was created."""

    description: str | None = None
    """Human-readable description."""

    metadata: dict[str, Any] | None = None
    """Optional metadata (license, tags, etc.)."""

    atdata_schema_version: int = 1
    """Format version discriminator for the schema record structure itself."""

    def to_record(self) -> dict[str, Any]:
        """Serialize to ATProto record dict."""
        d: dict[str, Any] = {
            "$type": f"{LEXICON_NAMESPACE}.schema",
            "name": self.name,
            "version": self.version,
            "schemaType": self.schema_type,
            "schema": self.schema.to_record(),
            "createdAt": self.created_at.isoformat(),
            "atdataSchemaVersion": self.atdata_schema_version,
        }
        if self.description is not None:
            d["description"] = self.description
        if self.metadata is not None:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_record(cls, d: dict[str, Any]) -> LexSchemaRecord:
        """Deserialize from ATProto record dict."""
        return cls(
            name=d["name"],
            version=d["version"],
            schema_type=d["schemaType"],
            schema=JsonSchemaFormat.from_record(d["schema"]),
            created_at=datetime.fromisoformat(d["createdAt"]),
            description=d.get("description"),
            metadata=d.get("metadata"),
            atdata_schema_version=_get_schema_version(d),
        )


# ---------------------------------------------------------------------------
# Dataset record
# ---------------------------------------------------------------------------


@dataclass
class LexDatasetEntry:
    """Dataset index entry pointing to WebDataset storage.

    Mirrors ``science.alt.dataset.entry`` (main record).
    """

    name: str
    """Human-readable dataset name."""

    schema_ref: str
    """AT-URI reference to the schema record."""

    storage: StorageUnion
    """Storage location for dataset shards."""

    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """Timestamp when this record was created."""

    description: str | None = None
    """Human-readable description."""

    metadata: DatasetMetadata | None = None
    """Structured dataset metadata."""

    tags: list[str] | None = None
    """Searchable tags for discovery."""

    size: DatasetSize | None = None
    """Dataset size information."""

    license: str | None = None
    """SPDX license identifier or URL."""

    metadata_schema_ref: str | None = None
    """AT-URI reference to a schema record for content metadata."""

    content_metadata: dict[str, Any] | None = None
    """Dataset-level content metadata (e.g., instrument settings)."""

    manifests: list[ShardManifestRef] | None = None
    """Per-shard manifest metadata for query-based access."""

    def to_record(self) -> dict[str, Any]:
        """Serialize to ATProto record dict."""
        d: dict[str, Any] = {
            "$type": f"{LEXICON_NAMESPACE}.entry",
            "name": self.name,
            "schemaRef": self.schema_ref,
            "storage": self.storage.to_record(),
            "createdAt": self.created_at.isoformat(),
        }
        if self.description is not None:
            d["description"] = self.description
        if self.metadata is not None:
            d["metadata"] = self.metadata.to_record()
        if self.tags is not None:
            d["tags"] = self.tags
        if self.size is not None:
            d["size"] = self.size.to_record()
        if self.license is not None:
            d["license"] = self.license
        if self.metadata_schema_ref is not None:
            d["metadataSchemaRef"] = self.metadata_schema_ref
        if self.content_metadata is not None:
            d["contentMetadata"] = self.content_metadata
        if self.manifests is not None:
            d["manifests"] = [m.to_record() for m in self.manifests]
        return d

    @classmethod
    def from_record(cls, d: dict[str, Any]) -> LexDatasetEntry:
        """Deserialize from ATProto record dict."""
        size = None
        if "size" in d:
            size = DatasetSize.from_record(d["size"])

        decoded_meta = decode_metadata_raw(d.get("metadata"))
        metadata: DatasetMetadata | None = None
        if decoded_meta is not None:
            metadata = DatasetMetadata.from_dict(decoded_meta)

        raw_manifests = d.get("manifests")
        manifests: list[ShardManifestRef] | None = None
        if raw_manifests is not None:
            manifests = [ShardManifestRef.from_record(m) for m in raw_manifests]

        return cls(
            name=d["name"],
            schema_ref=d["schemaRef"],
            storage=storage_from_record(d["storage"]),
            created_at=datetime.fromisoformat(d["createdAt"]),
            description=d.get("description"),
            metadata=metadata,
            tags=d.get("tags"),
            size=size,
            license=d.get("license"),
            metadata_schema_ref=d.get("metadataSchemaRef"),
            content_metadata=d.get("contentMetadata"),
            manifests=manifests,
        )


# ---------------------------------------------------------------------------
# Lens record
# ---------------------------------------------------------------------------


@dataclass
class LexLensRecord:
    """Bidirectional transformation between two sample types.

    Mirrors ``science.alt.dataset.lens`` (main record).
    ``getter_code`` and ``putter_code`` are required per the lexicon.
    """

    name: str
    """Human-readable lens name."""

    source_schema: str
    """AT-URI reference to source schema."""

    target_schema: str
    """AT-URI reference to target schema."""

    getter_code: LexCodeReference
    """Code reference for getter function (Source -> Target)."""

    putter_code: LexCodeReference
    """Code reference for putter function (Target, Source -> Source)."""

    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """Timestamp when this lens was created."""

    description: str | None = None
    """What this transformation does."""

    language: str | None = None
    """Programming language (e.g., 'python')."""

    metadata: dict[str, Any] | None = None
    """Arbitrary metadata."""

    def to_record(self) -> dict[str, Any]:
        """Serialize to ATProto record dict."""
        d: dict[str, Any] = {
            "$type": f"{LEXICON_NAMESPACE}.lens",
            "name": self.name,
            "sourceSchema": self.source_schema,
            "targetSchema": self.target_schema,
            "getterCode": self.getter_code.to_record(),
            "putterCode": self.putter_code.to_record(),
            "createdAt": self.created_at.isoformat(),
        }
        if self.description is not None:
            d["description"] = self.description
        if self.language is not None:
            d["language"] = self.language
        if self.metadata is not None:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_record(cls, d: dict[str, Any]) -> LexLensRecord:
        """Deserialize from ATProto record dict."""
        return cls(
            name=d["name"],
            source_schema=d["sourceSchema"],
            target_schema=d["targetSchema"],
            getter_code=LexCodeReference.from_record(d["getterCode"]),
            putter_code=LexCodeReference.from_record(d["putterCode"]),
            created_at=datetime.fromisoformat(d["createdAt"]),
            description=d.get("description"),
            language=d.get("language"),
            metadata=d.get("metadata"),
        )


# ---------------------------------------------------------------------------
# Label record
# ---------------------------------------------------------------------------


@dataclass
class LexLabelRecord:
    """Named label pointing to a dataset record.

    Mirrors ``science.alt.dataset.label`` (main record).
    Multiple labels with the same name but different versions can coexist,
    enabling versioned references to immutable, CID-addressed dataset records.
    """

    name: str
    """User-facing label name, e.g. 'mnist'."""

    dataset_uri: str
    """AT-URI pointing to the dataset record."""

    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """Timestamp when this label was created."""

    version: str | None = None
    """Semver or free-form version string."""

    description: str | None = None
    """Optional description of this labeled version."""

    def to_record(self) -> dict[str, Any]:
        """Serialize to ATProto record dict."""
        d: dict[str, Any] = {
            "$type": f"{LEXICON_NAMESPACE}.label",
            "name": self.name,
            "datasetUri": self.dataset_uri,
            "createdAt": self.created_at.isoformat(),
        }
        if self.version is not None:
            d["version"] = self.version
        if self.description is not None:
            d["description"] = self.description
        return d

    @classmethod
    def from_record(cls, d: dict[str, Any]) -> LexLabelRecord:
        """Deserialize from ATProto record dict."""
        return cls(
            name=d["name"],
            dataset_uri=d["datasetUri"],
            created_at=datetime.fromisoformat(d["createdAt"]),
            version=d.get("version"),
            description=d.get("description"),
        )


__all__ = [
    "LEXICON_NAMESPACE",
    "ShardChecksum",
    "DatasetSize",
    "ShardManifestRef",
    "DatasetMetadata",
    "HttpShardEntry",
    "StorageHttp",
    "S3ShardEntry",
    "StorageS3",
    "BlobEntry",
    "StorageBlobs",
    "StorageUnion",
    "storage_from_record",
    "LexCodeReference",
    "JsonSchemaFormat",
    "LexSchemaRecord",
    "LexDatasetEntry",
    "LexLensRecord",
    "LexLabelRecord",
]
