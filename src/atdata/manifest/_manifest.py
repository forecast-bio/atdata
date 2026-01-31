"""ShardManifest data model.

Represents a loaded manifest with JSON header (metadata + aggregates)
and per-sample metadata (as a pandas DataFrame).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

MANIFEST_FORMAT_VERSION = "1.0.0"


@dataclass
class ShardManifest:
    """In-memory representation of a shard's manifest.

    Contains the JSON header (metadata + aggregates) and per-sample
    metadata stored as a pandas DataFrame for efficient columnar filtering.

    Attributes:
        shard_id: Shard identifier (path without extension).
        schema_type: Schema class name.
        schema_version: Schema version string.
        num_samples: Number of samples in the shard.
        size_bytes: Total shard size in bytes.
        created_at: When the manifest was created.
        aggregates: Dict of field name to aggregate summary dict.
        samples: DataFrame with ``__key__``, ``__offset__``, ``__size__``,
            and manifest field columns.
        provenance: Optional provenance metadata (job ID, parent shards, etc.).

    Examples:
        >>> manifest = ShardManifest.from_files(
        ...     "data/shard-000000.manifest.json",
        ...     "data/shard-000000.manifest.parquet",
        ... )
        >>> manifest.num_samples
        1000
        >>> manifest.aggregates["label"]["cardinality"]
        3
    """

    shard_id: str
    schema_type: str
    schema_version: str
    num_samples: int
    size_bytes: int
    created_at: datetime
    aggregates: dict[str, dict[str, Any]]
    samples: pd.DataFrame
    provenance: dict[str, Any] = field(default_factory=dict)

    def header_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable header including aggregates.

        Returns:
            Dict suitable for writing as the ``.manifest.json`` file.
        """
        header: dict[str, Any] = {
            "manifest_version": MANIFEST_FORMAT_VERSION,
            "shard_id": self.shard_id,
            "schema_type": self.schema_type,
            "schema_version": self.schema_version,
            "num_samples": self.num_samples,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at.isoformat(),
            "aggregates": self.aggregates,
        }
        if self.provenance:
            header["provenance"] = self.provenance
        return header

    @classmethod
    def from_files(
        cls, json_path: str | Path, parquet_path: str | Path
    ) -> ShardManifest:
        """Load a manifest from its JSON + parquet companion files.

        Args:
            json_path: Path to the ``.manifest.json`` file.
            parquet_path: Path to the ``.manifest.parquet`` file.

        Returns:
            A fully loaded ``ShardManifest``.

        Raises:
            FileNotFoundError: If either file does not exist.
            json.JSONDecodeError: If the JSON file is malformed.
        """
        json_path = Path(json_path)
        parquet_path = Path(parquet_path)

        with open(json_path, "r", encoding="utf-8") as f:
            header = json.load(f)

        samples = pd.read_parquet(parquet_path, engine="fastparquet")

        return cls(
            shard_id=header["shard_id"],
            schema_type=header["schema_type"],
            schema_version=header["schema_version"],
            num_samples=header["num_samples"],
            size_bytes=header["size_bytes"],
            created_at=datetime.fromisoformat(header["created_at"]),
            aggregates=header.get("aggregates", {}),
            samples=samples,
            provenance=header.get("provenance", {}),
        )

    @classmethod
    def from_json_only(cls, json_path: str | Path) -> ShardManifest:
        """Load header-only manifest for shard-level filtering.

        Loads just the JSON header without the parquet per-sample data.
        Useful for fast shard pruning via aggregates before loading
        the full parquet file.

        Args:
            json_path: Path to the ``.manifest.json`` file.

        Returns:
            A ``ShardManifest`` with an empty ``samples`` DataFrame.
        """
        json_path = Path(json_path)

        with open(json_path, "r", encoding="utf-8") as f:
            header = json.load(f)

        return cls(
            shard_id=header["shard_id"],
            schema_type=header["schema_type"],
            schema_version=header["schema_version"],
            num_samples=header["num_samples"],
            size_bytes=header["size_bytes"],
            created_at=datetime.fromisoformat(header["created_at"]),
            aggregates=header.get("aggregates", {}),
            samples=pd.DataFrame(),
            provenance=header.get("provenance", {}),
        )
