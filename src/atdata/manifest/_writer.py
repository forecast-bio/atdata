"""ManifestWriter for serializing ShardManifest to JSON + parquet files."""

from __future__ import annotations

import json
from pathlib import Path

from ._manifest import ShardManifest


class ManifestWriter:
    """Writes a ``ShardManifest`` to companion JSON and parquet files.

    Produces two files alongside each shard:

    - ``{base_path}.manifest.json`` -- header with metadata and aggregates
    - ``{base_path}.manifest.parquet`` -- per-sample metadata (columnar)

    Args:
        base_path: The shard path without the ``.tar`` extension.

    Examples:
        >>> writer = ManifestWriter("/data/shard-000000")
        >>> json_path, parquet_path = writer.write(manifest)
    """

    def __init__(self, base_path: str | Path) -> None:
        self._base_path = Path(base_path)

    @property
    def json_path(self) -> Path:
        """Path for the JSON header file."""
        return self._base_path.with_suffix(".manifest.json")

    @property
    def parquet_path(self) -> Path:
        """Path for the parquet per-sample file."""
        return self._base_path.with_suffix(".manifest.parquet")

    def write(self, manifest: ShardManifest) -> tuple[Path, Path]:
        """Write the manifest to JSON + parquet files.

        Args:
            manifest: The ``ShardManifest`` to serialize.

        Returns:
            Tuple of ``(json_path, parquet_path)``.
        """
        json_out = self.json_path
        parquet_out = self.parquet_path

        # Ensure parent directory exists
        json_out.parent.mkdir(parents=True, exist_ok=True)

        # Write JSON header + aggregates
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(manifest.header_dict(), f, indent=2)

        # Write per-sample parquet
        if not manifest.samples.empty:
            manifest.samples.to_parquet(
                parquet_out,
                engine="fastparquet",
                index=False,
            )
        else:
            # Write an empty parquet with no rows
            manifest.samples.to_parquet(
                parquet_out,
                engine="fastparquet",
                index=False,
            )

        return json_out, parquet_out
