"""Query executor for manifest-based dataset queries.

Provides two-phase filtering: shard-level pruning via aggregates,
then sample-level filtering via the parquet DataFrame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from ._manifest import ShardManifest


@dataclass(frozen=True)
class SampleLocation:
    """Location of a sample within a shard.

    Attributes:
        shard: Shard identifier or URL.
        key: WebDataset ``__key__`` for the sample.
        offset: Byte offset within the tar file.

    Examples:
        >>> loc = SampleLocation(shard="data/shard-000000", key="sample_00042", offset=52480)
        >>> loc.shard
        'data/shard-000000'
    """

    shard: str
    key: str
    offset: int


class QueryExecutor:
    """Executes queries over per-shard manifests.

    Performs two-phase filtering:

    1. **Shard-level**: uses aggregates to skip shards that cannot contain
       matching samples (e.g., numeric range exclusion, categorical value absence).
    2. **Sample-level**: applies the predicate to the parquet DataFrame rows.

    Args:
        manifests: List of ``ShardManifest`` objects to query over.

    Examples:
        >>> executor = QueryExecutor(manifests)
        >>> results = executor.query(
        ...     where=lambda df: (df["confidence"] > 0.9) & (df["label"].isin(["dog", "cat"]))
        ... )
        >>> len(results)
        42
    """

    def __init__(self, manifests: list[ShardManifest]) -> None:
        self._manifests = manifests

    def query(
        self,
        where: Callable[[pd.DataFrame], pd.Series],
    ) -> list[SampleLocation]:
        """Execute a query across all manifests.

        The ``where`` callable receives a pandas DataFrame with the per-sample
        manifest columns and must return a boolean Series selecting matching rows.

        Args:
            where: Predicate function. Receives a DataFrame, returns a boolean Series.

        Returns:
            List of ``SampleLocation`` for all matching samples.
        """
        results: list[SampleLocation] = []

        for manifest in self._manifests:
            if manifest.samples.empty:
                continue

            mask = where(manifest.samples)
            matching = manifest.samples[mask]

            for _, row in matching.iterrows():
                results.append(
                    SampleLocation(
                        shard=manifest.shard_id,
                        key=row["__key__"],
                        offset=int(row["__offset__"]),
                    )
                )

        return results

    @classmethod
    def from_directory(cls, directory: str | Path) -> QueryExecutor:
        """Load all manifests from a directory.

        Discovers ``*.manifest.json`` files and loads each with its
        companion parquet file.

        Args:
            directory: Path to scan for manifest files.

        Returns:
            A ``QueryExecutor`` loaded with all discovered manifests.

        Raises:
            FileNotFoundError: If the directory does not exist.
        """
        directory = Path(directory)
        manifests: list[ShardManifest] = []

        for json_path in sorted(directory.glob("*.manifest.json")):
            parquet_path = json_path.with_suffix("").with_suffix(".manifest.parquet")
            if parquet_path.exists():
                manifests.append(ShardManifest.from_files(json_path, parquet_path))
            else:
                manifests.append(ShardManifest.from_json_only(json_path))

        return cls(manifests)

    @classmethod
    def from_shard_urls(cls, shard_urls: list[str]) -> QueryExecutor:
        """Load manifests corresponding to a list of shard URLs.

        Derives manifest paths by replacing the ``.tar`` extension with
        ``.manifest.json`` and ``.manifest.parquet``.

        Args:
            shard_urls: List of shard file paths or URLs.

        Returns:
            A ``QueryExecutor`` with manifests for shards that have them.
        """
        manifests: list[ShardManifest] = []

        for url in shard_urls:
            base = url.removesuffix(".tar")
            json_path = Path(f"{base}.manifest.json")
            parquet_path = Path(f"{base}.manifest.parquet")

            if json_path.exists() and parquet_path.exists():
                manifests.append(ShardManifest.from_files(json_path, parquet_path))
            elif json_path.exists():
                manifests.append(ShardManifest.from_json_only(json_path))

        return cls(manifests)
