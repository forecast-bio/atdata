"""ManifestBuilder for accumulating sample metadata during shard writes.

Creates one ``ManifestBuilder`` per shard. Call ``add_sample()`` for each
sample written, then ``build()`` to produce a finalized ``ShardManifest``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from ._aggregates import (
    create_aggregate,
    CategoricalAggregate,
    NumericAggregate,
    SetAggregate,
)
from ._fields import resolve_manifest_fields
from ._manifest import ShardManifest


@dataclass
class _SampleRow:
    """Internal per-sample metadata row."""

    key: str
    offset: int
    size: int
    fields: dict[str, Any]


class ManifestBuilder:
    """Accumulates sample metadata during shard writing.

    Extracts manifest-annotated fields from each sample, feeds running
    aggregate collectors, and accumulates per-sample metadata rows.
    Call ``build()`` after all samples are written to produce a
    ``ShardManifest``.

    Args:
        sample_type: The Packable type being written.
        shard_id: Identifier for this shard (e.g., path without extension).
        schema_version: Version string for the schema.
        source_job_id: Optional provenance job identifier.
        parent_shards: Optional list of input shard identifiers.
        pipeline_version: Optional pipeline version string.

    Examples:
        >>> builder = ManifestBuilder(
        ...     sample_type=ImageSample,
        ...     shard_id="train/shard-00042",
        ... )
        >>> builder.add_sample(key="abc", offset=0, size=1024, sample=my_sample)
        >>> manifest = builder.build()
        >>> manifest.num_samples
        1
    """

    def __init__(
        self,
        sample_type: type,
        shard_id: str,
        schema_version: str = "1.0.0",
        source_job_id: str | None = None,
        parent_shards: list[str] | None = None,
        pipeline_version: str | None = None,
    ) -> None:
        self._sample_type = sample_type
        self._shard_id = shard_id
        self._schema_version = schema_version
        self._source_job_id = source_job_id
        self._parent_shards = parent_shards or []
        self._pipeline_version = pipeline_version

        self._manifest_fields = resolve_manifest_fields(sample_type)
        self._aggregates: dict[
            str, CategoricalAggregate | NumericAggregate | SetAggregate
        ] = {
            name: create_aggregate(mf.aggregate)
            for name, mf in self._manifest_fields.items()
        }
        self._rows: list[_SampleRow] = []
        self._total_size: int = 0

    def add_sample(
        self,
        *,
        key: str,
        offset: int,
        size: int,
        sample: Any,
    ) -> None:
        """Record a sample's metadata.

        Extracts manifest-annotated fields from the sample, updates
        running aggregates, and appends a row to the internal list.

        Args:
            key: The WebDataset ``__key__`` for this sample.
            offset: Byte offset within the tar file.
            size: Size in bytes of this sample's tar entry.
            sample: The sample instance (dataclass with manifest fields).
        """
        field_values: dict[str, Any] = {}
        for name in self._manifest_fields:
            value = getattr(sample, name, None)
            if value is not None:
                self._aggregates[name].add(value)
                field_values[name] = value

        self._rows.append(
            _SampleRow(key=key, offset=offset, size=size, fields=field_values)
        )
        self._total_size += size

    def build(self) -> ShardManifest:
        """Finalize aggregates and produce a ``ShardManifest``.

        Returns:
            A complete ``ShardManifest`` with header, aggregates, and
            per-sample DataFrame.
        """
        # Build aggregates dict
        aggregates = {name: agg.to_dict() for name, agg in self._aggregates.items()}

        # Build per-sample DataFrame
        records: list[dict[str, Any]] = []
        for row in self._rows:
            record: dict[str, Any] = {
                "__key__": row.key,
                "__offset__": row.offset,
                "__size__": row.size,
            }
            record.update(row.fields)
            records.append(record)

        samples_df = pd.DataFrame(records) if records else pd.DataFrame()

        # Build provenance
        provenance: dict[str, Any] = {}
        if self._source_job_id:
            provenance["source_job_id"] = self._source_job_id
        if self._parent_shards:
            provenance["parent_shards"] = self._parent_shards
        if self._pipeline_version:
            provenance["pipeline_version"] = self._pipeline_version

        schema_name = self._sample_type.__name__

        return ShardManifest(
            shard_id=self._shard_id,
            schema_type=schema_name,
            schema_version=self._schema_version,
            num_samples=len(self._rows),
            size_bytes=self._total_size,
            created_at=datetime.now(timezone.utc),
            aggregates=aggregates,
            samples=samples_df,
            provenance=provenance,
        )
