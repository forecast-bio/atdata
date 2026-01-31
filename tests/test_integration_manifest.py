"""Integration tests for the manifest system.

Tests end-to-end manifest generation during shard writing,
round-trip loading, and query execution over real WebDataset files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import numpy as np
import webdataset as wds
from numpy.typing import NDArray

import atdata
from atdata.manifest import (
    ManifestBuilder,
    ManifestField,
    ManifestWriter,
    QueryExecutor,
    ShardManifest,
)


# =============================================================================
# Test Sample Types
# =============================================================================


@atdata.packable
class ImageClassSample:
    """Test sample mimicking an image classification dataset."""

    image: NDArray
    label: Annotated[str, ManifestField("categorical")]
    confidence: Annotated[float, ManifestField("numeric")]
    tags: Annotated[list[str], ManifestField("set")]


# =============================================================================
# End-to-End Tests
# =============================================================================


class TestManifestWriteDuringShardCreation:
    """Test manifest generation alongside shard writing using TarWriter."""

    def _write_shards_with_manifests(
        self, tmp_path: Path, samples_per_shard: int = 5
    ) -> tuple[list[Path], list[Path], list[Path]]:
        """Write shards with manifests and return file paths.

        Returns:
            Tuple of (tar_paths, json_paths, parquet_paths).
        """
        all_samples = [
            ImageClassSample(
                image=np.random.randn(8, 8).astype(np.float32),
                label=["dog", "cat", "bird"][i % 3],
                confidence=0.5 + (i % 10) * 0.05,
                tags=[["outdoor", "day"], ["indoor"], ["outdoor", "night"]][i % 3],
            )
            for i in range(15)
        ]

        tar_paths: list[Path] = []
        json_paths: list[Path] = []
        parquet_paths: list[Path] = []

        # Write in chunks of samples_per_shard
        for shard_idx in range(0, len(all_samples), samples_per_shard):
            chunk = all_samples[shard_idx : shard_idx + samples_per_shard]
            shard_name = f"data-{shard_idx // samples_per_shard:06d}"
            tar_path = tmp_path / f"{shard_name}.tar"

            builder = ManifestBuilder(
                sample_type=ImageClassSample,
                shard_id=str(tmp_path / shard_name),
                schema_version="1.0.0",
            )

            offset = 0
            with wds.writer.TarWriter(str(tar_path)) as writer:
                for sample in chunk:
                    wds_dict = sample.as_wds
                    writer.write(wds_dict)
                    packed_size = len(wds_dict.get("msgpack", b""))
                    builder.add_sample(
                        key=wds_dict["__key__"],
                        offset=offset,
                        size=packed_size,
                        sample=sample,
                    )
                    offset += 512 + packed_size + (512 - packed_size % 512) % 512

            manifest = builder.build()
            manifest_writer = ManifestWriter(tmp_path / shard_name)
            json_path, parquet_path = manifest_writer.write(manifest)

            tar_paths.append(tar_path)
            json_paths.append(json_path)
            parquet_paths.append(parquet_path)

        return tar_paths, json_paths, parquet_paths

    def test_manifest_files_created(self, tmp_path: Path):
        tar_paths, json_paths, parquet_paths = self._write_shards_with_manifests(
            tmp_path
        )

        assert len(tar_paths) == 3
        assert len(json_paths) == 3
        assert len(parquet_paths) == 3

        for jp in json_paths:
            assert jp.exists()
            assert jp.suffix == ".json"
        for pp in parquet_paths:
            assert pp.exists()
            assert pp.suffix == ".parquet"

    def test_manifest_content_matches_data(self, tmp_path: Path):
        tar_paths, json_paths, parquet_paths = self._write_shards_with_manifests(
            tmp_path
        )

        manifest = ShardManifest.from_files(json_paths[0], parquet_paths[0])
        assert manifest.num_samples == 5
        assert manifest.schema_type == "ImageClassSample"
        assert manifest.schema_version == "1.0.0"

        # Aggregates should reflect the first 5 samples
        assert "label" in manifest.aggregates
        assert manifest.aggregates["label"]["type"] == "categorical"
        assert "confidence" in manifest.aggregates
        assert manifest.aggregates["confidence"]["type"] == "numeric"
        assert "tags" in manifest.aggregates
        assert manifest.aggregates["tags"]["type"] == "set"

        # Per-sample data
        assert len(manifest.samples) == 5
        assert "__key__" in manifest.samples.columns
        assert "__offset__" in manifest.samples.columns
        assert "label" in manifest.samples.columns
        assert "confidence" in manifest.samples.columns

    def test_all_shards_covered(self, tmp_path: Path):
        _, json_paths, parquet_paths = self._write_shards_with_manifests(tmp_path)

        total_samples = 0
        for jp, pp in zip(json_paths, parquet_paths):
            m = ShardManifest.from_files(jp, pp)
            total_samples += m.num_samples

        assert total_samples == 15


class TestQueryOverWrittenDataset:
    """Test queries over manifests generated from real data."""

    def _setup_dataset_with_manifests(self, tmp_path: Path) -> list[ShardManifest]:
        """Create a multi-shard dataset with manifests."""
        samples = [
            ImageClassSample(
                image=np.random.randn(4, 4).astype(np.float32),
                label=label,
                confidence=conf,
                tags=tags,
            )
            for label, conf, tags in [
                ("dog", 0.95, ["outdoor", "day"]),
                ("cat", 0.40, ["indoor"]),
                ("dog", 0.85, ["outdoor", "night"]),
                ("bird", 0.60, ["outdoor"]),
                ("cat", 0.92, ["indoor", "night"]),
                ("dog", 0.30, ["outdoor", "day"]),
            ]
        ]

        manifests: list[ShardManifest] = []
        # Write 2 shards of 3 samples each
        for shard_idx in range(2):
            chunk = samples[shard_idx * 3 : (shard_idx + 1) * 3]
            shard_name = f"query-test-{shard_idx:06d}"
            tar_path = tmp_path / f"{shard_name}.tar"

            builder = ManifestBuilder(
                sample_type=ImageClassSample,
                shard_id=str(tmp_path / shard_name),
            )

            offset = 0
            with wds.writer.TarWriter(str(tar_path)) as writer:
                for sample in chunk:
                    wds_dict = sample.as_wds
                    writer.write(wds_dict)
                    packed_size = len(wds_dict.get("msgpack", b""))
                    builder.add_sample(
                        key=wds_dict["__key__"],
                        offset=offset,
                        size=packed_size,
                        sample=sample,
                    )
                    offset += 512 + packed_size + (512 - packed_size % 512) % 512

            manifest = builder.build()
            manifest_writer = ManifestWriter(tmp_path / shard_name)
            manifest_writer.write(manifest)
            manifests.append(manifest)

        return manifests

    def test_query_high_confidence(self, tmp_path: Path):
        manifests = self._setup_dataset_with_manifests(tmp_path)
        executor = QueryExecutor(manifests)

        results = executor.query(where=lambda df: df["confidence"] > 0.80)
        # Should match: dog@0.95, dog@0.85, cat@0.92
        assert len(results) == 3
        labels = []
        for r in results:
            row = None
            for m in manifests:
                match = m.samples[m.samples["__key__"] == r.key]
                if len(match) > 0:
                    row = match.iloc[0]
                    break
            assert row is not None
            labels.append(row["label"])
        assert sorted(labels) == ["cat", "dog", "dog"]

    def test_query_specific_label(self, tmp_path: Path):
        manifests = self._setup_dataset_with_manifests(tmp_path)
        executor = QueryExecutor(manifests)

        results = executor.query(where=lambda df: df["label"] == "bird")
        assert len(results) == 1

    def test_query_combined(self, tmp_path: Path):
        manifests = self._setup_dataset_with_manifests(tmp_path)
        executor = QueryExecutor(manifests)

        results = executor.query(
            where=lambda df: (df["label"] == "dog") & (df["confidence"] >= 0.85)
        )
        # dog@0.95 and dog@0.85
        assert len(results) == 2

    def test_query_from_directory(self, tmp_path: Path):
        self._setup_dataset_with_manifests(tmp_path)

        executor = QueryExecutor.from_directory(tmp_path)
        results = executor.query(where=lambda df: df["confidence"] > 0.90)
        # dog@0.95, cat@0.92
        assert len(results) == 2

    def test_query_no_results(self, tmp_path: Path):
        manifests = self._setup_dataset_with_manifests(tmp_path)
        executor = QueryExecutor(manifests)

        results = executor.query(where=lambda df: df["confidence"] > 0.99)
        assert len(results) == 0
