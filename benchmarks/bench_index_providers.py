"""Performance benchmarks for index provider operations.

Measures read/write latency and throughput for SQLite, Redis, and PostgreSQL
providers. Each benchmark skips gracefully if the backend is unavailable.
"""

from __future__ import annotations

import pytest

from atdata.local._entry import LocalDatasetEntry

from .conftest import BenchBasicSample, generate_basic_samples


# =============================================================================
# Helpers
# =============================================================================


def _make_entry(i: int) -> LocalDatasetEntry:
    return LocalDatasetEntry(
        name=f"bench_dataset_{i:06d}",
        schema_ref=f"atdata://local/sampleSchema/BenchBasicSample@1.0.0",
        data_urls=[f"/tmp/bench/data-{i:06d}.tar"],
        metadata={"index": i, "split": "train"},
    )


def _make_schema_json(name: str, version: str) -> str:
    return (
        f'{{"name": "{name}", "version": "{version}", '
        f'"fields": [{{"name": "x", "type": "int"}}]}}'
    )


def _prepopulate_entries(provider, n: int) -> list[LocalDatasetEntry]:
    entries = [_make_entry(i) for i in range(n)]
    for entry in entries:
        provider.store_entry(entry)
    return entries


def _prepopulate_schemas(provider, name: str, n: int) -> list[str]:
    versions = []
    for i in range(n):
        version = f"1.0.{i}"
        provider.store_schema(name, version, _make_schema_json(name, version))
        versions.append(version)
    return versions


# =============================================================================
# Write Benchmarks
# =============================================================================


@pytest.mark.bench_index
class TestProviderWriteBenchmarks:
    """Write operation benchmarks across all providers."""

    PARAM_LABELS = {"n": "entries to store", "any_provider": "storage backend"}

    def test_store_single_entry(self, benchmark, any_provider):
        entry = _make_entry(0)
        benchmark(any_provider.store_entry, entry)

    @pytest.mark.parametrize("n", [10, 100, 1000], ids=["10", "100", "1k"])
    def test_store_entries_bulk(self, benchmark, any_provider, n):
        entries = [_make_entry(i) for i in range(n)]

        def _store_all():
            for entry in entries:
                any_provider.store_entry(entry)

        benchmark(_store_all)

    def test_store_schema(self, benchmark, any_provider):
        benchmark(
            any_provider.store_schema,
            "BenchSample",
            "1.0.0",
            _make_schema_json("BenchSample", "1.0.0"),
        )

    @pytest.mark.parametrize("n", [10, 50], ids=["10v", "50v"])
    def test_store_schema_versions(self, benchmark, any_provider, n):
        def _store_versions():
            for i in range(n):
                v = f"1.0.{i}"
                any_provider.store_schema(
                    "BenchVersioned", v, _make_schema_json("BenchVersioned", v)
                )

        benchmark(_store_versions)


# =============================================================================
# Read Benchmarks
# =============================================================================


@pytest.mark.bench_index
class TestProviderReadBenchmarks:
    """Read operation benchmarks across all providers."""

    PARAM_LABELS = {"n": "entries in index", "any_provider": "storage backend"}

    def test_get_entry_by_name(self, benchmark, any_provider):
        entries = _prepopulate_entries(any_provider, 100)
        target = entries[50]
        benchmark(any_provider.get_entry_by_name, target.name)

    def test_get_entry_by_cid(self, benchmark, any_provider):
        entries = _prepopulate_entries(any_provider, 100)
        target = entries[50]
        benchmark(any_provider.get_entry_by_cid, target.cid)

    @pytest.mark.parametrize("n", [10, 100, 1000], ids=["10", "100", "1k"])
    def test_iter_entries(self, benchmark, any_provider, n):
        _prepopulate_entries(any_provider, n)
        benchmark(lambda: list(any_provider.iter_entries()))

    def test_get_schema_json(self, benchmark, any_provider):
        any_provider.store_schema(
            "BenchRead", "1.0.0", _make_schema_json("BenchRead", "1.0.0")
        )
        benchmark(any_provider.get_schema_json, "BenchRead", "1.0.0")

    @pytest.mark.parametrize("n", [5, 20, 50], ids=["5v", "20v", "50v"])
    def test_find_latest_version(self, benchmark, any_provider, n):
        _prepopulate_schemas(any_provider, "BenchLatest", n)
        benchmark(any_provider.find_latest_version, "BenchLatest")

    def test_iter_schemas(self, benchmark, any_provider):
        _prepopulate_schemas(any_provider, "BenchIterSchema", 20)
        benchmark(lambda: list(any_provider.iter_schemas()))


# =============================================================================
# Index-Level Benchmarks
# =============================================================================


@pytest.mark.bench_index
class TestIndexBenchmarks:
    """Benchmarks through the full Index API."""

    def test_index_insert_dataset(self, benchmark, tmp_path, sqlite_provider):
        import atdata
        from atdata.local._index import Index

        samples = generate_basic_samples(10)
        tar_path = tmp_path / "idx-bench-000000.tar"
        from .conftest import write_tar

        write_tar(tar_path, samples)

        index = Index(provider=sqlite_provider, atmosphere=None)
        ds = atdata.Dataset[BenchBasicSample](url=str(tar_path))

        counter = [0]

        def _insert():
            name = f"bench_ds_{counter[0]:06d}"
            counter[0] += 1
            index.insert_dataset(ds, name=name)

        benchmark(_insert)

    def test_index_get_dataset(self, benchmark, tmp_path, sqlite_provider):
        import atdata
        from atdata.local._index import Index

        samples = generate_basic_samples(10)
        tar_path = tmp_path / "idx-get-000000.tar"
        from .conftest import write_tar

        write_tar(tar_path, samples)

        index = Index(provider=sqlite_provider, atmosphere=None)
        ds = atdata.Dataset[BenchBasicSample](url=str(tar_path))
        index.insert_dataset(ds, name="bench_lookup_target")

        benchmark(index.get_dataset, "bench_lookup_target")

    def test_index_list_datasets(self, benchmark, tmp_path, sqlite_provider):
        import atdata
        from atdata.local._index import Index

        samples = generate_basic_samples(5)
        tar_path = tmp_path / "idx-list-000000.tar"
        from .conftest import write_tar

        write_tar(tar_path, samples)

        index = Index(provider=sqlite_provider, atmosphere=None)
        ds = atdata.Dataset[BenchBasicSample](url=str(tar_path))
        for i in range(100):
            index.insert_dataset(ds, name=f"bench_list_{i:04d}")

        benchmark(index.list_datasets)

    def test_index_publish_schema(self, benchmark, sqlite_provider):
        from atdata.local._index import Index

        index = Index(provider=sqlite_provider, atmosphere=None)
        counter = [0]

        def _publish():
            v = f"1.0.{counter[0]}"
            counter[0] += 1
            index.publish_schema(BenchBasicSample, version=v)

        benchmark(_publish)
