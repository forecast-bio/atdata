# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

## [0.3.0b2] - 2026-02-02

### Added
- **`LocalDiskStore`**: Local filesystem data store implementing `AbstractDataStore` protocol
  - Writes WebDataset shards to disk with `write_shards()`
  - Default root at `~/.atdata/data/`, configurable via constructor
- **`write_samples()`**: Module-level function to write samples directly to WebDataset tar files
  - Single tar or sharded output via `maxcount`/`maxsize` parameters
  - Returns typed `Dataset[ST]` wrapping the written files
- **`Index.write()`**: Write samples and create an index entry in one step
  - Combines `write_samples()` + `insert_dataset()` into a single call
  - Auto-creates `LocalDiskStore` when no data store is configured
- **`Index.promote_entry()` and `Index.promote_dataset()`**: Atmosphere promotion via Index
  - Promote locally-indexed datasets to ATProto without standalone functions
  - Schema deduplication and automatic publishing
- Top-level exports: `atdata.Index`, `atdata.LocalDiskStore`, `atdata.write_samples`
- `write()` method added to `AbstractIndex` protocol
- 38 new tests: `test_write_samples.py`, `test_disk_store.py`, `test_index_write.py`

### Changed
- Update CLAUDE.md to reflect recent additions and fix divergences (#540)
- Reorganize .planning/ directory for temporal clarity (#547)
- Investigate and fix code coverage reduction from recent changes (#541)
- Add tests for dataset.py uncovered edge cases (#546)
- Add tests for _index.py uncovered error/edge paths (#545)
- Add tests for testing.py list field, Optional field, and fixture paths (#544)
- Add tests for providers/_factory.py postgres and unknown provider paths (#543)
- Add tests for _helpers.py object dtype and legacy npy paths (#542)
- Create local skills for release, changelog, and adversarial review (#536)
- Create /changelog skill (#539)
- Update /ad skill — less aggressive docstring trimming (#538)
- Create /release skill (#537)
- Fix CI failures on v0.3.0b2 release (#535)
- Release v0.3.0b2 beta (#534)
- `promote.py` updated as backward-compat wrapper delegating to `Index.promote_entry()`
- Trimmed `_protocols.py` docstrings by 30% (487 → 343 lines)
- Trimmed verbose test docstrings across test suite (−173 lines)
- Strengthened weak test assertions (isinstance checks, tautological tests)
- Removed dead code: `parse_cid()` function and tests
- Added `@pytest.mark.filterwarnings` to tests exercising deprecated APIs

## [0.3.0b1] - 2026-01-31

### Added
- **Structured logging**: `atdata.configure_logging()` with pluggable logger protocol
- **Partial failure handling**: `PartialFailureError` and shard-level error handling in `Dataset.map()`
- **Testing utilities**: `atdata.testing` module with mock clients, fixtures, and helpers
- **Per-shard manifest and query system** (GH#35)
  - `ManifestBuilder`, `ManifestWriter`, `QueryExecutor`, `SampleLocation`
  - `ManifestField` annotation and `resolve_manifest_fields()`
  - Aggregate collectors (categorical, numeric, set)
  - Integrated into write path and `Dataset.query()`
- **Performance benchmark suite**: `bench_dataset_io`, `bench_index_providers`, `bench_query`, `bench_atmosphere`
  - HTML benchmark reports with CI integration
  - Median/IQR statistics with per-sample columns
- **SQLite/PostgreSQL index providers** (GH#42)
  - `SqliteIndexProvider`, `PostgresIndexProvider`, `RedisIndexProvider`
  - `IndexProvider` protocol in `_protocols.py`
  - SQLite as default provider (replacing Redis)
- **Developer experience improvements** (GH#38)
  - CLI: `atdata inspect`, `atdata preview`, `atdata schema show/diff`
  - `Dataset.head()`, `Dataset.__iter__`, `Dataset.__len__`, `Dataset.select()`
  - `Dataset.filter()`, `Dataset.map()`, `Dataset.describe()`
  - `Dataset.get(key)`, `Dataset.schema`, `Dataset.column_names`
  - `Dataset.to_pandas()`, `Dataset.to_dict()`
  - Custom exception hierarchy with actionable error messages
- **Consolidated Index with Repository system**
  - `Repository` dataclass and `_AtmosphereBackend`
  - Prefix routing for multi-backend index operations
  - Default `Index` singleton with `load_dataset` integration
  - `AtmosphereIndex` deprecated in favor of `Index(atmosphere=client)`
- Comprehensive test coverage: 1155 tests

### Changed
- Split `local.py` monolith into `local/` package (`_index.py`, `_entry.py`, `_schema.py`, `_s3.py`, `_repo_legacy.py`)
- Migrated CLI from argparse to typer
- Migrated type annotations from `PackableSample` to `Packable` protocol
- Multiple adversarial review passes with code quality improvements
- CI: fixed duplicate runs, scoped permissions, benchmark auto-commit

## [0.2.2b1] - 2026-01-28

### Added
- **Blob storage for atmosphere datasets**: Full support for storing dataset shards as ATProto blobs via PDS
  - `DatasetPublisher.publish_with_blobs()` for uploading shards as blobs
  - `DatasetLoader.get_blobs()` and `get_blob_urls()` for retrieval
  - `AtmosphereClient.upload_blob()` and `get_blob()` wrappers
- **HuggingFace-style API**: `load_dataset()` function with path resolution, split handling, and streaming support
  - WebDataset brace notation, glob patterns, local directories, remote URLs
  - `DatasetDict` class for multi-split datasets
  - `@handle/dataset` path resolution via atmosphere index
- **Protocol-based architecture**: Abstract protocols for backend interoperability
  - `IndexEntry`, `AbstractIndex`, `AbstractDataStore` protocols
  - Enables polymorphic code across local and atmosphere backends
- **Local to atmosphere promotion**: `promote_to_atmosphere()` workflow with schema deduplication
- **Quarto documentation site**: Tutorials, reference docs, and API reference at docs/
- **Comprehensive integration test suite**: 593 tests covering E2E flows, error handling, edge cases

### Changed
- **Architecture refactor**: `LocalIndex` + `S3DataStore` composable pattern
  - `LocalIndex` now accepts optional `data_store` parameter
  - `S3DataStore` implements `AbstractDataStore` for S3 operations
- **Deprecated `Repo` class**: Use `LocalIndex(data_store=S3DataStore(...))` instead
  - `Repo` remains as thin backwards-compatibility wrapper with deprecation warning
- Renamed `BasicIndexEntry` to `LocalDatasetEntry` with CID-based identity
- Added ATProto-compatible CID generation via libipld
- Performance improvements: cached `sample_type` property, precompiled regex patterns

### Fixed
- Dark theme styling for callouts and code blocks in Quarto docs
- Browser chrome color updates on dark/light mode toggle

## [0.2.0] - 2026-01-06

### Added
- Initial atmosphere module with ATProto integration
- Schema, dataset, and lens publishing to ATProto PDS
- `AtmosphereClient` for ATProto authentication and record management
- `AtmosphereIndex` for querying published datasets and schemas
- Dynamic sample type reconstruction from published schemas

### Changed
- Improved type hint coverage throughout codebase
- Enhanced error messages for common failure modes

## [0.1.0] - 2025-12-15

### Added
- Core `PackableSample` and `@packable` decorator for typed samples
- `Dataset[ST]` generic typed dataset with WebDataset backend
- `SampleBatch[DT]` with automatic attribute aggregation
- `Lens[S, V]` bidirectional transformations
- Local storage with Redis index and S3 data store
- WebDataset tar file reading and writing
- NumPy array serialization via msgpack
