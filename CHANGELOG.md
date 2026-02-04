# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

## [0.3.3b2] - 2026-02-04

### Testing
- **Coverage improvements** (92% → 94%): 61 new tests across atmosphere client (swap_commit, model conversion fallbacks), DatasetLoader (HTTP/S3 storage paths, `get_typed`, `to_dataset`, checksum validation), DatasetPublisher (`publish_with_s3`), Redis/Postgres provider label CRUD, Redis schema edge cases (bytes decoding, legacy format), and lexicon loading/validation

### Fixed
- **CI**: Use `cp -f` in bench workflow to avoid interactive prompt on file overwrite

## [0.3.3b1] - 2026-02-04

### Added
- **Dataset labels**: Named, versioned pointers to dataset records — separating identity (CID-addressed) from naming (mutable labels). `store_label()`, `get_label()`, `list_labels()`, `delete_label()` across all index providers (SQLite, Redis, PostgreSQL)
- **Atmosphere label records**: `LabelPublisher` and `LabelLoader` for publishing and resolving `ac.foundation.dataset.label` records on ATProto PDS, with `ac.foundation.dataset.resolveLabel` query lexicon
- **Label-aware `load_dataset()`**: Path resolution now tries label lookup before falling back to dataset name, enabling `load_dataset("@local/mnist")` to resolve through labels

### Changed
- **Git flow**: Adopted standard git flow branching model — `develop` as integration branch, `feature/*` from `develop`, `release/*` cut from `develop`. Updated `/release`, `/feature`, `/publish`, and `/featree` skills accordingly
- **Worktree chainlink sharing**: `/featree` now symlinks `.chainlink/issues.db` to the base clone's copy so all worktrees share a single authoritative issue database

## [0.3.2b3] - 2026-02-04

### Fixed
- **`Atmosphere.upload_blob()` TypeError**: The timeout heuristic passed `timeout=` to `Client.upload_blob()` which only accepts `(data: bytes)`. Switched to the namespace method `com.atproto.repo.upload_blob()` which forwards kwargs through to httpx

### Testing
- **ATProto SDK signature compatibility tests**: New `test_atproto_compat.py` with 7 tests that instantiate a real atproto `Client` (with `ClientRaw._invoke` patched) to validate method signatures without network I/O. Covers `upload_blob`, `create_record`, `list_records`, `get_record`, `delete_record`, and `export_session`

## [0.3.2b2] - 2026-02-03

### Added
- **Lexicon-mirror type system**: `StorageHttp`, `StorageS3`, `StorageBlobs`, `BlobEntry`, `ShardChecksum` dataclasses that mirror ATProto lexicon definitions, with `storage_from_record()` union deserializer
- **`ShardUploadResult`**: Typed return from `PDSBlobStore.write_shards()` carrying both AT URIs and blob ref dicts
- **Lexicon reference docs**: Auto-generated documentation page for the `ac.foundation.dataset.*` namespace
- **Example docs**: dataset-profiler, lens-graph, and query-cookbook with plots and interactive tabsets
- **Typed proxy DSL** for manifest queries (`foundation-ac #43`)

### Changed
- **`DatasetPublisher` refactored**: Extracted `_create_record()` helper, fixing a bug where `publish()` used `dataset.url` instead of `dataset.list_shards()` for multi-shard datasets
- **`PDSBlobStore.write_shards()`** returns `ShardUploadResult` instead of using a `_last_blob_refs` side-channel
- **Blob storage uploads**: PDS blob uploads now use `storageBlobs` with embedded blob ref objects instead of string AT URIs in `storageExternal`, preventing PDS garbage collection of uploaded blobs
- Replaced lexicon symlinks with real files
- Guarded `redis` imports behind `TYPE_CHECKING` in `index/_entry.py` and `index/_index.py`
- Standardized benchmark outputs to `.benchmarks/` directory

### Fixed
- `publish()` multi-shard bug: was passing single URL instead of full shard list
- Double-write eliminated in `PDSBlobStore`
- Lens-graph example: removed float rounding in calibrate lens that broke law assertions
- Unused imports and E402 violations in atmosphere module
- Unused variable and import in test files

### Testing
- Strengthened weak mock assertions with argument verification across 4 test files
- Fixed misleading unicode tests: real emoji (🌍🎉🚀) and CJK characters (日本語テスト, 中文测试, 한국어시험) instead of ASCII placeholders
- Exact shard count assertions instead of `>= 2` bounds
- Fixed self-referential assertion in `test_publish_schema`
- Removed unnecessary `isinstance` builtin patch
- Added content assertions for empty/corrupted shard recovery tests

## [0.3.2b1] - 2026-02-03

### Changed
- **`Index.write()` → `Index.write_samples()`**: Renamed with atmosphere-aware defaults — automatic PDS blob upload, 50 MB per-shard limit, 1 GB total dataset guard
  - New `force` flag bypasses PDS size limits for large datasets
  - New `copy` flag forces data transfer from private/remote sources to destination store
  - New `data_store` kwarg to override the default storage backend
- **`Index.insert_dataset()` overhaul**: Smart source routing for atmosphere targets
  - Local files auto-upload via `PDSBlobStore`
  - Remote HTTP/HTTPS URLs referenced as external storage (zero-copy)
  - Credentialed `S3Source` errors by default to prevent leaking private endpoints; pass `copy=True` to copy data to the destination store
- **PDS constants**: `PDS_BLOB_LIMIT_BYTES` (50 MB) and `PDS_TOTAL_DATASET_LIMIT_BYTES` (1 GB) in `atmosphere/store.py`; `PDSBlobStore.write_shards()` defaults to 50 MB shard size
- **CI overhaul**: Sequential Lint → Pilot → Matrix flow; codecov uploads once per run instead of per-matrix-cell; benchmarks split to separate workflow
- Lazy-import `pandas` and `requests` in `dataset.py` to reduce import time

### Fixed
- **Atmosphere blob uploads**: `Index.write_samples()` targeting atmosphere now uploads data as PDS blobs instead of publishing local temp file paths in the ATProto record

### Deprecated
- `Index.add_entry()` — use `Index.insert_dataset()` instead
- `Index.promote_entry()` and `Index.promote_dataset()` — use `Index.insert_dataset()` with an atmosphere-backed Index instead
- `URLSource.shard_list` and `S3Source.shard_list` properties — use `list_shards()` method instead

## [0.3.1b1] - 2026-02-03

### Added
- **Lexicon packaging**: ATProto lexicon JSON files bundled in `src/atdata/lexicons/` with `importlib.resources` access via `atdata.lexicons.get_lexicon()` and `list_lexicons()`
- **`DatasetDict` single-split proxy**: When a `DatasetDict` has one split, `.ordered()`, `.shuffled()`, `.list_shards()`, and other `Dataset` methods are proxied directly
- **`write_samples(manifest=True)`**: Opt-in manifest generation during sample writing for query-based access
- **Example documentation**: Five executable Quarto example docs covering typed pipelines, lens transforms, manifest queries, index workflows, and multi-split datasets
- Bounds checking in `bytes_to_array()` for truncated/corrupted input buffers

### Changed
- Production hardening: observability and checkpoint/resume (GH#39 5.1/5.2) (#590)
- Expand logging coverage across write/read/index/atmosphere paths (#593)
- Add checkpoint/resume and on_shard_error to process_shards (#592)
- Add log_operation context manager to _logging.py (#591)
- Add reference documentation for atdata's atproto lexicons (#589)
- Add version auto-suggest to /release and /publish skills (#588)
- Create /publish skill for post-merge release tagging and PyPI publish (#587)
- Fix wheel build: duplicate filename in ZIP archive rejected by PyPI (#586)
- Update /release skill to run ruff format --check before committing (#585)
- **`AtmosphereClient` → `Atmosphere`**: Renamed with factory classmethods `Atmosphere.login()` and `Atmosphere.from_env()`; `AtmosphereClient` remains as a deprecated alias
- **`sampleSchema` → `schema`**: Lexicon record type renamed from `ac.foundation.dataset.sampleSchema` to `ac.foundation.dataset.schema` (clean break, no backward compat)
- **Module reorganization**: `local/` split into `index/` (Index, entries, schema management) and `stores/` (LocalDiskStore, S3DataStore); `local/` remains as backward-compat re-export shim
- **CLI rename**: `atdata local` subcommand renamed to `atdata infra`
- **Uniform Repository model**: `Index` now treats `"local"` as a regular `Repository`, collapsing 3-way routing (local/named/atmosphere) to 2-way (repo/atmosphere)
- `SampleBatch` aggregation uses `np.stack()` instead of `np.array(list(...))` for efficiency
- Numpy scalar coercion in `_make_packable` — numpy scalars are now extracted to Python primitives before msgpack serialization
- Removed dead legacy aliases in `StubManager` (`_stub_filename`, `_stub_path`, `_stub_is_current`, `_write_stub_atomic`)
- Streamlined homepage and updated docs site to reflect new APIs

### Fixed
- Schema round-trip in `Index.write()` — schemas with NDArray fields now survive publish/decode correctly
- Test isolation: protocol tests now use temporary SQLite databases instead of shared default

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
