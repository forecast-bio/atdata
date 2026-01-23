# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

## [0.2.1a1] - 2026-01-20

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
- Review and address human-review.md feedback (#344)
- Fix load_dataset overloads and AbstractIndex compatibility (#348)
- Connect load_dataset to index data_store for S3 credentials (#361)
- Fix load_dataset overload return types for DictSample (#360)
- Add data_store to AbstractIndex protocol (#359)
- Audit and fix xs/list_xs naming convention (#347)
- Fix AtmosphereIndex: list_datasets/list_schemas return types (#357)
- Refactor DataSource/Dataset: shards()/shard_list -> shards/list_shards() (#356)
- Refactor local.py: entries/all_entries -> entries/list_entries (#355)
- Update AbstractIndex protocol to match new naming convention (#358)
- Investigate Redis index entry removal issue (#346)
- Implement 'atdata diagnose' command for Redis health check (#354)
- Implement 'atdata local up' command to run Redis + MinIO (#353)
- Create atdata.cli module with entry point (#352)
- Evaluate PackableSample → Packable protocol migration (#345)
- Update publish_schema and other signatures to use Packable protocol (#351)
- Update @packable decorator return type annotation (#350)
- Define Packable protocol in _protocols.py (#349)
- Review and update README for v0.2.2 release (#343)
- Streamline Dataset API with DictSample default type (#338)
- Add tests for DictSample and new API (#342)
- Update load_dataset default type to DictSample (#341)
- Update @packable to auto-register DictSample lens (#340)
- Implement DictSample class with __getattr__ and __getitem__ (#339)
- Fix failing tests in test_integration_error_handling.py (#337)
- v0.2.2 beta release improvements (#326)
- Document to_parquet() memory usage (#336)
- Evaluate splitting local.py into modules (#335)
- Add error path tests (timeouts, partial failures) (#334)
- Add deployment guide to docs (#333)
- Add troubleshooting/FAQ section to docs (#332)
- Document __orig_class__ assumption in Dataset docstring (#331)
- Centralize tar creation helper in test fixtures (#330)
- Consolidate duplicate test sample types to conftest.py (#329)
- Document expected filterwarnings in test suite (#328)
- Complete truncated atmosphere.qmd documentation (#327)
- Comprehensive v0.2.2 beta release review (#321)
- Compile findings into .review/comprehensive-review.md (#325)
- Review documentation website and examples (#324)
- Review test suite coverage and quality (#323)
- Review core codebase architecture and code quality (#322)
- Human Review: Local Workflow API Improvements (#274)
- Update documentation and examples for current codebase (#316)
- Update README.md with current API (#320)
- Update examples/*.py files for current API (#319)
- Update reference/protocols.qmd with DataSource protocol (#318)
- Update reference/datasets.qmd for DataSource API (#317)
- Adversarial review: Post-DataSource refactor assessment (#307)
- Clean up unused TypeAlias definitions in dataset.py (#315)
- Remove verbose docstrings that restate function signatures (#314)
- Consolidate schema reference parsing logic in local.py (#313)
- Add error tests for corrupted msgpack data in Dataset.wrap() (#312)
- Remove or implement skipped test_repo_insert_round_trip (#311)
- Fix bare exception handlers in _stub_manager.py and _cid.py (#310)
- Replace assertion with ValueError in lens.py input validation (#309)
- Replace assertions with ValueError in dataset.py msgpack validation (#308)
- Refactor Dataset to use DataSource abstraction (#299)
- Research WebDataset streaming alternatives beyond HTTP/S URLs (#298)
- Write tests for DataSource implementations (#306)
- Update load_dataset to use DataSource (#305)
- Update S3DataStore to create S3Source instances (#304)
- Refactor Dataset to accept DataSource | str (#303)
- Implement S3Source with boto3 streaming (#302)
- Implement URLSource in new _sources.py module (#301)
- Add DataSource protocol to _protocols.py (#300)
- Fix S3 mock fixture regionname typo in tests (#297)
- Human review feedback: API improvements from human-review-01 (#290)
- AbstractIndex: Protocol vs subclass causing linting errors (#296)
- load_dataset linting: no matching overloads error (#295)
- @atdata.lens linting: LocalTextSample not recognized as PackableSample subclass (#291)
- LocalDatasetEntry: underscore-prefixed attributes should be public (#294)
- Default batch_size should be None for Dataset.ordered/shuffled (#292)
- Improve SchemaNamespace typing for IDE support (#289)
- Schema namespace API: index.load_schema() + index.schemas.MyType (#288)
- Auto-typed get_schema/decode_schema return type (#287)
- Improve decode_schema typing for IDE support (#286)
- Fix stub filename collisions with authority-based namespacing (#285)
- Auto-generate stubs on schema access (#281)
- Add tests for auto-stub functionality (#284)
- Integrate auto-stub into Index class (#283)
- Add StubManager class for stub file management (#282)
- Improve decoded_type dynamic typing/signatures (#279)
- Document atdata URI specification (#280)
- Create proper SampleSchema Python type (#278)
- Fix @atdata.packable decorator class identity (#275)
- Fix @atdata.packable decorator class identity (#275)
- Fix @atdata.packable decorator class identity (#275)
- Improve index.publish_schema API (#276)
- Improve list_schemas API semantics (#277)
- Fix @atdata.packable decorator class identity (#275)
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
