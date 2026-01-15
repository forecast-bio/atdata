# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added

### Fixed

### Changed
- Phase 2: Refactor local.py to use new protocols (#113)
- Add CID utilities module (_cid.py) with ATProto-compatible CID generation (#132)
- Rename BasicIndexEntry to LocalDatasetEntry with CID + name dual identity (#127)
- Add LocalIndex alias for Index class (#129)
- Update Repo.insert() to require name parameter (#130)
- Update test_local.py for new LocalDatasetEntry API (#131)
- Revise AbstractIndex: Remove single-type generic, add schema decoding (#123)
- Implement dynamic PackableSample class generation from schema (#126)
- Add decode_schema() method to AbstractIndex (#125)
- Remove generic type parameter from AbstractIndex (#124)
- Phase 1: Define Abstract Protocols (_protocols.py) (#112)
- Export protocols from __init__.py (#122)
- Define AbstractDataStore protocol (#121)
- Define AbstractIndex protocol (#120)
- Define IndexEntry protocol (#119)
- Review ATProto vs Local integration architecture convergence (#110)
- Add HuggingFace Datasets-style API to atdata (#103)
- Support streaming mode parameter (#108)
- Add split parameter handling (train/test/validation) (#107)
- Implement path/URL resolution and shard discovery (#106)
- Add DatasetDict class for multi-split datasets (#105)
- Implement load_dataset() entry point function (#104)
- Write test suite for _hf_api.py module (#109)
- Investigate test-bucket directory creation issue (#105)
- Add remaining Dataset edge case tests (#104)
- Improve test coverage for edge cases (#103)
- Phase 1: Lexicon Design & Schema Definition (#17)
