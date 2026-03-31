---
title: "atdata Development History (chainlink issues 1-877)"
tags: ["history", "architecture", "migration"]
sources: []
contributors: ["maxine--futura"]
created: 2026-03-31
updated: 2026-03-31
---

# Development History

This page summarizes the development arc of atdata as tracked through 877 chainlink issues (the predecessor to crosslink). The original issue database is preserved in `.chainlink.bak/issues.db` for reference.

## Phase 1: Foundation (issues 1-100)
**Period:** Early development  
**Focus:** Local storage, testing infrastructure, initial ATProto design

Built the core local storage layer (`atdata.local`), test infrastructure (fixtures, parametrized tests, Redis/S3 cleanup), and initial lexicon design. Key milestones:
- Local disk and S3 data stores with index providers
- Test suite foundations (moto for S3, Redis fixtures)
- Lexicon design phases planned (schema definition, client library, AppView, codegen)
- CLAUDE.md established with project conventions

**22 closed, 78 archived**

## Phase 2: Core Dataset (issues 101-200)
**Period:** Dataset and lens system buildout  
**Focus:** PackableSample, Lens laws, schema codegen, documentation

Implemented the typed dataset system with lens transformations. Key work:
- Lens law tests (GetPut, PutGet, PutPut)
- Schema codec and stub generation for IDE support
- decode_schema() for AbstractIndex
- Documentation site (quartodoc + quarto)
- First live ATProto integration tests

**1 closed, 99 archived**

## Phase 3: Index & Storage (issues 201-400)
**Period:** Pluggable storage architecture  
**Focus:** Index providers, Repository system, CLI tools

Major architectural refactor introducing pluggable index providers and the Repository system:
- SQLite (default), Redis, and PostgreSQL index providers
- Repository dataclass pairing provider + data store with prefix routing
- Deprecated AtmosphereIndex in favor of unified Index
- CLI tools: `atdata inspect`, `atdata preview`, `atdata schema`, `atdata local`, `atdata diagnose`
- DataSource protocol for streaming shard data
- Consolidated duplicate logic across index/repository

**10 closed, 190 archived**

## Phase 4: Atmosphere & ATProto (issues 401-600)
**Period:** Federation and publishing  
**Focus:** ATProto record publishing, schema management, developer experience

Most active development phase (189 closed issues). Built the full atmosphere integration:
- Schema/dataset/lens publishers and loaders
- PDSBlobStore for shard storage on ATProto
- CID generation via libipld
- `@packable` decorator improvements
- Developer experience: feature branch CLI, chainlink integration
- Quartodoc documentation improvements (Examples section rendering)
- Blob reference validation in write_shards

**189 closed, 11 archived**

## Phase 5: Manifests & Query (issues 601-750)
**Period:** Query system and metadata  
**Focus:** Per-shard manifests, query execution, metadata consistency

Added the manifest and query system for accessing dataset subsets:
- ManifestField and QueryExecutor for per-shard metadata queries
- Fixed triplicated metadata dispatch logic (extracted helpers)
- Comprehensive atmosphere/records.py test coverage
- Duplicate import cleanup and test assertion tightening
- Metadata decoding consolidation across records.py and __init__.py

**149 closed, 1 archived**

## Phase 6: AppView & Lexicons (issues 751-877)
**Period:** Current — federation features and lexicon evolution  
**Focus:** Cross-account resolution, AppView, array formats, verification

Most recent work before the crosslink migration:
- NDArray v1.1.0 annotations (dtype, shape, dimensionNames)
- New array format serialization (safetensors, zarr support in schema)
- Lens verification and schema compatibility checks
- AbstractIndex deprecation (replaced by concrete Index)
- Lexicon namespace migration: `ac.foundation.dataset` → `science.alt.dataset`
- v0.7.0b1 release

**111 closed, 0 archived, 16 open (migrated to crosslink as 10 deduplicated issues)**

## Open Work Streams (migrated to crosslink)

The 16 open chainlink issues were deduplicated to 10 crosslink issues:

| Crosslink | Title | Priority | Origin |
|-----------|-------|----------|--------|
| #2 | Support @handle/TypeName@version in get_schema | high | #763,777,778,779 |
| #3 | Fix get_schema cross-account resolution | high | #769 |
| #4 | Add manifest property to dataset record lexicon | high | #771,772 |
| #5 | Rename atdataSchemaVersion remove $ prefix | medium | #784,785 |
| #6 | Consume lexicons from atdata-lexicon repo | high | #799 |
| #7 | Switch CI Redis to ghcr.io | medium | #813 |
| #8 | Client-side AppView integration | medium | #815 |
| #9 | Write tests for AppView integration | high | #823 |
| #10 | Unified search API with pluggable backends | medium | #832 |
| #11 | Lens lexicon E2E validation | medium | #841 |
