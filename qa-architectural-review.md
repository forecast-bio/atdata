---
title: "QA Architectural Review — Findings and Implementation Plan"
tags: ["qa", "architecture", "refactoring"]
sources:
  - url: "https://github.com/forecast-bio/atdata/issues/84"
    title: ""
    accessed_at: "2026-04-01"
contributors: ["maxine--futura"]
created: 2026-04-01
updated: 2026-04-01
---

## Codebase Snapshot (2026-04-01)

| Metric | Value |
|--------|-------|
| Production code | 29,183 lines (src/atdata/) |
| Tests | 2,021 collected (1,911 active), pytest + Codecov CI |
| Modules | 40+ across 8 packages |
| Python | 3.12+ (generics, \`|\` union syntax) |
| TODOs/FIXMEs | **0** in production code |
| Hardcoded secrets | **0** |
| Bare \`except:\` | **0** (all use specific types or \`Exception\`) |

## Architectural Assessment

### Facade Pattern — Intentional, Not God Objects

The three largest classes are user-facing facades. Splitting them would hurt DX:

- **Index** (42 public methods, 1,878 lines) — facade over datasets, schemas, labels, lenses. Flat API is correct. Internal implementation needs delegation.
- **Dataset** (32 public methods, 1,557 lines) — follows pandas/HuggingFace pattern. Chainable, fluent API. Do not split.
- **Atmosphere** (30 public methods, 1,036 lines) — serves two audiences (casual + power user). Benefits from partial namespacing.

The fix is internal delegation and DRY cleanup, not API decomposition.

### Bug Discovered During Review

**Batched filter/map silently dropped:** \`ds.filter(fn).ordered(batch_size=32)\` ignores the filter. The batched paths in \`ordered()\` and \`shuffled()\` skip \`_post_wrap_stages()\` entirely. Only unbatched paths apply filter/map.

---

## Detailed Findings

### Quick Wins (Low Effort, High Value)

#### 1. Dataset._build_pipeline() — eliminate 4x duplication
**File:** \`dataset.py:1127-1219\`
\`ordered()\` and \`shuffled()\` each have unbatched/batched variants — 4 near-identical pipeline constructions. Extract \`_build_pipeline(shuffle_shards, shuffle_samples, batch_size)\`.

#### 2. DatasetMeta dataclass — parameter explosion
**Files:** \`_index.py:695\`, \`_index.py:874\`, \`atmosphere/records.py\`
\`insert_dataset\` (15 params), \`write_samples\` (11 params), \`DatasetPublisher.publish\` (10 params) share 6 metadata fields (name, description, tags, license, schema_ref, metadata). Bundle into \`DatasetMeta\` dataclass. Accept both flat kwargs and config object.

#### 3. AppView fallback pattern — narrow exception handling
**Files:** All 6 files in \`atmosphere/\`
13 sites use identical pattern: try AppView → \`except Exception:\` → log → fall back to client-side. Problems: catches KeyboardInterrupt/SystemExit, lazy import of _logging in each block (8x duplication), silent degradation.
**Fix:** Extract \`with_appview_fallback()\` helper, narrow to \`(httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException)\`.

#### 4. SQLite indexes for schema/label lookups
**File:** \`providers/_sqlite.py:32\`
Only \`idx_entries_name\` exists. Add composite indexes for schema \`(name, version)\` and labels \`(name, created_at)\`.

#### 5. DeprecationWarning → FutureWarning
18 sites use \`DeprecationWarning\` (suppressed by default in user code). Switch to \`FutureWarning\` for APIs being removed in v1.0.

### Medium Effort

#### 6. Index internal delegation to Repository
**Files:** \`_index.py:695-872\`, \`_index.py:874-1069\`, \`repository.py\`
\`insert_dataset\` (195 lines) and \`write_samples\` (200 lines) inline routing, store selection, size guards, blob extraction — all duplicated. Repository is a passive dataclass with zero methods.
**Fix:** Give Repository \`insert_dataset()\` and \`write_samples()\` methods. Index becomes ~20-line dispatcher.

#### 7. Redis label resolution — correctness bug
**File:** \`providers/_redis.py:159-181\`
\`get_label()\` without version returns last \`scan_iter\` result — effectively random. SQLite/Postgres both use \`ORDER BY created_at DESC\`.
**Fix:** Add \`created_at\` field to Redis labels, sort by timestamp. Old labels without timestamp sort last (no backfill needed).

#### 8. Redis N+1 query in iter_entries
**File:** \`providers/_redis.py:74-79\`
Each entry fetched with separate \`get_entry_by_cid()\`. N entries = N round-trips.
**Fix:** Use Redis pipeline for batch retrieval.

#### 9. _schema_codec field type mapping duplication
**File:** \`_schema_codec.py\`
\`_field_type_to_python()\` and \`_field_type_to_stub_str()\` duplicate type mapping. New field type must be added to both.
**Fix:** Single mapping table or \`FieldTypeMapper\`.

### Larger Effort

#### 10. Atmosphere namespacing
**File:** \`atmosphere/client.py\`
Split into \`RecordOps\`, \`BlobOps\`, \`XrpcClient\`. Common ops (list/search/auth) stay on root. Power-user methods get namespaced: \`atmo.records.create()\`, \`atmo.blobs.upload()\`, \`atmo.xrpc.query()\`.

#### 11. StubManager split
**File:** \`_stub_manager.py\`
Handles schema + lens stubs with duplicated naming/path/write logic. Split into \`SchemaStubManager\` / \`LensStubManager\` or extract shared \`StubWriter\` base.

### Cleanup

#### 12. Deprecated method removal schedule
18 \`DeprecationWarning\` sites. Document removal version (\`# Removal: v1.0\`).

#### 13. Thread-safety: Atmosphere lazy import
**File:** \`atmosphere/client.py\`
Module-level \`_atproto_client_class\` global with lazy init, no lock. Fix with \`threading.Lock\` double-check pattern.

#### 14. _AtmosphereBackend temporal coupling
**File:** \`repository.py\`
Every public method calls \`_ensure_loaders()\`. Switch to eager initialization in \`__init__\`.

---

## Implementation Plan

### Phase Summary

| Phase | Items | Risk | Effort | Dependencies |
|-------|-------|------|--------|--------------|
| 1. Dataset Pipeline | #1 + bug fix | Low | Small | None |
| 2. Parameter Objects | #2 | Low | Small | None |
| 3. Internal Delegation | #3, #6, #14 | Medium | Medium | Phase 2 |
| 4. Provider Fixes | #4, #7, #8 | Low | Small | None |
| 5. Atmosphere + Cleanup | #5, #10-13 | Medium | Large | Phase 3 |

Phases 1, 2, 4 are independent and can run in parallel.

### Design Decisions (resolved)

1. **Pipeline bug fix:** Apply filter/map before batching (not after with batch-aware wrappers). Matches user intent, no SampleBatch internals dependency.
2. **Atmosphere namespacing:** Common ops stay on root. Only power-user methods (records/blobs/xrpc) get namespaced.
3. **DatasetMeta in atmosphere:** Yes — DatasetPublisher.publish* accepts DatasetMeta internally.
4. **Redis migration:** No backfill. Old labels without \`created_at\` sort last naturally.

### Non-Issues (for future reviewers)

- **Index 42 public methods** — facade over 4 domains. Flat API is correct DX. Fix internals, not surface.
- **Dataset 32 public methods** — pandas/HuggingFace pattern. Do not split.
- **Global singletons** (LensNetwork, _default_index, _type_cache, _logger) — all intentional.
- **\`except Exception\` in CLI** (15 sites) — CLI is system boundary, broad catch is correct.

---

## Tracking

- **GH Issue:** https://github.com/forecast-bio/atdata/issues/84
- **Design Doc:** \`.design/084-architectural-refactoring.md\`
- **Phase 1 Agent:** Launched 2026-04-01, session \`M2KE-5GVB-extract-dataset-build-pipeline-to-eliminate-4x-af25\`
