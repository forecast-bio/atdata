---
title: "Ecosystem Coordination Work Plan"
tags: ["design-doc", "workplan"]
sources: []
contributors: ["maxine--futura"]
created: 2026-04-05
updated: 2026-04-05
---


## Design Specification

### design document

See `.design/ecosystem-coordination.md` in the atdata repo for the full design.
Also stored as crosslink knowledge page `ecosystem-coordination`.

### issue structure

- **#15** (parent): Add ecosystem coordination design document
  - **#16** (WP1, atdata-lexicon): Test vectors, CI dispatch, aggregator
  - **#17** (WP2, atdata, blocked by #16): Configurable AppView, drift test, vector runner
  - **#18** (WP3, atdata-rs, blocked by #16): Three-tier routing, drift test, vector runner
  - **#19** (WP4, atdata-app, blocked by #16): Drift test, ecosystem json, compat workflow

### repo locations

- atdata: ~/code/forecast/atdata (Python SDK, primary repo)
- atdata-lexicon: ~/code/forecast/atdata-lexicon (lexicon contract + test vectors)
- atdata-rs: ~/code/forecast/atdata-rs (Rust SDK)
- atdata-app: ~/code/forecast/atdata-app (AppView, Python/FastAPI)

### execution order

1. WP1 first (atdata-lexicon) — creates test vectors and CI infra
2. WP2, WP3, WP4 can proceed in parallel after WP1

### key design decisions (resolved)

1. **Q1**: Hybrid lexicon sync — hand-coded types + CI drift-detection tests
2. **Q2**: Three-tier XRPC routing — generic AppView (Tier 1), user PDS (Tier 2), atdata AppView (Tier 3). Both AppView URLs configurable via env vars (ATDATA_APPVIEW, ATDATA_GENERIC_APPVIEW) and constructor params.
3. **Q3**: Sequenced releases with gates — lexicon first, consumers release independently
4. **Q4**: Test vectors in atdata-lexicon as behavioral contract, both SDKs run them
5. **Q5**: Distributed capability declarations (.atdata-ecosystem.json in each repo) + auto-generated central dashboard

### three-tier routing model

- Tier 1 (Generic ATProto, unauthenticated): com.atproto.repo.getRecord/listRecords, identity.resolveHandle, sync.getBlob -> configurable generic AppView (default bsky.social)
- Tier 2 (Authenticated ATProto): createRecord, deleteRecord, own-account reads -> user's PDS
- Tier 3 (atdata custom XRPC): all 16 science.alt.dataset.* endpoints -> configurable atdata AppView (default appview.foundation.ac)

### test vector format

Each vector is a self-contained JSON file in atdata-lexicon/test-vectors/category/:

```json
{
  "name": "simple-100-samples",
  "category": "shard-roundtrip",
  "capability": "shard_read_write",
  "inputs": { "samples": [...], "schema": {...} },
  "expected": { "shard_count": 1, "sample_count": 100, "keys": [...], "field_checks": {...} }
}
```

### capability list (for .atdata-ecosystem.json)

- shard_read_write: Read/write WebDataset shards
- schema_publish: Publish schema records to ATProto
- schema_resolve_xrpc: Resolve schemas via AppView XRPC
- dataset_search_xrpc: Search datasets via AppView XRPC
- lens_transforms: Lens-based schema transformations
- manifest_queries: Query-based access via manifests
- load_dataset_hf_api: HuggingFace-style load_dataset()
- atmosphere_crud: Basic ATProto record CRUD
- label_resolve_xrpc: Resolve labels via AppView XRPC
- blob_resolve_xrpc: Resolve blob URLs via AppView XRPC

### xrpc call map (from codebase analysis)

### atdata (Python) — 16 custom endpoints routed to atdata AppView (Tier 3):

Queries: resolveSchema, listSchemas, getEntry, listEntries, resolveBlobs, resolveLabel, listLenses, searchLenses, searchDatasets, describeService
Procedures: publishSchema, publishDataset, publishLabel, publishLens, publishLensVerification

All use with_appview_fallback() — fall back to Tier 2 PDS calls if AppView unavailable.

### atdata-rs (Rust) — PDS only today (Tier 2):

Standard ATProto: createRecord, getRecord, listRecords, deleteRecord, getBlob, resolveHandle
No custom XRPC endpoints implemented yet. Has ATDATA_APPVIEW env plumbing but unused.

### atdata-app (AppView) — Implements the custom endpoints:

Python/FastAPI service. Lexicons consumed via git submodule. NSIDs hardcoded in database.py.

