---
title: "Ecosystem coordination for atdata repos"
tags: ["design-doc"]
sources: []
contributors: ["maxine--futura"]
created: 2026-04-05
updated: 2026-04-05
---


## Design Specification

### Summary

Establish synchronized development and release infrastructure across the atdata ecosystem (atdata, atdata-rs, atdata-lexicon, atdata-app) using distributed capability declarations, shared test vectors, a three-tier XRPC routing model, and auto-generated compatibility dashboards. The goal is to enable coordinated cross-repo development without a fragile central manifest.

### Requirements

- REQ-1: Both SDKs (atdata, atdata-rs) must route XRPC calls through a harmonized three-tier model: generic AppView for unauthenticated ATProto reads, user PDS for authenticated operations, atdata AppView for custom `science.alt.dataset.*` endpoints.
- REQ-2: Both AppView URLs (generic and atdata-specific) must be configurable via constructor parameters and environment variables (`ATDATA_APPVIEW`, `ATDATA_GENERIC_APPVIEW`), with sensible production defaults.
- REQ-3: Lexicon type changes must be detectable via CI drift-detection tests in each consumer repo, comparing hand-coded types against the lexicon JSON schema.
- REQ-4: Behavioral semantics must be defined as language-agnostic test vectors in `atdata-lexicon`, runnable by both SDKs in their native test frameworks.
- REQ-5: Each consumer repo must declare its capabilities and lexicon pin in a local `.atdata-ecosystem.json`, not in a central manifest.
- REQ-6: Lexicon PRs must trigger cross-repo CI in all consumers via GitHub Actions `repository_dispatch`.
- REQ-7: Releases must follow lexicon-first sequencing: lexicon tags first (always additive), consumers update and release independently on their own schedule.
- REQ-8: A scheduled CI job in `atdata-lexicon` must aggregate consumer status into a generated compatibility dashboard.

### Acceptance Criteria

- [ ] AC-1: `atdata` Atmosphere client accepts `appview=` and reads `ATDATA_APPVIEW` for Tier 3 routing (already done). Generic AppView URL (`_APPVIEW_URL`) is configurable via `ATDATA_GENERIC_APPVIEW` env var instead of hardcoded `bsky.social`.
- [ ] AC-2: `atdata-rs` Atmosphere client routes authenticated ops through user PDS, unauthenticated reads through configurable generic AppView, and custom XRPC through configurable atdata AppView. Both URLs readable from env vars.
- [ ] AC-3: Each consumer repo (atdata, atdata-rs, atdata-app) has a test that parses `entry.json` (and other lexicon files) and asserts all properties are covered by the hand-coded types. Test fails if a new lexicon property is not implemented.
- [ ] AC-4: `atdata-lexicon` contains `test-vectors/shard-roundtrip/` with at least 3 vectors (simple, numpy, metadata samples). Both `atdata` and `atdata-rs` CI run these vectors and pass.
- [ ] AC-5: Each consumer repo contains `.atdata-ecosystem.json` declaring role, language, lexicon ref, and capabilities list.
- [ ] AC-6: `atdata-lexicon` has a GitHub Actions workflow that fires `repository_dispatch` to atdata, atdata-rs, and atdata-app on every PR to `develop` or `main`.
- [ ] AC-7: Consumer repos have a `compat-check` workflow triggered by `repository_dispatch` that syncs the dispatched lexicon ref and runs tests.
- [ ] AC-8: `atdata-lexicon` has a scheduled weekly workflow that fetches each consumer's `.atdata-ecosystem.json` and latest CI status, then publishes `ecosystem-status.json` as a workflow artifact.

### Architecture

### Three-tier XRPC routing model

Both SDKs implement the same routing logic:

**Tier 1 — Generic ATProto reads (unauthenticated):**
Calls: `com.atproto.repo.getRecord`, `listRecords`, `identity.resolveHandle`, `sync.getBlob`
Route: configurable generic AppView (default: `bsky.social`)
Configurable via: `ATDATA_GENERIC_APPVIEW` env var or constructor param
Use case: cross-account reads, public data, no login required

**Tier 2 — Authenticated ATProto operations (logged in):**
Calls: `com.atproto.repo.createRecord`, `deleteRecord`, plus all reads
Route: user's own PDS (handles auth, rate limiting, proxying)
Use case: writing records, authenticated reads of own data

**Tier 3 — atdata-specific XRPC (custom endpoints):**
Calls: all 16 `science.alt.dataset.*` endpoints (resolve, list, search, publish, etc.)
Route: configurable atdata AppView (default: `appview.foundation.ac`)
Configurable via: `ATDATA_APPVIEW` env var or constructor param
Use case: dataset search, schema resolution, server-validated writes

In `atdata` (Python), Tier 3 is already implemented with `with_appview_fallback()` — if the AppView is unavailable, operations fall back to Tier 2 (direct PDS calls). Tier 1 currently hardcodes `bsky.social` in `_APPVIEW_URL`; this needs to become configurable.

In `atdata-rs` (Rust), only Tier 2 exists today. Tier 1 and Tier 3 need to be added. The `appview_url` field and `ATDATA_APPVIEW` env var are already plumbed but unused for custom XRPC.

Relevant files:
- `src/atdata/atmosphere/client.py` — Python Atmosphere class (all three tiers)
- `src/atdata/atmosphere/_appview.py` — `with_appview_fallback()` helper
- `atdata-rs/atdata/src/atmosphere/client.rs` — Rust Atmosphere struct (Tier 2 only)

### Lexicon drift detection (hybrid approach)

Each consumer keeps hand-coded lexicon types (Python dataclasses, Rust serde structs, SQL mappings) for idiomatic language support. A CI drift-detection test in each repo verifies coverage:

```
Test logic:
1. Parse entry.json (and other lexicon files) from synced lexicons
2. Extract all property names from "properties" objects
3. Assert each property has a corresponding field in the hand-coded type
4. Fail the test if any property is missing
```

This catches "forgot to add `manifests` to the Rust struct" without requiring codegen. Hand-coded types stay idiomatic; drift is caught at PR time.

Relevant files:
- `src/atdata/atmosphere/_lexicon_types.py` — Python lexicon types
- `atdata-rs/atdata/src/atmosphere/lexicon.rs` — Rust lexicon types
- `atdata-app/src/atdata_app/database.py` — AppView column mappings

### Test vectors in atdata-lexicon

`atdata-lexicon` becomes the behavioral contract, not just the wire format contract. A `test-vectors/` directory contains language-agnostic test cases:

```
atdata-lexicon/
  lexicons/science/alt/dataset/*.json   # wire format contract
  test-vectors/
    shard-roundtrip/                    # write samples → read back → verify
      simple-100.json                   # 100 string samples, expected keys/values
      numpy-arrays.json                 # NDArray samples, expected shapes/dtypes
      metadata-samples.json             # samples with metadata dict
    schema-resolution/                  # resolve("@handle/Type@v") → expected record
    record-serialization/               # sample type → ATProto record JSON → verify
    ...
  ecosystem.json                        # (auto-generated, not hand-maintained)
```

Each vector is a self-contained JSON file specifying inputs and expected outputs. SDKs consume these in their native test frameworks:
- Python: `pytest.mark.parametrize` over globbed vector files
- Rust: `#[test_case]` or build-script-generated tests

New features start as a test vector (the spec), then get implemented in each SDK. The vector IS the specification.

### Distributed capability declarations

Each consumer repo has `.atdata-ecosystem.json`:

```json
{
  "role": "sdk",
  "language": "python",
  "lexicon_sync": { "method": "tarball", "ref": "v1.2.0" },
  "capabilities": [
    "shard_read_write",
    "schema_publish",
    "schema_resolve_xrpc",
    "dataset_search_xrpc",
    "lens_transforms",
    "manifest_queries",
    "load_dataset_hf_api"
  ]
}
```

Capabilities map to test vector directories. CI runs only the vectors matching declared capabilities. A failing vector falsifies the capability claim.

### Cross-repo CI dispatch

A GitHub Actions workflow in `atdata-lexicon` (`.github/workflows/dispatch-consumers.yml`) fires `repository_dispatch` to consumers on lexicon PRs:

```yaml
on:
  pull_request:
    branches: [develop, main]

jobs:
  dispatch:
    runs-on: ubuntu-latest
    steps:
      - uses: peter-evans/repository-dispatch@v3
        with:
          token: ${{ secrets.ECOSYSTEM_DISPATCH_TOKEN }}
          repository: forecast-bio/atdata
          event-type: lexicon-compat-check
          client-payload: '{"lexicon_ref": "${{ github.head_ref }}"}'
      # ... repeat for atdata-rs, atdata-app
```

Each consumer has a `compat-check.yml` workflow triggered by `repository_dispatch`:

```yaml
on:
  repository_dispatch:
    types: [lexicon-compat-check]

jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: just sync-lexicons ref=${{ github.event.client_payload.lexicon_ref }}
      - run: uv run pytest tests/
```

### Release sequencing

1. Lexicon releases first — always additive (new optional fields, new defs, new token values). Tags like `v1.2.0`.
2. Consumers pin a minimum lexicon version in `.atdata-ecosystem.json` and their sync mechanism (`ref=` in justfile, submodule commit, etc.).
3. Consumers release independently when their implementation of new lexicon features is ready and CI passes.
4. For AppView + SDK co-development (new XRPC endpoints), use the dev Railway instance (`ATDATA_APPVIEW=https://dev-appview.up.railway.app`) for integration testing before either side releases.

### Auto-generated ecosystem dashboard

A weekly scheduled workflow in `atdata-lexicon` aggregates ecosystem status:

1. Fetch `.atdata-ecosystem.json` from each consumer's default branch via `gh api`
2. Check each consumer's latest CI status via `gh run list`
3. Generate `ecosystem-status.json` as a workflow artifact
4. Optionally publish to `docs/` as a rendered status page

This replaces the hand-maintained central manifest with a verified, always-current view.

### Out of Scope

- Codegen from lexicon JSON to language-specific types (hand-coded types stay, drift detection catches mismatches)
- Monorepo or git submodule approach for the full ecosystem (repos stay independent)
- Automated release orchestration across repos (releases are manual, sequencing is by convention)
- AppView endpoint implementation in atdata-rs (Tier 3 plumbing, not full XRPC client — the Rust SDK adds the routing infrastructure, actual endpoints are tracked separately)
- Container-based dev AppView setup (the dev Railway instance is managed via `ops-atdata-app`, not this design)

