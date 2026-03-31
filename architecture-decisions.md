---
title: "Key Architecture Decisions"
tags: ["architecture", "decisions"]
sources: []
contributors: ["maxine--futura"]
created: 2026-03-31
updated: 2026-03-31
---

# Key Architecture Decisions

Significant architectural choices made during atdata development, reconstructed from the chainlink issue history.

## 1. SQLite as Default Index Provider (issue #429)
**Decision:** Switch default from Redis to SQLite.  
**Why:** Zero external dependencies for basic usage. Redis and PostgreSQL remain as pluggable alternatives for production/multi-process scenarios.

## 2. Repository System over AtmosphereIndex (issues #424-428)
**Decision:** Consolidate Index + AtmosphereIndex into a single Index class with Repository-based prefix routing.  
**Why:** AtmosphereIndex was a parallel implementation with duplicated logic. Repository dataclass pairs a provider + data store, enabling mixed local/remote storage through prefix routing (e.g., \`atdata://\` vs \`at://\`).

## 3. Concrete Index over AbstractIndex Protocol (issues #868-873)
**Decision:** Deprecate AbstractIndex protocol, use Index class directly.  
**Why:** The protocol added indirection without benefit — there was only ever one implementation. Simplified type annotations and imports across the codebase.

## 4. WebDataset as Storage Format
**Decision:** Use WebDataset (tar-based sharding) for all dataset storage.  
**Why:** Streaming-friendly, works with local disk and cloud storage, proven at scale. Samples are serialized via msgpack within tar entries.

## 5. Lens Network for Type Transformations (core design)
**Decision:** Global singleton LensNetwork with bidirectional lenses.  
**Why:** Enables viewing any dataset through any compatible type schema. Well-behavedness laws (GetPut/PutGet/PutPut) ensure round-trip consistency. Lenses are registered via @lens decorator.

## 6. ATProto for Federation (core design)
**Decision:** Use ATProto (the protocol behind Bluesky) for dataset discovery and metadata.  
**Why:** Decentralized identity (DIDs), schema-first design (lexicons), existing infrastructure (PDS hosting). Datasets published as ATProto records, shards stored as blobs on PDS.

## 7. Lexicon Namespace Migration (issue #799, ongoing)
**Decision:** Rename from \`ac.foundation.dataset\` to \`science.alt.dataset\`.  
**Why:** Moving to a community-governed namespace. Consuming lexicons from separate atdata-lexicon repo rather than bundling.

## 8. NDArray v1.1 Annotations (issues #850-856)
**Decision:** Extend schema field types to carry dtype, shape, and dimensionNames metadata.  
**Why:** Enables richer dataset profiling and validation. Supports new serialization formats (safetensors, zarr) alongside the original numpy format.
