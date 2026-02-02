# ATProto Integration - Overview

## Vision

Transform `atdata` from a local/centralized dataset library into a **distributed dataset federation** built on AT Protocol. Datasets, schemas, and transformations become discoverable, versioned records on the ATProto network, enabling:

- **Decentralized dataset publishing**: Anyone can publish datasets without centralized infrastructure
- **Schema sharing & reuse**: Sample type definitions become reusable records with automatic code generation
- **Discoverable transformations**: Lens transformations are published as bidirectional mappings between schemas
- **Interoperability**: Different tools and languages can consume the same datasets using generated code
- **Versioning & provenance**: Immutable records provide audit trails for dataset evolution

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         AT Protocol Network                      │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐ │
│  │ Schema Records   │  │ Dataset Records  │  │ Lens Records  │ │
│  │ (Lexicon)        │  │ (Lexicon)        │  │ (Lexicon)     │ │
│  └──────────────────┘  └──────────────────┘  └───────────────┘ │
│         ▲                      ▲                     ▲          │
│         │                      │                     │          │
└─────────┼──────────────────────┼─────────────────────┼──────────┘
          │                      │                     │
          │ publish/query        │                     │
          │                      │                     │
    ┌─────┴──────────────────────┴─────────────────────┴─────┐
    │           Python Client Library (atdata)                │
    │                                                          │
    │  ┌────────────┐  ┌────────────┐  ┌──────────────────┐  │
    │  │ ATProto    │  │ Schema     │  │ Dataset          │  │
    │  │ Auth       │  │ Publisher  │  │ Loader           │  │
    │  └────────────┘  └────────────┘  └──────────────────┘  │
    │                                                          │
    │  Existing:                                               │
    │  - PackableSample, Dataset, Lens                         │
    │  - WebDataset integration                                │
    └──────────────────────────────────────────────────────────┘
                              │
                              │ queries (optional)
                              ▼
                    ┌─────────────────────┐
                    │   AppView Service   │
                    │  (Index Aggregator) │
                    │                     │
                    │ - Fast search       │
                    │ - Schema browser    │
                    │ - Metadata cache    │
                    └─────────────────────┘
```

## Core Concepts

### 1. Schema Records (PackableSample definitions)

Published ATProto records containing:
- Field names and types (with special handling for NDArray)
- Serialization metadata
- Version information
- Author/provenance

These become the **source of truth** for sample types across the network.

### 2. Dataset Index Records

Published ATProto records containing:
- Reference to schema record (the sample type)
- WebDataset URL(s) using brace notation (e.g., `s3://bucket/data-{000000..000099}.tar`)
- Msgpack-encoded metadata (arbitrary key-value pairs)
- Dataset description, tags, author

Users discover datasets by querying these records, then load them using existing `Dataset` class.

### 3. Lens Transformation Records

Published ATProto records containing:
- Source schema reference
- Target schema reference
- Transformation code (or reference to code)
- Bidirectional mapping metadata (getter/putter)

Enables building a **network of transformations** between schemas.

## Integration with Existing `atdata`

The ATProto integration is **additive**:

1. **Existing functionality unchanged**: `PackableSample`, `Dataset`, `Lens` continue to work as-is
2. **New methods added**:
   - `sample_type.publish_to_atproto(client)` - Publish schema
   - `dataset.publish_to_atproto(client)` - Publish index record
   - `Dataset.from_atproto(client, record_uri)` - Load from published record
   - `lens.publish_to_atproto(client)` - Publish transformation
3. **Optional AppView**: Query service for faster discovery (like Bluesky's AppView)

## Development Phases

### Phase 1: Lexicon Design (Issues #17, #22-25)
- Design three Lexicon schemas (sample, dataset, lens)
- Evaluate schema representation formats
- Create reference documentation

**Deliverable**: Lexicon JSON definitions ready for use

### Phase 2: Python Client Library (Issues #18, #26-31)
- ATProto SDK integration (auth, session management)
- Publishing implementations for all three record types
- Query/discovery functionality
- Extend `Dataset` class with `from_atproto()` method

**Deliverable**: Working Python library that can publish/load from ATProto

### Phase 3: AppView Service (Issues #19, #32-35)
- Optional aggregation service
- Firehose ingestion
- Search/query API
- Performance optimization

**Deliverable**: Hosted service for fast dataset discovery

### Phase 4: Code Generation (Issues #20, #36-39)
- Template system for Python codegen
- CLI tool for generating classes from schema records
- Type validation and compatibility checking

**Deliverable**: Tool to generate Python code from published schemas

### Phase 5: Integration & Testing (Issues #21, #40-43)
- End-to-end workflows and examples
- Integration test suite
- Documentation and guides
- Performance benchmarks

**Deliverable**: Production-ready feature with complete documentation

## Open Design Questions

### Schema Representation Format
**Question**: How should we represent `PackableSample` schemas in Lexicon records?

**Options**:
1. **JSON Schema** - Standard, well-supported, validation tools exist
2. **Protobuf** - Compact, has codegen ecosystem, good for cross-language
3. **Custom format** - Tailored to `PackableSample` specifics (NDArray handling, msgpack serialization)

**Considerations**:
- Need to represent `NDArray` types specially (dtype, shape constraints?)
- Should support future extensions (constraints, validation rules)
- Must be human-readable and machine-processable
- Codegen tooling needs to parse it

**Decision needed**: See Issue #25

### WebDataset Storage Location
**Question**: Should actual WebDataset `.tar` files be stored on ATProto, or just references to external storage?

**Current approach**: References only (S3, HTTP URLs, etc.)
- Pros: No storage limits, existing infrastructure works
- Cons: Centralization risk if datasets disappear

**Future consideration**: ATProto blob storage for datasets
- Pros: Truly decentralized
- Cons: Storage costs, size limits, performance

### Lens Code Storage
**Question**: How should Lens transformation code be stored?

**Options**:
1. Python code as string in record (security concerns!)
2. Reference to GitHub/GitLab repo + commit hash
3. Bytecode or AST representation
4. Only store metadata, expect manual implementation

**Decision needed**: See Phase 1 planning

## Success Metrics

- **Functionality**: Can publish schema, publish dataset, discover, load end-to-end
- **Performance**: Dataset discovery <100ms (with AppView), load time unchanged
- **Adoption**: Easy enough that external users publish datasets
- **Interop**: Schema records usable from other languages (future)

## Timeline & Dependencies

```
Phase 1 (Lexicon Design)
    ↓
Phase 2 (Python Client) ← CRITICAL PATH
    ↓
    ├── Phase 3 (AppView) [parallel, optional]
    └── Phase 4 (Codegen) [parallel]
         ↓
Phase 5 (Integration & Testing)
```

Phase 2 is the critical path. Phases 3 & 4 can proceed in parallel once Phase 2 foundations are in place.

## Related Documents

- `02_lexicon_design.md` - Detailed Lexicon schema specifications
- `03_python_client.md` - Python library architecture and API design
- `04_appview.md` - AppView service architecture
- `05_codegen.md` - Code generation approach and templates
