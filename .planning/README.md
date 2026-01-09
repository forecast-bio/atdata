# ATProto Integration Planning

This directory contains comprehensive planning documents for integrating AT Protocol into the `atdata` library, transforming it into a distributed dataset federation.

## Planning Documents

### Design Decisions

📋 **[decisions/](decisions/)** - Critical design decisions with detailed analysis
- Each decision has its own document with options, recommendations, and rationale
- See [decisions/README.md](decisions/README.md) for navigation guide
- **Must be reviewed and finalized before Phase 1 implementation**

### Architecture & Design

1. **[01_overview.md](01_overview.md)** - High-level vision, architecture, and project roadmap
   - Overall vision for distributed datasets on ATProto
   - System architecture diagram
   - Development phases and dependencies
   - Open design questions

2. **[02_lexicon_design.md](02_lexicon_design.md)** - Detailed Lexicon schema specifications
   - Schema Record Lexicon (for PackableSample types)
   - Dataset Record Lexicon (for dataset indexes)
   - Lens Record Lexicon (for transformations)
   - Schema representation format decision
   - Example records

3. **[03_python_client.md](03_python_client.md)** - Python library architecture and API design
   - ATProtoClient for authentication
   - SchemaPublisher/Loader
   - DatasetPublisher/Loader
   - LensPublisher
   - Integration with existing Dataset class
   - Testing strategy

4. **[04_appview.md](04_appview.md)** - AppView aggregation service design
   - Service architecture
   - Database schema (PostgreSQL, ElasticSearch)
   - HTTP API endpoints
   - Firehose consumer
   - Deployment options
   - Performance considerations

5. **[05_codegen.md](05_codegen.md)** - Code generation tooling
   - Python code generator from schema records
   - CLI interface
   - Template system
   - Type validation and compatibility checking
   - Future multi-language support

## Milestone Tracking

**Milestone**: ATProto Integration (Milestone #1)
**Total Issues**: 34 (6 parent issues + 28 subissues)

### Planning Phase (Issue #44)

**Status**: In progress
**Priority**: High (blocks Phase 1)

Critical decisions needed before implementation:
- Decide on schema representation format (#45)
- Decide on Lens code storage approach (#46)
- Decide on WebDataset storage strategy (#47)
- Design schema evolution and versioning strategy (#48)
- Finalize Lexicon namespace and NSID structure (#49)
- Review and validate Lexicon JSON definitions (#50)

**All decisions have detailed analysis in planning documents with recommendations.**

### Phase Breakdown

#### Phase 1: Lexicon Design & Schema Definition (Issue #17)
- Design Lexicon for PackableSample schema storage (#22)
- Design Lexicon for dataset index records (#23)
- Design Lexicon for Lens transformation records (#24)
- Evaluate schema representation formats (#25)

**Status**: Blocked by Planning (#44)
**Priority**: High (blocks all other phases)

#### Phase 2: Python Client Library (Issue #18)
- Implement ATProto authentication and session management (#26)
- Implement schema publishing to ATProto (#27)
- Implement dataset index record publishing (#28)
- Implement Lens transformation publishing (#29)
- Implement querying and discovery of datasets (#30)
- Extend Dataset class to load from ATProto records (#31)

**Status**: Blocked by Phase 1
**Priority**: High (critical path)

#### Phase 3: AppView & Index Aggregation Service (Issue #19)
- Design AppView architecture and data model (#32)
- Implement record ingestion from ATProto firehose (#33)
- Implement search and query API (#34)
- Add caching and indexing for performance (#35)

**Status**: Blocked by Phase 2
**Priority**: Medium (optional infrastructure)

#### Phase 4: Code Generation Tooling (Issue #20)
- Design code generation template system (#36)
- Implement Python code generator from schema records (#37)
- Add CLI for code generation (#38)
- Support type validation and compatibility checking (#39)

**Status**: Blocked by Phase 2
**Priority**: Medium (can run parallel with Phase 3)

#### Phase 5: End-to-End Integration & Testing (Issue #21)
- Create end-to-end example workflows (#40)
- Write integration tests for full publish/discover/load cycle (#41)
- Create comprehensive documentation (#42)
- Performance testing and optimization (#43)

**Status**: Blocked by Phase 2
**Priority**: High (required for production release)

## Getting Started

To begin implementation:

1. **Review design decisions** in `decisions/` directory - these need your input first
2. **Review architecture documents** (01-05) to understand the full scope
3. **Provide feedback** on the design decisions and open questions
4. **Finalize decisions** for issues #45-49
5. **Validate Lexicons** (issue #50) once decisions are made
6. **Begin Phase 1 implementation** after validation
7. **Track progress** using chainlink issues

### Quick Start for Decision Review

1. Read [decisions/README.md](decisions/README.md) for overview
2. Review each decision document (01-06)
3. For each decision:
   - Agree with recommendation? → Comment on issue
   - Disagree? → Propose alternative in issue
   - Unsure? → Discuss open questions
4. Once all decisions made → Proceed to issue #50 (validation)

## Key Design Decisions Needed

Before starting implementation, we need decisions on (see Issue #44 and subissues #45-50):

1. **Schema representation format** (Issue #45)
   - Recommendation: Custom format within ATProto Lexicon
   - Alternative: JSON Schema or Protobuf
   - Details in `02_lexicon_design.md`

2. **Lens code storage** (Issue #46)
   - Recommendation: Code references (GitHub + commit) only
   - Alternative: Allow inline code (security concerns)
   - Details in `02_lexicon_design.md`

3. **WebDataset storage location** (Issue #47)
   - Phase 1: External storage (S3, HTTP) - just URLs
   - Future: ATProto blob storage for smaller datasets
   - Details in `02_lexicon_design.md`

4. **Schema evolution strategy** (Issue #48)
   - How to handle versioning and compatibility
   - Migration path for breaking changes
   - Details in `05_codegen.md`

5. **Lexicon namespace** (Issue #49)
   - Current proposal: `app.bsky.atdata.*`
   - May need to coordinate with ATProto/Bluesky team
   - Details in `02_lexicon_design.md`

6. **Lexicon validation** (Issue #50)
   - Validate all Lexicon JSON against ATProto spec
   - Create example records for testing
   - Blocked by decisions #45-49

## Questions for Discussion

Review the "Open Design Questions" sections in each planning document, particularly:

- `01_overview.md` - Overall architecture questions
- `02_lexicon_design.md` - Lexicon-specific design questions (CRITICAL for Phase 1)

## Next Steps

1. Review planning documents
2. Discuss and finalize design decisions
3. Begin Phase 1 implementation
4. Iterate and refine as we learn

---

**Milestone Created**: 2026-01-07
**Last Updated**: 2026-01-07
**Status**: Planning complete, ready for review
