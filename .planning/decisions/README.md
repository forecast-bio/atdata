# Critical Design Decisions for ATProto Integration

This directory contains detailed analysis and recommendations for the critical design decisions needed before implementing ATProto integration in `atdata`.

## Decision Documents (In Dependency Order)

### Core Design Decisions (Can be made in parallel)

1. **[01_schema_representation_format.md](01_schema_representation_format.md)** (Issue #45)
   - **Question**: How to represent PackableSample types in Lexicon records?
   - **Options**: Custom format, JSON Schema, Protobuf
   - **Recommendation**: Custom format within ATProto Lexicon
   - **Impact**: Code generation, cross-language support
   - **Blocks**: Issue #50 (validation)

2. **[02_lens_code_storage.md](02_lens_code_storage.md)** (Issue #46)
   - **Question**: How to store Lens transformation code?
   - **Options**: Code references, inline code, metadata only
   - **Recommendation**: Code references (GitHub + commit hash) only
   - **Impact**: Security, usability, trust model
   - **Blocks**: Issue #50 (validation)
   - ⚠️ **CRITICAL SECURITY DECISION**

3. **[03_webdataset_storage.md](03_webdataset_storage.md)** (Issue #47)
   - **Question**: Where to store actual WebDataset .tar files?
   - **Options**: External URLs, ATProto blobs, hybrid
   - **Recommendation**: External URLs (Phase 1), hybrid (future)
   - **Impact**: Decentralization, scalability, costs
   - **Blocks**: Issue #50 (validation)

4. **[04_schema_evolution.md](04_schema_evolution.md)** (Issue #48)
   - **Question**: How do schemas evolve without breaking changes?
   - **Options**: Semantic versioning, compatibility rules, migrations
   - **Recommendation**: Semantic versioning + Lenses for migration
   - **Impact**: Long-term maintainability, compatibility
   - **Blocks**: Issue #50 (validation), Issue #39 (type validation)

5. **[05_lexicon_namespace.md](05_lexicon_namespace.md)** (Issue #49)
   - **Question**: What namespace (NSID) to use for Lexicons?
   - **Options**: `app.bsky.atdata.*`, `io.atdata.*`, others
   - **Recommendation**: `io.atdata.*` (Phase 1), request `app.bsky.*` later
   - **Impact**: Discoverability, ownership, migration
   - **Blocks**: Issue #50 (validation)

### Final Validation (Depends on all above)

6. **[06_lexicon_validation.md](06_lexicon_validation.md)** (Issue #50)
   - **Question**: How to validate finalized Lexicon designs?
   - **Process**: Validation checklist, example records, tests
   - **Deliverables**: Finalized Lexicon JSON files, validation report
   - **Blocked By**: Issues #45, #46, #47, #48, #49 (all completed ✅)
   - **Blocks**: Phase 1 completion (Issue #17)
   - **Status**: Ready to proceed

### Architectural Assessment

7. **[assessment.md](assessment.md)** (Issue #51) ✅ **Complete**
   - **Comprehensive appraisal** of all finalized design decisions
   - **Overall Grade**: A- (Excellent with caveats)
   - **Analysis**: Strengths, synergies, trade-offs, risks, long-term trajectory
   - **Recommendations**: Immediate next steps and phasing guidance

## Decision Status

| Issue | Decision | Status | Final Decision |
|-------|----------|--------|----------------|
| #45 | Schema format | ✅ Decided | JSON Schema with NDArray shim |
| #46 | Lens code storage | ✅ Decided | External repos (GitHub + tangled.org) |
| #47 | WebDataset storage | ✅ Decided | Hybrid (URLs + blobs from start) |
| #48 | Schema evolution | ✅ Decided | rkey={NSID}@{semver} + migration Lenses |
| #49 | Lexicon namespace | ✅ Decided | `ac.foundation.dataset.*` |
| #50 | Validation process | ⏳ Ready | Proceed with finalized decisions |
| #51 | Architectural appraisal | ✅ Complete | See [assessment.md](assessment.md) |

**Overall Assessment**: Grade A- (Excellent with caveats) - See [assessment.md](assessment.md) for detailed analysis

## How to Use These Documents

### For Review

1. **Read in order** (01 through 06) to understand dependencies
2. **Focus on recommendations** - detailed analysis supports them
3. **Check open questions** - some need your input
4. **Provide feedback** - comment on issues or update documents

### For Implementation

1. **After decisions made** - use as reference during coding
2. **Check success criteria** - ensure implementation meets goals
3. **Follow recommendations** - they're based on thorough analysis
4. **Update as needed** - decisions can evolve with learning

## Key Insights

### Security First
- **Issue #46** (Lens code storage) is a critical security decision
- Recommendation: Code references only (no arbitrary code execution)
- Can add inline code later if we solve sandboxing

### Pragmatic Approach
- Start with what works (external URLs, custom format)
- Add sophistication later (ATProto blobs, advanced features)
- Don't block on perfect solutions

### Independence
- Use `io.atdata.*` namespace (don't wait for Bluesky approval)
- Can migrate to `app.bsky.atdata.*` later if desired
- Maintain control over project direction

### Future-Proof
- Semantic versioning enables evolution
- Hybrid storage approach allows flexibility
- Custom format gives us full control

## Decision Dependencies

```
┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
│   #45   │  │   #46   │  │   #47   │  │   #48   │  │   #49   │
│ Format  │  │  Lens   │  │ Storage │  │Evolution│  │Namespace│
└────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘
     │            │            │            │            │
     └────────────┴────────────┴────────────┴────────────┘
                              │
                         ┌────▼────┐
                         │   #50   │
                         │Validate │
                         └────┬────┘
                              │
                         ┌────▼────┐
                         │ Phase 1 │
                         │Complete │
                         └─────────┘
```

All decisions #45-49 can be made in parallel, then #50 validates everything before Phase 1 completion.

## Timeline

**Recommended**:
1. **Week 1**: Review and decide on #45-49 (can be done in parallel)
2. **Week 2**: Validation (#50) - create Lexicon JSON files and examples
3. **Week 3**: Begin Phase 2 implementation

**Flexible**: Can make decisions incrementally, but all needed before #50

## Questions?

- Review individual decision documents for detailed analysis
- Check "Open Questions" sections for items needing input
- See "References" sections for related planning documents
- Consult `../02_lexicon_design.md` for technical details

---

**Created**: 2026-01-07
**Status**: All decisions pending review
**Next Step**: Review decision documents and provide feedback
