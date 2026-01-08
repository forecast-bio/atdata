# Architectural Assessment of Design Decisions

**Issue**: #51
**Date**: 2026-01-07
**Status**: Complete

## Overall Impression: **Ambitious but Coherent**

The finalized design decisions prioritize **flexibility and future-proofing** over initial simplicity. This is a deliberate trade-off that makes sense given the scope of building a distributed dataset federation.

---

## Decision Summary

1. **Schema Format (#45)**: JSON Schema with NDArray shim, extensible via open union
2. **Lens Code (#46)**: External repos (GitHub + tangled.org), language metadata, future attestation
3. **Storage (#47)**: Hybrid (URLs + blobs) from start, AppView proxy for blobs
4. **Evolution (#48)**: rkey as {NSID}@{semver}, getLatestSchema query, optional migration Lenses
5. **Namespace (#49)**: `ac.foundation.dataset.*` (sampleSchema, record, lens)

---

## Key Strengths

### 1. **Ecosystem Integration** (JSON Schema + External Repos)

**Decision**: JSON Schema for type definitions, external repos for code storage

**Strength**: Leveraging existing ecosystems rather than building in isolation. JSON Schema brings:
- Extensive tooling (validators, codegen, IDE support)
- Multi-language support out of the box
- Familiarity for developers

Pairing this with GitHub/tangled.org for Lenses means developers can use existing workflows.

**Implication**: Lower barrier to entry, faster time to value. The NDArray shim is the only custom piece, which is appropriate since that's the unique requirement.

---

### 2. **Progressive Decentralization** (Hybrid Storage)

**Decision**: Hybrid storage from day one (URLs + PDS blobs)

**Strength**: This is pragmatic yet principled. Not forcing decentralization where it doesn't make sense (TB-scale datasets), but enabling it where it does (smaller datasets, self-hosters).

**Key Insight**: The AppView proxy for blobs is clever - it means users can work with a unified WebDataset URL interface regardless of backend storage. This abstraction is powerful.

**Implication**: More implementation complexity upfront, but avoids a painful migration later. The open union pattern makes this clean.

---

### 3. **Versioning as Identity** (rkey = NSID@semver)

**Decision**: Embed version in record key, use NSID for permanent identity

**Strength**: This is elegant. By making versioning part of the identity (rkey), you get:
- Immutable version records (can't accidentally update a published version)
- Natural query pattern (`getLatestSchema` Lexicon)
- Clear semantic versioning enforcement

**Synergy**: Combining this with Lenses for migration is brilliant. The rkey structure makes it trivial to discover what migrations exist (e.g., "show me all versions of schema X").

**Implication**: This requires custom rkey handling (type `any` in Lexicon), which ATProto supports but isn't the default pattern. Need to ensure tooling understands this convention.

---

### 4. **Trust Layer** (Attestation + Verification)

**Decision**: Language metadata + future attestation/verification records for Lenses

**Strength**: Thinking ahead about the trust problem. In a distributed system, trust is critical. This approach:
- Short-term: Language metadata helps users understand what they're running
- Long-term: Attestation (formal correctness proofs) + verification (trusted DIDs)

This is a **strong security model** that's missing from many distributed systems.

**Implication**: This is a research-level feature (formal verification of Lenses). Starting with language metadata is right, but the attestation system will require significant design work. Consider this Phase 6+.

---

## Architectural Tensions (Intentional Trade-offs)

### 1. **Complexity Budget**

**Observation**: Sophisticated solutions across the board:
- JSON Schema (standard but verbose)
- Hybrid storage (two code paths)
- Custom rkey scheme (non-standard)
- Future attestation system (advanced)

**Assessment**: This increases initial implementation cost significantly. However, each choice is justified:
- JSON Schema: Ecosystem benefits outweigh verbosity
- Hybrid storage: Essential for real-world use cases
- Custom rkey: Enables clean versioning
- Attestation: Future-proofing for trust

**Recommendation**: ✅ Accept the complexity, but **phase implementation carefully**:
- Phase 1-2: Core functionality (schemas, datasets, basic lenses)
- Phase 3: Hybrid storage in AppView
- Phase 4: Codegen for JSON Schema
- Phase 5+: Attestation/verification system

---

### 2. **ATProto Conventions vs. Custom Patterns**

**Observation**: Using some non-standard ATProto patterns:
- rkey type `any` (not typical)
- Custom versioning scheme in rkey
- `getLatestSchema` query Lexicon (not standard CRUD)

**Assessment**: This is **justified innovation**. ATProto is designed to support custom use cases. The versioning scheme in particular is a good use of flexible rkey.

**Caveat**: Need to document these conventions clearly, since they won't match typical ATProto examples.

---

### 3. **JSON Schema for NDArray**

**Observation**: JSON Schema wasn't designed for NDArray types. The shim approach treats them as "serialized bytes" with metadata.

**Assessment**: This is **pragmatic but leaky**. The abstraction leaks because:
- JSON Schema describes serialized form (bytes), not semantic form (array with dtype/shape)
- Codegen will need custom handling for NDArray types
- Validation happens at deserialization, not schema level

**Alternative Considered**: Custom format would give cleaner NDArray representation, but traded that for ecosystem benefits.

**Mitigation**: Ensure the NDArray shim is well-documented and becomes a de facto standard within the atdata ecosystem. Consider publishing it as a reusable JSON Schema extension.

---

## Synergies (Where Decisions Reinforce Each Other)

### 1. **Versioning + Lenses + rkey Scheme**

This trilogy works beautifully together:
- rkey embeds version → easy to list all versions
- Lenses enable migration → versions can evolve safely
- `getLatestSchema` query → discoverable latest version

This creates a **complete version management story** that's rare in distributed systems.

---

### 2. **Hybrid Storage + AppView Proxy**

The hybrid storage decision unlocks the proxy pattern:
- Large datasets stay on S3/R2 (practical)
- Small datasets can use PDS blobs (decentralized)
- AppView proxies both → uniform interface

This means the **client code is simple** (just WebDataset URLs) even though the backend is sophisticated.

---

### 3. **JSON Schema + Attestation + Language Metadata**

This builds a **tiered trust model**:
1. Base layer: JSON Schema validates structure
2. Language metadata: Users know what they're executing
3. Attestation (future): Formal proofs of correctness
4. Verification (future): Social trust (trusted DIDs)

Each layer adds security without requiring the next layer to exist.

---

## Implementation Risks & Mitigations

### Risk 1: JSON Schema Complexity

**Risk**: JSON Schema is verbose and can be confusing for users defining NDArray-heavy schemas.

**Mitigation**:
- Build **high-quality codegen** that hides the complexity (users write Python, get JSON Schema)
- Provide **NDArray shim library** that handles the serialization/deserialization
- Create **examples and templates** for common patterns

---

### Risk 2: Hybrid Storage Code Paths

**Risk**: Two storage backends means 2x testing, 2x bugs, 2x maintenance.

**Mitigation**:
- Use **abstraction layer** in Dataset class (already planned)
- **Prioritize external URLs** for Phase 1-2 (blob support can be added incrementally)
- Test both paths from the start (CI/CD)

---

### Risk 3: Custom rkey Convention

**Risk**: Tools that expect standard TID-based rkeys might break.

**Mitigation**:
- **Document clearly** in all Lexicon definitions
- Provide **helper functions** in SDK (`parseSchemaRkey`, `formatSchemaRkey`)
- Ensure `getLatestSchema` query is the primary discovery mechanism (hides rkey complexity)

---

### Risk 4: Attestation System Scope Creep

**Risk**: Formal verification and trust systems are research-level hard. Could delay entire project.

**Mitigation**:
- Mark as **explicitly future work** (Phase 6+)
- Start with **language metadata only** (low-hanging fruit)
- Consider **social trust first** (verified DIDs, reputation) before formal verification
- Partner with PL/verification researchers if pursuing formal proofs

---

## Long-Term Trajectory

The decisions set up a compelling long-term vision:

**Year 1**: Core dataset federation
- Publish/discover datasets
- JSON Schema for types
- External URL storage
- Basic Lenses

**Year 2**: Decentralization
- PDS blob storage for small datasets
- AppView with proxy
- Migration Lenses widely used
- Community schemas emerging

**Year 3**: Trust & verification
- Language metadata standard
- Verified DID system (social trust)
- Attestation for critical Lenses
- Cross-language support (TypeScript, Rust)

**Year 4+**: Research frontier
- Formal verification of Lenses
- Advanced query capabilities
- Federated learning on distributed datasets
- Integration with compute-over-data systems

---

## Concrete Recommendations

### 1. **Immediate** (Before Phase 1 Implementation)

- [ ] Define the **NDArray JSON Schema shim** precisely (schema structure, examples)
- [ ] Spec out the **rkey format** (`{NSID}@{semver}` - what's valid NSID here? full NSID or partial?)
- [ ] Design the **`getLatestSchema` query Lexicon** (parameters, return type)
- [ ] Define the **storage union type** (external URL variant vs PDS blob variant)

### 2. **Phase 1-2** (Lexicon + Python Client)

- [ ] Implement **external URLs only** for storage (defer blobs to Phase 3)
- [ ] Build **NDArray shim library** (serialize/deserialize with metadata)
- [ ] Create **basic codegen** (Python dataclass ↔ JSON Schema)
- [ ] Defer **language metadata** on Lenses to Phase 2 (start with just repo reference)

### 3. **Phase 3** (AppView)

- [ ] Implement **hybrid storage support** in AppView
- [ ] Build **proxy for PDS blobs** (unified WebDataset URL interface)
- [ ] Add **getLatestSchema endpoint**

### 4. **Phase 4+** (Future Work)

- [ ] Add **language metadata** to Lens records
- [ ] Design **attestation Lexicon** (separate from Lens records)
- [ ] Design **verification Lexicon** (trusted DIDs)
- [ ] Research formal verification feasibility

---

## Summary Assessment

**Grade: A-** (Excellent with caveats)

### Strengths
- ✅ Leverages existing ecosystems (JSON Schema, GitHub)
- ✅ Future-proof (extensible via open unions, versioning built-in)
- ✅ Pragmatic decentralization (hybrid storage)
- ✅ Innovative versioning (rkey scheme)
- ✅ Strong security model (multi-layered trust)

### Concerns
- ⚠️ High implementation complexity (manageable with phasing)
- ⚠️ JSON Schema for NDArray is a leaky abstraction (acceptable trade-off)
- ⚠️ Custom rkey convention requires good documentation
- ⚠️ Attestation system is ambitious (defer to future)

### Overall Assessment

This is a **well-considered architecture** that makes intentional trade-offs. The bet is on ecosystem integration and flexibility over simplicity, which is appropriate for a distributed dataset federation. The key to success will be **disciplined phasing** - implement the core first, add sophistication incrementally.

The decisions form a **coherent whole** where each piece reinforces the others. The versioning scheme, Lenses, and hybrid storage create a system that's greater than the sum of its parts.

**Recommendation**: ✅ **Proceed with these decisions**. Document the NDArray shim and rkey conventions thoroughly, and commit to incremental implementation.

---

## Next Steps

1. Close decision issues #45-49 as decided
2. Update planning documents with finalized decisions
3. Proceed to Issue #50 (Lexicon validation) with:
   - NDArray JSON Schema shim definition
   - rkey format specification
   - `getLatestSchema` query Lexicon design
   - Storage union type definition
4. Begin Phase 1 implementation after validation complete
