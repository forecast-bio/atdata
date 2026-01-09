# Decision: Schema Representation Format

**Issue**: #45
**Status**: Needs decision
**Blocks**: #50 (Lexicon validation)
**Priority**: Critical for Phase 1

## DECISION

Let's go with the **JSON schema** approach; the only real issue we have to worry about here is the `NDArray` support, and we can solve that by

* Adding a standardized JSON Schema shim to represent an `NDArray` as its serialized bytes
* Referencing this as the type within other schemas, and making this the standard we use

We'll make this decision future-proof by adding a property in the Lexicon for schemas that gives the type of schema definition, with one currently supported value (for JSON Schema), and then leave the standard overall as an open union, as is standard for atproto lexicons.

---

## Problem Statement

We need to decide how to represent `PackableSample` type definitions within ATProto Lexicon records. This affects:
- How schemas are stored and transmitted
- Code generation complexity
- Cross-language interoperability
- Tooling ecosystem availability

## Context

`PackableSample` types have specific requirements:
- Support for primitive types (str, int, float, bool, bytes)
- **Special handling for `NDArray` types** with dtype and shape information
- Msgpack serialization metadata
- Optional/required field semantics
- Future extensibility (constraints, validation, nested types)

## Options

### Option 1: Custom Format within ATProto Lexicon ⭐ RECOMMENDED

**Description**: Define our own type system using ATProto Lexicon primitives

**Example**:
```json
{
  "name": "image",
  "type": {
    "kind": "ndarray",
    "dtype": "uint8",
    "shape": [null, null, 3]
  },
  "optional": false,
  "description": "RGB image with variable height/width"
}
```

**Pros**:
- ✅ Native to ATProto - no external dependencies
- ✅ Tailored exactly to `PackableSample` needs
- ✅ Clean representation of NDArray (dtype, shape constraints)
- ✅ Full control over codegen implementation
- ✅ Can evolve independently
- ✅ Easy to extend (add constraints, validation rules, etc.)

**Cons**:
- ❌ Need to implement our own codegen tooling
- ❌ Less ecosystem tooling available
- ❌ Need to maintain custom parsers

**Implementation Effort**: Medium
- Lexicon design: ~2-3 days
- Python codegen: ~5-7 days
- Validation: ~2-3 days

---

### Option 2: JSON Schema

**Description**: Use JSON Schema as the type definition format

**Example**:
```json
{
  "type": "object",
  "properties": {
    "image": {
      "type": "object",
      "x-atdata-type": "ndarray",
      "x-dtype": "uint8",
      "x-shape": [null, null, 3]
    }
  },
  "required": ["image"]
}
```

**Pros**:
- ✅ Industry standard, widely understood
- ✅ Extensive validation tooling exists
- ✅ Many language implementations

**Cons**:
- ❌ Not designed for code generation
- ❌ Awkward NDArray representation (need custom extensions like `x-atdata-type`)
- ❌ Overly complex for our needs
- ❌ Still need custom codegen despite standard format
- ❌ Doesn't map cleanly to Python dataclasses

**Implementation Effort**: Medium-High
- Still need custom codegen despite standard format
- JSON Schema parsers available but adaptation needed

---

### Option 3: Protobuf (Protocol Buffers)

**Description**: Use Protobuf schema definitions

**Example**:
```protobuf
message ImageSample {
  bytes image = 1;  // NDArray serialized
  string label = 2;
  optional float confidence = 3;
}
```

**Pros**:
- ✅ Excellent codegen ecosystem (Python, TypeScript, Rust, etc.)
- ✅ Compact binary format
- ✅ Strong cross-language support
- ✅ Built-in versioning/evolution support

**Cons**:
- ❌ Not ATProto-native (different ecosystem)
- ❌ NDArray handling is awkward (just bytes, lose dtype/shape info)
- ❌ Requires compilation step
- ❌ Less human-readable than JSON
- ❌ Doesn't integrate well with msgpack serialization we already use
- ❌ Would need to convert between Protobuf and our existing serialization

**Implementation Effort**: High
- Need to bridge Protobuf and PackableSample worlds
- Complexity of maintaining two serialization systems

## Recommendation: Option 1 (Custom Format)

**Rationale**:

1. **Perfect fit for PackableSample**: Our custom format can represent NDArray types with full dtype and shape information, which is critical for ML/data applications.

2. **ATProto-native**: Using Lexicon primitives means everything stays within the ATProto ecosystem. No external schema dependencies.

3. **Full control**: We can optimize the codegen for our exact use case. Want to generate dataclasses with specific decorators? Easy. Want to add custom validation? We control it.

4. **Simplicity**: Despite being "custom", it's actually simpler than adapting JSON Schema or Protobuf to our needs. Less impedance mismatch.

5. **Future-proof**: Easy to add features like:
   - Shape constraints and validation
   - Custom serialization hooks
   - Nested PackableSample types
   - Union types for polymorphic samples

## Implementation Plan

If we choose Option 1:

1. **Finalize Lexicon structure** (see `02_lexicon_design.md`)
   - Field type definitions (primitive, ndarray, nested)
   - Union types for extensibility
   - Metadata fields

2. **Implement Python codegen** (see `05_codegen.md`)
   - Jinja2 templates for dataclass generation
   - Type annotation mapping
   - NDArray handling with dtype/shape comments

3. **Build validation tooling**
   - Schema validator (ensure schemas are well-formed)
   - Sample validator (ensure samples match schemas)
   - Compatibility checker (schema evolution)

4. **Document the format**
   - Clear spec for the type system
   - Examples for common patterns
   - Migration guide from JSON Schema if needed

## Alternative Approaches Considered

**Hybrid approach**: Use JSON Schema for validation + custom codegen
- Still has awkward NDArray representation
- Added complexity of two systems
- Not recommended

**Defer decision**: Use simple types only, add NDArray later
- Defeats the purpose - NDArray is core to ML datasets
- Would require breaking changes later
- Not recommended

## Impact on Other Decisions

- **Code generation (#36-39)**: Custom format means we fully control codegen
- **Validation (#50)**: Need to implement custom validators
- **Cross-language support (future)**: Need to write codegen for each language, but format is language-agnostic

## Success Criteria

After implementing this decision:
- ✅ Can represent all current PackableSample types
- ✅ NDArray types include dtype and shape information
- ✅ Generated code is idiomatic Python (dataclasses with type hints)
- ✅ Schema records are human-readable
- ✅ Codegen is fast (<1s for typical schemas)

## Open Questions

1. **Should we support shape constraints beyond documentation?**
   - e.g., should [224, 224, 3] be enforced at runtime?
   - Recommendation: Document only initially, add validation later

2. **How to handle nested PackableSample types?**
   - Reference by schema URI?
   - Inline nested schema?
   - Recommendation: URI reference for Phase 1

3. **Should we generate both classes and validators?**
   - Just classes, or also Pydantic models?
   - Recommendation: Start with dataclasses, add Pydantic later if needed

## References

- Full Lexicon design: `../02_lexicon_design.md`
- Code generation plan: `../05_codegen.md`
- Example schemas: `../02_lexicon_design.md` (Schema Record Lexicon section)

---

**Decision Needed By**: Before starting Phase 1 Issue #22 (Lexicon design)
**Decision Maker**: Project maintainer (max)
**Date Created**: 2026-01-07
