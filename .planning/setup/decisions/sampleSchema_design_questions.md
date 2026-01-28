# sampleSchema Lexicon Design Questions

This document captures open design questions for the `ac.foundation.dataset.sampleSchema` Lexicon that require user decisions before implementation.

## Q1: Key Format Validation

**Context:**
- Schema uses `"key": "any"` in Lexicon
- Documentation says rkey format is `{NSID}@{semver}`
- ATProto might not support regex validation on rkey in Lexicons

**Question:**
Should we add validation for the rkey format in the Lexicon definition, or is this enforced elsewhere?

**Options:**
1. Add rkey pattern validation if ATProto Lexicons support it
2. Document expected format but rely on application-level validation
3. Use a structured key type instead of "any"

**Impact:**
- Option 1: Strongest validation, prevents malformed rkeys
- Option 2: Simpler, but allows invalid rkeys to be created
- Option 3: May not be compatible with ATProto Lexicon spec

**Decision:** [TBD]

---

## Q2: Required Fields in JSON Schema

**Context:**
- The `jsonSchema` field accepts any JSON Schema object
- JSON Schemas can have zero required fields (all optional)
- PackableSample types in atdata typically have at least one field

**Question:**
Should we enforce that JSON Schemas must have at least one required field?

**Options:**
1. No constraint - allow empty required arrays
2. Require at least one field in required array
3. No constraint but document best practices

**Impact:**
- Option 1: Maximum flexibility, but allows degenerate schemas
- Option 2: Forces meaningful sample definitions
- Option 3: Middle ground - guidance without enforcement

**Recommendation:** Option 3 (document best practices)

**Decision:** [TBD]

---

## Q3: Schema Type Extension Path

**Context:**
- `schemaType` field has `enum: ["jsonschema"]` only
- Future may want to support other formats (Avro, Protobuf, etc.)
- Lexicon schema evolution unclear

**Question:**
How should we design for future schema format support?

**Options:**
1. Keep enum as-is, add new formats in major version bump
2. Use open union type instead of closed enum
3. Add `schemaFormat` union field alongside `jsonSchema`

**Example for Option 3:**
```json
{
  "schemaFormat": {
    "type": "union",
    "refs": ["#jsonSchemaFormat", "#avroSchemaFormat", "#protobufSchemaFormat"]
  }
}
```

**Impact:**
- Option 1: Breaking change required for new formats
- Option 2: No validation of format string
- Option 3: Clean extensibility but more complex now

**Recommendation:** Option 1 (YAGNI - wait for actual need)

**Decision:** [TBD]

---

## Q4: Metadata Field Structure

**Context:**
- `metadata` is currently `"type": "object"` with no structure
- Common fields like `author`, `license`, `tags` are documented in examples
- No validation on these fields

**Question:**
Should we define a structured schema for common metadata fields?

**Options:**
1. Keep fully unstructured (current)
2. Define optional but structured fields (author, license, tags, etc.)
3. Create separate metadata Lexicon type and reference it

**Example for Option 2:**
```json
{
  "metadata": {
    "type": "object",
    "properties": {
      "author": {"type": "string", "maxLength": 200},
      "license": {"type": "string", "maxLength": 100},
      "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 20}
    }
  }
}
```

**Impact:**
- Option 1: Maximum flexibility, no validation
- Option 2: Standardization with optional compliance
- Option 3: Reusability but added complexity

**Recommendation:** Option 2 (structured but optional)

**Decision:** [TBD]

---

## Q5: NDArray Shim URI Default

**Context:**
- `ndarrayShimUri` is optional with default mentioned in description
- Standard shim is at `https://foundation.ac/schemas/atdata-ndarray-bytes/1.0.0`
- No explicit default value in Lexicon

**Question:**
Should we add an explicit default value for `ndarrayShimUri`?

**Options:**
1. Add `"default": "https://foundation.ac/schemas/atdata-ndarray-bytes/1.0.0"`
2. Keep as optional, codegen assumes standard shim if missing
3. Make required - always explicit

**Impact:**
- Option 1: Clearest behavior, but locks in URI
- Option 2: Flexibility for future shim versions
- Option 3: Most explicit but verbose

**Recommendation:** Option 2 (implicit default in codegen)

**Decision:** [TBD]

---

## Notes

These questions should be resolved before finalizing the sampleSchema Lexicon design. Some can be deferred to Phase 2 implementation based on priority.

**Priority:**
- Q1: High (affects rkey strategy)
- Q2: Low (can document later)
- Q3: Low (YAGNI until needed)
- Q4: Medium (affects metadata usage patterns)
- Q5: Medium (affects codegen implementation)
