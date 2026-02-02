# Record Lexicon Assessment

## Overview

Comprehensive assessment of `ac.foundation.dataset.record` Lexicon design against ATProto standards and atdata project requirements.

**Assessment Date:** 2026-01-07
**Lexicon Version:** Initial design
**Assessor:** Claude Sonnet 4.5

---

## Executive Summary

**Grade: B+** (Good with improvements needed)

The record Lexicon provides a solid foundation for dataset indexing with hybrid storage support. Key strengths include clean union-based storage design and appropriate use of ATProto primitives. However, several issues need addressing:

- ⚠️ **Critical**: schemaRef should use format validation
- ⚠️ **High**: Metadata structure inconsistency with sampleSchema pattern
- ⚠️ **Medium**: Missing $type discriminators in union variants
- ✅ **Strength**: Clean storage union design
- ✅ **Strength**: Appropriate use of tid keys for datasets

---

## Detailed Analysis

### 1. Key Type Choice ✅ **Appropriate**

```json
"key": "tid"
```

**Assessment:** Correct choice for dataset records.

**Rationale:**
- TIDs provide temporal ordering (useful for "recent datasets" queries)
- Auto-generated, no collision risk
- Appropriate for records without natural semantic keys
- Consistent with ATProto patterns for user-generated content

**Comparison to sampleSchema:**
- sampleSchema uses `"key": "any"` for versioned rkeys like `{NSID}@{semver}`
- record uses `"key": "tid"` for chronological dataset entries
- Both choices are appropriate for their use cases

---

### 2. Field Validation Issues

#### Issue 2.1: schemaRef Missing Format Validation ⚠️ **Critical**

```json
"schemaRef": {
  "type": "string",
  "description": "AT-URI reference...",
  "maxLength": 500
}
```

**Problem:** Should use `"format": "at-uri"` like we did for sampleSchema fields.

**Fix:**
```json
"schemaRef": {
  "type": "string",
  "format": "at-uri",
  "description": "AT-URI reference to the sampleSchema record",
  "maxLength": 500
}
```

**Impact:** Without format validation, malformed references could be stored.

---

#### Issue 2.2: License Field Inconsistency ⚠️ **Medium**

sampleSchema metadata:
```json
"license": {
  "type": "string",
  "description": "... SPDX identifiers recommended ... or full SPDX URLs ...",
  "maxLength": 200
}
```

record:
```json
"license": {
  "type": "string",
  "description": "License (SPDX identifier preferred)",
  "maxLength": 100
}
```

**Problem:** Inconsistent maxLength and less detailed guidance.

**Recommendation:** Align with sampleSchema:
- maxLength: 200 (to support full URLs)
- Enhanced description with examples
- Reference Schema.org license property

---

#### Issue 2.3: Tags Field Inconsistency ⚠️ **Medium**

sampleSchema metadata:
```json
"tags": {
  "type": "array",
  "items": {"type": "string", "maxLength": 150},
  "maxLength": 30
}
```

record:
```json
"tags": {
  "type": "array",
  "items": {"type": "string", "maxLength": 50},
  "maxLength": 20
}
```

**Problem:** Different limits with no clear rationale.

**Recommendation:** Use consistent limits or document why datasets need different constraints than schemas.

---

### 3. Metadata Structure ⚠️ **High Priority**

#### Current Design

record:
```json
"metadata": {
  "type": "bytes",
  "description": "Msgpack-encoded metadata dict",
  "maxLength": 100000
},
"tags": {...},
"license": {...}
```

sampleSchema:
```json
"metadata": {
  "type": "object",
  "properties": {
    "license": {...},
    "tags": {...}
  }
}
```

**Problem:** Inconsistent approach between lexicons.

**Analysis:**

**Option A: Keep Separate (Current)**
- Pros: More discoverable (top-level fields, indexed/searchable)
- Pros: Validated by Lexicon
- Cons: Duplicates structure with metadata blob
- Cons: Inconsistent with sampleSchema pattern

**Option B: Unified Metadata Object**
- Pros: Consistent with sampleSchema
- Pros: Single source of truth
- Cons: Less discoverable for search
- Cons: Can't validate blob contents

**Recommendation:** Keep current approach but clarify relationship:
- Top-level fields: Core, searchable metadata (license, tags, size)
- metadata blob: Extended, arbitrary key-value pairs
- Update descriptions to explain this pattern

---

### 4. Storage Union Design ✅ **Excellent**

```json
"storage": {
  "type": "union",
  "refs": ["#storageExternal", "#storageBlobs"]
}
```

**Strengths:**
- Clean separation of storage types
- Extensible (closed: false by default)
- Well-defined variants

#### Issue 4.1: Missing $type in Union Variants ⚠️ **Critical**

storageExternal:
```json
{
  "type": "object",
  "required": ["type", "urls"],
  "properties": {
    "type": {"type": "string", "const": "external"}
  }
}
```

**Problem:** Uses `type` field as discriminator instead of ATProto's `$type`.

**ATProto Spec:** "Unions require discriminator fields... union variants: Always include `$type`"

**Fix:**
```json
{
  "type": "object",
  "required": ["$type", "urls"],
  "properties": {
    "$type": {
      "type": "string",
      "const": "ac.foundation.dataset.record#storageExternal"
    }
  }
}
```

**Impact:** Current design violates ATProto conventions and may cause issues with SDKs.

---

### 5. Size Information ✅ **Good Design**

```json
"size": {
  "type": "ref",
  "ref": "#datasetSize",
  "description": "Dataset size information (optional)"
}
```

**Strengths:**
- Optional (appropriate, not all datasets track this)
- Structured with useful fields (samples, bytes, shards)
- Uses ref for reusability

**Minor Suggestion:** Consider renaming `datasetSize` to `sizeInfo` or `datasetSizeInfo` for clarity.

---

### 6. Blob Storage Design ⚠️ **Needs Verification**

```json
"blobs": {
  "type": "array",
  "items": {
    "type": "blob",
    "description": "Blob reference to a WebDataset tar archive"
  }
}
```

**Questions:**
1. Does ATProto Lexicon support `"type": "blob"` for array items?
2. Should this be a ref like `"type": "ref", "ref": "#blobRef"`?
3. Are blob mime types validated?

**Example shows:**
```json
{
  "$type": "blob",
  "ref": {"$link": "..."},
  "mimeType": "application/x-tar",
  "size": 1234567
}
```

**Recommendation:** Verify against ATProto blob specification and potentially add validation constraints (maxSize, accept mimeType patterns).

---

### 7. Closed Union Consideration 🤔

```json
"storage": {
  "type": "union",
  "refs": ["#storageExternal", "#storageBlobs"]
}
```

**Current:** `closed: false` (default)

**Question:** Should storage union be closed?

**Arguments for closed: true:**
- Core storage types unlikely to change frequently
- Breaking change to add new storage after launch
- More predictable for clients

**Arguments for closed: false (current):**
- Future extensibility (e.g., IPFS-native, Filecoin, Arweave)
- Consistent with sampleSchema schema union pattern
- Graceful degradation for unknown types

**Recommendation:** Keep open but document in description that external/blobs are the canonical types maintained by foundation.ac.

---

### 8. Missing Fields from Standard Patterns

Comparing to Schema.org Dataset and sampleSchema patterns:

**Consider Adding:**

1. **Publisher/Creator** - Who published this dataset?
   - Could use top-level `creator` field (DID/handle)
   - Or rely on record author (implicit in AT-URI)

2. **Version** - Dataset versioning?
   - Current approach: New record per version (via tid)
   - Alternative: Add explicit `version` field like sampleSchema
   - **Recommendation:** Document that versioning is via new records, reference via AT-URI with tid

3. **Citation** - How to cite this dataset?
   - Optional field for academic datasets
   - Could go in metadata blob for now

4. **Related Datasets** - Links to variants, subsets, etc.
   - Could be array of AT-URIs
   - Or handle via separate "collection" Lexicon later

**Recommendation:** Current fields are sufficient for v1. Document these as future extensions.

---

### 9. ATProto Compliance Checklist

| Requirement | Status | Notes |
|-------------|--------|-------|
| Valid Lexicon version | ✅ | lexicon: 1 |
| NSID format | ✅ | ac.foundation.dataset.record |
| Key type specified | ✅ | tid (appropriate) |
| Required fields present | ✅ | name, schemaRef, storage, createdAt |
| Union discriminators | ⚠️ | Missing $type in variants |
| Format validators | ⚠️ | Missing at-uri format |
| Blob type usage | ⚠️ | Needs verification |
| Description fields | ✅ | All fields documented |
| maxLength constraints | ✅ | Present on strings |
| Datetime format | ✅ | createdAt uses datetime |

---

### 10. Example Record Validation

#### External Storage Example ✅

```json
{
  "$type": "ac.foundation.dataset.record",
  "name": "CIFAR-10 Training Set",
  "schemaRef": "at://did:plc:abc123/ac.foundation.dataset.sampleSchema/imageclassification@1.0.0",
  "storage": {"type": "external", "urls": ["..."]}
}
```

**Issues:**
- schemaRef is well-formed but not validated (missing format check)
- storage.type should be $type
- Otherwise structurally correct

#### Blob Storage Example ⚠️

```json
{
  "storage": {
    "type": "blobs",
    "blobs": [{
      "$type": "blob",
      "ref": {"$link": "..."},
      "mimeType": "application/x-tar"
    }]
  }
}
```

**Issues:**
- storage.type should be $type
- Blob structure needs verification against ATProto spec
- mimeType not validated in Lexicon

---

## Priority Issues Summary

### Critical (Must Fix)

1. **Add format validation to schemaRef** - Use `"format": "at-uri"`
2. **Fix union discriminators** - Use `$type` instead of `type` in storage variants
3. **Verify blob type usage** - Confirm ATProto compliance

### High Priority (Should Fix)

4. **Align metadata pattern** - Clarify relationship between top-level fields and metadata blob
5. **Standardize license field** - Match sampleSchema maxLength and description
6. **Standardize tags field** - Use consistent limits or document rationale

### Medium Priority (Consider)

7. **Add $type requirement to union variants** - Make explicit in required array
8. **Document versioning strategy** - Clarify that new versions = new records
9. **Add blob validation** - Consider maxSize, mimeType constraints

### Low Priority (Future)

10. **Consider closed union** - Evaluate after Phase 1 usage patterns
11. **Add creator field** - If needed based on user feedback
12. **Collection/relationship fields** - Phase 2 feature

---

## Consistency Matrix

Comparison of patterns between sampleSchema and record Lexicons:

| Pattern | sampleSchema | record | Status |
|---------|--------------|--------|--------|
| AT-URI format | ✅ Uses format | ❌ Missing | **Fix** |
| License field | 200 chars, detailed | 100 chars, basic | **Align** |
| Tags limits | 150/30 | 50/20 | **Decide** |
| Metadata structure | Structured object | Blob + top-level | **Document** |
| Union discriminator | Uses $type | Uses type | **Fix** |
| Versioning | Explicit version field | Implicit (tid) | **Different OK** |
| Key type | any (semantic) | tid (temporal) | **Both OK** |

---

## Recommendations

### Immediate Actions

1. Add `"format": "at-uri"` to schemaRef field
2. Change storage union variants to use `$type` discriminator
3. Verify blob array item type with ATProto specification
4. Align license field with sampleSchema (maxLength: 200, enhanced description)
5. Decide on tags limits (recommend matching sampleSchema: 150/30)

### Documentation Improvements

6. Add description clarifying metadata blob vs top-level fields relationship
7. Document that dataset versioning is via new records (tids)
8. Add note about storage union extensibility
9. Cross-reference with sampleSchema Lexicon

### Consider for Phase 2

10. Add creator/publisher field if user feedback indicates need
11. Evaluate closed union after observing extension patterns
12. Consider collection/relationship Lexicon for dataset hierarchies

---

## Conclusion

The record Lexicon provides a solid foundation but needs refinement for ATProto compliance and consistency with sampleSchema patterns. The storage union design is excellent, and the use of tids is appropriate. Primary concerns are format validation, union discriminators, and metadata pattern clarity.

**Estimated effort to address critical issues:** 2-3 hours
**Recommended timeline:** Before Phase 1 completion

After fixes, expected grade: **A-** (Excellent and production-ready)
