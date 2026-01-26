# Decision: Lexicon Validation Process

**Issue**: #50
**Status**: Needs decision
**Blocked By**: #45, #46, #47, #48, #49 (all design decisions)
**Priority**: Critical - Final step before Phase 1 completion

## Problem Statement

Once we've finalized all design decisions, we need to validate that our Lexicon JSON definitions:
1. Follow ATProto Lexicon specification correctly
2. Are internally consistent
3. Support all our use cases
4. Can be implemented as designed

This is the final checkpoint before Phase 1 (Lexicon Design) is complete and we move to Phase 2 (Implementation).

## What Needs Validation

### 1. Schema Record Lexicon (`io.atdata.schema`)
- Field type system (primitive, ndarray, nested)
- Type unions are properly structured
- Required vs optional fields
- Constraints (maxLength, etc.) are reasonable
- Example schema records validate against the Lexicon

### 2. Dataset Record Lexicon (`io.atdata.dataset`)
- URL array handling
- Metadata blob size limits
- Schema reference format
- Tag array constraints
- Example dataset records validate against the Lexicon

### 3. Lens Record Lexicon (`io.atdata.lens`)
- Code reference structure
- Schema reference handling
- Union types for different code storage options (if applicable)
- Example lens records validate against the Lexicon

## Validation Checklist

### ATProto Spec Compliance

**Lexicon Structure**:
- [ ] All Lexicons have required fields: `lexicon`, `id`, `defs`
- [ ] `lexicon` field is set to `1` (current version)
- [ ] `id` follows NSID format (reverse domain notation)
- [ ] `defs.main` exists and has `type: "record"`
- [ ] Record `key` is set appropriately (`tid` for time-ordered)

**Field Types**:
- [ ] All field types are valid ATProto types
  - `string`, `integer`, `boolean`, `bytes`, `object`, `array`
  - `ref`, `union` for complex types
- [ ] String fields have appropriate `maxLength`
- [ ] Array fields have `items` definition
- [ ] Object fields have `properties` definition
- [ ] Refs point to valid def names (e.g., `#fieldType`)

**Constraints**:
- [ ] `maxLength` values are reasonable (not too small, not too large)
- [ ] `minLength` constraints make sense
- [ ] Required fields are marked correctly
- [ ] Optional fields have appropriate defaults

### Internal Consistency

**Cross-References**:
- [ ] Schema refs (e.g., `schemaRef` in datasets) use correct format
  - Should be AT-URI format: `at://did:plc:.../io.atdata.schema/...`
- [ ] Union refs point to existing defs
- [ ] No circular references

**Type System**:
- [ ] Field types are well-defined
  - Primitive types map clearly (str, int, float, bool, bytes)
  - NDArray type includes dtype and optional shape
  - Nested types have schema reference
- [ ] Optional vs required semantics are clear

**Metadata**:
- [ ] Descriptions are present and helpful
- [ ] Examples match the schema
- [ ] Deprecations are noted (if any)

### Use Case Coverage

**Can we represent...**:
- [ ] All current PackableSample types?
- [ ] NDArray with dtype and shape information?
- [ ] Optional fields?
- [ ] Nested PackableSample types (future)?
- [ ] Dataset metadata (arbitrary key-value)?
- [ ] Multiple WebDataset shard URLs?
- [ ] Lens code references (repo + commit + path)?

**Can we implement...**:
- [ ] Python codegen from schema records?
- [ ] Dataset publishing with external URLs?
- [ ] Dataset loading from records?
- [ ] Lens publishing with code references?
- [ ] Schema versioning (version field present)?

## Validation Methods

### 1. Schema Validation Tools

**Use ATProto Tools** (if available):
```bash
# If ATProto has a Lexicon validator
atproto-lexicon validate io.atdata.schema.json
atproto-lexicon validate io.atdata.dataset.json
atproto-lexicon validate io.atdata.lens.json
```

**Create Custom Validator**:
```python
# src/atdata/atproto/validation.py
from jsonschema import validate, ValidationError

def validate_lexicon(lexicon_json: dict) -> tuple[bool, list[str]]:
    """Validate Lexicon against ATProto spec."""
    errors = []

    # Check required fields
    if 'lexicon' not in lexicon_json:
        errors.append("Missing 'lexicon' field")
    if 'id' not in lexicon_json:
        errors.append("Missing 'id' field")
    if 'defs' not in lexicon_json:
        errors.append("Missing 'defs' field")

    # Check NSID format
    nsid = lexicon_json.get('id', '')
    if not is_valid_nsid(nsid):
        errors.append(f"Invalid NSID: {nsid}")

    # More validations...

    return len(errors) == 0, errors
```

### 2. Example Record Validation

**Create Example Records**:

```python
# examples/schema_record.json
{
  "$type": "io.atdata.schema",
  "name": "ImageSample",
  "version": "1.0.0",
  "description": "Sample with image and label",
  "fields": [
    {
      "name": "image",
      "type": {"kind": "ndarray", "dtype": "uint8", "shape": [null, null, 3]},
      "optional": false
    },
    {
      "name": "label",
      "type": {"kind": "primitive", "primitive": "str"},
      "optional": false
    }
  ],
  "metadata": {"author": "alice"},
  "createdAt": "2025-01-06T12:00:00Z"
}
```

**Validate Against Lexicon**:
```python
def validate_record(record: dict, lexicon: dict) -> tuple[bool, list[str]]:
    """Validate a record against its Lexicon."""
    errors = []

    # Check $type matches Lexicon id
    record_type = record.get('$type')
    lexicon_id = lexicon.get('id')
    if record_type != lexicon_id:
        errors.append(f"Type mismatch: {record_type} != {lexicon_id}")

    # Validate required fields
    main_def = lexicon['defs']['main']['record']
    required = main_def.get('required', [])
    for field in required:
        if field not in record:
            errors.append(f"Missing required field: {field}")

    # Validate field types
    properties = main_def.get('properties', {})
    for field, value in record.items():
        if field in properties:
            # Type checking logic
            pass

    return len(errors) == 0, errors
```

### 3. Roundtrip Testing

**Test Full Cycle**:
1. Create PackableSample class
2. Generate schema record from class
3. Validate schema record against Lexicon
4. Generate code from schema record
5. Verify generated code matches original class

```python
def test_roundtrip():
    # 1. Original class
    @atdata.packable
    class TestSample:
        x: int
        y: str

    # 2. Generate schema record
    generator = SchemaRecordGenerator()
    record = generator.from_class(TestSample)

    # 3. Validate against Lexicon
    is_valid, errors = validate_record(record, SCHEMA_LEXICON)
    assert is_valid, f"Validation failed: {errors}"

    # 4. Generate code from record
    codegen = PythonGenerator()
    code = codegen.generate_from_record(record)

    # 5. Execute generated code and compare
    exec_globals = {}
    exec(code, exec_globals)
    GeneratedClass = exec_globals['TestSample']

    # Should be equivalent
    original_instance = TestSample(x=1, y="test")
    generated_instance = GeneratedClass(x=1, y="test")

    assert original_instance.packed == generated_instance.packed
```

### 4. Edge Case Testing

**Test Corner Cases**:
- [ ] Empty optional fields
- [ ] Very long strings (maxLength boundary)
- [ ] Large arrays (maxItems boundary)
- [ ] Complex nested types
- [ ] Unicode in strings
- [ ] Special characters in names
- [ ] Large metadata blobs

## Validation Artifacts

After validation, we should have:

### 1. Finalized Lexicon JSON Files

```
.planning/lexicons/
  io.atdata.schema.json
  io.atdata.dataset.json
  io.atdata.lens.json
```

Each file:
- Validates against ATProto Lexicon spec
- Has complete documentation
- Includes examples

### 2. Example Records

```
.planning/examples/
  schema_example.json
  dataset_example.json
  lens_example.json
```

Each example:
- Validates against its Lexicon
- Demonstrates all key features
- Includes comments explaining choices

### 3. Validation Test Suite

```python
# tests/test_lexicons.py

def test_schema_lexicon_valid():
    """Test schema Lexicon is valid."""
    with open('.planning/lexicons/io.atdata.schema.json') as f:
        lexicon = json.load(f)
    is_valid, errors = validate_lexicon(lexicon)
    assert is_valid, errors

def test_schema_example_valid():
    """Test schema example validates against Lexicon."""
    with open('.planning/lexicons/io.atdata.schema.json') as f:
        lexicon = json.load(f)
    with open('.planning/examples/schema_example.json') as f:
        example = json.load(f)
    is_valid, errors = validate_record(example, lexicon)
    assert is_valid, errors

# Similar tests for dataset and lens
```

### 4. Validation Report

```markdown
# Lexicon Validation Report

## Summary
- Schema Lexicon: ✅ Valid
- Dataset Lexicon: ✅ Valid
- Lens Lexicon: ✅ Valid

## Validation Results

### io.atdata.schema
- ATProto compliance: ✅ Pass
- Internal consistency: ✅ Pass
- Example validation: ✅ Pass
- Edge cases: ✅ Pass

### io.atdata.dataset
...

## Issues Found
None

## Recommendations
1. Consider adding X field to Y
2. Might want to increase maxLength for Z
...
```

## Implementation Plan

### Step 1: Create Lexicon JSON Files (depends on decisions #45-49)

Based on finalized decisions:
- Schema representation format (#45)
- Lens code storage (#46)
- WebDataset storage (#47)
- Schema evolution (#48)
- Lexicon namespace (#49)

Create three JSON files with complete Lexicon definitions.

### Step 2: Create Example Records

For each Lexicon, create 2-3 example records demonstrating:
- Minimal record
- Full-featured record
- Edge cases

### Step 3: Write Validation Tests

Implement validation test suite that:
- Validates Lexicons against ATProto spec
- Validates examples against Lexicons
- Tests roundtrip (class → record → code → class)

### Step 4: Manual Review

Have team members review:
- Lexicon designs
- Example records
- Any edge cases or concerns

### Step 5: Document Issues and Resolutions

Track any issues found:
- What was wrong?
- How was it fixed?
- Why was this decision made?

### Step 6: Final Sign-off

Once all validation passes:
- Mark Issue #50 as complete
- Unblock Phase 1 (Issue #17)
- Proceed to Phase 2 implementation

## Tools and Resources

**ATProto Resources**:
- Lexicon specification: https://atproto.com/specs/lexicon
- NSID specification: https://atproto.com/specs/nsid
- Example Lexicons: https://github.com/bluesky-social/atproto/tree/main/lexicons

**Validation Tools**:
- JSON Schema validator (jsonschema library)
- ATProto SDK validation (if available)
- Custom validators (we'll write)

**Documentation**:
- All planning docs in `.planning/`
- Decision docs in `.planning/decisions/`
- Lexicon design in `02_lexicon_design.md`

## Success Criteria

Phase 1 Issue #17 is complete when:
- ✅ All three Lexicons are finalized and validated
- ✅ Example records validate against Lexicons
- ✅ Roundtrip tests pass
- ✅ Team has reviewed and approved
- ✅ Documentation is complete
- ✅ Ready to begin Phase 2 implementation

## Next Steps After Validation

Once Issue #50 is complete:
1. Close Issue #50
2. Unblock and close Issue #17 (Phase 1)
3. Begin Phase 2 (Issue #18) - Python Client implementation
4. Reference finalized Lexicons during implementation

## Open Questions

1. **Should we submit Lexicons to ATProto for official review?**
   - Pro: Get expert feedback
   - Con: Delays, may not be necessary
   - Recommendation: Optional, do if time permits

2. **Should we create a Lexicon registry/index?**
   - Pro: Makes discovery easier
   - Con: Extra infrastructure
   - Recommendation: Defer to Phase 3 (AppView)

3. **How do we handle Lexicon updates after publication?**
   - Once records exist, changing Lexicons is breaking
   - Need clear versioning for Lexicons themselves
   - Recommendation: Lexicons are v1 for all Phase 1-5

## References

- All design decisions: `01-05_*.md` in this directory
- Lexicon design: `../02_lexicon_design.md`
- ATProto Lexicon spec: https://atproto.com/specs/lexicon

---

**Decision Needed By**: After all decisions #45-49 are finalized
**Decision Maker**: Project maintainer (max) + team review
**Date Created**: 2026-01-07

## Recommended Action

**After all design decisions are made**:
1. Create three Lexicon JSON files
2. Create example records for each
3. Write and run validation test suite
4. Review as team
5. Document any issues and fixes
6. Get final sign-off
7. Mark Phase 1 complete ✅
