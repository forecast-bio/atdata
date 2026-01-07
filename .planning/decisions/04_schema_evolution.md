# Decision: Schema Evolution and Versioning Strategy

**Issue**: #48
**Status**: Needs decision
**Blocks**: #50 (Lexicon validation), #39 (Type validation)
**Priority**: High

## DECISION

For this, let's take the following approach:

1. Let's make the `rkey` for the `ac.foundation.dataset.sampleSchema` records be of type `any`.
2. Then, we can have our own standard for the `rkey` being of the format `{NSID}@{semver}`, where `{NSID}` gives an NSID for the permanent identifier of this sample schema type.
    * This allows us to bookkeep on the version updates
    * We can make a `ac.foundation.dataset.getLatestSchema` `query` Lexicon that will provide the record for the latest version of a given schema, as well
3. We can build into the `atdata` SDK that whenever users update their own sample schema types, they can pass in optional `Lens`es between the two versions that give transformations to downgrade / upgrade records, so that there's an easy dev-facing way to auto-update any existing datasets using an older schema and maintain compatibility with older code for newer data.

---

## Problem Statement

We need to define how PackableSample schemas can evolve over time without breaking existing datasets or code. This includes:
- Version numbering scheme
- Compatibility rules (what changes are allowed?)
- Migration strategies
- Runtime validation

## Context

Schemas will evolve:
- **Adding new fields** (e.g., adding optional metadata)
- **Removing deprecated fields**
- **Changing field types** (e.g., int → float)
- **Changing field constraints** (e.g., making field optional)

Real-world example:
```python
# Version 1.0.0
@atdata.packable
class ImageSample:
    image: NDArray
    label: str

# Version 1.1.0 - add optional field (backward compatible)
@atdata.packable
class ImageSample:
    image: NDArray
    label: str
    confidence: Optional[float] = None  # NEW

# Version 2.0.0 - remove field (breaking change)
@atdata.packable
class ImageSample:
    image: NDArray
    # label removed - BREAKING
    class_id: int  # NEW, replaces label
```

## Goals

1. **Backward compatibility**: Old code can read new data (when possible)
2. **Forward compatibility**: New code can read old data (when possible)
3. **Clear breaking changes**: Users know when they need to update
4. **Safe migrations**: Data transformations are explicit and verifiable
5. **Developer-friendly**: Easy to understand and use

## Versioning Scheme

### Semantic Versioning (MAJOR.MINOR.PATCH)

**Recommendation**: Use semantic versioning for schemas

```
1.0.0 → 1.0.1 → 1.1.0 → 2.0.0
```

**Version Components**:
- **MAJOR**: Breaking changes (incompatible with previous versions)
- **MINOR**: Backward-compatible additions (new optional fields)
- **PATCH**: Documentation, clarifications, no functional changes

### Examples

```python
# 1.0.0 → 1.0.1 (PATCH)
# Change: Fixed documentation, added field description
# Compatible: ✅ Yes
# Action: None needed

# 1.0.0 → 1.1.0 (MINOR)
# Change: Added optional field 'metadata'
# Compatible: ✅ Yes (backward compatible)
# Action: Old code works, new code can use new field

# 1.0.0 → 2.0.0 (MAJOR)
# Change: Removed field 'old_field'
# Compatible: ❌ No (breaking change)
# Action: Users must migrate or use conversion lens
```

## Compatibility Rules

### Backward-Compatible Changes (MINOR version bump)

**Allowed**:
- ✅ Adding optional fields
- ✅ Making required field optional
- ✅ Widening type constraints (e.g., relaxing shape requirements)
- ✅ Adding documentation
- ✅ Adding metadata

**Example**:
```python
# v1.0.0
class Sample:
    x: int

# v1.1.0 - backward compatible
class Sample:
    x: int
    y: Optional[int] = None  # Added optional field
```

**Guarantee**: Code written for v1.0.0 continues to work with v1.1.0 schemas

---

### Breaking Changes (MAJOR version bump)

**Required**:
- ❌ Removing fields
- ❌ Changing field types (str → int)
- ❌ Making optional field required
- ❌ Narrowing type constraints (e.g., restricting shape)
- ❌ Renaming fields

**Example**:
```python
# v1.0.0
class Sample:
    x: int
    y: int

# v2.0.0 - breaking changes
class Sample:
    x: float  # Type changed
    # y removed
    z: int  # New required field
```

**Guarantee**: Code written for v1.0.0 will NOT work with v2.0.0 without updates

---

### Non-Breaking Changes (PATCH version bump)

**Allowed**:
- ✅ Documentation updates
- ✅ Metadata changes
- ✅ Clarifications
- ✅ Bug fixes in schema definition (not structure)

**No functional changes to schema structure**

## Compatibility Checking

### Automatic Compatibility Checker

Implement `SchemaValidator` to check compatibility:

```python
from atdata.codegen import SchemaValidator

validator = SchemaValidator()

old_schema = load_schema("at://alice/schema/sample/v1.0.0")
new_schema = load_schema("at://alice/schema/sample/v1.1.0")

is_compatible, issues = validator.is_compatible(old_schema, new_schema)

if not is_compatible:
    print("Incompatibilities found:")
    for issue in issues:
        print(f"  - {issue}")
```

**Checks**:
1. Field additions/removals
2. Type changes
3. Optional → Required changes
4. Shape constraint changes

See `../05_codegen.md` for implementation details.

### Version Constraints in Dataset Records

Datasets can specify schema version constraints:

```json
{
  "$type": "app.bsky.atdata.dataset",
  "schemaRef": "at://alice/schema/sample/v1.0.0",
  "schemaVersionConstraint": ">=1.0.0,<2.0.0",
  ...
}
```

**Semantics**:
- Dataset created with v1.0.0
- Compatible with v1.x.x (minor/patch updates)
- NOT compatible with v2.x.x (breaking changes)

## Migration Strategies

### Option 1: Lenses as Migration Paths ⭐ RECOMMENDED

**Concept**: Use Lens transformations to migrate between schema versions

```python
# Migration lens: v1.0.0 → v2.0.0
@atdata.lens
def sample_v1_to_v2(v1: SampleV1) -> SampleV2:
    """Migrate from v1.0.0 to v2.0.0"""
    return SampleV2(
        x=float(v1.x),  # int → float
        z=hash(v1.y) % 100  # derive z from removed y
    )

@sample_v1_to_v2.putter
def sample_v2_to_v1(v2: SampleV2, v1: SampleV1) -> SampleV1:
    """Reverse migration (lossy)"""
    return SampleV1(
        x=int(v2.x),
        y=0  # Can't recover removed field
    )
```

**Benefits**:
- ✅ Reuses existing Lens infrastructure
- ✅ Explicit transformation logic
- ✅ Bidirectional (when possible)
- ✅ Publishable and discoverable

**Limitations**:
- ❌ May be lossy (can't always reverse)
- ❌ Requires manual implementation

---

### Option 2: Automatic Migration

**Concept**: Generate migrations automatically based on schema diff

```python
migrator = SchemaM migrator()
v2_sample = migrator.migrate(v1_sample, target_version="2.0.0")
```

**Benefits**:
- ✅ Convenient for users
- ✅ No manual code needed

**Limitations**:
- ❌ Only works for simple changes (add/remove optional fields)
- ❌ Can't handle complex transformations (type changes)
- ❌ Risk of incorrect assumptions

**Recommendation**: Could implement for simple cases, but Lenses are more general

---

### Option 3: Manual Migration Scripts

**Concept**: Users write custom migration scripts

**Benefits**:
- ✅ Full control

**Limitations**:
- ❌ Not publishable/discoverable
- ❌ No standardization

**Recommendation**: Allow as fallback, but encourage Lenses

## Runtime Validation

### Sample Validation Against Schema

```python
from atdata.codegen import TypeValidator

validator = TypeValidator()
schema = load_schema("at://alice/schema/sample/v1.0.0")

# Validate sample
sample = SampleV1(x=42, y=100)
is_valid, errors = validator.validate(sample, schema)

if not is_valid:
    print("Validation errors:")
    for error in errors:
        print(f"  - {error}")
```

**Checks**:
1. All required fields present
2. Field types match
3. NDArray dtypes match (if specified)
4. NDArray shapes match (if specified)

**When to validate**:
- ❓ Every sample creation? (slow)
- ✅ On dataset write? (good balance)
- ✅ On user request (explicit validation)

**Recommendation**: Validate on write, make runtime validation optional

## Schema Record Versioning

### Version Field in Schema Records

```json
{
  "$type": "app.bsky.atdata.schema",
  "name": "ImageSample",
  "version": "1.1.0",  # Semantic version
  ...
}
```

### Publishing New Versions

**Option A**: New record for each version (RECOMMENDED)
```
at://alice/schema/imagesample/v1.0.0  # Version 1.0.0
at://alice/schema/imagesample/v1.1.0  # Version 1.1.0
at://alice/schema/imagesample/v2.0.0  # Version 2.0.0
```

**Pros**:
- ✅ Immutable versions
- ✅ Easy to reference specific versions
- ✅ No breaking changes to existing references

**Cons**:
- ❌ More records to manage
- ❌ Harder to find "latest" version

**Option B**: Update existing record
```
at://alice/schema/imagesample  # Always points to latest
```

**Pros**:
- ✅ Single canonical reference
- ✅ Easy to find latest

**Cons**:
- ❌ Breaks immutability
- ❌ References become ambiguous over time

**Recommendation**: Option A (new record per version), with metadata linking to previous versions

### Linking Versions

```json
{
  "$type": "app.bsky.atdata.schema",
  "name": "ImageSample",
  "version": "2.0.0",
  "metadata": {
    "previousVersion": "at://alice/schema/imagesample/v1.1.0",
    "migrationLens": "at://alice/lens/imagesample-v1-to-v2"
  },
  ...
}
```

## Developer Workflow

### Publishing a New Schema Version

```python
# 1. Define new version
@atdata.packable
class ImageSampleV2:
    image: NDArray
    label: str
    confidence: Optional[float] = None  # NEW

# 2. Publish with version
schema_uri = publisher.publish_schema(
    ImageSampleV2,
    name="ImageSample",
    version="1.1.0",  # MINOR bump
    metadata={
        "previousVersion": "at://alice/schema/imagesample/v1.0.0"
    }
)

# 3. Optionally publish migration lens
migration_lens = publisher.publish_lens(
    v1_to_v2_lens,
    source_schema_uri="at://alice/schema/imagesample/v1.0.0",
    target_schema_uri=schema_uri,
    name="ImageSample v1→v2 Migration"
)
```

### Using Versioned Schemas

```python
# Load specific version
schema = loader.get_schema("at://alice/schema/imagesample/v1.0.0")

# Check compatibility
is_compatible = validator.is_compatible(
    "at://alice/schema/imagesample/v1.0.0",
    "at://alice/schema/imagesample/v2.0.0"
)

# Find migration path
migration = loader.find_migration(
    source="at://alice/schema/imagesample/v1.0.0",
    target="at://alice/schema/imagesample/v2.0.0"
)
```

## Tooling Support

### CLI Commands

```bash
# Check schema compatibility
atdata schema diff \
  at://alice/schema/sample/v1.0.0 \
  at://alice/schema/sample/v2.0.0

# Validate sample against schema
atdata validate mysample.msgpack \
  --schema at://alice/schema/sample/v1.0.0

# Find migration path
atdata schema migrate \
  --from at://alice/schema/sample/v1.0.0 \
  --to at://alice/schema/sample/v2.0.0
```

### IDE Support (Future)

- Autocomplete for schema versions
- Warnings for compatibility issues
- Quick fixes for migrations

## Open Questions

1. **Should we auto-bump versions on publish?**
   - Detect changes, suggest version bump?
   - Recommendation: Manual for Phase 1, auto-suggest later

2. **How to handle shape evolution for NDArray?**
   ```python
   # v1: image shape [224, 224, 3]
   # v2: image shape [256, 256, 3]  # Breaking or not?
   ```
   - If shape is documented (not enforced), this could be minor
   - If shape is validated, this is breaking
   - Recommendation: Document only initially

3. **Should we support version ranges in schema refs?**
   ```json
   "schemaRef": "at://alice/schema/sample@^1.0.0"  # npm-style
   ```
   - Pro: More flexible
   - Con: Ambiguous (which exact version?)
   - Recommendation: Explicit versions only for Phase 1

4. **What about deprecated fields?**
   ```python
   class Sample:
       x: int
       y: int  # @deprecated: Use z instead
       z: Optional[int] = None
   ```
   - Could add deprecation warnings
   - Could track in schema metadata
   - Recommendation: Metadata only for Phase 1

## Success Criteria

After implementing this decision:
- ✅ Schemas use semantic versioning
- ✅ Compatibility rules are clear and documented
- ✅ Compatibility checker validates schema changes
- ✅ Lenses can be used for migrations
- ✅ Dataset records can specify version constraints
- ✅ Breaking changes require major version bump

## References

- Code generation: `../05_codegen.md` (SchemaValidator, TypeValidator)
- Lexicon design: `../02_lexicon_design.md` (Schema versioning)
- Lens transformations: `02_lens_code_storage.md`

---

**Decision Needed By**: Before Phase 4 Issue #39 (Type validation)
**Decision Maker**: Project maintainer (max)
**Date Created**: 2026-01-07
