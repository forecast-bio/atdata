# Decision: Lexicon Namespace and NSID Structure

**Issue**: #49
**Status**: Needs decision
**Blocks**: #50 (Lexicon validation)
**Priority**: Critical for Phase 1

## DECISION

We're going to use an org NSID for the steward organization as the base:

```
ac.foundation.dataset.*
```

The choices we have then are

```
ac.foundation.dataset.sampleSchema
ac.foundation.dataset.record
ac.foundation.dataset.lens
```

---

## Problem Statement

We need to finalize the namespace (NSID - Namespaced Identifier) for atdata Lexicons. This is a critical decision because:
- NSIDs are permanent and hard to change
- They affect discoverability and organization
- They may require coordination with ATProto/Bluesky team

## Context

ATProto NSIDs follow reverse domain notation:
```
app.bsky.feed.post          # Bluesky official feed posts
com.example.myapp.record    # Third-party app
```

We need NSIDs for three record types:
1. Schema records (PackableSample definitions)
2. Dataset records (dataset indexes)
3. Lens records (transformations)

## Current Proposal

```
app.bsky.atdata.schema      # PackableSample schema records
app.bsky.atdata.dataset     # Dataset index records
app.bsky.atdata.lens        # Lens transformation records
```

## Options

### Option 1: `app.bsky.atdata.*` (Current Proposal)

**Full NSIDs**:
- `app.bsky.atdata.schema`
- `app.bsky.atdata.dataset`
- `app.bsky.atdata.lens`

**Pros**:
- ✅ Under Bluesky ecosystem umbrella
- ✅ High visibility and discoverability
- ✅ Official-looking namespace
- ✅ Good for adoption

**Cons**:
- ❌ May require approval from Bluesky team
- ❌ `app.bsky.*` typically for official Bluesky apps
- ❌ Could be rejected or need to change later
- ❌ Implies Bluesky endorsement/ownership

**Risk**: ⚠️ Medium (may need to change if not approved)

---

### Option 2: `io.atdata.*` or `org.atdata.*`

**Full NSIDs**:
- `io.atdata.schema`
- `io.atdata.dataset`
- `io.atdata.lens`

**Pros**:
- ✅ Independent namespace
- ✅ No approval needed
- ✅ Clear ownership (atdata project)
- ✅ Can use immediately

**Cons**:
- ❌ Less discoverable (not under Bluesky)
- ❌ Appears less "official"
- ❌ Need to own atdata.io domain (or just use anyway?)

**Risk**: ⭐ Low (we control it)

---

### Option 3: `app.bsky.atproto.atdata.*` (Nested)

**Full NSIDs**:
- `app.bsky.atproto.atdata.schema`
- `app.bsky.atproto.atdata.dataset`
- `app.bsky.atproto.atdata.lens`

**Pros**:
- ✅ Still under Bluesky but more specific
- ✅ Groups with other ATProto-related Lexicons
- ✅ Less likely to conflict

**Cons**:
- ❌ Longer NSIDs
- ❌ Awkward naming (`atproto.atdata`?)
- ❌ Still may need approval

**Risk**: ⚠️ Medium

---

### Option 4: Personal/Org namespace (e.g., `com.github.username.atdata.*`)

**Example with your GitHub**:
- `com.github.maxineishere.atdata.schema` (if that's your GH username)
- Or: `com.yourorg.atdata.schema`

**Pros**:
- ✅ Guaranteed to work (it's your namespace)
- ✅ No approval needed
- ✅ Clear ownership

**Cons**:
- ❌ Looks very unofficial
- ❌ Hard to discover
- ❌ Tied to individual/org, not project
- ❌ May need to migrate later if project grows

**Risk**: ⭐ Very Low (but not ideal for adoption)

## Recommendation: Start with Option 2 (`io.atdata.*`), Keep Option 1 as Goal

**Phased Approach**:

### Phase 1: Use `io.atdata.*` immediately
- No approvals needed
- Can start development right away
- Professional-looking namespace
- Independent from Bluesky governance

### Future: Request `app.bsky.atdata.*` if appropriate
- Once atdata has users and proven value
- Submit formal request to Bluesky/ATProto team
- Migrate if approved (see migration plan below)

**Rationale**:
1. **Speed**: Don't block development waiting for approval
2. **Safety**: If denied `app.bsky.*`, we haven't committed to it
3. **Flexibility**: Can migrate namespaces if needed
4. **Independence**: atdata can exist independently of Bluesky

## Implementation Details

### Namespace Structure

```
io.atdata
  ├── schema          # PackableSample schema definitions
  ├── dataset         # Dataset index records
  └── lens            # Lens transformations
```

**Lexicon IDs**:
```json
{
  "lexicon": 1,
  "id": "io.atdata.schema",
  ...
}
```

```json
{
  "lexicon": 1,
  "id": "io.atdata.dataset",
  ...
}
```

```json
{
  "lexicon": 1,
  "id": "io.atdata.lens",
  ...
}
```

### Record URIs

```
at://did:plc:abc123/io.atdata.schema/3jk2lo34klm
at://did:plc:abc123/io.atdata.dataset/7mn8op56pqr
at://did:plc:abc123/io.atdata.lens/2fg4hi78jkl
```

### Python Constants

```python
# src/atdata/atproto/_constants.py

SCHEMA_NSID = "io.atdata.schema"
DATASET_NSID = "io.atdata.dataset"
LENS_NSID = "io.atdata.lens"

# Can be changed in one place if we migrate namespaces
```

## Domain Ownership

**Question**: Do we need to own `atdata.io`?

**ATProto Spec**: NSIDs don't require domain ownership, but it's recommended for credibility.

**Options**:
1. **Register `atdata.io`** (~$12/year)
   - Pro: Professional, verifiable ownership
   - Con: Small cost
   - Recommendation: ✅ Do this

2. **Use without owning**
   - Pro: Free
   - Con: Someone else could register it and claim the namespace
   - Recommendation: ❌ Too risky

**Decision**: Register `atdata.io` domain

## Versioning in NSIDs

**Question**: Should version be part of NSID?

### Option A: Version in record (RECOMMENDED)
```
NSIDs: io.atdata.schema (constant)
Versions: In schema record "version" field
```

**Pros**:
- ✅ Stable NSIDs
- ✅ Versions can evolve independently
- ✅ Single collection for all versions

**Cons**:
- ❌ Need to look up version from record

### Option B: Version in NSID
```
NSIDs: io.atdata.schema.v1, io.atdata.schema.v2
```

**Pros**:
- ✅ Version explicit in URI

**Cons**:
- ❌ New NSID for each major version
- ❌ More Lexicons to maintain
- ❌ Harder to query across versions

**Recommendation**: Option A (version in record)

## Namespace Migration Plan

If we need to migrate from `io.atdata.*` to `app.bsky.atdata.*`:

### Migration Steps

1. **Dual Publishing** (transition period)
   ```python
   # Publish to both namespaces
   publisher.publish_schema(
       sample_type,
       nsid="io.atdata.schema"  # Old
   )
   publisher.publish_schema(
       sample_type,
       nsid="app.bsky.atdata.schema"  # New
   )
   ```

2. **Deprecation Notice**
   - Announce migration timeline
   - Update documentation
   - Add warnings to old namespace

3. **Update Client**
   - Default to new namespace
   - Still support old namespace (read-only)

4. **Sunset Old Namespace**
   - After 6-12 months, stop publishing to old namespace
   - Keep reading old records for compatibility

### Record Linking

Add migration metadata:
```json
{
  "$type": "app.bsky.atdata.schema",
  "metadata": {
    "migratedFrom": "at://did:plc:abc123/io.atdata.schema/3jk2lo34klm"
  },
  ...
}
```

## Additional Lexicons (Future)

Should we reserve NSIDs for future use?

**Potential Additions**:
- `io.atdata.collection` - Group multiple datasets
- `io.atdata.benchmark` - Evaluation results
- `io.atdata.annotation` - User comments/ratings
- `io.atdata.pipeline` - Data processing pipelines

**Recommendation**: Don't create yet, but document reserved names

## Community Input

**Before finalizing**:
1. Check if `io.atdata.*` is available (no conflicts)
2. Reach out to ATProto community (Discord, GitHub)
3. Ask Bluesky team about `app.bsky.atdata.*` feasibility
4. Document decision and rationale

## Open Questions

1. **Should we create a demo namespace first?**
   - `io.atdata.dev.schema` for testing?
   - Pro: Keeps production namespace clean
   - Con: More namespaces to manage
   - Recommendation: Not needed, use test DIDs instead

2. **What about language-specific namespaces?**
   - `io.atdata.py.schema` for Python-specific schemas?
   - Pro: Allows language-specific features
   - Con: Fragments ecosystem
   - Recommendation: ❌ Keep language-agnostic

3. **Should we namespace by domain (vision, NLP, etc.)?**
   - `io.atdata.vision.schema`, `io.atdata.nlp.schema`?
   - Pro: Better organization for large ecosystems
   - Con: Premature optimization
   - Recommendation: ❌ Not for Phase 1

## Success Criteria

After implementing this decision:
- ✅ NSIDs are finalized and documented
- ✅ Lexicon JSON files use correct NSIDs
- ✅ Python code uses constant definitions (easy to change)
- ✅ Migration plan exists if needed
- ✅ Domain `atdata.io` is registered (or plan to register)

## References

- ATProto NSID spec: https://atproto.com/specs/nsid
- Lexicon design: `../02_lexicon_design.md`
- All three Lexicon definitions need this decision

---

**Decision Needed By**: Before starting Phase 1 Issue #22, #23, #24 (all Lexicon designs)
**Decision Maker**: Project maintainer (max)
**Date Created**: 2026-01-07

## Recommended Action

**Immediate**:
1. ✅ Decide on `io.atdata.*` as working namespace
2. ✅ Plan to register `atdata.io` domain
3. ✅ Document migration path to `app.bsky.atdata.*` if desired later

**Before Phase 2**:
1. Register `atdata.io` domain
2. Optional: Reach out to Bluesky about `app.bsky.atdata.*` for future

**Phase 1**:
Use `io.atdata.*` in all Lexicon designs
