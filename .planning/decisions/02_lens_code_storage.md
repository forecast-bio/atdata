# Decision: Lens Code Storage Approach

**Issue**: #46
**Status**: Needs decision
**Blocks**: #50 (Lexicon validation)
**Priority**: Critical for Phase 1

## DECISION

Let's go with Option 1, using external repositories. We can actually make this work for

* GitHub
* tangled.org (the native ATProto git repository system)

Additionally, we'll want to keep track of metadata for lenses giving the language the referenced code is implemented in.

Longer-term, it will also be good to add another Lexicon specification for attestation of `Lens` formal correctness (where possible), as this will enable filtering lens implementations by provability. We'll also want to add our own `verification` records that give attestation of individual atproto DIDs (user identities) as being "trusted" for creating `Lens`es, etc.

---

## Problem Statement

We need to decide how to store the transformation code for Lens records on ATProto. Lenses define bidirectional transformations between sample types (getter: Source → Target, putter: Target × Source → Source).

This is a **critical security decision** because we're dealing with executable code.

## Context

Lens transformations are functions that:
- Take samples of one type and transform them to another
- Are bidirectional (getter + putter)
- Need to be reproducible and verifiable
- Potentially execute on untrusted data

Example Lens:
```python
@atdata.lens
def rgb_to_grayscale(rgb_sample: RGBSample) -> GrayscaleSample:
    gray = cv2.cvtColor(rgb_sample.image, cv2.COLOR_RGB2GRAY)
    return GrayscaleSample(image=gray, label=rgb_sample.label)

@rgb_to_grayscale.putter
def grayscale_to_rgb(gray: GrayscaleSample, rgb: RGBSample) -> RGBSample:
    # Convert back to RGB (approximate)
    rgb_img = cv2.cvtColor(gray.image, cv2.COLOR_GRAY2RGB)
    return RGBSample(image=rgb_img, label=gray.label)
```

## Options

### Option 1: Code References Only (GitHub/GitLab + Commit Hash) ⭐ RECOMMENDED

**Description**: Store only references to code in version control repositories

**Record Format**:
```json
{
  "getterCode": {
    "kind": "reference",
    "repository": "https://github.com/alice/lenses",
    "commit": "a1b2c3d4e5f6789...",
    "path": "lenses/vision.py:rgb_to_grayscale"
  },
  "putterCode": {
    "kind": "reference",
    "repository": "https://github.com/alice/lenses",
    "commit": "a1b2c3d4e5f6789...",
    "path": "lenses/vision.py:grayscale_to_rgb"
  }
}
```

**Pros**:
- ✅ **Secure**: No arbitrary code execution from ATProto records
- ✅ **Verifiable**: Commit hash ensures immutability
- ✅ **Auditable**: Users can review code before using
- ✅ **Version controlled**: Natural versioning through git
- ✅ **Professional workflow**: Encourages proper development practices

**Cons**:
- ❌ External dependency (repo could disappear)
- ❌ Requires users to have code in public/accessible repos
- ❌ Need to clone/fetch repos to use lenses
- ❌ Less convenient than self-contained records

**Security**: ⭐⭐⭐⭐⭐ Excellent
**Convenience**: ⭐⭐⭐ Good
**Implementation Effort**: Low-Medium

---

### Option 2: Inline Python Code with Sandboxing

**Description**: Store Python source code directly in records, execute in sandbox

**Record Format**:
```json
{
  "getterCode": {
    "kind": "python",
    "source": "def rgb_to_grayscale(rgb_sample: RGBSample) -> GrayscaleSample:\n    ..."
  }
}
```

**Pros**:
- ✅ Self-contained records
- ✅ No external dependencies
- ✅ More convenient for users
- ✅ Easier discovery and exploration

**Cons**:
- ❌ **MAJOR SECURITY RISK**: Executing untrusted code
- ❌ Sandboxing Python is extremely difficult
- ❌ Even with sandboxing, attack surface is large
- ❌ `eval()`/`exec()` considered harmful
- ❌ Would need extensive review and testing
- ❌ Potential for malicious code injection

**Security**: ⭐ Very Poor (even with sandboxing)
**Convenience**: ⭐⭐⭐⭐⭐ Excellent
**Implementation Effort**: Very High (sandboxing is complex)

**Why Sandboxing is Hard**:
- Python has many ways to break out of sandboxes
- Import system, file I/O, network access all need blocking
- `__import__`, `eval`, `exec`, `compile`, `open`, etc.
- Even readonly access can leak sensitive data
- See: [PyPy sandbox](https://doc.pypy.org/en/latest/sandbox.html) - discontinued

---

### Option 3: Bytecode or AST Representation

**Description**: Store compiled bytecode or AST instead of source

**Pros**:
- ✅ Slightly safer than raw source (no syntax injection)
- ✅ Self-contained

**Cons**:
- ❌ Still executes arbitrary code - same security issues
- ❌ Harder to audit than source
- ❌ Platform/version dependent (Python bytecode changes)
- ❌ Complex to implement
- ❌ Doesn't solve the fundamental problem

**Security**: ⭐⭐ Poor
**Convenience**: ⭐⭐ Poor (less readable)
**Implementation Effort**: High

---

### Option 4: Metadata Only (Manual Implementation)

**Description**: Store only metadata about transformations, require manual implementation

**Record Format**:
```json
{
  "description": "Converts RGB images to grayscale",
  "getterSignature": "(RGBSample) -> GrayscaleSample",
  "putterSignature": "(GrayscaleSample, RGBSample) -> RGBSample"
}
```

**Pros**:
- ✅ Completely safe
- ✅ Simple to implement

**Cons**:
- ❌ Lenses not actually usable
- ❌ Defeats the purpose of publishing transformations
- ❌ No network effect (can't compose lenses)

**Security**: ⭐⭐⭐⭐⭐ Excellent
**Convenience**: ⭐ Very Poor
**Implementation Effort**: Very Low

## Recommendation: Option 1 (Code References Only)

**Rationale**:

1. **Security First**: We cannot compromise on security. Publishing executable code to a public network is extremely dangerous without proper safeguards.

2. **Verifiable and Auditable**: With commit hashes, users can:
   - Review the exact code before execution
   - Verify it hasn't been tampered with
   - Make informed trust decisions

3. **Professional Workflow**: Requiring code in version control:
   - Encourages good practices (testing, documentation)
   - Makes lens development collaborative
   - Enables code review

4. **Future Extensibility**: We can add inline code later if we solve sandboxing, but we can't easily remove it once added.

## Implementation Plan

If we choose Option 1:

1. **Lexicon Design** (Phase 1)
   ```json
   "transformCode": {
     "type": "union",
     "refs": ["#codeReference"]
   },
   "codeReference": {
     "type": "object",
     "required": ["kind", "repository", "commit", "path"],
     "properties": {
       "kind": {"type": "string", "const": "reference"},
       "repository": {"type": "string", "maxLength": 500},
       "commit": {"type": "string", "maxLength": 40},
       "path": {"type": "string", "maxLength": 500}
     }
   }
   ```

2. **Lens Publisher** (Phase 2)
   - Automatically detect git repo and commit from function location
   - Validate that repo is accessible
   - Include function name and module path

3. **Lens Loader** (Phase 2)
   - Clone/fetch repository at specified commit
   - Import function from specified path
   - Cache cloned repos locally
   - Verify function signatures match schema

4. **Trust Model**
   - Users explicitly approve which repos to trust
   - Whitelist/blacklist mechanism
   - Warn on first use of any lens

## Alternative Approaches Considered

**Signed inline code**: Store inline code with cryptographic signatures
- Still has execution risk
- Signature only proves authorship, not safety
- Not recommended

**WASM modules**: Compile transformations to WebAssembly
- More sandboxed than Python
- Very complex to implement
- Would require rewriting lenses in Rust/C++
- Interesting future direction but not for Phase 1

## User Experience Implications

**Publishing a Lens**:
```python
# 1. Write lens code in your repo
# lenses/vision.py
@atdata.lens
def rgb_to_grayscale(rgb: RGBSample) -> GrayscaleSample:
    ...

# 2. Commit and push
git add lenses/vision.py
git commit -m "Add RGB to grayscale lens"
git push

# 3. Publish to ATProto (automatically detects git info)
client = ATProtoClient()
client.login("alice.bsky.social", "password")

lens_publisher = LensPublisher(client)
lens_uri = lens_publisher.publish_lens(
    rgb_to_grayscale,
    source_schema_uri="at://alice/schema/rgb",
    target_schema_uri="at://alice/schema/gray"
)
```

**Using a Lens**:
```python
# 1. Discover lens
loader = LensLoader(client)
lenses = loader.search_lenses(
    source_schema="at://alice/schema/rgb",
    target_schema="at://alice/schema/gray"
)

# 2. User reviews the repo/code (outside tool)
# 3. User approves the repo

# 4. Load and use lens
rgb_to_gray = loader.load_lens(lenses[0]['uri'])
gray_sample = rgb_to_gray(rgb_sample)
```

## Security Considerations

Even with code references:
- **Malicious repos**: Users could reference repos with malicious code
- **Mitigation**: Explicit user approval, warnings, sandboxing (future)

- **Repo compromise**: Git repos could be hacked
- **Mitigation**: Commit hash pins exact version, users can audit

- **Dependency injection**: Lens code could import malicious packages
- **Mitigation**: Users review code, standard Python security practices

## Future Enhancements

**If we want inline code later**:
1. Build robust Python sandbox (e.g., using PyPy, restrictedpython)
2. Add extensive security testing
3. Implement strict permissions model
4. Use WebAssembly for true isolation
5. Add code signing and reputation system

**For now**: Start with references, prove the concept, add inline code only if there's strong demand and we can do it safely.

## Open Questions

1. **Private repositories**: How to handle lenses in private repos?
   - Could support auth tokens (stored locally, not in record)
   - Could use SSH keys
   - Recommendation: Public repos only for Phase 1

2. **Repository availability**: What if repo goes offline?
   - Could encourage mirrors
   - Could cache code (with user permission)
   - Recommendation: Accept the risk, it's part of decentralization

3. **Non-Python lenses**: What about TypeScript, Rust, etc.?
   - References work for any language
   - Each language would need its own loader
   - Recommendation: Python-only for Phase 1

## Success Criteria

After implementing this decision:
- ✅ Lenses can be published with code references
- ✅ Users can load and execute lenses from approved repos
- ✅ No arbitrary code execution from untrusted sources
- ✅ Lens records include immutable commit hashes
- ✅ Clear warnings when using external code

## References

- Lexicon design: `../02_lexicon_design.md` (Lens Record Lexicon)
- Python client implementation: `../03_python_client.md` (LensPublisher)
- Security best practices: Python security guide

---

**Decision Needed By**: Before starting Phase 1 Issue #24 (Lens Lexicon design)
**Decision Maker**: Project maintainer (max)
**Date Created**: 2026-01-07
