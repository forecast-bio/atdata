# Expected Test Warnings

This document explains the expected warnings that are suppressed in the test suite using `@pytest.mark.filterwarnings` decorators.

## Design Philosophy

Per the project's testing conventions (see `CLAUDE.md`), warning suppression is kept **local to individual tests** rather than using global suppression in `conftest.py`. This approach:

1. Documents which specific tests have known warning behaviors
2. Makes it easier to track when warnings appear in unexpected places
3. Avoids masking genuine warnings from new code

## Warning Categories

### 1. s3fs/moto Async Incompatibility

**Warnings:**
```python
@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
@pytest.mark.filterwarnings("ignore:coroutine.*was never awaited:RuntimeWarning")
```

**Cause:** The `s3fs` library (used for S3 filesystem access) has async internals that don't fully clean up when used with `moto` (AWS mocking library) in a synchronous test context. When the test tears down, some async coroutines haven't been awaited, triggering these warnings.

**Affected tests:** Any test using the `mock_s3` fixture that interacts with `S3DataStore` or `S3FileSystem`.

**Impact:** None on test correctness. These are cleanup warnings that occur after the test has completed successfully.

**Resolution status:** This is a known interaction between s3fs and moto. A proper fix would require upstream changes to one or both libraries. The warnings are harmless and are expected behavior when using these libraries together.

### 2. Deprecated Repo Class

**Warning:**
```python
@pytest.mark.filterwarnings("ignore:Repo is deprecated:DeprecationWarning")
```

**Cause:** The `Repo` class in `atdata.local` is deprecated in favor of `Index`. Tests that verify backward compatibility or test the deprecated class directly will trigger this warning.

**Affected tests:** Tests in `TestRepoWorkflow`, `TestRepoDeprecation`, and any test explicitly using the `Repo` class.

**Impact:** None on test correctness. The deprecation warning is intentional to guide users toward `Index`.

**Resolution status:** These warnings will be removed when the `Repo` class is removed in a future major version. Until then, tests maintain backward compatibility verification.

## Adding New Warning Suppressions

When adding new `filterwarnings` markers:

1. **Verify the warning is expected** - Understand why the warning occurs and confirm it doesn't indicate a real problem
2. **Use specific patterns** - Target only the exact warning, not broad categories
3. **Document here** - Add an entry explaining the warning
4. **Keep it local** - Apply to individual tests, not globally

Example:
```python
@pytest.mark.filterwarnings("ignore:specific warning pattern:WarningType")
def test_something():
    ...
```

## Files with Warning Suppressions

- `tests/test_local.py` - s3fs/moto async warnings, Repo deprecation
- `tests/test_integration_local.py` - s3fs/moto async warnings, Repo deprecation

## Verifying Warnings Are Still Expected

Periodically check if upstream fixes have resolved these issues:

```bash
# Run tests without suppressions to see all warnings
uv run pytest tests/test_local.py -W default 2>&1 | grep -i warning
```
