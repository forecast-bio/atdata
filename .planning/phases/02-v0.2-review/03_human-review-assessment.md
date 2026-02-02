# Human Review Assessment & Implementation Plan

**Source**: `.review/human-review.md`
**Chainlink**: #374 (parent), #375-379 (subissues)
**Date**: 2026-01-26

---

## Issue 1: PackableSample → Packable Protocol Migration

**Chainlink**: #375

### Problem Statement

The `@packable` decorator creates a class that inherits from `PackableSample`, but type checkers don't recognize this inheritance at static analysis time. This causes linting errors when passing `@packable`-decorated classes to functions expecting `Type[Packable]`, such as `Index.publish_schema()`.

### Current State Analysis

```python
# Current: @packable creates a new class inheriting from PackableSample
@packable
class MyData:
    name: str

# Type checker sees: Type[MyData] (original class)
# Runtime sees: Type[MyData] where MyData inherits from PackableSample
```

The `Packable` protocol in `_protocols.py` is correctly defined as `@runtime_checkable`, and runtime checks work:

```python
>>> isinstance(MyData(name='x'), Packable)
True
```

But static type checkers don't see the transformed class structure.

### Pros and Cons of Migration

#### Option A: Keep `PackableSample` base class

**Pros:**
- Explicit inheritance visible in code
- IDE autocomplete works without additional configuration
- `__post_init__` hooks work naturally

**Cons:**
- `@packable` decorator doesn't play well with type checkers
- Dual patterns (`@packable` vs explicit inheritance) cause confusion
- Protocol-based signatures (`Type[Packable]`) don't lint cleanly

#### Option B: Pure Protocol-based approach

**Pros:**
- Structural typing is more Pythonic
- No inheritance needed - just implement the interface
- Works with any class that has the right methods/properties

**Cons:**
- Need to manually implement `from_data`, `from_bytes`, `packed`, `as_wds`
- Lose automatic `__post_init__` behavior
- More boilerplate for users

#### Option C: Hybrid with `@dataclass_transform` (Recommended)

**Pros:**
- Type checkers understand the transformation
- Keeps `@packable` as primary API
- Protocol checks work at both runtime and static analysis

**Cons:**
- Requires Python 3.11+ for full `@dataclass_transform` support
- May need plugin for older type checkers

### Implementation Plan

1. **Add `@dataclass_transform()` to `@packable`** (already partially done at line 935)
   - Verify it's working correctly with pyright/mypy

2. **Fix return type annotation** on `@packable`:
   ```python
   @dataclass_transform()
   def packable(cls: type[_T]) -> type[_T & Packable]:  # Intersection type
       ...
   ```

3. **Add Protocol verification** in tests:
   ```python
   def test_packable_satisfies_protocol():
       @packable
       class TestSample:
           x: int

       # Static type check (via reveal_type in comments)
       sample_type: Type[Packable] = TestSample  # Should not error
   ```

4. **Update `publish_schema` signature** to be more permissive:
   ```python
   def publish_schema(
       self,
       sample_type: Type[Packable],  # Keep this
       # OR use a TypeVar bound to Packable
       ...
   )
   ```

### Estimated Effort: 4-6 hours

---

## Issue 2: Redis Index Entry Expiration

**Chainlink**: #376

### Problem Statement

Redis is inconsistently removing index entries over time, even though no explicit TTL is set.

### Current State Analysis

Looking at the code in `local.py`:

```python
# Line 655 - Writing dataset entries
redis.hset(save_key, mapping=data)  # No TTL

# Line 1387 - Writing schemas
self._redis.set(redis_key, schema_json)  # No TTL
```

Redis configuration check shows:
```
maxmemory-policy: noeviction
maxmemory: 0 (unlimited)
```

With `noeviction` policy and unlimited memory, Redis should NOT automatically remove keys.

### Potential Causes

1. **External process clearing keys**: Another application or script may be running `FLUSHDB`/`FLUSHALL`

2. **Redis persistence configuration**:
   - If `RDB` or `AOF` persistence is disabled, data is lost on Redis restart
   - Check with: `redis-cli CONFIG GET save` and `redis-cli CONFIG GET appendonly`

3. **Different Redis instances**:
   - Development vs production Redis may have different configurations
   - Docker containers may be recreated with fresh data

4. **Key name collisions**:
   - If key prefix changed between versions, old keys appear "missing"

5. **Memory pressure** (unlikely with noeviction):
   - Even with noeviction, if Redis runs out of memory, writes fail

### Investigation Steps

```bash
# Check persistence settings
redis-cli CONFIG GET save
redis-cli CONFIG GET appendonly

# Check if keys exist (pattern match)
redis-cli KEYS "LocalDatasetEntry:*"
redis-cli KEYS "LocalSchema:*"

# Check memory
redis-cli INFO memory

# Monitor for FLUSHDB/DEL commands
redis-cli MONITOR
```

### Implementation Plan

1. **Add Redis health check to CLI**:
   ```python
   # atdata diagnose redis
   def diagnose_redis(redis: Redis) -> dict:
       return {
           "persistence": {
               "rdb_enabled": redis.config_get("save"),
               "aof_enabled": redis.config_get("appendonly"),
           },
           "memory": redis.info("memory"),
           "dataset_count": len(list(redis.scan_iter("LocalDatasetEntry:*"))),
           "schema_count": len(list(redis.scan_iter("LocalSchema:*"))),
       }
   ```

2. **Add logging on write/read**:
   ```python
   def write_to(self, redis: Redis):
       logger.debug(f"Writing entry {self.cid} to Redis key {save_key}")
       redis.hset(save_key, mapping=data)
       logger.debug(f"Verified entry exists: {redis.exists(save_key)}")
   ```

3. **Document Redis requirements**:
   - Add to README: "Requires Redis with persistence enabled"
   - Provide example `redis.conf` for production use

4. **Consider backup/restore utilities**:
   ```python
   # atdata local backup --output index-backup.json
   # atdata local restore --input index-backup.json
   ```

### Estimated Effort: 3-4 hours

---

## Issue 3: `xs` Property vs `list_xs()` Convention Audit

**Chainlink**: #377

### Convention Definition

```python
class Foo:
    @property
    def xs(self) -> Iterator[X]:
        """Lazy iteration over X items."""
        for x in self._get_xs():
            yield x

    def list_xs(self) -> list[X]:
        """Fully evaluated list of X items."""
        return list(self.xs)
```

### Current State Audit

| Class | Property | Method | Status |
|-------|----------|--------|--------|
| `Index` | `entries` (Generator) | `list_entries()` | ✅ Correct |
| `Index` | `datasets` (Generator) | `list_datasets()` | ✅ Correct |
| `Index` | `schemas` (Generator) | `list_schemas()` | ✅ Correct |
| `AtmosphereIndex` | `datasets` (Iterator) | `list_datasets()` | ✅ Correct |
| `AtmosphereIndex` | `schemas` (Iterator) | `list_schemas()` | ✅ Correct |
| `Dataset` | - | `list_shards()` | ⚠️ Missing `shards` property |
| `URLSource` | `shards` (Iterator) | `list_shards()` | ✅ Correct |
| `S3Source` | `shards` (Iterator) | `list_shards()` | ✅ Correct |
| `DataSource` (Protocol) | `shards` (Iterator) | `list_shards()` | ✅ Correct |

### Issues Found

1. **`Dataset` class** has `list_shards()` but no lazy `shards` property
2. **Legacy `shard_list` property** exists but is marked deprecated - should route to `list_shards()`
3. **`DatasetDict.num_shards`** uses `shard_list` internally - should use `list_shards()`

### Implementation Plan

1. **Add `Dataset.shards` property**:
   ```python
   @property
   def shards(self) -> Iterator[str]:
       """Lazily iterate over shard identifiers."""
       return iter(self._source.list_shards())
   ```

2. **Update `DatasetDict.num_shards`**:
   ```python
   @property
   def num_shards(self) -> dict[str, int]:
       return {name: len(ds.list_shards()) for name, ds in self.items()}
   ```

3. **Add deprecation warnings** to legacy properties:
   ```python
   @property
   def shard_list(self) -> list[str]:
       warnings.warn("shard_list is deprecated, use list_shards()", DeprecationWarning)
       return self.list_shards()
   ```

4. **Document convention** in CLAUDE.md:
   ```markdown
   ## Naming Conventions

   - `foo.xs` - @property returning Iterator/Generator (lazy)
   - `foo.list_xs()` - method returning list (eager)
   ```

### Estimated Effort: 2-3 hours

---

## Issue 4: `load_dataset` Source Credentials

**Chainlink**: #378

### Problem Statement

When `load_dataset` loads from an index with an S3 data store, the returned `Dataset` doesn't use the S3 credentials. Similarly, atproto-based loading should use the appropriate storage mechanism.

### Current State Analysis

In `_hf_api.py:620-627`:
```python
data_urls, schema_ref = _resolve_indexed_path(path, index)
# ...
url = _shards_to_wds_url(data_urls)
ds = Dataset[resolved_type](url)  # Uses URLSource (no credentials!)
```

The `_resolve_indexed_path` function does transform URLs through `data_store.read_url()`, but this only works for URL transformation (s3:// → https://), not for credential injection.

### Required Behavior

1. **Local index with S3DataStore**:
   - Extract credentials from `index.data_store`
   - Create `S3Source` with those credentials
   - Pass `S3Source` to `Dataset`

2. **AtmosphereIndex with blob storage**:
   - Resolve blob references to AT URIs
   - Create appropriate source (future `BlobSource`)
   - Pass source to `Dataset`

3. **Plain URLs** (no index or index without data_store):
   - Current behavior is correct (use `URLSource`)

### Implementation Plan

1. **Extend `_resolve_indexed_path` to return source**:
   ```python
   def _resolve_indexed_path(
       path: str,
       index: "AbstractIndex",
   ) -> tuple[DataSource, str]:
       """Resolve @handle/dataset path to DataSource and schema_ref."""
       handle_or_did, dataset_name = _parse_indexed_path(path)
       entry = index.get_dataset(dataset_name)

       # Build appropriate DataSource
       data_urls = entry.data_urls

       if hasattr(index, 'data_store') and index.data_store is not None:
           store = index.data_store
           if isinstance(store, S3DataStore):
               # Extract S3 credentials and create S3Source
               source = S3Source.from_urls(
                   data_urls,
                   endpoint=store.credentials.get('AWS_ENDPOINT'),
                   access_key=store.credentials.get('AWS_ACCESS_KEY_ID'),
                   secret_key=store.credentials.get('AWS_SECRET_ACCESS_KEY'),
               )
               return source, entry.schema_ref

       # Default: URL-based source
       url = _shards_to_wds_url(data_urls)
       source = URLSource(url)
       return source, entry.schema_ref
   ```

2. **Update `load_dataset` to use DataSource**:
   ```python
   if _is_indexed_path(path):
       # ...
       source, schema_ref = _resolve_indexed_path(path, index)
       resolved_type = sample_type if sample_type is not None else index.decode_schema(schema_ref)
       ds = Dataset[resolved_type](source)  # Pass source, not URL
       # ...
   ```

3. **Add `AbstractDataStore.create_source()` method** (optional but cleaner):
   ```python
   class AbstractDataStore(Protocol):
       def create_source(self, urls: list[str]) -> DataSource:
           """Create a DataSource for reading these URLs."""
           ...
   ```

4. **Future: AtmosphereIndex blob support**:
   ```python
   # In AtmosphereIndex or AtmosphereDataStore
   def create_source(self, urls: list[str]) -> DataSource:
       if all(url.startswith("at://") for url in urls):
           return BlobSource(urls, client=self.client)
       return URLSource(_shards_to_wds_url(urls))
   ```

### Estimated Effort: 4-6 hours

---

## Issue 5: `load_dataset` Overload Type Hints

**Chainlink**: #379

### Problem Statement

Calls like:
```python
ds = load_dataset("@local/data", TextSample, split='train', index=index)
```
produce linting errors because the `AbstractIndex` protocol doesn't align with `local.Index` for type checking.

### Current State Analysis

The overloads in `_hf_api.py:481-529` use:
```python
index: Optional["AbstractIndex"] = None
```

But `AbstractIndex` is a Protocol, and `local.Index` is a concrete class. The issue is likely one of:

1. **Protocol compatibility**: `Index` doesn't fully satisfy `AbstractIndex`
2. **Import issues**: `AbstractIndex` may not be recognized correctly
3. **Optional handling**: The `Optional[...]` wrapping may cause issues

### Investigation

Check if `Index` satisfies `AbstractIndex`:

```python
# AbstractIndex requires:
# - data_store: Optional[AbstractDataStore]
# - insert_dataset(...)
# - get_dataset(...)
# - datasets (property)
# - list_datasets()
# - publish_schema(...)
# - get_schema(...)
# - schemas (property)
# - list_schemas()
# - decode_schema(...)
```

Looking at `local.Index`, it has all these methods/properties. The issue is likely the `data_store` attribute type.

### Root Cause Hypothesis

In `_protocols.py:191`:
```python
data_store: Optional["AbstractDataStore"]
```

In `local.py:1006`:
```python
def data_store(self) -> AbstractDataStore | None:
```

The Protocol uses a class attribute annotation, but `Index` uses a property. Protocol properties vs class attributes can cause type checker confusion.

### Implementation Plan

1. **Fix Protocol `data_store` to be a property**:
   ```python
   class AbstractIndex(Protocol):
       @property
       def data_store(self) -> Optional["AbstractDataStore"]:
           """Optional data store for writing shards."""
           ...
   ```

2. **Verify all Protocol members match implementation**:
   - Run `pyright --verifytypes atdata` to check
   - Ensure return types match exactly

3. **Add explicit Protocol inheritance** (alternative approach):
   ```python
   # If structural typing continues to fail, use explicit registration
   from typing import runtime_checkable

   # In local.py
   AbstractIndex.register(Index)  # Explicit ABC registration
   ```

4. **Simplify overloads** if needed:
   ```python
   # Consider using @overload with Union types instead of Optional
   @overload
   def load_dataset(
       path: str,
       sample_type: Type[ST],
       *,
       split: str,
       index: AbstractIndex | None = None,  # Union instead of Optional
   ) -> Dataset[ST]: ...
   ```

5. **Add type test file**:
   ```python
   # tests/test_types.py (for pyright --verifytypes)
   from atdata import load_dataset, Dataset
   from atdata.local import Index
   from atdata._protocols import AbstractIndex

   def check_index_protocol(index: AbstractIndex) -> None:
       pass

   def test_index_satisfies_protocol():
       index = Index()
       check_index_protocol(index)  # Should not error
   ```

### Estimated Effort: 3-4 hours

---

## Summary & Prioritization

| Issue | Priority | Effort | Dependencies |
|-------|----------|--------|--------------|
| #378 load_dataset credentials | High | 4-6h | None |
| #379 Type hint fixes | High | 3-4h | None |
| #375 Packable protocol | Medium | 4-6h | None |
| #377 Naming convention | Low | 2-3h | None |
| #376 Redis investigation | Medium | 3-4h | User environment |

**Recommended order**: #378 → #379 → #375 → #377 → #376

The credentials issue (#378) is the most impactful for users. Type hints (#379) affect developer experience. The Redis issue (#376) requires user environment investigation first.

---

## Next Steps

1. Approve this assessment
2. Implement #378 (load_dataset credentials)
3. Implement #379 (type hints)
4. Revisit #375-377 based on priority
