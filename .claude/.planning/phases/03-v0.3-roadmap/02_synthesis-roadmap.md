# v0.3 Synthesis Roadmap

**Synthesized from:**
- `01_codebase-review.md` - Technical assessment
- `architecture-doc.md` - Data processing backend vision
- Chainlink issues backlog

---

## Vision Statement

v0.3 transforms atdata from a **dataset library** into a **dataset operations platform** by adding:

1. **Queryable manifests** - Filter/aggregate without loading data
2. **Serverless processing** - Modal-native parallel transforms
3. **Event-driven pipelines** - R2 notifications trigger processing
4. **Production tooling** - CLI, observability, error handling

---

## Feature Domains

### Domain 1: Manifest System

**Source**: architecture-doc.md sections on "Per-Shard Manifest Design" and "Indexing and Query Strategy"

**Goal**: Enable efficient queries over large datasets without full scans.

#### 1.1 Manifest Generation

```python
# During shard write
with ShardWriter(..., manifest=True) as sink:
    for sample in ds.ordered():
        sink.write(sample.as_wds)
# Automatically writes:
#   shard-00042.tar
#   shard-00042.manifest.json       (header + aggregates)
#   shard-00042.manifest.parquet    (per-sample metadata)
```

**Implementation pieces:**
- `ManifestBuilder` class tracking samples during write
- `ManifestField` definition in schema (which fields are queryable)
- YAML/JSON header format for shard metadata
- Parquet writer for per-sample queryable fields

#### 1.2 Manifest Schema Integration

```python
@packable
class ImageSample:
    image: NDArray               # Not in manifest (too large)
    label: str                   # manifest: categorical
    confidence: float            # manifest: numeric
    tags: list[str]              # manifest: set
```

**Implementation pieces:**
- `@manifest_field` decorator or field annotation
- Auto-derive manifest fields from schema
- Aggregate type inference (categorical/numeric/set)

#### 1.3 Query Executor

```python
# Find samples with confidence > 0.9 and label in ['dog', 'cat']
results = ds.query(
    where=lambda m: m.confidence > 0.9 and m.label in ['dog', 'cat']
)

# Returns SampleLocations (shard + offset) for direct access
for loc in results:
    sample = ds.get_sample(loc)
```

**Implementation pieces:**
- `ManifestLoader` for reading across shards
- `QueryPlan` with shard pruning via aggregates
- `SampleLocation` type for sample addressing
- Optional DuckDB backend for SQL queries

#### Dependencies: None (foundational)

---

### Domain 2: Processing Backend (Modal)

**Source**: architecture-doc.md sections on "Core Stack", "System Architecture", "Modal Code Structure"

**Goal**: Enable scalable map-reduce over datasets using Modal.

#### 2.1 Processor Decorator

```python
import atdata
from modal import App

app = App("my-processor")

@atdata.processor(app, cpu=2, memory=4096)
def embed_images(sample: ImageSample) -> EmbeddingSample:
    embedding = model.encode(sample.image)
    return EmbeddingSample(
        embedding=embedding,
        label=sample.label,
    )
```

**Implementation pieces:**
- `@atdata.processor` wrapping Modal function
- Automatic shard distribution via `.map()`
- Schema type extraction from function signature
- Output shard writing with manifest

#### 2.2 Dataset.map() Method

```python
# Transform dataset using Modal workers
output_ds = input_ds.map(
    embed_images,
    output_prefix="s3://bucket/embedded/v1",
    workers_per_shard=1,
)
```

**Implementation pieces:**
- Shard enumeration and dispatch
- Worker result collection
- Output dataset registration
- Progress tracking

#### 2.3 Dispatcher Pattern

```python
@app.function()
@modal.web_endpoint(method="POST")
async def dispatch(request: Request):
    """Central dispatcher for job coordination."""
    body = await request.json()
    shards = list_shards(body["input_prefix"])

    # Fan out to workers
    results = process_shard.map(shards)
    return {"status": "started", "num_shards": len(shards)}
```

**Implementation pieces:**
- `atdata.modal.Dispatcher` class
- Job tracking and result aggregation
- Error handling with partial results
- Retry semantics

#### Dependencies: Domain 1 (manifests for output)

---

### Domain 3: Event-Driven Pipeline (R2 Notifications)

**Source**: architecture-doc.md section on "R2 Event Notifications → Modal"

**Goal**: Trigger processing automatically when data arrives.

#### 3.1 Cloudflare Worker Bridge

```javascript
// workers/r2-to-modal.js
export default {
  async queue(batch, env) {
    for (const msg of batch.messages) {
      await fetch(`${env.MODAL_ENDPOINT}/ingest`, {
        method: "POST",
        body: JSON.stringify({
          bucket: msg.body.bucket,
          object: msg.body.object.key,
          action: msg.body.action,
        }),
      });
      msg.ack();
    }
  }
};
```

**Implementation pieces:**
- Reference Cloudflare Worker template
- Environment variable configuration guide
- Modal endpoint for receiving events
- Event filtering logic (ignore non-.tar files)

#### 3.2 Pipeline Definition

```python
@atdata.pipeline(
    trigger="r2://bucket/incoming",
    output="r2://bucket/processed",
)
def process_uploads(shard_url: str):
    ds = Dataset[RawSample](shard_url)
    return ds.map(transform_sample)
```

**Implementation pieces:**
- Pipeline registration system
- Trigger pattern matching
- Output destination configuration
- Idempotency/deduplication

#### Dependencies: Domain 2 (Modal processing)

---

### Domain 4: Developer Experience

**Source**: Codebase review findings

**Goal**: Make atdata pleasant to use day-to-day.

#### 4.1 CLI Enhancements

```bash
# Inspect dataset
atdata inspect s3://bucket/data.tar
# Output: sample count, schema, shard info

# Show schema
atdata schema show @local/my-dataset
# Output: field types, versions, description

# Compare schemas
atdata schema diff v1.0.0 v1.1.0

# Preview samples
atdata preview s3://bucket/data.tar --limit 5
# Output: rendered samples with truncated arrays
```

**Implementation pieces:**
- CLI module extension (`cli/inspect.py`, `cli/schema.py`, `cli/preview.py`)
- Rich terminal output formatting
- S3/local/atmosphere URL resolution

#### 4.2 Dataset Convenience Methods

```python
# Quick preview
samples = ds.head(10)

# Sample by key
sample = ds.get("sample_00042")

# Schema access
fields = ds.schema.fields
version = ds.schema.version

# Statistics (from manifest)
stats = ds.describe()
```

**Implementation pieces:**
- `Dataset.head()` method
- `Dataset.get()` for keyed access
- `Dataset.schema` property
- `Dataset.describe()` using manifest

#### 4.3 Error Messages

Current:
```
ValueError: No registered lens from source <class '__main__.A'> to view <class '__main__.B'>
```

Improved:
```
LensNotFoundError: No lens transforms A → B

Available lenses from A:
  - A → C (via name_lens)

Did you mean to define:
  @lens
  def a_to_b(a: A) -> B:
      return B(...)
```

**Implementation pieces:**
- Custom exception hierarchy
- Suggestion generation
- Contextual help in errors

#### Dependencies: None (can proceed in parallel)

---

### Domain 5: Production Hardening

**Source**: Codebase review findings + architecture-doc.md reliability concerns

**Goal**: Make atdata suitable for production workloads.

#### 5.1 Observability

```python
# Structured logging
import structlog
atdata.configure_logging(structlog.get_logger())

# Metrics
atdata.metrics.samples_processed.inc()
atdata.metrics.shard_processing_time.observe(duration)

# Tracing
with atdata.trace("process_shard", shard_id=shard):
    ...
```

**Implementation pieces:**
- Pluggable logging interface
- Optional metrics (prometheus-client compatible)
- OpenTelemetry trace integration

#### 5.2 Error Handling

```python
# Partial failures
try:
    results = ds.map(processor)
except PartialFailureError as e:
    succeeded = e.succeeded_shards
    failed = e.failed_shards
    # Retry only failed shards
    retry_results = ds.map(processor, shards=failed, retry=True)
```

**Implementation pieces:**
- `PartialFailureError` with detailed info
- Checkpoint/resume support
- Dead-letter queue for persistent failures

#### 5.3 Testing Infrastructure

```python
# Mock atmosphere client for tests
@pytest.fixture
def mock_atmosphere():
    with atdata.testing.mock_atmosphere() as client:
        yield client

# Test dataset fixtures
def test_processing(tmp_dataset):
    result = tmp_dataset.map(my_processor)
    assert len(result.head(10)) == 10
```

**Implementation pieces:**
- `atdata.testing` module
- Mock clients and indices
- Dataset fixtures
- Integration test patterns

#### Dependencies: None (can proceed in parallel)

---

## Phased Implementation

### Phase A: Foundation (Weeks 1-3)

**Focus**: Manifest system + Dev experience basics

| Task | Domain | Complexity | Dependencies |
|------|--------|------------|--------------|
| ManifestBuilder class | 1 | Medium | None |
| Manifest field annotations | 1 | Low | ManifestBuilder |
| Dataset.head() method | 4 | Low | None |
| CLI inspect command | 4 | Medium | None |
| Custom exception hierarchy | 4 | Low | None |

**Exit criteria:**
- Can write datasets with manifests
- Can inspect datasets via CLI
- Better error messages in common cases

### Phase B: Query + Processing (Weeks 4-6)

**Focus**: Manifest queries + Modal integration

| Task | Domain | Complexity | Dependencies |
|------|--------|------------|--------------|
| ManifestLoader | 1 | Medium | Phase A |
| Query executor with pruning | 1 | High | ManifestLoader |
| @atdata.processor decorator | 2 | Medium | None |
| Dataset.map() method | 2 | High | @processor |
| Modal dispatcher | 2 | Medium | Dataset.map() |

**Exit criteria:**
- Can query datasets without loading all data
- Can run parallel transforms on Modal
- End-to-end: ingest → transform → query

### Phase C: Pipelines + Production (Weeks 7-9)

**Focus**: Event-driven + production hardening

| Task | Domain | Complexity | Dependencies |
|------|--------|------------|--------------|
| R2 Worker template | 3 | Low | None |
| Pipeline definition API | 3 | Medium | Modal |
| Observability hooks | 5 | Medium | None |
| Error handling improvements | 5 | Medium | None |
| Testing infrastructure | 5 | Medium | None |

**Exit criteria:**
- Can trigger processing on R2 upload
- Structured logging and metrics available
- Integration test suite passing

### Phase D: Polish (Weeks 10-11)

**Focus**: Documentation, edge cases, performance

| Task | Domain | Complexity | Dependencies |
|------|--------|------------|--------------|
| End-to-end tutorial | - | Medium | All phases |
| API reference completion | - | Medium | Phase C |
| Performance benchmarks | 5 | Medium | Phase B |
| Edge case fixes | - | Variable | All phases |

**Exit criteria:**
- Documentation complete
- Benchmark results documented
- Release candidate ready

---

## Risk Assessment

### Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Modal API changes | Medium | High | Pin versions, abstract interface |
| Manifest format lock-in | Low | Medium | Version field, migration tools |
| R2 notification delays | Low | Low | Polling fallback |
| Parquet compatibility | Low | Low | Use pyarrow directly |

### Resource Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Scope creep | High | Medium | Strict phase gates |
| External dependency delays | Medium | Medium | Stub interfaces early |
| Testing infrastructure gaps | Medium | High | Invest in Phase A |

---

## Success Metrics

### Functional Metrics
- [ ] Manifest-based query completes 10x faster than full scan
- [ ] Modal processing achieves >80% worker utilization
- [ ] R2 → processing latency <30s (push mode)
- [ ] CLI commands execute in <1s for local datasets

### Quality Metrics
- [ ] Test coverage >80% for new code
- [ ] All public APIs documented with examples
- [ ] Zero known security vulnerabilities
- [ ] Error messages include actionable suggestions

### Adoption Metrics
- [ ] Tutorial completion rate tracked
- [ ] GitHub issues for UX friction points
- [ ] Community dataset publications via atmosphere

---

## Appendix: Architecture-doc.md Key Quotes

> "Scientists and data producers interact via thin clients (CLI, notebooks, simple UI) without managing infra day-to-day"

This drives Domain 4 (Developer Experience).

> "Natural parallelization unit: One shard = one worker, no coordination needed"

This validates the Modal shard-per-worker model in Domain 2.

> "Per-shard manifests ... enables shard skipping without reading sample details"

This is the core insight for Domain 1.

> "Emit a manifest alongside each shard during processing"

This suggests manifest generation should be automatic during writes.

---

## Next Steps

1. Review and approve this roadmap
2. Create chainlink issues for Phase A tasks
3. Begin implementation of ManifestBuilder
4. Set up Modal project for processing experiments
