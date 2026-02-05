# Data Processing Backend Architecture

This document outlines the architecture for a robust, serverless data processing backend designed to handle large-ish datasets while keeping scientists and data producers unencumbered by infrastructure concerns.

---

## Design Goals

1. **Invisible infrastructure**: Scientists and data producers interact via thin clients (CLI, notebooks, simple UI) without managing infra day-to-day
2. **Parallelizable processing**: Support map-reduce style workflows over sharded datasets
3. **Fire-and-forget execution**: Jobs complete asynchronously; results are queried from output storage
4. **Event-driven and manual triggers**: Support both automatic processing on data arrival and ad-hoc runs
5. **Flexible job granularity**: Different jobs can chunk work differently based on their needs

---

## Core Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Object Storage** | Cloudflare R2 | S3-compatible, zero egress fees, event notifications via Queues |
| **Compute** | Modal | Serverless Python functions, native `.map()` for parallelism, GPU support |
| **Data Format** | WebDataset (via atdata) | Sharded tar files, streaming-friendly, natural parallelization unit |
| **Job Queue** | None (Modal-native) | Modal's `.spawn()` and `.map()` provide sufficient queuing semantics |

### Why not an external queue (e.g., QStash)?

For this workload, Modal provides:
- `.spawn()` for fire-and-forget async calls
- `.map()` / `.starmap()` for parallel fan-out
- `@web_endpoint` to receive HTTP triggers
- Built-in retries at the function level

An external queue would add value for guaranteed delivery during Modal outages or complex rate limiting, but adds complexity. Since the source of truth is "what's in S3," failed jobs can be retried by re-triggering. The data remains in the bucket.

---

## System Architecture

```
┌─────────────────┐      ┌─────────────────┐
│  Scientist CLI  │      │  R2 Event       │
│  / Notebook     │      │  Notification   │
└────────┬────────┘      └────────┬────────┘
         │                        │
         │ HTTP POST              │ Cloudflare Queue → Worker
         ▼                        ▼
┌─────────────────────────────────────────────┐
│           Modal @web_endpoint               │
│         (job dispatcher / router)           │
└────────────────────┬────────────────────────┘
                     │
                     │ .map() / .starmap()
                     ▼
┌─────────────────────────────────────────────┐
│        Modal workers (CPU or GPU)           │
│   read from R2 → process → write to R2      │
└────────────────────┬────────────────────────┘
                     │
                     │ write alongside output shards
                     ▼
┌─────────────────────────────────────────────┐
│         Per-shard manifests + output        │
│        (enables indexing and queries)       │
└─────────────────────────────────────────────┘
```

---

## R2 Event Notifications → Modal

R2 event notifications are GA and send messages to Cloudflare Queues when bucket data changes. Two consumption patterns are available:

### Option A: Push via Cloudflare Worker (Recommended)

```
R2 bucket event → Cloudflare Queue → Consumer Worker → HTTP POST to Modal
```

A minimal Cloudflare Worker (~20 lines) forwards events to Modal's `@web_endpoint`. This provides:
- **Batching**: Aggregate multiple events before calling Modal
- **Filtering**: Ignore events you don't care about
- **Retry logic**: At the Cloudflare layer

Example Worker:

```javascript
export default {
  async queue(batch, env) {
    for (const msg of batch.messages) {
      const event = msg.body;
      await fetch("https://your-org--dispatcher.modal.run/ingest", {
        method: "POST",
        headers: { 
          "Content-Type": "application/json",
          "Authorization": `Bearer ${env.MODAL_SECRET}` 
        },
        body: JSON.stringify({
          bucket: event.bucket,
          object: event.object.key,
          action: event.action,
          timestamp: new Date().toISOString()
        })
      });
      msg.ack();
    }
  }
};
```

### Option B: Pull-based Consumer

A Modal scheduled function polls the queue periodically. Simpler (no Worker to maintain) but adds latency.

---

## Data Format: WebDataset with atdata Schemas

### Why WebDataset?

WebDataset stores data as sharded tar files where each sample consists of files sharing a common basename:

```
shard-00042.tar
├── sample_00000.jpg
├── sample_00000.json
├── sample_00001.jpg
├── sample_00001.json
└── ...
```

Benefits for this architecture:
- **Natural parallelization unit**: One shard = one worker, no coordination needed
- **Streaming-friendly**: Workers process tar files sequentially without loading into memory
- **S3-optimal**: Sequential reads, no random access penalty
- **Append is trivial**: New data = new shards, never rewrite existing data

### atdata Layer

atdata adds:
1. **Sample type schemas**: Each sample conforms to a known schema, enabling structured serialization/deserialization
2. **External schema management**: Schema is separate from data, allowing evolution

This gives excellent append semantics (just create new tar files) while maintaining type safety.

---

## Modal Code Structure

### Dispatcher

```python
@app.function()
@modal.web_endpoint(method="POST")
async def dispatch(request: Request):
    body = await request.json()
    input_prefix = body["input_prefix"]
    output_prefix = body["output_prefix"]
    job_type = body["job_type"]
    
    shards = list_shards(input_prefix)  # S3 list operation
    
    # Fan out to workers
    results = worker.map(
        [(s, output_prefix, job_type) for s in shards]
    )
    return {"status": "started", "num_shards": len(shards)}
```

### Worker

```python
@app.function(cpu=2, memory=4096, timeout=600)
def worker(shard_url: str, output_prefix: str, job_type: str):
    schema = get_schema_for_job(job_type)
    
    # Stream from R2, process, write output shard with manifest
    with ShardWriter(output_shard) as sink:
        for sample in stream_shard(shard_url, schema):
            processed = process_sample(sample, job_type)
            sink.write(processed)
    
    # Write manifest alongside shard
    write_manifest(output_shard, manifest)
```

### Shard Size Guidelines

Target shards that take 30 seconds to 5 minutes to process:
- **Too small**: Per-task overhead dominates (cold starts, container spin-up)
- **Too large**: Memory pressure, coarse retry granularity
- **Just right**: Good parallelism, efficient resource use

---

## Indexing and Query Strategy

The core tension: append-friendly storage (WebDataset) vs. queryable storage. The solution is a layered approach using per-shard manifests.

### The Indexing Spectrum

```
Less infrastructure                                      More infrastructure
       │                                                         │
       ▼                                                         ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Partitioned │  │  Per-shard   │  │  Centralized │  │  Materialized│
│  bucket paths│  │  manifests   │  │  metadata DB │  │  query views │
└──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘
```

### Recommended Phased Approach

**Phase 1: Per-shard manifests** (Start here)
- Emit a manifest alongside each shard during processing
- Simple query executor scans manifests and returns matching sample locations
- No external systems, manifests live with data

**Phase 2: Add aggregates for shard-level filtering**
- Include statistical summaries in manifests
- Skip entire shards that can't match a query (e.g., max_score < query_min_score)

**Phase 3: Materialized query views**
- Common queries produce new WebDatasets
- Results cached by query hash, reused across training runs

**Phase 4: Centralized DB (maybe never)**
- Only if manifest scanning becomes a bottleneck at millions of shards
- Or if you need complex queries (joins, full-text search)

---

## Per-Shard Manifest Design

Each shard gets a companion manifest with three layers of information:

### Layer 1: Shard-Level Metadata

Tiny header for coarse filtering—always read this first.

```yaml
shard_id: "dataset-v2/train/shard-00042"
schema_type: "image_classification"
schema_version: "2.3.0"

# Provenance
created_at: "2025-01-22T14:32:00Z"
source_job_id: "modal-job-abc123"
parent_shards: ["raw/shard-00010"]
pipeline_version: "1.4.2"

# Physical properties
num_samples: 1000
size_bytes: 524288000
checksum_sha256: "a1b2c3..."

# Temporal bounds
time_min: "2025-01-01T00:00:00Z"
time_max: "2025-01-15T23:59:59Z"
```

### Layer 2: Statistical Aggregates

Enable shard skipping without reading sample details.

**Categorical fields:**
```yaml
aggregates:
  label:
    type: categorical
    cardinality: 3
    values: {dog: 423, cat: 512, bird: 65}
```

**Numeric fields:**
```yaml
aggregates:
  confidence_score:
    type: numeric
    min: 0.12
    max: 0.98
    mean: 0.72
    histogram:
      buckets: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
      counts: [12, 45, 189, 402, 352]
```

**Set/tag fields:**
```yaml
aggregates:
  tags:
    type: set
    all_values: [outdoor, indoor, night, day, urban, rural]
    bloom_filter_base64: "..."  # For high-cardinality sets
```

### Layer 3: Per-Sample Metadata

Queryable fields for each sample in the shard.

```yaml
samples:
  - __key__: "sample_00000"
    __offset__: 0           # Byte offset for direct access
    __size__: 52480
    label: "dog"
    confidence: 0.92
    tags: [outdoor, day]
    
  - __key__: "sample_00001"
    __offset__: 52480
    __size__: 48192
    label: "cat"
    confidence: 0.87
    tags: [indoor]
```

**Field selection guidelines:**
- **Include**: Fields you'll filter on, aggregate on, or use for joins
- **Exclude**: Large blobs (images, embeddings), unique-per-sample values with no query utility

---

## Manifest Format Recommendations

### Hybrid Approach (Recommended)

```
shard-00042.tar                    # The data
shard-00042.manifest.json          # Header + aggregates (tiny, human-readable)
shard-00042.manifest.parquet       # Per-sample metadata (columnar, compressed)
```

### Why Parquet for Sample Metadata?

- **Columnar**: Read only the columns you need
- **Compressed**: 5-10x smaller than JSON
- **Ecosystem**: DuckDB, Polars, pandas read it natively
- **Aggregation**: Query across all manifests without loading into memory

```python
import duckdb

# Query across all manifests
result = duckdb.query("""
    SELECT label, COUNT(*) as count
    FROM 's3://bucket/dataset/**/*.manifest.parquet'
    WHERE confidence > 0.9
    GROUP BY label
""")
```

### Simple Alternative

For smaller scale (<10K samples per shard), plain JSON is fine:

```python
def write_manifest(manifest, path):
    if manifest.num_samples < 10_000:
        write_json(path + ".manifest.json", manifest)
    else:
        # Split for larger shards
        write_json(path + ".manifest.json", manifest.header_and_aggregates())
        write_jsonl(path + ".manifest.samples.jsonl", manifest.samples)
```

---

## Schema Integration

Derive manifest fields from atdata schemas:

```python
@dataclass
class ImageClassificationSchema:
    """atdata schema for image classification samples."""
    image: bytes
    label: str
    confidence: float
    tags: List[str]
    metadata: dict

def manifest_fields(schema: Type) -> List[ManifestField]:
    """Define which fields are queryable."""
    return [
        ManifestField("label", indexed=True, aggregate="categorical"),
        ManifestField("confidence", indexed=True, aggregate="numeric"),
        ManifestField("tags", indexed=True, aggregate="set"),
        # 'image' excluded - too large
        # 'metadata' excluded - unstructured
    ]
```

When you define a new schema type, you also define what's queryable. The manifest writer automatically extracts those fields.

---

## Write Path Integration

```python
@app.function()
def process_shard(input_shard: str, output_prefix: str, job_id: str):
    schema = get_schema_for_shard(input_shard)
    manifest_fields = schema.manifest_fields()
    
    manifest = ManifestBuilder(
        shard_id=output_shard_id,
        schema_type=schema.name,
        schema_version=schema.version,
        source_job_id=job_id,
        parent_shards=[input_shard],
    )
    
    output_shard = f"{output_prefix}/{output_shard_id}.tar"
    
    with ShardWriter(output_shard) as sink:
        for sample in stream_shard(input_shard, schema):
            processed = process_sample(sample)
            sink.write(processed)
            
            manifest.add_sample(
                key=processed["__key__"],
                offset=sink.current_offset,
                fields={f: processed[f] for f in manifest_fields}
            )
    
    manifest.finalize_aggregates()
    manifest.write(output_shard.replace(".tar", ""))
```

---

## Query Execution

```python
def execute_query(bucket_prefix: str, predicate: Callable) -> List[SampleLocation]:
    manifests = list_manifests(bucket_prefix)
    
    results = []
    for manifest in manifests:
        # Shard-level filtering using aggregates
        if not shard_might_match(manifest.aggregates, predicate):
            continue
        
        # Sample-level filtering
        for sample_meta in manifest.samples:
            if predicate(sample_meta):
                results.append(SampleLocation(
                    shard=manifest.shard_id,
                    key=sample_meta["__key__"],
                    offset=sample_meta.get("__offset__")
                ))
    
    return results

def shard_might_match(aggregates, predicate):
    """Quick check using aggregates before scanning samples."""
    if predicate.requires_label("dog") and "dog" not in aggregates["label"]["values"]:
        return False
    if predicate.min_score and aggregates["confidence_score"]["max"] < predicate.min_score:
        return False
    return True
```

---

## Open Design Questions

1. **Schema evolution in manifests**: When adding new queryable fields, how to handle old manifests?
   - Backfill (expensive but clean)
   - Nullable (queries skip old shards)
   - Versioned readers (handle missing fields gracefully)

2. **Manifest update story**: If manifests are immutable (written once with shard), life is simple. If updates are needed (e.g., adding derived fields later), need atomicity and versioning strategy.

3. **Sample ID structure**: If "find sample by ID" needs to be fast, consider structured IDs (e.g., `{shard_id}/{seq}`) so location is derivable, or maintain a separate key→location index.

---

## Future Considerations

### Materialized Query Views

For frequently-used queries, materialize results as new WebDatasets:

```python
@app.function()
def materialize_query(query: Query, output_prefix: str):
    matching_samples = execute_query(query)
    
    with ShardWriter(output_prefix, maxcount=10000) as sink:
        for sample in matching_samples:
            sink.write(sample)
    
    return {"output": output_prefix, "num_samples": sink.count}
```

Cache by query hash; reuse across training runs.

### Centralized Database

If manifest scanning becomes a bottleneck (millions of shards) or complex queries are needed (joins, full-text search), consider:
- **SQLite in S3**: Viable up to ~10M samples
- **DuckDB**: Better for analytical queries, can query Parquet directly
- **Postgres (Neon/Supabase)**: For concurrent writers, transactions, real-time updates

### Vector Search

If queries like "samples similar to this embedding" are needed, consider:
- Storing embedding vectors in manifests (if small enough)
- External vector DB (Pinecone, Qdrant, pgvector)
- Approximate nearest neighbor indexes alongside shards
