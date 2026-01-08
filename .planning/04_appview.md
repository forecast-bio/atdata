# AppView Service Architecture

## Overview

The AppView is an **optional aggregation service** that indexes dataset records from across the ATProto network, providing fast search and discovery. Think of it as the "search engine" for atdata datasets.

## Why AppView?

Without AppView, discovering datasets requires:
- Querying each user's Personal Data Server (PDS) individually
- No global search across all published datasets
- Slow, inefficient discovery

With AppView:
- **Fast global search** across all datasets
- **Rich metadata browsing** (schemas, tags, authors)
- **Recommendation systems** (similar datasets, popular datasets)
- **Analytics** (dataset usage, trends)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     ATProto Network                          │
│                                                              │
│  ┌─────┐  ┌─────┐  ┌─────┐           ┌──────────────┐      │
│  │ PDS │  │ PDS │  │ PDS │  ────────▶ │  Relay/      │      │
│  │  1  │  │  2  │  │  3  │           │  Firehose    │      │
│  └─────┘  └─────┘  └─────┘           └──────────────┘      │
│     │         │         │                    │              │
│     └─────────┴─────────┴────────────────────┘              │
│              (publish records)                 │              │
└────────────────────────────────────────────────┼──────────────┘
                                                 │
                                                 │ (subscribe)
                                                 ▼
                                    ┌─────────────────────────┐
                                    │   AppView Service       │
                                    │                         │
                                    │  ┌──────────────────┐   │
                                    │  │  Firehose        │   │
                                    │  │  Consumer        │   │
                                    │  └────────┬─────────┘   │
                                    │           │             │
                                    │           ▼             │
                                    │  ┌──────────────────┐   │
                                    │  │  Record          │   │
                                    │  │  Processor       │   │
                                    │  └────────┬─────────┘   │
                                    │           │             │
                                    │           ▼             │
                                    │  ┌──────────────────┐   │
                                    │  │  PostgreSQL      │   │
                                    │  │  Database        │   │
                                    │  └──────────────────┘   │
                                    │                         │
                                    │  ┌──────────────────┐   │
                                    │  │  Search Index    │   │
                                    │  │  (ElasticSearch) │   │
                                    │  └──────────────────┘   │
                                    │                         │
                                    │  ┌──────────────────┐   │
                                    │  │  HTTP API        │   │
                                    │  │  (FastAPI)       │   │
                                    │  └──────────────────┘   │
                                    └─────────────────────────┘
                                              │
                                              │ (query API)
                                              ▼
                                    ┌─────────────────────────┐
                                    │  Python Client          │
                                    │  (atdata.atproto)       │
                                    └─────────────────────────┘
```

## Components

### 1. Firehose Consumer

**Purpose**: Subscribe to ATProto firehose and receive real-time record updates

**Technology**: Python + `atproto` SDK

**Responsibilities**:
- Connect to ATProto relay/firehose
- Filter for relevant Lexicon types:
  - `app.bsky.atdata.schema`
  - `app.bsky.atdata.dataset`
  - `app.bsky.atdata.lens`
- Handle reconnection and backpressure
- Forward records to processor

**Implementation**:
```python
from atproto import FirehoseSubscribeReposClient, parse_subscribe_repos_message

class AtdataFirehoseConsumer:
    def __init__(self, processor: RecordProcessor):
        self.processor = processor
        self.client = FirehoseSubscribeReposClient()

    def start(self):
        """Start consuming firehose."""
        def on_message_handler(message):
            commit = parse_subscribe_repos_message(message)
            if not commit:
                return

            for op in commit.ops:
                if op.action == 'create' or op.action == 'update':
                    if op.path.startswith('app.bsky.atdata.'):
                        # Extract record
                        record = op.record
                        self.processor.process_record(
                            uri=op.uri,
                            cid=op.cid,
                            record=record
                        )

        self.client.start(on_message_handler)
```

### 2. Record Processor

**Purpose**: Parse and validate incoming records, update database and search index

**Responsibilities**:
- Validate records against Lexicon schemas
- Extract searchable fields
- Resolve references (schema URIs, etc.)
- Update PostgreSQL and ElasticSearch
- Handle deletions and updates

**Data Model**:

**PostgreSQL Tables**:
```sql
-- Schema records
CREATE TABLE schemas (
    uri TEXT PRIMARY KEY,
    cid TEXT NOT NULL,
    did TEXT NOT NULL,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    description TEXT,
    fields JSONB NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP NOT NULL,
    indexed_at TIMESTAMP NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_schemas_did ON schemas(did);
CREATE INDEX idx_schemas_name ON schemas(name);

-- Dataset records
CREATE TABLE datasets (
    uri TEXT PRIMARY KEY,
    cid TEXT NOT NULL,
    did TEXT NOT NULL,
    name TEXT NOT NULL,
    schema_ref TEXT NOT NULL REFERENCES schemas(uri),
    urls TEXT[] NOT NULL,
    description TEXT,
    metadata BYTEA,
    tags TEXT[],
    license TEXT,
    size_samples INTEGER,
    size_bytes BIGINT,
    created_at TIMESTAMP NOT NULL,
    indexed_at TIMESTAMP NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_datasets_did ON datasets(did);
CREATE INDEX idx_datasets_schema ON datasets(schema_ref);
CREATE INDEX idx_datasets_tags ON datasets USING GIN(tags);

-- Lens records
CREATE TABLE lenses (
    uri TEXT PRIMARY KEY,
    cid TEXT NOT NULL,
    did TEXT NOT NULL,
    name TEXT NOT NULL,
    source_schema TEXT NOT NULL REFERENCES schemas(uri),
    target_schema TEXT NOT NULL REFERENCES schemas(uri),
    description TEXT,
    created_at TIMESTAMP NOT NULL,
    indexed_at TIMESTAMP NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_lenses_source ON lenses(source_schema);
CREATE INDEX idx_lenses_target ON lenses(target_schema);

-- Lens network view (for finding transformation paths)
CREATE MATERIALIZED VIEW lens_network AS
SELECT
    source_schema,
    target_schema,
    uri,
    name
FROM lenses;
CREATE INDEX idx_lens_network_source ON lens_network(source_schema);
CREATE INDEX idx_lens_network_target ON lens_network(target_schema);
```

**ElasticSearch Index**:
```json
{
  "mappings": {
    "properties": {
      "uri": { "type": "keyword" },
      "type": { "type": "keyword" },
      "did": { "type": "keyword" },
      "name": { "type": "text", "fields": { "keyword": { "type": "keyword" } } },
      "description": { "type": "text" },
      "tags": { "type": "keyword" },
      "created_at": { "type": "date" },
      "schema_ref": { "type": "keyword" },
      "license": { "type": "keyword" }
    }
  }
}
```

### 3. HTTP API

**Purpose**: Expose search and query endpoints for clients

**Technology**: FastAPI + Pydantic

**Endpoints**:

```python
from fastapi import FastAPI, Query
from pydantic import BaseModel

app = FastAPI()

# Search datasets
@app.get("/api/v1/datasets/search")
async def search_datasets(
    q: str = Query(None, description="Text search query"),
    tags: list[str] = Query(None, description="Filter by tags"),
    schema_uri: str = Query(None, description="Filter by schema"),
    author_did: str = Query(None, description="Filter by author DID"),
    limit: int = Query(20, le=100),
    offset: int = Query(0)
) -> list[dict]:
    """Search for datasets."""
    # Query ElasticSearch + PostgreSQL
    pass

# Get dataset details
@app.get("/api/v1/datasets/{uri:path}")
async def get_dataset(uri: str) -> dict:
    """Get dataset record by URI."""
    # Query PostgreSQL
    pass

# List schemas
@app.get("/api/v1/schemas")
async def list_schemas(
    limit: int = Query(20, le=100),
    offset: int = Query(0)
) -> list[dict]:
    """List available schemas."""
    pass

# Get schema details
@app.get("/api/v1/schemas/{uri:path}")
async def get_schema(uri: str) -> dict:
    """Get schema record by URI."""
    pass

# Find lens path between schemas
@app.get("/api/v1/lenses/path")
async def find_lens_path(
    source: str = Query(..., description="Source schema URI"),
    target: str = Query(..., description="Target schema URI")
) -> list[dict]:
    """Find transformation path between two schemas."""
    # Graph search on lens_network
    pass

# Stats and analytics
@app.get("/api/v1/stats")
async def get_stats() -> dict:
    """Get aggregate statistics."""
    return {
        "total_datasets": await count_datasets(),
        "total_schemas": await count_schemas(),
        "total_lenses": await count_lenses()
    }
```

### 4. Caching Layer

**Purpose**: Reduce database load for frequent queries

**Technology**: Redis

**Cached Items**:
- Popular dataset queries
- Schema lookups (high read frequency)
- Search results (with short TTL)
- Aggregate statistics

**Implementation**:
```python
import redis
import json
from functools import wraps

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache_result(ttl: int = 300):
    """Decorator to cache function results in Redis."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key from function name and args
            cache_key = f"{func.__name__}:{hash((args, frozenset(kwargs.items())))}"

            # Check cache
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)

            # Compute result
            result = await func(*args, **kwargs)

            # Store in cache
            redis_client.setex(cache_key, ttl, json.dumps(result))

            return result
        return wrapper
    return decorator

@cache_result(ttl=60)
async def get_popular_datasets():
    """Get popular datasets (cached for 1 minute)."""
    # Query database
    pass
```

## Deployment

### Infrastructure

**Option 1: Simple (single server)**
```
- PostgreSQL (datasets, schemas, lenses)
- ElasticSearch (search index)
- Redis (cache)
- FastAPI app (HTTP API)
- Firehose consumer (background process)
```

**Option 2: Scalable (cloud)**
```
- AWS RDS PostgreSQL (managed database)
- AWS OpenSearch (managed ElasticSearch)
- AWS ElastiCache (managed Redis)
- AWS ECS/Fargate (containerized FastAPI app)
- AWS ECS/Fargate (containerized firehose consumer)
- AWS ALB (load balancer)
```

### Docker Compose (Development)

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: atdata_appview
      POSTGRES_USER: atdata
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
    volumes:
      - es_data:/usr/share/elasticsearch/data
    ports:
      - "9200:9200"

  redis:
    image: redis:7
    ports:
      - "6379:6379"

  appview-api:
    build:
      context: .
      dockerfile: Dockerfile.api
    environment:
      DATABASE_URL: postgresql://atdata:password@postgres/atdata_appview
      ELASTICSEARCH_URL: http://elasticsearch:9200
      REDIS_URL: redis://redis:6379
    depends_on:
      - postgres
      - elasticsearch
      - redis
    ports:
      - "8000:8000"

  appview-firehose:
    build:
      context: .
      dockerfile: Dockerfile.firehose
    environment:
      DATABASE_URL: postgresql://atdata:password@postgres/atdata_appview
      ELASTICSEARCH_URL: http://elasticsearch:9200
      REDIS_URL: redis://redis:6379
      FIREHOSE_URL: wss://bsky.network/xrpc/com.atproto.sync.subscribeRepos
    depends_on:
      - postgres
      - elasticsearch
      - redis

volumes:
  postgres_data:
  es_data:
```

## Client Integration

### Python Client Updates

Add AppView support to `atdata.atproto.dataset.DatasetLoader`:

```python
class DatasetLoader:
    def __init__(
        self,
        client: ATProtoClient,
        appview_url: Optional[str] = None
    ):
        self.client = client
        self.appview_url = appview_url or "https://appview.atdata.network"

    def search_datasets(
        self,
        query: Optional[str] = None,
        tags: Optional[list[str]] = None,
        schema_uri: Optional[str] = None,
        limit: int = 20
    ) -> list[dict]:
        """Search datasets using AppView."""
        import httpx

        params = {"limit": limit}
        if query:
            params["q"] = query
        if tags:
            params["tags"] = tags
        if schema_uri:
            params["schema_uri"] = schema_uri

        response = httpx.get(f"{self.appview_url}/api/v1/datasets/search", params=params)
        response.raise_for_status()
        return response.json()
```

**Usage**:
```python
from atdata.atproto import ATProtoClient, DatasetLoader

client = ATProtoClient()
loader = DatasetLoader(client, appview_url="https://appview.atdata.network")

# Search for computer vision datasets
results = loader.search_datasets(
    tags=["computer-vision"],
    limit=10
)

for dataset in results:
    print(f"{dataset['name']}: {dataset['description']}")
```

## Performance Considerations

### Indexing Speed
- **Goal**: Index records in <1 second from firehose receipt
- **Approach**: Async processing, batch inserts

### Search Performance
- **Goal**: Search queries return in <100ms
- **Approach**: ElasticSearch indexing, query optimization, caching

### Scalability
- **Goal**: Handle 1000+ datasets, 100+ schemas
- **Approach**: Horizontal scaling of API servers, read replicas for PostgreSQL

## Monitoring & Observability

### Metrics
- Firehose lag (time behind current)
- Indexing throughput (records/second)
- API request latency (p50, p95, p99)
- Cache hit rate
- Database query performance

### Logging
- Structured JSON logs
- Log aggregation (e.g., CloudWatch, Datadog)
- Error tracking (e.g., Sentry)

### Health Checks
```python
@app.get("/health")
async def health_check():
    """Check service health."""
    return {
        "status": "healthy",
        "components": {
            "database": await check_db_health(),
            "elasticsearch": await check_es_health(),
            "redis": await check_redis_health(),
            "firehose": await check_firehose_health()
        }
    }
```

## Implementation Checklist (Phase 3)

- [ ] Design database schema (PostgreSQL)
- [ ] Design search index (ElasticSearch)
- [ ] Implement firehose consumer
- [ ] Implement record processor with validation
- [ ] Implement HTTP API with FastAPI
- [ ] Add caching layer (Redis)
- [ ] Create Docker Compose for local development
- [ ] Write integration tests
- [ ] Set up monitoring and logging
- [ ] Deploy to staging environment
- [ ] Performance testing and optimization

## Future Enhancements

### Advanced Search
- Fuzzy matching
- Relevance scoring
- Autocomplete for tags/names

### Recommendations
- "Datasets similar to this one"
- "Popular datasets in this category"
- "Datasets by authors you follow"

### Analytics
- Dataset usage tracking (downloads, views)
- Trending datasets
- Schema adoption statistics

### Social Features
- Dataset comments/reviews
- Ratings
- Curation lists (e.g., "Best datasets for X")

### Federation
- Multiple AppView instances
- Cross-AppView search
- Regional AppViews for performance

## Security Considerations

- **Rate limiting**: Prevent abuse of search API
- **Input validation**: Sanitize all query parameters
- **DDoS protection**: Use CloudFlare or similar
- **Authentication** (optional): API keys for heavy users
- **Data validation**: Verify record signatures from ATProto
