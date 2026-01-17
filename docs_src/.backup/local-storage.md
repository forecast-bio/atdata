# Local Storage

The local storage module provides a Redis + S3 backend for storing and managing datasets before publishing to the ATProto federation.

## Overview

Local storage uses:
- **Redis** for indexing and tracking dataset metadata
- **S3-compatible storage** for dataset tar files

This enables development and small-scale deployment before promoting to the full ATProto infrastructure.

## LocalIndex

The index tracks datasets in Redis:

```python
from atdata.local import LocalIndex

# Default connection (localhost:6379)
index = LocalIndex()

# Custom Redis connection
import redis
r = redis.Redis(host='custom-host', port=6379)
index = LocalIndex(redis=r)

# With connection kwargs
index = LocalIndex(host='custom-host', port=6379, db=1)
```

### Adding Entries

```python
dataset = atdata.Dataset[ImageSample]("data-{000000..000009}.tar")

entry = index.add_entry(
    dataset,
    name="my-dataset",
    schema_ref="local://schemas/mymodule.ImageSample@1.0.0",  # optional
    metadata={"description": "Training images"},              # optional
)

print(entry.cid)        # Content identifier
print(entry.name)       # "my-dataset"
print(entry.data_urls)  # ["data-{000000..000009}.tar"]
```

### Listing and Retrieving

```python
# Iterate all entries
for entry in index.entries:
    print(f"{entry.name}: {entry.cid}")

# Get as list
all_entries = index.all_entries

# Get by name
entry = index.get_entry_by_name("my-dataset")

# Get by CID
entry = index.get_entry("bafyrei...")
```

## Repo

The Repo class combines S3 storage with Redis indexing:

```python
from atdata.local import Repo

# From credentials file
repo = Repo(
    s3_credentials="path/to/.env",
    hive_path="my-bucket/datasets",
)

# From credentials dict
repo = Repo(
    s3_credentials={
        "AWS_ENDPOINT": "http://localhost:9000",
        "AWS_ACCESS_KEY_ID": "minioadmin",
        "AWS_SECRET_ACCESS_KEY": "minioadmin",
    },
    hive_path="my-bucket/datasets",
)
```

### Credentials File Format

The `.env` file should contain:

```
AWS_ENDPOINT=http://localhost:9000
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
```

For AWS S3, omit `AWS_ENDPOINT` to use the default endpoint.

### Inserting Datasets

```python
@atdata.packable
class ImageSample:
    image: NDArray
    label: str

# Create dataset from samples
samples = [ImageSample(...) for _ in range(1000)]
with wds.writer.TarWriter("temp.tar") as sink:
    for i, s in enumerate(samples):
        sink.write({**s.as_wds, "__key__": f"{i:06d}"})

dataset = atdata.Dataset[ImageSample]("temp.tar")

# Insert into repo (writes to S3 + indexes in Redis)
entry, stored_dataset = repo.insert(
    dataset,
    name="training-images-v1",
    cache_local=False,  # Stream directly to S3
)

print(entry.cid)                # Content identifier
print(stored_dataset.url)       # S3 URL for the stored data
print(stored_dataset.shard_list)  # Individual shard URLs
```

### Insert Options

```python
entry, ds = repo.insert(
    dataset,
    name="my-dataset",
    cache_local=True,   # Write locally first, then copy (faster for some workloads)
    maxcount=10000,     # Samples per shard
    maxsize=100_000_000,  # Max shard size in bytes
)
```

## LocalDatasetEntry

Index entries provide content-addressable identification:

```python
entry = index.get_entry_by_name("my-dataset")

# Core properties (IndexEntry protocol)
entry.name        # Human-readable name
entry.schema_ref  # Schema reference
entry.data_urls   # WebDataset URLs
entry.metadata    # Arbitrary metadata dict or None

# Content addressing
entry.cid         # ATProto-compatible CID (content identifier)

# Legacy compatibility
entry.wds_url     # First data URL
entry.sample_kind # Same as schema_ref
```

The CID is generated from the entry's content (schema_ref + data_urls), ensuring identical data produces identical CIDs whether stored locally or in the atmosphere.

## Schema Storage

Schemas can be stored and retrieved from the index:

```python
# Publish a schema
schema_ref = index.publish_schema(
    ImageSample,
    version="1.0.0",
    description="Image with label annotation",
)
# Returns: "local://schemas/mymodule.ImageSample@1.0.0"

# Retrieve schema record
schema = index.get_schema(schema_ref)
# {
#     "name": "ImageSample",
#     "version": "1.0.0",
#     "fields": [...],
#     "description": "...",
#     "createdAt": "...",
# }

# List all schemas
for schema in index.list_schemas():
    print(f"{schema['name']}@{schema['version']}")

# Reconstruct sample type from schema
SampleType = index.decode_schema(schema_ref)
dataset = atdata.Dataset[SampleType](entry.data_urls[0])
```

## S3DataStore

For direct S3 operations without Redis indexing:

```python
from atdata.local import S3DataStore

store = S3DataStore(
    credentials="path/to/.env",
    bucket="my-bucket",
)

# Write dataset shards
urls = store.write_shards(
    dataset,
    prefix="datasets/v1",
    maxcount=10000,
)
# Returns: ["s3://my-bucket/datasets/v1/data--uuid--000000.tar", ...]

# Check capabilities
store.supports_streaming()  # True
```

## Complete Workflow Example

```python
import numpy as np
from numpy.typing import NDArray
import atdata
from atdata.local import Repo, LocalIndex
import webdataset as wds

# 1. Define sample type
@atdata.packable
class TrainingSample:
    features: NDArray
    label: int
    source: str

# 2. Create samples
samples = [
    TrainingSample(
        features=np.random.randn(128).astype(np.float32),
        label=i % 10,
        source="synthetic",
    )
    for i in range(10000)
]

# 3. Write to local tar
with wds.writer.TarWriter("local-data.tar") as sink:
    for i, sample in enumerate(samples):
        sink.write({**sample.as_wds, "__key__": f"{i:06d}"})

# 4. Create repo and insert
repo = Repo(
    s3_credentials={
        "AWS_ENDPOINT": "http://localhost:9000",
        "AWS_ACCESS_KEY_ID": "minioadmin",
        "AWS_SECRET_ACCESS_KEY": "minioadmin",
    },
    hive_path="datasets-bucket/training",
)

local_ds = atdata.Dataset[TrainingSample]("local-data.tar")
entry, stored_ds = repo.insert(local_ds, name="training-v1")

# 5. Retrieve later
index = LocalIndex()
entry = index.get_entry_by_name("training-v1")
dataset = atdata.Dataset[TrainingSample](entry.data_urls[0])

for batch in dataset.ordered(batch_size=32):
    print(batch.features.shape)  # (32, 128)
```

## Related

- [Datasets](datasets.md) - Dataset iteration and batching
- [Protocols](protocols.md) - AbstractIndex and IndexEntry interfaces
- [Promotion](promotion.md) - Promoting local datasets to ATProto
- [Atmosphere](atmosphere.md) - ATProto federation
