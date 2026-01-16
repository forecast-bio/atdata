# Promotion Workflow

The promotion workflow migrates datasets from local storage (Redis + S3) to the ATProto atmosphere network, enabling federation and discovery.

## Overview

Promotion handles:
- **Schema deduplication**: Avoids publishing duplicate schemas
- **Data URL preservation**: Keeps existing S3 URLs or copies to new storage
- **Metadata transfer**: Preserves tags, descriptions, and custom metadata

## Basic Usage

```python
from atdata.local import LocalIndex
from atdata.atmosphere import AtmosphereClient
from atdata.promote import promote_to_atmosphere

# Setup
local_index = LocalIndex()
client = AtmosphereClient()
client.login("handle.bsky.social", "app-password")

# Get local entry
entry = local_index.get_entry_by_name("my-dataset")

# Promote to atmosphere
at_uri = promote_to_atmosphere(entry, local_index, client)
print(f"Published: {at_uri}")
```

## With Metadata

```python
at_uri = promote_to_atmosphere(
    entry,
    local_index,
    client,
    name="my-dataset-v2",           # Override name
    description="Training images",  # Add description
    tags=["images", "training"],    # Add discovery tags
    license="MIT",                  # Specify license
)
```

## Schema Deduplication

The promotion workflow automatically checks for existing schemas:

```python
# First promotion: publishes schema
uri1 = promote_to_atmosphere(entry1, local_index, client)

# Second promotion with same schema type + version: reuses existing schema
uri2 = promote_to_atmosphere(entry2, local_index, client)
```

Schema matching is based on:
- `{module}.{class_name}` (e.g., `mymodule.ImageSample`)
- Version string (e.g., `1.0.0`)

## Data Storage Options

### Use Existing URLs (Default)

By default, promotion keeps the original data URLs:

```python
# Data stays in original S3 location
at_uri = promote_to_atmosphere(entry, local_index, client)
```

### Copy to New Storage

To copy data to a different storage location:

```python
from atdata.local import S3DataStore

# Create new data store
new_store = S3DataStore(
    credentials="new-s3-creds.env",
    bucket="public-datasets",
)

# Promote with data copy
at_uri = promote_to_atmosphere(
    entry,
    local_index,
    client,
    data_store=new_store,  # Copy data to new storage
)
```

## Complete Workflow Example

```python
import numpy as np
from numpy.typing import NDArray
import atdata
from atdata.local import LocalIndex, Repo
from atdata.atmosphere import AtmosphereClient
from atdata.promote import promote_to_atmosphere
import webdataset as wds

# 1. Define sample type
@atdata.packable
class FeatureSample:
    features: NDArray
    label: int

# 2. Create local dataset
samples = [
    FeatureSample(
        features=np.random.randn(128).astype(np.float32),
        label=i % 10,
    )
    for i in range(1000)
]

with wds.writer.TarWriter("features.tar") as sink:
    for i, s in enumerate(samples):
        sink.write({**s.as_wds, "__key__": f"{i:06d}"})

# 3. Store in local repo
repo = Repo(
    s3_credentials={
        "AWS_ENDPOINT": "http://localhost:9000",
        "AWS_ACCESS_KEY_ID": "minioadmin",
        "AWS_SECRET_ACCESS_KEY": "minioadmin",
    },
    hive_path="datasets-bucket/features",
)

dataset = atdata.Dataset[FeatureSample]("features.tar")
local_entry, _ = repo.insert(dataset, name="feature-vectors-v1")

# 4. Publish schema to local index
local_index = LocalIndex()
local_index.publish_schema(FeatureSample, version="1.0.0")

# 5. Promote to atmosphere
client = AtmosphereClient()
client.login("myhandle.bsky.social", "app-password")

at_uri = promote_to_atmosphere(
    local_entry,
    local_index,
    client,
    description="Feature vectors for classification",
    tags=["features", "embeddings"],
    license="MIT",
)

print(f"Dataset published: {at_uri}")

# 6. Verify on atmosphere
from atdata.atmosphere import AtmosphereIndex

atm_index = AtmosphereIndex(client)
entry = atm_index.get_dataset(at_uri)
print(f"Name: {entry.name}")
print(f"Schema: {entry.schema_ref}")
print(f"URLs: {entry.data_urls}")
```

## Error Handling

```python
try:
    at_uri = promote_to_atmosphere(entry, local_index, client)
except KeyError as e:
    # Schema not found in local index
    print(f"Missing schema: {e}")
except ValueError as e:
    # Entry has no data URLs
    print(f"Invalid entry: {e}")
```

## Requirements

Before promotion:
1. Dataset must be in local index (via `Repo.insert()` or `Index.add_entry()`)
2. Schema must be published to local index (via `Index.publish_schema()`)
3. AtmosphereClient must be authenticated

## Related

- [Local Storage](local-storage.md) - Setting up local datasets
- [Atmosphere](atmosphere.md) - ATProto integration
- [Protocols](protocols.md) - AbstractIndex and AbstractDataStore
