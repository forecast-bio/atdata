# Protocols

The protocols module defines abstract interfaces that enable interchangeable index backends (local Redis vs ATProto) and data stores (S3 vs PDS blobs).

## Overview

Both local and atmosphere implementations solve the same problem: indexed dataset storage with external data URLs. These protocols formalize that common interface:

- **IndexEntry**: Common interface for dataset index entries
- **AbstractIndex**: Protocol for index operations
- **AbstractDataStore**: Protocol for data storage operations

## IndexEntry Protocol

Represents a dataset entry in any index:

```python
from atdata._protocols import IndexEntry

def process_entry(entry: IndexEntry) -> None:
    print(f"Name: {entry.name}")
    print(f"Schema: {entry.schema_ref}")
    print(f"URLs: {entry.data_urls}")
    print(f"Metadata: {entry.metadata}")
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `name` | `str` | Human-readable dataset name |
| `schema_ref` | `str` | Schema reference (local:// or at://) |
| `data_urls` | `list[str]` | WebDataset URLs for the data |
| `metadata` | `dict \| None` | Arbitrary metadata dictionary |

### Implementations

- `LocalDatasetEntry` (from `atdata.local`)
- `AtmosphereIndexEntry` (from `atdata.atmosphere`)

## AbstractIndex Protocol

Defines operations for managing schemas and datasets:

```python
from atdata._protocols import AbstractIndex

def list_all_datasets(index: AbstractIndex) -> None:
    """Works with LocalIndex or AtmosphereIndex."""
    for entry in index.list_datasets():
        print(f"{entry.name}: {entry.schema_ref}")
```

### Dataset Operations

```python
# Insert a dataset
entry = index.insert_dataset(
    dataset,
    name="my-dataset",
    schema_ref="local://schemas/MySample@1.0.0",  # optional
)

# Get by name/reference
entry = index.get_dataset("my-dataset")

# List all datasets
for entry in index.list_datasets():
    print(entry.name)
```

### Schema Operations

```python
# Publish a schema
schema_ref = index.publish_schema(
    MySample,
    version="1.0.0",
)

# Get schema record
schema = index.get_schema(schema_ref)
print(schema["name"], schema["version"])

# List all schemas
for schema in index.list_schemas():
    print(f"{schema['name']}@{schema['version']}")

# Decode schema to Python type
SampleType = index.decode_schema(schema_ref)
dataset = atdata.Dataset[SampleType](entry.data_urls[0])
```

### Implementations

- `LocalIndex` / `Index` (from `atdata.local`)
- `AtmosphereIndex` (from `atdata.atmosphere`)

## AbstractDataStore Protocol

Abstracts over different storage backends:

```python
from atdata._protocols import AbstractDataStore

def write_dataset(store: AbstractDataStore, dataset) -> list[str]:
    """Works with S3DataStore or future PDS blob store."""
    urls = store.write_shards(dataset, prefix="datasets/v1")
    return urls
```

### Methods

```python
# Write dataset shards
urls = store.write_shards(
    dataset,
    prefix="datasets/mnist/v1",
    maxcount=10000,  # samples per shard
)

# Resolve URL for reading
readable_url = store.read_url("s3://bucket/path.tar")

# Check streaming support
if store.supports_streaming():
    # Can stream directly
    pass
```

### Implementations

- `S3DataStore` (from `atdata.local`)

## Using Protocols for Polymorphism

Write code that works with any backend:

```python
from atdata._protocols import AbstractIndex, IndexEntry
from atdata import Dataset

def backup_all_datasets(
    source: AbstractIndex,
    target: AbstractIndex,
) -> None:
    """Copy all datasets from source index to target."""
    for entry in source.list_datasets():
        # Decode schema from source
        SampleType = source.decode_schema(entry.schema_ref)

        # Publish schema to target
        target_schema = target.publish_schema(SampleType)

        # Load and re-insert dataset
        ds = Dataset[SampleType](entry.data_urls[0])
        target.insert_dataset(
            ds,
            name=entry.name,
            schema_ref=target_schema,
        )
```

## Schema Reference Formats

Schema references vary by backend:

| Backend | Format | Example |
|---------|--------|---------|
| Local | `local://schemas/{module.Class}@{version}` | `local://schemas/myapp.ImageSample@1.0.0` |
| Atmosphere | `at://{did}/{collection}/{rkey}` | `at://did:plc:abc123/ac.foundation.dataset.sampleSchema/xyz` |

## Type Checking

Protocols are runtime-checkable:

```python
from atdata._protocols import IndexEntry, AbstractIndex

# Check if object implements protocol
entry = index.get_dataset("test")
assert isinstance(entry, IndexEntry)

# Type hints work with protocols
def process(index: AbstractIndex) -> None:
    ...  # IDE provides autocomplete
```

## Complete Example

```python
import atdata
from atdata.local import LocalIndex, S3DataStore
from atdata.atmosphere import AtmosphereClient, AtmosphereIndex
from atdata._protocols import AbstractIndex
import numpy as np
from numpy.typing import NDArray

# Define sample type
@atdata.packable
class FeatureSample:
    features: NDArray
    label: int

# Function works with any index
def count_datasets(index: AbstractIndex) -> int:
    return sum(1 for _ in index.list_datasets())

# Use with local index
local_index = LocalIndex()
print(f"Local datasets: {count_datasets(local_index)}")

# Use with atmosphere index
client = AtmosphereClient()
client.login("handle.bsky.social", "app-password")
atm_index = AtmosphereIndex(client)
print(f"Atmosphere datasets: {count_datasets(atm_index)}")

# Migrate from local to atmosphere
def migrate_dataset(
    name: str,
    source: AbstractIndex,
    target: AbstractIndex,
) -> None:
    entry = source.get_dataset(name)
    SampleType = source.decode_schema(entry.schema_ref)

    # Publish schema
    schema_ref = target.publish_schema(SampleType)

    # Create dataset and insert
    ds = atdata.Dataset[SampleType](entry.data_urls[0])
    target.insert_dataset(ds, name=name, schema_ref=schema_ref)

migrate_dataset("my-features", local_index, atm_index)
```

## Related

- [Local Storage](local-storage.md) - LocalIndex and S3DataStore
- [Atmosphere](atmosphere.md) - AtmosphereIndex
- [Promotion](promotion.md) - Local to atmosphere migration
- [load_dataset](load-dataset.md) - Using indexes with load_dataset()
