# Atmosphere (ATProto Integration)

The atmosphere module enables publishing and discovering datasets on the ATProto network, creating a federated ecosystem for typed datasets.

## Installation

```bash
pip install atdata[atmosphere]
# or
pip install atproto
```

## Overview

ATProto integration publishes datasets, schemas, and lenses as records in the `ac.foundation.dataset.*` namespace. This enables:

- **Discovery** through the ATProto network
- **Federation** across different hosts
- **Verifiability** through content-addressable records

## AtmosphereClient

The client handles authentication and record operations:

```python
from atdata.atmosphere import AtmosphereClient

client = AtmosphereClient()

# Login with app-specific password (not your main password!)
client.login("alice.bsky.social", "app-password")

print(client.did)     # 'did:plc:...'
print(client.handle)  # 'alice.bsky.social'
```

### Session Management

Save and restore sessions to avoid re-authentication:

```python
# Export session for later
session_string = client.export_session()

# Later: restore session
new_client = AtmosphereClient()
new_client.login_with_session(session_string)
```

### Custom PDS

Connect to a custom PDS instead of bsky.social:

```python
client = AtmosphereClient(base_url="https://pds.example.com")
```

## AtmosphereIndex

The unified interface for ATProto operations, implementing the AbstractIndex protocol:

```python
from atdata.atmosphere import AtmosphereClient, AtmosphereIndex

client = AtmosphereClient()
client.login("handle.bsky.social", "app-password")

index = AtmosphereIndex(client)
```

### Publishing Schemas

```python
@atdata.packable
class ImageSample:
    image: NDArray
    label: str
    confidence: float

# Publish schema
schema_uri = index.publish_schema(
    ImageSample,
    version="1.0.0",
    description="Image classification sample",
)
# Returns: "at://did:plc:.../ac.foundation.dataset.sampleSchema/..."
```

### Publishing Datasets

```python
dataset = atdata.Dataset[ImageSample]("data-{000000..000009}.tar")

entry = index.insert_dataset(
    dataset,
    name="imagenet-subset",
    schema_ref=schema_uri,           # Optional - auto-publishes if omitted
    description="ImageNet subset",
    tags=["images", "classification"],
    license="MIT",
)

print(entry.uri)        # AT URI of the record
print(entry.data_urls)  # WebDataset URLs
```

### Listing and Retrieving

```python
# List your datasets
for entry in index.list_datasets():
    print(f"{entry.name}: {entry.schema_ref}")

# List from another user
for entry in index.list_datasets(repo="did:plc:other-user"):
    print(entry.name)

# Get specific dataset
entry = index.get_dataset("at://did:plc:.../ac.foundation.dataset.record/...")

# List schemas
for schema in index.list_schemas():
    print(f"{schema['name']} v{schema['version']}")

# Decode schema to Python type
SampleType = index.decode_schema(schema_uri)
```

## Lower-Level Publishers

For more control, use the individual publisher classes:

### SchemaPublisher

```python
from atdata.atmosphere import SchemaPublisher

publisher = SchemaPublisher(client)

uri = publisher.publish(
    ImageSample,
    name="ImageSample",
    version="1.0.0",
    description="Image with label",
    metadata={"source": "training"},
)
```

### DatasetPublisher

```python
from atdata.atmosphere import DatasetPublisher

publisher = DatasetPublisher(client)

uri = publisher.publish(
    dataset,
    name="training-images",
    schema_uri=schema_uri,           # Required if auto_publish_schema=False
    auto_publish_schema=True,        # Publish schema automatically
    description="Training images",
    tags=["training", "images"],
    license="MIT",
)
```

#### Blob Storage

For smaller datasets (up to ~50MB per shard), you can store data directly in ATProto blobs instead of external URLs:

```python
import io
import webdataset as wds

# Create tar data in memory
tar_buffer = io.BytesIO()
with wds.writer.TarWriter(tar_buffer) as sink:
    for i, sample in enumerate(samples):
        sink.write({**sample.as_wds, "__key__": f"{i:06d}"})

# Publish with blob storage
uri = publisher.publish_with_blobs(
    blobs=[tar_buffer.getvalue()],
    schema_uri=schema_uri,
    name="small-dataset",
    description="Dataset stored in ATProto blobs",
    tags=["small", "demo"],
)
```

To load datasets with blob storage:

```python
from atdata.atmosphere import DatasetLoader

loader = DatasetLoader(client)

# Check storage type
storage_type = loader.get_storage_type(uri)  # "external" or "blobs"

if storage_type == "blobs":
    # Get blob URLs for direct access
    blob_urls = loader.get_blob_urls(uri)

# to_dataset() handles both storage types automatically
dataset = loader.to_dataset(uri, MySample)
for batch in dataset.ordered(batch_size=32):
    process(batch)
```

### LensPublisher

```python
from atdata.atmosphere import LensPublisher

publisher = LensPublisher(client)

# With code references
uri = publisher.publish(
    name="simplify",
    source_schema=full_schema_uri,
    target_schema=simple_schema_uri,
    description="Extract label only",
    getter_code={
        "repository": "https://github.com/org/repo",
        "commit": "abc123def...",
        "path": "transforms/simplify.py:simplify_getter",
    },
    putter_code={
        "repository": "https://github.com/org/repo",
        "commit": "abc123def...",
        "path": "transforms/simplify.py:simplify_putter",
    },
)

# Or publish from a Lens object
from atdata.lens import lens

@lens
def simplify(src: FullSample) -> SimpleSample:
    return SimpleSample(label=src.label)

uri = publisher.publish_from_lens(
    simplify,
    source_schema=full_schema_uri,
    target_schema=simple_schema_uri,
)
```

## AT URIs

ATProto records are identified by AT URIs:

```python
from atdata.atmosphere import AtUri

# Parse an AT URI
uri = AtUri.parse("at://did:plc:abc123/ac.foundation.dataset.sampleSchema/xyz")

print(uri.authority)   # 'did:plc:abc123'
print(uri.collection)  # 'ac.foundation.dataset.sampleSchema'
print(uri.rkey)        # 'xyz'

# Format back to string
print(str(uri))  # 'at://did:plc:abc123/ac.foundation.dataset.sampleSchema/xyz'
```

## Record Types

### SchemaRecord

```python
from atdata.atmosphere import SchemaRecord, FieldDef, FieldType

schema = SchemaRecord(
    name="ImageSample",
    version="1.0.0",
    fields=[
        FieldDef(
            name="image",
            field_type=FieldType(kind="ndarray", dtype="float32"),
        ),
        FieldDef(
            name="label",
            field_type=FieldType(kind="primitive", primitive="str"),
        ),
    ],
    description="Image with label",
)

record_dict = schema.to_record()
```

### DatasetRecord

```python
from atdata.atmosphere import DatasetRecord, StorageLocation

dataset_record = DatasetRecord(
    name="training-images",
    schema_ref="at://did:plc:.../...",
    storage=StorageLocation(
        kind="external",
        urls=["s3://bucket/data-{000000..000009}.tar"],
    ),
    tags=["training"],
    license="MIT",
)
```

### LensRecord

```python
from atdata.atmosphere import LensRecord, CodeReference

lens_record = LensRecord(
    name="simplify",
    source_schema="at://did:plc:.../.../source",
    target_schema="at://did:plc:.../.../target",
    description="Simplify sample",
    getter_code=CodeReference(
        repository="https://github.com/org/repo",
        commit="abc123",
        path="transforms.py:simplify",
    ),
)
```

## Supported Field Types

Schemas support these field types:

| Python Type | ATProto Type |
|-------------|--------------|
| `str` | `primitive/str` |
| `int` | `primitive/int` |
| `float` | `primitive/float` |
| `bool` | `primitive/bool` |
| `bytes` | `primitive/bytes` |
| `NDArray` | `ndarray` (default dtype: float32) |
| `NDArray[np.float64]` | `ndarray` (dtype: float64) |
| `list[str]` | `array` with items |
| `T \| None` | Optional field |

## Complete Example

```python
import numpy as np
from numpy.typing import NDArray
import atdata
from atdata.atmosphere import AtmosphereClient, AtmosphereIndex
import webdataset as wds

# 1. Define and create samples
@atdata.packable
class FeatureSample:
    features: NDArray
    label: int
    source: str

samples = [
    FeatureSample(
        features=np.random.randn(128).astype(np.float32),
        label=i % 10,
        source="synthetic",
    )
    for i in range(1000)
]

# 2. Write to tar
with wds.writer.TarWriter("features.tar") as sink:
    for i, s in enumerate(samples):
        sink.write({**s.as_wds, "__key__": f"{i:06d}"})

# 3. Authenticate
client = AtmosphereClient()
client.login("myhandle.bsky.social", "app-password")

index = AtmosphereIndex(client)

# 4. Publish schema
schema_uri = index.publish_schema(
    FeatureSample,
    version="1.0.0",
    description="Feature vectors with labels",
)

# 5. Publish dataset
dataset = atdata.Dataset[FeatureSample]("features.tar")
entry = index.insert_dataset(
    dataset,
    name="synthetic-features-v1",
    schema_ref=schema_uri,
    tags=["features", "synthetic"],
)

print(f"Published: {entry.uri}")

# 6. Later: discover and load
for dataset_entry in index.list_datasets():
    print(f"Found: {dataset_entry.name}")

    # Reconstruct type from schema
    SampleType = index.decode_schema(dataset_entry.schema_ref)

    # Load dataset
    ds = atdata.Dataset[SampleType](dataset_entry.data_urls[0])
    for batch in ds.ordered(batch_size=32):
        print(batch.features.shape)
        break
```

## Related

- [Local Storage](local-storage.md) - Redis + S3 backend
- [Promotion](promotion.md) - Promoting local datasets to ATProto
- [Protocols](protocols.md) - AbstractIndex interface
- [Packable Samples](packable-samples.md) - Defining sample types
