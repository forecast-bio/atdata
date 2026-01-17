# atdata

A loose federation of distributed, typed datasets built on WebDataset.

## What is atdata?

atdata provides a typed dataset abstraction for machine learning workflows with:

- **Typed samples** with automatic msgpack serialization
- **NDArray handling** with transparent numpy array conversion
- **Lens transformations** for viewing datasets through different schemas
- **Batch aggregation** with automatic numpy stacking
- **WebDataset integration** for efficient large-scale storage
- **ATProto federation** for publishing and discovering datasets

## Installation

```bash
pip install atdata

# With ATProto support
pip install atdata[atmosphere]
```

## Quick Start

### Define a Sample Type

```python
import numpy as np
from numpy.typing import NDArray
import atdata

@atdata.packable
class ImageSample:
    image: NDArray
    label: str
    confidence: float
```

### Create and Write Samples

```python
import webdataset as wds

samples = [
    ImageSample(
        image=np.random.rand(224, 224, 3).astype(np.float32),
        label="cat",
        confidence=0.95,
    )
    for _ in range(100)
]

with wds.writer.TarWriter("data-000000.tar") as sink:
    for i, sample in enumerate(samples):
        sink.write({**sample.as_wds, "__key__": f"sample_{i:06d}"})
```

### Load and Iterate

```python
dataset = atdata.Dataset[ImageSample]("data-000000.tar")

# Iterate with batching
for batch in dataset.shuffled(batch_size=32):
    images = batch.image      # numpy array (32, 224, 224, 3)
    labels = batch.label      # list of 32 strings
    confs = batch.confidence  # list of 32 floats
```

### Use Lenses for Type Transformations

```python
@atdata.packable
class SimplifiedSample:
    label: str

@atdata.lens
def simplify(src: ImageSample) -> SimplifiedSample:
    return SimplifiedSample(label=src.label)

# View dataset through a different type
simple_ds = dataset.as_type(SimplifiedSample)
for batch in simple_ds.ordered(batch_size=16):
    print(batch.label)
```

## HuggingFace-Style Loading

```python
# Load from local path
ds = atdata.load_dataset("path/to/data-{000000..000009}.tar", split="train")

# Load with split detection
ds_dict = atdata.load_dataset("path/to/data/")
train_ds = ds_dict["train"]
test_ds = ds_dict["test"]
```

## Local Storage with Redis + S3

```python
from atdata.local import LocalIndex, Repo

# Set up local index
index = LocalIndex()  # Connects to Redis

# Create repo with S3 storage
repo = Repo(
    s3_credentials={"AWS_ENDPOINT": "http://localhost:9000", ...},
    bucket="my-bucket",
    index=index,
)

# Insert dataset
entry = repo.insert(samples, name="my-dataset")
print(f"Stored at: {entry.data_urls}")
```

## Publish to ATProto Federation

```python
from atdata.atmosphere import AtmosphereClient
from atdata.promote import promote_to_atmosphere

# Authenticate
client = AtmosphereClient()
client.login("handle.bsky.social", "app-password")

# Promote local dataset to federation
entry = index.get_dataset("my-dataset")
at_uri = promote_to_atmosphere(entry, index, client)
print(f"Published at: {at_uri}")
```

## Documentation

- [Packable Samples](packable-samples.md) - Defining typed samples
- [Datasets](datasets.md) - Loading and iterating datasets
- [Lenses](lenses.md) - Type transformations
- [Local Storage](local-storage.md) - Redis + S3 backend
- [Atmosphere](atmosphere.md) - ATProto federation
- [Promotion](promotion.md) - Local to atmosphere workflow
- [load_dataset](load-dataset.md) - HuggingFace-style API
- [Protocols](protocols.md) - Abstract interfaces

## License

MIT
