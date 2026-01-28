# atdata

[![codecov](https://codecov.io/gh/foundation-ac/atdata/branch/main/graph/badge.svg)](https://codecov.io/gh/foundation-ac/atdata)

A loose federation of distributed, typed datasets built on WebDataset.

**atdata** provides a type-safe, composable framework for working with large-scale datasets. It combines the efficiency of WebDataset's tar-based storage with Python's type system and functional programming patterns.

## Features

- **Typed Samples** - Define dataset schemas using Python dataclasses with automatic msgpack serialization
- **Schema-free Exploration** - Load datasets without defining a schema first using `DictSample`
- **Lens Transformations** - Bidirectional, composable transformations between different dataset views
- **Automatic Batching** - Smart batch aggregation with numpy array stacking
- **WebDataset Integration** - Efficient storage and streaming for large-scale datasets
- **Flexible Data Sources** - Stream from local files, HTTP URLs, or S3-compatible storage
- **HuggingFace-style API** - `load_dataset()` with path resolution and split handling
- **Local & Atmosphere Storage** - Index datasets locally with Redis or publish to ATProto network

## Installation

```bash
pip install atdata
```

Requires Python 3.12 or later.

## Quick Start

### Loading Datasets

The primary way to load datasets is with `load_dataset()`:

```python
from atdata import load_dataset

# Load without specifying a type - returns Dataset[DictSample]
ds = load_dataset("path/to/data.tar", split="train")

# Explore the data
for sample in ds.ordered():
    print(sample.keys())      # See available fields
    print(sample["text"])     # Dict-style access
    print(sample.label)       # Attribute access
    break
```

### Defining Typed Schemas

Once you understand your data, define a typed schema with `@packable`:

```python
import atdata
from numpy.typing import NDArray

@atdata.packable
class ImageSample:
    image: NDArray
    label: str
    metadata: dict
```

### Loading with Types

```python
# Load with explicit type
ds = load_dataset("path/to/data-{000000..000009}.tar", ImageSample, split="train")

# Or convert from DictSample
ds = load_dataset("path/to/data.tar", split="train").as_type(ImageSample)

# Iterate over samples
for sample in ds.ordered():
    print(f"Label: {sample.label}, Image shape: {sample.image.shape}")

# Iterate with shuffling and batching
for batch in ds.shuffled(batch_size=32):
    # batch.image is automatically stacked into shape (32, ...)
    # batch.label is a list of 32 labels
    process_batch(batch.image, batch.label)
```

### Lens Transformations

Define reusable transformations between sample types:

```python
@atdata.packable
class ProcessedSample:
    features: NDArray
    label: str

@atdata.lens
def preprocess(sample: ImageSample) -> ProcessedSample:
    features = extract_features(sample.image)
    return ProcessedSample(features=features, label=sample.label)

# Apply lens to view dataset as ProcessedSample
processed_ds = dataset.as_type(ProcessedSample)

for sample in processed_ds.ordered(batch_size=None):
    # sample is now a ProcessedSample
    print(sample.features.shape)
```

## Core Concepts

### DictSample

The default sample type for schema-free exploration. Provides both attribute and dict-style access:

```python
ds = load_dataset("data.tar", split="train")

for sample in ds.ordered():
    # Dict-style access
    print(sample["field_name"])

    # Attribute access
    print(sample.field_name)

    # Introspection
    print(sample.keys())
    print(sample.to_dict())
```

### PackableSample

Base class for typed, serializable samples. Fields annotated as `NDArray` are automatically handled:

```python
@atdata.packable
class MySample:
    array_field: NDArray      # Automatically serialized
    optional_array: NDArray | None
    regular_field: str
```

Every `@packable` class automatically registers a lens from `DictSample`, enabling seamless conversion via `.as_type()`.

### Lens

Bidirectional transformations with getter/putter semantics:

```python
@atdata.lens
def my_lens(source: SourceType) -> ViewType:
    # Transform source -> view
    return ViewType(...)

@my_lens.putter
def my_lens_put(view: ViewType, source: SourceType) -> SourceType:
    # Transform view -> source
    return SourceType(...)
```

### Data Sources

Datasets support multiple backends via the `DataSource` protocol:

```python
# String URLs (most common) - automatically wrapped in URLSource
dataset = atdata.Dataset[ImageSample]("data-{000000..000009}.tar")

# S3 with authentication (private buckets, Cloudflare R2, MinIO)
source = atdata.S3Source(
    bucket="my-bucket",
    keys=["data-000000.tar", "data-000001.tar"],
    endpoint="https://my-account.r2.cloudflarestorage.com",
    access_key="...",
    secret_key="...",
)
dataset = atdata.Dataset[ImageSample](source)
```

### Dataset URLs

Uses WebDataset brace expansion for sharded datasets:

- Single file: `"data/dataset-000000.tar"`
- Multiple shards: `"data/dataset-{000000..000099}.tar"`
- Multiple patterns: `"data/{train,val}/dataset-{000000..000009}.tar"`

### HuggingFace-style API

Load datasets with a familiar interface:

```python
from atdata import load_dataset

# Load without type for exploration (returns Dataset[DictSample])
ds = load_dataset("./data/train-*.tar", split="train")

# Load with explicit type
ds = load_dataset("./data/train-*.tar", ImageSample, split="train")

# Load from S3 with brace notation
ds = load_dataset("s3://bucket/data-{000000..000099}.tar", ImageSample, split="train")

# Load all splits (returns DatasetDict)
ds_dict = load_dataset("./data", ImageSample)
train_ds = ds_dict["train"]
test_ds = ds_dict["test"]

# Convert DictSample to typed schema
ds = load_dataset("./data/train.tar", split="train").as_type(ImageSample)
```

## Development

### Setup

```bash
# Install uv if not already available
python -m pip install uv

# Install dependencies
uv sync
```

### Testing

```bash
# Run all tests with coverage
uv run pytest

# Run specific test file
uv run pytest tests/test_dataset.py

# Run single test
uv run pytest tests/test_lens.py::test_lens
```

### Building

```bash
uv build
```

## Contributing

Contributions are welcome! This project is in beta, so the API may still evolve.

## License

This project is licensed under the Mozilla Public License 2.0. See [LICENSE](LICENSE) for details.
