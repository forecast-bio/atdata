# atdata

[![codecov](https://codecov.io/gh/foundation-ac/atdata/branch/main/graph/badge.svg)](https://codecov.io/gh/foundation-ac/atdata)

A loose federation of distributed, typed datasets built on WebDataset.

**atdata** provides a type-safe, composable framework for working with large-scale datasets. It combines the efficiency of WebDataset's tar-based storage with Python's type system and functional programming patterns.

## Features

- **Typed Samples** - Define dataset schemas using Python dataclasses with automatic msgpack serialization
- **Lens Transformations** - Bidirectional, composable transformations between different dataset views
- **Automatic Batching** - Smart batch aggregation with numpy array stacking
- **WebDataset Integration** - Efficient storage and streaming for large-scale datasets

## Installation

```bash
pip install atdata
```

Requires Python 3.12 or later.

## Quick Start

### Defining Sample Types

Use the `@packable` decorator to create typed dataset samples:

```python
import atdata
from numpy.typing import NDArray

@atdata.packable
class ImageSample:
    image: NDArray
    label: str
    metadata: dict
```

### Creating Datasets

```python
# Create a dataset
dataset = atdata.Dataset[ImageSample]("path/to/data-{000000..000009}.tar")

# Iterate over samples in order
for sample in dataset.ordered(batch_size=None):
    print(f"Label: {sample.label}, Image shape: {sample.image.shape}")

# Iterate with shuffling and batching
for batch in dataset.shuffled(batch_size=32):
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

### PackableSample

Base class for serializable samples. Fields annotated as `NDArray` are automatically handled:

```python
@atdata.packable
class MySample:
    array_field: NDArray      # Automatically serialized
    optional_array: NDArray | None
    regular_field: str
```

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

### Dataset URLs

Uses WebDataset brace expansion for sharded datasets:

- Single file: `"data/dataset-000000.tar"`
- Multiple shards: `"data/dataset-{000000..000099}.tar"`
- Multiple patterns: `"data/{train,val}/dataset-{000000..000009}.tar"`

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
pytest

# Run specific test file
pytest tests/test_dataset.py

# Run single test
pytest tests/test_lens.py::test_lens
```

### Building

```bash
uv build
```

## Contributing

Contributions are welcome! This project is in beta, so the API may still evolve.

## License

This project is licensed under the Mozilla Public License 2.0. See [LICENSE](LICENSE) for details.
