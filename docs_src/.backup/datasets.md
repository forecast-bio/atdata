# Datasets

The `Dataset` class provides typed iteration over WebDataset tar files with automatic batching and lens transformations.

## Creating a Dataset

```python
import atdata

@atdata.packable
class ImageSample:
    image: NDArray
    label: str

# Single shard
dataset = atdata.Dataset[ImageSample]("data-000000.tar")

# Multiple shards with brace notation
dataset = atdata.Dataset[ImageSample]("data-{000000..000009}.tar")
```

The type parameter `[ImageSample]` specifies what sample type the dataset contains. This enables type-safe iteration and automatic deserialization.

## Iteration Modes

### Ordered Iteration

Iterate through samples in their original order:

```python
# With batching (default batch_size=1)
for batch in dataset.ordered(batch_size=32):
    images = batch.image  # numpy array (32, H, W, C)
    labels = batch.label  # list of 32 strings

# Without batching (raw samples)
for sample in dataset.ordered(batch_size=None):
    print(sample.label)
```

### Shuffled Iteration

Iterate with randomized order at both shard and sample levels:

```python
for batch in dataset.shuffled(batch_size=32):
    # Samples are shuffled
    process(batch)

# Control shuffle buffer sizes
for batch in dataset.shuffled(
    buffer_shards=100,    # Shards to buffer (default: 100)
    buffer_samples=10000, # Samples to buffer (default: 10,000)
    batch_size=32,
):
    process(batch)
```

Larger buffer sizes increase randomness but use more memory.

## SampleBatch

When iterating with a `batch_size`, each iteration yields a `SampleBatch` with automatic attribute aggregation.

```python
@atdata.packable
class Sample:
    features: NDArray  # shape (256,)
    label: str
    score: float

for batch in dataset.ordered(batch_size=16):
    # NDArray fields are stacked with a batch dimension
    features = batch.features  # numpy array (16, 256)

    # Other fields become lists
    labels = batch.label       # list of 16 strings
    scores = batch.score       # list of 16 floats
```

Results are cached, so accessing the same attribute multiple times is efficient.

## Type Transformations with Lenses

View a dataset through a different sample type using registered lenses:

```python
@atdata.packable
class SimplifiedSample:
    label: str

@atdata.lens
def simplify(src: ImageSample) -> SimplifiedSample:
    return SimplifiedSample(label=src.label)

# Transform dataset to different type
simple_ds = dataset.as_type(SimplifiedSample)

for batch in simple_ds.ordered(batch_size=16):
    print(batch.label)  # Only label field available
```

See [Lenses](lenses.md) for details on defining transformations.

## Dataset Properties

### Shard List

Get the list of individual tar files:

```python
dataset = atdata.Dataset[Sample]("data-{000000..000009}.tar")
shards = dataset.shard_list
# ['data-000000.tar', 'data-000001.tar', ..., 'data-000009.tar']
```

### Metadata

Datasets can have associated metadata from a URL:

```python
dataset = atdata.Dataset[Sample](
    "data-{000000..000009}.tar",
    metadata_url="https://example.com/metadata.msgpack"
)

# Fetched and cached on first access
metadata = dataset.metadata  # dict or None
```

## Writing Datasets

Use WebDataset's `TarWriter` or `ShardWriter` to create datasets:

```python
import webdataset as wds

samples = [
    ImageSample(image=np.random.rand(224, 224, 3).astype(np.float32), label="cat")
    for _ in range(100)
]

# Single tar file
with wds.writer.TarWriter("data-000000.tar") as sink:
    for i, sample in enumerate(samples):
        sink.write({**sample.as_wds, "__key__": f"sample_{i:06d}"})

# Multiple shards with automatic splitting
with wds.writer.ShardWriter("data-%06d.tar", maxcount=1000) as sink:
    for i, sample in enumerate(samples):
        sink.write({**sample.as_wds, "__key__": f"sample_{i:06d}"})
```

## Parquet Export

Export dataset contents to parquet format:

```python
# Export entire dataset
dataset.to_parquet("output.parquet")

# Export with custom field mapping
def extract_fields(sample):
    return {"label": sample.label, "score": sample.confidence}

dataset.to_parquet("output.parquet", sample_map=extract_fields)

# Export in segments
dataset.to_parquet("output.parquet", maxcount=10000)
# Creates output-000000.parquet, output-000001.parquet, etc.
```

## URL Formats

WebDataset supports various URL formats:

```python
# Local files
dataset = atdata.Dataset[Sample]("./data/file.tar")
dataset = atdata.Dataset[Sample]("/absolute/path/file-{000000..000009}.tar")

# S3 (requires s3fs)
dataset = atdata.Dataset[Sample]("s3://bucket/path/file-{000000..000009}.tar")

# HTTP/HTTPS
dataset = atdata.Dataset[Sample]("https://example.com/data-{000000..000009}.tar")
```

## Related

- [Packable Samples](packable-samples.md) - Defining typed samples
- [Lenses](lenses.md) - Type transformations
- [load_dataset](load-dataset.md) - HuggingFace-style loading API
