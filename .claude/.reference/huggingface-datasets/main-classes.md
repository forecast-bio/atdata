# HuggingFace Datasets - Main Classes API Reference

Source: https://huggingface.co/docs/datasets/en/package_reference/main_classes

## Dataset

A map-style dataset backed by Apache Arrow table, supporting random access.

### Creation Methods

```python
# From various file formats
Dataset.from_csv('path/to/file.csv')
Dataset.from_json('path/to/file.json')
Dataset.from_parquet('path/to/file.parquet')
Dataset.from_text('path/to/file.txt')
Dataset.from_sql("SELECT * FROM table", "sqlite:///db.sqlite")

# From in-memory data
Dataset.from_dict({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
Dataset.from_pandas(df)
Dataset.from_generator(generator_function)

# From Arrow
Dataset.from_file('dataset.arrow')
Dataset.from_buffer(arrow_buffer)
```

### Key Properties

```python
ds.num_rows          # Number of rows
ds.num_columns       # Number of columns
ds.column_names      # List of column names
ds.shape             # (num_rows, num_columns)
ds.features          # Features schema
ds.info              # DatasetInfo object
ds.data              # Apache Arrow table
ds.cache_files       # Cache file locations
```

### Data Transformation Methods

#### `map()` - Apply function to examples

```python
# Apply to individual examples
ds = ds.map(lambda x: {'text': x['text'].upper()})

# Apply to batches
ds = ds.map(lambda batch: {'text': [t.upper() for t in batch['text']]},
            batched=True, batch_size=32)

# With indices
ds = ds.map(lambda x, idx: {'id': idx, **x}, with_indices=True)

# Multiprocessing
ds = ds.map(process_fn, num_proc=4)

# Remove original columns
ds = ds.map(tokenize_fn, remove_columns=['text'])

# Add new columns
ds = ds.map(lambda x: {'length': len(x['text'])})
```

#### `filter()` - Keep examples matching condition

```python
# Filter by condition
ds_filtered = ds.filter(lambda x: x['label'] == 1)

# Batched filtering
ds_filtered = ds.filter(lambda batch: [l == 1 for l in batch['label']],
                        batched=True)

# With indices
ds_filtered = ds.filter(lambda x, idx: idx % 2 == 0, with_indices=True)
```

#### `select()` - Select specific rows

```python
ds_subset = ds.select(range(100))         # First 100 rows
ds_subset = ds.select([0, 10, 20, 30])    # Specific indices
```

#### `shuffle()` - Randomize row order

```python
ds_shuffled = ds.shuffle(seed=42)
# For large datasets, flatten indices for better performance
ds_shuffled = ds.shuffle(seed=42).flatten_indices()
```

#### `sort()` - Sort by column(s)

```python
ds_sorted = ds.sort('label')
ds_sorted = ds.sort(['label', 'text'], reverse=[True, False])
```

#### `train_test_split()` - Split into train/test

```python
train_test = ds.train_test_split(test_size=0.2, seed=42)
train_ds = train_test['train']
test_ds = train_test['test']

# Stratified split
train_test = ds.train_test_split(test_size=0.2, stratify_by_column='label')
```

### Column Operations

```python
# Add column
ds = ds.add_column('new_col', new_data)

# Remove columns
ds = ds.remove_columns(['col1', 'col2'])

# Rename column
ds = ds.rename_column('old_name', 'new_name')

# Rename multiple columns
ds = ds.rename_columns({'old1': 'new1', 'old2': 'new2'})

# Select columns
ds = ds.select_columns(['col1', 'col2'])

# Cast column to new type
from datasets import ClassLabel
ds = ds.cast_column('label', ClassLabel(names=['neg', 'pos']))

# Cast all features
from datasets import Features, Value
new_features = Features({'text': Value('string'), 'label': Value('int32')})
ds = ds.cast(new_features)

# Flatten nested features
ds_flat = ds.flatten()

# Class encode (convert to ClassLabel)
ds = ds.class_encode_column('label')
```

### Slicing & Iteration

```python
# Index access
item = ds[0]           # Single item
batch = ds[0:10]       # Slice
batch = ds[[0, 5, 9]]  # Multiple indices

# Iteration
for example in ds:
    pass

# Batched iteration
for batch in ds.iter(batch_size=32):
    pass

# Take first n
subset = ds.take(5)

# Skip first n
subset = ds.skip(10)

# Shard dataset
shard = ds.shard(num_shards=4, index=0)
```

### Data Format Control

```python
# Set format for ML framework
ds.set_format(type='torch', columns=['input_ids', 'attention_mask'])
ds.set_format(type='numpy')
ds.set_format(type='pandas')
ds.set_format(type='jax')
ds.set_format(type='tensorflow')

# With context manager
with ds.formatted_as(type='pandas'):
    df = ds[:]

# Reset to default (dict of lists)
ds.reset_format()

# With on-the-fly transforms
ds = ds.with_transform(tokenize_fn)
```

### Persistence

```python
# Save to disk (Arrow format)
ds.save_to_disk('path/to/dataset')

# Load from disk
ds = Dataset.load_from_disk('path/to/dataset')

# Export to formats
ds.to_csv('output.csv')
ds.to_json('output.jsonl')
ds.to_parquet('output.parquet')
df = ds.to_pandas()
d = ds.to_dict()

# Push to Hub
ds.push_to_hub('username/dataset-name', private=True)
```

---

## DatasetDict

Dictionary-like container for multiple Dataset splits (train, validation, test, etc.).

### Creation

```python
from datasets import DatasetDict, load_dataset

# From dict of datasets
dataset_dict = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset,
    'test': test_dataset
})

# Load from Hub (returns DatasetDict if split not specified)
dataset_dict = load_dataset('dataset_name')
```

### Access Splits

```python
train_ds = dataset_dict['train']
all_splits = list(dataset_dict.keys())  # ['train', 'validation', 'test']

# Iterate
for split_name, dataset in dataset_dict.items():
    print(f"{split_name}: {len(dataset)} examples")
```

### Properties

```python
dataset_dict.num_rows        # {'train': N, 'validation': M, ...}
dataset_dict.num_columns     # {'train': K, ...}
dataset_dict.column_names    # {'train': [...], ...}
dataset_dict.shape           # {'train': (N, K), ...}
```

### Collective Operations

All Dataset methods can be called on DatasetDict and will be applied to all splits:

```python
# Map over all splits
dataset_dict = dataset_dict.map(lambda x: {'text': x['text'].upper()})

# Filter all splits
dataset_dict = dataset_dict.filter(lambda x: x['label'] == 1)

# Remove columns from all splits
dataset_dict = dataset_dict.remove_columns(['col1'])

# Rename column in all splits
dataset_dict = dataset_dict.rename_column('old', 'new')

# Sort all splits
dataset_dict = dataset_dict.sort('label')

# Shuffle all splits
dataset_dict = dataset_dict.shuffle(seed=42)

# Format all splits
dataset_dict = dataset_dict.set_format(type='torch')
```

### Persistence

```python
# Save all splits
dataset_dict.save_to_disk('path/to/dataset')

# Load from disk
dataset_dict = DatasetDict.load_from_disk('path/to/dataset')

# Push to Hub
dataset_dict.push_to_hub('username/dataset-name')
```

---

## IterableDataset

Iterable dataset for streaming/lazy loading, backed by Python generators.

### Creation

```python
from datasets import IterableDataset

# From generator function
def gen():
    for i in range(1000):
        yield {'text': f'Example {i}', 'label': i % 2}

ds = IterableDataset.from_generator(gen)

# With sharded data
def gen(shards):
    for shard in shards:
        with open(shard) as f:
            for line in f:
                yield {'line': line}

shards = [f'data{i}.txt' for i in range(32)]
ds = IterableDataset.from_generator(gen, gen_kwargs={'shards': shards})

# From load_dataset with streaming=True
ds = load_dataset('dataset_name', split='train', streaming=True)
```

### Iteration

```python
# Basic iteration (no random access!)
for example in ds:
    process(example)

# Batched iteration
for batch in ds.iter(batch_size=32):
    process_batch(batch)

# Take n examples
subset = ds.take(100)

# Skip n examples
subset = ds.skip(10)
```

### Transformations (applied lazily during iteration)

```python
# Map
ds = ds.map(lambda x: {'text': x['text'].upper()})

# Filter
ds = ds.filter(lambda x: x['label'] == 1)

# Shuffle with buffer (approximate)
ds = ds.shuffle(seed=42, buffer_size=1000)

# Batch into groups
ds = ds.batch(batch_size=32)

# Remove/select columns
ds = ds.remove_columns(['unwanted_col'])
ds = ds.select_columns(['text', 'label'])

# Rename column
ds = ds.rename_column('old', 'new')

# Cast features
ds = ds.cast(new_features)
```

### Format Control

```python
# Set format
ds = ds.with_format('torch')
ds = ds.with_format('numpy')
```

### Distributed Processing

```python
# Shard across workers
ds = ds.shard(num_shards=4, index=0)

# State management for resumable iteration
state = ds.state_dict()
# ... resume later ...
ds.load_state_dict(state)
```

---

## Features

Schema definition for dataset structure, specifying column names and types.

### Creating Features

```python
from datasets import Features, Value, ClassLabel, Sequence, Array2D, Audio, Image

features = Features({
    'text': Value('string'),
    'label': ClassLabel(names=['neg', 'pos']),
    'score': Value('float32'),
    'tokens': Sequence(Value('string')),
    'embeddings': Sequence(Value('float32')),
    'audio': Audio(sampling_rate=16000),
    'image': Image(),
})
```

### Feature Types

| Type | Description |
|------|-------------|
| `Value('string')` | String scalar |
| `Value('int32')`, `Value('int64')` | Integer scalars |
| `Value('float32')`, `Value('float64')` | Float scalars |
| `Value('bool')` | Boolean |
| `ClassLabel(names=['a', 'b'])` | Classification labels |
| `Sequence(Value('string'))` | Variable-length sequence |
| `Array2D(shape=(28, 28), dtype='uint8')` | Fixed-shape 2D array |
| `Array3D(shape=(3, 224, 224), dtype='float32')` | Fixed-shape 3D array |
| `Audio(sampling_rate=16000)` | Audio file |
| `Image()` | Image file |
| `Translation(languages=['en', 'fr'])` | Translation pair |

### Using Features

```python
# Specify features when loading
features = Features({'text': Value('string'), 'label': ClassLabel(names=['neg', 'pos'])})
ds = load_dataset('csv', data_files='data.csv', features=features)

# Access features from dataset
print(ds.features)

# Cast existing dataset
ds = ds.cast(new_features)
ds = ds.cast_column('label', ClassLabel(names=['a', 'b', 'c']))
```

---

## Key Differences: Dataset vs IterableDataset

| Feature | Dataset | IterableDataset |
|---------|---------|-----------------|
| **Access** | Random access (`ds[0]`) | Sequential only |
| **Speed** | Fast for batch ops | Better for streaming |
| **Memory** | Arrow memory-mapped | Lazy evaluation |
| **Shuffling** | Full dataset | Approximate (buffer) |
| **Use Case** | Training with epochs | Streaming/large data |
| **len()** | Supported | Not supported |
