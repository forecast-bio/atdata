# HuggingFace Datasets - Loading Guide

Source: https://huggingface.co/docs/datasets/en/loading

## Overview

Data can be loaded from multiple sources:
- The Hugging Face Hub
- Local files (CSV, JSON, Parquet, etc.)
- In-memory data (dicts, lists, generators, DataFrames)
- SQL databases
- Remote URLs

---

## Loading from Hugging Face Hub

```python
from datasets import load_dataset

# Basic usage
dataset = load_dataset("lhoestq/demo1")

# Specific version (git tag, branch, or commit)
dataset = load_dataset("lhoestq/custom_squad", revision="main")

# Map data files to splits
data_files = {"train": "train.csv", "test": "test.csv"}
dataset = load_dataset("namespace/your_dataset_name", data_files=data_files)

# Load subset of files with patterns
c4_subset = load_dataset("allenai/c4", data_files="en/c4-train.0000*-of-01024.json.gz")

# Load from subdirectory
c4_subset = load_dataset("allenai/c4", data_dir="en")
```

---

## Loading Local Files

### CSV

```python
from datasets import load_dataset

# Single file
dataset = load_dataset("csv", data_files="my_file.csv")

# Multiple files
dataset = load_dataset("csv", data_files=["file1.csv", "file2.csv"])

# With split mapping
dataset = load_dataset("csv", data_files={"train": "train.csv", "test": "test.csv"})
```

### JSON

```python
# Standard JSON lines format (one object per line)
dataset = load_dataset("json", data_files="my_file.json")

# Nested JSON with field parameter
# File: {"version": "0.1.0", "data": [{"a": 1}, {"a": 2}]}
dataset = load_dataset("json", data_files="my_file.json", field="data")

# Remote JSON
base_url = "https://example.com/data/"
dataset = load_dataset("json", data_files={
    "train": base_url + "train.json",
    "validation": base_url + "dev.json"
}, field="data")
```

### Parquet

```python
# Local
dataset = load_dataset("parquet", data_files={'train': 'train.parquet', 'test': 'test.parquet'})

# Remote
base_url = "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.ab/"
data_files = {"train": base_url + "train-00000-of-00001.parquet"}
wiki = load_dataset("parquet", data_files=data_files, split="train")
```

### Arrow

```python
# Via load_dataset
dataset = load_dataset("arrow", data_files={'train': 'train.arrow'})

# Direct memory mapping (faster, no cache)
from datasets import Dataset
dataset = Dataset.from_file("data.arrow")
```

### Text

```python
dataset = load_dataset("text", data_files="my_file.txt")
```

### WebDataset (TAR archives)

```python
# Best used with streaming for large datasets
path = "path/to/train/*.tar"
dataset = load_dataset("webdataset", data_files={"train": path}, split="train", streaming=True)

# Remote WebDataset
base_url = "https://example.com/dataset/"
urls = [base_url + f"shard-{i:06d}.tar" for i in range(4)]
dataset = load_dataset("webdataset", data_files={"train": urls}, split="train", streaming=True)
```

### HDF5

```python
dataset = load_dataset("hdf5", data_files="data.h5")
```

### SQL Databases

```python
from datasets import Dataset

# Load entire table
dataset = Dataset.from_sql("data_table_name", con="sqlite:///sqlite_file.db")

# Load from query
dataset = Dataset.from_sql(
    "SELECT text FROM table WHERE length(text) > 100 LIMIT 10",
    con="sqlite:///sqlite_file.db"
)
```

---

## Loading In-Memory Data

### Python Dictionary

```python
from datasets import Dataset

my_dict = {"a": [1, 2, 3], "b": ["x", "y", "z"]}
dataset = Dataset.from_dict(my_dict)
```

### Python List of Dictionaries

```python
my_list = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}, {"a": 3, "b": "z"}]
dataset = Dataset.from_list(my_list)
```

### Python Generator

```python
from datasets import Dataset, IterableDataset

# For data larger than memory
def my_gen():
    for i in range(1, 1000000):
        yield {"a": i, "text": f"example {i}"}

dataset = Dataset.from_generator(my_gen)

# Sharded generator for distributed processing
def gen(shards):
    for shard in shards:
        with open(shard) as f:
            for line in f:
                yield {"line": line}

shards = [f"data{i}.txt" for i in range(32)]
ds = IterableDataset.from_generator(gen, gen_kwargs={"shards": shards})
ds = ds.shuffle(seed=42, buffer_size=10_000)
```

### Pandas DataFrame

```python
import pandas as pd
from datasets import Dataset

df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
dataset = Dataset.from_pandas(df)
```

---

## Multiprocessing

Speed up loading with multiple processes:

```python
from datasets import load_dataset

# Each process handles a subset of shards
imagenet = load_dataset("timm/imagenet-1k-wds", num_proc=8)
```

---

## Slicing Splits

### String API

```python
import datasets

# Concatenate splits
train_test_ds = datasets.load_dataset("dataset_name", split="train+test")

# Select rows by index
train_10_20_ds = datasets.load_dataset("dataset_name", split="train[10:20]")

# Select by percentage
train_10pct_ds = datasets.load_dataset("dataset_name", split="train[:10%]")

# Combine percentage slices
train_10_80pct_ds = datasets.load_dataset("dataset_name", split="train[:10%]+train[-80%:]")

# Cross-validation splits
val_ds = datasets.load_dataset("dataset_name",
    split=[f"train[{k}%:{k+10}%]" for k in range(0, 100, 10)])
train_ds = datasets.load_dataset("dataset_name",
    split=[f"train[:{k}%]+train[{k+10}%:]" for k in range(0, 100, 10)])
```

### ReadInstruction API

```python
import datasets

# Concatenate
ri = datasets.ReadInstruction("train") + datasets.ReadInstruction("test")
train_test_ds = datasets.load_dataset("dataset_name", split=ri)

# Percentage with rounding control
ri = datasets.ReadInstruction("train", from_=50, to=52, unit="%", rounding="pct1_dropremainder")
train_50_52_ds = datasets.load_dataset("dataset_name", split=ri)
```

---

## Specifying Features

Override auto-inferred features:

```python
from datasets import load_dataset, Features, Value, ClassLabel

# Define custom features
class_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]
emotion_features = Features({
    'text': Value('string'),
    'label': ClassLabel(names=class_names)
})

# Apply when loading
dataset = load_dataset('csv', data_files='data.csv', features=emotion_features)

# Verify
print(dataset['train'].features)
# {'text': Value('string'), 'label': ClassLabel(names=['sadness', 'joy', ...])}
```

---

## Offline Mode

Use cached datasets without internet:

```bash
# Set environment variable
export HF_HUB_OFFLINE=1
```

```python
# Will use cache only
dataset = load_dataset("dataset_name")
```

---

## Image/Audio/Video Datasets

### ImageFolder

```python
# Directory structure: images/{class_name}/{image_file}
dataset = load_dataset("imagefolder", data_dir="path/to/images", split="train")
```

### AudioFolder

```python
dataset = load_dataset("audiofolder", data_dir="path/to/audio", split="train")
```

### VideoFolder

```python
dataset = load_dataset("videofolder", data_dir="path/to/videos", split="train")
```
