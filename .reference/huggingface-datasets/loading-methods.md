# HuggingFace Datasets - Loading Methods API Reference

Source: https://huggingface.co/docs/datasets/en/package_reference/loading_methods

## datasets.load_dataset

Load a dataset from the Hugging Face Hub, or a local dataset.

```python
load_dataset(
    path: str,
    name: Optional[str] = None,
    data_dir: Optional[str] = None,
    data_files: Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]], None] = None,
    split: Union[str, Split, list[str], list[Split], None] = None,
    cache_dir: Optional[str] = None,
    features: Optional[Features] = None,
    download_config: Optional[DownloadConfig] = None,
    download_mode: Union[DownloadMode, str, None] = None,
    verification_mode: Union[VerificationMode, str, None] = None,
    keep_in_memory: Optional[bool] = None,
    save_infos: bool = False,
    revision: Union[Version, str, None] = None,
    token: Union[bool, str, None] = None,
    streaming: bool = False,
    num_proc: Optional[int] = None,
    storage_options: Optional[dict] = None,
    **config_kwargs,
)
```

### What it does under the hood

1. **Load a dataset builder:**
   - Find the most common data format in the dataset and pick its associated builder (JSON, CSV, Parquet, Webdataset, ImageFolder, AudioFolder, etc.)
   - Find which file goes into which split (e.g. train/test) based on file and directory names or on the YAML configuration
   - Can specify `data_files` manually, and which dataset builder to use (e.g. "parquet")

2. **Run the dataset builder:**
   - Download the data files from the dataset if they are not already available locally or cached
   - Process and cache the dataset in typed Arrow tables
   - In streaming mode: don't download or cache anything, dataset is lazily loaded

3. **Return a dataset built from the requested splits**

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str` | Path or name of the dataset. Can be: Hub repo (`'username/dataset_name'`), local directory (`'./path/to/data'`), or builder name with `data_files`/`data_dir` (`'parquet'`) |
| `name` | `str`, optional | Dataset configuration name |
| `data_dir` | `str`, optional | Directory containing the data files |
| `data_files` | `str` or `Sequence` or `Mapping`, optional | Path(s) to source data file(s) |
| `split` | `str` or `Split`, optional | Which split to load. If `None`, returns `DatasetDict` with all splits |
| `cache_dir` | `str`, optional | Directory to read/write data. Default: `~/.cache/huggingface/datasets` |
| `features` | `Features`, optional | Set the features type to use |
| `download_mode` | `DownloadMode`, optional | Download/generate mode. Default: `REUSE_DATASET_IF_EXISTS` |
| `verification_mode` | `VerificationMode`, optional | Checks to run on downloaded data. Default: `BASIC_CHECKS` |
| `keep_in_memory` | `bool`, optional | Whether to copy the dataset in-memory |
| `revision` | `str`, optional | Version (git tag/commit/branch) of the dataset to load |
| `token` | `str` or `bool`, optional | Bearer token for remote files on the Hub |
| `streaming` | `bool` | If `True`, returns `IterableDataset` without downloading. Default: `False` |
| `num_proc` | `int`, optional | Number of processes for downloading and generating |
| `storage_options` | `dict`, optional | Key/value pairs for file-system backend |

### Returns

- If `split` is not `None`: `Dataset` (or `IterableDataset` if streaming)
- If `split` is `None`: `DatasetDict` (or `IterableDatasetDict` if streaming)

### Examples

```python
from datasets import load_dataset

# Load from Hugging Face Hub
ds = load_dataset('cornell-movie-review-data/rotten_tomatoes', split='train')

# Load a subset/configuration
ds = load_dataset('nyu-mll/glue', 'sst2', split='train')

# Manual mapping of data files to splits
data_files = {'train': 'train.csv', 'test': 'test.csv'}
ds = load_dataset('namespace/your_dataset_name', data_files=data_files)

# Load local CSV file
ds = load_dataset('csv', data_files='path/to/local/my_dataset.csv')

# Load local JSON file
ds = load_dataset('json', data_files='path/to/local/my_dataset.json')

# Streaming mode (no download)
ds = load_dataset('cornell-movie-review-data/rotten_tomatoes', split='train', streaming=True)

# ImageFolder
ds = load_dataset('imagefolder', data_dir='/path/to/images', split='train')

# WebDataset
ds = load_dataset('webdataset', data_files='path/to/train/*.tar', split='train', streaming=True)
```

---

## datasets.load_from_disk

Loads a dataset that was previously saved using `save_to_disk()`.

```python
load_from_disk(
    dataset_path: str,
    keep_in_memory: Optional[bool] = None,
    storage_options: Optional[dict] = None,
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `dataset_path` | `path-like` | Path or remote URI (e.g. `"s3://my-bucket/dataset/train"`) |
| `keep_in_memory` | `bool`, optional | Whether to copy the dataset in-memory |
| `storage_options` | `dict`, optional | Key/value pairs for file-system backend |

### Returns

- `Dataset` or `DatasetDict`

### Example

```python
from datasets import load_from_disk

ds = load_from_disk('path/to/dataset/directory')
```

---

## datasets.load_dataset_builder

Load a dataset builder for inspection or streaming without full download.

```python
load_dataset_builder(
    path: str,
    name: Optional[str] = None,
    data_dir: Optional[str] = None,
    data_files: Optional[...] = None,
    cache_dir: Optional[str] = None,
    features: Optional[Features] = None,
    download_config: Optional[DownloadConfig] = None,
    download_mode: Optional[...] = None,
    revision: Optional[...] = None,
    token: Optional[...] = None,
    storage_options: Optional[dict] = None,
    **config_kwargs,
)
```

### Example

```python
from datasets import load_dataset_builder

ds_builder = load_dataset_builder('cornell-movie-review-data/rotten_tomatoes')
print(ds_builder.info.features)
# {'label': ClassLabel(names=['neg', 'pos']), 'text': Value('string')}
```

---

## datasets.get_dataset_config_names

Get the list of available config names for a dataset.

```python
from datasets import get_dataset_config_names

get_dataset_config_names("nyu-mll/glue")
# ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', ...]
```

---

## datasets.get_dataset_split_names

Get the list of available splits for a dataset.

```python
from datasets import get_dataset_split_names

get_dataset_split_names('cornell-movie-review-data/rotten_tomatoes')
# ['train', 'validation', 'test']
```

---

## Built-in Dataset Builders

The following builders are available for loading different file formats:

| Builder | File Types |
|---------|------------|
| `text` | `.txt` |
| `csv` | `.csv`, `.tsv` |
| `json` | `.json`, `.jsonl` |
| `parquet` | `.parquet` |
| `arrow` | `.arrow` |
| `xml` | `.xml` |
| `sql` | SQL databases |
| `webdataset` | `.tar` (WebDataset format) |
| `imagefolder` | Image directories |
| `audiofolder` | Audio directories |
| `videofolder` | Video directories |
| `pdffolder` | PDF directories |
| `hdf5` | `.h5`, `.hdf5` |

### Builder-specific options

Each builder has its own configuration class with specific options:

```python
# CSV with custom separator
load_dataset("csv", data_files="data.csv", sep="\t")

# JSON with nested field
load_dataset("json", data_files="data.json", field="data")

# Parquet with column selection
load_dataset("parquet", data_files="data.parquet", columns=["col1", "col2"])

# Parquet with filters (pushed down to file)
load_dataset("parquet", data_files="data.parquet", filters=[("col", "==", 0)])
```
