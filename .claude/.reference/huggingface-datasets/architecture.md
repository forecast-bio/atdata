# HuggingFace Datasets - Architecture Overview

Source: https://huggingface.co/docs/datasets/en/about_dataset_load

## How load_dataset Works (ELI5)

A dataset is a directory that contains:
- Data files in generic formats (JSON, CSV, Parquet, text, etc.)
- A dataset card (`README.md`) with documentation and YAML configuration

`load_dataset()` fetches the requested dataset locally or from the Hugging Face Hub.

### Automatic Format Detection

If the dataset only contains data files, `load_dataset()` automatically infers how to load them based on file extensions. Under the hood, it uses an appropriate `DatasetBuilder`:

| Format | Builder Class |
|--------|---------------|
| `.txt` | `datasets.packaged_modules.text.Text` |
| `.csv`, `.tsv` | `datasets.packaged_modules.csv.Csv` |
| `.json`, `.jsonl` | `datasets.packaged_modules.json.Json` |
| `.parquet` | `datasets.packaged_modules.parquet.Parquet` |
| `.arrow` | `datasets.packaged_modules.arrow.Arrow` |
| SQL | `datasets.packaged_modules.sql.Sql` |
| Image folders | `datasets.packaged_modules.imagefolder.ImageFolder` |
| Audio folders | `datasets.packaged_modules.audiofolder.AudioFolder` |
| WebDataset TAR | `datasets.packaged_modules.webdataset.WebDataset` |

---

## Building a Dataset

Two main classes are responsible for building a dataset:

### BuilderConfig

Configuration class containing dataset attributes:

| Attribute | Description |
|-----------|-------------|
| `name` | Short name of the dataset |
| `version` | Dataset version identifier |
| `data_dir` | Path to local folder containing data files |
| `data_files` | Paths to local data files |
| `description` | Description of the dataset |

Custom attributes (like class labels) can be added by subclassing `BuilderConfig`.

Configuration can be populated:
1. Via predefined `BuilderConfig` instances in `DatasetBuilder.BUILDER_CONFIGS`
2. Via keyword arguments to `load_dataset()` (overrides predefined)

### DatasetBuilder

Accesses `BuilderConfig` attributes to build the actual dataset.

Three main methods:

#### 1. `_info()` - Define dataset attributes

- Defines dataset attributes returned by `dataset.info`
- Specifies `Features` (schema with column names and types)

#### 2. `_split_generator()` - Organize data files

- Downloads or retrieves data files
- Uses `DownloadManager` for downloading/extracting
- Organizes files into splits via `SplitGenerator`
- Returns keyword arguments for `_generate_examples`

#### 3. `_generate_examples()` - Parse and yield examples

- Reads and parses data files for each split
- Yields examples as Python dicts matching the schema
- Uses Python generator (memory efficient)
- Examples buffered in `ArrowWriter` before writing to disk

---

## Data Flow

```
load_dataset("name", split="train")
        │
        ▼
┌───────────────────────────────────────┐
│ 1. Resolve dataset path               │
│    - Hub repo? Local dir? Builder?    │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│ 2. Load DatasetBuilder                │
│    - Auto-detect format               │
│    - Apply BuilderConfig              │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│ 3. Download & prepare (if not cached) │
│    - _split_generator() downloads     │
│    - _generate_examples() yields      │
│    - Arrow tables cached to disk      │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│ 4. Load from cache                    │
│    - Memory-map Arrow files           │
│    - Return Dataset/DatasetDict       │
└───────────────────────────────────────┘
```

---

## Caching

- Datasets are cached as Arrow tables in `~/.cache/huggingface/datasets`
- Subsequent loads use the cache (fast!)
- Cache can be disabled or customized via `cache_dir` parameter
- `download_mode` controls cache behavior:
  - `REUSE_DATASET_IF_EXISTS` (default): Use cache if available
  - `FORCE_REDOWNLOAD`: Re-download everything
  - `REUSE_CACHE_IF_EXISTS`: Reuse cache for downloads but regenerate dataset

---

## Streaming Mode

With `streaming=True`:
- No downloading or caching
- Data streamed on-the-fly during iteration
- Returns `IterableDataset` instead of `Dataset`
- Best for large datasets

```python
ds = load_dataset("large_dataset", split="train", streaming=True)
for example in ds:
    process(example)  # Examples fetched as needed
```

---

## Integrity Verification

`load_dataset()` verifies downloaded data:
- Number of splits in generated `DatasetDict`
- Number of samples in each split
- List of downloaded files
- SHA256 checksums (disabled by default)

Disable with `verification_mode="no_checks"` if needed.

---

## Key Design Patterns for atdata Integration

### Pattern 1: Path Resolution
HF Datasets supports multiple path types:
- Hub repository: `"username/dataset"`
- Local directory: `"./path/to/data"`
- Builder name: `"parquet"` with `data_files`

### Pattern 2: Split Handling
- `split=None` → `DatasetDict` with all splits
- `split="train"` → Single `Dataset`
- Split string algebra: `"train+test"`, `"train[:10%]"`

### Pattern 3: Lazy Loading
- Streaming mode for large datasets
- Generator-based iteration
- Buffer-based shuffling

### Pattern 4: Format Abstraction
- Single API for multiple formats
- Auto-detection based on file extensions
- Builder-specific configuration via kwargs

### Pattern 5: Type System
- `Features` schema defines column types
- Automatic type inference with override capability
- Special types for media (Audio, Image, Video)
