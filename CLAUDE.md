# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`atdata` is a Python library that implements a loose federation of distributed, typed datasets built on top of WebDataset. It provides:

- **Typed samples** with automatic serialization via msgpack
- **Lens-based transformations** between different dataset schemas
- **Batch aggregation** with automatic numpy array stacking
- **WebDataset integration** for efficient large-scale dataset storage

## Development Commands

### Environment Setup
```bash
# Uses uv for dependency management
python -m pip install uv  # if not already installed
uv sync
```

### Testing
```bash
# Run all tests with coverage
pytest

# Run specific test file
pytest tests/test_dataset.py
pytest tests/test_lens.py

# Run single test
pytest tests/test_dataset.py::test_create_sample
pytest tests/test_lens.py::test_lens
```

### Building
```bash
# Build the package
uv build
```

## Architecture

### Core Components

The codebase has three main modules under `src/atdata/`:

1. **dataset.py** - Core dataset and sample infrastructure
   - `PackableSample`: Base class for samples that can be serialized with msgpack
   - `Dataset[ST]`: Generic typed dataset wrapping WebDataset tar files
   - `SampleBatch[DT]`: Automatic batching with attribute aggregation
   - `@packable` decorator: Converts dataclasses into PackableSample subclasses

2. **lens.py** - Type transformation system
   - `Lens[S, V]`: Bidirectional transformations between sample types (getter/putter)
   - `LensNetwork`: Singleton registry for lens transformations
   - `@lens` decorator: Registers lens getters globally

3. **_helpers.py** - Serialization utilities
   - `array_to_bytes()` / `bytes_to_array()`: numpy array serialization

### Key Design Patterns

**Sample Type Definition**

Two approaches for defining sample types:

```python
# Approach 1: Explicit inheritance
@dataclass
class MySample(atdata.PackableSample):
    field1: str
    field2: NDArray

# Approach 2: Decorator (recommended)
@atdata.packable
class MySample:
    field1: str
    field2: NDArray
```

**NDArray Handling**

Fields annotated as `NDArray` or `NDArray | None` are automatically:
- Converted from bytes during deserialization
- Converted to bytes during serialization (via `_helpers.array_to_bytes`)
- Handled by `_ensure_good()` method in `PackableSample.__post_init__`

**Lens Transformations**

Lenses enable viewing datasets through different type schemas:

```python
@atdata.lens
def my_lens(source: SourceType) -> ViewType:
    return ViewType(...)

@my_lens.putter
def my_lens_put(view: ViewType, source: SourceType) -> SourceType:
    return SourceType(...)

# Use with datasets
ds = atdata.Dataset[SourceType](url).as_type(ViewType)
```

The `LensNetwork` singleton (in `lens.py:183`) maintains a global registry of all lenses decorated with `@lens`.

**Batch Aggregation**

`SampleBatch` uses `__getattr__` magic to aggregate sample attributes:
- For `NDArray` fields: stacks into numpy array with batch dimension
- For other fields: creates list
- Results are cached in `_aggregate_cache`

### Dataset URLs

Datasets use WebDataset brace-notation URLs:
- Single shard: `path/to/file-000000.tar`
- Multiple shards: `path/to/file-{000000..000009}.tar`

### Important Implementation Details

**Type Parameters**

The codebase uses Python 3.12+ generics heavily:
- `Dataset[ST]` where `ST` is the sample type
- `SampleBatch[DT]` where `DT` is the sample type
- Uses `__orig_class__.__args__[0]` at runtime to extract type parameters

**Serialization Flow**

1. Sample → `as_wds` property → dict with `__key__` and `msgpack` bytes
2. Msgpack bytes created by `packed` property calling `_make_packable()` on fields
3. Deserialization: `from_bytes()` → `from_data()` → `__init__` → `_ensure_good()`

**WebDataset Integration**

- Uses `wds.writer.ShardWriter` / `wds.writer.TarWriter` for writing
  - **Important:** Always import from `wds.writer` (e.g., `wds.writer.TarWriter`) instead of `wds.TarWriter`
  - This avoids linting issues while functionally equivalent
- Dataset iteration via `wds.DataPipeline` with custom `wrap()` / `wrap_batch()` methods
- Supports `ordered()` and `shuffled()` iteration modes

## Testing Notes

- Tests use parametrization heavily via `@pytest.mark.parametrize`
- Test cases cover both decorator and inheritance syntax
- Temporary WebDataset tar files created in `tmp_path` fixture
- Tests verify both serialization and batch aggregation behavior
- Lens tests verify well-behavedness (GetPut/PutGet laws)

## Git Workflow

### Committing Changes

When using the `/commit` command or creating commits:
- **Always include `.chainlink/issues.db`** in commits alongside code changes
- This ensures issue tracking history is preserved across sessions
- The issues.db file tracks all chainlink issues, comments, and status changes

### Planning Documents

- **Track `.planning/` directory in git** - Do not ignore planning documents
- Planning documents in `.planning/` should be committed to preserve design history
- This includes architecture notes, implementation plans, and design decisions
