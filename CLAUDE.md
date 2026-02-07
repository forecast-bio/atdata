# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`atdata` is a Python library that implements a loose federation of distributed, typed datasets built on top of WebDataset and ATProto. It provides:

- **Typed samples** with automatic serialization via msgpack
- **Local and atmosphere storage** with pluggable index providers (SQLite, Redis, PostgreSQL)
- **Lens-based transformations** between different dataset schemas
- **ATProto integration** for publishing and discovering datasets on the atmosphere
- **HuggingFace-style API** with `load_dataset()` for convenient access
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
# Always run tests through uv to use the correct virtual environment
# Run all tests with coverage
uv run pytest

# Run specific test file
uv run pytest tests/test_dataset.py
uv run pytest tests/test_local.py

# Run single test
uv run pytest tests/test_dataset.py::test_create_sample -v
```

### Building
```bash
# Build the package
uv build
```

### Development Scripts (justfile)

Development tasks are managed with [just](https://github.com/casey/just), a command runner. Available commands:

```bash
just test              # Run all tests with coverage
just test tests/test_dataset.py  # Run specific test file
just lint              # Run ruff check + format check
just docs              # Build documentation (runs quartodoc + quarto)
just bench             # Run full benchmark suite
just bench-io          # Run I/O benchmarks only
just bench-index       # Run index provider benchmarks
just bench-query       # Run query benchmarks
just bench-report      # Generate HTML benchmark report
just bench-save <name> # Save benchmark results
just bench-compare a b # Compare two benchmark runs
```

> **Note on `just docs`:** The recipe sets `QUARTO_PYTHON` to the project
> venv's Python so that Quarto uses the correct interpreter with project
> dependencies. Without this, Quarto may pick up a system Python that lacks
> `quartodoc` and other required packages.

The `justfile` is in the project root. Add new dev tasks there rather than creating shell scripts.

### Running Python
```bash
# Always use uv run for Python commands to use the correct virtual environment
uv run python -c "import atdata; print(atdata.__version__)"
uv run python script.py

# Never use bare python/python3 - it may not have project dependencies
# BAD: python3 -c "import webdataset"
# GOOD: uv run python -c "import webdataset"
```

## Architecture

### Module Overview

The codebase lives under `src/atdata/` with these main components:

**Core modules:**
- `dataset.py` — `PackableSample`, `DictSample`, `Dataset[ST]`, `SampleBatch[DT]`, `@packable`, `write_samples()`
- `lens.py` — `Lens[S, V]`, `LensNetwork`, `@lens` decorator
- `_protocols.py` — Protocol definitions: `Packable`, `IndexEntry`, `AbstractIndex`, `AbstractDataStore`, `DataSource`
- `_hf_api.py` — `load_dataset()`, `DatasetDict`, HuggingFace-style path resolution
- `_exceptions.py` — Custom exception hierarchy (`AtdataError`, `SchemaError`, `ShardError`, etc.)

**Index and storage:**
- `index/` — `Index`, `LocalDatasetEntry`, `LocalSchemaRecord`, schema management
- `stores/` — `LocalDiskStore`, `S3DataStore` (data store implementations)
- `local/` — Backward-compat shim re-exporting from `index/` and `stores/`
- `providers/` — Pluggable index backends: `SqliteProvider` (default), `RedisProvider`, `PostgresProvider`
- `repository.py` — `Repository` dataclass pairing provider + data store, prefix routing

**ATProto integration:**
- `atmosphere/` — `Atmosphere`, schema/dataset/lens publishers and loaders, `PDSBlobStore`
- `promote.py` — Local-to-atmosphere promotion (deprecated in favor of `Index.promote_entry()`)

**Data pipeline:**
- `_sources.py` — `URLSource`, `S3Source`, `BlobSource` (streaming shard data to Dataset)
- `manifest/` — Per-shard metadata manifests for query-based access (`ManifestField`, `QueryExecutor`)

**Utilities:**
- `_helpers.py` — NumPy array serialization (`array_to_bytes` / `bytes_to_array`)
- `_cid.py` — ATProto-compatible CID generation via libipld
- `_schema_codec.py` — Dynamic Python type generation from stored schemas
- `_stub_manager.py` — IDE stub file generation for dynamic types
- `_type_utils.py` — Shared type conversion utilities
- `_logging.py` — Pluggable structured logging
- `testing.py` — Mock clients, fixtures, and test helpers

**CLI:**
- `cli/` — Typer-based CLI: `atdata inspect`, `atdata preview`, `atdata schema show/diff`, `atdata local up/down/status`, `atdata diagnose`

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

**Writing and Indexing Data**

```python
# Write samples directly to tar files
ds = atdata.write_samples(samples, "output/data.tar")

# Or use Index for managed storage
index = atdata.Index(data_store=atdata.LocalDiskStore())
entry = index.write(samples, name="my-dataset")
```

**Index with Pluggable Storage**

```python
# SQLite backend (default, zero dependencies)
index = atdata.Index()

# With local disk storage
index = atdata.Index(data_store=atdata.LocalDiskStore())

# With S3 storage
from atdata.local import S3DataStore
index = atdata.Index(data_store=S3DataStore(credentials, bucket="my-bucket"))

# With atmosphere backend
from atdata.atmosphere import AtmosphereClient
client = AtmosphereClient.login("handle", "password")
index = atdata.Index(atmosphere=client)
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

The `LensNetwork` singleton (in `lens.py`) maintains a global registry of all lenses decorated with `@lens`.

**Batch Aggregation**

`SampleBatch` uses `__getattr__` magic to aggregate sample attributes:
- For `NDArray` fields: stacks into numpy array with batch dimension
- For other fields: creates list
- Results are cached in `_aggregate_cache`

### Dataset URLs

Datasets use WebDataset brace-notation URLs:
- Single shard: `path/to/file-000000.tar`
- Multiple shards: `path/to/file-{000000..000009}.tar`

### Naming Conventions

**Property vs Method Pattern for Collections**

When exposing collections of items, follow this convention:

- `foo.xs` - `@property` returning `Iterator[X]` (lazy iteration)
- `foo.list_xs()` - method returning `list[X]` (eager, fully evaluated)

Examples:
- `index.datasets` / `index.list_datasets()`
- `index.schemas` / `index.list_schemas()`
- `dataset.shards` / `dataset.list_shards()`

The lazy property enables memory-efficient iteration over large collections,
while the method provides a concrete list when needed.

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

- 1550+ tests across 40+ test files
- Tests use parametrization via `@pytest.mark.parametrize` where appropriate
- Temporary WebDataset tar files created in `tmp_path` fixture
- Shared sample types defined in `conftest.py` (`SharedBasicSample`, `SharedNumpySample`)
- Lens tests verify well-behavedness (GetPut/PutGet/PutPut laws)
- Integration tests cover local, atmosphere, cross-backend, and error handling scenarios

### ATProto SDK Signature Validation

`tests/test_atproto_compat.py` validates that our `Atmosphere` wrapper calls
the atproto SDK with compatible method signatures. It uses a **real** atproto
`Client` instance (not a mock) with `ClientRaw._invoke` patched so no network
I/O occurs. This catches TypeErrors like passing unsupported kwargs that
unspecced mocks would silently accept.

Key details:
- The fixture injects a mock session with a far-future JWT expiry so the SDK's
  session-refresh guard passes through without attempting a token refresh
- Both `client._session` (SDK) and `atmo._session` (wrapper) must be set
- The `Client.upload_blob()` wrapper does NOT accept `**kwargs`; use the
  namespace method `Client.com.atproto.repo.upload_blob()` which forwards
  kwargs through to httpx

### Warning Suppression Convention

**Keep warning suppression local to individual tests, not global.**

When tests generate expected warnings (e.g., from deprecated APIs or third-party library incompatibilities), suppress them using `@pytest.mark.filterwarnings` decorators on each affected test rather than global suppression in `conftest.py`. This:
- Documents which specific tests have known warning behaviors
- Makes it easier to track when warnings appear in unexpected places
- Avoids masking genuine warnings from new code

Example for deprecated API tests:
```python
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestAtmosphereIndex:
    """Tests for deprecated AtmosphereIndex backward compat."""
    ...
```

## Docstring Formatting

This project uses **Google-style docstrings** with quartodoc for API documentation generation. The most important formatting requirement is for **Examples sections**.

### Examples Section Format

Use `Examples:` (plural) for code examples. This is recognized by griffe's Google docstring parser and rendered with proper syntax highlighting by quartodoc:

```python
def my_function():
    """Short description.

    Longer description if needed.

    Args:
        param: Description of parameter.

    Returns:
        Description of return value.

    Examples:
        >>> result = my_function()
        >>> print(result)
        'output'
    """
```

**Key formatting rules:**

1. Use `Examples:` (plural, not `Example:` singular)
2. Code examples are indented 8 spaces (4 more than `Examples:`)
3. Use `>>>` for Python prompts and `...` for continuation lines
4. No `::` marker needed - griffe handles the parsing automatically

**Incorrect format (will not render with syntax highlighting):**
```python
    Example:  # Wrong - singular form is treated as an admonition
        ::    # Wrong - reST literal block marker not needed
            >>> code_here()
```

**Correct format:**
```python
    Examples:
        >>> code_here()  # Correct - plural form, proper indentation
```

### Multiple Examples

For multiple examples, continue in the same section:

```python
    Examples:
        >>> # First example
        >>> x = create_thing()

        >>> # Second example
        >>> y = other_thing()
```

### Class and Method Docstrings

Apply the same format to class docstrings and method docstrings:

```python
class MyClass:
    """Class description.

    Examples:
        >>> obj = MyClass()
        >>> obj.do_something()
    """

    def method(self):
        """Method description.

        Examples:
            >>> self.method()
        """
```

## Issue Tracking

This project uses **chainlink** for issue tracking. Chainlink commands do NOT need to be prefixed with `uv run`:
```bash
# Correct - run chainlink directly
chainlink list
chainlink close 123
chainlink show 123

# Incorrect - don't use uv run
uv run chainlink list  # Not needed
```

## Custom Skills

Project-level Claude Code skills are defined in `.claude/commands/`:

- `/release <version>` — Full release flow: branch from previous release, merge develop, version bump, changelog, PR to `main`
- `/publish` — Post-merge: create GitHub release, monitor PyPI publish, sync `develop`
- `/feature <description>` — Create a feature branch from `develop` with a slugified name and chainlink issue
- `/featree <description>` — Create a feature branch in a new git worktree (symlinks chainlink db)
- `/kickoff <description>` — Create a worktree via `/featree`, write a self-contained prompt, and launch an autonomous agent in a tmux session
- `/check [session]` — Check status of background feature agents (reads tmux panes and `.kickoff-status` sentinel files)
- `/adr` — Adversarial review with docstring-preservation rules for quartodoc
- `/changelog` — Generate clean CHANGELOG entry from chainlink history
- `/commit` — Analyze changes and create a well-formatted commit

User-level skills (in `~/.claude/commands/`) take precedence over project-level skills with the same name.

### Background Agent Workflow (`/kickoff` → `/check`)

The `/kickoff` command automates feature implementation by launching an autonomous Claude agent in an isolated worktree:

1. **`/kickoff <description>`** creates a worktree (via `/featree`), writes a detailed prompt file, and launches `claude --model opus` in a tmux session named `feat-<slug>`.
2. The background agent works autonomously: reads `CLAUDE.md`, implements the feature, runs tests, uses `/commit`, runs `/adr`, fixes issues, and writes `DONE` to `.kickoff-status` when finished.
3. **`/check`** monitors progress by reading the tmux pane output and checking the sentinel file. It reports status (Working/Waiting/Done/Error) and suggests next actions.

**Common issues with background agents:**
- Agents get stuck on Claude Code's trust/permission prompts — approve with `tmux send-keys -t <session> Enter`
- Context compaction happens around 5% remaining — agents with large changes may compact mid-work
- Monitor directly: `tmux capture-pane -t <session> -p | tail -40`
- Attach interactively: `tmux attach -t <session>`

## Git Workflow

This project follows **git flow** branching:

### Branch Model

| Branch | Purpose | Branches from | Merges to |
|--------|---------|---------------|-----------|
| `main` | Production releases, always deployable | — | — |
| `develop` | Integration branch, all features land here | `main` (initial) | `main` (via release) |
| `feature/*` | Individual work items | `develop` | `develop` |
| `release/*` | Release prep (version bump, changelog) | `develop` | `main` (via PR) |
| `hotfix/*` | Urgent fixes to production | `main` | `main` + `develop` |

### Feature Development

1. Branch from `develop`: `git checkout develop && git checkout -b feature/my-feature`
2. Do work, commit
3. Merge back to `develop` with `--no-ff`
4. Delete the feature branch

### Release Flow

Releases follow this pattern (automated by `/release` skill):
1. Create `release/v<version>` branch **from the previous release branch** (e.g., `release/v0.4.0b2`)
2. Merge `develop` into the release branch with `--no-ff`
3. Bump version in `pyproject.toml`, run `uv lock`
4. Write CHANGELOG entry (Keep a Changelog format)
5. Push and create PR to `main`
6. After merge, use `/publish` to create GitHub release, then sync develop: `git checkout develop && git merge main --no-ff`

### Committing Changes

When using the `/commit` command or creating commits:
- **Always include `.chainlink/issues.db`** in commits alongside code changes
- This ensures issue tracking history is preserved across sessions
- The issues.db file tracks all chainlink issues, comments, and status changes

### Git Hooks

The `.githooks/` directory contains shared hooks. Activate them after cloning:

```bash
just setup   # sets core.hooksPath to .githooks/
```

**Included hooks:**
- **`pre-commit`** — Blocks commits where `issues.db` is staged as a symlink (mode 120000). Prevents worktree artifacts from overwriting the real database on merge.
- **`pre-merge-commit`** — Backs up `issues.db` to `issues.db.pre-merge` before every merge, so the database can be restored if a merge corrupts it.

### Worktrees and Chainlink

When using git worktrees (via `/featree`), the worktree's `.chainlink/issues.db`
is replaced with a **symlink** to the base clone's copy. This ensures all
worktrees share a single authoritative database on the `develop` branch.

**Protection layers (in order of defense):**
1. `/featree` adds `.chainlink/issues.db` to the worktree's `.git/info/exclude` so the symlink is never staged.
2. The `pre-commit` hook blocks any commit that stages `issues.db` as a symlink (mode 120000).
3. The `pre-merge-commit` hook backs up the db before merges so it can be restored if a symlink slips through.

**If the database is corrupted after a merge:**
```bash
cp .chainlink/issues.db.pre-merge .chainlink/issues.db
```

### CLI Module

- **Track `src/atdata/cli/`** — Always include the CLI module in commits
- The CLI is built with typer and provides `atdata inspect`, `atdata preview`, `atdata schema`, `atdata local`, and `atdata diagnose` commands
- Changes to CLI should be committed with the related feature changes

### Planning Documents

- **Track `.planning/` directory in git** — Do not ignore planning documents
- Planning documents are organized by phase in `.planning/phases/`:
  - `01-atproto-foundation/` — Initial ATProto integration design, lexicon definitions, architecture decisions
  - `02-v0.2-review/` — Human review assessments from v0.2 cycle
  - `03-v0.3-roadmap/` — Codebase review and synthesis roadmap for v0.3

### Reference Materials

- **Track `.reference/` directory in git** — Include reference documentation in commits
- The `.reference/` directory contains external specifications and reference materials
- This includes API specs, lexicon definitions, and other reference documentation used for development
