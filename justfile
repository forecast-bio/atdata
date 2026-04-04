_bench_base := "uv run pytest benchmarks/ --override-ini='python_files=bench_*.py' --benchmark-enable --benchmark-sort=mean --no-cov"

setup:
    git config core.hooksPath .githooks
    @echo "Git hooks activated from .githooks/"

# Fetch latest lexicons from atdata-lexicon repo and sync to both local copies.
# Uses gh CLI to download a tarball — no clone needed.
# Pass ref=<branch/tag> to pin a specific version (default: main).
sync-lexicons ref="main":
    #!/usr/bin/env bash
    set -euo pipefail
    REPO="forecast-bio/atdata-lexicon"
    VENDOR="lexicons/science/alt/dataset"
    PKG="src/atdata/lexicons/science/alt/dataset"
    TMPDIR=$(mktemp -d)
    trap 'rm -rf "$TMPDIR"' EXIT
    echo "Fetching lexicons from $REPO@{{ref}}..."
    gh api "repos/$REPO/tarball/{{ref}}" > "$TMPDIR/archive.tar.gz"
    tar xzf "$TMPDIR/archive.tar.gz" -C "$TMPDIR" --strip-components=1
    # Copy NSID lexicons to both vendor and package directories
    for f in "$TMPDIR/lexicons/science/alt/dataset/"*.json; do
        name=$(basename "$f")
        cp "$f" "$VENDOR/$name"
        cp "$f" "$PKG/$name"
        echo "  synced $name"
    done
    echo "Lexicons synced from $REPO@{{ref}}"
    # Also sync top-level shim files if they exist upstream
    for f in "$TMPDIR/lexicons/"*.json; do
        [ -f "$f" ] || continue
        name=$(basename "$f")
        cp "$f" "lexicons/$name"
        [ -f "src/atdata/lexicons/$name" ] && cp "$f" "src/atdata/lexicons/$name"
    done

# Sync local vendored lexicons → package (no network, for offline use)
sync-lexicons-local:
    #!/usr/bin/env bash
    set -euo pipefail
    for f in lexicons/science/alt/dataset/*.json; do
        name=$(basename "$f")
        cp "$f" "src/atdata/lexicons/science/alt/dataset/$name"
    done
    for f in lexicons/*.json; do
        name=$(basename "$f")
        [ -f "src/atdata/lexicons/$name" ] && cp "$f" "src/atdata/lexicons/$name"
    done
    echo "Local lexicon sync complete"

gen-lexicon-docs:
    uv run python scripts/gen_lexicon_docs.py

test *args:
    just sync-lexicons-local
    uv run pytest {{args}}

lint:
    uv run ruff check src/ tests/
    uv run ruff format --check src/ tests/

bench:
    mkdir -p .benchmarks
    just bench-serial
    just bench-index
    just bench-io
    just bench-query
    just bench-s3
    just bench-report

bench-serial *args:
    {{ _bench_base }} -m bench_serial --benchmark-json=.benchmarks/serial.json {{args}}

bench-index *args:
    {{ _bench_base }} -m bench_index --benchmark-json=.benchmarks/index.json {{args}}

bench-io *args:
    {{ _bench_base }} -m bench_io --benchmark-json=.benchmarks/io.json {{args}}

bench-query *args:
    {{ _bench_base }} -m bench_query --benchmark-json=.benchmarks/query.json {{args}}

bench-s3 *args:
    {{ _bench_base }} -m bench_s3 --benchmark-json=.benchmarks/s3.json {{args}}

bench-report:
    uv run python -m benchmarks.render_report .benchmarks/*.json -o .benchmarks/report.html
    @echo "Report: .benchmarks/report.html"

bench-save name:
    {{ _bench_base }} --benchmark-save={{name}}

bench-compare a b:
    uv run pytest-benchmark compare {{a}} {{b}}

[working-directory: 'docs_src']
docs:
    just gen-lexicon-docs
    uv run quartodoc build
    # Use the project venv Python so quarto finds quartodoc and other deps
    # (without this, quarto may pick up a system Python that lacks them).
    QUARTO_PYTHON={{justfile_directory()}}/.venv/bin/python quarto render
    mkdir -p ../docs/benchmarks
    cp ../.benchmarks/report.html ../docs/benchmarks/index.html || echo "No benchmark report found — run 'just bench' first"
