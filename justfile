_bench_base := "uv run pytest benchmarks/ --override-ini='python_files=bench_*.py' --benchmark-enable --benchmark-sort=mean --no-cov"

test *args:
    uv run pytest {{args}}

lint:
    uv run ruff check src/ tests/
    uv run ruff format --check src/ tests/

bench:
    mkdir -p .bench
    just bench-serial
    just bench-index
    just bench-io
    just bench-query
    just bench-s3
    just bench-report

bench-serial *args:
    {{ _bench_base }} -m bench_serial --benchmark-json=.bench/serial.json {{args}}

bench-index *args:
    {{ _bench_base }} -m bench_index --benchmark-json=.bench/index.json {{args}}

bench-io *args:
    {{ _bench_base }} -m bench_io --benchmark-json=.bench/io.json {{args}}

bench-query *args:
    {{ _bench_base }} -m bench_query --benchmark-json=.bench/query.json {{args}}

bench-s3 *args:
    {{ _bench_base }} -m bench_s3 --benchmark-json=.bench/s3.json {{args}}

bench-report:
    uv run python -m benchmarks.render_report .bench/*.json -o .bench/report.html
    @echo "Report: .bench/report.html"

bench-save name:
    {{ _bench_base }} --benchmark-save={{name}}

bench-compare a b:
    uv run pytest-benchmark compare {{a}} {{b}}

[working-directory: 'docs_src']
docs:
    uv run quartodoc build
    quarto render
    mkdir -p ../docs/benchmarks
    cp ../.bench/report.html ../docs/benchmarks/index.html || echo "No benchmark report found — run 'just bench' first"
