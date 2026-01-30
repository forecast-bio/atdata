test *args:
    uv run pytest {{args}}

lint:
    uv run ruff check src/ tests/
    uv run ruff format --check src/ tests/

bench *args:
    uv run pytest benchmarks/ --override-ini="python_files=bench_*.py" --benchmark-enable --benchmark-sort=mean --no-cov {{args}}

bench-save name:
    uv run pytest benchmarks/ --override-ini="python_files=bench_*.py" --benchmark-enable --benchmark-save={{name}} --no-cov

bench-compare a b:
    uv run pytest-benchmark compare {{a}} {{b}}

[working-directory: 'docs_src']
docs:
    uv run quartodoc build
    quarto render