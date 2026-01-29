test *args:
    uv run pytest {{args}}

lint:
    uv run ruff check src/ tests/
    uv run ruff format --check src/ tests/

[working-directory: 'docs_src']
docs:
    uv run quartodoc build
    quarto render