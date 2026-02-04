"""Live integration tests against remote service sandboxes.

All tests in this package are marked with ``@pytest.mark.integration``
and are excluded from default ``pytest`` runs via ``addopts`` in
``pyproject.toml``.  Run them explicitly::

    uv run pytest -m integration
"""
