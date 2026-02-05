"""Live PostgreSQL provider integration tests.

Exercises ``PostgresProvider`` against a real PostgreSQL service
container.  Requires ``POSTGRES_DSN`` environment variable.
"""

from __future__ import annotations

import json
import threading

import pytest

from .conftest import unique_name

# ── Helpers ───────────────────────────────────────────────────────


def _make_entry(**overrides):
    """Create a LocalDatasetEntry with sensible defaults.

    Each entry gets a unique name AND unique data_urls so that the
    content-derived CID is distinct (CID is based on schema_ref +
    data_urls, not name).
    """
    from atdata.index._entry import LocalDatasetEntry

    entry_name = overrides.pop("name", unique_name("entry"))
    defaults = dict(
        name=entry_name,
        schema_ref="local://schemas/TestSample@1.0.0",
        data_urls=[f"/data/{entry_name}/shard-000000.tar"],
        metadata={"split": "train"},
    )
    defaults.update(overrides)
    return LocalDatasetEntry(**defaults)


# ── Basic CRUD ────────────────────────────────────────────────────


class TestEntryOperations:
    """Store, retrieve, and iterate dataset entries."""

    def test_store_and_get_by_name(self, postgres_provider):
        entry = _make_entry()
        postgres_provider.store_entry(entry)

        fetched = postgres_provider.get_entry_by_name(entry.name)
        assert fetched.name == entry.name
        assert fetched.schema_ref == entry.schema_ref
        assert fetched.data_urls == entry.data_urls
        assert fetched.metadata == entry.metadata

    def test_store_and_get_by_cid(self, postgres_provider):
        entry = _make_entry()
        postgres_provider.store_entry(entry)

        fetched = postgres_provider.get_entry_by_cid(entry.cid)
        assert fetched.cid == entry.cid
        assert fetched.name == entry.name

    def test_get_nonexistent_entry_raises(self, postgres_provider):
        with pytest.raises(KeyError):
            postgres_provider.get_entry_by_name("does-not-exist-ever")

    def test_iter_entries(self, postgres_provider):
        entries = [_make_entry() for _ in range(5)]
        for e in entries:
            postgres_provider.store_entry(e)

        stored = list(postgres_provider.iter_entries())
        stored_names = {e.name for e in stored}
        for e in entries:
            assert e.name in stored_names

    def test_upsert_entry(self, postgres_provider):
        entry = _make_entry()
        postgres_provider.store_entry(entry)

        # Update with same CID but different data_urls
        from atdata.index._entry import LocalDatasetEntry

        updated = LocalDatasetEntry(
            name=entry.name,
            schema_ref=entry.schema_ref,
            data_urls=["/data/new-shard.tar"],
            metadata={"split": "test"},
            _cid=entry.cid,
        )
        postgres_provider.store_entry(updated)

        fetched = postgres_provider.get_entry_by_cid(entry.cid)
        assert fetched.data_urls == ["/data/new-shard.tar"]
        assert fetched.metadata == {"split": "test"}


# ── Schema operations ────────────────────────────────────────────


class TestSchemaOperations:
    """Store and retrieve schema records."""

    def test_store_and_get_schema(self, postgres_provider):
        name = unique_name("schema")
        schema_json = json.dumps(
            {"name": name, "fields": [{"name": "x", "type": "int"}]}
        )
        postgres_provider.store_schema(name, "1.0.0", schema_json)

        result = postgres_provider.get_schema_json(name, "1.0.0")
        assert result is not None
        parsed = json.loads(result)
        assert parsed["name"] == name

    def test_get_missing_schema_returns_none(self, postgres_provider):
        result = postgres_provider.get_schema_json("nonexistent", "0.0.0")
        assert result is None

    def test_iter_schemas(self, postgres_provider):
        names = [unique_name("iter-schema") for _ in range(3)]
        for n in names:
            postgres_provider.store_schema(n, "1.0.0", json.dumps({"name": n}))

        all_schemas = list(postgres_provider.iter_schemas())
        stored_names = {s[0] for s in all_schemas}
        for n in names:
            assert n in stored_names

    def test_find_latest_version(self, postgres_provider):
        name = unique_name("versioned")
        postgres_provider.store_schema(name, "1.0.0", "{}")
        postgres_provider.store_schema(name, "1.1.0", "{}")
        postgres_provider.store_schema(name, "2.0.0", "{}")

        latest = postgres_provider.find_latest_version(name)
        assert latest == "2.0.0"

    def test_schema_upsert(self, postgres_provider):
        name = unique_name("upsert-schema")
        postgres_provider.store_schema(name, "1.0.0", '{"v":1}')
        postgres_provider.store_schema(name, "1.0.0", '{"v":2}')

        result = postgres_provider.get_schema_json(name, "1.0.0")
        assert json.loads(result)["v"] == 2


# ── Label operations ─────────────────────────────────────────────


class TestLabelOperations:
    """Store and retrieve labels."""

    def test_store_and_get_label(self, postgres_provider):
        name = unique_name("label")
        postgres_provider.store_label(
            name, "cid-abc", version="1.0.0", description="test label"
        )

        cid, version = postgres_provider.get_label(name, version="1.0.0")
        assert cid == "cid-abc"
        assert version == "1.0.0"

    def test_get_latest_label(self, postgres_provider):
        name = unique_name("latest-label")
        postgres_provider.store_label(name, "cid-old", version="1.0.0")
        postgres_provider.store_label(name, "cid-new", version="2.0.0")

        cid, version = postgres_provider.get_label(name)
        # get_label without version returns most recent by created_at
        assert cid in ("cid-old", "cid-new")

    def test_get_missing_label_raises(self, postgres_provider):
        with pytest.raises(KeyError):
            postgres_provider.get_label("nonexistent-label-xyz")

    def test_iter_labels(self, postgres_provider):
        names = [unique_name("iter-label") for _ in range(3)]
        for n in names:
            postgres_provider.store_label(n, f"cid-{n}")

        all_labels = list(postgres_provider.iter_labels())
        stored_names = {lbl[0] for lbl in all_labels}
        for n in names:
            assert n in stored_names


# ── Concurrent access ────────────────────────────────────────────


class TestConcurrentAccess:
    """Verify PostgresProvider handles concurrent writes safely."""

    def test_concurrent_schema_writes(self, postgres_dsn: str):
        """Multiple threads writing different schemas concurrently."""
        from atdata.providers._postgres import PostgresProvider

        errors: list[Exception] = []
        names = [unique_name(f"concurrent-{i}") for i in range(10)]

        def _write_schema(schema_name: str) -> None:
            try:
                provider = PostgresProvider(dsn=postgres_dsn)
                provider.store_schema(
                    schema_name, "1.0.0", json.dumps({"name": schema_name})
                )
                result = provider.get_schema_json(schema_name, "1.0.0")
                assert result is not None
                provider.close()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_write_schema, args=(n,)) for n in names]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert errors == [], f"Concurrent writes produced errors: {errors}"

    def test_concurrent_entry_writes(self, postgres_dsn: str):
        """Multiple threads writing different entries concurrently."""
        from atdata.providers._postgres import PostgresProvider

        errors: list[Exception] = []

        def _write_entry(idx: int) -> None:
            try:
                provider = PostgresProvider(dsn=postgres_dsn)
                entry = _make_entry(name=unique_name(f"conc-entry-{idx}"))
                provider.store_entry(entry)
                fetched = provider.get_entry_by_name(entry.name)
                assert fetched.name == entry.name
                provider.close()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_write_entry, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert errors == [], f"Concurrent writes produced errors: {errors}"
