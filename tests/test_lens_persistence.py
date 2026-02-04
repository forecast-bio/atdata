"""Tests for persistent lens storage and reconstitution.

Covers:
- Lens serialization/deserialization via _lens_codec
- Provider storage (SQLite) for lens records
- Index integration (store_lens, get_lens, load_lens, list_lenses, find_lenses)
- Field-mapping lens reconstitution (full round-trip)
- Code-reference lens reconstitution
- StubManager lens stub generation
- Lens law verification on reconstituted lenses (GetPut, PutGet, PutPut)
- Version auto-increment
- Cache behaviour
"""

# NOTE: Do NOT use `from __future__ import annotations` here.
# The @lens decorator needs concrete type annotations (not strings) to
# extract source_type and view_type at decoration time.

import json
import pytest

from atdata import Index, lens, packable
from atdata._lens_codec import (
    lens_to_json,
    lens_from_record,
    generate_lens_stub,
    clear_lens_cache,
    _detect_field_mapping,
    _resolve_function_ref,
)
from atdata.providers._sqlite import SqliteProvider
from atdata.index._schema import _parse_lens_ref


# ---------------------------------------------------------------------------
# Fixtures & sample types
# ---------------------------------------------------------------------------


@packable
class PersonFull:
    name: str
    age: int
    height: float


@packable
class PersonName:
    name: str


@packable
class PersonPhysical:
    name: str
    height: float


# A module-level lens so it has a resolvable code reference
@lens
def person_name_lens(s: PersonFull) -> PersonName:
    return PersonName(name=s.name)


@person_name_lens.putter
def person_name_lens_put(v: PersonName, s: PersonFull) -> PersonFull:
    return PersonFull(name=v.name, age=s.age, height=s.height)


@pytest.fixture
def sqlite_provider(tmp_path):
    provider = SqliteProvider(path=tmp_path / "test.db")
    yield provider
    provider.close()


@pytest.fixture
def index(tmp_path):
    idx = Index(provider="sqlite", path=tmp_path / "index.db", atmosphere=None)
    yield idx


@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear lens cache between tests."""
    clear_lens_cache()
    yield
    clear_lens_cache()


# ---------------------------------------------------------------------------
# _lens_codec: Serialization
# ---------------------------------------------------------------------------


class TestLensSerialization:
    """Tests for lens_to_json serialization."""

    def test_serialize_field_mapping_lens(self):
        """Field-mapping lens (all view fields exist in source) is detected."""

        @packable
        class Src:
            name: str
            age: int

        @packable
        class View:
            name: str

        @lens
        def simple_lens(s: Src) -> View:
            return View(name=s.name)

        result = json.loads(lens_to_json(simple_lens, name="simple"))
        assert result["name"] == "simple"
        assert result["version"] == "1.0.0"
        assert result["source_schema"] == "Src"
        assert result["view_schema"] == "View"
        assert result["getter"]["kind"] == "field_mapping"
        assert len(result["getter"]["mappings"]) == 1
        assert result["getter"]["mappings"][0]["source_field"] == "name"

    def test_serialize_code_reference_lens(self):
        """Module-level lens getter has a resolvable code reference."""
        result = json.loads(lens_to_json(person_name_lens, name="person_name"))
        # person_name_lens maps PersonFull -> PersonName where PersonName.name
        # is a subset of PersonFull, so it will be detected as field_mapping.
        # That's correct because it IS a simple field mapping.
        assert result["getter"]["kind"] == "field_mapping"

    def test_serialize_with_explicit_schemas(self):
        result = json.loads(
            lens_to_json(
                person_name_lens,
                name="test",
                source_schema="PersonFull@1.0.0",
                view_schema="PersonName@1.0.0",
                description="Extract name only",
            )
        )
        assert result["source_schema"] == "PersonFull@1.0.0"
        assert result["view_schema"] == "PersonName@1.0.0"
        assert result["description"] == "Extract name only"

    def test_serialize_version(self):
        result = json.loads(
            lens_to_json(person_name_lens, name="test", version="2.3.4")
        )
        assert result["version"] == "2.3.4"

    def test_serialize_has_created_at(self):
        result = json.loads(lens_to_json(person_name_lens, name="test"))
        assert "created_at" in result


# ---------------------------------------------------------------------------
# _lens_codec: Reconstitution
# ---------------------------------------------------------------------------


class TestLensReconstitution:
    """Tests for lens_from_record reconstitution."""

    def test_roundtrip_field_mapping(self):
        """Serialize and reconstitute a field-mapping lens."""
        record_json = lens_to_json(person_name_lens, name="person_name")
        record = json.loads(record_json)

        reconstituted = lens_from_record(
            record,
            source_type=PersonFull,
            view_type=PersonName,
            register=False,
        )

        source = PersonFull(name="Alice", age=30, height=170.0)
        view = reconstituted.get(source)
        assert view.name == "Alice"

    def test_reconstituted_putter(self):
        """Reconstituted field-mapping putter correctly updates source."""
        record_json = lens_to_json(person_name_lens, name="person_name")
        record = json.loads(record_json)

        reconstituted = lens_from_record(
            record,
            source_type=PersonFull,
            view_type=PersonName,
            register=False,
        )

        source = PersonFull(name="Alice", age=30, height=170.0)
        new_view = PersonName(name="Bob")
        updated = reconstituted.put(new_view, source)

        assert updated.name == "Bob"
        assert updated.age == 30  # Preserved from source
        assert updated.height == 170.0  # Preserved from source

    def test_lens_laws_on_reconstituted(self):
        """Reconstituted field-mapping lens satisfies lens laws."""
        record_json = lens_to_json(person_name_lens, name="person_name")
        record = json.loads(record_json)

        reconstituted = lens_from_record(
            record,
            source_type=PersonFull,
            view_type=PersonName,
            register=False,
        )

        s = PersonFull(name="Alice", age=30, height=170.0)
        v = PersonName(name="Updated")

        # GetPut: get(put(v, s)) == v
        assert reconstituted.get(reconstituted.put(v, s)) == v

        # PutGet: put(get(s), s) == s
        assert reconstituted.put(reconstituted.get(s), s) == s

        # PutPut: put(v2, put(v1, s)) == put(v2, s)
        v2 = PersonName(name="Another")
        assert reconstituted.put(v2, reconstituted.put(v, s)) == reconstituted.put(
            v2, s
        )

    def test_reconstitute_requires_types_for_field_mapping(self):
        """Field-mapping reconstitution raises ValueError without types."""
        record_json = lens_to_json(person_name_lens, name="person_name")
        record = json.loads(record_json)

        with pytest.raises(ValueError, match="view_type is required"):
            lens_from_record(record, register=False)

    def test_reconstitute_opaque_raises(self):
        """Opaque getter kind raises ValueError."""
        record = {
            "name": "opaque_lens",
            "version": "1.0.0",
            "source_schema": "Src",
            "view_schema": "View",
            "getter": {"kind": "opaque"},
            "putter": {"kind": "opaque"},
        }
        with pytest.raises(ValueError, match="Cannot reconstitute"):
            lens_from_record(record, register=False)

    def test_caching(self):
        """Reconstituted lenses are cached by name+version."""
        record_json = lens_to_json(person_name_lens, name="cached_lens")
        record = json.loads(record_json)

        l1 = lens_from_record(
            record,
            source_type=PersonFull,
            view_type=PersonName,
            register=False,
            use_cache=True,
        )
        l2 = lens_from_record(
            record,
            source_type=PersonFull,
            view_type=PersonName,
            register=False,
            use_cache=True,
        )
        assert l1 is l2

    def test_no_caching(self):
        """With use_cache=False, separate instances are returned."""
        record_json = lens_to_json(person_name_lens, name="nocache_lens")
        record = json.loads(record_json)

        l1 = lens_from_record(
            record,
            source_type=PersonFull,
            view_type=PersonName,
            register=False,
            use_cache=False,
        )
        l2 = lens_from_record(
            record,
            source_type=PersonFull,
            view_type=PersonName,
            register=False,
            use_cache=False,
        )
        assert l1 is not l2


# ---------------------------------------------------------------------------
# _lens_codec: Field mapping detection
# ---------------------------------------------------------------------------


class TestFieldMappingDetection:
    """Tests for _detect_field_mapping."""

    def test_detects_subset_fields(self):
        mappings = _detect_field_mapping(person_name_lens, PersonFull, PersonName)
        assert mappings is not None
        assert len(mappings) == 1
        assert mappings[0]["source_field"] == "name"

    def test_rejects_non_subset(self):
        """Returns None when view has fields not in source."""

        @packable
        class Src:
            name: str

        @packable
        class View:
            name: str
            extra: int

        @lens
        def bad_lens(s: Src) -> View:
            return View(name=s.name, extra=0)

        result = _detect_field_mapping(bad_lens, Src, View)
        assert result is None


# ---------------------------------------------------------------------------
# _lens_codec: Function reference resolution
# ---------------------------------------------------------------------------


class TestFunctionRefResolution:
    """Tests for _resolve_function_ref."""

    def test_module_level_function(self):
        ref = _resolve_function_ref(person_name_lens._getter)
        assert ref is not None
        assert "module" in ref
        assert "qualname" in ref

    def test_lambda_returns_none(self):
        f = lambda x: x  # noqa: E731
        assert _resolve_function_ref(f) is None

    def test_local_function_returns_none(self):
        def local_func(x):
            return x

        assert _resolve_function_ref(local_func) is None


# ---------------------------------------------------------------------------
# _lens_codec: Stub generation
# ---------------------------------------------------------------------------


class TestLensStubGeneration:
    """Tests for generate_lens_stub."""

    def test_generates_stub_content(self):
        record = {
            "name": "test_lens",
            "version": "1.0.0",
            "source_schema": "SourceType",
            "view_schema": "ViewType",
            "getter": {
                "kind": "field_mapping",
                "mappings": [
                    {"source_field": "name", "view_field": "name", "transform": None},
                ],
            },
            "putter": {"kind": "field_mapping", "mappings": []},
        }
        stub = generate_lens_stub(record)
        assert "test_lens" in stub
        assert "SourceType" in stub
        assert "ViewType" in stub
        assert "LENS_NAME" in stub
        assert "LENS_VERSION" in stub

    def test_stub_with_description(self):
        record = {
            "name": "described_lens",
            "version": "2.0.0",
            "source_schema": "A",
            "view_schema": "B",
            "getter": {"kind": "opaque"},
            "putter": {"kind": "opaque"},
            "description": "A test description",
        }
        stub = generate_lens_stub(record)
        assert "A test description" in stub


# ---------------------------------------------------------------------------
# Provider: SQLite lens storage
# ---------------------------------------------------------------------------


class TestSqliteProviderLens:
    """Tests for SqliteProvider lens CRUD operations."""

    def test_store_and_get(self, sqlite_provider):
        lens_json = '{"name": "test", "version": "1.0.0"}'
        sqlite_provider.store_lens("test", "1.0.0", lens_json)

        result = sqlite_provider.get_lens_json("test", "1.0.0")
        assert result == lens_json

    def test_get_nonexistent(self, sqlite_provider):
        assert sqlite_provider.get_lens_json("missing", "1.0.0") is None

    def test_upsert(self, sqlite_provider):
        sqlite_provider.store_lens("test", "1.0.0", '{"v": 1}')
        sqlite_provider.store_lens("test", "1.0.0", '{"v": 2}')

        result = sqlite_provider.get_lens_json("test", "1.0.0")
        assert json.loads(result) == {"v": 2}

    def test_iter_lenses(self, sqlite_provider):
        sqlite_provider.store_lens("a", "1.0.0", '{"name": "a"}')
        sqlite_provider.store_lens("b", "1.0.0", '{"name": "b"}')

        results = list(sqlite_provider.iter_lenses())
        assert len(results) == 2
        names = {r[0] for r in results}
        assert names == {"a", "b"}

    def test_find_latest_lens_version(self, sqlite_provider):
        sqlite_provider.store_lens("test", "1.0.0", "{}")
        sqlite_provider.store_lens("test", "1.0.1", "{}")
        sqlite_provider.store_lens("test", "2.0.0", "{}")

        assert sqlite_provider.find_latest_lens_version("test") == "2.0.0"

    def test_find_latest_lens_version_none(self, sqlite_provider):
        assert sqlite_provider.find_latest_lens_version("missing") is None

    def test_find_lenses_by_schemas(self, sqlite_provider):
        lens1 = json.dumps({"source_schema": "A", "view_schema": "B"})
        lens2 = json.dumps({"source_schema": "A", "view_schema": "C"})
        lens3 = json.dumps({"source_schema": "X", "view_schema": "Y"})

        sqlite_provider.store_lens("l1", "1.0.0", lens1)
        sqlite_provider.store_lens("l2", "1.0.0", lens2)
        sqlite_provider.store_lens("l3", "1.0.0", lens3)

        # Find by source only
        results = sqlite_provider.find_lenses_by_schemas("A")
        assert len(results) == 2

        # Find by source and view
        results = sqlite_provider.find_lenses_by_schemas("A", "B")
        assert len(results) == 1
        assert results[0][0] == "l1"

        # No match
        results = sqlite_provider.find_lenses_by_schemas("Z")
        assert len(results) == 0


# ---------------------------------------------------------------------------
# Index: Lens operations
# ---------------------------------------------------------------------------


class TestIndexLensOperations:
    """Tests for Index lens storage, retrieval, and reconstitution."""

    def test_store_lens_returns_ref(self, index):
        ref = index.store_lens(person_name_lens, name="person_name")
        assert ref == "atdata://local/lens/person_name@1.0.0"

    def test_store_lens_with_explicit_version(self, index):
        ref = index.store_lens(person_name_lens, name="person_name", version="3.0.0")
        assert ref == "atdata://local/lens/person_name@3.0.0"

    def test_get_lens_record(self, index):
        index.store_lens(person_name_lens, name="person_name")
        record = index.get_lens("atdata://local/lens/person_name@1.0.0")

        assert record["name"] == "person_name"
        assert record["version"] == "1.0.0"
        assert record["source_schema"] == "PersonFull"
        assert record["view_schema"] == "PersonName"
        assert "$ref" in record

    def test_get_lens_not_found(self, index):
        with pytest.raises(KeyError, match="Lens not found"):
            index.get_lens("atdata://local/lens/missing@1.0.0")

    def test_get_lens_invalid_ref(self, index):
        with pytest.raises(ValueError, match="Invalid lens reference"):
            index.get_lens("invalid://ref")

    def test_load_lens_field_mapping(self, index):
        """Store and reconstitute a field-mapping lens via Index."""
        index.store_lens(person_name_lens, name="person_name")
        reconstituted = index.load_lens(
            "atdata://local/lens/person_name@1.0.0",
            source_type=PersonFull,
            view_type=PersonName,
        )

        source = PersonFull(name="Alice", age=30, height=170.0)
        view = reconstituted.get(source)
        assert view.name == "Alice"

        updated = reconstituted.put(PersonName(name="Bob"), source)
        assert updated.name == "Bob"
        assert updated.age == 30

    def test_list_lenses(self, index):
        index.store_lens(person_name_lens, name="lens_a")
        index.store_lens(person_name_lens, name="lens_b", version="1.0.0")

        result = index.list_lenses()
        assert len(result) == 2
        names = {r["name"] for r in result}
        assert names == {"lens_a", "lens_b"}

    def test_lenses_property_lazy(self, index):
        """The lenses property yields records lazily."""
        index.store_lens(person_name_lens, name="lazy_lens")
        gen = index.lenses
        # It's a generator
        record = next(gen)
        assert record["name"] == "lazy_lens"

    def test_find_lenses(self, index):
        index.store_lens(
            person_name_lens,
            name="l1",
            source_schema="PersonFull",
            view_schema="PersonName",
        )
        index.store_lens(
            person_name_lens,
            name="l2",
            source_schema="PersonFull",
            view_schema="Other",
        )

        results = index.find_lenses("PersonFull")
        assert len(results) == 2

        results = index.find_lenses("PersonFull", "PersonName")
        assert len(results) == 1
        assert results[0]["name"] == "l1"

    def test_version_auto_increment(self, index):
        ref1 = index.store_lens(person_name_lens, name="auto_ver")
        assert ref1.endswith("@1.0.0")

        ref2 = index.store_lens(person_name_lens, name="auto_ver")
        assert ref2.endswith("@1.0.1")

        ref3 = index.store_lens(person_name_lens, name="auto_ver")
        assert ref3.endswith("@1.0.2")


# ---------------------------------------------------------------------------
# Index with auto_stubs: lens stubs
# ---------------------------------------------------------------------------


class TestIndexLensStubs:
    """Tests for Index auto_stubs integration with lenses."""

    def test_store_lens_generates_stub(self, tmp_path):
        stub_dir = tmp_path / "stubs"
        idx = Index(
            provider="sqlite",
            path=tmp_path / "index.db",
            atmosphere=None,
            auto_stubs=True,
            stub_dir=stub_dir,
        )
        idx.store_lens(person_name_lens, name="stub_test")

        stubs = idx._stub_manager.list_lens_stubs()
        assert len(stubs) == 1
        content = stubs[0].read_text()
        assert "stub_test" in content


# ---------------------------------------------------------------------------
# _parse_lens_ref
# ---------------------------------------------------------------------------


class TestParseLensRef:
    """Tests for _parse_lens_ref."""

    def test_valid_ref(self):
        name, version = _parse_lens_ref("atdata://local/lens/my_lens@1.0.0")
        assert name == "my_lens"
        assert version == "1.0.0"

    def test_invalid_prefix(self):
        with pytest.raises(ValueError, match="Invalid lens reference"):
            _parse_lens_ref("local://lenses/something@1.0.0")

    def test_missing_version(self):
        with pytest.raises(ValueError, match="must include version"):
            _parse_lens_ref("atdata://local/lens/no_version")


# ---------------------------------------------------------------------------
# Multi-field lens round-trip
# ---------------------------------------------------------------------------


class TestMultiFieldLensRoundTrip:
    """End-to-end test with a multi-field lens."""

    def test_physical_view_roundtrip(self, index):
        """Store and reconstitute a two-field view lens."""

        @lens
        def physical_lens(s: PersonFull) -> PersonPhysical:
            return PersonPhysical(name=s.name, height=s.height)

        @physical_lens.putter
        def physical_put(v: PersonPhysical, s: PersonFull) -> PersonFull:
            return PersonFull(name=v.name, age=s.age, height=v.height)

        ref = index.store_lens(physical_lens, name="physical")
        recon = index.load_lens(ref, source_type=PersonFull, view_type=PersonPhysical)

        src = PersonFull(name="Alice", age=25, height=165.0)

        # Getter works
        view = recon.get(src)
        assert view.name == "Alice"
        assert view.height == 165.0

        # Putter works
        updated = recon.put(PersonPhysical(name="Bob", height=180.0), src)
        assert updated.name == "Bob"
        assert updated.age == 25  # Preserved
        assert updated.height == 180.0

        # Lens laws
        v = PersonPhysical(name="Test", height=175.0)
        assert recon.get(recon.put(v, src)) == v  # GetPut
        assert recon.put(recon.get(src), src) == src  # PutGet

        v2 = PersonPhysical(name="Other", height=190.0)
        assert recon.put(v2, recon.put(v, src)) == recon.put(v2, src)  # PutPut


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_store_lens_with_description(self, index):
        ref = index.store_lens(
            person_name_lens,
            name="described",
            description="Extract name from person",
        )
        record = index.get_lens(ref)
        assert record["description"] == "Extract name from person"

    def test_empty_lenses_list(self, index):
        assert index.list_lenses() == []

    def test_find_lenses_empty(self, index):
        assert index.find_lenses("NonExistent") == []

    def test_multiple_versions_coexist(self, index):
        index.store_lens(person_name_lens, name="versioned", version="1.0.0")
        index.store_lens(person_name_lens, name="versioned", version="2.0.0")

        r1 = index.get_lens("atdata://local/lens/versioned@1.0.0")
        r2 = index.get_lens("atdata://local/lens/versioned@2.0.0")
        assert r1["version"] == "1.0.0"
        assert r2["version"] == "2.0.0"

    def test_lenses_have_ref_field(self, index):
        index.store_lens(person_name_lens, name="with_ref")
        record = index.get_lens("atdata://local/lens/with_ref@1.0.0")
        assert record["$ref"] == "atdata://local/lens/with_ref@1.0.0"

    def test_lenses_iterator_includes_ref(self, index):
        index.store_lens(person_name_lens, name="iter_ref")
        records = list(index.lenses)
        assert records[0]["$ref"] == "atdata://local/lens/iter_ref@1.0.0"

    def test_find_lenses_by_schemas_skips_malformed_json(self, tmp_path):
        """find_lenses_by_schemas gracefully skips records with invalid JSON."""
        provider = SqliteProvider(path=tmp_path / "bad.db")
        # Insert valid lens
        provider.store_lens(
            "good",
            "1.0.0",
            json.dumps(
                {
                    "source_schema": "Src",
                    "view_schema": "View",
                }
            ),
        )
        # Insert malformed JSON directly
        provider._conn.execute(
            "INSERT INTO lenses (name, version, lens_json) VALUES (?, ?, ?)",
            ("bad", "1.0.0", "{truncated"),
        )
        provider._conn.commit()

        results = provider.find_lenses_by_schemas("Src")
        assert len(results) == 1
        assert results[0][0] == "good"
        provider.close()

    def test_reconstitute_missing_getter_key_raises(self):
        """lens_from_record raises KeyError when 'getter' is missing."""
        record = {"name": "bad", "version": "1.0.0", "putter": {"kind": "opaque"}}
        with pytest.raises(KeyError):
            lens_from_record(record, use_cache=False, register=False)

    def test_reconstitute_opaque_getter_raises(self):
        """lens_from_record raises ValueError for opaque getter kind."""
        record = {
            "name": "opaque_test",
            "version": "1.0.0",
            "getter": {"kind": "opaque"},
            "putter": {"kind": "opaque"},
        }
        with pytest.raises(ValueError, match="Cannot reconstitute"):
            lens_from_record(record, use_cache=False, register=False)

    def test_reconstitute_field_mapping_without_view_type_raises(self):
        """field_mapping getter requires view_type to be provided."""
        record = {
            "name": "no_view",
            "version": "1.0.0",
            "getter": {
                "kind": "field_mapping",
                "mappings": [{"source_field": "x", "view_field": "x"}],
            },
            "putter": {"kind": "opaque"},
        }
        with pytest.raises(ValueError, match="view_type is required"):
            lens_from_record(record, view_type=None, use_cache=False, register=False)
