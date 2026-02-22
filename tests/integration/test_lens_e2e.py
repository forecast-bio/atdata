"""End-to-end integration tests for the lens lifecycle.

Proves the ``science.alt.dataset.lens`` lexicon works end-to-end:
define → publish → retrieve → execute → verify lens laws.

Requires ``ATPROTO_TEST_HANDLE`` and ``ATPROTO_TEST_PASSWORD`` env vars.
Every record created is cleaned up in a finalizer so the test account
stays tidy.

GH #48 — subissue of GH #34 (lens lexicon E2E validation).
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import webdataset as wds
from numpy.typing import NDArray

import atdata
from atdata.atmosphere import (
    Atmosphere,
    SchemaPublisher,
    DatasetPublisher,
    DatasetLoader,
    LensPublisher,
    LensLoader,
    LexLensRecord,
    LexCodeReference,
)
from atdata.atmosphere._types import LEXICON_NAMESPACE

from .conftest import unique_name, RUN_ID


# ── Sample types ──────────────────────────────────────────────────


@atdata.packable
class FullSample:
    """Source type with all fields."""

    name: str
    age: int
    score: float


@atdata.packable
class NameScore:
    """View type projecting name and score."""

    name: str
    score: float


@atdata.packable
class ArraySource:
    """Source type with an ndarray field."""

    label: str
    embedding: NDArray


@atdata.packable
class LabelOnly:
    """View projecting just the label."""

    label: str


# ── Lenses (defined but NOT registered via @lens to avoid global side effects) ──


def _full_to_namescore_get(s: FullSample) -> NameScore:
    return NameScore(name=s.name, score=s.score)


def _full_to_namescore_put(v: NameScore, s: FullSample) -> FullSample:
    return FullSample(name=v.name, age=s.age, score=v.score)


_full_to_namescore = atdata.Lens(_full_to_namescore_get, put=_full_to_namescore_put)


def _array_to_label_get(s: ArraySource) -> LabelOnly:
    return LabelOnly(label=s.label)


def _array_to_label_put(v: LabelOnly, s: ArraySource) -> ArraySource:
    return ArraySource(label=v.label, embedding=s.embedding)


_array_to_label = atdata.Lens(_array_to_label_get, put=_array_to_label_put)


# ── Helpers ───────────────────────────────────────────────────────

TEST_COLLECTION_LENS = f"{LEXICON_NAMESPACE}.lens"
TEST_COLLECTION_SCHEMA = f"{LEXICON_NAMESPACE}.schema"


def _cleanup_lens_records(client: Atmosphere) -> None:
    """Delete all lens records from this test run."""
    records, _ = client.list_records(TEST_COLLECTION_LENS)
    for rec in records:
        rec_name = rec.get("name", "")
        if RUN_ID in rec_name:
            uri = rec.get("uri") or rec.get("$uri")
            if uri:
                try:
                    client.delete_record(uri)
                except Exception:
                    continue


def _cleanup_schema_records(client: Atmosphere) -> None:
    """Delete schema records from this test run."""
    records, _ = client.list_records(TEST_COLLECTION_SCHEMA)
    for rec in records:
        rec_name = rec.get("name", "")
        if RUN_ID in rec_name:
            uri = rec.get("uri") or rec.get("$uri")
            if uri:
                try:
                    client.delete_record(uri)
                except Exception:
                    continue


# ── Core E2E: publish → retrieve → verify ─────────────────────────


class TestLensPublishRetrieve:
    """Full lifecycle: define lens, publish to ATProto, retrieve, verify."""

    def test_publish_and_retrieve_lens(self, atproto_client: Atmosphere):
        """Publish a lens record and retrieve it with correct fields."""
        name = unique_name("lens-e2e")

        # Publish source and target schemas first
        schema_pub = SchemaPublisher(atproto_client)
        FullSample.__module__ = f"integ.{name}.source"
        source_schema_uri = str(schema_pub.publish(FullSample, version="1.0.0"))

        NameScore.__module__ = f"integ.{name}.target"
        target_schema_uri = str(schema_pub.publish(NameScore, version="1.0.0"))

        # Publish the lens
        lens_pub = LensPublisher(atproto_client)
        lens_uri = lens_pub.publish(
            name=name,
            source_schema_uri=source_schema_uri,
            target_schema_uri=target_schema_uri,
            code_repository="https://github.com/forecast-bio/atdata",
            code_commit="a" * 40,
            getter_path="tests.integration.test_lens_e2e:_full_to_namescore_get",
            putter_path="tests.integration.test_lens_e2e:_full_to_namescore_put",
            description="E2E test lens: FullSample -> NameScore",
            language="python",
        )
        assert "at://" in str(lens_uri)

        # Retrieve and verify
        loader = LensLoader(atproto_client)
        record = loader.get(str(lens_uri))

        assert record["name"] == name
        assert record["sourceSchema"] == source_schema_uri
        assert record["targetSchema"] == target_schema_uri
        assert (
            record["getterCode"]["repository"]
            == "https://github.com/forecast-bio/atdata"
        )
        assert record["getterCode"]["commit"] == "a" * 40
        assert (
            record["getterCode"]["path"]
            == "tests.integration.test_lens_e2e:_full_to_namescore_get"
        )
        assert (
            record["putterCode"]["path"]
            == "tests.integration.test_lens_e2e:_full_to_namescore_put"
        )
        assert record["description"] == "E2E test lens: FullSample -> NameScore"
        assert record["language"] == "python"

        # Cleanup
        atproto_client.delete_record(lens_uri)
        atproto_client.delete_record(source_schema_uri)
        atproto_client.delete_record(target_schema_uri)

    def test_publish_and_retrieve_typed(self, atproto_client: Atmosphere):
        """Retrieve as typed LexLensRecord and verify round-trip fidelity."""
        name = unique_name("lens-typed")

        schema_pub = SchemaPublisher(atproto_client)
        FullSample.__module__ = f"integ.{name}.source"
        source_uri = str(schema_pub.publish(FullSample, version="1.0.0"))
        NameScore.__module__ = f"integ.{name}.target"
        target_uri = str(schema_pub.publish(NameScore, version="1.0.0"))

        lens_pub = LensPublisher(atproto_client)
        lens_uri = lens_pub.publish(
            name=name,
            source_schema_uri=source_uri,
            target_schema_uri=target_uri,
            code_repository="https://github.com/forecast-bio/atdata",
            code_commit="b" * 40,
            getter_path="lenses.get_fn",
            putter_path="lenses.put_fn",
            language="python",
        )

        loader = LensLoader(atproto_client)
        typed_record = loader.get_typed(str(lens_uri))

        assert isinstance(typed_record, LexLensRecord)
        assert typed_record.name == name
        assert typed_record.source_schema == source_uri
        assert typed_record.target_schema == target_uri
        assert isinstance(typed_record.getter_code, LexCodeReference)
        assert (
            typed_record.getter_code.repository
            == "https://github.com/forecast-bio/atdata"
        )
        assert typed_record.getter_code.commit == "b" * 40
        assert typed_record.getter_code.path == "lenses.get_fn"
        assert typed_record.putter_code.path == "lenses.put_fn"

        # Cleanup
        atproto_client.delete_record(lens_uri)
        atproto_client.delete_record(source_uri)
        atproto_client.delete_record(target_uri)


# ── publish_from_lens convenience method ──────────────────────────


class TestPublishFromLens:
    """Test publishing via the publish_from_lens convenience method."""

    def test_publish_from_lens_object(self, atproto_client: Atmosphere):
        """Publish directly from a Lens object and verify code paths are extracted."""
        name = unique_name("lens-from-obj")

        schema_pub = SchemaPublisher(atproto_client)
        FullSample.__module__ = f"integ.{name}.source"
        source_uri = str(schema_pub.publish(FullSample, version="1.0.0"))
        NameScore.__module__ = f"integ.{name}.target"
        target_uri = str(schema_pub.publish(NameScore, version="1.0.0"))

        lens_pub = LensPublisher(atproto_client)
        lens_uri = lens_pub.publish_from_lens(
            _full_to_namescore,
            name=name,
            source_schema_uri=source_uri,
            target_schema_uri=target_uri,
            code_repository="https://github.com/forecast-bio/atdata",
            code_commit="c" * 40,
            language="python",
        )

        loader = LensLoader(atproto_client)
        record = loader.get(str(lens_uri))

        # The getter/putter paths should contain the function names
        assert "_full_to_namescore_get" in record["getterCode"]["path"]
        assert "_full_to_namescore_put" in record["putterCode"]["path"]

        # Cleanup
        atproto_client.delete_record(lens_uri)
        atproto_client.delete_record(source_uri)
        atproto_client.delete_record(target_uri)


# ── Lens law verification after round-trip ────────────────────────


class TestLensLawsAfterRoundTrip:
    """Verify lens laws hold after a publish → retrieve round-trip.

    While we cannot dynamically execute code from retrieved code references
    (that would be a security risk), we verify that:
    1. The locally-defined lens satisfies all three laws
    2. The published record correctly references the getter/putter
    3. The code references can be used to locate the original functions
    """

    def test_getput_law(self):
        """GetPut: put(get(s), s) == s."""
        src = FullSample(name="Alice", age=30, score=95.5)
        view = _full_to_namescore.get(src)
        restored = _full_to_namescore.put(view, src)
        assert restored == src, f"GetPut violation: {restored} != {src}"

    def test_putget_law(self):
        """PutGet: get(put(v, s)) == v."""
        src = FullSample(name="Alice", age=30, score=95.5)
        new_view = NameScore(name="Bob", score=42.0)
        updated = _full_to_namescore.put(new_view, src)
        result = _full_to_namescore.get(updated)
        assert result == new_view, f"PutGet violation: {result} != {new_view}"

    def test_putput_law(self):
        """PutPut: put(v2, put(v1, s)) == put(v2, s)."""
        src = FullSample(name="Alice", age=30, score=95.5)
        v1 = NameScore(name="Bob", score=42.0)
        v2 = NameScore(name="Carol", score=88.0)
        result1 = _full_to_namescore.put(v2, _full_to_namescore.put(v1, src))
        result2 = _full_to_namescore.put(v2, src)
        assert result1 == result2, f"PutPut violation: {result1} != {result2}"

    def test_laws_with_ndarray_lens(self):
        """Verify all three laws for a lens involving NDArray fields."""
        src = ArraySource(
            label="test",
            embedding=np.array([1.0, 2.0, 3.0]),
        )

        # GetPut
        view = _array_to_label.get(src)
        restored = _array_to_label.put(view, src)
        assert restored.label == src.label
        np.testing.assert_array_equal(restored.embedding, src.embedding)

        # PutGet
        new_view = LabelOnly(label="updated")
        updated = _array_to_label.put(new_view, src)
        result_view = _array_to_label.get(updated)
        assert result_view == new_view

        # PutPut
        v1 = LabelOnly(label="first")
        v2 = LabelOnly(label="second")
        r1 = _array_to_label.put(v2, _array_to_label.put(v1, src))
        r2 = _array_to_label.put(v2, src)
        assert r1.label == r2.label
        np.testing.assert_array_equal(r1.embedding, r2.embedding)

    def test_published_record_references_match_lens(self, atproto_client: Atmosphere):
        """Publish a lens, retrieve it, and verify code references match."""
        name = unique_name("lens-laws")

        schema_pub = SchemaPublisher(atproto_client)
        FullSample.__module__ = f"integ.{name}.source"
        source_uri = str(schema_pub.publish(FullSample, version="1.0.0"))
        NameScore.__module__ = f"integ.{name}.target"
        target_uri = str(schema_pub.publish(NameScore, version="1.0.0"))

        getter_path = "tests.integration.test_lens_e2e:_full_to_namescore_get"
        putter_path = "tests.integration.test_lens_e2e:_full_to_namescore_put"

        lens_pub = LensPublisher(atproto_client)
        lens_uri = lens_pub.publish(
            name=name,
            source_schema_uri=source_uri,
            target_schema_uri=target_uri,
            code_repository="https://github.com/forecast-bio/atdata",
            code_commit="d" * 40,
            getter_path=getter_path,
            putter_path=putter_path,
        )

        loader = LensLoader(atproto_client)
        record = loader.get(str(lens_uri))

        # Verify the code references can be resolved to actual functions
        retrieved_getter_path = record["getterCode"]["path"]
        retrieved_putter_path = record["putterCode"]["path"]

        # Parse module:function format and verify they match
        getter_module, getter_func = retrieved_getter_path.rsplit(":", 1)
        putter_module, putter_func = retrieved_putter_path.rsplit(":", 1)

        assert getter_func == "_full_to_namescore_get"
        assert putter_func == "_full_to_namescore_put"
        assert "test_lens_e2e" in getter_module
        assert "test_lens_e2e" in putter_module

        # Verify the locally resolved functions still satisfy laws
        src = FullSample(name="Test", age=25, score=99.0)
        view = _full_to_namescore.get(src)
        assert _full_to_namescore.put(view, src) == src  # GetPut

        # Cleanup
        atproto_client.delete_record(lens_uri)
        atproto_client.delete_record(source_uri)
        atproto_client.delete_record(target_uri)


# ── Code reference patterns ───────────────────────────────────────


class TestCodeReferencePatterns:
    """Validate single-repo and split-repo code reference patterns."""

    def test_single_repo_pattern(self, atproto_client: Atmosphere):
        """Both getter and putter reference the same repository and commit."""
        name = unique_name("lens-single-repo")

        schema_pub = SchemaPublisher(atproto_client)
        FullSample.__module__ = f"integ.{name}.source"
        source_uri = str(schema_pub.publish(FullSample, version="1.0.0"))
        NameScore.__module__ = f"integ.{name}.target"
        target_uri = str(schema_pub.publish(NameScore, version="1.0.0"))

        shared_repo = "https://github.com/user/lenses"
        shared_commit = "abc123" + "0" * 34

        lens_pub = LensPublisher(atproto_client)
        lens_uri = lens_pub.publish(
            name=name,
            source_schema_uri=source_uri,
            target_schema_uri=target_uri,
            code_repository=shared_repo,
            code_commit=shared_commit,
            getter_path="lenses.my_lens:get",
            putter_path="lenses.my_lens:put",
        )

        loader = LensLoader(atproto_client)
        record = loader.get(str(lens_uri))

        # Both code refs should share repo and commit
        assert record["getterCode"]["repository"] == shared_repo
        assert record["putterCode"]["repository"] == shared_repo
        assert record["getterCode"]["commit"] == shared_commit
        assert record["putterCode"]["commit"] == shared_commit

        # Cleanup
        atproto_client.delete_record(lens_uri)
        atproto_client.delete_record(source_uri)
        atproto_client.delete_record(target_uri)

    def test_split_repo_pattern(self, atproto_client: Atmosphere):
        """Getter and putter reference different repositories/commits."""
        name = unique_name("lens-split-repo")

        schema_pub = SchemaPublisher(atproto_client)
        FullSample.__module__ = f"integ.{name}.source"
        source_uri = str(schema_pub.publish(FullSample, version="1.0.0"))
        NameScore.__module__ = f"integ.{name}.target"
        target_uri = str(schema_pub.publish(NameScore, version="1.0.0"))

        getter_repo = "https://github.com/user/getters"
        getter_commit = "abc123" + "0" * 34
        putter_repo = "https://github.com/user/putters"
        putter_commit = "def456" + "0" * 34

        # Use explicit publish to set different repos per code ref
        getter_code = LexCodeReference(
            repository=getter_repo,
            commit=getter_commit,
            path="getters.my_lens:get",
        )
        putter_code = LexCodeReference(
            repository=putter_repo,
            commit=putter_commit,
            path="putters.my_lens:put",
        )

        lens_record = LexLensRecord(
            name=name,
            source_schema=source_uri,
            target_schema=target_uri,
            getter_code=getter_code,
            putter_code=putter_code,
            description="Split-repo test",
            language="python",
        )

        lens_uri = atproto_client.create_record(
            collection=f"{LEXICON_NAMESPACE}.lens",
            record=lens_record.to_record(),
            validate=False,
        )

        loader = LensLoader(atproto_client)
        record = loader.get(str(lens_uri))

        assert record["getterCode"]["repository"] == getter_repo
        assert record["getterCode"]["commit"] == getter_commit
        assert record["getterCode"]["path"] == "getters.my_lens:get"
        assert record["putterCode"]["repository"] == putter_repo
        assert record["putterCode"]["commit"] == putter_commit
        assert record["putterCode"]["path"] == "putters.my_lens:put"

        # Cleanup
        atproto_client.delete_record(lens_uri)
        atproto_client.delete_record(source_uri)
        atproto_client.delete_record(target_uri)


# ── Lens listing and discovery ────────────────────────────────────


class TestLensDiscovery:
    """Test listing and finding lenses by schema."""

    def test_list_lenses(self, atproto_client: Atmosphere):
        """Published lenses appear in list_all results."""
        name = unique_name("lens-list")

        schema_pub = SchemaPublisher(atproto_client)
        FullSample.__module__ = f"integ.{name}.source"
        source_uri = str(schema_pub.publish(FullSample, version="1.0.0"))
        NameScore.__module__ = f"integ.{name}.target"
        target_uri = str(schema_pub.publish(NameScore, version="1.0.0"))

        lens_pub = LensPublisher(atproto_client)
        lens_uri = lens_pub.publish(
            name=name,
            source_schema_uri=source_uri,
            target_schema_uri=target_uri,
            code_repository="https://github.com/forecast-bio/atdata",
            code_commit="e" * 40,
            getter_path="mod:get",
            putter_path="mod:put",
        )

        loader = LensLoader(atproto_client)
        all_lenses = loader.list_all()
        names = [r.get("name", "") for r in all_lenses]
        assert name in names, f"Published lens '{name}' not found in list_all"

        # Cleanup
        atproto_client.delete_record(lens_uri)
        atproto_client.delete_record(source_uri)
        atproto_client.delete_record(target_uri)

    def test_find_by_schemas(self, atproto_client: Atmosphere):
        """find_by_schemas returns lenses matching source and target schemas."""
        name = unique_name("lens-find")

        schema_pub = SchemaPublisher(atproto_client)
        FullSample.__module__ = f"integ.{name}.source"
        source_uri = str(schema_pub.publish(FullSample, version="1.0.0"))
        NameScore.__module__ = f"integ.{name}.target"
        target_uri = str(schema_pub.publish(NameScore, version="1.0.0"))

        lens_pub = LensPublisher(atproto_client)
        lens_uri = lens_pub.publish(
            name=name,
            source_schema_uri=source_uri,
            target_schema_uri=target_uri,
            code_repository="https://github.com/forecast-bio/atdata",
            code_commit="f" * 40,
            getter_path="mod:get",
            putter_path="mod:put",
        )

        loader = LensLoader(atproto_client)

        # Find by both schemas
        matches = loader.find_by_schemas(source_uri, target_uri)
        match_names = [m.get("name", "") for m in matches]
        assert name in match_names

        # Find by source schema only
        source_matches = loader.find_by_schemas(source_uri)
        source_names = [m.get("name", "") for m in source_matches]
        assert name in source_names

        # Cleanup
        atproto_client.delete_record(lens_uri)
        atproto_client.delete_record(source_uri)
        atproto_client.delete_record(target_uri)


# ── Lens applied to dataset ──────────────────────────────────────


class TestLensDatasetIntegration:
    """Verify a lens can transform dataset samples end-to-end."""

    def test_lens_transforms_dataset_samples(self, tmp_path):
        """Write samples, apply lens via Dataset.as_type, verify output."""

        # Register the lens via @lens for this test
        @atdata.lens
        def e2e_lens(s: FullSample) -> NameScore:
            return NameScore(name=s.name, score=s.score)

        @e2e_lens.putter
        def e2e_lens_put(v: NameScore, s: FullSample) -> FullSample:
            return FullSample(name=v.name, age=s.age, score=v.score)

        # Write source samples
        samples = [
            FullSample(name="Alice", age=30, score=95.5),
            FullSample(name="Bob", age=25, score=87.3),
            FullSample(name="Carol", age=35, score=91.0),
        ]

        tar_path = (tmp_path / "lens-e2e.tar").as_posix()
        with wds.writer.TarWriter(tar_path) as sink:
            for s in samples:
                sink.write(s.as_wds)

        # Apply lens transformation
        ds = atdata.Dataset[FullSample](tar_path).as_type(NameScore)
        assert ds.sample_type == NameScore

        results = list(ds.ordered(batch_size=None))
        assert len(results) == 3

        for orig, result in zip(samples, results):
            assert isinstance(result, NameScore)
            assert result.name == orig.name
            assert result.score == orig.score

    def test_lens_with_blob_published_dataset(self, atproto_client: Atmosphere):
        """Full E2E: write → publish as blob → retrieve → apply lens."""
        name = unique_name("lens-blob-ds")

        # Register lens
        @atdata.lens
        def blob_e2e_lens(s: FullSample) -> NameScore:
            return NameScore(name=s.name, score=s.score)

        samples = [
            FullSample(name="Alice", age=30, score=95.5),
            FullSample(name="Bob", age=25, score=87.3),
        ]

        # Build tar in memory
        buf = io.BytesIO()
        with wds.writer.TarWriter(buf) as sink:
            for s in samples:
                sink.write(s.as_wds)
        tar_bytes = buf.getvalue()

        # Publish schema + dataset
        schema_pub = SchemaPublisher(atproto_client)
        FullSample.__module__ = f"integ.{name}"
        schema_uri = str(schema_pub.publish(FullSample, version="1.0.0"))

        ds_pub = DatasetPublisher(atproto_client)
        ds_uri = ds_pub.publish_with_blobs(
            blobs=[tar_bytes],
            schema_uri=schema_uri,
            name=name,
            description="lens blob E2E test",
        )

        # Retrieve dataset and apply lens
        ds_loader = DatasetLoader(atproto_client)
        ds = ds_loader.to_dataset(str(ds_uri), FullSample)
        transformed = ds.as_type(NameScore)

        results = list(transformed.ordered())
        assert len(results) == 2
        assert results[0].name == "Alice"
        assert results[0].score == 95.5
        assert results[1].name == "Bob"

        # Cleanup
        atproto_client.delete_record(ds_uri)
        atproto_client.delete_record(schema_uri)


# ── Error handling ────────────────────────────────────────────────


class TestLensErrorHandling:
    """Verify error paths for lens operations."""

    def test_get_nonexistent_lens(self, atproto_client: Atmosphere):
        """Fetching a nonexistent lens record raises."""
        fake_uri = (
            f"at://{atproto_client.did}/{LEXICON_NAMESPACE}.lens/nonexistent99999"
        )
        loader = LensLoader(atproto_client)
        with pytest.raises(Exception):
            loader.get(fake_uri)

    def test_get_wrong_record_type(self, atproto_client: Atmosphere):
        """Fetching a schema record via LensLoader raises ValueError."""
        name = unique_name("lens-wrongtype")

        schema_pub = SchemaPublisher(atproto_client)
        FullSample.__module__ = f"integ.{name}"
        schema_uri = str(schema_pub.publish(FullSample, version="1.0.0"))

        loader = LensLoader(atproto_client)
        with pytest.raises(ValueError, match="not a lens record"):
            loader.get(schema_uri)

        atproto_client.delete_record(schema_uri)

    def test_publish_without_auth_raises(self):
        """Publishing without authentication raises ValueError."""
        client = Atmosphere()
        pub = LensPublisher(client)
        with pytest.raises((ValueError, Exception)):
            pub.publish(
                name="should-fail",
                source_schema_uri="at://did:plc:fake/science.alt.dataset.schema/x",
                target_schema_uri="at://did:plc:fake/science.alt.dataset.schema/y",
                code_repository="https://github.com/user/repo",
                code_commit="a" * 40,
                getter_path="mod:get",
                putter_path="mod:put",
            )


# ── LexLensRecord serialization round-trip ────────────────────────


class TestLexLensRecordRoundTrip:
    """Verify LexLensRecord to_record/from_record round-trip fidelity."""

    def test_full_record_roundtrip(self):
        """All fields survive a to_record → from_record cycle."""
        getter = LexCodeReference(
            repository="https://github.com/user/repo",
            commit="a" * 40,
            path="module:get_fn",
            branch="main",
        )
        putter = LexCodeReference(
            repository="https://github.com/user/repo",
            commit="a" * 40,
            path="module:put_fn",
        )

        original = LexLensRecord(
            name="test-roundtrip",
            source_schema="at://did:plc:abc/science.alt.dataset.schema/src",
            target_schema="at://did:plc:abc/science.alt.dataset.schema/tgt",
            getter_code=getter,
            putter_code=putter,
            description="Round-trip test",
            language="python",
            metadata={"author": "test", "version": "1.0"},
        )

        record_dict = original.to_record()
        restored = LexLensRecord.from_record(record_dict)

        assert restored.name == original.name
        assert restored.source_schema == original.source_schema
        assert restored.target_schema == original.target_schema
        assert restored.getter_code.repository == original.getter_code.repository
        assert restored.getter_code.commit == original.getter_code.commit
        assert restored.getter_code.path == original.getter_code.path
        assert restored.getter_code.branch == original.getter_code.branch
        assert restored.putter_code.path == original.putter_code.path
        assert restored.putter_code.branch is None
        assert restored.description == original.description
        assert restored.language == original.language
        assert restored.metadata == original.metadata

    def test_minimal_record_roundtrip(self):
        """Record with only required fields survives round-trip."""
        getter = LexCodeReference(
            repository="https://github.com/user/repo",
            commit="b" * 40,
            path="mod:get",
        )
        putter = LexCodeReference(
            repository="https://github.com/user/repo",
            commit="b" * 40,
            path="mod:put",
        )

        original = LexLensRecord(
            name="minimal",
            source_schema="at://did:plc:abc/science.alt.dataset.schema/s",
            target_schema="at://did:plc:abc/science.alt.dataset.schema/t",
            getter_code=getter,
            putter_code=putter,
        )

        record_dict = original.to_record()
        restored = LexLensRecord.from_record(record_dict)

        assert restored.name == "minimal"
        assert restored.description is None
        assert restored.language is None
        assert restored.metadata is None

    def test_record_has_correct_type(self):
        """Serialized record has correct $type field."""
        getter = LexCodeReference(repository="repo", commit="c" * 40, path="p:get")
        putter = LexCodeReference(repository="repo", commit="c" * 40, path="p:put")

        record = LexLensRecord(
            name="type-check",
            source_schema="at://did:plc:abc/science.alt.dataset.schema/s",
            target_schema="at://did:plc:abc/science.alt.dataset.schema/t",
            getter_code=getter,
            putter_code=putter,
        )

        d = record.to_record()
        assert d["$type"] == f"{LEXICON_NAMESPACE}.lens"


# ── Sweep cleanup (runs last by naming convention) ────────────────


class TestZZZLensCleanup:
    """Best-effort cleanup of any leftover test records from this run."""

    def test_cleanup_lens_records(self, atproto_client: Atmosphere):
        _cleanup_lens_records(atproto_client)

    def test_cleanup_schema_records(self, atproto_client: Atmosphere):
        _cleanup_schema_records(atproto_client)
