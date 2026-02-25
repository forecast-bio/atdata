"""Unit tests for new lexicon types and verification publisher/loader.

Tests LexCodeHash, LexLensVerification, updated LexCodeReference (language),
updated LexLensRecord (schema versions), VerificationPublisher, and
VerificationLoader.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from atdata.atmosphere._lexicon_types import (
    LexCodeHash,
    LexCodeReference,
    LexLensRecord,
    LexLensVerification,
)
from atdata.atmosphere._types import LEXICON_NAMESPACE
from atdata.atmosphere.verification import VerificationPublisher, VerificationLoader
from atdata.testing import MockAtmosphere


# ---------------------------------------------------------------------------
# LexCodeHash
# ---------------------------------------------------------------------------


class TestLexCodeHash:
    """Round-trip tests for LexCodeHash."""

    def test_roundtrip(self):
        original = LexCodeHash(algorithm="sha256", digest="abc123" * 10)
        d = original.to_record()
        restored = LexCodeHash.from_record(d)
        assert restored.algorithm == "sha256"
        assert restored.digest == "abc123" * 10

    def test_to_record_keys(self):
        h = LexCodeHash(algorithm="blake3", digest="deadbeef")
        d = h.to_record()
        assert set(d.keys()) == {"algorithm", "digest"}

    def test_from_record_required_fields(self):
        with pytest.raises(KeyError):
            LexCodeHash.from_record({"algorithm": "sha256"})
        with pytest.raises(KeyError):
            LexCodeHash.from_record({"digest": "abc"})


# ---------------------------------------------------------------------------
# LexCodeReference — language field
# ---------------------------------------------------------------------------


class TestLexCodeReferenceLanguage:
    """Tests for the new language field on LexCodeReference."""

    def test_language_roundtrip(self):
        ref = LexCodeReference(
            repository="https://github.com/user/repo",
            commit="a" * 40,
            path="mod:func",
            language="python",
        )
        d = ref.to_record()
        assert d["language"] == "python"
        restored = LexCodeReference.from_record(d)
        assert restored.language == "python"

    def test_language_none_omitted(self):
        ref = LexCodeReference(
            repository="https://github.com/user/repo",
            commit="b" * 40,
            path="mod:func",
        )
        d = ref.to_record()
        assert "language" not in d

    def test_backward_compat_missing_language(self):
        d = {"repository": "repo", "commit": "c" * 40, "path": "p:f"}
        ref = LexCodeReference.from_record(d)
        assert ref.language is None

    def test_language_with_branch(self):
        ref = LexCodeReference(
            repository="repo",
            commit="d" * 40,
            path="p:f",
            branch="main",
            language="rust",
        )
        d = ref.to_record()
        assert d["branch"] == "main"
        assert d["language"] == "rust"
        restored = LexCodeReference.from_record(d)
        assert restored.branch == "main"
        assert restored.language == "rust"


# ---------------------------------------------------------------------------
# LexLensRecord — schema version fields
# ---------------------------------------------------------------------------


class TestLexLensRecordSchemaVersions:
    """Tests for the new source/target schema version fields."""

    def _make_record(self, **kwargs):
        getter = LexCodeReference(repository="repo", commit="a" * 40, path="p:get")
        putter = LexCodeReference(repository="repo", commit="a" * 40, path="p:put")
        defaults = dict(
            name="test-lens",
            source_schema="at://did:plc:abc/science.alt.dataset.schema/src",
            target_schema="at://did:plc:abc/science.alt.dataset.schema/tgt",
            getter_code=getter,
            putter_code=putter,
        )
        defaults.update(kwargs)
        return LexLensRecord(**defaults)

    def test_schema_versions_roundtrip(self):
        record = self._make_record(
            source_schema_version="1.0.0",
            target_schema_version=">=2.0.0 <3.0.0",
        )
        d = record.to_record()
        assert d["sourceSchemaVersion"] == "1.0.0"
        assert d["targetSchemaVersion"] == ">=2.0.0 <3.0.0"
        restored = LexLensRecord.from_record(d)
        assert restored.source_schema_version == "1.0.0"
        assert restored.target_schema_version == ">=2.0.0 <3.0.0"

    def test_schema_versions_none_omitted(self):
        record = self._make_record()
        d = record.to_record()
        assert "sourceSchemaVersion" not in d
        assert "targetSchemaVersion" not in d

    def test_backward_compat_missing_versions(self):
        """Records without version fields deserialize with None."""
        getter = LexCodeReference(repository="repo", commit="a" * 40, path="p:get")
        putter = LexCodeReference(repository="repo", commit="a" * 40, path="p:put")
        d = {
            "$type": f"{LEXICON_NAMESPACE}.lens",
            "name": "old-lens",
            "sourceSchema": "at://did:plc:abc/s/src",
            "targetSchema": "at://did:plc:abc/s/tgt",
            "getterCode": getter.to_record(),
            "putterCode": putter.to_record(),
            "createdAt": datetime.now(timezone.utc).isoformat(),
        }
        restored = LexLensRecord.from_record(d)
        assert restored.source_schema_version is None
        assert restored.target_schema_version is None

    def test_only_source_version(self):
        record = self._make_record(source_schema_version="1.2.3")
        d = record.to_record()
        assert d["sourceSchemaVersion"] == "1.2.3"
        assert "targetSchemaVersion" not in d
        restored = LexLensRecord.from_record(d)
        assert restored.source_schema_version == "1.2.3"
        assert restored.target_schema_version is None


# ---------------------------------------------------------------------------
# LexLensVerification
# ---------------------------------------------------------------------------


class TestLexLensVerification:
    """Round-trip tests for LexLensVerification."""

    def test_minimal_roundtrip(self):
        ts = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        v = LexLensVerification(
            lens="at://did:plc:abc/science.alt.dataset.lens/xyz",
            lens_commit="bafyabc123",
            verification_method="codeReview",
            created_at=ts,
        )
        d = v.to_record()
        assert d["$type"] == f"{LEXICON_NAMESPACE}.lensVerification"
        assert d["lens"] == "at://did:plc:abc/science.alt.dataset.lens/xyz"
        assert d["lensCommit"] == "bafyabc123"
        assert d["verificationMethod"] == "codeReview"
        assert "codeHash" not in d
        assert "proofRef" not in d
        assert "description" not in d

        restored = LexLensVerification.from_record(d)
        assert restored.lens == v.lens
        assert restored.lens_commit == v.lens_commit
        assert restored.verification_method == "codeReview"
        assert restored.code_hash is None
        assert restored.proof_ref is None
        assert restored.description is None

    def test_full_roundtrip(self):
        ts = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        code_hash = LexCodeHash(algorithm="sha256", digest="abc" * 20)
        proof = LexCodeReference(
            repository="https://github.com/user/proofs",
            commit="f" * 40,
            path="proofs/lens_proof.lean",
            language="lean",
        )
        v = LexLensVerification(
            lens="at://did:plc:abc/science.alt.dataset.lens/xyz",
            lens_commit="bafyabc123",
            verification_method="formalProof",
            created_at=ts,
            code_hash=code_hash,
            proof_ref=proof,
            description="Verified via Lean4 proof",
        )
        d = v.to_record()
        assert d["codeHash"]["algorithm"] == "sha256"
        assert d["proofRef"]["repository"] == "https://github.com/user/proofs"
        assert d["proofRef"]["language"] == "lean"
        assert d["description"] == "Verified via Lean4 proof"

        restored = LexLensVerification.from_record(d)
        assert restored.code_hash is not None
        assert restored.code_hash.algorithm == "sha256"
        assert restored.code_hash.digest == "abc" * 20
        assert restored.proof_ref is not None
        assert restored.proof_ref.repository == "https://github.com/user/proofs"
        assert restored.proof_ref.language == "lean"
        assert restored.description == "Verified via Lean4 proof"

    def test_from_record_required_fields(self):
        with pytest.raises(KeyError):
            LexLensVerification.from_record({"lens": "x"})


# ---------------------------------------------------------------------------
# VerificationPublisher (mock)
# ---------------------------------------------------------------------------


class TestVerificationPublisher:
    """Unit tests for VerificationPublisher with MockAtmosphere."""

    @pytest.fixture()
    def client(self):
        c = MockAtmosphere()
        c.login("test.handle", "password")
        return c

    def test_publish_minimal(self, client):
        pub = VerificationPublisher(client)
        uri = pub.publish(
            lens_uri="at://did:plc:mock000000000000/science.alt.dataset.lens/abc",
            lens_commit="bafyabc",
            verification_method="codeReview",
        )
        assert "at://" in str(uri)
        assert "lensVerification" in str(uri)

        record = client.get_record(str(uri))
        assert record["$type"] == f"{LEXICON_NAMESPACE}.lensVerification"
        assert record["verificationMethod"] == "codeReview"

    def test_publish_with_code_hash(self, client):
        pub = VerificationPublisher(client)
        code_hash = LexCodeHash(algorithm="sha256", digest="dead" * 16)
        uri = pub.publish(
            lens_uri="at://did:plc:mock000000000000/science.alt.dataset.lens/abc",
            lens_commit="bafyabc",
            verification_method="signedHash",
            code_hash=code_hash,
            description="Signed hash verification",
        )
        record = client.get_record(str(uri))
        assert record["codeHash"]["algorithm"] == "sha256"
        assert record["description"] == "Signed hash verification"

    def test_publish_with_proof_ref(self, client):
        pub = VerificationPublisher(client)
        proof = LexCodeReference(
            repository="https://github.com/user/proofs",
            commit="a" * 40,
            path="proofs/test_suite.py",
            language="python",
        )
        uri = pub.publish(
            lens_uri="at://did:plc:mock000000000000/science.alt.dataset.lens/abc",
            lens_commit="bafyabc",
            verification_method="automatedTest",
            proof_ref=proof,
        )
        record = client.get_record(str(uri))
        assert record["proofRef"]["repository"] == "https://github.com/user/proofs"
        assert record["proofRef"]["language"] == "python"

    def test_publish_with_rkey(self, client):
        pub = VerificationPublisher(client)
        uri = pub.publish(
            lens_uri="at://did:plc:mock000000000000/science.alt.dataset.lens/abc",
            lens_commit="bafyabc",
            verification_method="codeReview",
            rkey="custom-key",
        )
        assert "custom-key" in str(uri)


# ---------------------------------------------------------------------------
# VerificationLoader (mock)
# ---------------------------------------------------------------------------


class TestVerificationLoader:
    """Unit tests for VerificationLoader with MockAtmosphere."""

    @pytest.fixture()
    def client(self):
        c = MockAtmosphere()
        c.login("test.handle", "password")
        return c

    def _publish(self, client, *, lens_uri, method="codeReview", **kwargs):
        pub = VerificationPublisher(client)
        return pub.publish(
            lens_uri=lens_uri,
            lens_commit="bafyabc",
            verification_method=method,
            **kwargs,
        )

    def test_get(self, client):
        uri = self._publish(
            client,
            lens_uri="at://did:plc:mock000000000000/science.alt.dataset.lens/abc",
        )
        loader = VerificationLoader(client)
        record = loader.get(str(uri))
        assert record["verificationMethod"] == "codeReview"

    def test_get_typed(self, client):
        uri = self._publish(
            client,
            lens_uri="at://did:plc:mock000000000000/science.alt.dataset.lens/abc",
            description="Test typed",
        )
        loader = VerificationLoader(client)
        typed = loader.get_typed(str(uri))
        assert isinstance(typed, LexLensVerification)
        assert typed.verification_method == "codeReview"
        assert typed.description == "Test typed"

    def test_get_wrong_type_raises(self, client):
        # Create a lens record (not verification)
        client.create_record(
            collection=f"{LEXICON_NAMESPACE}.lens",
            record={"$type": f"{LEXICON_NAMESPACE}.lens", "name": "test"},
            rkey="wrong-type",
        )
        uri = f"at://{client.did}/{LEXICON_NAMESPACE}.lens/wrong-type"
        loader = VerificationLoader(client)
        with pytest.raises(ValueError, match="not a lensVerification record"):
            loader.get(uri)

    def test_list_for_lens(self, client):
        lens_a = "at://did:plc:mock000000000000/science.alt.dataset.lens/a"
        lens_b = "at://did:plc:mock000000000000/science.alt.dataset.lens/b"
        self._publish(client, lens_uri=lens_a, method="codeReview")
        self._publish(client, lens_uri=lens_a, method="automatedTest")
        self._publish(client, lens_uri=lens_b, method="codeReview")

        loader = VerificationLoader(client)
        results = loader.list_for_lens(lens_a)
        assert len(results) == 2
        methods = {r["verificationMethod"] for r in results}
        assert methods == {"codeReview", "automatedTest"}

    def test_find_by_method(self, client):
        lens_a = "at://did:plc:mock000000000000/science.alt.dataset.lens/a"
        self._publish(client, lens_uri=lens_a, method="codeReview")
        self._publish(client, lens_uri=lens_a, method="automatedTest")
        self._publish(client, lens_uri=lens_a, method="codeReview")

        loader = VerificationLoader(client)
        results = loader.find_by_method("codeReview")
        assert len(results) == 2

    def test_list_for_lens_empty(self, client):
        loader = VerificationLoader(client)
        results = loader.list_for_lens(
            "at://did:plc:mock000000000000/science.alt.dataset.lens/nonexistent"
        )
        assert results == []

    def test_find_by_method_empty(self, client):
        loader = VerificationLoader(client)
        results = loader.find_by_method("formalProof")
        assert results == []


# ---------------------------------------------------------------------------
# LensPublisher schema version passthrough (mock)
# ---------------------------------------------------------------------------


class TestLensPublisherSchemaVersions:
    """Verify LensPublisher passes schema version params to the record."""

    @pytest.fixture()
    def client(self):
        c = MockAtmosphere()
        c.login("test.handle", "password")
        return c

    def test_publish_with_schema_versions(self, client):
        from atdata.atmosphere.lens import LensPublisher, LensLoader

        pub = LensPublisher(client)
        uri = pub.publish(
            name="versioned-lens",
            source_schema_uri="at://did:plc:abc/science.alt.dataset.schema/src",
            target_schema_uri="at://did:plc:abc/science.alt.dataset.schema/tgt",
            code_repository="https://github.com/user/repo",
            code_commit="a" * 40,
            getter_path="mod:get",
            putter_path="mod:put",
            source_schema_version="1.0.0",
            target_schema_version=">=2.0.0 <3.0.0",
        )

        record = client.get_record(str(uri))
        assert record["sourceSchemaVersion"] == "1.0.0"
        assert record["targetSchemaVersion"] == ">=2.0.0 <3.0.0"

        loader = LensLoader(client)
        typed = loader.get_typed(str(uri))
        assert typed.source_schema_version == "1.0.0"
        assert typed.target_schema_version == ">=2.0.0 <3.0.0"

    def test_publish_without_schema_versions(self, client):
        from atdata.atmosphere.lens import LensPublisher

        pub = LensPublisher(client)
        uri = pub.publish(
            name="no-version-lens",
            source_schema_uri="at://did:plc:abc/science.alt.dataset.schema/src",
            target_schema_uri="at://did:plc:abc/science.alt.dataset.schema/tgt",
            code_repository="https://github.com/user/repo",
            code_commit="b" * 40,
            getter_path="mod:get",
            putter_path="mod:put",
        )

        record = client.get_record(str(uri))
        assert "sourceSchemaVersion" not in record
        assert "targetSchemaVersion" not in record
