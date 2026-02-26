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
        assert restored == original

    def test_dict_inverse(self):
        """to_record → from_record → to_record produces identical dict."""
        original = LexCodeHash(algorithm="blake3", digest="deadbeef")
        assert (
            LexCodeHash.from_record(original.to_record()).to_record()
            == original.to_record()
        )

    def test_to_record_keys(self):
        h = LexCodeHash(algorithm="blake3", digest="deadbeef")
        d = h.to_record()
        assert set(d.keys()) == {"algorithm", "digest"}

    def test_from_record_missing_digest(self):
        with pytest.raises(KeyError, match="digest"):
            LexCodeHash.from_record({"algorithm": "sha256"})

    def test_from_record_missing_algorithm(self):
        with pytest.raises(KeyError, match="algorithm"):
            LexCodeHash.from_record({"digest": "abc"})

    def test_from_record_empty_dict(self):
        with pytest.raises(KeyError):
            LexCodeHash.from_record({})

    def test_from_record_ignores_unknown_keys(self):
        """Extra keys (future lexicon fields) are silently ignored."""
        d = {"algorithm": "sha256", "digest": "abc", "futureField": 42}
        h = LexCodeHash.from_record(d)
        assert h.algorithm == "sha256"
        assert h.digest == "abc"


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
        assert restored == ref

    def test_dict_inverse(self):
        ref = LexCodeReference(
            repository="repo",
            commit="a" * 40,
            path="p:f",
            branch="main",
            language="rust",
        )
        assert (
            LexCodeReference.from_record(ref.to_record()).to_record() == ref.to_record()
        )

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

    def test_dict_inverse(self):
        record = self._make_record(
            source_schema_version="1.0.0",
            target_schema_version=">=2.0.0 <3.0.0",
            description="test",
            language="python",
        )
        assert (
            LexLensRecord.from_record(record.to_record()).to_record()
            == record.to_record()
        )

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
        assert restored.created_at == ts
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
        assert restored.created_at == ts
        assert restored.code_hash == code_hash
        assert restored.proof_ref == proof
        assert restored.description == "Verified via Lean4 proof"

    def test_dict_inverse(self):
        """to_record → from_record → to_record produces identical dict."""
        ts = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        v = LexLensVerification(
            lens="at://did:plc:abc/science.alt.dataset.lens/xyz",
            lens_commit="bafyabc123",
            verification_method="automatedTest",
            created_at=ts,
            code_hash=LexCodeHash(algorithm="sha256", digest="dead" * 16),
            description="test",
        )
        assert (
            LexLensVerification.from_record(v.to_record()).to_record() == v.to_record()
        )

    def test_from_record_missing_lens_commit(self):
        with pytest.raises(KeyError, match="lensCommit"):
            LexLensVerification.from_record(
                {
                    "lens": "at://x",
                    "verificationMethod": "codeReview",
                    "createdAt": "2025-06-01T12:00:00+00:00",
                }
            )

    def test_from_record_missing_verification_method(self):
        with pytest.raises(KeyError, match="verificationMethod"):
            LexLensVerification.from_record(
                {
                    "lens": "at://x",
                    "lensCommit": "bafy",
                    "createdAt": "2025-06-01T12:00:00+00:00",
                }
            )

    def test_from_record_missing_created_at(self):
        with pytest.raises(KeyError, match="createdAt"):
            LexLensVerification.from_record(
                {
                    "lens": "at://x",
                    "lensCommit": "bafy",
                    "verificationMethod": "codeReview",
                }
            )

    def test_from_record_ignores_unknown_keys(self):
        """Future lexicon fields don't break deserialization."""
        d = {
            "lens": "at://did:plc:abc/science.alt.dataset.lens/xyz",
            "lensCommit": "bafy",
            "verificationMethod": "codeReview",
            "createdAt": "2025-06-01T12:00:00+00:00",
            "unknownFutureField": {"nested": "data"},
        }
        v = LexLensVerification.from_record(d)
        assert v.verification_method == "codeReview"


# ---------------------------------------------------------------------------
# VerificationPublisher (mock)
# ---------------------------------------------------------------------------

_MOCK_DID = "did:plc:mock000000000000"
_LENS_URI = f"at://{_MOCK_DID}/science.alt.dataset.lens/abc"
_VERIFICATION_COLLECTION = f"{LEXICON_NAMESPACE}.lensVerification"


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
            lens_uri=_LENS_URI,
            lens_commit="bafyabc",
            verification_method="codeReview",
        )
        uri_str = str(uri)
        assert uri_str.startswith(f"at://{_MOCK_DID}/")
        assert f"/{_VERIFICATION_COLLECTION}/" in uri_str

        record = client.get_record(uri_str)
        assert record["$type"] == _VERIFICATION_COLLECTION
        assert record["verificationMethod"] == "codeReview"
        assert record["lens"] == _LENS_URI

    def test_publish_with_code_hash(self, client):
        pub = VerificationPublisher(client)
        code_hash = LexCodeHash(algorithm="sha256", digest="dead" * 16)
        uri = pub.publish(
            lens_uri=_LENS_URI,
            lens_commit="bafyabc",
            verification_method="signedHash",
            code_hash=code_hash,
            description="Signed hash verification",
        )
        record = client.get_record(str(uri))
        assert record["codeHash"]["algorithm"] == "sha256"
        assert record["codeHash"]["digest"] == "dead" * 16
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
            lens_uri=_LENS_URI,
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
            lens_uri=_LENS_URI,
            lens_commit="bafyabc",
            verification_method="codeReview",
            rkey="custom-key",
        )
        assert str(uri).endswith("/custom-key")

    def test_published_record_survives_typed_roundtrip(self, client):
        """Publish via publisher, then load via get_typed — full pipeline."""
        pub = VerificationPublisher(client)
        uri = pub.publish(
            lens_uri=_LENS_URI,
            lens_commit="bafyabc",
            verification_method="formalProof",
            code_hash=LexCodeHash(algorithm="blake3", digest="cafe" * 16),
            description="Full pipeline test",
        )
        loader = VerificationLoader(client)
        typed = loader.get_typed(str(uri))
        assert typed.lens == _LENS_URI
        assert typed.verification_method == "formalProof"
        assert typed.code_hash is not None
        assert typed.code_hash.algorithm == "blake3"
        assert typed.description == "Full pipeline test"


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
        uri = self._publish(client, lens_uri=_LENS_URI)
        loader = VerificationLoader(client)
        record = loader.get(str(uri))
        assert record["verificationMethod"] == "codeReview"
        assert record["lens"] == _LENS_URI

    def test_get_typed(self, client):
        uri = self._publish(
            client,
            lens_uri=_LENS_URI,
            description="Test typed",
        )
        loader = VerificationLoader(client)
        typed = loader.get_typed(str(uri))
        assert isinstance(typed, LexLensVerification)
        assert typed.verification_method == "codeReview"
        assert typed.description == "Test typed"

    def test_get_wrong_type_raises(self, client):
        lens_uri = client.create_record(
            collection=f"{LEXICON_NAMESPACE}.lens",
            record={"$type": f"{LEXICON_NAMESPACE}.lens", "name": "test"},
            rkey="wrong-type",
        )
        loader = VerificationLoader(client)
        with pytest.raises(ValueError, match="not a lensVerification record"):
            loader.get(str(lens_uri))

    def test_list_for_lens(self, client):
        lens_a = f"at://{_MOCK_DID}/science.alt.dataset.lens/a"
        lens_b = f"at://{_MOCK_DID}/science.alt.dataset.lens/b"
        self._publish(client, lens_uri=lens_a, method="codeReview")
        self._publish(client, lens_uri=lens_a, method="automatedTest")
        self._publish(client, lens_uri=lens_b, method="codeReview")

        loader = VerificationLoader(client)
        results = loader.list_for_lens(lens_a)
        assert len(results) == 2
        methods = {r["verificationMethod"] for r in results}
        assert methods == {"codeReview", "automatedTest"}

    def test_list_for_lens_does_not_return_plain_lens_records(self, client):
        """Regression: list_for_lens must not return lens records (substring collision)."""
        lens_a = f"at://{_MOCK_DID}/science.alt.dataset.lens/a"
        self._publish(client, lens_uri=lens_a, method="codeReview")
        # Also create a plain lens record whose lens URI happens to match
        client.create_record(
            collection=f"{LEXICON_NAMESPACE}.lens",
            record={
                "$type": f"{LEXICON_NAMESPACE}.lens",
                "name": "decoy",
                "lens": lens_a,  # same key name, wrong collection
            },
        )
        loader = VerificationLoader(client)
        results = loader.list_for_lens(lens_a)
        # Should only get the verification record, not the lens record
        assert len(results) == 1
        assert results[0]["verificationMethod"] == "codeReview"

    def test_find_by_method(self, client):
        lens_a = f"at://{_MOCK_DID}/science.alt.dataset.lens/a"
        self._publish(client, lens_uri=lens_a, method="codeReview")
        self._publish(client, lens_uri=lens_a, method="automatedTest")
        self._publish(client, lens_uri=lens_a, method="codeReview")

        loader = VerificationLoader(client)
        results = loader.find_by_method("codeReview")
        assert len(results) == 2

    def test_list_for_lens_empty(self, client):
        loader = VerificationLoader(client)
        results = loader.list_for_lens(
            f"at://{_MOCK_DID}/science.alt.dataset.lens/nonexistent"
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

    def test_language_propagated_to_code_references(self, client):
        """language= should propagate to both getter and putter code refs."""
        from atdata.atmosphere.lens import LensPublisher, LensLoader

        pub = LensPublisher(client)
        uri = pub.publish(
            name="lang-lens",
            source_schema_uri="at://did:plc:abc/science.alt.dataset.schema/src",
            target_schema_uri="at://did:plc:abc/science.alt.dataset.schema/tgt",
            code_repository="https://github.com/user/repo",
            code_commit="c" * 40,
            getter_path="mod:get",
            putter_path="mod:put",
            language="python",
        )

        record = client.get_record(str(uri))
        # Top-level deprecated field
        assert record["language"] == "python"
        # Per-reference language
        assert record["getterCode"]["language"] == "python"
        assert record["putterCode"]["language"] == "python"

        # Round-trip through typed loader
        loader = LensLoader(client)
        typed = loader.get_typed(str(uri))
        assert typed.getter_code.language == "python"
        assert typed.putter_code.language == "python"

    def test_no_language_omits_field_from_code_references(self, client):
        """When language is None, code references should omit the field."""
        from atdata.atmosphere.lens import LensPublisher

        pub = LensPublisher(client)
        uri = pub.publish(
            name="no-lang-lens",
            source_schema_uri="at://did:plc:abc/science.alt.dataset.schema/src",
            target_schema_uri="at://did:plc:abc/science.alt.dataset.schema/tgt",
            code_repository="https://github.com/user/repo",
            code_commit="d" * 40,
            getter_path="mod:get",
            putter_path="mod:put",
        )

        record = client.get_record(str(uri))
        assert "language" not in record.get("getterCode", {})
        assert "language" not in record.get("putterCode", {})
