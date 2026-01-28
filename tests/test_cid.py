"""Tests for CID generation utilities."""

import pytest
import libipld

from atdata._cid import (
    generate_cid,
    generate_cid_from_bytes,
    verify_cid,
    parse_cid,
)


class TestGenerateCid:
    """Tests for generate_cid function."""

    def test_generates_valid_cid_from_dict(self):
        """CID is generated from a dictionary."""
        data = {"name": "TestSample", "version": "1.0.0"}
        cid = generate_cid(data)

        # CIDv1 base32 starts with 'bafy'
        assert cid.startswith("bafy")
        assert len(cid) > 40  # CIDs are typically 59 chars

    def test_deterministic_output(self):
        """Same data always produces same CID."""
        data = {"name": "TestSample", "version": "1.0.0", "fields": []}

        cid1 = generate_cid(data)
        cid2 = generate_cid(data)

        assert cid1 == cid2

    def test_different_data_different_cid(self):
        """Different data produces different CIDs."""
        data1 = {"name": "Sample1", "version": "1.0.0"}
        data2 = {"name": "Sample2", "version": "1.0.0"}

        cid1 = generate_cid(data1)
        cid2 = generate_cid(data2)

        assert cid1 != cid2

    def test_key_order_matters_in_dag_cbor(self):
        """DAG-CBOR has deterministic key ordering, so key order in input doesn't matter."""
        # DAG-CBOR sorts keys, so these should produce the same CID
        data1 = {"a": 1, "b": 2}
        data2 = {"b": 2, "a": 1}

        cid1 = generate_cid(data1)
        cid2 = generate_cid(data2)

        # DAG-CBOR canonicalizes key order
        assert cid1 == cid2

    def test_handles_nested_structures(self):
        """CID can be generated from nested data structures."""
        data = {
            "name": "NestedSample",
            "fields": [
                {"name": "field1", "type": "str"},
                {"name": "field2", "type": "int"},
            ],
            "metadata": {"author": "test", "tags": ["a", "b", "c"]},
        }

        cid = generate_cid(data)
        assert cid.startswith("bafy")

    def test_handles_various_types(self):
        """CID handles various Python types."""
        data = {
            "string": "hello",
            "integer": 42,
            "float": 3.14,
            "boolean": True,
            "null": None,
            "bytes": b"binary data",
            "list": [1, 2, 3],
        }

        cid = generate_cid(data)
        assert cid.startswith("bafy")

    def test_invalid_data_raises_error(self):
        """Non-CBOR-serializable data raises ValueError."""
        # Functions can't be serialized to CBOR
        data = {"func": lambda x: x}

        with pytest.raises(ValueError, match="Failed to encode"):
            generate_cid(data)


class TestGenerateCidFromBytes:
    """Tests for generate_cid_from_bytes function."""

    def test_generates_cid_from_bytes(self):
        """CID is generated from raw bytes."""
        data_bytes = b"some raw bytes"
        cid = generate_cid_from_bytes(data_bytes)

        assert cid.startswith("bafy")

    def test_matches_manual_encoding(self):
        """CID from bytes matches CID from pre-encoded data."""
        data = {"key": "value"}
        cbor_bytes = libipld.encode_dag_cbor(data)

        cid_from_data = generate_cid(data)
        cid_from_bytes = generate_cid_from_bytes(cbor_bytes)

        assert cid_from_data == cid_from_bytes


class TestVerifyCid:
    """Tests for verify_cid function."""

    def test_verify_matching_data(self):
        """verify_cid returns True for matching data."""
        data = {"name": "test", "value": 123}
        cid = generate_cid(data)

        assert verify_cid(cid, data) is True

    def test_verify_non_matching_data(self):
        """verify_cid returns False for non-matching data."""
        data = {"name": "test", "value": 123}
        cid = generate_cid(data)

        different_data = {"name": "test", "value": 456}
        assert verify_cid(cid, different_data) is False

    def test_verify_with_complex_data(self):
        """verify_cid works with complex nested structures."""
        data = {
            "schema": {
                "name": "ImageSample",
                "version": "1.0.0",
                "fields": [
                    {"name": "image", "type": "ndarray"},
                    {"name": "label", "type": "str"},
                ],
            }
        }
        cid = generate_cid(data)

        assert verify_cid(cid, data) is True


class TestParseCid:
    """Tests for parse_cid function."""

    def test_parse_cid_components(self):
        """parse_cid extracts CID components."""
        data = {"test": "data"}
        cid = generate_cid(data)

        parsed = parse_cid(cid)

        assert parsed["version"] == 1
        assert parsed["codec"] == 0x71  # dag-cbor
        assert parsed["hash"]["code"] == 0x12  # sha256
        assert parsed["hash"]["size"] == 32

    def test_parse_cid_digest_matches(self):
        """Parsed digest matches the SHA-256 of the data."""
        import hashlib

        data = {"test": "data"}
        cid = generate_cid(data)

        cbor_bytes = libipld.encode_dag_cbor(data)
        expected_digest = hashlib.sha256(cbor_bytes).digest()

        parsed = parse_cid(cid)
        assert parsed["hash"]["digest"] == expected_digest

    @pytest.mark.parametrize(
        "malformed_cid",
        [
            "",  # empty
            "invalid",  # not a CID
            "bafy123",  # truncated CID
            "Qm123",  # v0 prefix but invalid
        ],
    )
    def test_parse_cid_malformed_raises_valueerror(self, malformed_cid):
        """Malformed CID strings raise ValueError."""
        with pytest.raises(ValueError, match="Failed to decode CID"):
            parse_cid(malformed_cid)


class TestAtprotoCompatibility:
    """Tests verifying ATProto SDK compatibility."""

    def test_cid_decodable_by_atproto(self):
        """Generated CIDs can be decoded by atproto SDK."""
        from atproto_core.cid.cid import CID

        data = {"name": "TestSchema", "version": "1.0.0"}
        cid_str = generate_cid(data)

        # Should not raise
        cid_obj = CID.decode(cid_str)

        assert cid_obj.version == 1
        assert cid_obj.codec == 0x71

    def test_hash_matches_atproto_decode(self):
        """Hash in generated CID matches when decoded by atproto."""
        import hashlib
        from atproto_core.cid.cid import CID

        data = {"name": "TestSchema", "version": "1.0.0"}
        cid_str = generate_cid(data)

        cbor_bytes = libipld.encode_dag_cbor(data)
        expected_hash = hashlib.sha256(cbor_bytes).digest()

        cid_obj = CID.decode(cid_str)
        assert cid_obj.hash.digest == expected_hash
