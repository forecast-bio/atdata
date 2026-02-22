"""Tests for schematized dataset content metadata (GH #38).

Covers:
- LexDatasetEntry with metadataSchemaRef and contentMetadata
- write_samples() with content_metadata parameter (Packable and dict)
- Dataset.content_metadata property
- DatasetPublisher content metadata pass-through
- Round-trip serialization of content metadata fields
"""

from unittest.mock import Mock

import pytest

import atdata
from atdata.atmosphere._lexicon_types import (
    LEXICON_NAMESPACE,
    DatasetMetadata,
    LexDatasetEntry,
    StorageHttp,
)
from atdata.atmosphere.records import DatasetPublisher, _packable_to_dict


# ---------------------------------------------------------------------------
# Test metadata types
# ---------------------------------------------------------------------------


@atdata.packable
class InstrumentMetadata:
    """Typed content metadata for testing."""

    instrument: str
    acquisition_date: str
    operator: str


@atdata.packable
class CalibrationMetadata:
    """Content metadata with nested dict field."""

    instrument: str
    settings: dict


# ---------------------------------------------------------------------------
# LexDatasetEntry: metadataSchemaRef and contentMetadata
# ---------------------------------------------------------------------------


class TestLexDatasetEntryContentMetadata:
    """Tests for the new metadataSchemaRef and contentMetadata fields."""

    def _make_record(self, **kwargs):
        defaults = dict(
            name="TestDS",
            schema_ref="at://did:plc:abc/science.alt.dataset.schema/xyz",
            storage=StorageHttp(shards=[]),
        )
        defaults.update(kwargs)
        return LexDatasetEntry(**defaults)

    def test_to_record_omits_none_content_metadata(self):
        rec = self._make_record()
        d = rec.to_record()
        assert "metadataSchemaRef" not in d
        assert "contentMetadata" not in d

    def test_to_record_includes_metadata_schema_ref(self):
        rec = self._make_record(
            metadata_schema_ref="at://did:plc:abc/science.alt.dataset.schema/meta1"
        )
        d = rec.to_record()
        assert (
            d["metadataSchemaRef"]
            == "at://did:plc:abc/science.alt.dataset.schema/meta1"
        )

    def test_to_record_includes_content_metadata(self):
        rec = self._make_record(
            content_metadata={"instrument": "Zeiss LSM 880", "gain": 1.2}
        )
        d = rec.to_record()
        assert d["contentMetadata"] == {"instrument": "Zeiss LSM 880", "gain": 1.2}

    def test_to_record_includes_both_fields(self):
        rec = self._make_record(
            metadata_schema_ref="at://did:plc:abc/collection/meta1",
            content_metadata={"instrument": "Zeiss"},
        )
        d = rec.to_record()
        assert d["metadataSchemaRef"] == "at://did:plc:abc/collection/meta1"
        assert d["contentMetadata"]["instrument"] == "Zeiss"

    def test_roundtrip_with_content_metadata(self):
        original = self._make_record(
            metadata_schema_ref="at://did:plc:abc/science.alt.dataset.schema/meta1",
            content_metadata={
                "instrument": "Nikon A1R",
                "acquisition_date": "2025-06-15",
                "settings": {"gain": 1.5, "offset": 0.1},
            },
        )
        d = original.to_record()
        restored = LexDatasetEntry.from_record(d)
        assert restored.metadata_schema_ref == original.metadata_schema_ref
        assert restored.content_metadata == original.content_metadata

    def test_roundtrip_without_content_metadata(self):
        """Backward compat: old records without content metadata still parse."""
        d = {
            "$type": f"{LEXICON_NAMESPACE}.entry",
            "name": "OldDataset",
            "schemaRef": "at://did:plc:abc/collection/key",
            "storage": {"$type": f"{LEXICON_NAMESPACE}.storageHttp", "shards": []},
            "createdAt": "2025-01-01T00:00:00+00:00",
        }
        restored = LexDatasetEntry.from_record(d)
        assert restored.metadata_schema_ref is None
        assert restored.content_metadata is None

    def test_content_metadata_coexists_with_record_metadata(self):
        """Both metadata (record-level) and contentMetadata can exist."""
        rec = self._make_record(
            metadata=DatasetMetadata(split="train"),
            metadata_schema_ref="at://did:plc:abc/collection/meta1",
            content_metadata={"instrument": "Zeiss"},
        )
        d = rec.to_record()
        restored = LexDatasetEntry.from_record(d)
        assert restored.metadata is not None
        assert restored.metadata.split == "train"
        assert restored.content_metadata == {"instrument": "Zeiss"}
        assert restored.metadata_schema_ref is not None


# ---------------------------------------------------------------------------
# _packable_to_dict helper
# ---------------------------------------------------------------------------


class TestPackableToDict:
    def test_converts_packable_to_dict(self):
        meta = InstrumentMetadata(
            instrument="Zeiss LSM 880",
            acquisition_date="2025-06-15",
            operator="Alice",
        )
        d = _packable_to_dict(meta)
        assert d == {
            "instrument": "Zeiss LSM 880",
            "acquisition_date": "2025-06-15",
            "operator": "Alice",
        }

    def test_converts_nested_dict_field(self):
        meta = CalibrationMetadata(
            instrument="Nikon", settings={"gain": 1.5, "offset": 0.1}
        )
        d = _packable_to_dict(meta)
        assert d["settings"] == {"gain": 1.5, "offset": 0.1}

    def test_rejects_non_dataclass(self):
        with pytest.raises(TypeError, match="Cannot convert"):
            _packable_to_dict("not a dataclass")


# ---------------------------------------------------------------------------
# write_samples with content_metadata
# ---------------------------------------------------------------------------


class TestWriteSamplesContentMetadata:
    def test_dict_metadata_attached(self, tmp_path):
        @atdata.packable
        class SimpleSample:
            text: str

        samples = [SimpleSample(text="hello"), SimpleSample(text="world")]
        ds = atdata.write_samples(
            samples,
            str(tmp_path / "data.tar"),
            content_metadata={"instrument": "Zeiss"},
        )
        assert ds.content_metadata == {"instrument": "Zeiss"}

    def test_packable_metadata_attached(self, tmp_path):
        @atdata.packable
        class SimpleSample:
            text: str

        meta = InstrumentMetadata(
            instrument="Nikon",
            acquisition_date="2025-01-15",
            operator="Bob",
        )
        samples = [SimpleSample(text="hello")]
        ds = atdata.write_samples(
            samples,
            str(tmp_path / "data.tar"),
            content_metadata=meta,
        )
        assert isinstance(ds.content_metadata, InstrumentMetadata)
        assert ds.content_metadata.instrument == "Nikon"

    def test_none_metadata_default(self, tmp_path):
        @atdata.packable
        class SimpleSample:
            text: str

        ds = atdata.write_samples(
            [SimpleSample(text="hello")],
            str(tmp_path / "data.tar"),
        )
        assert ds.content_metadata is None

    def test_invalid_metadata_type_raises(self, tmp_path):
        @atdata.packable
        class SimpleSample:
            text: str

        with pytest.raises(TypeError, match="content_metadata must be"):
            atdata.write_samples(
                [SimpleSample(text="hello")],
                str(tmp_path / "data.tar"),
                content_metadata=42,
            )


# ---------------------------------------------------------------------------
# Dataset.content_metadata property
# ---------------------------------------------------------------------------


class TestDatasetContentMetadata:
    def test_default_is_none(self, tmp_path):
        @atdata.packable
        class SimpleSample:
            text: str

        ds = atdata.write_samples([SimpleSample(text="a")], str(tmp_path / "data.tar"))
        assert ds.content_metadata is None

    def test_setter_accepts_dict(self, tmp_path):
        @atdata.packable
        class SimpleSample:
            text: str

        ds = atdata.write_samples([SimpleSample(text="a")], str(tmp_path / "data.tar"))
        ds.content_metadata = {"key": "value"}
        assert ds.content_metadata == {"key": "value"}

    def test_setter_accepts_packable(self, tmp_path):
        @atdata.packable
        class SimpleSample:
            text: str

        ds = atdata.write_samples([SimpleSample(text="a")], str(tmp_path / "data.tar"))
        meta = InstrumentMetadata(
            instrument="Zeiss", acquisition_date="2025-01-01", operator="Eve"
        )
        ds.content_metadata = meta
        assert ds.content_metadata.instrument == "Zeiss"

    def test_setter_accepts_none(self, tmp_path):
        @atdata.packable
        class SimpleSample:
            text: str

        ds = atdata.write_samples(
            [SimpleSample(text="a")],
            str(tmp_path / "data.tar"),
            content_metadata={"x": 1},
        )
        ds.content_metadata = None
        assert ds.content_metadata is None


# ---------------------------------------------------------------------------
# DatasetPublisher content metadata pass-through
# ---------------------------------------------------------------------------


class TestDatasetPublisherContentMetadata:
    def test_create_record_with_content_metadata(self):
        """_create_record passes content metadata to LexDatasetEntry."""
        mock_client = Mock()
        mock_client.create_record.return_value = Mock(
            uri="at://did:plc:abc/science.alt.dataset.entry/xyz"
        )
        publisher = DatasetPublisher(mock_client)

        publisher._create_record(
            StorageHttp(shards=[]),
            name="TestDS",
            schema_uri="at://did:plc:abc/collection/schema1",
            metadata_schema_ref="at://did:plc:abc/collection/meta_schema1",
            content_metadata={"instrument": "Zeiss"},
        )

        call_args = mock_client.create_record.call_args
        record_dict = call_args.kwargs.get("record") or call_args[1].get("record")
        assert (
            record_dict["metadataSchemaRef"]
            == "at://did:plc:abc/collection/meta_schema1"
        )
        assert record_dict["contentMetadata"] == {"instrument": "Zeiss"}

    def test_create_record_without_content_metadata(self):
        """_create_record omits content metadata fields when None."""
        mock_client = Mock()
        mock_client.create_record.return_value = Mock(
            uri="at://did:plc:abc/science.alt.dataset.entry/xyz"
        )
        publisher = DatasetPublisher(mock_client)

        publisher._create_record(
            StorageHttp(shards=[]),
            name="TestDS",
            schema_uri="at://did:plc:abc/collection/schema1",
        )

        call_args = mock_client.create_record.call_args
        record_dict = call_args.kwargs.get("record") or call_args[1].get("record")
        assert "metadataSchemaRef" not in record_dict
        assert "contentMetadata" not in record_dict
