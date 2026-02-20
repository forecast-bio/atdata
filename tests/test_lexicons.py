"""Tests for atdata.lexicons module — lexicon loading and listing."""

import json
from pathlib import Path

import pytest

from atdata.lexicons import (
    LEXICON_IDS,
    NAMESPACE,
    list_lexicons,
    load_lexicon,
    load_ndarray_shim,
)

# Lexicon JSON files now live under the NSID-to-path structure inside the
# package directory (src/atdata/lexicons/science/alt/dataset/) and also in
# the top-level vendor directory (lexicons/science/alt/dataset/).
REPO_ROOT = Path(__file__).resolve().parent.parent
LEXICON_DIR = REPO_ROOT / "lexicons" / "science" / "alt" / "dataset"
_HAS_LEXICON_FILES = (LEXICON_DIR / "schema.json").exists()


def _lexicon_path(lexicon_id: str) -> Path:
    """Convert an NSID to the NSID-to-path file location."""
    parts = lexicon_id.split(".")
    return REPO_ROOT / "lexicons" / Path(*parts[:-1]) / f"{parts[-1]}.json"


class TestNamespace:
    """Verify module constants."""

    def test_namespace_value(self):
        assert NAMESPACE == "science.alt.dataset"

    def test_lexicon_ids_are_namespaced(self):
        for lid in LEXICON_IDS:
            assert lid.startswith(NAMESPACE), f"{lid} missing namespace prefix"

    def test_lexicon_ids_count(self):
        assert len(LEXICON_IDS) >= 10


@pytest.mark.skipif(not _HAS_LEXICON_FILES, reason="lexicon JSON files not found")
class TestLexiconFilesExist:
    """Validate that all declared lexicon IDs have corresponding JSON files."""

    def test_all_lexicon_ids_have_files(self):
        for lid in LEXICON_IDS:
            path = _lexicon_path(lid)
            assert path.exists(), f"Missing lexicon file: {path}"

    def test_lexicon_files_are_valid_json(self):
        for lid in LEXICON_IDS:
            path = _lexicon_path(lid)
            data = json.loads(path.read_text(encoding="utf-8"))
            assert data["id"] == lid

    def test_all_lexicons_declare_version_1(self):
        for lid in LEXICON_IDS:
            path = _lexicon_path(lid)
            data = json.loads(path.read_text(encoding="utf-8"))
            assert data.get("lexicon") == 1, f"{lid} missing lexicon: 1"

    def test_lexicon_roundtrips_through_json(self):
        path = _lexicon_path("science.alt.dataset.schema")
        data = json.loads(path.read_text(encoding="utf-8"))
        roundtripped = json.loads(json.dumps(data))
        assert roundtripped == data

    def test_ndarray_shim_exists(self):
        path = REPO_ROOT / "lexicons" / "ndarray_shim.json"
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "$defs" in data
        assert data["$defs"]["ndarray"]["type"] == "string"


class TestLoadLexicon:
    """Tests for load_lexicon() function."""

    def test_nonexistent_raises_file_not_found(self):
        load_lexicon.cache_clear()
        with pytest.raises(FileNotFoundError, match="No lexicon file found"):
            load_lexicon("science.alt.dataset.nonexistent")

    def test_caching_returns_same_object(self):
        """Repeated calls with same ID return identical object (lru_cache)."""
        load_lexicon.cache_clear()
        a = load_lexicon("science.alt.dataset.schema")
        b = load_lexicon("science.alt.dataset.schema")
        assert a is b

    def test_load_all_lexicons(self):
        """Every declared lexicon ID can be loaded via load_lexicon."""
        load_lexicon.cache_clear()
        for lid in LEXICON_IDS:
            data = load_lexicon(lid)
            assert data["id"] == lid

    def test_label_and_resolve_label_present(self):
        """Verify new label and resolveLabel lexicons are loadable."""
        load_lexicon.cache_clear()
        label = load_lexicon("science.alt.dataset.label")
        assert label["defs"]["main"]["type"] == "record"
        resolve = load_lexicon("science.alt.dataset.resolveLabel")
        assert resolve["defs"]["main"]["type"] == "query"


class TestLoadNdarrayShim:
    """Tests for load_ndarray_shim()."""

    def test_returns_dict(self):
        load_ndarray_shim.cache_clear()
        shim = load_ndarray_shim()
        assert isinstance(shim, dict)
        assert "$defs" in shim

    def test_ndarray_shim_id(self):
        load_ndarray_shim.cache_clear()
        shim = load_ndarray_shim()
        assert shim["$id"] == "https://alt.science/schemas/atdata-ndarray-bytes/1.0.0"


class TestListLexicons:
    """Tests for list_lexicons()."""

    def test_returns_tuple(self):
        result = list_lexicons()
        assert isinstance(result, tuple)

    def test_contains_known_ids(self):
        result = list_lexicons()
        assert "science.alt.dataset.schema" in result
        assert "science.alt.dataset.entry" in result
        assert "science.alt.dataset.lens" in result
        assert "science.alt.dataset.label" in result
        assert "science.alt.dataset.resolveLabel" in result

    def test_matches_constant(self):
        assert list_lexicons() is LEXICON_IDS
