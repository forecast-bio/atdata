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

# The JSON files live in the repo root `lexicons/` directory and are only
# available via importlib.resources when installed from the built wheel
# (via pyproject.toml force-include). In editable installs, we fall back
# to loading directly from the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
LEXICON_DIR = REPO_ROOT / "lexicons"
_HAS_LEXICON_FILES = (LEXICON_DIR / "ac.foundation.dataset.schema.json").exists()


class TestNamespace:
    """Verify module constants."""

    def test_namespace_value(self):
        assert NAMESPACE == "ac.foundation.dataset"

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
            path = LEXICON_DIR / f"{lid}.json"
            assert path.exists(), f"Missing lexicon file: {path}"

    def test_lexicon_files_are_valid_json(self):
        for lid in LEXICON_IDS:
            path = LEXICON_DIR / f"{lid}.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            assert data["id"] == lid

    def test_all_lexicons_declare_version_1(self):
        for lid in LEXICON_IDS:
            path = LEXICON_DIR / f"{lid}.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            assert data.get("lexicon") == 1, f"{lid} missing lexicon: 1"

    def test_lexicon_roundtrips_through_json(self):
        path = LEXICON_DIR / "ac.foundation.dataset.schema.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        roundtripped = json.loads(json.dumps(data))
        assert roundtripped == data

    def test_ndarray_shim_exists(self):
        path = LEXICON_DIR / "ndarray_shim.json"
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "$defs" in data
        assert data["$defs"]["ndarray"]["type"] == "string"


class TestLoadLexicon:
    """Tests for load_lexicon() function."""

    def test_nonexistent_raises_file_not_found(self):
        load_lexicon.cache_clear()
        with pytest.raises(FileNotFoundError, match="No lexicon file found"):
            load_lexicon("ac.foundation.dataset.nonexistent")

    def test_caching_returns_same_object(self):
        """Repeated calls with same ID return identical object (lru_cache)."""
        load_lexicon.cache_clear()
        # This will raise if files aren't installed, but the cache behavior
        # is testable even with the error path.
        try:
            a = load_lexicon("ac.foundation.dataset.schema")
            b = load_lexicon("ac.foundation.dataset.schema")
            assert a is b
        except FileNotFoundError:
            pytest.skip("lexicon files not installed in dev mode")


class TestLoadNdarrayShim:
    """Tests for load_ndarray_shim()."""

    def test_returns_dict_or_raises(self):
        load_ndarray_shim.cache_clear()
        try:
            shim = load_ndarray_shim()
            assert isinstance(shim, dict)
            assert "$defs" in shim
        except FileNotFoundError:
            pytest.skip("ndarray_shim.json not installed in dev mode")


class TestListLexicons:
    """Tests for list_lexicons()."""

    def test_returns_tuple(self):
        result = list_lexicons()
        assert isinstance(result, tuple)

    def test_contains_known_ids(self):
        result = list_lexicons()
        assert "ac.foundation.dataset.schema" in result
        assert "ac.foundation.dataset.record" in result
        assert "ac.foundation.dataset.lens" in result

    def test_matches_constant(self):
        assert list_lexicons() is LEXICON_IDS
