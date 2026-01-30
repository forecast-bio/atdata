"""Tests for the Repository dataclass and unified Index repository system.

Covers:
- Repository construction and create_repository factory
- Index with multiple named repositories
- Prefix routing via _resolve_prefix
- Cross-repository dataset operations
- Default Index singleton (get_default_index / set_default_index)
- AtmosphereIndex deprecation warning
"""

from dataclasses import dataclass
from pathlib import Path

import pytest
import webdataset as wds

import atdata
from atdata.repository import Repository, create_repository
from atdata.providers._sqlite import SqliteProvider
from atdata.local import Index


# ---------------------------------------------------------------------------
# Sample types
# ---------------------------------------------------------------------------


@dataclass
class RepoTestSample(atdata.PackableSample):
    """Sample type for repository tests."""

    text: str
    value: int


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sqlite_provider(tmp_path: Path) -> SqliteProvider:
    """Fresh SQLite provider for local repository."""
    return SqliteProvider(path=tmp_path / "local.db")


@pytest.fixture
def extra_provider(tmp_path: Path) -> SqliteProvider:
    """Fresh SQLite provider for a named repository."""
    return SqliteProvider(path=tmp_path / "lab.db")


@pytest.fixture
def tar_file(tmp_path: Path) -> Path:
    """Create a test tar file with one sample."""
    tar_path = tmp_path / "data-000000.tar"
    sample = RepoTestSample(text="hello", value=42)
    with wds.writer.TarWriter(str(tar_path)) as writer:
        writer.write(sample.as_wds)
    return tar_path


# ---------------------------------------------------------------------------
# Repository dataclass
# ---------------------------------------------------------------------------


class TestRepository:
    """Tests for the Repository dataclass."""

    def test_basic_construction(self, sqlite_provider):
        repo = Repository(provider=sqlite_provider)
        assert repo.provider is sqlite_provider
        assert repo.data_store is None

    def test_construction_with_data_store(self, sqlite_provider):
        # Use None as data_store placeholder (real S3DataStore needs creds)
        repo = Repository(provider=sqlite_provider, data_store=None)
        assert repo.data_store is None


class TestCreateRepository:
    """Tests for the create_repository factory."""

    def test_sqlite(self, tmp_path):
        repo = create_repository("sqlite", path=tmp_path / "test.db")
        assert isinstance(repo, Repository)
        assert isinstance(repo.provider, SqliteProvider)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            create_repository("unknown_backend")


# ---------------------------------------------------------------------------
# Index with repositories
# ---------------------------------------------------------------------------


class TestIndexRepos:
    """Tests for Index with named repositories."""

    def test_default_index_has_no_extra_repos(self, sqlite_provider):
        index = Index(provider=sqlite_provider, atmosphere=None)
        assert index.repos == {}

    def test_index_with_named_repos(self, sqlite_provider, extra_provider):
        lab_repo = Repository(provider=extra_provider)
        index = Index(
            provider=sqlite_provider,
            repos={"lab": lab_repo},
            atmosphere=None,
        )
        assert "lab" in index.repos
        assert index.repos["lab"].provider is extra_provider

    def test_reserved_local_name_raises(self, sqlite_provider, extra_provider):
        with pytest.raises(ValueError, match='"local" is reserved'):
            Index(
                provider=sqlite_provider,
                repos={"local": Repository(provider=extra_provider)},
                atmosphere=None,
            )

    def test_invalid_repo_type_raises(self, sqlite_provider):
        with pytest.raises(TypeError, match="must be a Repository"):
            Index(
                provider=sqlite_provider,
                repos={"bad": "not a repo"},  # type: ignore[dict-item]
                atmosphere=None,
            )

    def test_atmosphere_none_disables(self, sqlite_provider):
        index = Index(provider=sqlite_provider, atmosphere=None)
        assert index.atmosphere is None


# ---------------------------------------------------------------------------
# Prefix routing
# ---------------------------------------------------------------------------


class TestResolvePrefix:
    """Tests for Index._resolve_prefix routing."""

    @pytest.fixture
    def index(self, sqlite_provider, extra_provider):
        return Index(
            provider=sqlite_provider,
            repos={"lab": Repository(provider=extra_provider)},
            atmosphere=None,
        )

    def test_bare_name_routes_to_local(self, index):
        key, ref, handle = index._resolve_prefix("mnist")
        assert key == "local"
        assert ref == "mnist"
        assert handle is None

    def test_local_prefix(self, index):
        key, ref, handle = index._resolve_prefix("local/mnist")
        assert key == "local"
        assert ref == "mnist"

    def test_named_repo_prefix(self, index):
        key, ref, handle = index._resolve_prefix("lab/mnist")
        assert key == "lab"
        assert ref == "mnist"

    def test_at_prefix_routes_to_atmosphere(self, index):
        key, ref, handle = index._resolve_prefix("@maxine.science/mnist")
        assert key == "_atmosphere"
        assert ref == "mnist"
        assert handle == "maxine.science"

    def test_at_uri_routes_to_atmosphere(self, index):
        key, ref, handle = index._resolve_prefix("at://did:plc:abc/collection/rkey")
        assert key == "_atmosphere"
        assert ref == "at://did:plc:abc/collection/rkey"

    def test_atdata_uri_local(self, index):
        key, ref, handle = index._resolve_prefix("atdata://local/record/mnist")
        assert key == "local"
        assert ref == "mnist"

    def test_atdata_uri_named_repo(self, index):
        key, ref, handle = index._resolve_prefix("atdata://lab/record/mnist")
        assert key == "lab"
        assert ref == "mnist"

    def test_unknown_prefix_treated_as_local(self, index):
        """A prefix/ path where prefix is NOT a known repo is treated as local."""
        key, ref, handle = index._resolve_prefix("unknownrepo/dataset")
        # unknownrepo is not a registered repo, so it falls through to local
        assert key == "local"
        assert ref == "unknownrepo/dataset"

    def test_at_handle_no_slash(self, index):
        """@handle with no slash returns the handle as ref, no handle_or_did."""
        key, ref, handle = index._resolve_prefix("@maxine.science")
        assert key == "_atmosphere"
        assert ref == "maxine.science"
        assert handle is None

    def test_atdata_uri_unknown_prefix_routes_to_atmosphere(self, index):
        """atdata:// with unrecognized prefix routes to atmosphere."""
        key, ref, handle = index._resolve_prefix("atdata://unknownrepo/record/mnist")
        assert key == "_atmosphere"
        assert ref == "mnist"
        assert handle == "unknownrepo"


# ---------------------------------------------------------------------------
# Cross-repository dataset operations
# ---------------------------------------------------------------------------


class TestCrossRepoOperations:
    """Tests for dataset operations across multiple repositories."""

    def test_insert_and_get_local(self, sqlite_provider, tar_file):
        index = Index(provider=sqlite_provider, atmosphere=None)
        ds = atdata.Dataset[RepoTestSample](str(tar_file))

        index.insert_dataset(ds, name="mnist")
        retrieved = index.get_dataset("mnist")
        assert retrieved.name == "mnist"

    def test_insert_and_get_named_repo(self, sqlite_provider, extra_provider, tar_file):
        index = Index(
            provider=sqlite_provider,
            repos={"lab": Repository(provider=extra_provider)},
            atmosphere=None,
        )
        ds = atdata.Dataset[RepoTestSample](str(tar_file))

        # Insert into lab repo
        entry = index.insert_dataset(ds, name="lab/mnist")
        assert entry.name == "mnist"

        # Retrieve from lab repo
        retrieved = index.get_dataset("lab/mnist")
        assert retrieved.name == "mnist"

        # Should NOT be in local
        with pytest.raises(KeyError):
            index.get_dataset("mnist")

    def test_insert_local_explicit_prefix(
        self, sqlite_provider, extra_provider, tar_file
    ):
        index = Index(
            provider=sqlite_provider,
            repos={"lab": Repository(provider=extra_provider)},
            atmosphere=None,
        )
        ds = atdata.Dataset[RepoTestSample](str(tar_file))

        entry = index.insert_dataset(ds, name="local/mnist")
        assert entry.name == "mnist"

        retrieved = index.get_dataset("local/mnist")
        assert retrieved.name == "mnist"

    def test_list_datasets_aggregates(self, sqlite_provider, extra_provider, tar_file):
        index = Index(
            provider=sqlite_provider,
            repos={"lab": Repository(provider=extra_provider)},
            atmosphere=None,
        )
        ds = atdata.Dataset[RepoTestSample](str(tar_file))

        index.insert_dataset(ds, name="local_ds")
        index.insert_dataset(ds, name="lab/lab_ds")

        # Aggregated: both repos
        all_entries = index.list_datasets()
        names = [e.name for e in all_entries]
        assert "local_ds" in names
        assert "lab_ds" in names

    def test_list_datasets_filtered(self, sqlite_provider, extra_provider, tar_file):
        index = Index(
            provider=sqlite_provider,
            repos={"lab": Repository(provider=extra_provider)},
            atmosphere=None,
        )
        ds = atdata.Dataset[RepoTestSample](str(tar_file))

        index.insert_dataset(ds, name="local_ds")
        index.insert_dataset(ds, name="lab/lab_ds")

        local_only = index.list_datasets(repo="local")
        assert len(local_only) == 1
        assert local_only[0].name == "local_ds"

        lab_only = index.list_datasets(repo="lab")
        assert len(lab_only) == 1
        assert lab_only[0].name == "lab_ds"

    def test_get_dataset_atmosphere_disabled_raises(self, sqlite_provider):
        index = Index(provider=sqlite_provider, atmosphere=None)
        with pytest.raises(ValueError, match="Atmosphere backend required"):
            index.get_dataset("@handle/dataset")

    def test_insert_dataset_atmosphere_disabled_raises(self, sqlite_provider, tar_file):
        index = Index(provider=sqlite_provider, atmosphere=None)
        ds = atdata.Dataset[RepoTestSample](str(tar_file))
        with pytest.raises(ValueError, match="Atmosphere backend required"):
            index.insert_dataset(ds, name="@handle/dataset")

    def test_list_datasets_unknown_repo_raises(self, sqlite_provider):
        index = Index(provider=sqlite_provider, atmosphere=None)
        with pytest.raises(KeyError, match="Unknown repository"):
            index.list_datasets(repo="nonexistent")

    def test_list_datasets_atmosphere_disabled_returns_empty(self, sqlite_provider):
        index = Index(provider=sqlite_provider, atmosphere=None)
        result = index.list_datasets(repo="_atmosphere")
        assert result == []


# ---------------------------------------------------------------------------
# Default Index singleton
# ---------------------------------------------------------------------------


class TestDefaultIndex:
    """Tests for get_default_index / set_default_index."""

    def test_set_and_get(self, sqlite_provider):
        import atdata._hf_api as hf

        custom = Index(provider=sqlite_provider, atmosphere=None)
        original = hf._default_index

        try:
            atdata.set_default_index(custom)
            assert atdata.get_default_index() is custom
        finally:
            # Restore original state
            hf._default_index = original

    def test_get_creates_default_lazily(self):
        import atdata._hf_api as hf

        original = hf._default_index
        try:
            hf._default_index = None
            idx = atdata.get_default_index()
            assert isinstance(idx, Index)
        finally:
            hf._default_index = original


# ---------------------------------------------------------------------------
# AtmosphereIndex deprecation
# ---------------------------------------------------------------------------


class TestAtmosphereIndexDeprecation:
    """Tests that AtmosphereIndex emits a deprecation warning."""

    @pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
    def test_deprecation_warning(self):
        from unittest.mock import MagicMock

        from atdata.atmosphere import AtmosphereIndex, AtmosphereClient

        # Create a mock client to avoid network calls
        mock_client = MagicMock(spec=AtmosphereClient)
        mock_client.is_authenticated = True
        mock_client.did = "did:plc:test"

        with pytest.warns(DeprecationWarning, match="AtmosphereIndex is deprecated"):
            AtmosphereIndex(mock_client)
