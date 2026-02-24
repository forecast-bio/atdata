"""Tests for the unified search API (search.py and Index.search).

Covers:
- SearchResult and SearchResults dataclass construction
- AppViewSearchBackend with mocked xrpc_query
- LocalSearchBackend with a real SQLite-backed Index
- SearchAggregator deduplication and sorting
- Index.search() end-to-end with both backends
- Graceful degradation when AppView is unavailable
- Edge cases: empty results, pagination, backend failures
"""

from unittest.mock import Mock

import pytest

import atdata
from atdata.search import (
    SearchResult,
    SearchResults,
    SearchBackend,
    AppViewSearchBackend,
    LocalSearchBackend,
    SearchAggregator,
    _deduplicate,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sqlite_index(tmp_path):
    """Index backed by a temp SQLite database with sample data."""
    from atdata.stores._disk import LocalDiskStore

    store = LocalDiskStore(root=tmp_path / "data")
    index = atdata.Index(
        provider="sqlite",
        path=str(tmp_path / "index.db"),
        data_store=store,
        atmosphere=None,
    )
    return index


@pytest.fixture
def populated_index(sqlite_index, tmp_path):
    """Index with several datasets already written."""

    @atdata.packable
    class SearchSample:
        name: str
        value: int

    samples_a = [SearchSample(name=f"a-{i}", value=i) for i in range(5)]
    samples_b = [SearchSample(name=f"b-{i}", value=i) for i in range(5)]
    samples_c = [SearchSample(name=f"c-{i}", value=i) for i in range(5)]

    sqlite_index.write_samples(
        samples_a,
        name="mnist-train",
        metadata={"tags": ["vision", "digits"], "description": "MNIST training set"},
    )
    sqlite_index.write_samples(
        samples_b,
        name="mnist-test",
        metadata={"tags": ["vision", "digits"], "description": "MNIST test set"},
    )
    sqlite_index.write_samples(
        samples_c,
        name="cifar-train",
        metadata={"tags": ["vision", "objects"], "description": "CIFAR training set"},
    )

    return sqlite_index


@pytest.fixture
def mock_atmosphere():
    """Mocked Atmosphere client with AppView."""
    atmo = Mock()
    atmo.has_appview = True
    atmo.xrpc_query.return_value = {
        "entries": [
            {
                "uri": "at://did:plc:abc/science.alt.dataset.entry/tid1",
                "name": "genomics-v1",
                "description": "Genomics dataset",
                "tags": ["bio", "genomics"],
                "schemaRef": "at://did:plc:abc/science.alt.dataset.schema/s1",
            },
            {
                "uri": "at://did:plc:abc/science.alt.dataset.entry/tid2",
                "name": "proteomics-v1",
                "description": "Proteomics dataset",
                "tags": ["bio", "proteomics"],
                "schemaRef": None,
            },
        ],
        "cursor": "next-page-token",
    }
    return atmo


# =============================================================================
# SearchResult / SearchResults dataclass tests
# =============================================================================


class TestSearchResult:
    def test_minimal_construction(self):
        r = SearchResult(uri="cid:abc", name="test")
        assert r.uri == "cid:abc"
        assert r.name == "test"
        assert r.description is None
        assert r.tags == []
        assert r.schema_ref is None
        assert r.source == "unknown"
        assert r.score is None
        assert r.record is None

    def test_full_construction(self):
        r = SearchResult(
            uri="at://did:plc:x/science.alt.dataset.entry/y",
            name="mnist",
            description="Hand-written digits",
            tags=["vision"],
            schema_ref="at://did:plc:x/science.alt.dataset.schema/z",
            source="atmosphere",
            score=0.95,
            record={"raw": True},
        )
        assert r.score == 0.95
        assert r.tags == ["vision"]
        assert r.record == {"raw": True}

    def test_default_tags_not_shared(self):
        """Each instance gets its own tags list."""
        a = SearchResult(uri="a", name="a")
        b = SearchResult(uri="b", name="b")
        a.tags.append("x")
        assert b.tags == []


class TestSearchResults:
    def test_empty(self):
        r = SearchResults()
        assert r.items == []
        assert r.cursor is None
        assert r.total_hint is None
        assert r.source == "unknown"

    def test_with_items(self):
        items = [SearchResult(uri="1", name="one"), SearchResult(uri="2", name="two")]
        r = SearchResults(items=items, cursor="abc", total_hint=100, source="local")
        assert len(r.items) == 2
        assert r.cursor == "abc"
        assert r.total_hint == 100


# =============================================================================
# SearchBackend protocol compliance
# =============================================================================


class TestSearchBackendProtocol:
    def test_appview_backend_satisfies_protocol(self):
        backend = AppViewSearchBackend(Mock())
        assert isinstance(backend, SearchBackend)

    def test_local_backend_satisfies_protocol(self):
        backend = LocalSearchBackend(Mock())
        assert isinstance(backend, SearchBackend)

    def test_custom_backend_satisfies_protocol(self):
        """A third-party class satisfying the protocol is recognized."""

        class CustomBackend:
            @property
            def name(self) -> str:
                return "custom"

            def search(
                self,
                text=None,
                tags=None,
                schema_ref=None,
                repo=None,
                limit=50,
                cursor=None,
            ):
                return SearchResults(source="custom")

            def health_check(self) -> bool:
                return True

        assert isinstance(CustomBackend(), SearchBackend)


# =============================================================================
# AppViewSearchBackend
# =============================================================================


class TestAppViewSearchBackend:
    def test_search_returns_results(self, mock_atmosphere):
        backend = AppViewSearchBackend(mock_atmosphere)
        results = backend.search(text="genomics")

        assert results.source == "atmosphere"
        assert len(results.items) == 2
        assert results.items[0].name == "genomics-v1"
        assert results.items[0].source == "atmosphere"
        assert results.items[0].tags == ["bio", "genomics"]
        assert results.cursor == "next-page-token"

    def test_search_passes_params(self, mock_atmosphere):
        backend = AppViewSearchBackend(mock_atmosphere)
        backend.search(
            text="test",
            tags=["bio"],
            schema_ref="at://x/y/z",
            repo="did:plc:abc",
            limit=10,
            cursor="prev",
        )

        mock_atmosphere.xrpc_query.assert_called_once()
        call_args = mock_atmosphere.xrpc_query.call_args
        params = call_args[1]["params"] if "params" in call_args[1] else call_args[0][1]
        assert params["q"] == "test"
        assert params["tags"] == ["bio"]
        assert params["schemaRef"] == "at://x/y/z"
        assert params["repo"] == "did:plc:abc"
        assert params["limit"] == 10
        assert params["cursor"] == "prev"

    def test_search_limit_capped_at_100(self, mock_atmosphere):
        backend = AppViewSearchBackend(mock_atmosphere)
        backend.search(limit=200)

        call_args = mock_atmosphere.xrpc_query.call_args
        params = call_args[1]["params"] if "params" in call_args[1] else call_args[0][1]
        assert params["limit"] == 100

    def test_search_no_text_omits_q_param(self, mock_atmosphere):
        backend = AppViewSearchBackend(mock_atmosphere)
        backend.search()

        call_args = mock_atmosphere.xrpc_query.call_args
        params = call_args[1]["params"] if "params" in call_args[1] else call_args[0][1]
        assert "q" not in params

    def test_search_empty_response(self):
        atmo = Mock()
        atmo.xrpc_query.return_value = {"entries": []}
        backend = AppViewSearchBackend(atmo)
        results = backend.search(text="nothing")
        assert results.items == []
        assert results.cursor is None

    def test_health_check_success(self):
        atmo = Mock()
        atmo.xrpc_query.return_value = {"did": "did:web:test"}
        backend = AppViewSearchBackend(atmo)
        assert backend.health_check() is True

    def test_health_check_failure(self):
        atmo = Mock()
        atmo.xrpc_query.side_effect = Exception("unreachable")
        backend = AppViewSearchBackend(atmo)
        assert backend.health_check() is False

    def test_name_property(self, mock_atmosphere):
        backend = AppViewSearchBackend(mock_atmosphere)
        assert backend.name == "atmosphere"


# =============================================================================
# LocalSearchBackend
# =============================================================================


class TestLocalSearchBackend:
    def test_search_by_text(self, populated_index):
        backend = LocalSearchBackend(populated_index)
        results = backend.search(text="mnist")

        assert results.source == "local"
        assert len(results.items) == 2
        names = {r.name for r in results.items}
        assert names == {"mnist-train", "mnist-test"}

    def test_search_case_insensitive(self, populated_index):
        backend = LocalSearchBackend(populated_index)
        results = backend.search(text="MNIST")
        assert len(results.items) == 2

    def test_search_by_tags(self, populated_index):
        backend = LocalSearchBackend(populated_index)
        results = backend.search(tags=["digits"])

        assert len(results.items) == 2
        for r in results.items:
            assert "digits" in r.tags

    def test_search_by_multiple_tags(self, populated_index):
        backend = LocalSearchBackend(populated_index)
        results = backend.search(tags=["vision", "objects"])

        assert len(results.items) == 1
        assert results.items[0].name == "cifar-train"

    def test_search_by_schema_ref(self, populated_index):
        backend = LocalSearchBackend(populated_index)
        # Get actual schema ref from an entry
        entries = populated_index.list_entries()
        ref = entries[0].schema_ref

        results = backend.search(schema_ref=ref)
        # All entries share the same schema
        assert len(results.items) == 3

    def test_search_combined_text_and_tags(self, populated_index):
        backend = LocalSearchBackend(populated_index)
        results = backend.search(text="mnist", tags=["digits"])
        assert len(results.items) == 2

    def test_search_no_matches(self, populated_index):
        backend = LocalSearchBackend(populated_index)
        results = backend.search(text="nonexistent")
        assert results.items == []
        assert results.total_hint == 0

    def test_search_no_filters_returns_all(self, populated_index):
        backend = LocalSearchBackend(populated_index)
        results = backend.search()
        assert len(results.items) == 3

    def test_search_limit(self, populated_index):
        backend = LocalSearchBackend(populated_index)
        results = backend.search(limit=2)
        assert len(results.items) == 2
        assert results.cursor is not None  # more results available

    def test_search_pagination(self, populated_index):
        backend = LocalSearchBackend(populated_index)

        page1 = backend.search(limit=2)
        assert len(page1.items) == 2
        assert page1.cursor is not None

        page2 = backend.search(limit=2, cursor=page1.cursor)
        assert len(page2.items) == 1
        assert page2.cursor is None  # no more results

        # Pages should not overlap
        page1_names = {r.name for r in page1.items}
        page2_names = {r.name for r in page2.items}
        assert page1_names.isdisjoint(page2_names)

    def test_search_description_from_metadata(self, populated_index):
        backend = LocalSearchBackend(populated_index)
        results = backend.search(text="mnist-train")
        assert len(results.items) == 1
        assert results.items[0].description == "MNIST training set"

    def test_health_check_always_true(self, populated_index):
        backend = LocalSearchBackend(populated_index)
        assert backend.health_check() is True

    def test_name_property(self, populated_index):
        backend = LocalSearchBackend(populated_index)
        assert backend.name == "local"

    def test_search_empty_index(self, sqlite_index):
        backend = LocalSearchBackend(sqlite_index)
        results = backend.search(text="anything")
        assert results.items == []


# =============================================================================
# Deduplication
# =============================================================================


class TestDeduplication:
    def test_dedup_by_uri(self):
        items = [
            SearchResult(uri="abc", name="dataset-a", source="local"),
            SearchResult(uri="abc", name="dataset-a-dup", source="atmosphere"),
        ]
        result = _deduplicate(items)
        assert len(result) == 1
        assert result[0].source == "local"  # first occurrence kept

    def test_dedup_by_name_case_insensitive(self):
        items = [
            SearchResult(uri="local-1", name="MNIST", source="local"),
            SearchResult(uri="at://x/y/z", name="mnist", source="atmosphere"),
        ]
        result = _deduplicate(items)
        assert len(result) == 1
        assert result[0].source == "local"

    def test_dedup_preserves_order(self):
        items = [
            SearchResult(uri="1", name="alpha"),
            SearchResult(uri="2", name="beta"),
            SearchResult(uri="3", name="gamma"),
        ]
        result = _deduplicate(items)
        assert [r.name for r in result] == ["alpha", "beta", "gamma"]

    def test_dedup_empty_list(self):
        assert _deduplicate([]) == []

    def test_dedup_all_unique(self):
        items = [
            SearchResult(uri="a", name="one"),
            SearchResult(uri="b", name="two"),
            SearchResult(uri="c", name="three"),
        ]
        result = _deduplicate(items)
        assert len(result) == 3


# =============================================================================
# SearchAggregator
# =============================================================================


class TestSearchAggregator:
    def test_aggregates_multiple_backends(self):
        local = Mock(spec=["name", "search", "health_check"])
        local.name = "local"
        local.health_check.return_value = True
        local.search.return_value = SearchResults(
            items=[SearchResult(uri="local-1", name="ds-local", source="local")],
            source="local",
        )

        remote = Mock(spec=["name", "search", "health_check"])
        remote.name = "atmosphere"
        remote.health_check.return_value = True
        remote.search.return_value = SearchResults(
            items=[SearchResult(uri="at://x", name="ds-remote", source="atmosphere")],
            source="atmosphere",
        )

        agg = SearchAggregator([local, remote])
        results = agg.search(text="ds")

        assert results.source == "aggregated"
        assert len(results.items) == 2

    def test_deduplicates_across_backends(self):
        local = Mock(spec=["name", "search", "health_check"])
        local.name = "local"
        local.health_check.return_value = True
        local.search.return_value = SearchResults(
            items=[SearchResult(uri="local-1", name="mnist", source="local")],
            source="local",
        )

        remote = Mock(spec=["name", "search", "health_check"])
        remote.name = "atmosphere"
        remote.health_check.return_value = True
        remote.search.return_value = SearchResults(
            items=[SearchResult(uri="at://x", name="MNIST", source="atmosphere")],
            source="atmosphere",
        )

        agg = SearchAggregator([local, remote])
        results = agg.search(text="mnist")

        assert len(results.items) == 1
        assert results.items[0].source == "local"  # local listed first

    def test_skips_unhealthy_backend(self):
        healthy = Mock(spec=["name", "search", "health_check"])
        healthy.name = "local"
        healthy.health_check.return_value = True
        healthy.search.return_value = SearchResults(
            items=[SearchResult(uri="1", name="ok", source="local")],
            source="local",
        )

        unhealthy = Mock(spec=["name", "search", "health_check"])
        unhealthy.name = "atmosphere"
        unhealthy.health_check.return_value = False

        agg = SearchAggregator([healthy, unhealthy])
        results = agg.search(text="test")

        assert len(results.items) == 1
        unhealthy.search.assert_not_called()

    def test_handles_backend_exception(self):
        ok = Mock(spec=["name", "search", "health_check"])
        ok.name = "local"
        ok.health_check.return_value = True
        ok.search.return_value = SearchResults(
            items=[SearchResult(uri="1", name="ok", source="local")],
            source="local",
        )

        broken = Mock(spec=["name", "search", "health_check"])
        broken.name = "broken"
        broken.health_check.return_value = True
        broken.search.side_effect = RuntimeError("boom")

        agg = SearchAggregator([ok, broken])
        results = agg.search(text="test")

        assert len(results.items) == 1
        assert results.items[0].name == "ok"

    def test_applies_global_limit(self):
        backend = Mock(spec=["name", "search", "health_check"])
        backend.name = "local"
        backend.health_check.return_value = True
        backend.search.return_value = SearchResults(
            items=[
                SearchResult(uri=str(i), name=f"ds-{i}", source="local")
                for i in range(20)
            ],
            source="local",
        )

        agg = SearchAggregator([backend])
        results = agg.search(limit=5)
        assert len(results.items) == 5

    def test_sorts_scored_results_first(self):
        backend = Mock(spec=["name", "search", "health_check"])
        backend.name = "mixed"
        backend.health_check.return_value = True
        backend.search.return_value = SearchResults(
            items=[
                SearchResult(uri="1", name="unscored", source="local", score=None),
                SearchResult(uri="2", name="low-score", source="atmo", score=0.3),
                SearchResult(uri="3", name="high-score", source="atmo", score=0.9),
            ],
            source="mixed",
        )

        agg = SearchAggregator([backend])
        results = agg.search()

        assert results.items[0].name == "high-score"
        assert results.items[1].name == "low-score"
        assert results.items[2].name == "unscored"

    def test_no_backends(self):
        agg = SearchAggregator([])
        results = agg.search(text="anything")
        assert results.items == []
        assert results.source == "aggregated"


# =============================================================================
# Index.search() integration
# =============================================================================


class TestIndexSearch:
    def test_local_only_search(self, populated_index):
        results = populated_index.search(text="mnist")
        assert len(results.items) == 2
        assert all(r.source == "local" for r in results.items)

    def test_search_with_no_atmosphere(self, populated_index):
        """Index with atmosphere=None still searches local."""
        results = populated_index.search(text="cifar")
        assert len(results.items) == 1
        assert results.items[0].name == "cifar-train"

    def test_search_include_atmosphere_false(self, populated_index):
        results = populated_index.search(text="mnist", include_atmosphere=False)
        assert len(results.items) == 2
        assert all(r.source == "local" for r in results.items)

    def test_search_returns_search_results_type(self, populated_index):
        results = populated_index.search()
        assert isinstance(results, SearchResults)
        for item in results.items:
            assert isinstance(item, SearchResult)

    def test_search_empty_query(self, populated_index):
        """No filters returns all datasets."""
        results = populated_index.search()
        assert len(results.items) == 3

    def test_search_by_tags(self, populated_index):
        results = populated_index.search(tags=["objects"])
        assert len(results.items) == 1
        assert results.items[0].name == "cifar-train"

    def test_search_limit(self, populated_index):
        results = populated_index.search(limit=1)
        assert len(results.items) == 1

    def test_search_no_results(self, populated_index):
        results = populated_index.search(text="nonexistent-dataset")
        assert results.items == []

    def test_search_with_atmosphere_appview(self, tmp_path):
        """Index.search includes atmosphere when appview is configured."""
        index = atdata.Index(
            provider="sqlite",
            path=str(tmp_path / "index.db"),
            atmosphere=None,
        )

        # Mock the atmosphere backend
        mock_atmo_backend = Mock()
        mock_client = Mock()
        mock_client.has_appview = True
        mock_client.xrpc_query.return_value = {
            "entries": [
                {
                    "uri": "at://did:plc:x/science.alt.dataset.entry/t1",
                    "name": "remote-ds",
                    "tags": [],
                }
            ],
            "cursor": None,
        }
        mock_atmo_backend.client = mock_client
        index._atmosphere = mock_atmo_backend
        index._atmosphere_deferred = False

        results = index.search(text="remote")
        assert any(r.source == "atmosphere" for r in results.items)

    def test_search_graceful_degradation(self, tmp_path):
        """If atmosphere health check fails, search still returns local results."""

        @atdata.packable
        class S:
            name: str
            value: int

        store = atdata.LocalDiskStore(root=tmp_path / "data")
        index = atdata.Index(
            provider="sqlite",
            path=str(tmp_path / "index.db"),
            data_store=store,
            atmosphere=None,
        )
        index.write_samples(
            [S(name="x", value=1)],
            name="local-ds",
        )

        # Set up a broken atmosphere backend
        mock_atmo_backend = Mock()
        mock_client = Mock()
        mock_client.has_appview = True
        mock_client.xrpc_query.side_effect = ConnectionError("offline")
        mock_atmo_backend.client = mock_client
        index._atmosphere = mock_atmo_backend
        index._atmosphere_deferred = False

        results = index.search(text="local")
        assert len(results.items) == 1
        assert results.items[0].source == "local"


# =============================================================================
# Public API exports
# =============================================================================


class TestExports:
    def test_search_result_importable(self):
        from atdata import SearchResult as SR

        assert SR is SearchResult

    def test_search_results_importable(self):
        from atdata import SearchResults as SRs

        assert SRs is SearchResults

    def test_search_backend_importable(self):
        from atdata import SearchBackend as SB

        assert SB is SearchBackend

    def test_appview_search_backend_importable(self):
        from atdata import AppViewSearchBackend as ASB

        assert ASB is AppViewSearchBackend

    def test_local_search_backend_importable(self):
        from atdata import LocalSearchBackend as LSB

        assert LSB is LocalSearchBackend

    def test_search_aggregator_importable(self):
        from atdata import SearchAggregator as SA

        assert SA is SearchAggregator
