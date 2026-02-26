"""Unified search API with pluggable backends.

Provides a ``SearchBackend`` protocol for pluggable search providers, concrete
implementations for the AppView (``AppViewSearchBackend``) and local index
(``LocalSearchBackend``), and a ``SearchAggregator`` that merges and
deduplicates results from multiple backends.

Examples:
    >>> from atdata.search import LocalSearchBackend, SearchAggregator
    >>> backend = LocalSearchBackend(index)
    >>> results = backend.search(text="mnist")
    >>> for r in results.items:
    ...     print(r.name, r.source)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from atdata._logging import get_logger


# ------------------------------------------------------------------ #
# Result types
# ------------------------------------------------------------------ #


@dataclass
class SearchResult:
    """A single search result from any backend.

    Attributes:
        uri: AT-URI or local identifier (e.g. CID).
        name: Human-readable dataset name.
        description: Optional description.
        tags: Tags associated with the dataset.
        schema_ref: Schema reference, if available.
        source: Backend name that produced this result
            (e.g. ``"local"``, ``"atmosphere"``).
        score: Relevance score (if the backend provides one).
        record: Raw record data, if available.
    """

    uri: str
    name: str
    description: str | None = None
    tags: list[str] = field(default_factory=list)
    schema_ref: str | None = None
    source: str = "unknown"
    score: float | None = None
    record: dict | None = None


@dataclass
class SearchResults:
    """Paginated collection of search results.

    Attributes:
        items: The list of results for this page.
        cursor: Opaque cursor for fetching the next page, or ``None``
            if there are no more results.
        total_hint: Estimated total number of matching results across
            all pages (may be ``None`` if the backend doesn't provide it).
        source: Describes which backend(s) contributed
            (e.g. ``"local"``, ``"atmosphere"``, ``"aggregated"``).
    """

    items: list[SearchResult] = field(default_factory=list)
    cursor: str | None = None
    total_hint: int | None = None
    source: str = "unknown"


# ------------------------------------------------------------------ #
# SearchBackend protocol
# ------------------------------------------------------------------ #


@runtime_checkable
class SearchBackend(Protocol):
    """Interface for pluggable search backends.

    Implementations must provide a ``name`` property, a ``search`` method
    returning ``SearchResults``, and a ``health_check`` method.

    Examples:
        >>> class MyBackend:
        ...     @property
        ...     def name(self) -> str:
        ...         return "custom"
        ...     def search(self, text=None, **kw) -> SearchResults:
        ...         return SearchResults(items=[], source="custom")
        ...     def health_check(self) -> bool:
        ...         return True
    """

    @property
    def name(self) -> str:
        """Backend identifier (e.g. ``"appview"``, ``"local"``)."""
        ...

    def search(
        self,
        text: str | None = None,
        tags: list[str] | None = None,
        schema_ref: str | None = None,
        repo: str | None = None,
        limit: int = 50,
        cursor: str | None = None,
    ) -> SearchResults:
        """Search for datasets matching the given criteria.

        Args:
            text: Free-text query string.
            tags: Filter by tags (all must match).
            schema_ref: Filter by schema reference.
            repo: Filter by repository DID or handle.
            limit: Maximum number of results to return.
            cursor: Pagination cursor from a previous ``SearchResults``.

        Returns:
            A ``SearchResults`` with matching items.
        """
        ...

    def health_check(self) -> bool:
        """Return ``True`` if this backend is reachable and operational."""
        ...


# ------------------------------------------------------------------ #
# AppView search backend
# ------------------------------------------------------------------ #


class AppViewSearchBackend:
    """Search backend that delegates to an AppView's XRPC endpoints.

    Wraps :meth:`Atmosphere.xrpc_query` for the
    ``science.alt.dataset.searchDatasets`` endpoint and returns standardized
    ``SearchResults``.

    Args:
        atmosphere: An ``Atmosphere`` instance with an AppView configured.

    Examples:
        >>> from atdata.atmosphere.client import Atmosphere
        >>> atmo = Atmosphere(appview="https://datasets.atdata.blue")
        >>> backend = AppViewSearchBackend(atmo)
        >>> results = backend.search(text="genomics")
    """

    def __init__(self, atmosphere: Any) -> None:
        self._atmo = atmosphere

    @property
    def name(self) -> str:
        """Backend identifier."""
        return "atmosphere"

    def search(
        self,
        text: str | None = None,
        tags: list[str] | None = None,
        schema_ref: str | None = None,
        repo: str | None = None,
        limit: int = 50,
        cursor: str | None = None,
    ) -> SearchResults:
        """Search datasets via the AppView ``searchDatasets`` XRPC endpoint.

        Args:
            text: Free-text query string.
            tags: Filter by tags.
            schema_ref: Filter by schema reference.
            repo: Filter by repository DID or handle.
            limit: Maximum results (capped at 100 by the AppView).
            cursor: Pagination cursor.

        Returns:
            ``SearchResults`` with atmosphere-sourced items.

        Raises:
            AppViewRequiredError: If no AppView is configured.
            AppViewUnavailableError: If the AppView is unreachable.
        """
        from atdata.atmosphere._types import LEXICON_NAMESPACE

        params: dict[str, Any] = {"limit": min(limit, 100)}
        if text is not None:
            params["q"] = text
        if tags is not None:
            params["tags"] = tags
        if schema_ref is not None:
            params["schemaRef"] = schema_ref
        if repo is not None:
            params["repo"] = repo
        if cursor is not None:
            params["cursor"] = cursor

        result = self._atmo.xrpc_query(
            f"{LEXICON_NAMESPACE}.searchDatasets", params=params
        )

        items: list[SearchResult] = []
        for entry in result.get("entries", []):
            items.append(
                SearchResult(
                    uri=entry.get("uri", ""),
                    name=entry.get("name", ""),
                    description=entry.get("description"),
                    tags=entry.get("tags", []),
                    schema_ref=entry.get("schemaRef"),
                    source="atmosphere",
                    record=entry,
                )
            )

        return SearchResults(
            items=items,
            cursor=result.get("cursor"),
            total_hint=None,
            source="atmosphere",
        )

    def health_check(self) -> bool:
        """Check AppView reachability via ``describeService``."""
        try:
            from atdata.atmosphere._types import LEXICON_NAMESPACE

            self._atmo.xrpc_query(f"{LEXICON_NAMESPACE}.describeService")
            return True
        except Exception:
            return False


# ------------------------------------------------------------------ #
# Local search backend
# ------------------------------------------------------------------ #


class LocalSearchBackend:
    """Search backend for the local index.

    Searches ``LocalDatasetEntry`` records by substring matching on name
    and filtering by tags and schema reference.  Does not require any
    external services.

    Args:
        index: An ``Index`` instance.

    Examples:
        >>> from atdata import Index
        >>> index = Index()
        >>> backend = LocalSearchBackend(index)
        >>> results = backend.search(text="train")
    """

    def __init__(self, index: Any) -> None:
        self._index = index

    @property
    def name(self) -> str:
        """Backend identifier."""
        return "local"

    def search(
        self,
        text: str | None = None,
        tags: list[str] | None = None,
        schema_ref: str | None = None,
        repo: str | None = None,
        limit: int = 50,
        cursor: str | None = None,
    ) -> SearchResults:
        """Search local index entries by name substring and metadata filters.

        Args:
            text: Substring to match against entry names (case-insensitive).
            tags: Filter by tags stored in entry metadata.
            schema_ref: Filter by schema reference.
            repo: Ignored for local search (local entries have no repo DID).
            limit: Maximum results.
            cursor: Offset-based pagination cursor (stringified integer).

        Returns:
            ``SearchResults`` with locally-sourced items.
        """
        offset = int(cursor) if cursor else 0

        matched: list[SearchResult] = []
        text_lower = text.lower() if text else None

        for entry in self._index.datasets:
            # Text filter: substring match on name
            if text_lower is not None and text_lower not in entry.name.lower():
                continue

            # Schema filter
            if schema_ref is not None and entry.schema_ref != schema_ref:
                continue

            # Tag filter: check entry metadata for tags
            entry_tags: list[str] = []
            if entry.metadata and isinstance(entry.metadata.get("tags"), list):
                entry_tags = entry.metadata["tags"]

            if tags is not None:
                if not all(t in entry_tags for t in tags):
                    continue

            matched.append(
                SearchResult(
                    uri=entry.cid,
                    name=entry.name,
                    description=(
                        entry.metadata.get("description") if entry.metadata else None
                    ),
                    tags=entry_tags,
                    schema_ref=entry.schema_ref,
                    source="local",
                )
            )

        # Apply offset-based pagination
        total = len(matched)
        page = matched[offset : offset + limit]
        next_cursor = str(offset + limit) if offset + limit < total else None

        return SearchResults(
            items=page,
            cursor=next_cursor,
            total_hint=total,
            source="local",
        )

    def health_check(self) -> bool:
        """Local backend is always available."""
        return True


# ------------------------------------------------------------------ #
# Search aggregator
# ------------------------------------------------------------------ #


class SearchAggregator:
    """Aggregates and deduplicates results from multiple search backends.

    Results are collected from every healthy backend, deduplicated by URI,
    and returned as a single ``SearchResults``. When a dataset appears in
    multiple backends the first occurrence is kept (backends are tried in
    the order given).

    Args:
        backends: Ordered list of search backends to query.

    Examples:
        >>> agg = SearchAggregator([local_backend, appview_backend])
        >>> results = agg.search(text="mnist")
    """

    def __init__(self, backends: list[SearchBackend]) -> None:
        self._backends = backends

    def search(
        self,
        text: str | None = None,
        tags: list[str] | None = None,
        schema_ref: str | None = None,
        repo: str | None = None,
        limit: int = 50,
        cursor: str | None = None,
    ) -> SearchResults:
        """Query all healthy backends and return deduplicated results.

        Args:
            text: Free-text query string.
            tags: Filter by tags.
            schema_ref: Filter by schema reference.
            repo: Filter by repository DID or handle.
            limit: Maximum total results returned.
            cursor: Not used by the aggregator (each backend may paginate
                independently).

        Returns:
            ``SearchResults`` with items from all backends, deduplicated
            by URI and name.
        """
        log = get_logger()
        all_items: list[SearchResult] = []

        for backend in self._backends:
            if not backend.health_check():
                log.warning(
                    "Search backend %s failed health check, skipping", backend.name
                )
                continue
            try:
                results = backend.search(
                    text=text,
                    tags=tags,
                    schema_ref=schema_ref,
                    repo=repo,
                    limit=limit,
                )
                all_items.extend(results.items)
            except Exception:
                log.warning(
                    "Search backend %s raised an exception, skipping",
                    backend.name,
                    exc_info=True,
                )

        deduplicated = _deduplicate(all_items)

        # Sort: scored results first (descending), then alphabetical by name
        deduplicated.sort(key=lambda r: (r.score is None, -(r.score or 0), r.name))

        # Apply global limit
        page = deduplicated[:limit]

        return SearchResults(
            items=page,
            cursor=None,
            total_hint=len(deduplicated),
            source="aggregated",
        )


def _deduplicate(items: list[SearchResult]) -> list[SearchResult]:
    """Remove duplicate results, keeping the first occurrence.

    Deduplication keys:
    1. By URI (exact match).
    2. By name (case-insensitive) — a dataset named "mnist" from local
       and atmosphere should appear only once.
    """
    seen_uris: set[str] = set()
    seen_names: set[str] = set()
    unique: list[SearchResult] = []

    for item in items:
        if item.uri in seen_uris:
            continue
        name_key = item.name.lower()
        if name_key in seen_names:
            continue
        seen_uris.add(item.uri)
        seen_names.add(name_key)
        unique.append(item)

    return unique
