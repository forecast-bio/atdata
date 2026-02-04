"""Tests for the typed proxy DSL for manifest queries.

Tests FieldProxy operators, Predicate composition and compilation,
query_fields() factory, untyped F proxy, and integration with QueryExecutor.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated

import pandas as pd
import pytest
from numpy.typing import NDArray

import atdata
from atdata.manifest import (
    F,
    FieldProxy,
    ManifestField,
    Predicate,
    QueryExecutor,
    ShardManifest,
    query_fields,
)


# =============================================================================
# Test Sample Types
# =============================================================================


@atdata.packable
class ProxySample:
    """Sample type for proxy DSL tests."""

    image: NDArray
    label: Annotated[str, ManifestField("categorical")]
    confidence: Annotated[float, ManifestField("numeric")]
    tags: Annotated[list[str], ManifestField("set")]


@atdata.packable
class SimpleNumericSample:
    """Minimal sample with auto-inferred numeric field."""

    score: float
    name: str


# =============================================================================
# Shared Test DataFrame
# =============================================================================


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """A small DataFrame mimicking manifest per-sample data."""
    return pd.DataFrame(
        {
            "__key__": ["s0", "s1", "s2", "s3", "s4"],
            "__offset__": [0, 100, 200, 300, 400],
            "__size__": [50, 50, 50, 50, 50],
            "label": ["dog", "cat", "dog", "bird", "cat"],
            "confidence": [0.95, 0.80, 0.60, 0.92, 0.45],
            "tags": [["a"], ["b"], ["a", "b"], ["c"], ["a"]],
        }
    )


# =============================================================================
# FieldProxy Tests
# =============================================================================


class TestFieldProxy:
    """Tests for FieldProxy comparison operators."""

    def test_gt_returns_predicate(self) -> None:
        fp = FieldProxy("x")
        result = fp > 5
        assert isinstance(result, Predicate)

    def test_lt_returns_predicate(self) -> None:
        fp = FieldProxy("x")
        result = fp < 5
        assert isinstance(result, Predicate)

    def test_ge_returns_predicate(self) -> None:
        fp = FieldProxy("x")
        result = fp >= 5
        assert isinstance(result, Predicate)

    def test_le_returns_predicate(self) -> None:
        fp = FieldProxy("x")
        result = fp <= 5
        assert isinstance(result, Predicate)

    def test_eq_returns_predicate(self) -> None:
        fp = FieldProxy("x")
        result = fp == 5
        assert isinstance(result, Predicate)

    def test_ne_returns_predicate(self) -> None:
        fp = FieldProxy("x")
        result = fp != 5
        assert isinstance(result, Predicate)

    def test_isin_returns_predicate(self) -> None:
        fp = FieldProxy("x")
        result = fp.isin([1, 2, 3])
        assert isinstance(result, Predicate)

    def test_between_returns_predicate(self) -> None:
        fp = FieldProxy("x")
        result = fp.between(1, 10)
        assert isinstance(result, Predicate)

    def test_repr(self) -> None:
        fp = FieldProxy("my_field")
        assert repr(fp) == "FieldProxy('my_field')"


# =============================================================================
# Predicate Compilation Tests
# =============================================================================


class TestPredicate:
    """Tests for Predicate compilation and evaluation."""

    def test_compile_gt(self, sample_df: pd.DataFrame) -> None:
        pred = FieldProxy("confidence") > 0.9
        result = pred(sample_df)
        assert list(result) == [True, False, False, True, False]

    def test_compile_lt(self, sample_df: pd.DataFrame) -> None:
        pred = FieldProxy("confidence") < 0.8
        result = pred(sample_df)
        assert list(result) == [False, False, True, False, True]

    def test_compile_ge(self, sample_df: pd.DataFrame) -> None:
        pred = FieldProxy("confidence") >= 0.80
        result = pred(sample_df)
        assert list(result) == [True, True, False, True, False]

    def test_compile_le(self, sample_df: pd.DataFrame) -> None:
        pred = FieldProxy("confidence") <= 0.80
        result = pred(sample_df)
        assert list(result) == [False, True, True, False, True]

    def test_compile_eq(self, sample_df: pd.DataFrame) -> None:
        pred = FieldProxy("label") == "dog"
        result = pred(sample_df)
        assert list(result) == [True, False, True, False, False]

    def test_compile_ne(self, sample_df: pd.DataFrame) -> None:
        pred = FieldProxy("label") != "dog"
        result = pred(sample_df)
        assert list(result) == [False, True, False, True, True]

    def test_compile_isin(self, sample_df: pd.DataFrame) -> None:
        pred = FieldProxy("label").isin(["dog", "cat"])
        result = pred(sample_df)
        assert list(result) == [True, True, True, False, True]

    def test_compile_between(self, sample_df: pd.DataFrame) -> None:
        pred = FieldProxy("confidence").between(0.60, 0.92)
        result = pred(sample_df)
        assert list(result) == [False, True, True, True, False]

    def test_and_composition(self, sample_df: pd.DataFrame) -> None:
        pred = (FieldProxy("confidence") > 0.9) & (FieldProxy("label") == "dog")
        result = pred(sample_df)
        assert list(result) == [True, False, False, False, False]

    def test_or_composition(self, sample_df: pd.DataFrame) -> None:
        pred = (FieldProxy("label") == "bird") | (FieldProxy("confidence") > 0.9)
        result = pred(sample_df)
        assert list(result) == [True, False, False, True, False]

    def test_not(self, sample_df: pd.DataFrame) -> None:
        pred = ~(FieldProxy("label") == "dog")
        result = pred(sample_df)
        assert list(result) == [False, True, False, True, True]

    def test_nested_composition(self, sample_df: pd.DataFrame) -> None:
        pred = ((FieldProxy("confidence") > 0.9) & (FieldProxy("label") == "dog")) | (
            FieldProxy("label") == "cat"
        )
        result = pred(sample_df)
        # dog w/ >0.9: s0; cat: s1, s4
        assert list(result) == [True, True, False, False, True]

    def test_callable_protocol(self, sample_df: pd.DataFrame) -> None:
        """Predicate can be called directly like a function."""
        pred = FieldProxy("confidence") > 0.9
        series = pred(sample_df)
        assert isinstance(series, pd.Series)
        assert series.dtype == bool

    def test_compile_caches(self) -> None:
        pred = FieldProxy("x") > 5
        fn1 = pred.compile()
        fn2 = pred.compile()
        assert fn1 is fn2

    def test_repr_comparison(self) -> None:
        pred = FieldProxy("x") > 5
        assert "x" in repr(pred)
        assert "gt" in repr(pred)

    def test_repr_and(self) -> None:
        pred = (FieldProxy("x") > 1) & (FieldProxy("y") < 2)
        r = repr(pred)
        assert "&" in r

    def test_repr_not(self) -> None:
        pred = ~(FieldProxy("x") > 1)
        assert "~" in repr(pred)

    def test_and_flattens(self) -> None:
        a = FieldProxy("x") > 1
        b = FieldProxy("y") > 2
        c = FieldProxy("z") > 3
        combined = a & b & c
        # Should flatten into a single AND with 3 children
        assert combined._kind == "and"
        assert len(combined._children) == 3

    def test_or_flattens(self) -> None:
        a = FieldProxy("x") > 1
        b = FieldProxy("y") > 2
        c = FieldProxy("z") > 3
        combined = a | b | c
        assert combined._kind == "or"
        assert len(combined._children) == 3

    def test_and_with_non_predicate_returns_not_implemented(self) -> None:
        pred = FieldProxy("x") > 1
        assert pred.__and__(42) is NotImplemented

    def test_or_with_non_predicate_returns_not_implemented(self) -> None:
        pred = FieldProxy("x") > 1
        assert pred.__or__(42) is NotImplemented

    def test_not_hashable(self) -> None:
        pred = FieldProxy("x") > 1
        with pytest.raises(TypeError):
            hash(pred)


# =============================================================================
# query_fields() Tests
# =============================================================================


class TestQueryFields:
    """Tests for the query_fields() factory function."""

    def test_creates_proxies_for_manifest_fields(self) -> None:
        Q = query_fields(ProxySample)
        assert hasattr(Q, "label")
        assert hasattr(Q, "confidence")
        assert hasattr(Q, "tags")

    def test_field_types_are_fieldproxy(self) -> None:
        Q = query_fields(ProxySample)
        assert isinstance(Q.label, FieldProxy)
        assert isinstance(Q.confidence, FieldProxy)
        assert isinstance(Q.tags, FieldProxy)

    def test_excludes_ndarray_fields(self) -> None:
        Q = query_fields(ProxySample)
        assert not hasattr(Q, "image")

    def test_auto_inferred_fields(self) -> None:
        Q = query_fields(SimpleNumericSample)
        assert isinstance(Q.score, FieldProxy)
        assert isinstance(Q.name, FieldProxy)

    def test_raises_on_non_dataclass(self) -> None:
        with pytest.raises(TypeError):
            query_fields(int)

    def test_proxy_repr(self) -> None:
        Q = query_fields(ProxySample)
        r = repr(Q)
        assert "ProxySample" in r

    def test_usable_in_expression(self, sample_df: pd.DataFrame) -> None:
        Q = query_fields(ProxySample)
        pred = (Q.confidence > 0.9) & (Q.label == "dog")
        result = pred(sample_df)
        assert list(result) == [True, False, False, False, False]


# =============================================================================
# Untyped F Proxy Tests
# =============================================================================


class TestUntypedF:
    """Tests for the untyped F convenience proxy."""

    def test_any_attribute_returns_field_proxy(self) -> None:
        assert isinstance(F.anything, FieldProxy)
        assert isinstance(F.some_field, FieldProxy)

    def test_field_proxy_has_correct_name(self) -> None:
        fp = F.confidence
        assert fp._name == "confidence"

    def test_private_attribute_raises(self) -> None:
        with pytest.raises(AttributeError):
            _ = F._private

    def test_repr(self) -> None:
        assert repr(F) == "F"

    def test_usable_in_expression(self, sample_df: pd.DataFrame) -> None:
        pred = (F.confidence > 0.9) & (F.label == "dog")
        result = pred(sample_df)
        assert list(result) == [True, False, False, False, False]


# =============================================================================
# Integration with QueryExecutor
# =============================================================================


def _make_test_manifest(sample_df: pd.DataFrame) -> ShardManifest:
    """Create a minimal ShardManifest for testing."""
    return ShardManifest(
        shard_id="test/shard-000000",
        schema_type="ProxySample",
        schema_version="1.0.0",
        num_samples=len(sample_df),
        size_bytes=1000,
        created_at=datetime.now(timezone.utc),
        aggregates={},
        samples=sample_df,
    )


class TestIntegrationWithQueryExecutor:
    """Tests that Predicate objects work with QueryExecutor.query()."""

    def test_typed_proxy_with_executor(self, sample_df: pd.DataFrame) -> None:
        manifest = _make_test_manifest(sample_df)
        executor = QueryExecutor([manifest])

        Q = query_fields(ProxySample)
        results = executor.query(where=(Q.confidence > 0.9))

        assert len(results) == 2
        keys = {r.key for r in results}
        assert keys == {"s0", "s3"}

    def test_f_proxy_with_executor(self, sample_df: pd.DataFrame) -> None:
        manifest = _make_test_manifest(sample_df)
        executor = QueryExecutor([manifest])

        results = executor.query(where=(F.label == "cat"))

        assert len(results) == 2
        keys = {r.key for r in results}
        assert keys == {"s1", "s4"}

    def test_compound_predicate_with_executor(self, sample_df: pd.DataFrame) -> None:
        manifest = _make_test_manifest(sample_df)
        executor = QueryExecutor([manifest])

        pred = (F.confidence > 0.9) & (F.label == "dog")
        results = executor.query(where=pred)

        assert len(results) == 1
        assert results[0].key == "s0"

    def test_lambda_still_works(self, sample_df: pd.DataFrame) -> None:
        """Existing lambda-based API is unchanged."""
        manifest = _make_test_manifest(sample_df)
        executor = QueryExecutor([manifest])

        results = executor.query(where=lambda df: df["confidence"] > 0.9)

        assert len(results) == 2
        keys = {r.key for r in results}
        assert keys == {"s0", "s3"}

    def test_negated_predicate_with_executor(self, sample_df: pd.DataFrame) -> None:
        manifest = _make_test_manifest(sample_df)
        executor = QueryExecutor([manifest])

        results = executor.query(where=~(F.label == "dog"))

        assert len(results) == 3
        keys = {r.key for r in results}
        assert keys == {"s1", "s3", "s4"}

    def test_isin_with_executor(self, sample_df: pd.DataFrame) -> None:
        manifest = _make_test_manifest(sample_df)
        executor = QueryExecutor([manifest])

        results = executor.query(where=F.label.isin(["dog", "bird"]))

        assert len(results) == 3
        keys = {r.key for r in results}
        assert keys == {"s0", "s2", "s3"}
