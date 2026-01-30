"""Statistical aggregate collectors for manifest fields.

Each aggregate type tracks running statistics during shard writing and
produces a summary dict for inclusion in the manifest JSON header.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CategoricalAggregate:
    """Aggregate for categorical (string/enum) fields.

    Tracks value counts and cardinality across all samples in a shard.

    Examples:
        >>> agg = CategoricalAggregate()
        >>> agg.add("dog")
        >>> agg.add("cat")
        >>> agg.add("dog")
        >>> agg.to_dict()
        {'type': 'categorical', 'cardinality': 2, 'value_counts': {'dog': 2, 'cat': 1}}
    """

    value_counts: dict[str, int] = field(default_factory=dict)

    @property
    def cardinality(self) -> int:
        """Number of distinct values observed."""
        return len(self.value_counts)

    def add(self, value: Any) -> None:
        """Record a value observation."""
        key = str(value)
        self.value_counts[key] = self.value_counts.get(key, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "type": "categorical",
            "cardinality": self.cardinality,
            "value_counts": dict(self.value_counts),
        }


@dataclass
class NumericAggregate:
    """Aggregate for numeric (int/float) fields.

    Tracks min, max, sum, and count for computing summary statistics.

    Examples:
        >>> agg = NumericAggregate()
        >>> agg.add(1.0)
        >>> agg.add(3.0)
        >>> agg.add(2.0)
        >>> agg.to_dict()
        {'type': 'numeric', 'min': 1.0, 'max': 3.0, 'mean': 2.0, 'count': 3}
    """

    _min: float = field(default=float("inf"))
    _max: float = field(default=float("-inf"))
    _sum: float = 0.0
    count: int = 0

    @property
    def min(self) -> float:
        """Minimum observed value."""
        return self._min

    @property
    def max(self) -> float:
        """Maximum observed value."""
        return self._max

    @property
    def mean(self) -> float:
        """Running mean of observed values."""
        if self.count == 0:
            return 0.0
        return self._sum / self.count

    def add(self, value: float | int) -> None:
        """Record a numeric observation."""
        v = float(value)
        if v < self._min:
            self._min = v
        if v > self._max:
            self._max = v
        self._sum += v
        self.count += 1

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "type": "numeric",
            "min": self._min,
            "max": self._max,
            "mean": self.mean,
            "count": self.count,
        }


@dataclass
class SetAggregate:
    """Aggregate for set/tag (list) fields.

    Tracks the union of all observed values across samples.

    Examples:
        >>> agg = SetAggregate()
        >>> agg.add(["outdoor", "day"])
        >>> agg.add(["indoor"])
        >>> agg.to_dict()
        {'type': 'set', 'all_values': ['day', 'indoor', 'outdoor']}
    """

    all_values: set[str] = field(default_factory=set)

    def add(self, values: list | set | tuple) -> None:
        """Record a collection of values."""
        for v in values:
            self.all_values.add(str(v))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "type": "set",
            "all_values": sorted(self.all_values),
        }


def create_aggregate(
    kind: str,
) -> CategoricalAggregate | NumericAggregate | SetAggregate:
    """Create an aggregate collector for the given kind.

    Args:
        kind: One of ``"categorical"``, ``"numeric"``, ``"set"``.

    Returns:
        A new aggregate collector instance.

    Raises:
        ValueError: If kind is not recognized.
    """
    if kind == "categorical":
        return CategoricalAggregate()
    if kind == "numeric":
        return NumericAggregate()
    if kind == "set":
        return SetAggregate()
    raise ValueError(f"Unknown aggregate kind: {kind!r}")
