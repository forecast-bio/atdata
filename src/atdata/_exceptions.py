"""Custom exception hierarchy for atdata.

Provides actionable error messages with contextual help, available
alternatives, and suggested fix code snippets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Type


class AtdataError(Exception):
    """Base exception for all atdata errors."""


class LensNotFoundError(AtdataError):
    """No lens registered to transform between two sample types.

    Attributes:
        source_type: The source sample type.
        view_type: The target view type.
        available_targets: Types reachable from the source via registered lenses.
    """

    def __init__(
        self,
        source_type: Type,
        view_type: Type,
        available_targets: list[tuple[Type, str]] | None = None,
    ) -> None:
        self.source_type = source_type
        self.view_type = view_type
        self.available_targets = available_targets or []

        src_name = source_type.__name__
        view_name = view_type.__name__

        lines = [f"No lens transforms {src_name} \u2192 {view_name}"]

        if self.available_targets:
            lines.append("")
            lines.append(f"Available lenses from {src_name}:")
            for target_type, lens_name in self.available_targets:
                lines.append(f"  - {src_name} \u2192 {target_type.__name__} (via {lens_name})")

        lines.append("")
        lines.append("Did you mean to define:")
        lines.append(f"  @lens")
        lines.append(f"  def {src_name.lower()}_to_{view_name.lower()}(source: {src_name}) -> {view_name}:")
        lines.append(f"      return {view_name}(...)")

        super().__init__("\n".join(lines))


class SchemaError(AtdataError):
    """Schema mismatch during sample deserialization.

    Raised when the data in a shard doesn't match the expected sample type.

    Attributes:
        expected_fields: Fields expected by the sample type.
        actual_fields: Fields found in the data.
        sample_type_name: Name of the target sample type.
    """

    def __init__(
        self,
        sample_type_name: str,
        expected_fields: list[str],
        actual_fields: list[str],
    ) -> None:
        self.sample_type_name = sample_type_name
        self.expected_fields = expected_fields
        self.actual_fields = actual_fields

        missing = sorted(set(expected_fields) - set(actual_fields))
        extra = sorted(set(actual_fields) - set(expected_fields))

        lines = [f"Schema mismatch for {sample_type_name}"]
        if missing:
            lines.append(f"  Missing fields: {', '.join(missing)}")
        if extra:
            lines.append(f"  Unexpected fields: {', '.join(extra)}")
        lines.append("")
        lines.append(f"Expected: {', '.join(sorted(expected_fields))}")
        lines.append(f"Got:      {', '.join(sorted(actual_fields))}")

        super().__init__("\n".join(lines))


class SampleKeyError(AtdataError, KeyError):
    """Sample with the given key was not found in the dataset.

    Attributes:
        key: The key that was not found.
    """

    def __init__(self, key: str) -> None:
        self.key = key
        super().__init__(
            f"Sample with key '{key}' not found in dataset. "
            f"Note: key lookup requires scanning all shards and is O(n)."
        )


class ShardError(AtdataError):
    """Error accessing or reading a dataset shard.

    Attributes:
        shard_id: Identifier of the shard that failed.
        reason: Human-readable description of what went wrong.
    """

    def __init__(self, shard_id: str, reason: str) -> None:
        self.shard_id = shard_id
        self.reason = reason
        super().__init__(f"Failed to read shard '{shard_id}': {reason}")
