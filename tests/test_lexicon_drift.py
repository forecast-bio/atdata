"""Lexicon drift detection tests.

Ensures every property defined in the lexicon JSON files has a corresponding
field in the Python dataclass types in ``_lexicon_types.py``.  When a new
property is added to a lexicon but not to the Python type, this test fails.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from atdata.atmosphere._lexicon_types import (
    LexDatasetEntry,
    LexLabelRecord,
    LexLensRecord,
    LexLensVerification,
    LexSchemaRecord,
)

LEXICON_DIR = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "atdata"
    / "lexicons"
    / "science"
    / "alt"
    / "dataset"
)

# Map lexicon JSON file stems to their corresponding Python types.
RECORD_TYPES: dict[str, type] = {
    "entry": LexDatasetEntry,
    "schema": LexSchemaRecord,
    "lens": LexLensRecord,
    "label": LexLabelRecord,
    "lensVerification": LexLensVerification,
}


def _camel_to_snake(name: str) -> str:
    """Convert camelCase to snake_case."""
    s1 = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def _get_lexicon_properties(lexicon_path: Path) -> set[str]:
    """Extract main record property names from a lexicon JSON file."""
    data = json.loads(lexicon_path.read_text())
    main_def = data.get("defs", {}).get("main", {})
    record = main_def.get("record", {})
    return set(record.get("properties", {}).keys())


def _get_dataclass_fields(cls: type) -> set[str]:
    """Get field names from a dataclass, including inherited fields."""
    import dataclasses

    return {f.name for f in dataclasses.fields(cls)}


def _build_test_cases() -> list[tuple[str, Path, type]]:
    """Build parametrized test cases for each lexicon/type pair."""
    cases = []
    for stem, py_type in RECORD_TYPES.items():
        lexicon_path = LEXICON_DIR / f"{stem}.json"
        if lexicon_path.exists():
            cases.append((stem, lexicon_path, py_type))
    return cases


@pytest.mark.parametrize(
    "stem,lexicon_path,py_type",
    _build_test_cases(),
    ids=[c[0] for c in _build_test_cases()],
)
def test_lexicon_properties_covered_by_python_type(
    stem: str, lexicon_path: Path, py_type: type
) -> None:
    """Every lexicon record property must have a Python dataclass field."""
    lexicon_props = _get_lexicon_properties(lexicon_path)
    python_fields = _get_dataclass_fields(py_type)

    # Build mapping: camelCase lexicon prop -> expected snake_case Python field
    missing = []
    for prop in sorted(lexicon_props):
        # Skip $type — it's not a stored field
        if prop.startswith("$"):
            continue
        expected_field = _camel_to_snake(prop)
        if expected_field not in python_fields:
            missing.append(f"  {prop} -> {expected_field}")

    assert not missing, (
        f"Lexicon '{stem}' has properties not covered by {py_type.__name__}:\n"
        + "\n".join(missing)
        + "\nAdd the missing fields to the Python dataclass."
    )
