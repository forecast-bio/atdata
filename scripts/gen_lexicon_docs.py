#!/usr/bin/env python3
"""Generate Quarto lexicon reference documentation from lexicon JSON files.

Reads all ``ac.foundation.dataset.*.json`` files in the ``lexicons/`` directory,
parses them according to the ATProto Lexicon v1 specification, and emits a
single Quarto ``.qmd`` page suitable for the Reference section of the docs.

Usage::

    python scripts/gen_lexicon_docs.py                  # default paths
    python scripts/gen_lexicon_docs.py -o docs_src/reference/lexicons.qmd
    python scripts/gen_lexicon_docs.py --lexicon-dir lexicons/

Integrated into the docs build via ``just docs`` (runs before ``quartodoc``).
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Lexicon ordering — records first, then supporting objects, then tokens/query
# ---------------------------------------------------------------------------

_NSID_ORDER = [
    "ac.foundation.dataset.schema",
    "ac.foundation.dataset.record",
    "ac.foundation.dataset.lens",
    "ac.foundation.dataset.storageHttp",
    "ac.foundation.dataset.storageS3",
    "ac.foundation.dataset.storageBlobs",
    "ac.foundation.dataset.storageExternal",
    "ac.foundation.dataset.schemaType",
    "ac.foundation.dataset.arrayFormat",
    "ac.foundation.dataset.resolveSchema",
]


def _sort_key(nsid: str) -> tuple[int, str]:
    try:
        return (_NSID_ORDER.index(nsid), nsid)
    except ValueError:
        return (len(_NSID_ORDER), nsid)


# ---------------------------------------------------------------------------
# Type rendering helpers
# ---------------------------------------------------------------------------

_TYPE_BADGES: dict[str, str] = {
    "record": "Record",
    "object": "Object",
    "query": "Query",
    "token": "Token",
    "string": "String",
}


def _badge(def_type: str) -> str:
    label = _TYPE_BADGES.get(def_type, def_type.title())
    return f"<span class='lexicon-badge lexicon-badge-{def_type}'>{label}</span>"


def _resolve_ref(ref: str, parent_nsid: str) -> str:
    """Resolve a possibly-short ref like ``#foo`` to a full ``nsid#foo``."""
    if ref.startswith("#"):
        return parent_nsid + ref
    return ref


def _type_label(prop: dict[str, Any], parent_nsid: str = "") -> str:
    """Render a human-readable type string for a property definition."""
    t = prop.get("type", "unknown")
    if t == "ref":
        ref = prop.get("ref", "")
        if ref:
            full = _resolve_ref(ref, parent_nsid)
            return f"[`{ref}`](#{_anchor(full)})"
        return "ref"
    if t == "union":
        refs = prop.get("refs", [])
        parts = []
        for r in refs:
            full = _resolve_ref(r, parent_nsid)
            parts.append(f"[`{r}`](#{_anchor(full)})")
        return " \\| ".join(parts) if parts else "union"
    if t == "array":
        items = prop.get("items", {})
        inner = _type_label(items, parent_nsid) if isinstance(items, dict) else "any"
        return f"array\\<{inner}\\>"
    if t == "blob":
        return "blob"
    if t == "bytes":
        return "bytes"
    if t == "string":
        fmt = prop.get("format")
        if fmt:
            return f"string ({fmt})"
        return "string"
    if t == "integer":
        return "integer"
    if t == "object":
        return "object"
    return t


def _anchor(ref: str) -> str:
    """Create an HTML anchor id from a lexicon NSID or def ref."""
    return ref.replace("#", "-").replace(".", "-").lower()


def _constraints(prop: dict[str, Any]) -> list[str]:
    """Extract constraint annotations from a property definition."""
    parts: list[str] = []
    if "maxLength" in prop:
        parts.append(f"max length: {prop['maxLength']}")
    if "minLength" in prop:
        parts.append(f"min length: {prop['minLength']}")
    if "maxProperties" in prop:
        parts.append(f"max properties: {prop['maxProperties']}")
    if "minProperties" in prop:
        parts.append(f"min properties: {prop['minProperties']}")
    if "minimum" in prop:
        parts.append(f"min: {prop['minimum']}")
    if "maximum" in prop:
        parts.append(f"max: {prop['maximum']}")
    if "pattern" in prop:
        parts.append(f"pattern: `{prop['pattern']}`")
    if "const" in prop:
        parts.append(f"const: `{prop['const']}`")
    if "format" in prop and prop.get("type") == "string":
        pass  # already shown in type label
    if prop.get("type") == "blob":
        accept = prop.get("accept")
        if accept:
            parts.append(f"accept: {', '.join(f'`{a}`' for a in accept)}")
        max_size = prop.get("maxSize")
        if max_size is not None:
            mb = max_size / (1024 * 1024)
            parts.append(f"max size: {mb:.0f} MB")
    if "knownValues" in prop:
        vals = ", ".join(f"`{v}`" for v in prop["knownValues"])
        parts.append(f"known values: {vals}")
    return parts


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_properties_table(
    properties: dict[str, Any],
    required: list[str],
    parent_nsid: str = "",
) -> str:
    """Render a markdown table of properties."""
    lines: list[str] = []
    lines.append("| Property | Type | Required | Description |")
    lines.append("|----------|------|----------|-------------|")

    for name, prop in properties.items():
        type_str = _type_label(prop, parent_nsid)
        req = "**yes**" if name in required else "no"
        desc = prop.get("description", "")
        cons = _constraints(prop)
        if cons:
            desc += " " + " · ".join(f"*{c}*" for c in cons)
        # Escape pipes in description
        desc = desc.replace("|", "\\|")
        lines.append(f"| `{name}` | {type_str} | {req} | {desc} |")

    return "\n".join(lines)


def _render_query_params(params: dict[str, Any], parent_nsid: str = "") -> str:
    """Render query parameters section."""
    properties = params.get("properties", {})
    required = params.get("required", [])
    if not properties:
        return ""
    lines = ["#### Parameters\n"]
    lines.append(_render_properties_table(properties, required, parent_nsid))
    return "\n".join(lines)


def _render_query_output(output: dict[str, Any], parent_nsid: str = "") -> str:
    """Render query output section."""
    encoding = output.get("encoding", "")
    schema = output.get("schema", {})
    lines = [f"#### Output\n"]
    if encoding:
        lines.append(f"Encoding: `{encoding}`\n")
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    if properties:
        lines.append(_render_properties_table(properties, required, parent_nsid))
    return "\n".join(lines)


def _render_query_errors(errors: list[dict[str, Any]]) -> str:
    """Render query errors section."""
    if not errors:
        return ""
    lines = ["#### Errors\n"]
    for err in errors:
        lines.append(f"- **{err['name']}** — {err.get('description', '')}")
    return "\n".join(lines)


def _render_def(
    nsid: str,
    def_name: str,
    def_body: dict[str, Any],
) -> str:
    """Render a single definition (main or auxiliary)."""
    def_type = def_body.get("type", "unknown")
    desc = def_body.get("description", "")
    deprecated = "(Deprecated" in desc

    if def_name == "main":
        full_id = nsid
        heading = f"### `{nsid}` {_badge(def_type)}"
    else:
        full_id = f"{nsid}#{def_name}"
        heading = f"#### `{nsid}#{def_name}` {_badge(def_type)}"

    anchor_id = _anchor(full_id)
    lines: list[str] = []
    lines.append(f"{heading} {{#{anchor_id}}}\n")

    if deprecated:
        lines.append("::: {.callout-warning}")
        lines.append("This type is deprecated.")
        lines.append(":::\n")

    if desc:
        lines.append(f"{desc}\n")

    # Record type: unwrap the inner record object
    if def_type == "record":
        key = def_body.get("key", "")
        if key:
            lines.append(f"**Record key:** `{key}`\n")
        record_obj = def_body.get("record", {})
        properties = record_obj.get("properties", {})
        required = record_obj.get("required", [])
        if properties:
            lines.append(_render_properties_table(properties, required, nsid))
            lines.append("")

    # Object type
    elif def_type == "object":
        properties = def_body.get("properties", {})
        required = def_body.get("required", [])
        if properties:
            lines.append(_render_properties_table(properties, required, nsid))
            lines.append("")

    # Query type
    elif def_type == "query":
        params = def_body.get("parameters", {})
        if params:
            lines.append(_render_query_params(params, nsid))
            lines.append("")
        output = def_body.get("output", {})
        if output:
            lines.append(_render_query_output(output, nsid))
            lines.append("")
        errors = def_body.get("errors", [])
        if errors:
            lines.append(_render_query_errors(errors))
            lines.append("")

    # String type (extensible enum)
    elif def_type == "string":
        cons = _constraints(def_body)
        if cons:
            lines.append(" · ".join(f"*{c}*" for c in cons))
            lines.append("")

    # Token type
    elif def_type == "token":
        pass  # description is sufficient

    return "\n".join(lines)


def _render_lexicon(path: Path) -> str:
    """Render a full lexicon file into markdown sections."""
    data = json.loads(path.read_text(encoding="utf-8"))
    nsid = data.get("id", path.stem)
    defs = data.get("defs", {})

    parts: list[str] = []

    # Render main def first
    if "main" in defs:
        parts.append(_render_def(nsid, "main", defs["main"]))

    # Render auxiliary defs
    for def_name, def_body in defs.items():
        if def_name == "main":
            continue
        parts.append(_render_def(nsid, def_name, def_body))

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Page assembly
# ---------------------------------------------------------------------------

_PREAMBLE = textwrap.dedent("""\
    ---
    title: "Lexicon Reference"
    description: "ATProto Lexicon definitions for the ac.foundation.dataset namespace"
    ---

    {{< include _lexicon-styles.qmd >}}

    This page documents the [ATProto Lexicon](https://atproto.com/specs/lexicon)
    definitions that make up the `ac.foundation.dataset` namespace. These
    lexicons define the record types, objects, tokens, and queries used by
    atdata for publishing schemas, datasets, and lenses to the AT Protocol
    network.

    All lexicons conform to ATProto Lexicon version 1 and are published
    from the canonical source in
    [`lexicons/`](https://github.com/forecast-bio/atdata/tree/main/lexicons).

    ::: {.callout-note}
    ## Reading this reference

    Each entry shows the **NSID** (Namespaced Identifier), its **type**
    (Record, Object, Token, Query, or String), and a property table with
    type information and constraints.  Internal cross-references link to
    auxiliary definitions within the same namespace.
    :::

    ## Namespace overview

    | NSID | Type | Description |
    |------|------|-------------|
""")

_STYLES = textwrap.dedent("""\
    ```{=html}
    <style>
    .lexicon-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75em;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        vertical-align: middle;
        margin-left: 4px;
    }
    .lexicon-badge-record { background: #dbeafe; color: #1e40af; }
    .lexicon-badge-object { background: #dcfce7; color: #166534; }
    .lexicon-badge-query  { background: #fef3c7; color: #92400e; }
    .lexicon-badge-token  { background: #f3e8ff; color: #6b21a8; }
    .lexicon-badge-string { background: #e0e7ff; color: #3730a3; }
    </style>
    ```
""")


def _overview_row(nsid: str, defs: dict[str, Any]) -> str:
    main = defs.get("main", {})
    def_type = main.get("type", "—")
    desc = main.get("description", "")
    # Truncate long descriptions
    if len(desc) > 120:
        desc = desc[:117] + "..."
    desc = desc.replace("|", "\\|")
    anchor = _anchor(nsid)
    return f"| [`{nsid}`](#{anchor}) | {_TYPE_BADGES.get(def_type, def_type)} | {desc} |"


def generate(lexicon_dir: Path, output_path: Path) -> None:
    """Generate the lexicon reference .qmd file."""
    # Collect lexicon files (skip ndarray_shim which is JSON Schema, not Lexicon)
    paths: list[tuple[str, Path]] = []
    for p in sorted(lexicon_dir.glob("ac.foundation.dataset.*.json")):
        nsid = p.stem
        paths.append((nsid, p))

    paths.sort(key=lambda t: _sort_key(t[0]))

    # Build overview table
    overview_rows: list[str] = []
    for nsid, p in paths:
        data = json.loads(p.read_text(encoding="utf-8"))
        overview_rows.append(_overview_row(nsid, data.get("defs", {})))

    # Build detail sections
    detail_sections: list[str] = []
    for nsid, p in paths:
        detail_sections.append(_render_lexicon(p))

    # Assemble page
    page = _PREAMBLE
    page += "\n".join(overview_rows) + "\n\n"
    page += "## Definitions\n\n"
    page += "\n---\n\n".join(detail_sections)

    # Also write the styles include file
    styles_path = output_path.parent / "_lexicon-styles.qmd"
    styles_path.write_text(_STYLES, encoding="utf-8")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page, encoding="utf-8")

    print(f"Generated {output_path} ({len(paths)} lexicons)")
    print(f"Generated {styles_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Quarto lexicon reference from lexicon JSON files."
    )
    parser.add_argument(
        "--lexicon-dir",
        type=Path,
        default=Path("lexicons"),
        help="Directory containing lexicon JSON files (default: lexicons/)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("docs_src/reference/lexicons.qmd"),
        help="Output .qmd file path",
    )
    args = parser.parse_args()
    generate(args.lexicon_dir, args.output)


if __name__ == "__main__":
    main()
