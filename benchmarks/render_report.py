"""Render pytest-benchmark JSON results into a standalone HTML report.

Usage:
    uv run python -m benchmarks.render_report results/*.json -o bench-report.html

Reads one or more pytest-benchmark JSON files (one per group) and produces
a single HTML page with grouped tables and test descriptions extracted from
the benchmark source files.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import jinja2


# =============================================================================
# Docstring extraction
# =============================================================================


@dataclass
class _SourceMeta:
    """Metadata extracted from benchmark source files via AST."""

    docstrings: dict[str, str] = field(default_factory=dict)
    param_labels: dict[str, dict[str, str]] = field(default_factory=dict)


def _extract_source_meta(bench_dir: Path) -> _SourceMeta:
    """Walk benchmark .py files and extract class/method docstrings and PARAM_LABELS.

    Returns a ``_SourceMeta`` with:
    - ``docstrings``: mapping qualified names like
      ``"TestSerializationBenchmarks"`` or
      ``"TestSerializationBenchmarks.test_serialize_basic_sample"``
      to their docstring text.
    - ``param_labels``: mapping class names to their ``PARAM_LABELS`` dict
      (e.g. ``{"n": "samples per shard"}``).
    """
    meta = _SourceMeta()
    for py_file in sorted(bench_dir.glob("bench_*.py")):
        tree = ast.parse(py_file.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                cls_doc = ast.get_docstring(node)
                if cls_doc:
                    meta.docstrings[node.name] = cls_doc
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        fn_doc = ast.get_docstring(item)
                        if fn_doc:
                            meta.docstrings[f"{node.name}.{item.name}"] = fn_doc
                    # Extract PARAM_LABELS = {...} class variable
                    if isinstance(item, ast.Assign):
                        for target in item.targets:
                            if (
                                isinstance(target, ast.Name)
                                and target.id == "PARAM_LABELS"
                            ):
                                try:
                                    value = ast.literal_eval(item.value)
                                    if isinstance(value, dict):
                                        meta.param_labels[node.name] = value
                                except (ValueError, TypeError):
                                    pass
    return meta


# =============================================================================
# Data model
# =============================================================================

# Friendly names for each marker group
GROUP_TITLES: dict[str, str] = {
    "bench_serial": "Serialization (μs scale)",
    "bench_index": "Index Providers (μs–ms scale)",
    "bench_io": "Dataset I/O (ms scale)",
    "bench_query": "Query System (ms scale)",
    "bench_s3": "S3 Storage (ms+ scale)",
}


@dataclass
class BenchRow:
    name: str
    fullname: str
    description: str
    raw_params: str
    params_desc: str
    median: float
    iqr: float
    ops: float
    n_samples: int | None = None


@dataclass
class BenchGroup:
    key: str
    title: str
    class_description: str
    param_labels: dict[str, str] = field(default_factory=dict)
    has_samples: bool = False
    rows: list[BenchRow] = field(default_factory=list)


def _format_time(seconds: float) -> str:
    """Format seconds into a human-readable string with appropriate unit."""
    if seconds < 1e-6:
        return f"{seconds * 1e9:.1f} ns"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.2f} μs"
    if seconds < 1.0:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.3f} s"


def _format_ops(ops: float) -> str:
    """Format operations per second with SI prefix."""
    if ops >= 1e6:
        return f"{ops / 1e6:.2f} Mops/s"
    if ops >= 1e3:
        return f"{ops / 1e3:.2f} Kops/s"
    return f"{ops:.1f} ops/s"


def _class_from_fullname(fullname: str) -> str:
    """Extract class name from a pytest fullname like
    ``benchmarks/bench_dataset_io.py::TestSerializationBenchmarks::test_foo``.
    """
    parts = fullname.split("::")
    if len(parts) >= 2:
        return parts[1]
    return ""


def _method_from_fullname(fullname: str) -> str:
    """Extract method name from a pytest fullname."""
    parts = fullname.split("::")
    if len(parts) >= 3:
        return parts[2]
    return parts[-1]


# =============================================================================
# Build groups from JSON
# =============================================================================


def _format_params(
    params_dict: dict | None,
    param_labels: dict[str, str],
) -> str:
    """Format a benchmark's params dict using human-readable labels.

    Given ``{"n": 1000}`` and labels ``{"n": "samples per shard"}``,
    returns ``"1000 samples per shard"``.
    """
    if not params_dict:
        return ""
    parts: list[str] = []
    for key, value in params_dict.items():
        label = param_labels.get(key)
        if label:
            parts.append(f"{value} {label}")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts)


def _build_groups(
    json_paths: list[Path],
    meta: _SourceMeta,
) -> list[BenchGroup]:
    """Load JSON files and assemble BenchGroup objects."""
    # Each JSON file corresponds to one marker group.  We infer the group key
    # from the filename produced by the justfile (e.g. ``serial.json``).
    groups: dict[str, BenchGroup] = {}

    for path in sorted(json_paths):
        data = json.loads(path.read_text())
        benchmarks = data.get("benchmarks", [])
        if not benchmarks:
            continue

        group_key = path.stem  # e.g. "serial", "index", "io", "query", "s3"
        marker_key = f"bench_{group_key}"
        title = GROUP_TITLES.get(marker_key, group_key.title())

        # Collect unique class descriptions and param labels for the group
        class_names_seen: set[str] = set()
        class_descs: list[str] = []
        merged_param_labels: dict[str, str] = {}
        rows: list[BenchRow] = []

        for bench in benchmarks:
            fullname = bench["fullname"]
            stats = bench["stats"]
            cls_name = _class_from_fullname(fullname)
            method_name = _method_from_fullname(fullname)

            if cls_name and cls_name not in class_names_seen:
                class_names_seen.add(cls_name)
                cls_doc = meta.docstrings.get(cls_name, "")
                if cls_doc:
                    class_descs.append(cls_doc)
                # Merge param labels from this class
                cls_labels = meta.param_labels.get(cls_name, {})
                for k, v in cls_labels.items():
                    if k not in merged_param_labels:
                        merged_param_labels[k] = v

            # Build description: prefer method docstring, fall back to
            # readable version of test name (strip bracket suffix)
            base_method = method_name.split("[")[0]
            qualified = f"{cls_name}.{base_method}" if cls_name else base_method
            desc = meta.docstrings.get(qualified, "")
            if not desc:
                # Convert test_serialize_basic_sample -> Serialize basic sample
                readable = base_method.removeprefix("test_").replace("_", " ").capitalize()
                desc = readable

            # Raw param ID for the top line (e.g. "sqlite-5v")
            raw_param = bench.get("param") or ""
            # Human-readable param description for the subtitle
            params_dict = bench.get("params")
            params_desc = _format_params(params_dict, merged_param_labels)

            extra = bench.get("extra_info", {})
            n_samples = extra.get("n_samples")
            if n_samples is not None:
                n_samples = int(n_samples)

            rows.append(
                BenchRow(
                    name=bench["name"],
                    fullname=fullname,
                    description=desc,
                    raw_params=str(raw_param) if raw_param else "",
                    params_desc=params_desc,
                    median=stats["median"],
                    iqr=stats["iqr"],
                    ops=stats["ops"],
                    n_samples=n_samples,
                )
            )

        groups[group_key] = BenchGroup(
            key=group_key,
            title=title,
            class_description="; ".join(class_descs) if class_descs else "",
            param_labels=merged_param_labels,
            has_samples=any(r.n_samples is not None for r in rows),
            rows=rows,
        )

    return list(groups.values())


# =============================================================================
# HTML template
# =============================================================================

TEMPLATE = jinja2.Template(
    """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>atdata benchmark report</title>
<style>
  :root {
    --bg: #fdfdfd; --fg: #1a1a1a; --muted: #6b7280;
    --border: #e5e7eb; --accent: #2563eb; --row-alt: #f9fafb;
    --green: #16a34a; --header-bg: #f1f5f9;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #0f172a; --fg: #e2e8f0; --muted: #94a3b8;
      --border: #334155; --accent: #60a5fa; --row-alt: #1e293b;
      --green: #4ade80; --header-bg: #1e293b;
    }
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg); color: var(--fg);
    line-height: 1.5; padding: 2rem; max-width: 1200px; margin: 0 auto;
  }
  h1 { font-size: 1.5rem; margin-bottom: 0.25rem; }
  .subtitle { color: var(--muted); font-size: 0.875rem; margin-bottom: 2rem; }
  .machine { color: var(--muted); font-size: 0.8rem; margin-bottom: 2rem;
             padding: 0.75rem; background: var(--row-alt); border-radius: 6px;
             border: 1px solid var(--border); }
  .machine code { font-size: 0.8rem; }
  section { margin-bottom: 2.5rem; }
  h2 { font-size: 1.15rem; margin-bottom: 0.25rem; color: var(--accent); }
  .group-desc { color: var(--muted); font-size: 0.85rem; margin-bottom: 0.75rem; }
  table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
  th {
    text-align: left; padding: 0.5rem 0.75rem;
    background: var(--header-bg); border-bottom: 2px solid var(--border);
    font-weight: 600; white-space: nowrap;
  }
  th.num { text-align: right; }
  td { padding: 0.45rem 0.75rem; border-bottom: 1px solid var(--border); }
  td.num { text-align: right; font-variant-numeric: tabular-nums; font-family: monospace; }
  tr:nth-child(even) td { background: var(--row-alt); }
  .best { color: var(--green); font-weight: 600; }
  .test-name { font-family: monospace; font-weight: 600; color: var(--fg); }
  .test-params { color: var(--accent); font-family: monospace; }
  .desc { color: var(--muted); font-size: 0.8rem; }
  .param-legend { color: var(--muted); font-size: 0.8rem; margin-bottom: 0.5rem; }
  .param-legend code { background: var(--header-bg); padding: 0.1rem 0.35rem;
                        border-radius: 3px; font-size: 0.8rem; }
  footer { margin-top: 3rem; color: var(--muted); font-size: 0.75rem;
           border-top: 1px solid var(--border); padding-top: 1rem; }
</style>
</head>
<body>
<h1>atdata benchmark report</h1>
<p class="subtitle">Generated from pytest-benchmark JSON output</p>

{% if machine %}
<div class="machine">
  <strong>{{ machine.node }}</strong> &mdash;
  {{ machine.cpu.brand_raw }} ({{ machine.cpu.count }} cores) &middot;
  Python {{ machine.python_version }} &middot;
  {{ machine.system }} {{ machine.release }}
  {% if commit %}
  &middot; <code>{{ commit.branch }}@{{ commit.id[:8] }}{% if commit.dirty %} (dirty){% endif %}</code>
  {% endif %}
</div>
{% endif %}

{% for group in groups %}
<section id="{{ group.key }}">
  <h2>{{ group.title }}</h2>
  {% if group.class_description %}
  <p class="group-desc">{{ group.class_description }}</p>
  {% endif %}
  {% if group.param_labels %}
  <p class="param-legend">Parameters: {% for key, label in group.param_labels.items() %}<code>{{ key }}</code> = {{ label }}{% if not loop.last %}, {% endif %}{% endfor %}</p>
  {% endif %}
  <table>
    <thead>
      <tr>
        <th>Test</th>
        <th class="num">Median</th>
        <th class="num">IQR</th>
        <th class="num">OPS</th>
        {% if group.has_samples %}<th class="num">Med/sample</th>
        <th class="num">Samples/s</th>{% endif %}
      </tr>
    </thead>
    <tbody>
    {% for row in group.rows %}
      <tr>
        <td>
          {% if "[" in row.name %}<span class="test-name">{{ row.name.split("[")[0] }}</span><span class="test-params">[{{ row.name.split("[")[1] }}</span>{% else %}<span class="test-name">{{ row.name }}</span>{% endif %}
          <br><span class="desc">{{ row.description }}{% if row.params_desc %} [{{ row.params_desc }}]{% endif %}</span>
        </td>
        <td class="num{% if loop.first %} best{% endif %}">{{ fmt_time(row.median) }}</td>
        <td class="num">{{ fmt_time(row.iqr) }}</td>
        <td class="num">{{ fmt_ops(row.ops) }}</td>
        {% if group.has_samples %}<td class="num">{% if row.n_samples %}{{ fmt_time(row.median / row.n_samples) }}{% else %}&mdash;{% endif %}</td>
        <td class="num">{% if row.n_samples %}{{ fmt_ops(row.n_samples / row.median) }}{% else %}&mdash;{% endif %}</td>{% endif %}
      </tr>
    {% endfor %}
    </tbody>
  </table>
</section>
{% endfor %}

<footer>
  Report generated by <code>benchmarks/render_report.py</code> from
  {{ groups | length }} benchmark group{{ "s" if groups | length != 1 }},
  {{ total_benchmarks }} total benchmarks.
</footer>
</body>
</html>
""",
    undefined=jinja2.StrictUndefined,
)


# =============================================================================
# Main
# =============================================================================


def render_html(json_paths: list[Path], bench_dir: Path) -> str:
    """Render benchmark JSON files into an HTML string."""
    meta = _extract_source_meta(bench_dir)
    groups = _build_groups(json_paths, meta)

    # Extract machine/commit info from the first JSON file
    machine = None
    commit = None
    for path in json_paths:
        data = json.loads(path.read_text())
        if "machine_info" in data:
            machine = data["machine_info"]
            commit = data.get("commit_info")
            break

    total = sum(len(g.rows) for g in groups)

    return TEMPLATE.render(
        groups=groups,
        machine=machine,
        commit=commit,
        total_benchmarks=total,
        fmt_time=_format_time,
        fmt_ops=_format_ops,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Render pytest-benchmark JSON into HTML report",
    )
    parser.add_argument(
        "json_files",
        nargs="+",
        type=Path,
        help="One or more pytest-benchmark JSON files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("bench-report.html"),
        help="Output HTML file (default: bench-report.html)",
    )
    parser.add_argument(
        "--bench-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Directory containing bench_*.py files for docstring extraction",
    )
    args = parser.parse_args(argv)

    existing = [p for p in args.json_files if p.exists()]
    if not existing:
        print(f"Error: no JSON files found: {args.json_files}", file=sys.stderr)
        sys.exit(1)

    html = render_html(existing, args.bench_dir)
    args.output.write_text(html)
    print(f"Wrote {args.output} ({len(html):,} bytes, from {len(existing)} JSON files)")


if __name__ == "__main__":
    main()
