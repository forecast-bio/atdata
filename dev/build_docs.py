#!/usr/bin/env python3
"""Build documentation using quartodoc and quarto.

This script can be run from any directory within the project.
It finds the project root by locating pyproject.toml.
"""

import subprocess
import sys
from pathlib import Path


def find_project_root() -> Path:
    """Find project root by searching for pyproject.toml."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    raise RuntimeError("Could not find project root (no pyproject.toml found)")


def main() -> int:
    """Build documentation."""
    project_root = find_project_root()
    docs_src = project_root / "docs_src"
    docs_out = project_root / "docs"

    if not docs_src.exists():
        print(f"Error: docs_src directory not found at {docs_src}", file=sys.stderr)
        return 1

    print(f"Building docs from {docs_src}")

    # Run quartodoc build
    result = subprocess.run(
        ["quartodoc", "build"],
        cwd=docs_src,
    )
    if result.returncode != 0:
        print("Error: quartodoc build failed", file=sys.stderr)
        return result.returncode

    # Run quarto render
    result = subprocess.run(
        ["quarto", "render", "--output-dir", str(docs_out)],
        cwd=docs_src,
    )
    if result.returncode != 0:
        print("Error: quarto render failed", file=sys.stderr)
        return result.returncode

    print(f"Documentation built successfully in {docs_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
