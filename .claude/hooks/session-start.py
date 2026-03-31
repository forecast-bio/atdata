#!/usr/bin/env python3
"""
Session start hook that loads crosslink context and reminds about session workflow.
"""

import json
import subprocess
import sys
import os


def run_crosslink(args):
    """Run a crosslink command and return output."""
    try:
        result = subprocess.run(
            ["crosslink"] + args,
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return None


def check_crosslink_initialized():
    """Check if .crosslink directory exists."""
    cwd = os.getcwd()
    current = cwd

    while True:
        candidate = os.path.join(current, ".crosslink")
        if os.path.isdir(candidate):
            return True
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent

    return False


def main():
    if not check_crosslink_initialized():
        # No crosslink repo, skip
        sys.exit(0)

    context_parts = ["<chainlink-session-context>"]

    # Try to get session status
    session_status = run_crosslink(["session", "status"])
    if session_status:
        context_parts.append(f"## Current Session\n{session_status}")

    # Get ready issues (unblocked work)
    ready_issues = run_crosslink(["issue", "ready"])
    if ready_issues:
        context_parts.append(f"## Ready Issues (unblocked)\n{ready_issues}")

    # Get open issues summary
    open_issues = run_crosslink(["issue", "list", "-s", "open"])
    if open_issues:
        context_parts.append(f"## Open Issues\n{open_issues}")

    context_parts.append("""
## Chainlink Workflow Reminder
- Use `crosslink session start` at the beginning of work
- Use `crosslink session work <id>` to mark current focus
- Add comments as you discover things: `crosslink issue comment <id> "..."`
- End with handoff notes: `crosslink session end --notes "..."`
</chainlink-session-context>""")

    print("\n\n".join(context_parts))
    sys.exit(0)


if __name__ == "__main__":
    main()
