---
allowed-tools: Bash(git *), Bash(crosslink *), Bash(uuidgen), Bash(ls *), Bash(ln *), Bash(rm *), Bash(tmux *), Bash(claude *), Bash(cat *), Bash(grep *), Bash(which *), Read, Write, Edit, Glob, Grep, Skill, Task
description: Create a worktree and launch a background claude agent in tmux to implement a feature
---

## Context

- Current repo root: !`git rev-parse --show-toplevel`
- Current branch: !`git branch --show-current`
- Existing worktrees: !`git worktree list`
- Working tree status: !`git status --short`
- tmux available: !`which tmux`
- CLAUDE.md head: !`head -5 CLAUDE.md 2>/dev/null || echo "no CLAUDE.md"`
- Project root files: !`ls -1`

## Your task

The user provides a feature description (e.g. "add batch retry logic") and optionally additional context, file references, or constraints. You will create an isolated worktree, then launch a background `claude` process in a tmux session to implement the feature autonomously.

### 1. Validate prerequisites

- Confirm `tmux` is installed (`which tmux`). If not, abort with a message telling the user to install it.
- Confirm `claude` CLI is installed (`which claude`). If not, abort.

### 2. Create the worktree via /featree

- Invoke the `/featree` skill with the user's feature description.
- Capture the worktree path and branch name from the output.
- The worktree will be at `<repo-root>/.worktrees/<slug>` (inside the repo, inheriting trust scope).

### 3. Detect project conventions

Before writing the prompt, detect what tools the project uses so the child agent gets appropriate instructions:

- **Test runner**: Check for `justfile` (`just test`), `Makefile`, `package.json` (`npm test`), `pyproject.toml` (`uv run pytest` or `pytest`), `Cargo.toml` (`cargo test`), etc.
- **Linter**: Check for `ruff.toml`, `.eslintrc`, `clippy`, etc.
- **Task runner**: `just`, `make`, `npm run`, `cargo`, etc.
- **CLAUDE.md**: If present, the child agent will read it for full project conventions.

### 4. Prepare the agent prompt

Build a detailed prompt for the child agent. The prompt must be self-contained — the child has no access to this conversation's context. Include:

- The feature description from the user
- Any specific files, modules, or code areas the user mentioned
- Any constraints or requirements the user specified
- Instructions to:
  1. **Read the project's CLAUDE.md** (if it exists) for conventions before starting
  2. Explore relevant code before making changes
  3. Implement the feature fully (no stubs or placeholders)
  4. **Run the project's test suite** to verify changes don't break anything (use the detected test command)
  5. Use `/commit` to commit the work when implementation is complete
  6. Review the diff of all changes and fix any issues found
  7. Use `/commit` again after any fixes
  8. When completely finished, write the word `DONE` to a file called `.kickoff-status` in the worktree root

Write the prompt to `KICKOFF.md` in the worktree root. Ensure it's excluded from git by adding to the **main repo's** `.git/info/exclude` (worktrees share `info/exclude` from `$GIT_COMMON_DIR`):

```bash
# Get the common git dir (main repo's .git/)
common_dir=$(git -C <worktree-path> rev-parse --git-common-dir)
# Add KICKOFF.md and .kickoff-status if not already present
grep -qxF 'KICKOFF.md' "$common_dir/info/exclude" || echo "KICKOFF.md" >> "$common_dir/info/exclude"
grep -qxF '.kickoff-status' "$common_dir/info/exclude" || echo ".kickoff-status" >> "$common_dir/info/exclude"
```

### 5. Derive the tmux session name

- Use the feature branch slug as the tmux session name (e.g. `feat-add-batch-retry-logic`).
- Prefix with `feat-` and truncate to 50 characters if needed.
- Replace any characters invalid for tmux session names (periods, colons) with hyphens.

### 6. Launch the tmux session

```bash
tmux new-session -d -s <session-name> -c <worktree-path>
```

Then send the claude command into the session. **Important:**
- Prefix with `env -u CLAUDECODE` to clear the nested-session detection variable inherited from the parent shell.
- Use `--allowedTools` to auto-approve the listed tools so the agent can work without interactive permission prompts.
- Do **NOT** use `--dangerously-skip-permissions` — the agent operates within the normal permission model with `--allowedTools` for auto-approval.

**Critical quoting rules for tmux send-keys:**
- `--allowedTools` is variadic (consumes all subsequent space-separated args). You MUST pass tools as a **single comma-separated string** so the prompt isn't consumed as a tool name.
- Use `--` before the positional prompt argument to terminate option parsing.
- The prompt file content must be passed as a single shell argument. Use `"$(cat KICKOFF.md)"` with proper quoting. Since the tmux session's working directory is already the worktree, `KICKOFF.md` resolves correctly.

```bash
tmux send-keys -t <session-name> "env -u CLAUDECODE claude --model opus --allowedTools 'Read,Write,Edit,Glob,Grep,Skill,Task,WebSearch,WebFetch,Bash(git *),Bash(ls *),Bash(mkdir *),Bash(test *),Bash(which *),Bash(touch *),Bash(cat *),Bash(head *),Bash(tail *),Bash(wc *),Bash(diff *),Bash(echo *),<project-specific-tools>' -- \"\$(cat KICKOFF.md)\"" Enter
```

**Project-specific tool additions** (add to the comma-separated list based on what was detected):
- Python/uv project: `Bash(uv *)`
- Node project: `Bash(npm *),Bash(npx *)`
- Rust project: `Bash(cargo *)`
- Just task runner: `Bash(just *)`
- Make task runner: `Bash(make *)`
- Crosslink present: `Bash(crosslink *)`

**Permission rationale:**
- `Read`, `Write`, `Edit`, `Glob`, `Grep` — full file operations for implementation
- `Skill` — enables `/commit` and other skill workflows
- `Task` — subagent exploration and code search
- `WebSearch`, `WebFetch` — look up docs and API references
- `Bash(git *)` — version control operations
- Project-specific tools — test runner, package manager, task runner
- Standard utilities (`ls/mkdir/test/which/touch/cat/head/tail/wc/diff/echo`) — file inspection
- **Excluded**: unrestricted `Bash`, `NotebookEdit`, destructive commands (`rm`, `curl|sh`, etc.)

### 7. Report to user

Print a summary:

```
Feature agent launched.

  Worktree: <path>
  Branch:   feature/<slug>
  Session:  <tmux-session-name>

  Check status:    /check <tmux-session-name>
  Attach directly: tmux attach -t <tmux-session-name>
```

## Constraints

- Never force-push or delete branches.
- Do not push the branch to a remote.
- The child agent prompt must be fully self-contained — assume it has zero context from this conversation beyond what you explicitly include.
- Leave `KICKOFF.md` in the worktree for reference (it's git-excluded).
- If a tmux session with the same name already exists, append a short random suffix rather than failing.
