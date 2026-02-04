---
allowed-tools: Bash(git *), Bash(chainlink *), Bash(uuidgen), Bash(ls *), Bash(ln *), Bash(rm *), Bash(tmux *), Bash(claude *), Bash(cat *), Bash(mktemp), Bash(which *), Skill
description: Create a worktree and launch a background claude agent in tmux to implement a feature
---

## Context

- Current repo root: !`git rev-parse --show-toplevel`
- Current branch: !`git branch --show-current`
- Existing worktrees: !`git worktree list`
- Working tree status: !`git status --short`
- tmux available: !`which tmux`
- Active tmux sessions: !`tmux list-sessions 2>/dev/null || echo "no tmux server running"`

## Your task

The user provides a feature description (e.g. "add batch retry logic") and optionally additional context, file references, or constraints. You will create an isolated worktree, then launch a background `claude` process in a tmux session to implement the feature autonomously.

### 1. Validate prerequisites

- Confirm `tmux` is installed (`which tmux`). If not, abort with a message telling the user to install it.
- Confirm `claude` CLI is installed (`which claude`). If not, abort.

### 2. Create the worktree via /featree

- Invoke the `/featree` skill with the user's feature description.
- Capture the worktree path and branch name from the output.

### 3. Prepare the agent prompt

Build a detailed prompt for the child agent. The prompt must be self-contained — the child has no access to this conversation's context. Include:

- The feature description from the user
- Any specific files, modules, or code areas the user mentioned
- Any constraints or requirements the user specified
- Instructions to:
  1. Read the CLAUDE.md for project conventions
  2. Explore relevant code before making changes
  3. Implement the feature fully (no stubs)
  4. Run `uv run pytest` to verify changes don't break tests
  5. Use `/commit` to commit the work when implementation is complete
  6. Run `/adr` to perform an adversarial review of the changes
  7. Use `/commit` again after any adr fixes
  8. When completely finished, write the word `DONE` to a file called `.kickoff-status` in the worktree root

Write this prompt to a temp file so it can be passed to claude cleanly.

### 4. Derive the tmux session name

- Use the feature branch slug as the tmux session name (e.g. `feat-add-batch-retry-logic`).
- Prefix with `feat-` and truncate to 50 characters if needed.
- Replace any characters invalid for tmux session names (periods, colons) with hyphens.

### 5. Launch the tmux session

```bash
tmux new-session -d -s <session-name> -c <worktree-path>
```

Then send the claude command into the session. The child agent gets explicit tool permissions — broad enough to work autonomously, scoped enough to prevent destructive actions.

**Critical quoting rules for tmux send-keys:**
- `--allowedTools` is variadic (consumes all subsequent space-separated args). You MUST pass tools as a **single comma-separated string** so the prompt isn't consumed as a tool name.
- Use `--` before the positional prompt argument to terminate option parsing.
- The prompt file content must be passed as a single shell argument. Use `"$(cat <prompt-file>)"` with proper quoting.

```bash
tmux send-keys -t <session-name> "claude --model opus --allowedTools 'Read,Write,Edit,Glob,Grep,Skill,Task,WebSearch,WebFetch,Bash(git *),Bash(uv *),Bash(chainlink *),Bash(just *),Bash(ls *),Bash(mkdir *),Bash(test *),Bash(which *),Bash(touch *),Bash(cat *),Bash(head *),Bash(tail *),Bash(wc *),Bash(diff *),Bash(echo *)' -- \"\$(cat <prompt-file>)\"" Enter
```

**Permission rationale:**
- `Read`, `Write`, `Edit`, `Glob`, `Grep` — full file operations for implementation
- `Skill` — enables `/commit` and `/adr` workflows
- `Task` — subagent exploration and code search
- `WebSearch`, `WebFetch` — look up docs and API references
- `Bash(git *)` — version control operations
- `Bash(uv *)` — run tests (`uv run pytest`), linting (`uv run ruff`), and any uv commands
- `Bash(chainlink *)` — issue tracking
- `Bash(just *)` — project task runner
- `Bash(ls/mkdir/test/which/touch/cat/head/tail/wc/diff/echo *)` — standard file inspection utilities
- **Excluded**: unrestricted `Bash`, `NotebookEdit`, destructive commands (`rm`, `curl|sh`, etc.)

### 6. Report to user

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
- Clean up the temp prompt file after launching (it's been read by the shell).
- If a tmux session with the same name already exists, append a short random suffix rather than failing.
