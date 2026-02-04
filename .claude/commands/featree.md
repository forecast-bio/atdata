---
allowed-tools: Bash(git *), Bash(chainlink *), Bash(uuidgen), Bash(ls *), Skill
description: Create a feature branch and move it to a new git worktree
---

## Context

- Current repo root: !`git rev-parse --show-toplevel`
- Current branch: !`git branch --show-current`
- Existing worktrees: !`git worktree list`
- Working tree status: !`git status --short`

## Your task

The user provides a human-readable feature description (e.g. "add batch retry logic"). You will first create a feature branch using the `/feature` skill, then move it into a new git worktree.

### 1. Create the feature branch

- Invoke the `/feature` skill with the user's description as the argument.
- This creates the `feature/<slug>` branch and a chainlink issue.
- Note the branch name that was created.

### 2. Generate worktree path

- Generate a short random suffix using `uuidgen | cut -c1-8` (8-char hex).
- The worktree path is `<repo-root>--<suffix>` as a sibling of the current repo directory.
  - Example: if repo root is `/Users/max/git-forecast/atdata`, the worktree is `/Users/max/git-forecast/atdata--a1b2c3d4`.

### 3. Create the worktree

- Switch back to the previous branch (the one we were on before `/feature` created the new branch): `git checkout -`
- Create the worktree pointing at the feature branch: `git worktree add <worktree-path> feature/<slug>`

### 4. Report to user

Print a summary:
```
Worktree: <path>
Branch:   feature/<slug>

To start working:
  cd <worktree-path>
```

## Constraints

- Never force-push or delete branches.
- Do not push the branch to a remote — the user will do that when ready.
