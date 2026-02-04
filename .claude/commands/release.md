---
allowed-tools: Bash(git *), Bash(gh *), Bash(uv lock*), Bash(uv run ruff*), Bash(uv run pytest*), Bash(chainlink *), Bash(uv run ruff format*)
description: Prepare and submit a beta release
---

## Context

- Current branch: !`git branch --show-current`
- Recent commits: !`git log --oneline -15`
- All branches: !`git branch --list 'release/*' | tail -5`
- Current version: !`grep '^version' pyproject.toml`
- Repo identity: !`gh repo view --json nameWithOwner --jq .nameWithOwner`
- Remotes: !`git remote -v`

## Your task

The user will provide a version string (e.g. `v0.3.0b2`). If no version is provided, suggest options based on the current version using `AskUserQuestion`:

Given current version `X.Y.ZbN`:
- **Next beta**: `vX.Y.ZbN+1` (e.g. `0.3.1b1` → `v0.3.1b2`) — increments pre-release number
- **Stable release**: `vX.Y.Z` (e.g. `0.3.1b1` → `v0.3.1`) — drops pre-release suffix
- **Next minor beta**: `vX.Y+1.0b1` (e.g. `0.3.1b1` → `v0.4.0b1`) — bumps minor, resets patch and beta
- **Next patch beta**: `vX.Y.Z+1b1` (e.g. `0.3.1b1` → `v0.3.2b1`) — bumps patch, resets beta

Given current version `X.Y.Z` (stable):
- **Next patch beta**: `vX.Y.Z+1b1` (e.g. `0.3.1` → `v0.3.2b1`)
- **Next minor beta**: `vX.Y+1.0b1` (e.g. `0.3.1` → `v0.4.0b1`)
- **Next major beta**: `vX+1.0.0b1` (e.g. `0.3.1` → `v1.0.0b1`)

Present 3-4 options with the most likely choice first (marked as recommended). Then proceed with the selected version.

Detect the repo owner/name dynamically via `gh repo view --json nameWithOwner --jq .nameWithOwner`.

Perform the full release flow:

### 1. Validate preconditions
- Confirm all tests pass: `uv run pytest tests/ -x -q`
- Confirm lint is clean: `uv run ruff check src/ tests/`
- Confirm formatting is clean: `uv run ruff format --check src/ tests/` (fix with `uv run ruff format src/ tests/` if needed)
- Confirm no uncommitted changes (other than `.chainlink/issues.db`)
- Confirm `develop` branch is up to date with `main` (merge main into develop if needed)

### 2. Create release branch from develop
- Stash any uncommitted changes
- `git checkout develop`
- `git pull origin develop`
- `git checkout -b release/<version>`
- `git stash pop` (if anything was stashed)

### 3. Prepare release
- Bump version in `pyproject.toml`
- Run `uv lock` to update the lockfile
- Run `/changelog` skill to generate a clean CHANGELOG entry (or generate one manually following Keep a Changelog format with Added/Changed/Fixed sections)
- Run `uv run ruff check src/ tests/` and fix any lint errors
- Run `uv run ruff format --check src/ tests/` and fix any format errors (run `uv run ruff format src/ tests/` to auto-fix)
- Run `uv run pytest tests/ -x -q` to confirm tests pass

### 4. Commit and push
- `git add pyproject.toml uv.lock CHANGELOG.md .chainlink/issues.db`
- `git commit -m "release: prepare <version>"`
- `git push -u origin release/<version>`

### 5. Create PR
- Create PR to `main` using `gh pr create`:
  - `--repo <owner>/<repo>`
  - `--base main`
  - `--head release/<version>`
  - Title: `release: <version>`
  - Body: summary of changes from CHANGELOG, test plan with pass counts

### 6. Post-merge: sync develop
After the PR is merged to `main`, sync develop:
```bash
git checkout develop
git merge main --no-ff --no-edit
git push origin develop
```
Remind the user to do this after merge, or do it if the PR is already merged.

### 7. Track in chainlink
- Create a chainlink issue for the release, close when PR is submitted

## Constraints

- **Branch from `develop`**, not from previous release branches
- Always use `--no-ff` for merges to preserve branch topology
- Always run `uv lock` after version bumps — stale lockfiles break CI
- Always run both `ruff check` and `ruff format --check` before committing — either will fail CI
- Never force-push to release branches
- The CHANGELOG should follow Keep a Changelog format with proper Added/Changed/Fixed sections, not a flat list of chainlink issues
