---
allowed-tools: Bash(gh *), Bash(git *), Bash(uv build*), Bash(python3 -c *), Bash(chainlink *)
description: Tag, create GitHub release, and publish to PyPI after a release PR is merged
---

## Context

- Current branch: !`git branch --show-current`
- Current version: !`grep '^version' pyproject.toml`
- Recent tags: !`git tag --sort=-creatordate | head -5`
- Latest release PR: !`gh pr list --repo forecast-bio/atdata --state merged --search "release:" --limit 3 --json number,title,mergedAt --jq '.[] | "#\(.number) \(.title) (\(.mergedAt))"'`
- Remotes: !`git remote -v`

## Your task

The user will provide a version string (e.g. `v0.3.1b1`). If no version is provided, infer it:

1. Check the current version in `pyproject.toml` — this is the most likely version to publish
2. Check for recently merged release PRs that match
3. Present the version to the user for confirmation using `AskUserQuestion` with options:
   - **`v<current_version>`** (Recommended) — publish the version currently in pyproject.toml
   - **Other** — let the user type a different version

Perform the post-merge publish flow:

### 1. Validate preconditions
- Confirm the release branch PR has been merged to `main`
- Confirm the version in `pyproject.toml` matches the requested version (strip leading `v`)
- Confirm no existing tag for this version: `git tag -l <version>`
- Verify the wheel builds cleanly with no duplicate entries:
  ```
  uv build
  python3 -c "import zipfile; z=zipfile.ZipFile('dist/atdata-<pypi_version>-py3-none-any.whl'); names=z.namelist(); dupes=[n for n in names if names.count(n)>1]; print(f'{len(dupes)} duplicates') if dupes else print('OK: {len(names)} entries, no duplicates')"
  ```

### 2. Wait for CI on main
- Check all CI checks on `main` are passing:
  ```
  gh api repos/forecast-bio/atdata/commits/main/check-runs \
    --jq '.check_runs | map({name, status, conclusion}) | .[]'
  ```
- If any checks are still running, poll every 15 seconds until all complete
- If any check fails, STOP and report the failure — do not create the release

### 3. Create GitHub release
- Determine if this is a pre-release (version contains `a`, `b`, `rc`, or `dev`)
- Read the CHANGELOG.md entry for this version to use as release notes
- Create the release:
  ```
  gh release create <version> \
    --repo forecast-bio/atdata \
    --target main \
    --title "<version>" \
    [--prerelease] \
    --notes "<notes from CHANGELOG>"
  ```
- The `uv-publish-pypi.yml` workflow triggers automatically on `release: published`

### 4. Verify publish
- Monitor the publish workflow:
  ```
  gh run list --repo forecast-bio/atdata --workflow=uv-publish-pypi.yml --limit 1
  ```
- Report the PyPI URL: `https://pypi.org/project/atdata/<pypi_version>/`

### 5. Post-publish
- Fetch the new tag locally: `git fetch --tags`
- Update the dev branch if needed: merge `main` into dev branch with `--no-ff`
- Close any chainlink release tracking issues

## Constraints

- NEVER create a tag or release if CI checks on `main` are not all green
- NEVER skip the wheel duplicate check — this was a real issue that blocked v0.3.1b1
- Always use `--prerelease` flag for alpha/beta/rc versions
- The publish workflow uses trusted publishing (OIDC) — no API tokens needed
- If the tag already exists and needs to be recreated, delete it from both `origin` and `upstream` first:
  ```
  git tag -d <version>
  git push origin :refs/tags/<version>
  git push upstream :refs/tags/<version>
  ```
