---
allowed-tools: Bash(git log:*), Bash(git tag:*), Bash(git diff:*), Bash(chainlink *)
description: Generate a clean CHANGELOG entry from recent work
---

## Context

- Current version: !`grep '^version' pyproject.toml`
- Recent tags: !`git tag --sort=-creatordate | head -5`
- CHANGELOG head: !`head -20 CHANGELOG.md`
- Recent chainlink issues: !`chainlink list`

## Your task

Generate a properly structured CHANGELOG entry for the current release, following [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) format.

### 1. Gather changes

Identify all changes since the last release by examining:
- `git log --oneline <last-release-tag-or-branch>..HEAD` for commit messages
- `chainlink list` for closed issues and their descriptions
- `git diff --stat <last-release-tag-or-branch>..HEAD` for files changed

### 2. Categorize changes

Sort changes into Keep a Changelog sections:

- **Added**: New features, new files, new public APIs, new test suites
- **Changed**: Modifications to existing behavior, refactors, dependency updates, CI changes
- **Fixed**: Bug fixes, lint fixes, CI fixes
- **Deprecated**: Newly deprecated APIs (with migration path)
- **Removed**: Removed features, deleted files, removed APIs

### 3. Write the entry

Follow these formatting rules:
- Each item should be a concise, user-facing description — not a chainlink issue title
- Group related changes under bold sub-headers (e.g. **`LocalDiskStore`**: description)
- Use nested bullets for sub-items that belong to a feature group
- Omit internal-only changes (individual subissue closes, review assessments, investigation tickets)
- Include GitHub issue references where relevant (e.g. `(GH#42)`)
- Do NOT include chainlink issue numbers — these are internal tracking

### 4. Update CHANGELOG.md

- Insert the new version section between `## [Unreleased]` and the previous release
- Leave `## [Unreleased]` empty at the top
- Do not modify any existing release sections below

### 5. Verify

- Confirm the CHANGELOG renders as valid markdown
- Confirm no chainlink auto-appended entries leaked into existing release sections

## Constraints

- Follow Keep a Changelog format strictly
- Write for the library's users, not for internal tracking
- Consolidate — 5 well-written bullets are better than 30 issue titles
- Preserve existing release sections exactly as they are
- If chainlink has appended noise to existing sections, clean it up
