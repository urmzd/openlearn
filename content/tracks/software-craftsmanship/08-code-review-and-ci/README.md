# Code Review and CI

The gates every change to your course system passes through: a CI workflow that runs on every push and pull request, commit messages a machine can read, and (in Pass 7) reviewing a change someone else wrote against the system you own.

## Overview

- **What lives here**: the course chapters `craft.01` (your repo and CI gate, Pass 0) and `craft.08` (code review of a seeded pull request, Pass 7).
- **Prerequisites**: [Shell, git, make, processes](../12-language-and-tool-primers/02-shell-git-make.md) (`lang.02`).
- **Optional depth**: *Software Engineering at Google*, ch. 9 "Code Review" and ch. 23 "Continuous Integration" (abseil.io/resources/swe-book, free); Conventional Commits 1.0.0 (conventionalcommits.org, free).

## Key Takeaways

- CI is a function from a commit to pass or fail; branch protection turns it into a gate.
- One rule, one implementation: the hook you run locally is the step CI runs.
- Conventional commit headers make history machine-readable, which is what releases (craft.11) are computed from.

## How to Study

- Read `craft.01` right after the two Pass 0 primers: [Pass 0](../../../courses/build-an-llm/paths/course-p00-setup/milestone.md) ends when the repository's CI is green.
- Keep the gate strict from the first commit: loosening it later is easy, tightening it means rewriting history.

## Chapters

| # | Module | Chapter | Kind | Pass |
|---|---|---|---|---|
| 1 | `craft.01` | [Your repo and CI gate (conventional commits enforced)](01-your-repo-and-ci-gate.md) | practice | 0 |
| 2 | `craft.08` | [Code review: review a seeded PR against your system](02-code-review.md) | practice | 7 |
