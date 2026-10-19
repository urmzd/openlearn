# What the system does after Pass 0

There is no model yet. What exists is a repository whose history and CI are trustworthy, which every later pass depends on: each pass adds code in four languages, and the only way to know a change did not break an earlier layer is a CI run on every commit.

## What runs

| Check | What it does | Why it is needed |
|---|---|---|
| HEAD is a conventional commit | `git log -1` matches Conventional Commits (`feat: ...`, `fix(scope): ...`), enforced by a `commit-msg` hook | release notes, changelogs, and version bumps in later passes are generated from commit types |
| CI is green | with a GitHub remote, the latest run on the default branch concluded `success` (`gh run list`); without one, the local CI script exits 0 | a red build caught at the commit that broke it is cheap; one found three passes later is not |

Both checks keep running in every later pass: a red CI or a non-conventional HEAD is a broken system, whatever else works.

## The local CI script

Without a hosted CI, the same gates run from one script, [`ci-local.sh`](../../../../../course/solutions/entry/scripts/ci-local.sh), in three stages:

1. **commit-lint**: run the repository's `.githooks/commit-msg` hook on the message of HEAD, so the hook and CI enforce the same rule.
2. **native-tests**: build and run the shell/make primer (`make -C primers/lang.02`), clean it again, and import the Python primer modules, so a broken toolchain fails here rather than in a later pass.
3. **the test suite**: run the system's tests. (The script's third stage still calls the old course checker, which has been removed; in your own copy, replace it with the test commands of your build.)

`set -euo pipefail` makes the first failing command fail the script, which is what makes it usable as a gate.
