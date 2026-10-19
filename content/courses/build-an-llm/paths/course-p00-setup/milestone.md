# What the System Does After Pass 0

There is no model yet. What exists after Pass 0 is a repository whose history and CI you can trust, and every later pass depends on that. Each pass adds code in four languages, and the only way to know a change did not break an earlier layer is a CI run on every commit.

## What runs

Two checks define the end of this pass. The first is that HEAD is a conventional commit: `git log -1` matches the Conventional Commits format (`feat: ...`, `fix(scope): ...`), and a `commit-msg` hook enforces it. This matters later, because release notes, changelogs, and version bumps in later passes are generated from commit types. The second is that CI is green. With a GitHub remote, the latest run on the default branch has concluded `success` (`gh run list` shows it); without one, the local CI script exits 0. A red build caught at the commit that broke it is cheap to fix; one found three passes later is not.

Both checks keep running in every later pass. A red CI or a non-conventional HEAD is a broken system, whatever else works.

## The local CI script

Without a hosted CI, the same gates run from one script, [`ci-local.sh`](../../../../../course/solutions/entry/scripts/ci-local.sh), in two stages:

1. **commit-lint** runs the repository's `.githooks/commit-msg` hook on the message of HEAD, so the hook and CI enforce the same rule.
2. **native-tests** builds and runs the shell and make primer (`make -C primers/lang.02`), cleans it again, and imports the Python primer modules, so a broken toolchain fails here rather than in a later pass.

Each language's own test command (`pytest`, `cargo test`, `go test ./...`, `make test`) joins the native-tests stage as that language enters the system. The script starts with `set -euo pipefail`, which makes the first failing command fail the whole script; that is what makes it usable as a gate. The sections behind it are [The repository and its CI gate](../../../software-craftsmanship/05-code-review-and-ci/01-your-repo-and-ci-gate.md) and [Shell, git, make, and processes](../../../programming-languages/02-language-and-tool-primers/02-shell-git-make.md).

[Pass 1](../course-p01-tracer/README.md) puts the first code in four languages through this gate.
