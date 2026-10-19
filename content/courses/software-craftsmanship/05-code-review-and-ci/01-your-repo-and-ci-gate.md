# The repository and its CI gate

At the start of the project, the repository holds two primers and nothing else, and nothing stops a broken change from landing in it: a commit that breaks the build, a message like "wip" that tells a future reader nothing, a build that only works on one laptop. Soon every change to the system lands here. If tests run only when someone remembers to run them, regressions land silently and are found weeks later, far from the change that caused them. This section makes the repository defend itself, with a CI gate that runs on every push and commit messages that a machine can read.

The design rests on three ideas. Continuous integration is a function from a commit to pass or fail, run by a machine on every push and pull request, and each step passes or fails by its exit status and nothing else. A Conventional Commit header, `type(scope)!: description`, makes history machine-readable, which is what release tools later compute version bumps from. And there is one rule in one file: the same `.githooks/commit-msg` script rejects a bad message on a laptop and in CI, so the two can never disagree. Along the way you will meet the trap that catches most first attempts: a shallow clone hides history, so unless the lint job fetches everything, "every new commit" silently means "the last one".

This section builds on [Shell, git, make, processes](../../programming-languages/02-language-and-tool-primers/02-shell-git-make.md) from *Programming Languages*, for processes, exit codes, and git. You need it as reading only; none of its code is called here. Completing this section is part of [the setup milestone](../../build-an-llm/paths/course-p00-setup/milestone.md) in *Build Your Own LLM System*, which ends when HEAD is a conventional commit and CI is green. The finished solution is three files: [`.githooks/commit-msg`](../../../../course/solutions/entry/.githooks/commit-msg), a Conventional Commits check run locally and in CI; [`.github/workflows/ci.yml`](../../../../course/solutions/entry/.github/workflows/ci.yml), with the jobs `commit-lint` and `native-tests`; and [`scripts/ci-local.sh`](../../../../course/solutions/entry/scripts/ci-local.sh), the same gates for a repository without GitHub. There is no contract file: CI and hooks belong to the repository. Before you read the hook, try writing its regular expression yourself from the grammar in [Conventional Commits](#conventional-commits).

> **Learning objectives**
> - Explain what a CI workflow, job, and step are, and how branch protection turns CI into a gate.
> - Write the Conventional Commits header grammar as a regular expression and say which rule each near miss breaks.
> - Install a tracked `commit-msg` hook and explain why a local hook alone is not enough.
> - Compute the set of commits a push or pull request brings, including on a first push and with a shallow clone.
> - Build a GitHub Actions workflow that lints every new commit and runs the repository's tests.

## What CI is

**Continuous integration** (CI) is a service that watches your repository and, on every push and every pull request, runs a **workflow** written for it. A workflow is a list of **jobs**, each on a fresh machine, and each job is a list of **steps**. A step is either an action someone published (`uses: actions/checkout@v4` clones your repository) or a shell command (`run: make test`). A step fails when its command exits nonzero, as described in [Shell, git, make, processes](../../programming-languages/02-language-and-tool-primers/02-shell-git-make.md); a job fails when a step fails; and the run fails when any job fails. Jobs run in parallel unless one declares `needs:` another.

On its own, CI only reports. With **branch protection**, the hosting service refuses to merge into the default branch until named jobs pass, and that is the moment CI becomes a **gate**. GitHub Actions reads workflows from `.github/workflows/*.yml`. A workflow can read a handful of values about the event that triggered it, and these are the ones this section uses:

| Symbol | Meaning | Type / shape |
|---|---|---|
| `${{ github.sha }}` | the commit the run is for (on a push: the new tip of the branch) | 40-hex string |
| `${{ github.event.before }}` | on a push: the tip of the branch before the push; 40 zeros when the branch is new | 40-hex string |
| `${{ github.event.pull_request.base.sha }}` | on a pull request: the commit of the target branch it is compared with | 40-hex string |
| `${{ github.event.pull_request.head.sha }}` | on a pull request: the newest commit of the proposed branch | 40-hex string |
| `$RUNNER_TEMP` | a scratch directory on the job's machine, emptied after the job | path |

## Conventional Commits

The gate needs something to check besides tests, and the commit message is the cheapest place to start. A commit message has a **header** (its first line), then optionally a blank line and a **body** (free text), then optionally **footers** (`Refs: lang.02`, `BREAKING CHANGE: ...`). The Conventional Commits 1.0.0 format fixes the header's shape:

```text
<type>[(<scope>)][!]: <description>
```

Each part has one rule, summarized here:

| Part | Rule in this book | Example |
|---|---|---|
| type | one of `feat fix docs style refactor perf test build ci chore revert`, lowercase | `feat` |
| scope | optional; one word in parentheses, not empty | `(make)` |
| `!` | optional; marks a breaking change | `feat!:` |
| `: ` | a colon and exactly one space | `: ` |
| description | not empty, starts with a non-space character | `rebuild objects when greet.h changes` |

Written as one regular expression, which is the one the hook applies and the one the setup milestone expects of HEAD:

```text
^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)(\([^()\s]+\))?!?: \S.*$
```

The format pays for itself because tools read it. `feat` means a new capability (a minor version bump), `fix` a bug fix (a patch bump), and `!` or a `BREAKING CHANGE:` footer a major bump. [Commits, semver, and releases](../06-releases-and-deprecation/01-commits-semver-and-releases.md) releases `v1.0.0` from exactly this history. The specification itself lets types be any case; this book and the common commitlint preset accept lowercase only.

## Hooks: checking before the commit exists

The cheapest place to reject a bad message is on the author's laptop, before the commit exists. git runs **hooks**, executable files with fixed names, at fixed moments. The `commit-msg` hook runs after you write a message and before the commit is created. git passes it **one argument, the path of a file holding the message**, and aborts the commit if the hook exits nonzero. That file is the raw editor text: it may still contain git's comment lines (starting with `#`), which git strips only after the hook has run.

By default hooks live in `.git/hooks/`, which git does not track, so a hook kept there is lost on the next clone. Keep it tracked in `.githooks/` instead and point git at it once per clone with `git config core.hooksPath .githooks`. Even then, a local hook is advisory: `git commit --no-verify` skips it, and a fresh clone has no `core.hooksPath`. That is why CI runs the same file again.

## Which commits a push brings

For CI to re-run the hook, it needs to know which commits are new. `git rev-list A..B` lists the commits reachable from `B` that are not reachable from `A`. On a push, that is the commits between the old tip `github.event.before` and the new tip `github.sha`; on a pull request, those between the base and the head.

Two edge cases matter. When a branch is pushed for the first time, `before` is 40 zeros, which names no commit, and the right set is everything reachable from the tip. And `actions/checkout` clones **one commit** by default (a shallow clone), so `before` is not in the clone and the range cannot be computed at all. `fetch-depth: 0` asks for the full history.

## CI without GitHub

A repository with no GitHub remote has no hosted runner. The same gates then run from one local script, `scripts/ci-local.sh`, which lints HEAD's message with the hook and runs the native tests. It exits nonzero as soon as any gate fails, exactly like a CI job.

## Working it by hand

With the pieces in place, walk a message and a push through them before reading any code. This message passes:

```text
feat(make): rebuild objects when greet.h changes

main.o and greet.o both include greet.h, so both depend on it.

Refs: lang.02
```

The header is the first line that is neither blank nor a comment. `feat` is in the type list, `(make)` is a one-word scope, there is no `!`, `: ` follows, and the description starts with `r`. The body and the footer are free text. A release tool would read this commit as a minor bump.

The near misses are more instructive. Each of the following headers breaks exactly one rule of the grammar:

| Header | Broken rule |
|---|---|
| `Add bigram counter` | no type |
| `feat:count bigrams` | no space after the colon |
| `feat : count bigrams` | a space before the colon |
| `feature: count bigrams` | `feature` is not a type |
| `Feat: count bigrams` | uppercase type |
| `fix(): empty scope` | empty scope |
| `fix(make files): spaces in scope` | the scope is two words |
| `feat: ` | empty description |

Now a push. The branch is `A - B - C` on the server, and a push adds `D - E` on top. GitHub sets `before = C` and `sha = E`; `git rev-list C..E` lists `E` and `D`, and the commit-lint job runs the hook on both messages. Had this been the branch's first push, `before` would be 40 zeros and the job would lint `A` through `E`.

The workflow has two jobs, and it helps to fix their ids and responsibilities before writing them:

| Job id | Runs | Fails when |
|---|---|---|
| `commit-lint` | checkout with `fetch-depth: 0`; `.githooks/commit-msg` on every new commit's message | any message is not a Conventional Commit |
| `native-tests` | the repository's own tests with each language's tool; at first, build and run the primers | any test command exits nonzero |

## The artifacts

The solution is three files:

| File | Holds |
|---|---|
| `.githooks/commit-msg` | an executable bash script: reads the file named by its first argument, takes the first line that is neither blank nor a comment as the header, exits 0 when it matches the grammar, otherwise prints the rule and an example on stderr and exits 1 |
| `.github/workflows/ci.yml` | `on:` both `push` and `pull_request`; the two jobs above, each with `runs-on: ubuntu-latest` |
| `scripts/ci-local.sh` | the same gates, run locally |

The hook is enabled per clone with `git config core.hooksPath .githooks`; after that, `git commit --allow-empty -m "wip"` must be refused. The first commit that adds these files is itself conventional, for example `ci: gate pushes on commit lint and native tests`.

## Building it step by step

Six observations fix the design before any code is written.

1. **The rule must live in one place.** If the laptop and CI each had their own check, they would drift. So the rule is one executable file, run by git locally and by CI over every pushed commit.
2. **git hands the hook a file, not a string, and the file is raw editor text.** So the hook must drop `#` comment lines and blank lines, then take the first remaining line as the header.
3. **A regular expression is the right tool**, because the header grammar is regular: a fixed alternation of types, an optional parenthesized token with no spaces, an optional `!`, a literal `: `, and a non-space start.
4. **Local hooks are advisory** (`--no-verify`, fresh clones). So CI must re-run the same hook on every commit the push or pull request brings, not just the tip.
5. **The commit range needs history and has an edge case.** Computing `BASE..HEAD` requires `fetch-depth: 0`, and a first push has no valid base, so the job must fall back to all commits reachable from HEAD.
6. **Exit status is the only signal.** Every script uses `set -euo pipefail`, so a failed command fails the step instead of being ignored.

### The hook

```bash
#!/usr/bin/env bash
set -euo pipefail
msg_file="${1:?usage: commit-msg <message file>}"   # git passes the message file; refuse to run without it

# The header is the first line that is neither a comment (git strips lines
# starting with '#') nor blank.
header="$(grep -v '^#' "$msg_file" | sed '/^[[:space:]]*$/d' | head -n 1 || true)"
#        drop comments        drop blank lines            first line   an empty message gives "" and fails below

types='feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert'
# ( type )( optional (scope) with no spaces or parens )( optional ! ): then a non-space
pattern="^($types)(\([^()[:space:]]+\))?!?: [^[:space:]].*$"
if [[ "$header" =~ $pattern ]]; then     # unquoted $pattern: bash treats it as an ERE
  exit 0
fi
{
  echo "commit-msg: not a Conventional Commits header: '$header'"
  echo "  want: <type>[(scope)][!]: <description>"
  echo "  type: ${types//|/ }"
  echo "  e.g.: feat(make): rebuild objects when greet.h changes"
} >&2                                    # stderr: git shows it when it refuses the commit
exit 1
```

Each near miss from the hand-worked table fails one part of the pattern: `feat:count` fails the `: ` literal, `fix()` the `+` in the scope (at least one character), `fix(make files)` the `[^()[:space:]]` class, `Feat` the lowercase alternation, and `feat: ` the `[^[:space:]]`. The `|| true` matters under `pipefail`: `grep -v` exits 1 when every line is a comment, and without it the hook would die before printing its message. The pattern uses `[[:space:]]` rather than `\s` because bash's `=~` uses POSIX EREs, which have no `\s`. The full hook is in [`commit-msg`](../../../../course/solutions/entry/.githooks/commit-msg).

### The workflow

```yaml
on:
  push:
  pull_request:                 # both: a PR from a fork never triggers "push" here

jobs:
  commit-lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0          # the whole history: a shallow clone holds one commit
      - name: every new commit message is a Conventional Commit
        env:
          # a pull request compares base..head; a push compares before..sha
          BASE: ${{ github.event.pull_request.base.sha || github.event.before }}
          HEAD: ${{ github.event.pull_request.head.sha || github.sha }}
        run: |
          set -euo pipefail
          # 40 zeros (a new branch) or an unknown sha (a force-push) is not a commit here
          if [ -z "$BASE" ] || ! git cat-file -e "$BASE^{commit}" 2>/dev/null; then
            commits="$(git rev-list "$HEAD")"        # first push of a branch: all of it
          else
            commits="$(git rev-list "$BASE..$HEAD")"
          fi
          for c in $commits; do
            git log -1 --format=%B "$c" > "$RUNNER_TEMP/msg"   # the full message, as git would pass it
            .githooks/commit-msg "$RUNNER_TEMP/msg" || { echo "in commit $c" >&2; exit 1; }
          done

  native-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
      # One step per language as it is started: `uv run --project python pytest
      # python/tests`, `make -C c test`, `cargo test --manifest-path
      # rust/Cargo.toml`, `cd go && go test ./...`.
      - name: C primer builds and runs
        run: make -C primers/lang.02 && primers/lang.02/greet ci
      - name: Python primer imports inside its uv project
        run: cd primers/lang.01 && uv run python -c "import bigram, worksheet"
```

`git cat-file -e "$BASE^{commit}"` handles both edge cases with one test: 40 zeros and a sha missing after a force-push both fail it, and the job lints everything reachable instead of crashing. Naming the failing commit (`in commit $c`) turns a red job into a fix. The two jobs have no `needs:`, so they run in parallel, and a lint failure does not hide a test failure. The full workflow is in [`ci.yml`](../../../../course/solutions/entry/.github/workflows/ci.yml). Its local equivalent, [`ci-local.sh`](../../../../course/solutions/entry/scripts/ci-local.sh), lints HEAD with the same hook, then builds the C primer, cleans it, and imports the Python primer.

### Why this design

A few alternatives were considered and rejected. **commitlint** needs Node and a config file; a 30-line bash script needs nothing a fresh clone lacks and is the same file in CI. The cost is fewer rules (no header length limit, for instance), which is acceptable because the format is the only rule that matters here.

**Linting only HEAD** would be simpler than linting every new commit, but a pull request with five commits and one bad message would then merge a non-conventional commit into history and break release tooling later.

**Exit codes rather than parsed output** keep every gate the same shape: "did the command exit 0?" Adding a language is one `run:` line.

Finally, the repository settings should use **squash or rebase merging**. A merge commit's message, `Merge pull request #3 ...`, is not conventional, and with squash or rebase HEAD always passes.

## What can go wrong

A correct solution handles each of the following cases. The table says what kind of check catches each one and why it matters later:

| Case | Kind of check | Why it matters downstream |
|---|---|---|
| the hook has the execute bit and a `#!` line | unit | git silently skips a hook it cannot exec |
| the hand-worked message passes, body and footer included | unit | the hook and the format agree |
| plain, scoped, `!`, scoped with `!` headers, and every type, pass | unit | valid history is never blocked |
| every near miss fails, with a reason on stderr | boundary | a near miss fails here, not later on HEAD |
| comment lines are skipped, only the header decides, an empty message fails | boundary | git's editor file is not the final message |
| HEAD passes the hook | conformance | the gate has judged real history |
| the workflow runs on both `push` and `pull_request` | unit | every change is checked before it merges |
| `commit-lint` and `native-tests` exist, each with `runs-on` and steps | unit | branch protection requires them by name |
| `commit-lint` uses `fetch-depth: 0` and runs `.githooks/commit-msg` | unit | every pushed commit is linted, by one rule |
| `native-tests` has at least one `run:` step | unit | the repository's tests run on every push |

The common mistakes map onto those cases.

> **Warning:** A loose pattern such as `^[a-z]+:` lets `feat:count`, `feature: x`, and `fix(): x` through. Use the full pattern: fixed types, a non-empty scope, `: `, and a non-space.

Matching the whole file, or a comment line, as the header produces two opposite failures: a valid commit is refused because of git's comment text, or a `feat:` hidden in the body passes. Drop `#` lines and blank lines, then take `head -n 1`.

Triggering only on pushes to `main` lets pull requests merge without a run; trigger on both `push` and `pull_request`. The default shallow checkout in `commit-lint` makes `git rev-list` fail on the missing `before` commit, or lints only the last commit; set `fetch-depth: 0`.

> **Warning:** A hook without the execute bit is skipped by git without a word, so every message "passes". Commit the file executable, with a `#!` line.

Merging pull requests with a merge commit makes HEAD `Merge pull request #3 ...`, which is not a Conventional Commit. Use squash or rebase merging.

## Where this leads

This gate looks back to [Shell, git, make, processes](../../programming-languages/02-language-and-tool-primers/02-shell-git-make.md): exit codes decide every step, hooks are scripts, and `git rev-list` and `git log` walk history. Every later change lands through it. The first runtime code of the system is the first code whose tests `native-tests` runs, and from then on the repository's own tests are judged by mutation testing, as described in [TDD, unit tests, and mutation testing](../03-testing-mentality/01-tdd-unit-tests-and-mutation-grading.md), with `native-tests` running them on every push. [CI for the platform repo](../../infrastructure/01-containers-kubernetes/06-ci-for-your-repo.md) in *Infrastructure* extends this workflow with an image build, a kind end-to-end run (`tilt ci`), and the perf gate. And [Commits, semver, and releases](../06-releases-and-deprecation/01-commits-semver-and-releases.md) releases `v1.0.0` and its changelog from the conventional history.

Each piece here has a production equivalent worth knowing:

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `.githooks/commit-msg` | commitlint with `@commitlint/config-conventional` | configurable rules (header length, case, allowed scopes) and editor integration | conventional-changelog/commitlint, `@commitlint/config-conventional` |
| `core.hooksPath .githooks` | the pre-commit framework | hooks pinned by version and installed per clone with one command | pre-commit.com |
| conventional history | release-please, semantic-release | the next version and the changelog computed from commit types | googleapis/release-please `README.md` |
| the `commit-lint` job | GitHub rulesets and required status checks | the merge button stays disabled until the named jobs pass | docs.github.com, "About protected branches" |
| `scripts/ci-local.sh` | nektos/act | runs the GitHub workflow itself in local containers | nektos/act `README.md` |

### Further reading

- Conventional Commits 1.0.0, conventionalcommits.org (free).
- *Software Engineering at Google*, ch. 23, "Continuous Integration", abseil.io/resources/swe-book (free).
- GitHub Actions, "Workflow syntax", docs.github.com (free).
- `man 5 githooks`.

## Key terms

- **Continuous integration (CI)**: a service that runs a workflow of jobs and steps on every push and pull request, passing or failing by exit status.
- **Branch protection**: a hosting rule that blocks merges into a branch until named CI jobs pass, turning CI into a gate.
- **Conventional Commit**: a commit whose header matches `type(scope)!: description`, so tools can derive version bumps and changelogs.
- **commit-msg hook**: an executable git runs with the path of the message file before creating a commit; a nonzero exit aborts the commit.
- **`core.hooksPath`**: the git setting that points hook lookup at a tracked directory such as `.githooks/`.
- **Shallow clone**: a checkout with truncated history (one commit by default in `actions/checkout`), which makes commit ranges uncomputable.

## Summary

- CI is a function from a commit to pass or fail; branch protection turns it into a gate.
- One rule, one implementation: the hook you run locally is the step CI runs.
- Conventional commit headers make history machine-readable, which is what releases are computed from.
- CI must lint every commit a push or pull request brings, which needs `fetch-depth: 0` and a fallback for a first push or a force-push.
- Exit status is the only signal; `set -euo pipefail` and `|| true` where an empty result is legitimate keep it honest.
- Keep the gate strict from the first commit: loosening it later is easy, while tightening it means rewriting history.

## Review questions

1. Why does the hook need `|| true` after the pipeline that extracts the header?
   <details><summary>Answer</summary>With `set -o pipefail`, `grep -v '^#'` exiting 1 (no non-comment lines) fails the whole assignment and `set -e` kills the script silently. `|| true` lets the header be empty so the hook reaches the clear error message and exits 1 itself.</details>
2. What goes wrong with the default `actions/checkout` in `commit-lint`?
   <details><summary>Answer</summary>It clones one commit, so `before` is not in the clone: `git cat-file` fails, and the job falls back to `git rev-list "$HEAD"`, which in a one-commit clone is just HEAD. Only the last commit is linted, and bad messages earlier in the push pass.</details>
3. Why both `push` and `pull_request` triggers?
   <details><summary>Answer</summary>A push runs on branches in the repository; a pull request (especially from a fork) is the change about to merge, compared against its base. Branch protection needs the PR run to gate the merge.</details>
4. Why is `fix(): empty scope` rejected by the pattern?
   <details><summary>Answer</summary>The scope group is `\([^()[:space:]]+\)`: the `+` requires at least one character between the parentheses.</details>
5. Why does CI run the same hook file instead of its own check?
   <details><summary>Answer</summary>Two implementations of one rule drift apart. A local hook is also advisory (`--no-verify` skips it and fresh clones lack `core.hooksPath`), so CI must enforce the rule, and running the identical file guarantees the laptop and CI agree.</details>
