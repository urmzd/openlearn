# The repository and its CI gate (conventional commits enforced)

## Overview

| | |
|---|---|
| **Module** | `craft.01` · practice · ops · Pass 0 · 1 to 2 h reading |
| **Solution** | [`.githooks/commit-msg`](../../../../course/solutions/entry/.githooks/commit-msg) (a Conventional Commits check run locally and in CI), [`.github/workflows/ci.yml`](../../../../course/solutions/entry/.github/workflows/ci.yml) (the jobs `commit-lint` and `native-tests`), and [`scripts/ci-local.sh`](../../../../course/solutions/entry/scripts/ci-local.sh) (the same gates for a repository without GitHub) |
| **Contract** | none: CI and hooks belong to the repository |
| **Needs** | [`lang.02`](../../programming-languages/02-language-and-tool-primers/02-shell-git-make.md) for processes, exit codes, and git (a reading prerequisite: nothing of it is called) |
| **Used by** | no code call site: every later change lands through this gate. `dep.05` extends this CI (image build, kind e2e, perf gate) and `craft.11` adds the release workflow. Part of [the Pass 0 setup](../../../courses/build-an-llm/paths/course-p00-setup/milestone.md): a conventional HEAD and a green CI |
| **Optional depth** | Conventional Commits 1.0.0 (conventionalcommits.org, free); *Software Engineering at Google*, ch. 23, "Continuous Integration" (abseil.io/resources/swe-book, free); GitHub Actions, "Workflow syntax" (docs.github.com, free); `man 5 githooks` |

## Key Takeaways

- **CI is a function from a commit to pass or fail**, run by a machine on every push and pull request. Each step passes or fails by its exit status, nothing else.
- **A Conventional Commit header** `type(scope)!: description` makes history machine-readable: release tools derive version bumps from it (craft.11).
- **One rule, one file.** The same `.githooks/commit-msg` rejects a bad message on a laptop and in CI, so the two can never disagree.
- **A shallow clone hides history.** The commit-lint job fetches everything, or "every new commit" silently means "the last one".

## How to read this chapter

Sections 1 to 3 explain CI, the commit format, hooks, and commit ranges, and work a message and a push by hand. Section 4 states the artifacts. Section 5 walks through the hook and the workflow line by line and explains why each line is there. Before reading the hook, write the regular expression yourself from section 2.2.

---

## 1. Why now

At this point the repository holds two primers. Nothing stops a broken change from landing in it: a commit that breaks a build, a message like "wip" that tells a future reader nothing, a build that only works on one laptop. From Pass 1 on, every change to the system lands here. If tests run only when someone remembers to run them, regressions land silently and are found three passes later. This module makes the repository defend itself: a CI gate that runs on every push, and commit messages a machine can read.

## 2. Principles

### 2.1 What CI is

**Continuous integration** (CI) is a service that watches your repository and, on every push and every pull request, runs a **workflow** written for it: a list of **jobs**, each on a fresh machine, each a list of **steps**. A step is either an action someone published (`uses: actions/checkout@v4` clones your repository) or a shell command (`run: make test`). A step fails when its command exits nonzero (lang.02, 2.1), a job fails when a step fails, and the run fails when any job fails. Jobs run in parallel unless one declares `needs:` another. With **branch protection**, the hosting service refuses to merge into the default branch until named jobs pass: that is when CI becomes a **gate**. GitHub Actions reads workflows from `.github/workflows/*.yml`. The values a workflow can read about its event:

| Symbol | Meaning | Type / shape |
|---|---|---|
| `${{ github.sha }}` | the commit the run is for (on a push: the new tip of the branch) | 40-hex string |
| `${{ github.event.before }}` | on a push: the tip of the branch before the push; 40 zeros when the branch is new | 40-hex string |
| `${{ github.event.pull_request.base.sha }}` | on a pull request: the commit of the target branch it is compared with | 40-hex string |
| `${{ github.event.pull_request.head.sha }}` | on a pull request: the newest commit of the proposed branch | 40-hex string |
| `$RUNNER_TEMP` | a scratch directory on the job's machine, emptied after the job | path |

### 2.2 Conventional Commits

A commit message has a **header** (its first line), then optionally a blank line and a **body** (free text), then optionally **footers** (`Refs: lang.02`, `BREAKING CHANGE: ...`). The Conventional Commits 1.0.0 format fixes the header's shape:

```text
<type>[(<scope>)][!]: <description>
```

| Part | Rule in this course | Example |
|---|---|---|
| type | one of `feat fix docs style refactor perf test build ci chore revert`, lowercase | `feat` |
| scope | optional; one word in parentheses, not empty | `(make)` |
| `!` | optional; marks a breaking change | `feat!:` |
| `: ` | a colon and exactly one space | `: ` |
| description | not empty, starts with a non-space character | `rebuild objects when greet.h changes` |

As one regular expression (the one the hook applies, and the one the Pass 0 setup expects of HEAD):

```text
^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)(\([^()\s]+\))?!?: \S.*$
```

The format pays for itself because tools read it: `feat` means a new capability (a minor version bump), `fix` a bug fix (a patch bump), `!` or a `BREAKING CHANGE:` footer a major bump; craft.11 releases `v1.0.0` from exactly this history. The specification itself lets types be any case; this course and the common commitlint preset accept lowercase only.

### 2.3 Hooks

git runs **hooks**, executable files with fixed names, at fixed moments. The `commit-msg` hook runs after you write a message and before the commit exists; git passes it **one argument, the path of a file holding the message**, and aborts the commit if the hook exits nonzero. That file is the raw editor text: it may still contain git's comment lines (starting with `#`), which git strips only after the hook has run. By default hooks live in `.git/hooks/`, which git does not track, so a hook kept there is lost on the next clone. Keep it tracked in `.githooks/` and point git at it once per clone with `git config core.hooksPath .githooks`. A local hook is advisory (`git commit --no-verify` skips it, and a fresh clone has no `core.hooksPath`), which is why CI runs the same file again.

### 2.4 Which commits a push brings

`git rev-list A..B` lists the commits reachable from `B` that are not reachable from `A`: on a push, the commits between the old tip `github.event.before` and the new tip `github.sha`; on a pull request, those between the base and the head. When a branch is pushed for the first time, `before` is 40 zeros, which names no commit, and the right set is everything reachable from the tip. One more trap: `actions/checkout` clones **one commit** by default (a shallow clone), so `before` is not in the clone and the range cannot be computed. `fetch-depth: 0` asks for the full history.

### 2.5 CI without GitHub

A repository with no GitHub remote has no hosted runner. The same gates then run from one local script, `scripts/ci-local.sh`, which lints HEAD's message with the hook and runs the native tests; it exits nonzero as soon as any gate fails, exactly like a CI job.

## 3. Worked example by hand

**A message, parsed.** This message passes:

```text
feat(make): rebuild objects when greet.h changes

main.o and greet.o both include greet.h, so both depend on it.

Refs: lang.02
```

The header is the first line that is neither blank nor a comment. `feat` is in the type list; `(make)` is a one-word scope; there is no `!`; `: ` follows; the description starts with `r`. The body and the footer are free text. A release tool would read it as a minor bump.

**Near misses**, each breaking exactly one rule of 2.2:

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

**The commits a push brings.** The branch is `A - B - C` on the server, and a push adds `D - E` on top. GitHub sets `before = C` and `sha = E`; `git rev-list C..E` lists `E` and `D`, and the commit-lint job runs the hook on both messages. Had this been the branch's first push, `before` would be 40 zeros and the job lints `A` to `E`.

**The two jobs**, with their ids:

| Job id | Runs | Fails when |
|---|---|---|
| `commit-lint` | checkout with `fetch-depth: 0`; `.githooks/commit-msg` on every new commit's message | any message is not a Conventional Commit |
| `native-tests` | the repository's own tests with each language's tool; in Pass 0, build and run the primers | any test command exits nonzero |

## 4. The artifacts

| File | Holds |
|---|---|
| `.githooks/commit-msg` | an executable bash script: reads the file named by its first argument, takes the first line that is neither blank nor a comment as the header, exits 0 when it matches 2.2, otherwise prints the rule and an example on stderr and exits 1 |
| `.github/workflows/ci.yml` | `on:` both `push` and `pull_request`; the two jobs of section 3, each with `runs-on: ubuntu-latest` |
| `scripts/ci-local.sh` | the same gates, run locally (2.5) |

The hook is enabled per clone with `git config core.hooksPath .githooks`; then `git commit --allow-empty -m "wip"` must be refused. The first commit that adds these files is itself conventional, for example `ci: gate pushes on commit lint and native tests`.

### What a correct solution must handle

| Case | Kind of check | Why it matters downstream |
|---|---|---|
| the hook has the execute bit and a `#!` line | unit | git silently skips a hook it cannot exec |
| the section 3 message passes, body and footer included | unit | the hook and the format agree |
| plain, scoped, `!`, scoped with `!` headers, and every type, pass | unit | valid history is never blocked |
| every near miss of section 3 fails, with a reason on stderr | boundary | a near miss fails here, not later on HEAD |
| comment lines are skipped, only the header decides, an empty message fails | boundary | git's editor file is not the final message |
| HEAD passes the hook | conformance | the gate has judged real history |
| the workflow runs on both `push` and `pull_request` | unit | every change is checked before it merges |
| `commit-lint` and `native-tests` exist, each with `runs-on` and steps | unit | branch protection requires them by name |
| `commit-lint` uses `fetch-depth: 0` and runs `.githooks/commit-msg` | unit | every pushed commit is linted, by one rule |
| `native-tests` has at least one `run:` step | unit | the repository's tests run on every push |

## 5. The solution, step by step

### 5.1 From first principles

1. **The rule must live in one place.** If the laptop and CI each had their own check, they would drift. So the rule is one executable file, run by git locally and by CI over every pushed commit.
2. **git hands the hook a file, not a string, and the file is raw editor text.** So the hook must drop `#` comment lines and blank lines, then take the first remaining line as the header.
3. **A regular expression is the right tool** because the header grammar of 2.2 is regular: a fixed alternation of types, an optional parenthesized token with no spaces, an optional `!`, a literal `: `, and a non-space start.
4. **Local hooks are advisory** (`--no-verify`, fresh clones). So CI must re-run the same hook, on every commit the push or pull request brings, not just the tip.
5. **The commit range needs history and has an edge case.** Computing `BASE..HEAD` requires `fetch-depth: 0`; a first push has no valid base, so the job must fall back to all commits reachable from HEAD.
6. **Exit status is the only signal.** Every script uses `set -euo pipefail`, so a failed command fails the step instead of being ignored.

### 5.2 The hook

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

Each near miss of section 3 fails one part: `feat:count` the `: ` literal, `fix()` the `+` in the scope (at least one character), `fix(make files)` the `[^()[:space:]]` class, `Feat` the lowercase alternation, `feat: ` the `[^[:space:]]`. The `|| true` matters under `pipefail`: `grep -v` exits 1 when every line is a comment, and without it the hook would die before printing its message. `[[:space:]]` is used because bash's `=~` uses POSIX EREs, which have no `\s`. Full hook: [`commit-msg`](../../../../course/solutions/entry/.githooks/commit-msg).

### 5.3 The workflow

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

`git cat-file -e "$BASE^{commit}"` handles both edge cases with one test: 40 zeros and a sha missing after a force-push both fail it, and the job lints everything reachable instead of crashing. Naming the failing commit (`in commit $c`) turns a red job into a fix. The two jobs have no `needs:`, so they run in parallel and a lint failure does not hide a test failure. Full workflow: [`ci.yml`](../../../../course/solutions/entry/.github/workflows/ci.yml); local equivalent: [`ci-local.sh`](../../../../course/solutions/entry/scripts/ci-local.sh), which lints HEAD with the same hook, then builds the C primer, cleans it, and imports the Python primer.

### 5.4 Why this solution

- **A bash hook versus commitlint.** commitlint needs Node and a config; a 30-line bash script needs nothing a fresh clone lacks and is the same file in CI. The cost is fewer rules (no header length limit), accepted because the format is the only rule that matters here.
- **Linting every new commit versus only HEAD.** A pull request with five commits and one bad message would otherwise merge a non-conventional commit into history, breaking release tooling later.
- **Exit codes versus parsing output.** Every gate is "did the command exit 0?", so adding a language is one `run:` line.
- **Squash or rebase merging.** A merge commit's message, `Merge pull request #3 ...`, is not conventional; the repository settings should use squash or rebase so HEAD always passes.

### 5.5 Explain it back

1. Why does the hook need `|| true` after the pipeline that extracts the header?
   <details><summary>Answer</summary>With `set -o pipefail`, `grep -v '^#'` exiting 1 (no non-comment lines) fails the whole assignment and `set -e` kills the script silently. `|| true` lets the header be empty so the hook reaches the clear error message and exits 1 itself.</details>
2. What goes wrong with the default `actions/checkout` in `commit-lint`?
   <details><summary>Answer</summary>It clones one commit, so `before` is not in the clone: `git cat-file` fails, and the job falls back to `git rev-list "$HEAD"`, which in a one-commit clone is just HEAD. Only the last commit is linted, and bad messages earlier in the push pass.</details>
3. Why both `push` and `pull_request` triggers?
   <details><summary>Answer</summary>A push runs on branches in the repository; a pull request (especially from a fork) is the change about to merge, compared against its base. Branch protection needs the PR run to gate the merge.</details>
4. Why is `fix(): empty scope` rejected by the pattern?
   <details><summary>Answer</summary>The scope group is `\([^()[:space:]]+\)`: the `+` requires at least one character between the parentheses.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. A loose pattern such as `^[a-z]+:` | `feat:count`, `feature: x`, and `fix(): x` pass the hook | the full pattern of 5.2: fixed types, non-empty scope, `: ` and a non-space |
| 2. Matching the whole file, or a comment line, as the header | a valid commit is refused because of git's comment text, or "feat:" hidden in the body passes | drop `#` lines and blank lines, then `head -n 1` |
| 3. Triggering only on pushes to `main` | pull requests merge without a run | `on: push` and `pull_request` |
| 4. The default shallow checkout in `commit-lint` | `git rev-list` fails on the missing `before` commit, or only the last commit is linted | `fetch-depth: 0` |
| 5. A hook without the execute bit | git skips it without a word: every message "passes" | the file is committed executable with a `#!` line |
| 6. Merging pull requests with a merge commit | HEAD becomes `Merge pull request #3 ...`, which is not a Conventional Commit | squash or rebase merging |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `lang.02` | exit codes decide every step; hooks are scripts; `git rev-list` and `git log` walk history |
| Forward | `rt.01` | the first Pass 1 code whose tests `native-tests` runs |
| Forward | `craft.03` | from Pass 2 the repository's own tests are judged by mutation testing; `native-tests` runs them on every push |
| Forward | `dep.05` | extends this workflow: image build, kind end to end (`tilt ci`), the perf gate |
| Forward | `craft.11` | releases `v1.0.0` and its changelog from the conventional history |

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `.githooks/commit-msg` | commitlint with `@commitlint/config-conventional` | configurable rules (header length, case, allowed scopes) and editor integration | conventional-changelog/commitlint, `@commitlint/config-conventional` |
| `core.hooksPath .githooks` | the pre-commit framework | hooks pinned by version and installed per clone with one command | pre-commit.com |
| conventional history | release-please, semantic-release | the next version and the changelog computed from commit types | googleapis/release-please `README.md` |
| the `commit-lint` job | GitHub rulesets and required status checks | the merge button stays disabled until the named jobs pass | docs.github.com, "About protected branches" |
| `scripts/ci-local.sh` | nektos/act | runs the GitHub workflow itself in local containers | nektos/act `README.md` |
