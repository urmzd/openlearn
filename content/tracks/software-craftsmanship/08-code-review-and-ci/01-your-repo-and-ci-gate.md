<!-- ol:module craft.01 -->
# Your repo and CI gate (conventional commits enforced)

## Overview

| | |
|---|---|
| **Module** | `craft.01` · practice · ops · Pass 0 · 2 to 3 h |
| **You build** | in your course repo: `.githooks/commit-msg` (a Conventional Commits check you run locally and in CI), `.github/workflows/ci.yml` with the jobs `commit-lint`, `native-tests`, and `course-check`, and your first conventional commit, pushed and green |
| **Contract** | none: CI and hooks are your territory (D16). `ol course ci` prints the exact recipe for the `course-check` job |
| **Tests** | `course/tests/craft.01/`, run by its `check` script (what they check: section 4) |
| **Needs** | [`lang.02`](../12-language-and-tool-primers/02-shell-git-make.md) for processes, exit codes, and git (a reading prerequisite: nothing of it is called) |
| **Used by** | no code call site: every later module is graded through this gate, because from Pass 1 `ol check --all --ci` checks each started module on every push. `dep.05` extends this CI (image build, kind e2e, perf gate) and `craft.11` adds the release workflow |
| **Milestone** | MS-P0 (page: `content/courses/build-an-llm/paths/course-p00-setup/milestone.md`) |
| **Optional depth** | Conventional Commits 1.0.0 (conventionalcommits.org, free); *Software Engineering at Google*, ch. 23, "Continuous Integration" (abseil.io/resources/swe-book, free); GitHub Actions, "Workflow syntax" (docs.github.com, free); `man 5 githooks` |

## Key Takeaways

- **CI is a function from a commit to pass or fail**, run by a machine on every push and pull request. Each step passes or fails by its exit status, nothing else.
- **A Conventional Commit header** `type(scope)!: description` makes history machine-readable: release tools derive version bumps from it (craft.11), and MS-P0 checks your HEAD with it.
- **One rule, one file.** The same `.githooks/commit-msg` rejects a bad message on your laptop and in CI, so the two can never disagree.
- **A shallow clone hides history.** The commit-lint job fetches everything, or "every new commit" silently means "the last one".
- **The course gate runs the tests your contracts came with**: CI checks out openlearn at the commit `contracts/VERSION` names, never at its latest commit.

## How to work this chapter

```bash
ol start craft.01                # records the start; there is no stub: CI and hooks are yours
ol course ci                     # prints the course-check recipe
ol tests craft.01                # read the test catalog first
ol check craft.01                # exit code is the verdict
ol milestone MS-P0               # the Pass 0 gate: lang.01, lang.02, craft.01, a conventional HEAD, green CI
```

---

## 1. Why now

So far your repository is a directory that `ol course init` made and you filled with two primers. Nothing stops a broken change from landing in it: a commit that fails `ol check`, a message like "wip" that tells your future self nothing, a build that only works on your laptop. From Pass 1 on, every change you make to your system lands here, and every module you start is checked by `ol check --all --ci`. If that check runs only when you remember to run it, regressions land silently and you find them three passes later. This module makes the repository defend itself: a CI gate that runs on every push, and commit messages a machine can read.

## 2. Principles

### 2.1 What CI is

**Continuous integration** (CI) is a service that watches your repository and, on every push and every pull request, runs a **workflow** you wrote: a list of **jobs**, each on a fresh machine, each a list of **steps**. A step is either an action someone published (`uses: actions/checkout@v4` clones your repository) or a shell command (`run: make test`). A step fails when its command exits nonzero (lang.02, 2.1), a job fails when a step fails, and the run fails when any job fails. Jobs run in parallel unless one declares `needs:` another. With **branch protection**, the hosting service refuses to merge into the default branch until named jobs pass: that is when CI becomes a **gate**. GitHub Actions reads workflows from `.github/workflows/*.yml`. The values a workflow can read about its event:

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

As one regular expression (the one MS-P0's `git-log` check applies to your HEAD):

```text
^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)(\([^()\s]+\))?!?: \S.*$
```

The format pays for itself because tools read it: `feat` means a new capability (a minor version bump), `fix` a bug fix (a patch bump), `!` or a `BREAKING CHANGE:` footer a major bump; craft.11 releases your `v1.0.0` from exactly this history. The specification itself lets types be any case; this course, the common commitlint preset, and MS-P0 accept lowercase only.

### 2.3 Hooks

git runs **hooks**, executable files with fixed names, at fixed moments. The `commit-msg` hook runs after you write a message and before the commit exists; git passes it **one argument, the path of a file holding the message**, and aborts the commit if the hook exits nonzero. That file is the raw editor text: it may still contain git's comment lines (starting with `#`), which git strips only after the hook has run. By default hooks live in `.git/hooks/`, which git does not track, so a hook you write there is lost on the next clone. Keep it tracked in `.githooks/` and point git at it once per clone with `git config core.hooksPath .githooks`. A local hook is advisory (`git commit --no-verify` skips it, and a fresh clone has no `core.hooksPath`), which is why CI runs the same file again.

### 2.4 Which commits a push brings

`git rev-list A..B` lists the commits reachable from `B` that are not reachable from `A`: on a push, the commits between the old tip `github.event.before` and the new tip `github.sha`; on a pull request, those between the base and the head. When a branch is pushed for the first time, `before` is 40 zeros, which names no commit, and the right set is everything reachable from the tip. One more trap: `actions/checkout` clones **one commit** by default (a shallow clone), so `before` is not in the clone and the range cannot be computed. `fetch-depth: 0` asks for the full history.

### 2.5 The course gate

Your repository does not contain openlearn; it contains `contracts/`, vendored at one openlearn commit, and `contracts/VERSION` names that commit's sha. The course tests that grade you must be the ones your contracts came with, so the `course-check` job clones openlearn, checks out **that** sha, and runs `ol check --all --ci` against your checkout (`OL_COURSE_HOME` points `ol` at it). `--ci` refuses `--ref-deps` and skips interactive rubrics: CI grades only your own code. A module counts as started in CI when one of its units differs from its stub, or when a file it grades exists (`primers/<ID>/`, `.githooks/commit-msg`, `.github/workflows/ci.yml`; the registry calls these its `artifacts`), because the ledger in `.ol/` is not committed. So in Pass 0 the gate checks `lang.01`, `lang.02`, and `craft.01`; from Pass 1 it checks every module whose files you have written.

### 2.6 CI without GitHub

If your repository has no GitHub remote, MS-P0 cannot ask GitHub for your last run. Its `ci-status` check then runs the commands you list under `[ci]` in `system.toml`, and each must exit 0:

```toml
[ci]
local = [["bash", "scripts/ci-local.sh"]]     # the same three gates, run on your machine
```

The script lints HEAD's message with your hook, runs your native tests, and runs `ol check --all --ci`. Your repository does not know where your openlearn clone is, so the script reads it from an environment variable you set once, `OL_BIN` (for example `export OL_BIN=~/src/openlearn/practice/bin/ol`), and otherwise uses the clone the CI recipe makes in `.ol/openlearn`.

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

**The commits a push brings.** Your branch is `A - B - C` on the server, and you push `D - E` on top. GitHub sets `before = C` and `sha = E`; `git rev-list C..E` lists `E` and `D`, and the commit-lint job runs your hook on both messages. Had this been the branch's first push, `before` would be 40 zeros and the job lints `A` to `E`.

**The three jobs**, with the ids the tests look for:

| Job id | Runs | Fails when |
|---|---|---|
| `commit-lint` | checkout with `fetch-depth: 0`; `.githooks/commit-msg` on every new commit's message | any message is not a Conventional Commit |
| `native-tests` | your own tests with each language's tool; in Pass 0, build and run your primers | any test command exits nonzero |
| `course-check` | the `ol course ci` recipe: clone openlearn, check out the `contracts/VERSION` sha, `ol check --all --ci` | any started module fails its course tests |

## 4. The artifact and its check

**Step 1, the hook.** Write `.githooks/commit-msg`: a script starting with `#!/usr/bin/env bash`, made executable with `chmod +x`. It reads the file named by its first argument, takes the first line that is neither blank nor a comment as the header, exits 0 when the header matches 2.2, and otherwise prints the rule and an example on stderr and exits 1. bash's `[[ "$header" =~ $pattern ]]` matches an extended regular expression; `[[:space:]]` is its spelling of `\s`.

**Step 2, the workflow.** Write `.github/workflows/ci.yml` with `on:` both `push` and `pull_request`, and the three jobs of section 3, each with `runs-on: ubuntu-latest`. For `commit-lint`, compute the range of 2.4 and loop over it:

```bash
for c in $(git rev-list "$BASE..$HEAD"); do
  git log -1 --format=%B "$c" > "$RUNNER_TEMP/msg"
  .githooks/commit-msg "$RUNNER_TEMP/msg"
done
```

For `native-tests`, Pass 0 has only primers: build and run `primers/lang.02` with `make`, and import your `primers/lang.01` modules inside their uv project (the `astral-sh/setup-uv` action installs uv). For `course-check`, paste the recipe `ol course ci` prints.

**Step 3, use the hook locally.** `git config core.hooksPath .githooks`, then try `git commit --allow-empty -m "wip"`: it must be refused with your message.

**Step 4, your first conventional commit.** Stage your primers, `.githooks/`, and `.github/`, and commit them with a conventional header, for example `ci: gate pushes on commit lint, native tests, and ol check`.

**Step 5, push and gate.** Create an empty repository on GitHub, `git remote add origin <url>`, `git push -u origin main`, and watch the run (`gh run watch`). Without GitHub, write `scripts/ci-local.sh` running the same three gates and declare it under `[ci]` (2.6). Then `ol check craft.01` and `ol milestone MS-P0`.

**The check.** `ol check craft.01` runs `course/tests/craft.01/check` in your repo. It refuses early if the repo is not a git repository or either file is missing, then runs the tests with pytest and PyYAML (`uv run --no-project`). The hook is graded by the messages it accepts and rejects; the workflow is read, not run (GitHub runs it; MS-P0 asks whether it was green).

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hook_is_an_executable_script` | unit | execute bit and a `#!` line | git silently skips a hook it cannot exec |
| `test_hook_accepts_the_worked_example` | unit | the section 3 message, body and footer included | you and the test agree on the format |
| `test_hook_accepts_conventional_headers` | unit | plain, scoped, `!`, scoped with `!`, every type | valid history is never blocked |
| `test_hook_rejects_malformed_headers` | boundary | the section 3 near misses, with a reason on stderr | a near miss fails here, not at MS-P0 |
| `test_hook_reads_the_first_line_and_ignores_comments` | boundary | comment lines skipped, only the header decides, an empty message fails | git's editor file is not the final message |
| `test_head_commit_passes_your_hook` | conformance | your HEAD passes your hook and MS-P0's pattern | the gate has judged real history |
| `test_workflow_runs_on_push_and_pull_request` | unit | both triggers | every change is checked before it merges |
| `test_workflow_has_the_three_gate_jobs` | unit | `commit-lint`, `native-tests`, `course-check`, each with `runs-on` and steps | branch protection requires them by name |
| `test_commit_lint_job_fetches_history_and_runs_your_hook` | unit | `fetch-depth: 0`; runs `.githooks/commit-msg` | every pushed commit is linted, by one rule |
| `test_native_tests_job_runs_commands` | unit | at least one `run:` step | your own tests run on every push |
| `test_course_check_job_pins_openlearn_to_contracts_version` | conformance | `contracts/VERSION`, `ol check --all --ci`, `OL_COURSE_HOME`, no `--ref-deps` | you are graded by the tests your contracts came with |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. A loose pattern such as `^[a-z]+:` | `feat:count`, `feature: x`, and `fix(): x` pass the hook, then fail MS-P0 on HEAD | `test_hook_rejects_malformed_headers` |
| 2. Matching the whole file, or a comment line, as the header | a valid commit is refused because of git's comment text, or "feat:" hidden in the body passes | `test_hook_reads_the_first_line_and_ignores_comments` |
| 3. Triggering only on pushes to `main` | pull requests merge without a run | `test_workflow_runs_on_push_and_pull_request` |
| 4. The default shallow checkout in `commit-lint` | `git rev-list` fails on the missing `before` commit, or only the last commit is linted | `test_commit_lint_job_fetches_history_and_runs_your_hook` |
| 5. Cloning openlearn's latest commit in `course-check` | new course tests run against your older contracts and fail for reasons that are not yours | `test_course_check_job_pins_openlearn_to_contracts_version` |
| 6. A hook without the execute bit | git skips it without a word: every message "passes" | `test_hook_is_an_executable_script` |
| 7. Merging pull requests with a merge commit | HEAD becomes `Merge pull request #3 ...`, which is not a Conventional Commit; use squash or rebase merging | `test_head_commit_passes_your_hook` |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `lang.02` | exit codes decide every step; hooks are scripts; `git rev-list` and `git log` walk history |
| Forward | `rt.01` | the first module `course-check` grades in Pass 1, together with every later started module |
| Forward | `craft.03` | from Pass 2 your own tests are graded by mutation testing; `native-tests` runs them on every push |
| Forward | `dep.05` | extends this workflow: image build, kind end to end (`tilt ci`, `ol milestone MS-prod --smoke`), the perf gate |
| Forward | `craft.11` | releases `v1.0.0` and its changelog from your conventional history |

A practice module has no code call site, so no module's `ol check` blocks on it. MS-P0 requires a fresh pass of `craft.01`, and its two smoke steps (a conventional HEAD and a green CI) are rerun by every later pass gate.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `.githooks/commit-msg` | commitlint with `@commitlint/config-conventional` | configurable rules (header length, case, allowed scopes) and editor integration | conventional-changelog/commitlint, `@commitlint/config-conventional` |
| `core.hooksPath .githooks` | the pre-commit framework | hooks pinned by version and installed per clone with one command | pre-commit.com |
| conventional history | release-please, semantic-release | the next version and the changelog computed from commit types | googleapis/release-please `README.md` |
| the `commit-lint` job | GitHub rulesets and required status checks | the merge button stays disabled until the named jobs pass | docs.github.com, "About protected branches" |
| `scripts/ci-local.sh` | nektos/act | runs the GitHub workflow itself in local containers | nektos/act `README.md` |
