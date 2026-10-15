<!-- ol:module dep.05 -->
# CI for the learner repo

## Overview

| | |
|---|---|
| **Module** | `dep.05` · practice · ops · Pass 7 · 3 to 5 h |
| **You build** | `.github/workflows/platform.yml`: jobs `lint`, `unit`, `images`, and `kind-e2e` (plus `perf-gate` if you did the optional `load.02`), next to `craft.01`'s `ci.yml` (which keeps commit-lint, native-tests, and course-check); `craft.11` adds `release.yml` |
| **Contract** | [GitHub Actions workflow syntax](https://docs.github.com/en/actions/writing-workflows/workflow-syntax-for-github-actions); the CI recipe of DESIGN 5.13 (openlearn at `contracts/VERSION`) |
| **Tests** | `course/tests/dep.05/` (`check` runs `artifacts.py`, which reads the workflow statically; section 4) |
| **Needs** | `craft.01` (the gate this extends), `dep.04` (`tilt ci` is the kind job) · optional: `load.02`'s `compare` is the perf gate |
| **Used by** | MS-prod and every later milestone run against what this pipeline keeps green; `ops.06` (dependency upgrade) and `ops.07` (perf regression) are graded by it going red |
| **Milestone** | MS-prod |
| **Optional depth** | [GitHub Actions security hardening](https://docs.github.com/en/actions/security-for-github-actions/security-guides/security-hardening-for-github-actions) (free), *Continuous Delivery* (Humble and Farley), ch. 5 |

## Key Takeaways

- Each job answers one question and has a fixed id, so branch protection can require it by name: does it lint, do the unit tests pass, do the images build, does the system deploy and pass its smoke milestone, is main slower than before.
- Pin what runs: actions by release tag or commit sha, downloaded tools by release URL; `permissions: contents: read` by default.
- The kind job builds the cluster from **your** `deploy/kind/cluster.yaml` and deploys with **your** `tilt ci`: CI and your laptop run the same loop.
- The perf gate (optional, with `load.02`) runs **on main only**: each green run uploads its load report, the next run compares against it with `load.02`'s statistics, and the commit that regressed is the one that turns red.
- Every job has a timeout; a hung cluster costs minutes, not six hours.

## How to work this chapter

```bash
ol start dep.05                       # records the start; there are no stubs
ol tests dep.05                       # read the test catalog first
# write .github/workflows/platform.yml (section 4), then:
ol check dep.05                       # reads the workflow; static only
git push                              # the real verdict: the run on GitHub
gh run list --workflow platform --limit 3
```

---

## 1. Why now

Your repo's CI (`craft.01`) lints commit messages, runs a few native tests, and runs `ol check --all --ci`. Since then the system grew a Rust engine, a Go gateway, Dockerfiles, charts, a Tiltfile, and a load generator, and none of them is exercised when you push. A pull request can break the engine image (a moved `COPY` source), the chart (a renamed value), or TTFT (a lock in the scheduler) and still be green. Pass 7's milestone is about running the system in production shape, and production shape includes a pipeline that refuses those changes.

## 2. Principles

### 2.1 One question per job

| Job | Question | Fails when |
|---|---|---|
| `lint` | is the code formatted and free of lint? | `ruff`, `gofmt`, `go vet`, `cargo fmt --check`, `cargo clippy -D warnings` |
| `unit` | do the native tests pass? | `pytest`, `go test -race`, `cargo test`, the C build |
| `images` | does every Dockerfile build, as non-root? | a `docker build` |
| `kind-e2e` | does the system deploy and serve? | `tilt ci`, then `ol milestone MS-prod --smoke` |
| `perf-gate` | is main slower than the last green main? | `{loadgen} compare base.json head.json --metric ttft_p95 --max-regress 5%` |

Jobs run in parallel unless `needs:` orders them: `kind-e2e` needs `images` (no point deploying images that do not build) and `unit`.

### 2.2 Supply chain: what runs in your CI

| Symbol | Meaning |
|---|---|
| `uses: owner/action@ref` | runs the code at `ref` of that repository with your job's token and secrets |
| moving ref | a branch (`main`, `master`, `stable`): its code changes without you |
| pinned ref | a release tag (`v4`, `v1.12.0`) or a 40-hex commit sha |

A workflow is code that runs with write access to your repository unless you say otherwise. Three habits close most of the gap: pin every action and downloaded tool, declare `permissions:` (read by default, a job asks for more), and pass secrets through `env`, never `echo` them (masking misses transformed values).

### 2.3 The kind job

`helm/kind-action` creates a kind cluster on the runner from the config you pass. Pass `deploy/kind/cluster.yaml`: the same cluster name and NodePort mappings as on your laptop, so `[deploy]` in `system.toml` is right in CI too. Then `tilt ci` (`dep.04`) builds and deploys, and `ol milestone MS-prod --smoke` runs the milestone's smoke steps from the openlearn commit your `contracts/VERSION` names (the `craft.01` recipe). A throwaway API key is generated per run, masked, and stored in the Secret the gateway chart reads.

### 2.4 A perf gate needs a baseline

This job is optional: it needs `load.02`, an optional module, and the check reads it only when your workflow has it. A load report is noisy: two identical runs differ by a few percent. `load.02`'s `compare` decides whether a difference is a regression with a permutation test and a threshold (`--max-regress 5%`). The baseline must be a run of the same code path on the same kind of machine: the report of the last green run on main. So the job runs only on pushes to main, downloads that report (`gh run download`), compares, and uploads its own report for the next run. On a pull request there is no stable baseline, and a gate that compares against a different runner type blocks good changes at random.

## 3. Worked example by hand

A pull request changes `go/gateway/route/route.go`. With the reference `needs:` graph and typical durations:

| Minute | lint | unit | images | kind-e2e | perf-gate |
|---|---|---|---|---|---|
| 0 | start | start | start | waiting on images, unit | skipped (not main) |
| 2 | pass | | | | |
| 6 | | pass | | | |
| 9 | | | pass | start | |
| 22 | | | | pass (`tilt ci` 9 min, smoke 4 min) | |

Wall time 22 minutes; the jobs that can run in parallel do. If `images` had no `needs:` relationship, `kind-e2e` would start at minute 0 and fail on an image that does not build, after paying for the cluster.

After merge, the push to main runs `perf-gate`. The last green main run's report says `ttft_p95` = 410 ms; this run measures 445 ms:

$$\frac{445 - 410}{410} = 0.085 = 8.5\% > 5\%$$

and `compare` also finds the shift significant, so it exits 1 and the commit that introduced it is red on main. Had this run measured 418 ms (+2.0%), it would pass, and its report would become the next baseline.

A pin, checked by hand: `uses: actions/checkout@v4` is a release tag (pinned); `uses: dtolnay/rust-toolchain@stable` is a branch (moving: rejected); `curl .../tilt/master/scripts/install.sh | bash` runs whatever that branch holds today (rejected); a release tarball URL with a version in it is pinned.

## 4. The interface

`.github/workflows/platform.yml` (the reference, abridged):

```yaml
name: platform
on: {push: {branches: [main]}, pull_request: {}}
permissions: {contents: read}
jobs:
  lint:      {runs-on: ubuntu-latest, timeout-minutes: 10, steps: [...]}  # ruff, gofmt, go vet, cargo fmt --check, clippy -D warnings
  unit:      {runs-on: ubuntu-latest, timeout-minutes: 20, steps: [...]}  # make -C c, pytest, go test -race, cargo test
  images:    {runs-on: ubuntu-latest, timeout-minutes: 30, steps: [...]}  # docker build -f deploy/docker/<part>.Dockerfile .
  kind-e2e:
    needs: [images, unit]
    timeout-minutes: 45
    steps:
      - uses: helm/kind-action@v1.12.0
        with: {cluster_name: forge, config: deploy/kind/cluster.yaml}
      - run: tilt ci -f deploy/Tiltfile --context kind-forge --timeout 15m -- --topology=disaggregated
      - run: |   # openlearn at contracts/VERSION, as in craft.01
          git -C .ol/openlearn checkout --quiet --detach "$SHA"
          OL_COURSE_HOME="$PWD" .ol/openlearn/practice/bin/ol milestone MS-prod --smoke
  perf-gate:
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    timeout-minutes: 30
    steps:   # start the stack, run the loadgen, fetch the last report, then:
      - run: go/bin/loadgen compare "$BASE" head.json --metric ttft_p95 --max-regress 5%
      - uses: actions/upload-artifact@v4
        with: {name: perf-report, path: head.json}
```

### What the tests check

GitHub Actions cannot run inside `ol check`, so the check reads the workflow; MS-P0's `ci-status` matcher is what reads the run on GitHub.

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_workflow_parses_and_triggers` | unit | one mapping; `on` has `push` and `pull_request` | every change is gated |
| `test_required_jobs` | unit | the five job ids, each with `runs-on`, steps, and `timeout-minutes` at most 60 | branch protection requires them by name |
| `test_actions_are_pinned` | boundary | every `uses:` has a version tag or a sha, no moving branch; no moving-branch script piped to a shell | what runs is what you reviewed |
| `test_lint_covers_every_language` | unit | ruff, gofmt, go vet, cargo fmt `--check`, clippy `-D warnings` | cheap failures stay cheap |
| `test_unit_runs_every_suite` | unit | pytest, `go test -race`, cargo test, `make -C c` | your own tests on every push |
| `test_images_job_builds_every_dockerfile` | conformance | each `deploy/docker/*.Dockerfile` is built | a broken image fails its own pull request |
| `test_kind_e2e_job` | conformance | kind from `deploy/kind/cluster.yaml`, `tilt ci`, `ol milestone MS-prod --smoke` at `contracts/VERSION`, `needs: images` | the deployed system on every pull request |
| `test_perf_gate_runs_on_main_against_the_last_report` | conformance | if the job exists: `if:` restricted to main; `compare ... --metric ... --max-regress ...`; the report uploaded | regressions fail the commit that made them |
| `test_least_privilege_and_no_echoed_secrets` | boundary | top-level `permissions:`; no step echoes a secret | a compromised step can do less |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. No `permissions:`, or `echo ${{ secrets.X }}` in a step | the token can push to your repo; the secret is in the log | `test_least_privilege_and_no_echoed_secrets` |
| 2. `uses: some/action@main`, or `curl .../master/install.sh \| bash` | a change upstream changes your CI without a commit of yours | `test_actions_are_pinned` |
| 3. `go test` without `-race` | the gateway's data races pass CI and fail under load | `test_unit_runs_every_suite` |
| 4. A kind cluster from the action's default config | NodePorts 30080, 30090, 30320 are not mapped; MS-prod's smoke cannot reach the gateway | `test_kind_e2e_job` |
| 5. No `timeout-minutes` | a stuck `tilt ci` holds a runner for 6 hours | `test_required_jobs` |
| 6. The perf gate on pull requests, or with no stored baseline | random failures, or a gate that never compares anything | `test_perf_gate_runs_on_main_against_the_last_report` |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `craft.01` | the three gates of `ci.yml` and the openlearn-at-`contracts/VERSION` recipe |
| Back | `dep.04` | `tilt ci` is the kind job's deploy step |
| Forward | `craft.11` | `release.yml`: tags, changelog, images pushed only from a release |
| Forward | `ops.06` | a dependency upgrade lands only with this pipeline green |
| Forward | `ops.07` | the perf gate's red commit is what `git bisect run` hunts |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| tag-pinned actions | sha-pinned actions with Dependabot or Renovate updates | immutable refs that still get updates, by pull request | [Dependabot for actions](https://docs.github.com/en/code-security/dependabot) |
| a kind cluster per run | ephemeral preview environments per pull request | a URL reviewers can click | Argo CD ApplicationSets, Vercel-style previews |
| a 60 s load run on main | continuous benchmarking on dedicated runners | stable baselines, trend dashboards | [Bencher](https://bencher.dev/), [Conbench](https://conbench.github.io/conbench/) |
| `docker build` only | SBOMs, signing, provenance (`craft.18`) | know and prove what is in each image | [SLSA](https://slsa.dev/), [cosign](https://docs.sigstore.dev/) |
