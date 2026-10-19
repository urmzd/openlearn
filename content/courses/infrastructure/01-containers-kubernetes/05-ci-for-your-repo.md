# CI for the platform repo

## Overview

| | |
|---|---|
| **Module** | `dep.05` · practice · ops · Pass 7 · 3 to 5 h |
| **Solution** | [`.github/workflows/platform.yml`](../../../../course/solutions/entry/.github/workflows/platform.yml): jobs `lint`, `unit`, `images`, `kind-e2e`, and `perf-gate` (the last uses the optional `load.02`), next to `craft.01`'s [`ci.yml`](../../../../course/solutions/entry/.github/workflows/ci.yml) (commit-lint and native tests); `craft.11` adds [`release.yml`](../../../../course/solutions/entry/.github/workflows/release.yml) |
| **Contract** | [GitHub Actions workflow syntax](https://docs.github.com/en/actions/writing-workflows/workflow-syntax-for-github-actions) |
| **Needs** | `craft.01` (the gate this extends), `dep.04` (`tilt ci` is the kind job) · optional: `load.02`'s `compare` is the perf gate |
| **Used by** | the Pass 7 production platform and every later pass run on what this pipeline keeps green; `ops.06` (dependency upgrade) and `ops.07` (perf regression) are detected by it going red |
| **Optional depth** | [GitHub Actions security hardening](https://docs.github.com/en/actions/security-for-github-actions/security-guides/security-hardening-for-github-actions) (free), *Continuous Delivery* (Humble and Farley), ch. 5 |

## Key Takeaways

- Each job answers one question and has a fixed id, so branch protection can require it by name: does it lint, do the unit tests pass, do the images build, does the system deploy and serve a smoke request, is main slower than before.
- Pin what runs: actions by release tag or commit sha, downloaded tools by release URL; `permissions: contents: read` by default.
- The kind job builds the cluster from the repo's own `deploy/kind/cluster.yaml` and deploys with its own `tilt ci`: CI and your laptop run the same loop.
- The perf gate (optional, with `load.02`) runs **on main only**: each green run uploads its load report, the next run compares against it with `load.02`'s statistics, and the commit that regressed is the one that turns red.
- Every job has a timeout; a hung cluster costs minutes, not six hours.

## How to read this chapter

Sections 1 to 3 derive the pipeline's jobs from the questions a change must answer, and time one pull request and one perf comparison by hand. Section 4 fixes the workflow's shape. Section 5 walks through the solution workflow job by job and explains each choice. Try to predict each job's steps before reading them.

The pipeline runs on GitHub on every push; from a terminal, `gh run list --workflow platform --limit 3` shows its latest runs.

---

## 1. Why now

The repo's CI (`craft.01`) lints commit messages and runs a few native tests. Since then the system grew a Rust engine, a Go gateway, Dockerfiles, charts, a Tiltfile, and a load generator, and none of them is exercised on a push. A pull request can break the engine image (a moved `COPY` source), the chart (a renamed value), or TTFT (a lock in the scheduler) and still be green. Pass 7 is about running the system in production shape, and production shape includes a pipeline that refuses those changes.

## 2. Principles

### 2.1 One question per job

| Job | Question | Fails when |
|---|---|---|
| `lint` | is the code formatted and free of lint? | `ruff`, `gofmt`, `go vet`, `cargo fmt --check`, `cargo clippy -D warnings` |
| `unit` | do the native tests pass? | `pytest`, `go test -race`, `cargo test`, the C build |
| `images` | does every Dockerfile build, as non-root? | a `docker build` |
| `kind-e2e` | does the system deploy and serve? | `tilt ci`, then a smoke run against the deployed gateway |
| `perf-gate` | is main slower than the last green main? | `{loadgen} compare base.json head.json --metric ttft_p95 --max-regress 5%` |

Jobs run in parallel unless `needs:` orders them: `kind-e2e` needs `images` (no point deploying images that do not build) and `unit`.

### 2.2 Supply chain: what runs in CI

| Symbol | Meaning |
|---|---|
| `uses: owner/action@ref` | runs the code at `ref` of that repository with the job's token and secrets |
| moving ref | a branch (`main`, `master`, `stable`): its code changes without a commit in this repo |
| pinned ref | a release tag (`v4`, `v1.12.0`) or a 40-hex commit sha |

A workflow is code that runs with write access to the repository unless it says otherwise. Three habits close most of the gap: pin every action and downloaded tool, declare `permissions:` (read by default, a job asks for more), and pass secrets through `env`, never `echo` them (masking misses transformed values).

### 2.3 The kind job

`helm/kind-action` creates a kind cluster on the runner from the config you pass. Pass `deploy/kind/cluster.yaml`: the same cluster name and NodePort mappings as on your laptop, so the deployment's coordinates (`dep.00` section 2.7) are right in CI too. Then `tilt ci` (`dep.04`) builds and deploys, and a smoke run streams a completion through the gateway NodePort. A throwaway API key is generated per run, masked, and stored in the Secret the gateway chart reads.

### 2.4 A perf gate needs a baseline

This job is optional: it needs `load.02`, an optional module. A load report is noisy: two identical runs differ by a few percent. `load.02`'s `compare` decides whether a difference is a regression with a permutation test and a threshold (`--max-regress 5%`). The baseline must be a run of the same code path on the same kind of machine: the report of the last green run on main. So the job runs only on pushes to main, downloads that report (`gh run download`), compares, and uploads its own report for the next run. On a pull request there is no stable baseline, and a gate that compares against a different runner type blocks good changes at random.

## 3. Worked example by hand

A pull request changes `go/gateway/route/route.go`. With the solution's `needs:` graph and typical durations:

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

`.github/workflows/platform.yml` (the solution, abridged; section 5 quotes it in full):

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
      # ... a smoke run against the deployed gateway
  perf-gate:
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    timeout-minutes: 30
    steps:   # start the stack, run the loadgen, fetch the last report, then:
      - run: go/bin/loadgen compare "$BASE" head.json --metric ttft_p95 --max-regress 5%
      - uses: actions/upload-artifact@v4
        with: {name: perf-report, path: head.json}
```

### What a correct solution must handle

GitHub Actions only runs on GitHub, so most properties are read from the workflow file itself; the run on GitHub is the final word.

| Case | Kind of check | What must hold | Why it matters downstream |
|---|---|---|---|
| workflow parses and triggers | static | one mapping; `on` has `push` and `pull_request` | every change is gated |
| required jobs | static | the job ids, each with `runs-on`, steps, and `timeout-minutes` at most 60 | branch protection requires them by name |
| actions are pinned | boundary | every `uses:` has a version tag or a sha, no moving branch; no moving-branch script piped to a shell | what runs is what you reviewed |
| lint covers every language | static | ruff, gofmt, go vet, cargo fmt `--check`, clippy `-D warnings` | cheap failures stay cheap |
| unit runs every suite | static | pytest, `go test -race`, cargo test, `make -C c` | every native test on every push |
| images job builds every Dockerfile | conformance | each `deploy/docker/*.Dockerfile` is built | a broken image fails its own pull request |
| kind e2e job | conformance | kind from `deploy/kind/cluster.yaml`, `tilt ci`, a smoke run against the deployed system, `needs: images` | the deployed system on every pull request |
| perf gate on main against the last report | conformance | if the job exists: `if:` restricted to main; `compare ... --metric ... --max-regress ...`; the report uploaded | regressions fail the commit that made them |
| least privilege, no echoed secrets | boundary | top-level `permissions:`; no step echoes a secret | a compromised step can do less |

## 5. The solution, step by step

### 5.1 From first principles

1. **A check is only useful if it can be required.** Branch protection requires jobs by id, so each question of 2.1 is one job with a fixed id, and a failure names the question that failed.
2. **Cheap questions first, expensive ones only when they can pass.** `lint` and `unit` take minutes; a kind cluster takes many. `needs: [images, unit]` keeps the expensive job from paying for a cluster to deploy images that do not build.
3. **CI must run the same thing as the laptop.** So the kind job uses the repo's own `deploy/kind/cluster.yaml` and `tilt ci`; any CI-only deploy path would test something nobody runs.
4. **Whatever CI runs, it runs with the repo's token.** So every action is pinned to a release tag or sha, every downloaded tool to a versioned URL, and `permissions: contents: read` is the default; a job that needs more asks for it.
5. **A perf comparison needs a comparable baseline.** Only the last green run of the same workflow on main, on the same runner type, measures the same code path on the same hardware class. So the gate runs on pushes to main only, and each run's report becomes the next baseline.
6. **Nothing may hang forever.** Every job has `timeout-minutes`; the default is six hours.

### 5.2 Triggers, concurrency, permissions

```yaml
name: platform
on:
  push:
    branches: [main]      # main: every merged commit, and the perf gate's baseline
  pull_request:           # every proposed change

# One run per branch: a new push cancels the old run instead of queueing.
concurrency:
  group: platform-${{ github.ref }}
  cancel-in-progress: true

# Least privilege: the default GITHUB_TOKEN can write to the repo.
permissions:
  contents: read

env:
  SYSTEM: forge
  TILT_VERSION: "0.35.1"  # one pin, used by the install step's URL
```

`concurrency` matters for cost more than correctness: on a busy pull request, three pushes would otherwise mean three kind clusters.

### 5.3 `lint` and `unit`

```yaml
  lint:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
      - uses: actions/setup-go@v5
        with:
          go-version-file: go/go.mod          # the Go version the module declares, not a second pin
      - uses: dtolnay/rust-toolchain@1.88.0   # a version, not @stable (a moving branch)
        with:
          components: rustfmt, clippy
      - name: python (ruff)
        run: uvx ruff@0.13.0 check python && uvx ruff@0.13.0 format --check python
      - name: go (gofmt, vet)
        run: |
          # gofmt -l prints files that need formatting and exits 0 regardless,
          # so the job fails on non-empty output instead.
          test -z "$(gofmt -l go)" || { gofmt -l go; exit 1; }
          cd go && go vet ./...
      - name: rust (fmt, clippy)
        run: |
          cargo fmt --manifest-path rust/Cargo.toml --all --check
          cargo clippy --manifest-path rust/Cargo.toml --workspace --all-targets -- -D warnings

  unit:
    runs-on: ubuntu-latest
    timeout-minutes: 20
    steps:
      # ... checkout and the same three toolchains
      - name: c (libtinyllm)
        run: make -C c                       # first: the engine and Python link it
      - name: python (pytest)
        run: uv run --project python --with pytest --with hypothesis pytest -q python/tests
      - name: go (race detector)
        run: cd go && go test -race -count=1 ./...   # -count=1: no cached results
      - name: rust
        run: cargo test --manifest-path rust/Cargo.toml --workspace
```

`-D warnings` turns clippy's warnings into errors, so lint debt cannot accumulate. `-race` is the one flag that catches the gateway's shared-state bugs (rate-limit buckets, cache, registry) before load does.

### 5.4 `images`

```yaml
  images:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v4
      - name: build every image from the repo root
        run: |
          for part in engine gateway durable worker agent; do
            docker build -f deploy/docker/$part.Dockerfile -t "$SYSTEM-$part:ci" .
            # The kubelet's runAsNonRoot check, done here: empty, 0, or root fails.
            user="$(docker image inspect -f '{{.Config.User}}' "$SYSTEM-$part:ci")"
            case "${user%%:*}" in ''|0|root) echo "$part runs as root" >&2; exit 1 ;; esac
          done
```

`${user%%:*}` strips a `:group` suffix, so `10001:10001` is checked as `10001`. The loop names every Dockerfile, including the durable, worker, and agent images of `dep.06` and `dep.07`.

### 5.5 `kind-e2e`

```yaml
  kind-e2e:
    needs: [images, unit]
    runs-on: ubuntu-latest
    timeout-minutes: 45
    steps:
      # ... checkout and toolchains
      # The cluster from your own config: same name, same NodePort mappings.
      - uses: helm/kind-action@v1.12.0
        with:
          cluster_name: ${{ env.SYSTEM }}
          config: deploy/kind/cluster.yaml
      - name: install tilt (pinned release)
        run: |
          curl -fsSL "https://github.com/tilt-dev/tilt/releases/download/v${TILT_VERSION}/tilt.${TILT_VERSION}.linux.x86_64.tar.gz" \
            | sudo tar -xz -C /usr/local/bin tilt
          tilt version
      - name: namespace and a throwaway API key
        run: |
          kubectl create namespace "$SYSTEM"
          key="tl_ci_$(openssl rand -hex 12)"
          echo "::add-mask::$key"             # masked before it can appear in any log
          kubectl -n "$SYSTEM" create secret generic "$SYSTEM-api-key" --from-literal=api-key="$key"
          kubectl -n "$SYSTEM" create secret generic "$SYSTEM-gateway-pepper" --from-literal=pepper="$(openssl rand -hex 32)"
          echo "TL_API_KEY=$key" >> "$GITHUB_ENV"   # later steps read it from env, never echo it
      - name: tilt ci
        run: tilt ci -f deploy/Tiltfile --context "kind-$SYSTEM" --timeout 15m -- --topology=disaggregated
      # ... a smoke run against the deployed system (see below)
```

The key is generated per run, so no long-lived secret exists for CI to leak; it is created as the same two Secrets the charts reference (`dep.00`, `dep.03`). `tilt ci` deploys the disaggregated topology, the production shape. The solution file's last step still calls the course's former milestone runner, which has been removed; what that step has to prove is the end-to-end tracer of `dep.00` section 3: a streamed completion of 32 or more chunks through `http://127.0.0.1:30080` with `$TL_API_KEY`, which works in CI because kind used the same port mappings as on your laptop.

### 5.6 `perf-gate`

```yaml
  perf-gate:
    # Only on main: a pull request compares against nothing stable yet, and
    # every main run's report becomes the next run's baseline.
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    needs: [unit]
    runs-on: ubuntu-latest
    timeout-minutes: 30
    permissions:
      contents: read
      actions: read            # gh run download of the previous report
    steps:
      # ... build and start tl-serve and the gateway on the runner; wait on /readyz
      - name: load run (head)
        run: TL_API_KEY=tl_perf_local go/bin/loadgen --target http://127.0.0.1:8080 --rate 10 --duration 60s --seed 1 --out head.json
      - name: last green main report (base)
        env:
          GH_TOKEN: ${{ github.token }}
        run: |
          id="$(gh run list --workflow platform --branch main --status success --limit 1 --json databaseId -q '.[0].databaseId')"
          if [ -n "$id" ] && gh run download "$id" --name perf-report --dir base; then echo "BASE=base/head.json" >> "$GITHUB_ENV"; fi
      - name: compare (load.02)
        run: |
          if [ -z "${BASE:-}" ]; then echo "no baseline yet: this run's report becomes it"; exit 0; fi
          go/bin/loadgen compare "$BASE" head.json --metric ttft_p95 --max-regress 5%
      - uses: actions/upload-artifact@v4
        with:
          name: perf-report
          path: head.json
          retention-days: 30
```

The job widens its own permissions (`actions: read`) for exactly one need. A fixed `--seed 1` and fixed rate make two runs of the same code generate the same request sequence, so the comparison measures the code, not the workload. The first run on a fresh repository has no baseline and passes on purpose: its report is the baseline. Because a failed run uploads nothing that `--status success` would select, a regression never becomes the new baseline. The stack runs as host processes rather than on kind: the gate measures the serving code, and a cluster would add variance without adding signal.

Full solution: [`platform.yml`](../../../../course/solutions/entry/.github/workflows/platform.yml).

### 5.7 Why this solution

- **Separate jobs over one long script.** Parallel jobs cut wall time (22 minutes instead of the sum), and each id is a separately requirable check.
- **Tag pins over sha pins.** Release tags of well-maintained actions are a reasonable middle ground; sha pins with Dependabot are stronger (Going further). Moving branches are never acceptable.
- **`tilt ci` over a CI-specific `helm install` script.** One deploy path, exercised on every pull request, is the one that works on laptops too.
- **Perf on main only over perf on every pull request.** A gate that fails at random gets disabled; a gate that compares like with like gets trusted. The cost is that a regression is caught one commit late, on main, where `ops.07`'s bisect finds it.
- **Tradeoff accepted:** GitHub-hosted runners are noisy neighbours; the 5% threshold and `compare`'s permutation test absorb some of that noise, and dedicated runners would absorb more.

### 5.8 Explain it back

1. Why does `kind-e2e` need `images` even though `tilt ci` builds the images itself?
   <details><summary>Answer</summary>Ordering, not artifacts: a broken Dockerfile should fail the cheap <code>images</code> job in minutes, not after the runner has created a kind cluster and Tilt has started building. <code>needs:</code> makes the expensive job wait for the cheap one to pass.</details>
2. Why does the perf gate not run on pull requests?
   <details><summary>Answer</summary>There is no stable baseline for a pull request: the last green main run may be on a different runner, and the branch's own history is not a release history. Comparing against an unstable baseline fails good changes at random. On main, each green run's report is the next baseline.</details>
3. The CI key is masked with `::add-mask::`. Why is masking alone not enough, and what else does the solution do?
   <details><summary>Answer</summary>Masking matches the literal string; a transformed value (base64, a substring) prints in clear. The solution never echoes the key: it passes it through <code>$GITHUB_ENV</code> to later steps and into a Kubernetes Secret, and it is generated per run, so a leak would expose a key that dies with the runner.</details>
4. What does `test -z "$(gofmt -l go)"` do that `gofmt -l go` alone does not?
   <details><summary>Answer</summary><code>gofmt -l</code> lists unformatted files but exits 0 either way. Wrapping it in <code>test -z</code> turns non-empty output into a failing exit code, and the <code>|| { gofmt -l go; exit 1; }</code> branch prints the list so the log names the files.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. No `permissions:`, or `echo ${{ secrets.X }}` in a step | the token can push to the repo; the secret is in the log | top-level `permissions: contents: read`; the key is masked and passed through `$GITHUB_ENV` (5.2, 5.5) |
| 2. `uses: some/action@main`, or `curl .../master/install.sh \| bash` | a change upstream changes CI without a commit in this repo | every `uses:` at a release tag; tilt from a versioned release URL (5.3, 5.5) |
| 3. `go test` without `-race` | the gateway's data races pass CI and fail under load | `go test -race -count=1` (5.3) |
| 4. A kind cluster from the action's default config | NodePorts 30080, 30090, 30320 are not mapped; the smoke run cannot reach the gateway | `config: deploy/kind/cluster.yaml` (5.5) |
| 5. No `timeout-minutes` | a stuck `tilt ci` holds a runner for 6 hours | `timeout-minutes` on every job, plus `tilt ci --timeout 15m` (5.5) |
| 6. The perf gate on pull requests, or with no stored baseline | random failures, or a gate that never compares anything | `if:` main only; the last green report downloaded, this run's uploaded (5.6) |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `craft.01` | the gates of `ci.yml` this workflow sits next to |
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
