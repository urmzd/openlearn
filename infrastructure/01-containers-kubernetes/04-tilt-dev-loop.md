<!-- ol:module dep.04 -->
# Tilt dev loop

## Overview

| | |
|---|---|
| **Module** | `dep.04` · practice · ops · Pass 7 · 2 to 4 h |
| **You build** | `deploy/Tiltfile`: images for the engine (full rebuild) and the gateway (live update of a host-compiled binary), your Helm charts deployed into `<system>`, the gateway after the engine, a guard against any context but `kind-<system>`, and a `tilt ci` budget; plus the gateway's dev image (`deploy/tilt/gateway.dev.Dockerfile` in the reference) |
| **Contract** | your own charts and Dockerfiles (`dep.00`, later `dep.01` to `dep.03`); [Tiltfile API](https://docs.tilt.dev/api.html) |
| **Tests** | `course/tests/dep.04/` (`check` runs `artifacts.py`, which evaluates your Tiltfile with the recording fakes in `_tiltfile.py`; section 4) |
| **Needs** | `dep.03` (the Pass 7 charts the loop deploys), with the Dockerfiles of `dep.00` and `dep.01` |
| **Used by** | `dep.05` runs `tilt ci` in its kind job; every later chart (`dep.06`, `dep.07`) joins the graph |
| **Milestone** | MS-prod (the deployment `tilt ci` produces) |
| **Optional depth** | [Tilt docs: Tiltfile concepts, live update](https://docs.tilt.dev/) (free), [restart_process extension](https://github.com/tilt-dev/tilt-extensions/tree/master/restart_process) (free) |

## Key Takeaways

- The loop is a **graph of resources**: image builds, local commands, and Kubernetes workloads, with explicit dependencies. Tilt watches files and reruns only what a change reaches.
- Tilt deploys a freshly built image only into containers whose image **repository equals** the `docker_build` ref. A mismatch builds your code and deploys the old image, silently.
- **Go**: compile on the host for `linux` with `CGO_ENABLED=0`, sync the one binary into the running container, restart the process: seconds. **Rust**: rebuild the image; the runtime image has no toolchain to compile in.
- A Tiltfile is code that deploys to whatever context kubectl points at: make it **refuse** every context but `kind-<system>`.
- `tilt ci` runs the same graph once and exits 0 when every resource is ready; give it a budget so a stuck pod fails CI in minutes, not half an hour.

## How to work this chapter

```bash
ol start dep.04                       # records the start; there are no stubs
ol tests dep.04                       # read the test catalog first
# write deploy/Tiltfile (section 4), then:
OL_SMOKE=1 ol check dep.04            # evaluates the Tiltfile without Tilt or a cluster
tilt up -f deploy/Tiltfile            # the loop: http://localhost:10350 shows the graph
tilt ci -f deploy/Tiltfile            # once, as CI runs it (dep.05)
ol check dep.04                       # adds the cluster tier: `tilt ci` on kind-<system>
```

---

## 1. Why now

Pass 7 changes the engine and the gateway every day: continuous batching, routing, rate limits, caching. On kind, each change today is `docker build` (a minute for the gateway, several for the engine), `kind load docker-image`, `helm upgrade`, wait for the rollout, then look at logs in another terminal, repeated for each service you touched and in the right order. That loop is slow enough that you stop testing on the cluster and start testing only locally, and the bugs that exist only in the cluster (probes, Service names, the `/artifacts` mount, the gateway reaching the engine) wait for the next milestone. A dev loop that redeploys in seconds keeps the cluster the place where you run your code.

## 2. Principles

### 2.1 Resources and the graph

| Term | Meaning |
|---|---|
| image target | how to build one image: `docker_build(ref, context, dockerfile=...)` |
| Kubernetes resource | one workload from the YAML you hand to `k8s_yaml`, named after it (`forge-engine`) |
| local resource | a command on your machine: `local_resource(name, cmd, deps=[...])`, rerun when a `deps` file changes |
| `resource_deps` | resources that must be ready before this one starts, in `k8s_resource(name, resource_deps=[...])` |
| live update | steps that patch a running container instead of replacing it: `sync(local, remote)`, `run(cmd)` |

Tilt matches images to workloads by name: every container whose image repository equals a `docker_build` ref gets the image Tilt built, retagged with a content hash (`forge-engine:tilt-3f9c...`), and is loaded into kind. The chart's own tag is ignored.

### 2.2 Two kinds of change

| Change | Fast path | Why |
|---|---|---|
| Go source of the gateway | `go build` on the host (2 s), `sync` the binary into the container, restart the process | Go cross-compiles to a static binary; the container needs nothing but the file |
| Rust or C source of the engine | full image rebuild, then a new pod | the engine links your C library and needs the toolchain; the runtime image (debian slim) has neither, by design (`dep.01` keeps images small) |
| a chart or a values file | redeploy the YAML | Tilt diffs the rendered objects |

For the Go path, two details decide whether it works: the kind node is Linux, so the host build needs `GOOS=linux` (a macOS binary dies with `exec format error`), and `CGO_ENABLED=0`, so the binary needs no libc in the image. Restarting the process is the `restart_process` extension's `docker_build_with_restart`, which wraps the entrypoint so a sync triggers a restart; it needs a shell in the image, so the dev image is alpine, not distroless.

### 2.3 Ordering

The gateway's `/readyz` waits on the engine. Started first, it fails readiness, restarts, and fills its log with 503s that hide real problems. `resource_deps` makes the order explicit. The same will hold for the durable server before its workers (`dep.06`).

### 2.4 The context guard and `tilt ci`

`k8s_context()` is the current kubectl context. Tilt deploys there, after a warning only for contexts it does not recognize as local. One `if k8s_context() != 'kind-forge': fail(...)` at the top makes a wrong context an error before anything is built. `tilt ci` evaluates the same Tiltfile, builds and deploys everything once, streams logs, and exits 0 when every resource is ready (non-zero when any fails); `ci_settings(timeout=...)` bounds it.

## 3. Worked example by hand

The reference graph after you save `go/gateway/route/route.go` (times are typical for a laptop; yours will differ, the ratio will not):

| Resource | Triggered by | Steps | Time |
|---|---|---|---|
| `gateway-compile` | `go/` changed (its `deps`) | `cd ../go && CGO_ENABLED=0 GOOS=linux go build -o ../deploy/tilt/build/gateway ./cmd/gateway` | 2.1 s |
| `forge-gateway` (image) | `deploy/tilt/build/gateway` changed (its `only`) | live update: `sync('tilt/build/gateway', '/usr/local/bin/gateway')`, restart the process | 0.6 s |
| `forge-engine` | nothing (no engine file changed) | none | 0 s |

About 3 seconds from save to the new gateway answering. Now edit `rust/crates/tl-engine/src/sched.rs`:

| Resource | Steps | Time |
|---|---|---|
| `forge-engine` (image) | `docker build -f docker/engine.Dockerfile ..` (contracts and C layers cached, `cargo build --release` of the changed crates) | 70 s |
| `forge-engine` | `kind load`, new ReplicaSet, readiness on `/healthz` | 12 s |
| `forge-gateway` | unchanged; its upstream is the Service, so it follows the new pod | 0 s |

**The name match.** The engine chart renders `image: "forge-engine:0.1.0"`. The ref `forge-engine` has the same repository, so Tilt deploys `forge-engine:tilt-<hash>` instead. Had the Tiltfile said `docker_build('engine', ...)`, Tilt would build `engine`, find no container using it, and deploy `forge-engine:0.1.0`, the image from the last manual build: your edit would never run, and no error would say so. `test_images_match_the_charts` reports exactly this mismatch.

**The guard.** With `kubectl config use-context prod-east`, evaluating the reference Tiltfile stops at line 17 with `deploy/Tiltfile deploys only to kind-forge; current context is prod-east (kubectl config use-context kind-forge)`. The check evaluates your Tiltfile twice, with `kind-<system>` (must succeed) and with `prod-cluster` (must fail).

## 4. The interface

```python
# deploy/Tiltfile (paths are relative to deploy/)
load('ext://restart_process', 'docker_build_with_restart')
SYSTEM = 'forge'
if k8s_context() != 'kind-' + SYSTEM:
    fail('deploy/Tiltfile deploys only to kind-%s' % SYSTEM)
ci_settings(timeout='15m')

docker_build(SYSTEM + '-engine', '..', dockerfile='docker/engine.Dockerfile', only=['contracts', 'c', 'rust'])
local_resource('gateway-compile', 'cd ../go && CGO_ENABLED=0 GOOS=linux go build -o ../deploy/tilt/build/gateway ./cmd/gateway',
               deps=['../go', '../contracts/go'])
docker_build_with_restart(SYSTEM + '-gateway', 'tilt', dockerfile='tilt/gateway.dev.Dockerfile',
                          entrypoint=['/usr/local/bin/gateway'], only=['build/gateway'],
                          live_update=[sync('tilt/build/gateway', '/usr/local/bin/gateway')])

k8s_yaml(helm('helm/%s-engine' % SYSTEM, name=SYSTEM + '-engine', namespace=SYSTEM))
k8s_yaml(helm('helm/%s-gateway' % SYSTEM, name=SYSTEM + '-gateway', namespace=SYSTEM))
k8s_resource(SYSTEM + '-gateway', resource_deps=[SYSTEM + '-engine', 'gateway-compile'])
```

`tilt up` runs one unified engine, which starts fastest. The production topology MS-prod checks is one engine release per role (`<system>-engine-prefill`, and `<system>-engine-decode` with 2 replicas); the reference Tiltfile switches with a Tilt config flag, `tilt up -f deploy/Tiltfile -- --topology=disaggregated`, and makes the gateway wait on every engine release.

The check does not run Tilt for the static tier: `_tiltfile.py` evaluates your Tiltfile as Python with a recording fake for each builtin (the list is at the top of that file), and `helm()` renders your charts with the real `helm template`. Use the builtins listed there; `load()` of other extensions fails with the name it did not know.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_tiltfile_evaluates` | unit | the Tiltfile runs with `k8s_context() == 'kind-<system>'` | every `tilt up` and `tilt ci` |
| `test_refuses_other_clusters` | fault | with `prod-cluster` it calls `fail()` | your dev images never reach a cluster you did not mean |
| `test_images_match_the_charts` | conformance | each chart's image repository has a `docker_build` of the same name | edits actually deploy |
| `test_engine_rebuilds_from_its_dockerfile` | unit | engine: no live update, `deploy/docker/engine.Dockerfile`, repo-root context | the same image `dep.00` and CI build |
| `test_gateway_live_updates_a_linux_binary` | unit | gateway: a sync, a restart, a `go build` with `CGO_ENABLED=0 GOOS=linux` that watches `../go` | seconds per Go change |
| `test_charts_deployed` | unit | `helm()` of both charts into `<system>`; both workloads rendered | the production charts, no dev copies |
| `test_gateway_starts_after_the_engine` | unit | `resource_deps` puts the engine first (and durable before workers, once they exist) | no crash loop at startup |
| `test_ci_has_a_budget` | boundary | `ci_settings(timeout=...)` of at most 20 minutes | `dep.05`'s kind job fails fast |
| `test_tilt_ci_on_kind` | conformance | `tilt ci` on `kind-<system>` exits 0 | the loop works end to end |

The last test is the **cluster tier**; `OL_SMOKE=1` skips it with the reason.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. No context guard | `tilt up` in the wrong terminal deploys dev images to a shared cluster | `test_refuses_other_clusters` |
| 2. `docker_build('engine', ...)` while the chart runs `forge-engine` | Tilt shows a green build; the pod runs yesterday's image | `test_images_match_the_charts` |
| 3. `go build` without `GOOS=linux` (on a Mac) or with cgo | the synced binary fails with `exec format error`, or `not found` for libc | `test_gateway_live_updates_a_linux_binary` |
| 4. No `resource_deps` | the gateway restarts until the engine is up; real errors drown in 503s | `test_gateway_starts_after_the_engine` |
| 5. Build context `..` without `only` or `.dockerignore` | every save under `rust/target` triggers an engine rebuild | `test_engine_rebuilds_from_its_dockerfile` (the Dockerfile and context), and the .dockerignore from dep.00 |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `dep.03` | the charts the loop deploys, into the kind cluster of `dep.02`, with the Dockerfiles of `dep.00` and `dep.01` |
| Forward | `dep.05` | CI's kind job runs `tilt ci -f deploy/Tiltfile` |
| Forward | `dep.06` | durable and worker charts join the graph, durable first |
| Forward | `obs.02` | the monitors and the collector can be applied from the same Tiltfile |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| Tilt on kind | Skaffold, Garden, DevSpace | the same loop with other tradeoffs (profiles, remote clusters, test stages) | [Skaffold](https://skaffold.dev/) |
| host compile and sync | remote dev containers, Telepresence | run one service locally against the cluster | [Telepresence](https://www.telepresence.io/) |
| `kind load` of every image | a local registry (`localhost:5001`, `dep.02`) | faster loads, the same flow as a real registry | [kind local registry](https://kind.sigs.k8s.io/docs/user/local-registry/) |
| one Tiltfile | Tilt extensions and `include()` | per-team Tiltfiles composed into one graph | [tilt-extensions](https://github.com/tilt-dev/tilt-extensions) |
