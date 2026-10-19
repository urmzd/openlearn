# Tilt dev loop

## Overview

| | |
|---|---|
| **Module** | `dep.04` · practice · ops · Pass 7 · 2 to 4 h |
| **Solution** | [`deploy/Tiltfile`](../../../../course/solutions/entry/deploy/Tiltfile): images for the engine (full rebuild) and the gateway (live update of a host-compiled binary), the Helm charts deployed into `<system>`, the gateway after the engine, a guard against any context but `kind-<system>`, and a `tilt ci` budget; plus the gateway's dev image [`deploy/tilt/gateway.dev.Dockerfile`](../../../../course/solutions/entry/deploy/tilt/gateway.dev.Dockerfile) |
| **Contract** | the system's charts and Dockerfiles (`dep.00`, later `dep.01` to `dep.03`); [Tiltfile API](https://docs.tilt.dev/api.html) |
| **Needs** | `dep.03` (the Pass 7 charts the loop deploys), with the Dockerfiles of `dep.00` and `dep.01` |
| **Used by** | `dep.05` runs `tilt ci` in its kind job, and the deployment it produces is the Pass 7 production platform; every later chart (`dep.06`, `dep.07`) joins the graph |
| **Optional depth** | [Tilt docs: Tiltfile concepts, live update](https://docs.tilt.dev/) (free), [restart_process extension](https://github.com/tilt-dev/tilt-extensions/tree/master/restart_process) (free) |

## Key Takeaways

- The loop is a **graph of resources**: image builds, local commands, and Kubernetes workloads, with explicit dependencies. Tilt watches files and reruns only what a change reaches.
- Tilt deploys a freshly built image only into containers whose image **repository equals** the `docker_build` ref. A mismatch builds the new code and deploys the old image, silently.
- **Go**: compile on the host for `linux` with `CGO_ENABLED=0`, sync the one binary into the running container, restart the process: seconds. **Rust**: rebuild the image; the runtime image has no toolchain to compile in.
- A Tiltfile is code that deploys to whatever context kubectl points at: make it **refuse** every context but `kind-<system>`.
- `tilt ci` runs the same graph once and exits 0 when every resource is ready; give it a budget so a stuck pod fails CI in minutes, not half an hour.

## How to read this chapter

Sections 1 to 3 explain why the cluster needs a fast inner loop, which kind of change takes which path, and time both paths by hand. Section 4 fixes the Tiltfile's shape. Section 5 walks through the solution Tiltfile and the gateway's dev image block by block. Try to predict each block before reading it.

The solution is run from the repo root:

```bash
tilt up -f deploy/Tiltfile            # the loop: http://localhost:10350 shows the graph
tilt ci -f deploy/Tiltfile            # once, as CI runs it (dep.05)
```

---

## 1. Why now

Pass 7 changes the engine and the gateway every day: continuous batching, routing, rate limits, caching. On kind, each change today is `docker build` (a minute for the gateway, several for the engine), `kind load docker-image`, `helm upgrade`, wait for the rollout, then look at logs in another terminal, repeated for each service you touched and in the right order. That loop is slow enough that you stop testing on the cluster and start testing only locally, and the bugs that exist only in the cluster (probes, Service names, the `/artifacts` mount, the gateway reaching the engine) wait until the next full deployment. A dev loop that redeploys in seconds keeps the cluster the place where the code runs.

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
| Rust or C source of the engine | full image rebuild, then a new pod | the engine links the system's C library and needs the toolchain; the runtime image (debian slim) has neither, by design (`dep.01` keeps images small) |
| a chart or a values file | redeploy the YAML | Tilt diffs the rendered objects |

For the Go path, two details decide whether it works: the kind node is Linux, so the host build needs `GOOS=linux` (a macOS binary dies with `exec format error`), and `CGO_ENABLED=0`, so the binary needs no libc in the image. Restarting the process is the `restart_process` extension's `docker_build_with_restart`, which wraps the entrypoint so a sync triggers a restart; it needs a shell in the image, so the dev image is alpine, not distroless.

### 2.3 Ordering

The gateway's `/readyz` waits on the engine. Started first, it fails readiness, restarts, and fills its log with 503s that hide real problems. `resource_deps` makes the order explicit. The same will hold for the durable server before its workers (`dep.06`).

### 2.4 The context guard and `tilt ci`

`k8s_context()` is the current kubectl context. Tilt deploys there, after a warning only for contexts it does not recognize as local. One `if k8s_context() != 'kind-forge': fail(...)` at the top makes a wrong context an error before anything is built. `tilt ci` evaluates the same Tiltfile, builds and deploys everything once, streams logs, and exits 0 when every resource is ready (non-zero when any fails); `ci_settings(timeout=...)` bounds it.

## 3. Worked example by hand

The solution's graph after you save `go/gateway/route/route.go` (times are typical for a laptop; yours will differ, the ratio will not):

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

**The name match.** The engine chart renders `image: "forge-engine:0.1.0"`. The ref `forge-engine` has the same repository, so Tilt deploys `forge-engine:tilt-<hash>` instead. Had the Tiltfile said `docker_build('engine', ...)`, Tilt would build `engine`, find no container using it, and deploy `forge-engine:0.1.0`, the image from the last manual build: your edit would never run, and no error would say so.

**The guard.** With `kubectl config use-context prod-east`, evaluating the solution Tiltfile stops at the guard with `deploy/Tiltfile deploys only to kind-forge; current context is prod-east (kubectl config use-context kind-forge)`.

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

`tilt up` runs one unified engine, which starts fastest. The production topology is one engine release per role (`<system>-engine-prefill`, and `<system>-engine-decode` with 2 replicas); the solution Tiltfile switches with a Tilt config flag, `tilt up -f deploy/Tiltfile -- --topology=disaggregated`, and makes the gateway wait on every engine release.

### What a correct solution must handle

| Case | Kind of check | What must hold | Why it matters downstream |
|---|---|---|---|
| the Tiltfile evaluates | static | it runs with `k8s_context() == 'kind-<system>'` | every `tilt up` and `tilt ci` |
| it refuses other clusters | fault | with any other context (say `prod-cluster`) it calls `fail()` | dev images never reach a cluster you did not mean |
| images match the charts | conformance | each chart's image repository has a `docker_build` of the same name | edits actually deploy |
| the engine rebuilds from its Dockerfile | static | engine: no live update, `deploy/docker/engine.Dockerfile`, repo-root context | the same image `dep.00` and CI build |
| the gateway live-updates a Linux binary | static | gateway: a sync, a restart, a `go build` with `CGO_ENABLED=0 GOOS=linux` that watches `../go` | seconds per Go change |
| the charts are deployed | static | `helm()` of both charts into `<system>`; both workloads rendered | the production charts, no dev copies |
| the gateway starts after the engine | static | `resource_deps` puts the engine first (and durable before workers, once they exist) | no crash loop at startup |
| CI has a budget | boundary | `ci_settings(timeout=...)` of at most 20 minutes | `dep.05`'s kind job fails fast |
| `tilt ci` on kind | cluster | `tilt ci` on `kind-<system>` exits 0 | the loop works end to end |

The static cases can be checked without Tilt or a cluster by evaluating the Tiltfile as Python with a recording fake for each builtin, and rendering the charts with the real `helm template`.

## 5. The solution, step by step

### 5.1 From first principles

1. **Only what a change reaches should rerun.** So the loop is a dependency graph of resources (2.1), and each resource declares exactly the files it depends on (`only`, `deps`, `ignore`).
2. **A built image must reach the workload that runs it.** Tilt pairs them by repository name, so each `docker_build` ref must equal the chart's `image.repository`: `forge-engine`, `forge-gateway`. No other mechanism links the two.
3. **The fastest correct path differs by language.** Go cross-compiles to a static Linux binary on the host in seconds, so the gateway's change is one file synced into a running container. Rust links native code and the runtime image has no toolchain, so the engine's change is a full rebuild with layer caching.
4. **Order follows readiness dependencies.** The gateway's readiness waits on the engine; durable workers poll the durable server. `resource_deps` encodes both, so startup has no crash loops.
5. **A deploy script must know where it may deploy.** A Tiltfile deploys to the current context, so it must fail on any other than `kind-<system>`, before building anything.
6. **CI must fail in minutes.** `tilt ci` without a budget waits 30 minutes on a stuck pod; `ci_settings(timeout='15m')` bounds it.

### 5.2 The guard and the budget

```python
load('ext://restart_process', 'docker_build_with_restart')

SYSTEM = 'forge'
CONTEXT = 'kind-' + SYSTEM

# Never deploy anywhere but your own kind cluster: a wrong current context
# would otherwise get your dev images and charts.
if k8s_context() != CONTEXT:
    fail('deploy/Tiltfile deploys only to %s; current context is %s (kubectl config use-context %s)' % (CONTEXT, k8s_context(), CONTEXT))
# Tilt warns before deploying to a context it does not recognize as local;
# naming it allowed removes the warning only after the guard has passed.
allow_k8s_contexts(CONTEXT)

# tilt ci gives up on a resource that never becomes ready after this long,
# instead of the default 30 minutes.
ci_settings(timeout='15m')
```

The guard comes before every `docker_build` and `k8s_yaml`, so a wrong context costs nothing but the error message, which names the fix.

### 5.3 The engine: rebuild the image

```python
# A Rust change needs the toolchain and a relink, so there is no live update:
# the image rebuilds (cargo's layer cache keeps that to the changed crates)
# and the pod is replaced. The image name matches the chart's image.repository,
# which is how Tilt knows to deploy the image it just built.
docker_build(
    SYSTEM + '-engine',
    '..',                                   # the repo root, as in dep.00
    dockerfile='docker/engine.Dockerfile',  # the production Dockerfile, not a dev copy
    only=['contracts', 'c', 'rust'],        # a docs edit does not rebuild the engine
    ignore=['rust/target', 'c/build'],      # build outputs never trigger a rebuild
)
```

`only` limits both what is watched and what is sent as context; `ignore` removes the build directories inside those trees, without which every `cargo build` on the host would retrigger the image build (pitfall 5).

### 5.4 The gateway: compile on the host, sync, restart

```python
# Cross-compile for the kind node (linux, no cgo), then copy the one file
# into the running container and restart the process: seconds, not a rebuild.
local_resource(
    'gateway-compile',
    'cd ../go && CGO_ENABLED=0 GOOS=linux go build -o ../deploy/tilt/build/gateway ./cmd/gateway',
    deps=['../go', '../contracts/go'],      # rerun when Go source changes
    ignore=['../go/bin'],
    labels=['build'],
)
docker_build_with_restart(
    SYSTEM + '-gateway',                    # = the gateway chart's image.repository
    'tilt',                                 # a tiny context: deploy/tilt/
    dockerfile='tilt/gateway.dev.Dockerfile',
    entrypoint=['/usr/local/bin/gateway'],  # wrapped, so a sync can restart it
    only=['build/gateway'],                 # the image depends on one file
    live_update=[sync('tilt/build/gateway', '/usr/local/bin/gateway')],
)
```

The dev image is the production runtime stage minus everything the host build replaced:

```dockerfile
# Tilt's restart_process live update needs a shell in the image, so the base
# is alpine, not distroless. Never pushed or released.
FROM alpine:3.19
COPY build/gateway /usr/local/bin/gateway
USER 10001:10001
EXPOSE 8080 9464
ENTRYPOINT ["/usr/local/bin/gateway"]
```

The chain is: a save under `go/` reruns `gateway-compile`; its output `deploy/tilt/build/gateway` is the only file in the image's `only`, so the image's live update fires; `sync` copies the binary into the container and the wrapped entrypoint restarts it. `deploy/tilt/build/` is git-ignored.

### 5.5 The charts and the graph

```python
# `tilt up` runs one unified engine (fast to start). The per-role topology:
#   tilt up -f deploy/Tiltfile -- --topology=disaggregated
config.define_string('topology')
TOPOLOGY = config.parse().get('topology', 'unified')
if TOPOLOGY == 'disaggregated':
    k8s_yaml(helm('helm/%s-engine' % SYSTEM, name=SYSTEM + '-engine-prefill', namespace=SYSTEM, set=['role=prefill']))
    k8s_yaml(helm('helm/%s-engine' % SYSTEM, name=SYSTEM + '-engine-decode', namespace=SYSTEM, set=['role=decode', 'replicaCount=2']))
    ENGINES = [SYSTEM + '-engine-prefill', SYSTEM + '-engine-decode']
else:
    k8s_yaml(helm('helm/%s-engine' % SYSTEM, name=SYSTEM + '-engine', namespace=SYSTEM))
    ENGINES = [SYSTEM + '-engine']
k8s_yaml(helm('helm/%s-gateway' % SYSTEM, name=SYSTEM + '-gateway', namespace=SYSTEM))
# ... (dep.06: the durable and worker charts, workers with autoscaling.enabled=false)

# The gateway's readiness waits on the engines; starting it first only
# produces a crash loop of 503s. Its image needs the compiled binary first.
for e in ENGINES:
    k8s_resource(e, labels=['serving'])
k8s_resource(SYSTEM + '-gateway', resource_deps=ENGINES + ['gateway-compile'],
             port_forwards=['8080:8080'], labels=['serving'])
# ... (dep.06: forge-worker after forge-durable)
```

`helm()` renders the same charts `dep.03` installs, values schema included, so the loop cannot drift from production. `ENGINES` is a list so the gateway waits on every engine release in either topology. Release names are the resource names, which is what `k8s_resource` refers to.

Full solution: [`Tiltfile`](../../../../course/solutions/entry/deploy/Tiltfile), [`gateway.dev.Dockerfile`](../../../../course/solutions/entry/deploy/tilt/gateway.dev.Dockerfile).

### 5.6 Why this solution

- **Host compile and sync over an in-cluster `go build`.** Building inside the container would need the Go toolchain in the dev image and a module cache per pod; the host already has both, warm.
- **A full rebuild for the engine over a synced binary.** A host-built Rust binary would link the host's glibc and C library; on macOS it would not even be a Linux binary. The image build is slower but always correct.
- **The production charts over dev manifests.** A separate dev YAML would drift; using `helm()` means a chart bug shows up in the loop first.
- **A `fail()` guard over `allow_k8s_contexts` alone.** `allow_k8s_contexts` only suppresses a warning for listed contexts; it does not stop a deploy to a context Tilt considers local or one you click through. `fail()` stops everything.
- **Tradeoff accepted:** the dev gateway image differs from the production one (no digest pin, no `HEALTHCHECK`, no labels). CI (`dep.05`) still builds and tests the production image.

### 5.7 Explain it back

1. The Tiltfile says `docker_build('engine', ...)`, the build is green, and the pod still runs old code. Why, and why is there no error?
   <details><summary>Answer</summary>Tilt substitutes its build only into containers whose image repository equals the ref. The chart's container uses <code>forge-engine</code>, so nothing matches <code>engine</code>; the pod keeps <code>forge-engine:0.1.0</code> from the last manual build. An unused image is not an error in Tilt's model.</details>
2. Why must the host build set `GOOS=linux` even when your laptop runs Linux?
   <details><summary>Answer</summary>On a Linux laptop it is redundant but harmless; on macOS the default would be a Darwin binary, which fails in the container with <code>exec format error</code>. Setting it makes the Tiltfile correct on every host.</details>
3. Why is the dev image alpine and not distroless?
   <details><summary>Answer</summary><code>docker_build_with_restart</code> wraps the entrypoint in a small restart helper that needs a shell. Distroless has none, so the restart after a sync would fail.</details>
4. What does `resource_deps=ENGINES + ['gateway-compile']` prevent?
   <details><summary>Answer</summary>Two failure modes at startup: the gateway image being built before the binary exists, and the gateway pod starting before any engine is ready, which fails <code>/readyz</code>, crash-loops, and floods the log with 503s.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. No context guard | `tilt up` in the wrong terminal deploys dev images to a shared cluster | `fail()` on any context but `kind-forge`, before any build (5.2) |
| 2. `docker_build('engine', ...)` while the chart runs `forge-engine` | Tilt shows a green build; the pod runs yesterday's image | refs `SYSTEM + '-engine'` and `SYSTEM + '-gateway'` equal the charts' repositories (5.3, 5.4) |
| 3. `go build` without `GOOS=linux` (on a Mac) or with cgo | the synced binary fails with `exec format error`, or `not found` for libc | `CGO_ENABLED=0 GOOS=linux` in `gateway-compile` (5.4) |
| 4. No `resource_deps` | the gateway restarts until the engine is up; real errors drown in 503s | `resource_deps=ENGINES + ['gateway-compile']` (5.5) |
| 5. Build context `..` without `only` or `.dockerignore` | every save under `rust/target` triggers an engine rebuild | `only` plus `ignore=['rust/target', 'c/build']`, and the `.dockerignore` from dep.00 (5.3) |

## 7. Where it's used next

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
