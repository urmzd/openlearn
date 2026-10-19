# The Tilt dev loop

While you build the serving platform, the engine and the gateway change every day: continuous batching, routing, rate limits, caching. On kind, each change is currently `docker build` (a minute for the gateway, several for the engine), `kind load docker-image`, `helm upgrade`, a wait for the rollout, and then a look at logs in another terminal, repeated for each service you touched and in the right order. That loop is slow enough that you stop testing on the cluster and start testing only locally. Then the bugs that exist only in the cluster, such as probes, Service names, the `/artifacts` mount, or the gateway reaching the engine, wait until the next full deployment.

A dev loop that redeploys in seconds keeps the cluster the place where the code runs. This section builds one with Tilt: a single `deploy/Tiltfile` that builds the engine image, live-updates a host-compiled gateway binary, deploys the production Helm charts into `<system>`, starts the gateway after the engine, refuses any kube context but `kind-<system>`, and gives `tilt ci` a time budget. A small dev image for the gateway, `deploy/tilt/gateway.dev.Dockerfile`, completes it.

The loop deploys the charts from [Helm charts and the observability stack](03-helm-charts-and-observability-stack.md), using the Dockerfiles from [the tracer deploy](00-tracer-deploy.md) and [Dockerfiles for the gateway and the engine](01-dockerfiles-for-gateway-and-engine.md), onto the cluster from [kind cluster, local registry, and port mappings](02-kind-cluster-and-local-registry.md). Its interface is the system's own charts and Dockerfiles plus the [Tiltfile API](https://docs.tilt.dev/api.html).

> **Learning objectives**
> - Model a dev loop as a graph of image builds, local commands, and Kubernetes workloads with explicit dependencies.
> - Explain how Tilt pairs a built image with a workload, and diagnose the silent failure when the names differ.
> - Choose between a live update (Go) and a full image rebuild (Rust) and justify the choice.
> - Guard a Tiltfile against deploying to the wrong cluster, and bound `tilt ci` with a budget.

## Running the loop

From the repo root:

```bash
tilt up -f deploy/Tiltfile            # the loop: http://localhost:10350 shows the graph
tilt ci -f deploy/Tiltfile            # once, as CI runs it (CI for your repo)
```

What Tilt does when you save a file depends on a small vocabulary, which is where to start.

## Resources and the graph

The loop is a **graph of resources** with explicit dependencies. Tilt watches files and reruns only what a change reaches. Five terms describe the graph:

| Term | Meaning |
|---|---|
| image target | how to build one image: `docker_build(ref, context, dockerfile=...)` |
| Kubernetes resource | one workload from the YAML you hand to `k8s_yaml`, named after it (`forge-engine`) |
| local resource | a command on your machine: `local_resource(name, cmd, deps=[...])`, rerun when a `deps` file changes |
| `resource_deps` | resources that must be ready before this one starts, in `k8s_resource(name, resource_deps=[...])` |
| live update | steps that patch a running container instead of replacing it: `sync(local, remote)`, `run(cmd)` |

Tilt matches images to workloads by name. Every container whose image repository **equals** a `docker_build` ref gets the image Tilt built, retagged with a content hash (`forge-engine:tilt-3f9c...`) and loaded into kind. The chart's own tag is ignored. A mismatch builds the new code and deploys the old image, silently; you will see exactly how below.

## Two kinds of change

Not every change deserves the same path. The fastest correct one depends on the language and on what the runtime image contains:

| Change | Fast path | Why |
|---|---|---|
| Go source of the gateway | `go build` on the host (2 s), `sync` the binary into the container, restart the process | Go cross-compiles to a static binary; the container needs nothing but the file |
| Rust or C source of the engine | full image rebuild, then a new Pod | the engine links the system's C library and needs the toolchain; the runtime image (Debian slim) has neither, by design, to keep images small |
| a chart or a values file | redeploy the YAML | Tilt diffs the rendered objects |

Two details decide whether the Go path works. The kind node is Linux, so the host build needs `GOOS=linux`; a macOS binary dies with `exec format error`. And it needs `CGO_ENABLED=0`, so the binary needs no libc in the image. Restarting the process is the job of the `restart_process` extension's `docker_build_with_restart`, which wraps the entrypoint so a sync triggers a restart. The wrapper needs a shell in the image, so the dev image is alpine, not distroless.

## Ordering

Fast paths are useless if services start in the wrong order. The gateway's `/readyz` waits on the engine. Started first, the gateway fails readiness, restarts, and fills its log with 503s that hide real problems. `resource_deps` makes the order explicit. The same will hold for the durable server before its workers, in [Durable and worker images and charts](06-durable-and-worker-charts.md).

## The context guard and `tilt ci`

A Tiltfile is code that deploys to whatever context kubectl points at. `k8s_context()` returns the current context, and Tilt deploys there after a warning only for contexts it does not recognize as local. One `if k8s_context() != 'kind-forge': fail(...)` at the top makes a wrong context an error before anything is built.

`tilt ci` evaluates the same Tiltfile, builds and deploys everything once, streams logs, and exits 0 when every resource is ready (non-zero when any fails). `ci_settings(timeout=...)` bounds it, so a stuck Pod fails CI in minutes, not half an hour.

## Timing both paths by hand

Here is the solution's graph after you save `go/gateway/route/route.go`. The times are typical for a laptop; yours will differ, but the ratio will not.

| Resource | Triggered by | Steps | Time |
|---|---|---|---|
| `gateway-compile` | `go/` changed (its `deps`) | `cd ../go && CGO_ENABLED=0 GOOS=linux go build -o ../deploy/tilt/build/gateway ./cmd/gateway` | 2.1 s |
| `forge-gateway` (image) | `deploy/tilt/build/gateway` changed (its `only`) | live update: `sync('tilt/build/gateway', '/usr/local/bin/gateway')`, restart the process | 0.6 s |
| `forge-engine` | nothing (no engine file changed) | none | 0 s |

That is about 3 seconds from save to the new gateway answering. Now edit `rust/crates/tl-engine/src/sched.rs`:

| Resource | Steps | Time |
|---|---|---|
| `forge-engine` (image) | `docker build -f docker/engine.Dockerfile ..` (contracts and C layers cached, `cargo build --release` of the changed crates) | 70 s |
| `forge-engine` | `kind load`, new ReplicaSet, readiness on `/healthz` | 12 s |
| `forge-gateway` | unchanged; its upstream is the Service, so it follows the new Pod | 0 s |

**The name match.** The engine chart renders `image: "forge-engine:0.1.0"`. The ref `forge-engine` has the same repository, so Tilt deploys `forge-engine:tilt-<hash>` instead. Had the Tiltfile said `docker_build('engine', ...)`, Tilt would build `engine`, find no container using it, and deploy `forge-engine:0.1.0`, the image from the last manual build. Your edit would never run, and no error would say so.

**The guard.** With `kubectl config use-context prod-east`, evaluating the solution Tiltfile stops at the guard with `deploy/Tiltfile deploys only to kind-forge; current context is prod-east (kubectl config use-context kind-forge)`.

## The shape of the Tiltfile

The whole interface fits on one screen. Paths are relative to `deploy/`:

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

`tilt up` runs one unified engine, which starts fastest. The production topology is one engine release per role (`<system>-engine-prefill`, and `<system>-engine-decode` with 2 replicas). The solution Tiltfile switches with a Tilt config flag, `tilt up -f deploy/Tiltfile -- --topology=disaggregated`, and makes the gateway wait on every engine release.

### What a correct solution must handle

The Tiltfile must evaluate with `k8s_context() == 'kind-<system>'`, since every `tilt up` and `tilt ci` depends on it, and with any other context (say `prod-cluster`) it must call `fail()`, so dev images never reach a cluster you did not mean. Each chart's image repository must have a `docker_build` of the same name, so edits actually deploy. The engine must have no live update and must build from `deploy/docker/engine.Dockerfile` with the repo-root context, the same image the tracer deploy and CI build. The gateway must have a sync, a restart, and a `go build` with `CGO_ENABLED=0 GOOS=linux` that watches `../go`, which is what makes a Go change take seconds. Both charts must be deployed through `helm()` into `<system>` with both workloads rendered: the production charts, no dev copies. `resource_deps` must put the engine first (and durable before workers, once they exist), so there is no crash loop at startup. `ci_settings(timeout=...)` must be at most 20 minutes, so the kind job in CI fails fast. And end to end, `tilt ci` on `kind-<system>` must exit 0.

The static cases can be checked without Tilt or a cluster, by evaluating the Tiltfile as Python with a recording fake for each builtin and rendering the charts with the real `helm template`.

## Building it step by step

### Reasoning from first principles

1. **Only what a change reaches should rerun.** So the loop is a dependency graph of resources, and each resource declares exactly the files it depends on (`only`, `deps`, `ignore`).
2. **A built image must reach the workload that runs it.** Tilt pairs them by repository name, so each `docker_build` ref must equal the chart's `image.repository`: `forge-engine`, `forge-gateway`. No other mechanism links the two.
3. **The fastest correct path differs by language.** Go cross-compiles to a static Linux binary on the host in seconds, so the gateway's change is one file synced into a running container. Rust links native code and the runtime image has no toolchain, so the engine's change is a full rebuild with layer caching.
4. **Order follows readiness dependencies.** The gateway's readiness waits on the engine; durable workers poll the durable server. `resource_deps` encodes both, so startup has no crash loops.
5. **A deploy script must know where it may deploy.** A Tiltfile deploys to the current context, so it must fail on any context other than `kind-<system>`, before building anything.
6. **CI must fail in minutes.** `tilt ci` without a budget waits 30 minutes on a stuck Pod; `ci_settings(timeout='15m')` bounds it.

### The guard and the budget

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

### The engine: rebuild the image

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

`only` limits both what is watched and what is sent as context. `ignore` removes the build directories inside those trees; without it, every `cargo build` on the host would retrigger the image build.

### The gateway: compile on the host, sync, restart

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

Follow the chain. A save under `go/` reruns `gateway-compile`. Its output, `deploy/tilt/build/gateway`, is the only file in the image's `only`, so the image's live update fires. `sync` copies the binary into the container, and the wrapped entrypoint restarts it. `deploy/tilt/build/` is git-ignored.

### The charts and the graph

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

`helm()` renders the same charts that [Helm charts and the observability stack](03-helm-charts-and-observability-stack.md) installs, values schema included, so the loop cannot drift from production. `ENGINES` is a list so the gateway waits on every engine release in either topology. Release names are the resource names, which is what `k8s_resource` refers to.

The full solution is [`Tiltfile`](../../../../course/solutions/entry/deploy/Tiltfile) and [`gateway.dev.Dockerfile`](../../../../course/solutions/entry/deploy/tilt/gateway.dev.Dockerfile).

### Why this design

- **Host compile and sync over an in-cluster `go build`.** Building inside the container would need the Go toolchain in the dev image and a module cache per Pod; the host already has both, warm.
- **A full rebuild for the engine over a synced binary.** A host-built Rust binary would link the host's glibc and C library; on macOS it would not even be a Linux binary. The image build is slower but always correct.
- **The production charts over dev manifests.** A separate dev YAML would drift; using `helm()` means a chart bug shows up in the loop first.
- **A `fail()` guard over `allow_k8s_contexts` alone.** `allow_k8s_contexts` only suppresses a warning for listed contexts; it does not stop a deploy to a context Tilt considers local or one you click through. `fail()` stops everything.
- **Tradeoff accepted:** the dev gateway image differs from the production one (no digest pin, no `HEALTHCHECK`, no labels). [CI for the platform repo](05-ci-for-your-repo.md) still builds and tests the production image.

## What can go wrong

Most Tilt failures are quiet: the dashboard is green and the cluster is wrong. These five cause most of them:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| No context guard | `tilt up` in the wrong terminal deploys dev images to a shared cluster | `fail()` on any context but `kind-forge`, before any build |
| `docker_build('engine', ...)` while the chart runs `forge-engine` | Tilt shows a green build; the Pod runs yesterday's image | refs `SYSTEM + '-engine'` and `SYSTEM + '-gateway'` equal the charts' repositories |
| `go build` without `GOOS=linux` (on a Mac) or with cgo | the synced binary fails with `exec format error`, or `not found` for libc | `CGO_ENABLED=0 GOOS=linux` in `gateway-compile` |
| No `resource_deps` | the gateway restarts until the engine is up; real errors drown in 503s | `resource_deps=ENGINES + ['gateway-compile']` |
| Build context `..` without `only` or `.dockerignore` | every save under `rust/target` triggers an engine rebuild | `only` plus `ignore=['rust/target', 'c/build']`, and the `.dockerignore` from the tracer deploy |

## Where this leads

The loop deploys the charts of [Helm charts and the observability stack](03-helm-charts-and-observability-stack.md) into the cluster of [kind cluster, local registry, and port mappings](02-kind-cluster-and-local-registry.md), with the Dockerfiles of [the tracer deploy](00-tracer-deploy.md) and [Dockerfiles for the gateway and the engine](01-dockerfiles-for-gateway-and-engine.md). Next, [CI for the platform repo](05-ci-for-your-repo.md) runs `tilt ci -f deploy/Tiltfile` in its kind job, and the deployment it produces is the production platform. The durable and worker charts of [Durable and worker images and charts](06-durable-and-worker-charts.md) join the graph, durable first, and every later chart does the same. The monitors and collector of [Metrics and log correlation](../../systems/04-observability/02-metrics-and-log-correlation.md) in *Systems and Architecture* can be applied from the same Tiltfile.

The local pieces have production cousins:

| Your piece | Production equivalent | What it adds |
|---|---|---|
| Tilt on kind | Skaffold, Garden, DevSpace | the same loop with other tradeoffs (profiles, remote clusters, test stages) |
| host compile and sync | remote dev containers, Telepresence | run one service locally against the cluster |
| `kind load` of every image | a local registry (`localhost:5001`) | faster loads, the same flow as a real registry |
| one Tiltfile | Tilt extensions and `include()` | per-team Tiltfiles composed into one graph |

### Further reading

- [Tilt docs: Tiltfile concepts, live update](https://docs.tilt.dev/)
- [restart_process extension](https://github.com/tilt-dev/tilt-extensions/tree/master/restart_process)
- [Skaffold](https://skaffold.dev/)
- [Telepresence](https://www.telepresence.io/)
- [kind local registry](https://kind.sigs.k8s.io/docs/user/local-registry/)
- [tilt-extensions](https://github.com/tilt-dev/tilt-extensions)

## Key terms

- **resource graph**: Tilt's model of image builds, local commands, and workloads, with dependencies, rerun only where a change reaches.
- **image target**: a `docker_build` declaration; Tilt substitutes its image into containers whose repository equals the ref.
- **local resource**: a command on your machine that reruns when its `deps` change.
- **live update**: patching a running container (sync files, run commands) instead of replacing the Pod.
- **`resource_deps`**: the resources that must be ready before another starts.
- **context guard**: a `fail()` at the top of the Tiltfile that refuses any kube context but the intended one.
- **`tilt ci`**: a one-shot run of the graph that exits 0 when every resource is ready, bounded by `ci_settings(timeout=...)`.

## Summary

- The loop is a graph of resources (image builds, local commands, Kubernetes workloads) with explicit dependencies; Tilt reruns only what a change reaches.
- Tilt deploys a freshly built image only into containers whose image repository equals the `docker_build` ref; a mismatch silently deploys the old image.
- Go: compile on the host for `linux` with `CGO_ENABLED=0`, sync the one binary, restart the process, in seconds. Rust: rebuild the image, because the runtime image has no toolchain.
- A Tiltfile deploys to whatever context kubectl points at, so make it refuse every context but `kind-<system>`.
- `tilt ci` runs the same graph once and exits 0 when every resource is ready; a budget makes a stuck Pod fail CI in minutes.

## Review questions

1. The Tiltfile says `docker_build('engine', ...)`, the build is green, and the Pod still runs old code. Why, and why is there no error?
   <details><summary>Answer</summary>Tilt substitutes its build only into containers whose image repository equals the ref. The chart's container uses <code>forge-engine</code>, so nothing matches <code>engine</code>; the Pod keeps <code>forge-engine:0.1.0</code> from the last manual build. An unused image is not an error in Tilt's model.</details>
2. Why must the host build set `GOOS=linux` even when your laptop runs Linux?
   <details><summary>Answer</summary>On a Linux laptop it is redundant but harmless; on macOS the default would be a Darwin binary, which fails in the container with <code>exec format error</code>. Setting it makes the Tiltfile correct on every host.</details>
3. Why is the dev image alpine and not distroless?
   <details><summary>Answer</summary><code>docker_build_with_restart</code> wraps the entrypoint in a small restart helper that needs a shell. Distroless has none, so the restart after a sync would fail.</details>
4. What does `resource_deps=ENGINES + ['gateway-compile']` prevent?
   <details><summary>Answer</summary>Two failure modes at startup: the gateway image being built before the binary exists, and the gateway Pod starting before any engine is ready, which fails <code>/readyz</code>, crash-loops, and floods the log with 503s.</details>
