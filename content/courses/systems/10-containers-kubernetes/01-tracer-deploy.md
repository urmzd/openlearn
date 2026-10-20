# Tracer deploy: images, a kind cluster, two Helm charts, and Jaeger

By the time you reach this section, the engine and the gateway stream tokens on your laptop, started by hand. That is a fragile place to stop. Nothing restarts them when they crash, nothing but your shell knows where the model file is, and the only way to reach them is through whatever port your laptop happened to have free. Everything that follows assumes a real deployment: [One trace](../11-observability/01-one-trace.md) in this book reads a trace out of a Jaeger running next to the two servers, [First drill: the engine crashloops](../12-incident-response-and-chaos/01-first-drill.md) breaks the engine on purpose and walks you through bringing it back, and the rest of this chapter grows the same cluster into the serving platform.

This section turns the two programs into two container images, two Helm releases, and one kind cluster with Jaeger beside them. It is the last link of the tracer: once it works, every layer from training a model to a streamed completion runs end to end on a cluster. It builds on [Containers and Kubernetes](../../programming-and-algorithms/01-language-primers/07-containers-and-kubernetes.md) in *Programming and Algorithms*, which defined images, layers, Pods, Deployments, Services, probes, Helm, and kind on a toy server. It packages the entry points of three programs you have already read: the `tinyllm` CLI from [Byte bigram](../../ai-engineering/06-numerics-and-autograd/01-byte-bigram.md) and the engine from [Your first endpoint](../../ai-engineering/23-model-serving/01-your-first-endpoint.md), both in *AI Engineering*, and [the tracer gateway](../../ai-engineering/25-ai-gateway/01-streaming-proxy.md) in *AI Engineering*. The [chapter opening](README.md) covers the general Kubernetes model this section relies on.

There is no contract of its own here. The charts run the tracer forms of the command lines in [`spec/cli-roles.md`](../../../../course/contracts/spec/cli-roles.md) on the fixed in-cluster ports described below, and the required file layout is laid out in [The layout every file must satisfy](#the-layout-every-file-must-satisfy). The solution's system is named `forge`; read `forge` wherever the text says `<system>`. As you go, try to predict each file before you read it.

> **Learning objectives**
> - Build the engine and gateway images from the repo root with a `.dockerignore`, multi-stage builds, a numeric non-root user, and an exec-form `ENTRYPOINT`.
> - Trace one request through all nine hops from your machine to the engine and back, naming the file that makes each hop work.
> - Explain the three-mount chain that carries the model from your disk into the engine Pod, and diagnose a break in it.
> - Choose readiness and liveness probes that match each server's dependencies, and keep the API key out of every file you commit.
> - Name the five coordinates that every later tool uses to operate on the cluster.

## The whole deployment in eleven commands

It helps to see the destination first. The solution comes up from the repo root with these commands, all of them the system's own tools:

```bash
{tinyllm} train bigram --data <any text file> --out artifacts/models/bigram
docker build -f deploy/docker/engine.Dockerfile  -t forge-engine:0.1.0 .
docker build -f deploy/docker/gateway.Dockerfile -t forge-gateway:0.1.0 .
kind create cluster --config deploy/kind/cluster.yaml
kind load docker-image forge-engine:0.1.0 forge-gateway:0.1.0 --name forge
kubectl create namespace forge
kubectl -n forge create secret generic forge-api-key --from-literal=api-key="$TL_API_KEY"
kubectl -n forge apply -f deploy/observability/jaeger.yaml
helm upgrade --install forge-engine  deploy/helm/forge-engine  -n forge
helm upgrade --install forge-gateway deploy/helm/forge-gateway -n forge
curl -sN http://127.0.0.1:30080/v1/completions -H "Authorization: Bearer $TL_API_KEY" \
  -H 'Content-Type: application/json' -d '{"model":"tracer","prompt":"Once","max_tokens":32,"stream":true}'
```

Each line hides a decision. The rest of this section adds only what changes when the programs in the toy examples become real ones.

## Build contexts at the repo root

The engine needs `rust/`. The gateway needs `go/` and `contracts/`: its `go.mod` replaces the contracts module with `../contracts/go`, a directory that arrives later with the generated contract code. `COPY contracts/ contracts/` copies whatever is there today, and copying a path that does not exist fails the build. Both images therefore build with the **repo root** as their context: `docker build -f deploy/docker/engine.Dockerfile .`.

That choice has a cost you must pay deliberately. The builder sends the whole context to the Docker daemon before the first instruction runs, and `rust/target/` alone can be several gigabytes. So a **`.dockerignore`** at the root, in the same syntax as `.gitignore`, must exclude build outputs (`**/target`), local state (`.ol`, `.venv`), data (`artifacts`), and anything secret. With that in place, each Dockerfile copies only the directories its stage needs.

**The engine image.** Stage 1 starts from `rust:<version>-slim-bookworm` and runs `cargo build --release -p tl-serve`. Stage 2 uses `debian:bookworm-slim`. The copy order follows change frequency: `contracts/` first, then `rust/` and `cargo build`.

**The gateway image.** Stage 1 starts from `golang:<version>-alpine`, copies `contracts/` (`COPY contracts/ contracts/`) and `go/go.mod` (plus `go.sum` if there is one), and runs `go mod download`, which stays cached until the module files change. Then it copies `go/` and builds `./cmd/gateway` with `CGO_ENABLED=0`, which gives a static binary. Stage 2 is `alpine` (or distroless) plus that binary.

Both final stages share three rules: a numeric non-root `USER`, an `EXPOSE` of the ports in the topology below, and an exec-form `ENTRYPOINT` with **no arguments**. The chart passes the arguments (`args:` in the Pod spec), and only the exec form receives them.

## The topology on kind

With the images defined, the next question is what runs them. Everything lives in the namespace `<system>`:

| Object | Kind | Ports | Made by |
|---|---|---|---|
| `<system>-engine` | Deployment, 1 replica | 8000 HTTP, 9464 health | Helm release `<system>-engine` |
| `<system>-engine` | Service, ClusterIP | 8000 | same release |
| `<system>-gateway` | Deployment, 1 replica | 8080 HTTP, 9464 health | Helm release `<system>-gateway` |
| `<system>-gateway` | Service, NodePort | 8080, nodePort **30080** | same release |
| `jaeger` | Deployment | 4317, 4318 (OTLP in), 16686 (query) | `deploy/observability/jaeger.yaml` |
| `jaeger` | Service, ClusterIP | 4317, 4318 | same file |
| `jaeger-query` | Service, NodePort | 16686, nodePort **30686** | same file |
| `<system>-api-key` | Secret | key `api-key` | you, by hand, once |

Ports inside the cluster are fixed. Every Pod has its own IP, so two Pods can both listen on 8000 without colliding, and fixed ports mean no chart has to discover anything. Only a run on your laptop, where every process shares one network namespace, needs free ports. The release name is the object name (`{{ .Release.Name }}`), so `helm install forge-engine` makes `deploy/forge-engine`, which is exactly what the first drill patches.

**Service DNS.** Every Service gets a DNS name, `<service>.<namespace>.svc.cluster.local`, and inside the same namespace the short name `<service>` resolves too. The gateway's `--upstream` is therefore `http://<system>-engine:8000`. No IP address ever appears in a chart, and a restarted engine Pod with a new IP is reached at the same name.

## The artifacts chain

The model is data, not code, so it is not in the image. It travels from your disk to the engine through three mounts:

| Where | Path | Declared by |
|---|---|---|
| your machine | `./artifacts/models/bigram/` (from `{tinyllm} train bigram --out ...`) | you |
| the kind node container | `/artifacts/models/bigram/` | `extraMounts` in `deploy/kind/cluster.yaml` (a relative `hostPath` is resolved against the directory you run `kind create cluster` from) |
| the engine Pod | `/artifacts/models/bigram/` | a `hostPath` volume of the node's `/artifacts`, mounted at `/artifacts` |

The engine's `--model-dir` is the Pod path. Break any link and the engine crashloops, which is precisely the failure the first drill rehearses. A `hostPath` volume is a single-node simplification; the closing section names what replaces it in production.

## Secrets

The model is the one piece of data that must reach the engine; the API key is the one piece that must reach the gateway without appearing anywhere else. A **Secret** is a Kubernetes object holding small byte strings, here the API key. A container reads it as an environment variable through `valueFrom.secretKeyRef: {name, key}`. You create the Secret once, by hand, from the `TL_API_KEY` variable in your shell, so the key's value lives in exactly one place and appears in no file you commit: not the chart, not `values.yaml`, not the image, and not `helm get values`.

> **Note:** Kubernetes stores Secrets base64-encoded, not encrypted. Access control and encryption at rest come later, in [Helm charts and the observability stack](04-helm-charts-and-observability-stack.md) and in [Secrets and authorization review](../../software-engineering/09-security/03-secrets-and-authorization-review.md) in *Software Engineering*.

## Probes that match the roles

Probes are where a deployment encodes what each program depends on. The two servers have different dependencies, so they get different probes:

| Container | readiness | liveness | Why |
|---|---|---|---|
| engine | `GET /healthz` on 9464 | `GET /healthz` on 9464 | the engine has no dependency; healthy is ready |
| gateway | `GET /readyz` on 9464 | `GET /healthz` on 9464 | `/readyz` answers 503 while the engine is unreachable, so the gateway leaves its Service instead of returning errors; liveness must not follow the engine, or an engine outage restarts the gateway too |

In short, the gateway is **ready** only when the engine is, and **live** whenever it is itself up.

## Telemetry wiring

Both servers read the standard OpenTelemetry variables defined in `spec/cli-roles.md`: `OTEL_EXPORTER_OTLP_ENDPOINT` (the base URL; unset means export nothing) and `OTEL_SERVICE_NAME`. The charts set the endpoint to `http://jaeger:4318`, Jaeger's OTLP/HTTP port reached through its Service, and the service names to the release names, `<system>-engine` and `<system>-gateway`. [One trace](../11-observability/01-one-trace.md) makes the export actually happen and looks the trace up by those names.

## The deployment's coordinates

Anything that operates on the cluster later, whether a trace lookup, a drill, or a load test, needs five facts. They follow directly from the files above:

| Coordinate | Value for `forge` | Comes from |
|---|---|---|
| kube context | `kind-forge` | `kind create cluster` with `name: forge` |
| namespace | `forge` | `kubectl create namespace forge` |
| gateway URL | `http://127.0.0.1:30080` | the gateway NodePort, mapped by kind |
| traces URL | `http://127.0.0.1:30686` | the Jaeger query NodePort |
| workloads | `deploy/forge-engine`, `deploy/forge-gateway` | the release names |

Every later operation on the cluster uses exactly these. Naming the context and namespace explicitly is a safety property: a tool that breaks things on purpose, like the incident drills in this book, must refuse any other context, or it could patch a cluster you did not mean to touch.

## Following one request by hand

Before you run anything, follow one request through every hop with `<system> = forge`. Each line is checkable against the files of the layout that follows.

```bash
curl -sN http://127.0.0.1:30080/v1/completions -H "Authorization: Bearer $TL_API_KEY" -d '...'
```

| # | Hop | Address | Because of |
|---|---|---|---|
| 1 | your machine to the kind node container | `127.0.0.1:30080` to node `:30080` | `extraPortMappings` 30080 to 30080 |
| 2 | node port to Service | node `:30080` to `forge-gateway:8080` | Service `type: NodePort`, `nodePort: 30080`, `port: 8080` |
| 3 | Service to gateway Pod | ClusterIP `:8080` to Pod IP `:8080` | selector `app.kubernetes.io/name: forge-gateway`, `targetPort: http`, container port `http` = 8080 |
| 4 | gateway checks the key | `Authorization: Bearer <key>` vs env `TL_API_KEY` | `secretKeyRef {name: forge-api-key, key: api-key}` |
| 5 | gateway to engine Service | `http://forge-engine:8000/v1/completions` | `--upstream http://forge-engine:8000`, DNS `forge-engine.forge.svc.cluster.local` |
| 6 | Service to engine Pod | ClusterIP `:8000` to Pod IP `:8000` | Service `forge-engine`, `targetPort: http` = 8000 |
| 7 | engine reads the model | `/artifacts/models/bigram/model.safetensors` | `--model-dir`, the `hostPath` volume, `extraMounts`, `train bigram` |
| 8 | tokens stream back | SSE, one `data:` line per token, then `data: [DONE]` | the gateway copies and flushes each chunk ([the tracer gateway](../../ai-engineering/25-ai-gateway/01-streaming-proxy.md)) |
| 9 | spans leave both Pods | `http://jaeger:4318/v1/traces` | `OTEL_EXPORTER_OTLP_ENDPOINT`, Service `jaeger` port 4318 |

The two health ports never appear in that path. The kubelet probes `:9464` on each Pod directly, and hops 3 and 6 happen only while those probes pass.

### Rendering one template by hand

The gateway chart's `values.yaml` holds `port: 8080`, `healthPort: 9464`, and `upstream: ""`, and its template computes a default:

```yaml
{{- $upstream := .Values.upstream | default (printf "http://%s-engine:8000" (trimSuffix "-gateway" .Release.Name)) -}}
args: ["--port", {{ .Values.port | quote }}, "--health-port", {{ .Values.healthPort | quote }}, "--upstream", {{ $upstream | quote }}]
```

Work it through for `helm install forge-gateway`. `.Release.Name` is `forge-gateway`; `trimSuffix "-gateway"` gives `forge`; `printf` gives `http://forge-engine:8000`; `upstream` is empty, so `default` picks the computed value. The rendered list is `["--port", "8080", "--health-port", "9464", "--upstream", "http://forge-engine:8000"]`. The `quote` matters: `args` must be strings, and a bare `8080` would render as a YAML integer, which Kubernetes rejects. Running `helm template forge-gateway deploy/helm/forge-gateway -n forge` prints exactly this.

### Checking the rollout by hand

With one replica each and the default strategy, `maxSurge` $= \lceil 0.25 \times 1 \rceil = 1$ and `maxUnavailable` $= \lfloor 0.25 \times 1 \rfloor = 0$, where $\lceil x \rceil$ rounds up and $\lfloor x \rfloor$ rounds down to an integer. An upgrade therefore starts the new engine Pod, waits for its readiness probe, and only then stops the old one: no gap, provided the new Pod becomes ready at all. When it does not, as in the first drill, the old Pod keeps serving and the rollout stalls, which is the safe failure.

## The layout every file must satisfy

The hand trace depends on specific names and fields, so the layout is fixed. In the table, `<system>` is the system's name, `forge` in the solution:

| Path | Requirement |
|---|---|
| `.dockerignore` | excludes `target` (any depth), `.ol`, and `artifacts` |
| `deploy/docker/engine.Dockerfile` | context: repo root; two or more stages, every `FROM` pinned; final stage `USER` numeric non-root, `EXPOSE 8000` and `9464`, exec-form `ENTRYPOINT` of `tl-serve`; the image takes the tracer flags of `spec/cli-roles.md` as arguments |
| `deploy/docker/gateway.Dockerfile` | the same rules; `EXPOSE 8080` and `9464`; `go/cmd/gateway` |
| `deploy/kind/cluster.yaml` | `name: <system>`; mappings 30080 to 30080 and 30686 to 30686; an `extraMounts` entry with `containerPath: /artifacts` |
| `deploy/helm/<system>-engine/` | `helm lint` clean; rendered as release `<system>-engine`: Deployment `<system>-engine` with `--model-dir` under `/artifacts/`, `--port 8000`, `--health-port 9464`, a `hostPath` volume of `/artifacts` mounted at `/artifacts`, readiness and liveness on `/healthz` at 9464, `resources.limits`, `runAsNonRoot`, a pinned tag with a pull policy other than `Always`, `OTEL_SERVICE_NAME=<system>-engine`, `OTEL_EXPORTER_OTLP_ENDPOINT` set; Service `<system>-engine` targeting 8000 |
| `deploy/helm/<system>-gateway/` | the same pod rules with readiness `/readyz` and liveness `/healthz` at 9464, `OTEL_SERVICE_NAME=<system>-gateway`; `--port 8080`, `--health-port 9464`, `--upstream http://<system>-engine:8000`; `TL_API_KEY` only through `secretKeyRef`; Service `<system>-gateway`, `type: NodePort`, `nodePort: 30080` targeting 8080; no file in either chart holds a `tl_<id>_<secret>` key |
| `deploy/observability/jaeger.yaml` | one Deployment of a pinned Jaeger image; a Service port 4318; a NodePort Service on 30686 |
| on kind | both Deployments available in namespace `<system>`; 32 or more chunks through the gateway URL; `<traces URL>/api/services` answers |

### What a correct solution must handle

You can prove the images on plain Docker before any cluster exists. Train a throwaway model, copy it into the engine container at `/artifacts/models/bigram` (copied, not bind-mounted, because Docker Desktop shares only some host directories), and run both images on a private Docker network with the engine under the alias `engine`.

The checks fall into a progression. Static checks come first: the layout is present, build outputs (`target`, `.ol`, `artifacts`) stay out of the context, and the Dockerfiles follow the image rules from *Programming and Algorithms* (stages, pins, user, ports, exec form). The kind config must have the name, both mappings, and the `/artifacts` mount, so that 30080 and 30686 are reachable from your machine. Both charts must pass `helm lint`, which catches template errors before install. The engine chart must render its args, volume, probes, limits, user, image, OTel environment, and Service, so that the engine starts and is found; the gateway chart must render its args, probes, OTel environment, and NodePort 30080, so that it reaches the engine and is itself reachable. The API key must come only from `secretKeyRef`, with no literal key in either chart; [Secrets and authorization review](../../software-engineering/09-security/03-secrets-and-authorization-review.md) audits this later and [Helm charts and the observability stack](04-helm-charts-and-observability-stack.md) enforces it. The Jaeger manifest must have a pinned image, port 4318, and NodePort 30686, because [One trace](../11-observability/01-one-trace.md) exports there and reads from there.

Then come the Docker checks. Both images must build from the repo root, exactly as CI later builds them in [CI for the platform repo](06-ci-for-your-repo.md). Their `User` field must be numeric non-root, which is what the kubelet's `runAsNonRoot` check needs. The containers must stream a completion, 24 chunks then `[DONE]` through the gateway container, proving the images work before any cluster. The gateway container must answer 401 to a request with no key, proving the key check survived packaging. And both containers must exit 0 within an 8 s grace period on `SIGTERM`, which rollouts and the first drill depend on.

Finally, the cluster checks: both Deployments have an available replica, which means the whole chart worked; the gateway NodePort returns 32 or more chunks with the key, the end-to-end tracer on a cluster; and `GET <traces>/api/services` at the Jaeger query port returns JSON, which the trace lookup needs.

## Building it step by step

The solution files are the final versions of these artifacts. Later sections pin base images by digest and add `HEALTHCHECK` and labels ([Dockerfiles for the gateway and the engine](02-dockerfiles-for-gateway-and-engine.md)), add more port mappings ([kind cluster, local registry, and port mappings](03-kind-cluster-and-local-registry.md)), and add values such as a pepper Secret, a usage ledger, and a `--config` form. The listings below keep the lines this section needs and mark the later additions.

### Reasoning from first principles

Seven constraints fix almost every line of the solution, and it is worth deriving them before reading any file.

1. **The programs must run where your laptop is not.** So each becomes an image, and an image must contain exactly the binary and nothing machine-specific. That forces a multi-stage build (toolchain in stage 1, binary only in stage 2) and a `.dockerignore` that keeps `target/` and `artifacts/` out of the context.
2. **The cluster, not the image, decides how a program is started.** Arguments differ per deployment (model directory, upstream), so the image carries an `ENTRYPOINT` with no arguments and the chart supplies `args:`. Only the exec form passes `args:` through and makes the server PID 1, so `SIGTERM` reaches it.
3. **The kubelet must be able to prove the container is not root.** `runAsNonRoot` can only check a numeric `USER`; a name like `app` cannot be verified without reading `/etc/passwd` in the image. So `USER 10001:10001`.
4. **The model is data and is large.** Baking it into the image would rebuild the image on every training run. So it rides the three-mount chain, and `--model-dir` names the Pod path.
5. **Pods are replaced, and IPs change.** So the gateway names the engine by Service DNS, and the default upstream is derived from the release name: one rule (`<system>-gateway` talks to `<system>-engine`) instead of a value to keep in sync.
6. **A secret must never be in a file that is copied.** Charts, values, and images are all copied, to git, to `helm get values`, to registries. So the key enters only through `secretKeyRef` to a Secret created out of band.
7. **Probes encode dependencies.** Readiness may follow a dependency (stop sending me traffic I cannot serve); liveness must not (restarting me does not fix the engine). That fixes the probe table above.

### The build context

```gitignore
# The build context is the repo root. Keep build outputs, local state, and
# secrets out: they are large, machine-specific, and must never end up in a layer.
.git
.ol
artifacts
# **/ means any depth: rust/target, but also a stray crate's target/
**/target
c/build
**/.venv
**/__pycache__
**/*.pyc
.env
*.pem
```

The `**/` prefix matters: a bare `target` matches only `./target`, and the multi-gigabyte one is `rust/target`. `.env` and `*.pem` are there because "anything secret" has to be listed, not hoped for. The full solution is [`.dockerignore`](../../../../course/solutions/entry/.dockerignore).

### The engine image

```dockerfile
# Stage 1: the toolchain. (dep.01 later appends @sha256:... to each FROM.)
FROM rust:1.88-slim-bookworm AS build
WORKDIR /src
# Least to most often changed: contracts, then Rust. A Rust edit
# invalidates only the layers from COPY rust/ on.
COPY contracts/ contracts/
COPY rust/ rust/
RUN cargo build --release --manifest-path rust/Cargo.toml -p tl-serve \
 && cp rust/target/release/tl-serve /tl-serve

# Stage 2: the same glibc family as the builder (bookworm), so the
# dynamically linked binary finds the libraries it was linked against.
FROM debian:bookworm-slim
# ... (dep.01: ARG VERSION, ARG REVISION, OCI LABELs)
COPY --from=build /tl-serve /usr/local/bin/tl-serve
# Numeric, so Kubernetes runAsNonRoot can verify it.
USER 10001:10001
# 8000: POST /v1/completions; 9464: /healthz.
EXPOSE 8000 9464
# ... (dep.01: HEALTHCHECK)
# Exec form: tl-serve is PID 1, gets SIGTERM, and receives the chart's args.
ENTRYPOINT ["/usr/local/bin/tl-serve"]
```

The `cp` to `/tl-serve` in the same `RUN` gives stage 2 a short, fixed path to copy from. The runtime base matches the builder's distribution because Rust links glibc dynamically; a `musl` target or a static build would be the alternative, at the cost of a different toolchain. The full solution is [`engine.Dockerfile`](../../../../course/solutions/entry/deploy/docker/engine.Dockerfile).

### The gateway image

```dockerfile
FROM golang:1.25-alpine AS build
WORKDIR /src
# The module files first: `go mod download` stays cached until they change.
COPY contracts/ contracts/
COPY go/go.mod go/go.sum* go/
# go.sum* matches zero or one file, so the COPY works before go.sum exists.
# The workspace names both modules, because go.mod replaces the contracts
# module with ../contracts/go.
RUN go work init ./go ./contracts/go && cd go && go mod download
COPY go/ go/
# CGO_ENABLED=0: a static binary that needs no libc at run time.
# -trimpath and -s -w: no build paths, no symbol table; a smaller image.
RUN cd go && CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/gateway ./cmd/gateway

FROM alpine:3.19
# ... (dep.01: ARG, LABEL)
COPY --from=build /out/gateway /usr/local/bin/gateway
USER 10001:10001
# 8080: the API; 9464: /healthz and /readyz.
EXPOSE 8080 9464
# ... (dep.01: HEALTHCHECK with busybox wget)
ENTRYPOINT ["/usr/local/bin/gateway"]
```

Splitting `COPY go/go.mod` from `COPY go/` is the whole caching trick: dependencies change weekly, source changes every minute, and the download layer survives every source edit. The full solution is [`gateway.Dockerfile`](../../../../course/solutions/entry/deploy/docker/gateway.Dockerfile).

### The kind cluster

```yaml
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
name: forge                      # the kube context becomes kind-forge
# ... (dep.02: containerdConfigPatches for the local registry)
nodes:
  - role: control-plane
    # Port mappings are fixed when the node container is created.
    # 127.0.0.1 only: nothing is exposed to your LAN.
    extraPortMappings:
      - containerPort: 30080   # gateway NodePort
        hostPort: 30080
        listenAddress: "127.0.0.1"
      - containerPort: 30686   # Jaeger query NodePort (Pass 1)
        hostPort: 30686
        listenAddress: "127.0.0.1"
      # ... (dep.02: 30090 Prometheus, 30300 Grafana, 30320 Tempo, 30733 durable)
    # ./artifacts on your machine is /artifacts on the node; pods mount it
    # with a hostPath volume. Relative: run kind from the repo root.
    extraMounts:
      - hostPath: ./artifacts
        containerPath: /artifacts
        readOnly: true
```

`readOnly: true` encodes that serving never writes the model. The node is a Docker container, so these two lists are the only doors between your machine and the cluster: hop 1 of the hand trace and link 2 of the artifacts chain. The full solution is [`cluster.yaml`](../../../../course/solutions/entry/deploy/kind/cluster.yaml).

### The engine chart

The values carry every number once; the template only arranges them.

```yaml
# values.yaml (excerpt)
image:
  repository: forge-engine
  tag: "0.1.0"                 # pinned; quoted so YAML keeps it a string
  pullPolicy: IfNotPresent     # kind load puts the image on the node; Always would try a registry
model:
  dir: /artifacts/models/bigram   # the Pod path, never a host path
service:
  httpPort: 8000
  healthPort: 9464
probes:
  liveness:  { path: /healthz, initialDelaySeconds: 5, periodSeconds: 10 }
  readiness: { path: /healthz, periodSeconds: 2 }
securityContext:
  runAsNonRoot: true
  runAsUser: 10001
  allowPrivilegeEscalation: false
  readOnlyRootFilesystem: true
resources:
  requests: { cpu: 100m, memory: 64Mi }
  limits:   { cpu: "1",  memory: 256Mi }
artifacts: { hostPath: /artifacts, mountPath: /artifacts }
otel:
  endpoint: http://jaeger:4318    # OTLP/HTTP, not 4317 (gRPC)
```

```yaml
# templates/deployment.yaml (excerpt)
metadata:
  name: {{ .Release.Name }}            # helm install forge-engine -> deploy/forge-engine
spec:
  replicas: {{ .Values.replicaCount }}
  selector:
    matchLabels:
      app.kubernetes.io/name: {{ .Release.Name }}   # the Service selects the same label
  template:
    spec:
      securityContext:
        runAsNonRoot: {{ .Values.securityContext.runAsNonRoot }}
        runAsUser: {{ .Values.securityContext.runAsUser }}
      containers:
        - name: engine
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}{{ with .Values.image.digest }}@{{ . }}{{ end }}"
          imagePullPolicy: {{ .Values.image.pullPolicy | default "IfNotPresent" }}
          args:
            {{- if .Values.runtime }}
            # ... (L10.5: --config /etc/tinyllm/runtime.toml)
            {{- else }}
            # The tracer form of spec/cli-roles.md: flags, no config file.
            - --model-dir
            - {{ .Values.model.dir | quote }}
            - --port
            - {{ .Values.service.httpPort | quote }}       # quote: args are strings
            - --health-port
            - {{ .Values.service.healthPort | quote }}
            {{- end }}
          env:
            - name: OTEL_SERVICE_NAME
              value: {{ .Values.otel.serviceName | default .Release.Name | quote }}
            - name: OTEL_EXPORTER_OTLP_ENDPOINT
              value: {{ .Values.otel.endpoint | quote }}
            # ... (TL_ENGINE__ROLE, TL_ENGINE__MODEL_DIR for the config form)
          ports:
            - name: http                 # named, so probes and Services refer to names
              containerPort: {{ .Values.service.httpPort }}
            - name: health
              containerPort: {{ .Values.service.healthPort }}
            # ... (grpc, kv: later passes)
          readinessProbe:
            httpGet:
              path: {{ .Values.probes.readiness.path }}
              port: health
            # ... (periodSeconds and initialDelaySeconds from values)
          livenessProbe:
            httpGet:
              path: {{ .Values.probes.liveness.path }}
              port: health
          volumeMounts:
            - name: artifacts
              mountPath: {{ .Values.artifacts.mountPath }}
              readOnly: true
          resources:
            {{- toYaml .Values.resources | nindent 12 }}
      volumes:
        - name: artifacts
          hostPath:
            path: {{ .Values.artifacts.hostPath }}
            type: Directory    # fail fast if the node has no /artifacts (broken chain)
```

The `runtime` branch is the config-file form introduced in [tl-serve: an OpenAI-compatible server on tokio](../../ai-engineering/23-model-serving/06-openai-server-on-tokio.md); the tracer uses the flags branch. The Service is short: a `ClusterIP` with `selector: app.kubernetes.io/name: {{ .Release.Name }}` and a port `http` mapped to `targetPort: http`. Because both sides use the name `http`, changing `httpPort` in one place moves the container port, the args, and the Service target together.

`type: Directory` is a deliberate choice. Without it, a missing `/artifacts` would be silently created empty and the engine would fail later, further from the cause.

The full solution is [`forge-engine/templates/deployment.yaml`](../../../../course/solutions/entry/deploy/helm/forge-engine/templates/deployment.yaml), [`service.yaml`](../../../../course/solutions/entry/deploy/helm/forge-engine/templates/service.yaml), and [`values.yaml`](../../../../course/solutions/entry/deploy/helm/forge-engine/values.yaml).

### The gateway chart

The gateway template differs from the engine's in three places: the computed upstream (which you rendered by hand above), the probes, and the key.

```yaml
{{- $upstream := .Values.upstream | default (printf "http://%s-engine:8000" (trimSuffix "-gateway" .Release.Name)) -}}
# ...
          args:
            - --port
            - {{ .Values.service.port | quote }}
            - --health-port
            - {{ .Values.service.healthPort | quote }}
            - --upstream
            - {{ $upstream | quote }}
          env:
            # The key enters only by reference; its value is in no file.
            - name: TL_API_KEY
              valueFrom:
                secretKeyRef:
                  name: {{ .Values.apiKey.secretName }}   # forge-api-key
                  key: {{ .Values.apiKey.secretKey }}     # api-key
            # ... (gw.01: TL_GATEWAY_PEPPER from forge-gateway-pepper; gw.07: TL_GATEWAY__USAGE_DB)
```

```yaml
# values.yaml (excerpt)
service:
  type: NodePort
  port: 8080
  nodePort: 30080      # mapped to 127.0.0.1:30080 by deploy/kind/cluster.yaml
  healthPort: 9464
probes:
  liveness:
    path: /healthz     # the gateway itself: an engine outage never restarts it
  readiness:
    path: /readyz      # ready when it can route
upstream: ""           # "" means http://<release name minus -gateway>-engine:8000
```

The Service template renders `nodePort` only `if eq .Values.service.type "NodePort"`, so the same chart can later sit behind a `ClusterIP` without an invalid field. In the final solution the template also references a pepper Secret (from [Server skeleton](../../ai-engineering/25-ai-gateway/02-server-skeleton.md)) and a ledger PersistentVolumeClaim (from [Usage ledger, metering, and admin API](../../ai-engineering/25-ai-gateway/05-usage-ledger-and-admin-api.md)); with that version both must exist before the gateway Pod starts, and `values.yaml` lists the `kubectl create secret` commands.

The full solution is [`forge-gateway/templates/deployment.yaml`](../../../../course/solutions/entry/deploy/helm/forge-gateway/templates/deployment.yaml), [`service.yaml`](../../../../course/solutions/entry/deploy/helm/forge-gateway/templates/service.yaml), and [`values.yaml`](../../../../course/solutions/entry/deploy/helm/forge-gateway/values.yaml).

### Jaeger

```yaml
# One process: OTLP in, query UI and API out. In-memory: traces vanish on restart.
containers:
  - name: jaeger
    image: jaegertracing/jaeger:2.10.0     # pinned
    ports:
      - { name: otlp-grpc, containerPort: 4317 }
      - { name: otlp-http, containerPort: 4318 }
      - { name: query,     containerPort: 16686 }
    readinessProbe:
      httpGet: { path: /, port: query }
---
# In-cluster OTLP endpoint: OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4318
kind: Service
metadata: { name: jaeger }
spec:
  ports:
    - { name: otlp-grpc, port: 4317, targetPort: otlp-grpc }
    - { name: otlp-http, port: 4318, targetPort: otlp-http }
---
# Query UI and API from your machine: http://127.0.0.1:30686
kind: Service
metadata: { name: jaeger-query }
spec:
  type: NodePort
  ports:
    - { name: query, port: 16686, targetPort: query, nodePort: 30686 }
```

There are two Services, not one: ingestion stays cluster-internal, and only the read side is exposed to your machine. The full solution is [`jaeger.yaml`](../../../../course/solutions/entry/deploy/observability/jaeger.yaml).

### Why this design

Each choice above trades something away, and it is worth naming the trade.

- **Multi-stage over single-stage.** A single `rust:` image is over a gigabyte and ships a compiler into production; the two-stage engine image is the binary plus a slim base.
- **`kind load` over a registry, for now.** It needs nothing running, at the cost of re-loading after every build and of "same tag, different bytes" (see the warnings below). [kind cluster, local registry, and port mappings](03-kind-cluster-and-local-registry.md) adds the registry when the image count grows.
- **`hostPath` over a PersistentVolume or object storage.** Correct only on one node, but it makes the model visible with zero infrastructure. The single-node assumption is written into the cluster file, not hidden.
- **A derived upstream over a required value.** The default works for every release that follows the naming rule, and `upstream:` still overrides it.
- **A hand-made Secret over a templated one.** A templated Secret would put the value into `helm get values`; a sealed or external secret is the production answer.

## What can go wrong

Most failures of this deployment are one wrong line away from the solution. These are the eight that bite most often, each with its symptom and the part of the solution that avoids it:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| No `.dockerignore` | every build starts with `transferring context: 4.2GB`; a `COPY rust/ rust/` carries your laptop's `target/` into the image | `.dockerignore` excludes `**/target`, `artifacts`, `.ol` |
| Shell-form `ENTRYPOINT`, or arguments baked into `CMD` | the chart's `args:` are ignored, so the engine starts with no `--model-dir`; or `SIGTERM` never reaches the server | exec-form `ENTRYPOINT` with no arguments; the chart owns `args:` |
| `TL_API_KEY: tl_...` in `values.yaml` | the key is in git history and in `helm get values` for anyone with cluster access | `TL_API_KEY` comes only from `secretKeyRef` to a hand-made Secret |
| `kind create cluster` run from `deploy/kind/`, or a mapping added later | the engine finds an empty `/artifacts`, or `curl` cannot connect to 30080 | run kind from the repo root; declare all mappings before creation |
| `--model-dir artifacts/models/bigram` (a host path) | the engine exits at startup: `CrashLoopBackOff`, the first drill before the drill | `model.dir` is the Pod path `/artifacts/models/bigram` |
| Gateway liveness on `/readyz` | every engine restart also restarts the gateway, and the outage doubles | liveness `/healthz`, readiness `/readyz` |
| Rebuilt an image without `kind load` | the cluster keeps running yesterday's code under the same tag | bump the tag (or restart the rollout) after every load; the next sections replace this with a registry and digests |
| OTLP endpoint `http://jaeger:4317` with an HTTP exporter | no trace: 4317 is gRPC, 4318 is HTTP | `otel.endpoint: http://jaeger:4318` in both charts |

> **Warning:** The same-tag trap is the subtlest of these. `kind load` copies bytes onto the node under a tag; rebuild without reloading, or reload without changing the tag, and the cluster happily keeps running the old bytes.

## Where this leads

This section stands on four earlier pieces: the image, chart, and kind skills from [Containers and Kubernetes](../../programming-and-algorithms/01-language-primers/07-containers-and-kubernetes.md); the engine binary and its tracer flags from [Your first endpoint](../../ai-engineering/23-model-serving/01-your-first-endpoint.md); the gateway binary, its key check, and `/readyz` from [the tracer gateway](../../ai-engineering/25-ai-gateway/01-streaming-proxy.md); and `train bigram`, which writes the model under `artifacts/`, from [Byte bigram](../../ai-engineering/06-numerics-and-autograd/01-byte-bigram.md).

Going forward, the engine and gateway export spans to this Jaeger and [One trace](../11-observability/01-one-trace.md) reads the trace at 30686. [First drill: the engine crashloops](../12-incident-response-and-chaos/01-first-drill.md) patches `deploy/<system>-engine` and walks through the recovery. The next five sections of this chapter harden what you built: pinned digests and `HEALTHCHECK` in [Dockerfiles for the gateway and the engine](02-dockerfiles-for-gateway-and-engine.md), a local registry and a recreated cluster in [kind cluster, local registry, and port mappings](03-kind-cluster-and-local-registry.md), chart policy and values schemas in [Helm charts and the observability stack](04-helm-charts-and-observability-stack.md), a fast inner loop in [The Tilt dev loop](05-tilt-dev-loop.md), and [CI for the platform repo](06-ci-for-your-repo.md).

Each simplification here has a production equivalent:

| Your piece | Production equivalent | What it adds |
|---|---|---|
| `hostPath` `/artifacts` | object storage (S3, GCS) or a ReadWriteMany volume | models reachable from any node, versioned and access-controlled |
| a Secret created by hand | External Secrets Operator, Sealed Secrets, a cloud KMS | secrets synced from a vault, encrypted in git |
| `kind load docker-image` | a registry (the next sections run one at `localhost:5001`) and image digests | the same bytes on every node and in every environment |
| NodePort 30080 | a LoadBalancer Service, Ingress, or the Gateway API (Envoy Gateway, the Envoy AI Gateway) | TLS, hostnames, and routing at the edge |
| Jaeger all-in-one, in memory | the OpenTelemetry Collector feeding Tempo, Prometheus, and Grafana ([Tracing for the serving path](../11-observability/02-tracing-for-the-serving-path.md)) | sampling, retention, metrics, and dashboards |

### Further reading

- [Kubernetes: Services and DNS](https://kubernetes.io/docs/concepts/services-networking/dns-pod-service/)
- [Kubernetes: Secrets](https://kubernetes.io/docs/concepts/configuration/secret/)
- [Kubernetes volumes](https://kubernetes.io/docs/concepts/storage/volumes/)
- [Jaeger getting started](https://www.jaegertracing.io/docs/latest/getting-started/)
- [External Secrets](https://external-secrets.io/)
- [kind local registry](https://kind.sigs.k8s.io/docs/user/local-registry/)
- [Gateway API](https://gateway-api.sigs.k8s.io/)
- [OpenTelemetry Collector](https://opentelemetry.io/docs/collector/)

## Key terms

- **build context**: the directory tree the Docker builder sends to the daemon before the first instruction; here, the repo root.
- **`.dockerignore`**: a `.gitignore`-style file that excludes paths from the build context.
- **exec-form `ENTRYPOINT`**: a JSON-array entrypoint that runs the binary directly as PID 1 and receives the Pod's `args:`.
- **Service DNS name**: `<service>.<namespace>.svc.cluster.local`, a stable name for a Service that survives Pod replacement.
- **artifacts chain**: the three mounts (host directory, kind `extraMounts`, Pod `hostPath`) that carry the model into the engine.
- **`secretKeyRef`**: a reference from a container's environment variable to one key of a Secret.
- **readiness probe / liveness probe**: checks that, on failure, remove a Pod from its Service or restart its container, respectively.
- **deployment coordinates**: kube context, namespace, gateway URL, traces URL, and workload names, the five facts any tool needs to operate on the cluster.

## Summary

- Each image builds from the repo root as its context, so a `.dockerignore` keeps `target/`, `.ol/`, and `artifacts/` out, and each Dockerfile copies only the directories its stage needs.
- The final stages run a numeric non-root user and an exec-form `ENTRYPOINT` with no arguments; the chart supplies `args:`.
- The engine reads its model through three mounts: `./artifacts` on your machine, `/artifacts` on the kind node (`extraMounts`), `/artifacts` in the Pod (a `hostPath` volume). Break any link and the engine crashloops.
- The gateway finds the engine by Service DNS name (`http://<system>-engine:8000`), is ready only when the engine is (`/readyz`), and is live whenever it is itself up (`/healthz`).
- The API key lives in exactly one place: a Secret you create by hand, read through `secretKeyRef`, never in a chart, a values file, or an image.
- Five coordinates (kube context, namespace, gateway URL, traces URL, workload names) describe the deployment, and every later operation on the cluster uses exactly these.

## Review questions

1. Why must the `ENTRYPOINT` be exec form with no arguments?
   <details><summary>Answer</summary>The chart's <code>args:</code> replace the image's <code>CMD</code> and are appended to an exec-form <code>ENTRYPOINT</code>. Shell form runs <code>/bin/sh -c "..."</code>, which ignores the extra arguments and becomes PID 1 itself, so <code>SIGTERM</code> goes to the shell and the server is killed after the grace period instead of exiting cleanly.</details>
2. Why does the gateway's readiness follow the engine but its liveness does not?
   <details><summary>Answer</summary>Readiness only removes the Pod from its Service, which is the right reaction to "I cannot serve right now". Liveness restarts the container; a restart cannot fix an unreachable engine, so a liveness probe that followed the engine would turn one outage into two.</details>
3. Why is `USER 10001:10001` numeric?
   <details><summary>Answer</summary>The kubelet enforces <code>runAsNonRoot</code> from the image's user field without reading the image's <code>/etc/passwd</code>. A numeric UID can be checked against 0; a name cannot, and the Pod is refused.</details>
4. The engine crashloops right after a fresh `kind create cluster`. Name the three links to check, in order.
   <details><summary>Answer</summary><code>./artifacts/models/bigram</code> exists on your machine (trained); <code>extraMounts</code> resolved <code>./artifacts</code> against the directory kind ran from (it must be the repo root), visible as <code>/artifacts</code> on the node; the Pod's <code>hostPath</code> volume mounts it at <code>/artifacts</code> and <code>--model-dir</code> names a path under it.</details>
5. Why does the solution quote ports in `args:` but not in `containerPort:`?
   <details><summary>Answer</summary><code>args</code> is a list of strings, so an unquoted <code>8080</code> renders as an integer and the API server rejects the Pod. <code>containerPort</code> is an integer field, so there the bare number is correct.</details>
6. With one replica and the default rolling-update strategy, why does a failed upgrade leave the old engine serving?
   <details><summary>Answer</summary><code>maxSurge</code> is ⌈0.25 × 1⌉ = 1 and <code>maxUnavailable</code> is ⌊0.25 × 1⌋ = 0, so Kubernetes starts the new Pod first and stops the old one only after the new one passes readiness. If it never becomes ready, the old Pod is never stopped and the rollout simply stalls.</details>
