<!-- ol:module dep.00 -->
# Tracer deploy: engine and gateway images, kind cluster, two Helm charts, Jaeger all-in-one

## Overview

| | |
|---|---|
| **Module** | `dep.00` · practice · ops · Pass 1 · 6 to 10 h |
| **You build** | `deploy/docker/{engine,gateway}.Dockerfile`, `.dockerignore`, `deploy/kind/cluster.yaml`, `deploy/helm/<system>-{engine,gateway}/`, `deploy/observability/jaeger.yaml`, and the `[deploy]` section of `system.toml` |
| **Contract** | none of its own: the charts run the tracer forms of [`spec/cli-roles.md`](../../../../course/contracts/spec/cli-roles.md) and the ports of DESIGN 2.13; section 4 is the checked layout |
| **Tests** | `course/tests/dep.00/` (`check` runs `artifacts.py`; what each test checks: section 4) |
| **Needs** | `lang.07` containers and Kubernetes, `L10.0` your engine, `gw.00` your gateway, `L0.0` your `tinyllm` CLI (all reading: this module packages their entry points) |
| **Used by** | `obs.00` one trace in Jaeger · `ops.00` the first drill |
| **Milestone** | MS-P1 |
| **Optional depth** | [Containers, Kubernetes & Workloads](README.md) (this topic), [Kubernetes: Services and DNS](https://kubernetes.io/docs/concepts/services-networking/dns-pod-service/) (free), [Secrets](https://kubernetes.io/docs/concepts/configuration/secret/) (free), [Jaeger getting started](https://www.jaegertracing.io/docs/latest/getting-started/) (free) |

## Key Takeaways

- Each image builds from the **repo root** as its context, so a `.dockerignore` keeps `target/`, `.ol/`, and `artifacts/` out, and the Dockerfile copies only the directories its stage needs.
- The engine reads its model through a chain of three mounts: `./artifacts` on your machine, `/artifacts` on the kind node (`extraMounts`), `/artifacts` in the pod (a `hostPath` volume). Break any link and the engine crashloops, which is the drill of `ops.00`.
- The gateway finds the engine by **Service DNS name** (`http://<system>-engine:8000`), is **ready** only when the engine is (`/readyz`), and is **live** whenever it is itself up (`/healthz`).
- The API key lives in exactly one place: a Secret you create by hand, read through `secretKeyRef`. It is never in a chart, a values file, or an image.
- `system.toml` `[deploy]` is how the harness finds your cluster: MS-P1's kind steps, `obs.00`, and every drill read it.

## How to work this chapter

Every command runs from your repo root. `forge` stands for your `<system>` name.

```bash
ol start dep.00                         # records the start; there are no stubs
ol tests dep.00                         # read the test catalog first
# write the files of section 4, then:
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
ol check dep.00                         # exit code is the verdict
OL_SMOKE=1 ol check dep.00              # without a cluster: static and docker tiers only
```

---

## 1. Why now

After `L10.0` and `gw.00` your engine and gateway stream tokens on your laptop, started by hand or by `ol milestone MS-P1 --smoke`. Nothing restarts them when they crash, nothing but your shell knows where the model file is, and the only way to reach them is a port your laptop happened to free. The rest of the course assumes a deployment: `obs.00` reads one trace out of Jaeger running next to them, `ops.00` breaks the engine on purpose and grades how you bring it back, and Pass 7 grows this same cluster into the serving platform. This module turns your two programs into two images, two Helm releases, and one kind cluster with Jaeger, with nothing in them that the course wrote.

## 2. Principles

`lang.07` defined images, layers, Pods, Deployments, Services, probes, Helm, and kind. This section adds only what changes when the programs are real.

### 2.1 Build contexts at the repo root

The engine needs `rust/`; the gateway needs `go/` and `contracts/` (its `go.mod` replaces the contracts module with `../contracts/go`, a directory that arrives with the generated code of Pass 9; `COPY contracts/ contracts/` copies whatever is there today, and copying a path that does not exist fails the build). Both images therefore build with the **repo root** as their context: `docker build -f deploy/docker/engine.Dockerfile .`. The builder sends the whole context to the Docker daemon before the first instruction runs. Your `rust/target/` alone can be several gigabytes, so a **`.dockerignore`** at the root (same syntax as `.gitignore`) must exclude build outputs (`**/target`), local state (`.ol`, `.venv`), data (`artifacts`), and anything secret.

**The engine image.** Stage 1 starts from `rust:<version>-slim-bookworm` and runs `cargo build --release -p tl-serve`. Stage 2 uses `debian:bookworm-slim`. Copy order follows change frequency: `contracts/`, then `rust/` and `cargo build`.

**The gateway image.** Stage 1 starts from `golang:<version>-alpine`, copies `contracts/` (`COPY contracts/ contracts/`) and `go/go.mod` (plus `go.sum` if you have one), runs `go mod download` (cached until the module files change), then copies `go/` and builds `./cmd/gateway` with `CGO_ENABLED=0`, which gives a static binary. Stage 2 is `alpine` (or distroless) plus that binary.

Both final stages: a numeric non-root `USER`, `EXPOSE` of the ports in 2.2, and an exec-form `ENTRYPOINT` with **no arguments**. The chart passes the arguments (`args:` in the Pod spec), and only exec form receives them.

### 2.2 The topology on kind

| Object (namespace `<system>`) | Kind | Ports | Made by |
|---|---|---|---|
| `<system>-engine` | Deployment, 1 replica | 8000 HTTP, 9464 health | Helm release `<system>-engine` |
| `<system>-engine` | Service, ClusterIP | 8000 | same release |
| `<system>-gateway` | Deployment, 1 replica | 8080 HTTP, 9464 health | Helm release `<system>-gateway` |
| `<system>-gateway` | Service, NodePort | 8080, nodePort **30080** | same release |
| `jaeger` | Deployment | 4317, 4318 (OTLP in), 16686 (query) | `deploy/observability/jaeger.yaml` |
| `jaeger` | Service, ClusterIP | 4317, 4318 | same file |
| `jaeger-query` | Service, NodePort | 16686, nodePort **30686** | same file |
| `<system>-api-key` | Secret | key `api-key` | you, by hand, once |

Ports inside the cluster are fixed (DESIGN 2.13); only `ol milestone` on your laptop allocates free ones. The release name is the object name (`{{ .Release.Name }}`), so `helm install forge-engine` makes `deploy/forge-engine`, which is what `ops.00` patches and what `[deploy].services.engine` names.

**Service DNS.** Every Service gets a DNS name, `<service>.<namespace>.svc.cluster.local`, and inside the same namespace the short name `<service>` resolves too. The gateway's `--upstream` is therefore `http://<system>-engine:8000`: no IP address ever appears in a chart, and a restarted engine Pod with a new IP is reached at the same name.

### 2.3 The artifacts chain

The model is data, not code, so it is not in the image. It travels through three mounts:

| Where | Path | Declared by |
|---|---|---|
| your machine | `./artifacts/models/bigram/` (from `{tinyllm} train bigram --out ...`) | you |
| the kind node container | `/artifacts/models/bigram/` | `extraMounts` in `deploy/kind/cluster.yaml` (a relative `hostPath` is resolved against the directory you run `kind create cluster` from) |
| the engine Pod | `/artifacts/models/bigram/` | a `hostPath` volume of the node's `/artifacts`, mounted at `/artifacts` |

The engine's `--model-dir` is the Pod path. A `hostPath` volume is a single-node simplification (DESIGN 2.13); the "Going further" table names what replaces it.

### 2.4 Secrets

A **Secret** is a Kubernetes object holding small byte strings, here the API key. A container reads it as an environment variable through `valueFrom.secretKeyRef: {name, key}`. The Secret is created once, by hand, from the `TL_API_KEY` variable in your shell, so the key's value appears in no file you commit: not the chart, not `values.yaml`, not the image, and not `helm get values`. (Kubernetes stores Secrets base64-encoded, not encrypted; access control and encryption at rest come in `dep.03` and `craft.19`.)

### 2.5 Probes that match the roles

| Container | readiness | liveness | Why |
|---|---|---|---|
| engine | `GET /healthz` on 9464 | `GET /healthz` on 9464 | the engine has no dependency; healthy is ready |
| gateway | `GET /readyz` on 9464 | `GET /healthz` on 9464 | `/readyz` answers 503 while the engine is unreachable, so the gateway leaves its Service instead of returning errors; liveness must not follow the engine, or an engine outage restarts the gateway too |

### 2.6 Telemetry wiring

Both servers read the standard OpenTelemetry variables (`spec/cli-roles.md`): `OTEL_EXPORTER_OTLP_ENDPOINT` (the base URL; unset means export nothing) and `OTEL_SERVICE_NAME`. The charts set the endpoint to `http://jaeger:4318`, Jaeger's OTLP/HTTP port through its Service, and the service names to the release names, `<system>-engine` and `<system>-gateway`. `obs.00` makes the export happen and MS-P1 looks the trace up by those names.

### 2.7 The `[deploy]` section

`system.toml` is the harness's only map of your cluster:

```toml
[deploy]
kube_context = "kind-forge"                 # kind create cluster --name forge
namespace    = "forge"
gateway_url  = "http://127.0.0.1:30080"     # the gateway NodePort, mapped by kind
traces       = "http://127.0.0.1:30686"     # the Jaeger query NodePort
services     = { engine = "deploy/forge-engine", gateway = "deploy/forge-gateway" }
```

`ol drill` refuses to touch any context but `kube_context` and any namespace but `namespace`; it patches the workloads `services` names.

## 3. Worked example by hand

One request, every hop, with `<system> = forge`, before you run anything. Each line is checkable from the files of section 4.

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
| 7 | engine reads the model | `/artifacts/models/bigram/model.safetensors` | `--model-dir`, the `hostPath` volume, `extraMounts`, your `train bigram` |
| 8 | tokens stream back | SSE, one `data:` line per token, then `data: [DONE]` | the gateway copies and flushes each chunk (`gw.00`) |
| 9 | spans leave both Pods | `http://jaeger:4318/v1/traces` | `OTEL_EXPORTER_OTLP_ENDPOINT`, Service `jaeger` port 4318 |

Two health ports never appear in that path: the kubelet probes `:9464` on each Pod directly (hop 3 and hop 6 happen only while those probes pass).

**Render one template by hand.** The gateway chart's `values.yaml` holds `port: 8080`, `healthPort: 9464`, and `upstream: ""`, and its template computes a default:

```yaml
{{- $upstream := .Values.upstream | default (printf "http://%s-engine:8000" (trimSuffix "-gateway" .Release.Name)) -}}
args: ["--port", {{ .Values.port | quote }}, "--health-port", {{ .Values.healthPort | quote }}, "--upstream", {{ $upstream | quote }}]
```

With `helm install forge-gateway`: `.Release.Name` is `forge-gateway`; `trimSuffix "-gateway"` gives `forge`; `printf` gives `http://forge-engine:8000`; `upstream` is empty, so `default` picks the computed value. The rendered list is `["--port", "8080", "--health-port", "9464", "--upstream", "http://forge-engine:8000"]`. `quote` matters: `args` must be strings, and a bare `8080` would render as a YAML integer, which Kubernetes rejects. `helm template forge-gateway deploy/helm/forge-gateway -n forge` prints exactly this, and `test_gateway_chart_renders` reads it the same way.

**Rollout check by hand.** One replica each, default strategy: `maxSurge` $= \lceil 0.25 \times 1 \rceil = 1$, `maxUnavailable` $= \lfloor 0.25 \times 1 \rfloor = 0$. An upgrade starts the new engine Pod, waits for its readiness probe, then stops the old one: no gap, provided the new Pod becomes ready at all. When it does not (the drill of `ops.00`), the old Pod keeps serving and the rollout stalls, which is the safe failure.

| Symbol | Meaning | Type |
|---|---|---|
| $\lceil x \rceil$, $\lfloor x \rfloor$ | round up, round down | integer |

## 4. The interface

The checked layout, in your repo (`<system>` is `[system].name`):

| Path | Requirement |
|---|---|
| `.dockerignore` | excludes `target` (any depth), `.ol`, and `artifacts` |
| `deploy/docker/engine.Dockerfile` | context: repo root; two or more stages, every `FROM` pinned; final stage `USER` numeric non-root, `EXPOSE 8000` and `9464`, exec-form `ENTRYPOINT` of your `tl-serve`; the image takes the tracer flags of `spec/cli-roles.md` as arguments |
| `deploy/docker/gateway.Dockerfile` | the same rules; `EXPOSE 8080` and `9464`; your `go/cmd/gateway` |
| `deploy/kind/cluster.yaml` | `name: <system>`; mappings 30080 to 30080 and 30686 to 30686; an `extraMounts` entry with `containerPath: /artifacts` |
| `deploy/helm/<system>-engine/` | `helm lint` clean; rendered as release `<system>-engine`: Deployment `<system>-engine` with `--model-dir` under `/artifacts/`, `--port 8000`, `--health-port 9464`, a `hostPath` volume of `/artifacts` mounted at `/artifacts`, readiness and liveness on `/healthz` at 9464, `resources.limits`, `runAsNonRoot`, a pinned tag with a pull policy other than `Always`, `OTEL_SERVICE_NAME=<system>-engine`, `OTEL_EXPORTER_OTLP_ENDPOINT` set; Service `<system>-engine` targeting 8000 |
| `deploy/helm/<system>-gateway/` | the same pod rules with readiness `/readyz` and liveness `/healthz` at 9464, `OTEL_SERVICE_NAME=<system>-gateway`; `--port 8080`, `--health-port 9464`, `--upstream http://<system>-engine:8000`; `TL_API_KEY` only through `secretKeyRef`; Service `<system>-gateway`, `type: NodePort`, `nodePort: 30080` targeting 8080; no file in either chart holds a `tl_<id>_<secret>` key |
| `deploy/observability/jaeger.yaml` | one Deployment of a pinned Jaeger image; a Service port 4318; a NodePort Service on 30686 |
| `system.toml` `[deploy]` | as in section 2.7 |
| on kind | both Deployments available in namespace `<system>`; 32 or more chunks through `gateway_url`; `traces/api/services` answers |

### What the tests check

`ol check dep.00` runs `course/tests/dep.00/check` in your repo, in three tiers. The docker tier trains a throwaway model with your `[entry].tinyllm` into `.ol/check/dep.00/`, copies it into the engine container at `/artifacts/models/bigram` (copied, not bind-mounted, because Docker Desktop shares only some host directories), and runs both images on a private Docker network, the engine under the alias `engine`.

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_layout_present` | unit | the files of the table above | `ops.00` and MS-P1 find them by these names |
| `test_dockerignore_keeps_build_outputs_out` | unit | `target`, `.ol`, `artifacts` excluded | small contexts, no laptop state in images |
| `test_dockerfiles_follow_the_image_rules` | unit | stages, pins, user, ports, exec form | the `lang.07` rules on real programs |
| `test_kind_cluster_maps_gateway_jaeger_and_artifacts` | unit | name, the two mappings, the `/artifacts` mount | MS-P1 reaches 30080 and 30686 from your machine |
| `test_charts_lint` | conformance | `helm lint` on both charts | template errors before install |
| `test_engine_chart_renders` | unit | args, volume, probes, limits, user, image, OTel env, Service | the engine starts and is found |
| `test_gateway_chart_renders` | unit | args, probes, OTel env, NodePort 30080 | the gateway reaches the engine and is reachable |
| `test_api_key_only_from_a_secret` | unit | `secretKeyRef`, no literal key in the charts | `craft.19` audits this; `dep.03` enforces it |
| `test_jaeger_manifest` | unit | pinned image, 4318, NodePort 30686 | `obs.00` exports there and reads from there |
| `test_system_toml_deploy_section` | unit | the `[deploy]` keys | MS-P1 kind steps, `obs.00`, `ol drill` |
| `test_images_build` | unit | both images build from the repo root | CI builds them the same way (`dep.05`) |
| `test_images_run_as_numeric_non_root` | boundary | the images' `User` | the kubelet's `runAsNonRoot` check |
| `test_containers_stream_a_completion` | conformance | 24 chunks then `[DONE]` through the gateway container | the images work before any cluster |
| `test_gateway_container_rejects_a_missing_key` | boundary | no key gives 401 | the key check survived packaging |
| `test_containers_exit_zero_on_sigterm` | fault | both exit 0 within an 8 s grace period | rollouts and the `ops.00` drill |
| `test_workloads_available_on_kind` | conformance | both Deployments have an available replica | the whole chart worked |
| `test_stream_through_the_gateway_nodeport` | conformance | 32 or more chunks at `gateway_url` with your key | MS-P1's kind step |
| `test_jaeger_query_reachable` | conformance | `GET <traces>/api/services` is JSON | MS-P1's trace step |

The last three are the **cluster tier**: without your `kube_context` they fail and say so, and `OL_SMOKE=1 ol check dep.00` skips them with the reason. The cluster tier reads your key from the variable `[endpoints].api_key_env` names (default `TL_API_KEY`).

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. No `.dockerignore` | every build starts with `transferring context: 4.2GB`; a `COPY rust/ rust/` carries your laptop's `target/` into the image | `test_dockerignore_keeps_build_outputs_out` |
| 2. Shell-form `ENTRYPOINT`, or arguments baked into `CMD` | the chart's `args:` are ignored, so the engine starts with no `--model-dir`; or `SIGTERM` never reaches the server | `test_dockerfiles_follow_the_image_rules`, `test_containers_exit_zero_on_sigterm` |
| 3. `TL_API_KEY: tl_...` in `values.yaml` | the key is in git history and in `helm get values` for anyone with cluster access | `test_api_key_only_from_a_secret` |
| 4. `kind create cluster` run from `deploy/kind/`, or a mapping added later | the engine finds an empty `/artifacts`, or `curl` cannot connect to 30080 | `test_kind_cluster_maps_gateway_jaeger_and_artifacts`, `test_workloads_available_on_kind` |
| 5. `--model-dir artifacts/models/bigram` (a host path) | the engine exits at startup: `CrashLoopBackOff`, the drill of `ops.00` before the drill | `test_engine_chart_renders` |
| 6. Gateway liveness on `/readyz` | every engine restart also restarts the gateway, and the outage doubles | `test_gateway_chart_renders` |
| 7. Rebuilt an image without `kind load` | the cluster keeps running yesterday's code under the same tag | `test_workloads_available_on_kind` only when the old code fails; bump the tag (or restart the rollout) after every load |
| 8. OTLP endpoint `http://jaeger:4317` with an HTTP exporter | no trace: 4317 is gRPC, 4318 is HTTP | `test_jaeger_manifest` checks the port exists; the obs.00 check proves the trace |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `lang.07` | the image, chart, and kind skills, on a toy server |
| Back | `L10.0` | the engine binary and its tracer flags |
| Back | `gw.00` | the gateway binary, its key check, and `/readyz` |
| Back | `L0.0` | `train bigram`, which writes the model under `artifacts/` |
| Forward | `obs.00` | the engine and gateway export spans to this Jaeger; the trace is read at 30686 |
| Forward | `ops.00` | the drill patches `deploy/<system>-engine` and grades your recovery |
| Forward | `dep.01` to `dep.05` | pinned digests and `HEALTHCHECK`, a local registry and a recreated cluster, chart policy and values schemas, Tilt, CI |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `hostPath` `/artifacts` | object storage (S3, GCS) or a ReadWriteMany volume | models reachable from any node, versioned and access-controlled | DESIGN 2.13; [Kubernetes volumes](https://kubernetes.io/docs/concepts/storage/volumes/) |
| a Secret created by hand | External Secrets Operator, Sealed Secrets, a cloud KMS | secrets synced from a vault, encrypted in git | [External Secrets](https://external-secrets.io/) |
| `kind load docker-image` | a registry (`dep.02` runs one at `localhost:5001`) and image digests | the same bytes on every node and in every environment | [kind local registry](https://kind.sigs.k8s.io/docs/user/local-registry/) |
| NodePort 30080 | a LoadBalancer Service, Ingress, or the Gateway API (Envoy Gateway, the Envoy AI Gateway) | TLS, hostnames, and routing at the edge | [Gateway API](https://gateway-api.sigs.k8s.io/) |
| Jaeger all-in-one, in memory | the OpenTelemetry Collector feeding Tempo, Prometheus, and Grafana (Pass 7, `obs.01`) | sampling, retention, metrics, and dashboards | [OTel Collector](https://opentelemetry.io/docs/collector/) |
