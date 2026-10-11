<!-- ol:module dep.03 -->
# Helm charts for the gateway and the engine, plus the observability stack

## Overview

| | |
|---|---|
| **Module** | `dep.03` · practice · ops · Pass 7 · 5 to 8 h |
| **You build** | `deploy/helm/<system>-gateway/` and `deploy/helm/<system>-engine/` grown into production charts (the course's values schema, PodDisruptionBudgets, the ledger's volume claim, one engine release per role, a runtime ConfigMap), and `deploy/observability/` (an umbrella chart pinning the collector, Prometheus with Grafana, and Tempo) |
| **Contract** | the values schemas [`helm/gateway.values.schema.json`](../../course/contracts/helm/gateway.values.schema.json) and [`helm/engine.values.schema.json`](../../course/contracts/helm/engine.values.schema.json), the pins and rule selector of [`helm/observability.md`](../../course/contracts/helm/observability.md) |
| **Tests** | `course/tests/dep.03/` (`check` runs `artifacts.py`; what each test checks: section 4) |
| **Needs** | `dep.01` the images, `dep.02` the cluster and registry; reading: `dep.00` the tracer charts, `gw.07` the ledger the gateway chart must keep, [Cloud native](../../systems/03-cloud-native/) |
| **Used by** | MS-prod deploys these charts; `dep.04` (Tilt), `obs.01` to `obs.04`, and the drills build on them |
| **Milestone** | MS-prod |
| **Optional depth** | [Helm: schema files](https://helm.sh/docs/topics/charts/#schema-files) (free); [Kubernetes: disruptions and PodDisruptionBudgets](https://kubernetes.io/docs/concepts/workloads/pods/disruptions/) (free); [kube-prometheus-stack](https://github.com/prometheus-community/helm-charts/tree/main/charts/kube-prometheus-stack) (free); [Helm: chart dependencies](https://helm.sh/docs/helm/helm_dependency/) (free) |

## Key Takeaways

- A chart's **`values.schema.json`** makes Helm refuse bad values before anything is installed: tag `latest`, a literal pepper, a second gateway replica. The course writes the schema; your chart carries it unchanged.
- The **policy** is checked on what Kubernetes receives (`helm template`): probes, requests and limits, a numeric non-root user, no privilege escalation, pinned images, a **PodDisruptionBudget** for every workload.
- A single-replica gateway's PDB must allow **one pod down**; `maxUnavailable: 0` makes every node drain hang.
- **Secrets only by reference**: the chart names a Secret and a key; you create the Secret by hand. The usage ledger lives on a **PersistentVolumeClaim**, because the root filesystem is read-only and an `emptyDir` dies with the pod.
- Upstream charts are **pinned** in `Chart.yaml` and `Chart.lock`, and the stack keeps kube-prometheus-stack's default **rule selector**: rules labelled `release: observability` load, anything else is silently ignored.

## How to work this chapter

Every command runs from your repo root. `forge` stands for your `<system>` name.

```bash
ol start dep.03
ol tests dep.03
cp contracts/helm/gateway.values.schema.json deploy/helm/forge-gateway/values.schema.json
cp contracts/helm/engine.values.schema.json  deploy/helm/forge-engine/values.schema.json
helm lint deploy/helm/forge-gateway deploy/helm/forge-engine
helm template x deploy/helm/forge-gateway --set image.tag=latest     # must fail
kubectl -n forge create secret generic forge-gateway-pepper --from-literal=pepper="$(openssl rand -hex 32)"
helm upgrade --install forge-engine  deploy/helm/forge-engine  -n forge --set image.repository=localhost:5001/forge-engine
helm upgrade --install forge-gateway deploy/helm/forge-gateway -n forge --set image.repository=localhost:5001/forge-gateway
helm dependency update deploy/observability                         # writes Chart.lock and charts/*.tgz
kubectl create namespace observability
kubectl -n observability create secret generic grafana-admin --from-literal=admin-user=admin --from-literal=admin-password="$(openssl rand -hex 16)"
helm upgrade --install observability deploy/observability -n observability
ol check dep.03                         # exit code is the verdict
```

Keep `deploy/observability/charts/` out of git (`.gitignore`): `Chart.lock` pins the versions and `helm dependency build` restores the archives.

---

## 1. Why now

The Pass 1 charts worked once, on one cluster, with values you remembered. Pass 7 runs them for the rest of the course: MS-prod installs the gateway, a unified engine, then a prefill and two decode engines, plus the observability stack; the drills (`ops.01` and later) kill pods and drain nodes while load runs; `dep.04` reinstalls them on every save. Three things that were fine for a tracer now break. A typo in `values.yaml` (`tag: latest`, a pepper pasted as a string) installs without complaint. A node drain or a rollout can take the only gateway pod down, or hang forever, depending on a PodDisruptionBudget you never wrote. And the gateway now writes a usage ledger (`gw.07`) that a read-only root filesystem cannot hold and a pod restart must not erase. This module makes the charts refuse bad values, encode the operating policy, and install the observability stack that `obs.*` fills.

## 2. Principles

### 2.1 Values schemas

Helm reads `values.schema.json` (JSON Schema draft-07) next to `Chart.yaml` and validates the merged values (defaults plus `-f` files plus `--set`) on `lint`, `template`, `install`, and `upgrade`. The course writes the schemas in `contracts/helm/` and changes them only through a contract version, so every chart in every learner's repo accepts the same shape: `image.{repository, tag, digest, pullPolicy}`, `resources.{requests, limits}.{cpu, memory}`, `securityContext`, `probes.{liveness, readiness}.path`, `pdb.{enabled, maxUnavailable}`, and per chart `replicaCount` (gateway: exactly 1), `service`, `pepper` (gateway), `role` and `model.dir` (engine). The schema rejects `tag: latest`, `runAsNonRoot: false`, an env var whose name looks secret but has a literal `value`, and a `pepper` that is not `{name, key}`.

### 2.2 The policy, on rendered output

`helm template` renders the manifests without a cluster. The check reads them as Kubernetes would:

| Rule | Why |
|---|---|
| liveness and readiness probes on every container | liveness restarts a hung process; readiness keeps traffic away until it can serve (gateway: `/readyz` follows routable workers) |
| `resources.requests` and `limits` for CPU and memory | requests drive scheduling; limits stop one engine from starving the node; without a memory limit the OOM killer picks a victim for you |
| `runAsNonRoot: true`, a numeric `runAsUser`, `allowPrivilegeEscalation: false` | the kubelet can verify a number, not a name; no setuid escapes |
| a pinned image, `imagePullPolicy` not `Always` | the same bytes on every node; kind does not re-pull |
| a PodDisruptionBudget selecting the pods | a voluntary disruption (drain, cluster upgrade) respects it |

A **PodDisruptionBudget** limits voluntary evictions: with `maxUnavailable: 1`, the eviction API lets one pod of the selected set go at a time. The arithmetic for one replica:

| Symbol | Meaning |
|---|---|
| $r$ | replicas of the Deployment |
| $u$ | `maxUnavailable` of its PDB |
| $a$ | pods that may be evicted at once, $a = \min(u, r)$ |

The gateway has $r = 1$. With $u = 1$, $a = 1$: a drain evicts the pod (a short outage, which is the honest cost of one replica). With $u = 0$, $a = 0$: the eviction API refuses forever and `kubectl drain` never finishes, which turns a routine node upgrade into an incident.

### 2.3 Secrets by reference

A chart value that holds a secret ends up in git, in `helm get values`, and in the release's stored manifest. So the chart only **references** a Secret: `env: [{name: TL_GATEWAY_PEPPER, valueFrom: {secretKeyRef: {name: forge-gateway-pepper, key: pepper}}}]`, and you create the Secret once with `kubectl create secret`. The same holds for the tracer's `TL_API_KEY` and Grafana's admin password (`grafana.admin.existingSecret`).

### 2.4 State: the ledger's volume

The gateway's root filesystem is read-only (`readOnlyRootFilesystem: true`), and the node's `/artifacts` mount is read-only too (`dep.02`). The ledger is a SQLite file in WAL mode (`gw.07`), so it needs a writable directory that outlives the pod: a **PersistentVolumeClaim**. kind ships a `local-path` provisioner (StorageClass `standard`), which binds the claim to a directory on the node. One replica and `ReadWriteOnce` fit together: the volume is mounted by one node, and SQLite's file locks keep a second process on the same node safe during a rolling update.

### 2.5 The observability stack

`deploy/observability` is an **umbrella chart**: no templates of its own, three `dependencies` pinned to the exact versions of `contracts/helm/observability.md` (`opentelemetry-collector` 0.173.1, `kube-prometheus-stack` 91.5.2, `tempo` 1.24.4). `helm dependency update` resolves them, writes `Chart.lock` (names, repositories, versions, and a digest of the requirements), and downloads the archives into `charts/`. Values for a subchart go under its name (or `alias`) in `values.yaml`. kube-prometheus-stack's Prometheus loads only the `PrometheusRule` and `ServiceMonitor` objects its selectors match; with the default (`ruleSelectorNilUsesHelmValues: true`) that is everything labelled `release: <release name>`, so the release is named `observability` and every rule `obs.03` writes carries `release: observability`.

## 3. Worked example by hand

Disaggregated serving installs the engine chart three times:

```bash
helm upgrade --install forge-engine         deploy/helm/forge-engine -n forge
helm upgrade --install forge-engine-prefill deploy/helm/forge-engine -n forge --set role=prefill
helm upgrade --install forge-engine-decode  deploy/helm/forge-engine -n forge --set role=decode --set replicaCount=2
```

For the third, Helm merges `values.yaml` with `role=decode` and `replicaCount=2` and validates: `role` is in `[unified, prefill, decode]`, `replicaCount` is an integer at least 0. The templates name everything after the release:

| Object | Name | Key fields |
|---|---|---|
| Deployment | `forge-engine-decode` | `replicas: 2`; pod labels `app.kubernetes.io/name: forge-engine-decode`, `tinyllm.role: decode`; env `TL_ENGINE__ROLE=decode` |
| Service | `forge-engine-decode` | ClusterIP, ports 8000, 50051, 50052 |
| PodDisruptionBudget | `forge-engine-decode` | `maxUnavailable: 1`, selector `app.kubernetes.io/name: forge-engine-decode` |

The PDB arithmetic of 2.2 with $r = 2$, $u = 1$: $a = \min(1, 2) = 1$, so a drain moves one decode pod at a time and the other keeps serving. `[deploy].services.decode = "deploy/forge-engine-decode"` is how `ops.01` finds the pod to kill. And `--set image.tag=latest` on any of the three stops at validation: `at '/image/tag': 'not' failed`, before Kubernetes sees anything. `test_engine_one_release_per_role` and `test_schema_rejects_unsafe_values` check this example.

## 4. The artifact and its check

| Path | Holds |
|---|---|
| `deploy/helm/<system>-gateway/` | `values.schema.json` copied from the contract; values in the schema's shape; Deployment (1 replica, probes on the health port, pod and container security context, env: pepper and API key by `secretKeyRef`, `TL_GATEWAY__USAGE_DB` on the claim), Service (NodePort 30080), PodDisruptionBudget (`maxUnavailable: 1`), PersistentVolumeClaim for the ledger, ConfigMap with `runtime.toml` when `runtime` is set (then `--config`; otherwise the tracer flags) |
| `deploy/helm/<system>-engine/` | the same for the engine, one release per `role`, the model under `/artifacts` read-only, the role passed to the engine |
| `deploy/observability/Chart.yaml`, `Chart.lock`, `values.yaml` | the three pinned dependencies; Prometheus on NodePort 30090, Grafana on 30300 with its admin password from a Secret, the default rule selector |
| `system.toml` `[deploy]` | `prometheus = "http://127.0.0.1:30090"` |

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_layout_present` | unit | charts, schemas, the stack's three files | |
| `test_values_schemas_are_the_contract` | conformance | each `values.schema.json` equals the contract | one shape for every repo |
| `test_charts_lint` | conformance | `helm lint` both charts | |
| `test_schema_rejects_unsafe_values` | boundary | `latest`, a literal pepper, 2 gateway replicas, `role=leader` all fail to render | the schema is live |
| `test_policy_over_rendered_charts` | unit | 2.2 on the gateway and on the engine in each role | the policy of DESIGN 2.13 |
| `test_secrets_only_by_reference` | unit | no rendered Secret with data, no literal secret env, the pepper by reference | nothing secret in git or Helm |
| `test_gateway_one_replica_that_can_drain` | boundary | 1 replica; no PDB that allows zero disruptions | drains in the drills finish |
| `test_gateway_ledger_on_a_volume_claim` | unit | the gateway mounts a PVC the chart creates | the bill survives restarts |
| `test_engine_one_release_per_role` | unit | section 3: names per release, the role reaches the engine | MS-prod's disaggregated run, `ops.01` |
| `test_observability_stack_is_pinned` | conformance | `Chart.yaml` and `Chart.lock` match the pins | reproducible stack; upgrades are explicit |
| `test_prometheus_rule_selector_and_ports` | unit | default selectors, NodePorts 30090 and 30300, Grafana password from a Secret | `obs.03` rules load |
| `test_system_toml_prometheus` | unit | `[deploy].prometheus` | MS-prod `promql` steps |
| `test_server_side_dry_run` | conformance | `kubectl apply --dry-run=server` of the gateway, engine, and a decode engine | the API server accepts every object |
| `test_releases_available` | conformance | gateway and engine available; release `observability` deployed | |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. no `values.schema.json`, or a loosened copy | `tag: latest` and a pasted pepper install without a word | `test_schema_rejects_unsafe_values`, `test_values_schemas_are_the_contract` |
| 2. a probe on the API port, or none | a hung engine keeps receiving traffic; a slow start restarts forever | `test_policy_over_rendered_charts` |
| 3. `maxUnavailable: 0` (or `minAvailable: 1`) on one replica | `kubectl drain` waits forever; the cluster upgrade stalls | `test_gateway_one_replica_that_can_drain` |
| 4. the ledger on an `emptyDir` or the root filesystem | usage resets on every restart, or the gateway cannot write at all | `test_gateway_ledger_on_a_volume_claim` |
| 5. a custom `ruleSelector`, or a release not named `observability` | the SLO alerts of `obs.03` are valid YAML and never fire | `test_prometheus_rule_selector_and_ports` |
| 6. `version: ">=91.0.0"` and no `Chart.lock` | the stack changes under you on the next install | `test_observability_stack_is_pinned` |
| 7. one engine release with `role` in a template conditional | prefill and decode share a name; installing one replaces the other | `test_engine_one_release_per_role` |
| 8. `adminPassword: admin` for Grafana | the password is in git and in `helm get values` | `test_prometheus_rule_selector_and_ports` |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `dep.01` | the images the charts run |
| Back | `dep.02` | the cluster, the registry at `localhost:5001`, the NodePorts |
| Forward | MS-prod | installs the gateway, the engines (unified, then prefill and two decode), and the stack; runs conformance, load, and `ops.01` against them |
| Forward | `dep.04` | Tilt reinstalls these charts on every change |
| Forward | `obs.01` to `obs.04` | the collector pipeline, ServiceMonitors, rules, and dashboards installed into this stack |
| Forward | `dep.06`, `dep.07` | the durable, worker, and agent charts follow the same schema and policy |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| the policy test over `helm template` | Kyverno, OPA Gatekeeper, Pod Security Admission | the same rules enforced by the API server for every install, not just yours | [Kyverno](https://kyverno.io/docs/) (free), [Pod Security Standards](https://kubernetes.io/docs/concepts/security/pod-security-standards/) (free) |
| Secrets created by hand | External Secrets Operator, Sealed Secrets | secrets synced from a vault, or encrypted in git | [External Secrets](https://external-secrets.io/) (free) |
| the umbrella chart | Argo CD, Flux | git as the source of truth for every release, with drift detection | [Argo CD](https://argo-cd.readthedocs.io/) (free) |
