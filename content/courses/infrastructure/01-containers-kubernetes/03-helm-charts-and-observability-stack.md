# Helm charts for the gateway and the engine, plus the observability stack

## Overview

| | |
|---|---|
| **Module** | `dep.03` · practice · ops · Pass 7 · 5 to 8 h |
| **Solution** | [`deploy/helm/forge-gateway/`](../../../../course/solutions/entry/deploy/helm/forge-gateway/values.yaml) and [`deploy/helm/forge-engine/`](../../../../course/solutions/entry/deploy/helm/forge-engine/values.yaml) grown into production charts (the contract's values schema, PodDisruptionBudgets, the ledger's volume claim, one engine release per role, a runtime ConfigMap), and [`deploy/observability/`](../../../../course/solutions/entry/deploy/observability/Chart.yaml) (an umbrella chart pinning the collector, Prometheus with Grafana, and Tempo) |
| **Contract** | the values schemas [`helm/gateway.values.schema.json`](../../../../course/contracts/helm/gateway.values.schema.json) and [`helm/engine.values.schema.json`](../../../../course/contracts/helm/engine.values.schema.json), the pins and rule selector of [`helm/observability.md`](../../../../course/contracts/helm/observability.md) |
| **Needs** | `dep.01` the images, `dep.02` the cluster and registry; reading: `dep.00` the tracer charts, `gw.07` the ledger the gateway chart must keep, [Cloud native](../../systems/03-cloud-native/) |
| **Used by** | the Pass 7 production platform runs on these charts; `dep.04` (Tilt), `obs.01` to `obs.04`, and the drills build on them |
| **Optional depth** | [Helm: schema files](https://helm.sh/docs/topics/charts/#schema-files) (free); [Kubernetes: disruptions and PodDisruptionBudgets](https://kubernetes.io/docs/concepts/workloads/pods/disruptions/) (free); [kube-prometheus-stack](https://github.com/prometheus-community/helm-charts/tree/main/charts/kube-prometheus-stack) (free); [Helm: chart dependencies](https://helm.sh/docs/helm/helm_dependency/) (free) |

## Key Takeaways

- A chart's **`values.schema.json`** makes Helm refuse bad values before anything is installed: tag `latest`, a literal pepper, a second gateway replica. The contract defines the schema; the chart carries it unchanged.
- The **policy** is checked on what Kubernetes receives (`helm template`): probes, requests and limits, a numeric non-root user, no privilege escalation, pinned images, a **PodDisruptionBudget** for every workload.
- A single-replica gateway's PDB must allow **one pod down**; `maxUnavailable: 0` makes every node drain hang.
- **Secrets only by reference**: the chart names a Secret and a key; you create the Secret by hand. The usage ledger lives on a **PersistentVolumeClaim**, because the root filesystem is read-only and an `emptyDir` dies with the pod.
- Upstream charts are **pinned** in `Chart.yaml` and `Chart.lock`, and the stack keeps kube-prometheus-stack's default **rule selector**: rules labelled `release: observability` load, anything else is silently ignored.

## How to read this chapter

Sections 1 to 3 derive what a production chart must refuse and guarantee, and render one engine release per role by hand. Section 4 fixes the artifacts. Section 5 walks through the solution's charts and observability stack and explains why each piece is there. Try to predict each template before reading it.

The solution is installed from the repo root like this (`forge` is the system's name):

```bash
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
```

Keep `deploy/observability/charts/` out of git (`.gitignore`): `Chart.lock` pins the versions and `helm dependency build` restores the archives.

---

## 1. Why now

The Pass 1 charts worked once, on one cluster, with values you remembered. Pass 7 runs them for the rest of the course: the production platform installs the gateway, a unified engine, then a prefill and two decode engines, plus the observability stack; the drills (`ops.01` and later) kill pods and drain nodes while load runs; `dep.04` reinstalls them on every save. Three things that were fine for a tracer now break. A typo in `values.yaml` (`tag: latest`, a pepper pasted as a string) installs without complaint. A node drain or a rollout can take the only gateway pod down, or hang forever, depending on a PodDisruptionBudget nobody wrote. And the gateway now writes a usage ledger (`gw.07`) that a read-only root filesystem cannot hold and a pod restart must not erase. This module makes the charts refuse bad values, encode the operating policy, and install the observability stack that `obs.*` fills.

## 2. Principles

### 2.1 Values schemas

Helm reads `values.schema.json` (JSON Schema draft-07) next to `Chart.yaml` and validates the merged values (defaults plus `-f` files plus `--set`) on `lint`, `template`, `install`, and `upgrade`. The schemas live in `contracts/helm/` and change only through a contract version, so every chart that carries them accepts the same shape: `image.{repository, tag, digest, pullPolicy}`, `resources.{requests, limits}.{cpu, memory}`, `securityContext`, `probes.{liveness, readiness}.path`, `pdb.{enabled, maxUnavailable}`, and per chart `replicaCount` (gateway: exactly 1), `service`, `pepper` (gateway), `role` and `model.dir` (engine). The schema rejects `tag: latest`, `runAsNonRoot: false`, an env var whose name looks secret but has a literal `value`, and a `pepper` that is not `{name, key}`.

### 2.2 The policy, on rendered output

`helm template` renders the manifests without a cluster, so the policy can be checked on exactly what Kubernetes would receive:

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

The PDB arithmetic of 2.2 with $r = 2$, $u = 1$: $a = \min(1, 2) = 1$, so a drain moves one decode pod at a time and the other keeps serving. The workload name `deploy/forge-engine-decode` is how `ops.01` finds the pod to kill. And `--set image.tag=latest` on any of the three stops at validation: `at '/image/tag': 'not' failed`, before Kubernetes sees anything.

## 4. The artifact and what it must do

| Path | Holds |
|---|---|
| `deploy/helm/<system>-gateway/` | `values.schema.json` copied from the contract; values in the schema's shape; Deployment (1 replica, probes on the health port, pod and container security context, env: pepper and API key by `secretKeyRef`, `TL_GATEWAY__USAGE_DB` on the claim), Service (NodePort 30080), PodDisruptionBudget (`maxUnavailable: 1`), PersistentVolumeClaim for the ledger, ConfigMap with `runtime.toml` when `runtime` is set (then `--config`; otherwise the tracer flags) |
| `deploy/helm/<system>-engine/` | the same for the engine, one release per `role`, the model under `/artifacts` read-only, the role passed to the engine |
| `deploy/observability/Chart.yaml`, `Chart.lock`, `values.yaml` | the three pinned dependencies; Prometheus on NodePort 30090, Grafana on 30300 with its admin password from a Secret, the default rule selector |

### What a correct solution must handle

| Case | Kind of check | What must hold | Why it matters downstream |
|---|---|---|---|
| layout present | static | charts, schemas, the stack's three files | |
| values schemas are the contract | conformance | each `values.schema.json` equals the contract | one shape for every chart that consumes it |
| charts lint | conformance | `helm lint` both charts | |
| schema rejects unsafe values | boundary | `latest`, a literal pepper, 2 gateway replicas, `role=leader` all fail to render | the schema is live |
| policy over rendered charts | static | 2.2 on the gateway and on the engine in each role | the operating policy holds for every release |
| secrets only by reference | static | no rendered Secret with data, no literal secret env, the pepper by reference | nothing secret in git or Helm |
| one gateway replica that can drain | boundary | 1 replica; no PDB that allows zero disruptions | drains in the drills finish |
| ledger on a volume claim | static | the gateway mounts a PVC the chart creates | the bill survives restarts |
| one engine release per role | static | section 3: names per release, the role reaches the engine | the disaggregated run, `ops.01` |
| observability stack pinned | conformance | `Chart.yaml` and `Chart.lock` match the pins | reproducible stack; upgrades are explicit |
| rule selector and ports | static | default selectors, NodePorts 30090 and 30300, Grafana password from a Secret | `obs.03` rules load |
| server-side dry run | cluster | `kubectl apply --dry-run=server` of the gateway, engine, and a decode engine | the API server accepts every object |
| releases available | cluster | gateway and engine available; release `observability` deployed | |

## 5. The solution, step by step

### 5.1 From first principles

1. **Bad values must fail before anything is installed.** Helm validates merged values against `values.schema.json` on every `lint`, `template`, `install`, and `upgrade`, so the schema is the earliest possible gate. It is a contract file, copied unchanged, so every consumer of the chart agrees on the shape.
2. **Every number lives in values, every rule in templates.** The templates only arrange values; the policy (probes, limits, non-root, pins, PDB) is therefore visible in `helm template` output, where it can be checked without a cluster.
3. **Everything is named after the release.** Names, selectors, and the PDB all use `{{ .Release.Name }}`, so installing the same chart three times (unified, prefill, decode) gives three independent sets of objects. A role switch inside one release would make them collide.
4. **A secret value never passes through Helm.** Whatever Helm sees ends up in its stored release, so the chart takes `{name, key}` references and the Secret is created out of band.
5. **State that outlives a pod needs a claim.** The root filesystem is read-only and an `emptyDir` dies with the pod; the ledger gets a PersistentVolumeClaim created by the chart, sized in values.
6. **Disruption policy follows from replica count.** With one replica, the only workable PDB is $u = 1$ (2.2); with two decode replicas the same $u = 1$ keeps one serving during a drain.
7. **The upstream stack must not drift.** Exact dependency versions plus `Chart.lock` make the stack reproducible; keeping the default rule selector and naming the release `observability` makes `release: observability` the one label rules need.

### 5.2 The schema at work

The contract schema is long; three of its definitions carry the policy:

```json
"tag": {
  "type": "string",
  "pattern": "^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$",
  "not": { "enum": ["latest", ""] }
}
```

```json
"secretRef": {
  "type": "object",
  "required": ["name", "key"],
  "additionalProperties": false
}
```

```json
"envVar": {
  "anyOf": [
    { "required": ["valueFrom"] },
    { "properties": { "name": { "not": { "pattern": "(?i)(KEY|SECRET|PEPPER|TOKEN|PASSWORD|CREDENTIAL)" } } } }
  ]
}
```

The tag rule produces the `'not' failed` of section 3. `secretRef` with `additionalProperties: false` is what makes `pepper: "s3cr3t"` (a string, not `{name, key}`) and `pepper: {value: ...}` both fail. The `envVar` rule reads: an env var either comes from a Secret, or its name must not look secret. The gateway schema also pins `"replicaCount": {"const": 1}`, because limits, cache, ledger, and registry are per process.

Full contract: [`gateway.values.schema.json`](../../../../course/contracts/helm/gateway.values.schema.json), [`engine.values.schema.json`](../../../../course/contracts/helm/engine.values.schema.json); the charts carry identical copies.

### 5.3 The gateway chart's production additions

The Deployment of `dep.00` (quoted in that chapter) gains the pepper, the ledger, and the config form:

```yaml
    spec:
      securityContext:
        runAsNonRoot: {{ .Values.securityContext.runAsNonRoot }}
        runAsUser: {{ .Values.securityContext.runAsUser }}
        {{- with .Values.securityContext.runAsGroup }}
        runAsGroup: {{ . }}
        fsGroup: {{ . }}          # the PVC's files become writable by group 10001
        {{- end }}
      containers:
        - name: gateway
          args:
            {{- if .Values.runtime }}
            - --config
            - /etc/tinyllm/runtime.toml
            {{- else }}
            # ... the tracer flags of dep.00
            {{- end }}
          env:
            # ... TL_API_KEY by secretKeyRef (dep.00)
            - name: TL_GATEWAY_PEPPER
              valueFrom:
                secretKeyRef:
                  name: {{ .Values.pepper.name }}
                  key: {{ .Values.pepper.key }}
            - name: TL_GATEWAY__USAGE_DB
              value: {{ printf "%s/usage.db" .Values.ledger.mountPath | quote }}
          securityContext:
            allowPrivilegeEscalation: {{ .Values.securityContext.allowPrivilegeEscalation }}
            readOnlyRootFilesystem: {{ .Values.securityContext.readOnlyRootFilesystem | default true }}
          volumeMounts:
            - name: ledger
              mountPath: {{ .Values.ledger.mountPath }}     # /var/lib/gateway: the one writable path
      volumes:
        - name: ledger
          persistentVolumeClaim:
            claimName: {{ .Release.Name }}-ledger
```

`fsGroup` is the detail that makes the claim usable: the `local-path` volume is created owned by root, and without `fsGroup` the non-root gateway could not create `usage.db` in it. The claim itself is four lines of substance:

```yaml
kind: PersistentVolumeClaim
metadata:
  name: {{ .Release.Name }}-ledger
spec:
  accessModes: ["ReadWriteOnce"]    # one node mounts it; matches replicaCount 1
  resources:
    requests:
      storage: {{ .Values.ledger.size }}    # 1Gi
```

When `runtime` is set, a ConfigMap renders it with `toToml` as `runtime.toml`, and the pod template carries `checksum/runtime: {{ include (print $.Template.BasePath "/configmap.yaml") . | sha256sum }}` as an annotation. A ConfigMap change alone does not restart pods; changing the annotation changes the pod template, so Kubernetes rolls the Deployment and the new config is actually read.

Full solution: [`forge-gateway/templates/`](../../../../course/solutions/entry/deploy/helm/forge-gateway/templates/deployment.yaml) (`deployment.yaml`, `pvc.yaml`, `configmap.yaml`, `pdb.yaml`), [`values.yaml`](../../../../course/solutions/entry/deploy/helm/forge-gateway/values.yaml).

### 5.4 The PodDisruptionBudget

Both charts use the same template:

```yaml
{{- if .Values.pdb.enabled }}     # the schema makes enabled const true
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: {{ .Release.Name }}
spec:
  maxUnavailable: {{ .Values.pdb.maxUnavailable }}   # 1: a single replica can still drain
  selector:
    matchLabels:
      app.kubernetes.io/name: {{ .Release.Name }}    # exactly this release's pods
{{- end }}
```

`maxUnavailable` rather than `minAvailable`: with `minAvailable: 1` and one replica, $a = 0$ and drains hang (pitfall 3); `maxUnavailable: 1` gives $a = \min(1, r) \geq 1$ for any $r \geq 1$.

### 5.5 One engine release per role

```yaml
# _helpers.tpl
{{- define "engine.labels" -}}
app.kubernetes.io/name: {{ .Release.Name }}
app.kubernetes.io/component: engine
tinyllm.role: {{ .Values.role }}
{{- end -}}
```

```yaml
# deployment.yaml (excerpt)
          env:
            - name: TL_ENGINE__ROLE
              value: {{ .Values.role | quote }}       # unified | prefill | decode
            - name: TL_ENGINE__MODEL_DIR
              value: {{ .Values.model.dir | quote }}
```

The role travels as a label (for selection and dashboards) and as an environment variable the engine reads, in the `TL_<SECTION>__<KEY>` form of `spec/cli-roles.md`, so the same image serves all three roles. Names come from the release, never from the role, which is why section 3 gets three disjoint sets of objects.

Full solution: [`forge-engine/templates/`](../../../../course/solutions/entry/deploy/helm/forge-engine/templates/deployment.yaml), [`values.yaml`](../../../../course/solutions/entry/deploy/helm/forge-engine/values.yaml).

### 5.6 The observability umbrella

```yaml
# Chart.yaml
apiVersion: v2
name: observability
type: application
version: 0.2.0
dependencies:                       # exact versions, from contracts/helm/observability.md
  - name: opentelemetry-collector
    alias: otel-collector           # values go under otel-collector:
    version: 0.173.1
    repository: https://open-telemetry.github.io/opentelemetry-helm-charts
  - name: kube-prometheus-stack
    version: 91.5.2
    repository: https://prometheus-community.github.io/helm-charts
  - name: tempo
    version: 1.24.4
    repository: https://grafana.github.io/helm-charts
```

```yaml
# values.yaml (excerpt)
kube-prometheus-stack:
  # Leave the rule selector at its default: Prometheus loads PrometheusRules
  # and ServiceMonitors labelled release: observability (the release name).
  prometheus:
    service: { type: NodePort, nodePort: 30090 }
    prometheusSpec:
      ruleSelectorNilUsesHelmValues: true
      serviceMonitorSelectorNilUsesHelmValues: true
  grafana:
    service: { type: NodePort, nodePort: 30300 }
    # The admin password comes from a Secret you create, never from this file.
    admin:
      existingSecret: grafana-admin
      userKey: admin-user
      passwordKey: admin-password
tempo:
  fullnameOverride: tempo           # a stable Service name: tempo.observability:3200
```

`fullnameOverride` on the collector and Tempo gives them Service names that do not depend on the release name, so the collector config of `obs.01` can address `tempo` directly. `charts/` is git-ignored: `Chart.lock` is the pin, the archives are a cache that `helm dependency build` restores.

Full solution: [`Chart.yaml`](../../../../course/solutions/entry/deploy/observability/Chart.yaml), [`values.yaml`](../../../../course/solutions/entry/deploy/observability/values.yaml), [`Chart.lock`](../../../../course/solutions/entry/deploy/observability/Chart.lock).

### 5.7 Why this solution

- **A JSON Schema over template-time `fail` calls.** `{{ fail }}` in templates works but spreads the rules across files and runs only when that template branch renders; the schema checks all values, every time, in one place that is also documentation.
- **Policy checked on rendered output over checking `values.yaml`.** What matters is what Kubernetes receives; a default in a template or an override with `--set` would escape a values-only check. In production an admission controller (Kyverno, Pod Security Admission) enforces the same rules at the API server.
- **A PVC over `hostPath` for the ledger.** `hostPath` would also survive restarts on kind, but it bypasses the storage API; the claim works unchanged on any cluster with a default StorageClass.
- **An umbrella chart over three separate `helm install`s.** One release, one lock file, one upgrade; the cost is that upgrading one component means bumping the umbrella.
- **Tradeoff accepted:** the single gateway replica means a drain is a short outage. That is stated (the schema's `const: 1`) rather than hidden behind a PDB that would hang the drain.

### 5.8 Explain it back

1. Why does `minAvailable: 1` hang a drain of the single gateway pod while `maxUnavailable: 1` does not?
   <details><summary>Answer</summary>With one replica, <code>minAvailable: 1</code> leaves $a = 1 - 1 = 0$ evictions allowed, so the eviction API refuses forever. <code>maxUnavailable: 1</code> allows $a = \min(1, 1) = 1$: the pod is evicted and rescheduled, at the cost of a short outage.</details>
2. Installing `forge-engine-prefill` and `forge-engine-decode` from the same chart gives two Deployments. What in the templates guarantees they do not share pods?
   <details><summary>Answer</summary>Every name and every selector uses <code>{{ .Release.Name }}</code> (the label <code>app.kubernetes.io/name</code>), so each Deployment, Service, and PDB selects only its own release's pods. The role only sets a label value and an environment variable.</details>
3. Why does the gateway pod set `fsGroup`?
   <details><summary>Answer</summary>The volume behind the ledger claim is created owned by root. <code>fsGroup</code> makes Kubernetes give the pod's group ownership of the mounted files, so the non-root gateway (uid and gid 10001) can create and write <code>usage.db</code>.</details>
4. A new `PrometheusRule` is valid YAML and applied, but its alert never fires. What is the first thing to check?
   <details><summary>Answer</summary>Its labels: with the default selector, Prometheus loads only rules labelled <code>release: observability</code>, matching the release name. A missing label, or a release installed under a different name, makes the rule silently ignored.</details>
5. Why does a change to `runtime` in values restart the gateway, given that ConfigMap changes alone do not?
   <details><summary>Answer</summary>The pod template carries a <code>checksum/runtime</code> annotation computed from the rendered ConfigMap. A different config gives a different checksum, which changes the pod template and triggers a rollout.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. no `values.schema.json`, or a loosened copy | `tag: latest` and a pasted pepper install without a word | the contract schema copied unchanged (5.2) |
| 2. a probe on the API port, or none | a hung engine keeps receiving traffic; a slow start restarts forever | both probes on the named `health` port, timings from values (see `dep.00`, section 5.6) |
| 3. `maxUnavailable: 0` (or `minAvailable: 1`) on one replica | `kubectl drain` waits forever; the cluster upgrade stalls | `maxUnavailable: 1` (5.4) |
| 4. the ledger on an `emptyDir` or the root filesystem | usage resets on every restart, or the gateway cannot write at all | a chart-created PVC with `fsGroup` (5.3) |
| 5. a custom `ruleSelector`, or a release not named `observability` | the SLO alerts of `obs.03` are valid YAML and never fire | default selectors, release named `observability` (5.6) |
| 6. `version: ">=91.0.0"` and no `Chart.lock` | the stack changes under you on the next install | exact versions plus `Chart.lock` (5.6) |
| 7. one engine release with `role` in a template conditional | prefill and decode share a name; installing one replaces the other | names from the release, role only as label and env (5.5) |
| 8. `adminPassword: admin` for Grafana | the password is in git and in `helm get values` | `grafana.admin.existingSecret` (5.6) |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `dep.01` | the images the charts run |
| Back | `dep.02` | the cluster, the registry at `localhost:5001`, the NodePorts |
| Forward | Pass 7 production run | installs the gateway, the engines (unified, then prefill and two decode), and the stack; conformance, load, and `ops.01` run against them |
| Forward | `dep.04` | Tilt reinstalls these charts on every change |
| Forward | `obs.01` to `obs.04` | the collector pipeline, ServiceMonitors, rules, and dashboards installed into this stack |
| Forward | `dep.06`, `dep.07` | the durable, worker, and agent charts follow the same schema and policy |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| the policy test over `helm template` | Kyverno, OPA Gatekeeper, Pod Security Admission | the same rules enforced by the API server for every install, not just one chart's | [Kyverno](https://kyverno.io/docs/) (free), [Pod Security Standards](https://kubernetes.io/docs/concepts/security/pod-security-standards/) (free) |
| Secrets created by hand | External Secrets Operator, Sealed Secrets | secrets synced from a vault, or encrypted in git | [External Secrets](https://external-secrets.io/) (free) |
| the umbrella chart | Argo CD, Flux | git as the source of truth for every release, with drift detection | [Argo CD](https://argo-cd.readthedocs.io/) (free) |
