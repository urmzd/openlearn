# Helm charts and the observability stack

The tracer charts from [the tracer deploy](00-tracer-deploy.md) worked once, on one cluster, with values you remembered. From here on they run the production platform. They install the gateway, a unified engine, then a prefill and two decode engines, plus the observability stack. The incident drills in *Systems and Architecture*, starting with [A decode worker dies mid-stream](../../systems/05-incident-response-and-chaos/01-kill-decode.md), kill Pods and drain nodes while load runs, and [the Tilt dev loop](04-tilt-dev-loop.md) reinstalls the charts on every save.

Three things that were fine for a tracer now break. A typo in `values.yaml`, such as `tag: latest` or a pepper pasted as a string, installs without complaint. A node drain or a rollout can take the only gateway Pod down, or hang forever, depending on a PodDisruptionBudget nobody wrote. And the gateway now writes a usage ledger, from [Usage ledger, metering, and admin API](../../ai-platform-engineering/12-gateway/07-usage-ledger-and-admin-api.md) in *AI Platform Engineering*, which a read-only root filesystem cannot hold and a Pod restart must not erase. This section makes the charts refuse bad values, encode the operating policy, keep the ledger's state, and install the observability stack that the observability chapter of *Systems and Architecture* fills.

You need the images from [Dockerfiles for the gateway and the engine](01-dockerfiles-for-gateway-and-engine.md) and the cluster and registry from [kind cluster, local registry, and port mappings](02-kind-cluster-and-local-registry.md). [Cloud native](../../systems/03-cloud-native/) in *Systems and Architecture* gives the broader context. This time there is a contract: the values schemas [`helm/gateway.values.schema.json`](../../../../course/contracts/helm/gateway.values.schema.json) and [`helm/engine.values.schema.json`](../../../../course/contracts/helm/engine.values.schema.json), and the pins and rule selector of [`helm/observability.md`](../../../../course/contracts/helm/observability.md).

> **Learning objectives**
> - Use a `values.schema.json` to make Helm refuse unsafe values before anything is installed.
> - Check an operating policy (probes, resources, non-root, pinned images, PDBs) on `helm template` output.
> - Compute how many Pods a PodDisruptionBudget lets a drain evict, and choose a budget that never hangs.
> - Keep secrets out of Helm by reference and keep stateful data on a PersistentVolumeClaim.
> - Install one chart several times as independent releases, and pin an umbrella chart of upstream dependencies.

## Installing the platform

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

Keep `deploy/observability/charts/` out of git (`.gitignore`): `Chart.lock` pins the versions and `helm dependency build` restores the archives. Notice the fourth line, a command you expect to fail. Making that failure happen is the first job of a production chart.

## Values schemas

Helm reads `values.schema.json` (JSON Schema draft-07) next to `Chart.yaml` and validates the merged values (defaults plus `-f` files plus `--set`) on `lint`, `template`, `install`, and `upgrade`. A schema therefore makes Helm refuse bad values before anything is installed.

The schemas live in `contracts/helm/` and change only through a contract version, so every chart that carries them accepts the same shape: `image.{repository, tag, digest, pullPolicy}`, `resources.{requests, limits}.{cpu, memory}`, `securityContext`, `probes.{liveness, readiness}.path`, `pdb.{enabled, maxUnavailable}`, and per chart `replicaCount` (gateway: exactly 1), `service`, `pepper` (gateway), and `role` and `model.dir` (engine). The schema rejects `tag: latest`, `runAsNonRoot: false`, an env var whose name looks secret but has a literal `value`, and a `pepper` that is not `{name, key}`. The contract defines the schema; the chart carries it unchanged.

## The policy, on rendered output

A schema checks values, but what matters is what Kubernetes receives. `helm template` renders the manifests without a cluster, so the policy can be checked on exactly that output:

| Rule | Why |
|---|---|
| liveness and readiness probes on every container | liveness restarts a hung process; readiness keeps traffic away until it can serve (gateway: `/readyz` follows routable workers) |
| `resources.requests` and `limits` for CPU and memory | requests drive scheduling; limits stop one engine from starving the node; without a memory limit the OOM killer picks a victim for you |
| `runAsNonRoot: true`, a numeric `runAsUser`, `allowPrivilegeEscalation: false` | the kubelet can verify a number, not a name; no setuid escapes |
| a pinned image, `imagePullPolicy` not `Always` | the same bytes on every node; kind does not re-pull |
| a PodDisruptionBudget selecting the Pods | a voluntary disruption (drain, cluster upgrade) respects it |

The last rule needs arithmetic. A **PodDisruptionBudget** limits voluntary evictions: with `maxUnavailable: 1`, the eviction API lets one Pod of the selected set go at a time. Let $r$ be the replicas of the Deployment, $u$ the `maxUnavailable` of its PDB, and $a$ the Pods that may be evicted at once:

$$a = \min(u, r)$$

The gateway has $r = 1$. With $u = 1$, $a = 1$: a drain evicts the Pod, a short outage that is the honest cost of one replica. With $u = 0$, $a = 0$: the eviction API refuses forever and `kubectl drain` never finishes, which turns a routine node upgrade into an incident. A single-replica gateway's PDB must allow one Pod down.

## Secrets by reference

A chart value that holds a secret ends up in git, in `helm get values`, and in the release's stored manifest. So the chart only **references** a Secret, naming a Secret and a key: `env: [{name: TL_GATEWAY_PEPPER, valueFrom: {secretKeyRef: {name: forge-gateway-pepper, key: pepper}}}]`. You create the Secret once with `kubectl create secret`. The same holds for the tracer's `TL_API_KEY` and for Grafana's admin password (`grafana.admin.existingSecret`).

## State: the ledger's volume

Secrets come from outside the chart; state must survive outside the Pod. The gateway's root filesystem is read-only (`readOnlyRootFilesystem: true`), and the node's `/artifacts` mount is read-only too. The ledger is a SQLite file in WAL mode, so it needs a writable directory that outlives the Pod, and an `emptyDir` dies with the Pod. The answer is a **PersistentVolumeClaim**. kind ships a `local-path` provisioner (StorageClass `standard`), which binds the claim to a directory on the node. One replica and `ReadWriteOnce` fit together: the volume is mounted by one node, and SQLite's file locks keep a second process on the same node safe during a rolling update.

## The observability stack

The last piece is not your code at all. `deploy/observability` is an **umbrella chart**: no templates of its own, and three `dependencies` pinned to the exact versions of `contracts/helm/observability.md` (`opentelemetry-collector` 0.173.1, `kube-prometheus-stack` 91.5.2, `tempo` 1.24.4). `helm dependency update` resolves them, writes `Chart.lock` (names, repositories, versions, and a digest of the requirements), and downloads the archives into `charts/`. Values for a subchart go under its name (or `alias`) in `values.yaml`.

One behavior of kube-prometheus-stack matters more than any other. Its Prometheus loads only the `PrometheusRule` and `ServiceMonitor` objects its selectors match. With the default (`ruleSelectorNilUsesHelmValues: true`), that is everything labelled `release: <release name>`. So the release is named `observability`, and every rule written in [SLOs and multi-window burn-rate alerts](../../systems/04-observability/03-slos-and-burn-rate-alerts.md) carries `release: observability`. Anything else is silently ignored.

## Working one release per role by hand

Disaggregated serving installs the engine chart three times:

```bash
helm upgrade --install forge-engine         deploy/helm/forge-engine -n forge
helm upgrade --install forge-engine-prefill deploy/helm/forge-engine -n forge --set role=prefill
helm upgrade --install forge-engine-decode  deploy/helm/forge-engine -n forge --set role=decode --set replicaCount=2
```

For the third, Helm merges `values.yaml` with `role=decode` and `replicaCount=2` and validates: `role` is in `[unified, prefill, decode]`, and `replicaCount` is an integer at least 0. The templates name everything after the release, which gives these objects:

| Object | Name | Key fields |
|---|---|---|
| Deployment | `forge-engine-decode` | `replicas: 2`; Pod labels `app.kubernetes.io/name: forge-engine-decode`, `tinyllm.role: decode`; env `TL_ENGINE__ROLE=decode` |
| Service | `forge-engine-decode` | ClusterIP, ports 8000, 50051, 50052 |
| PodDisruptionBudget | `forge-engine-decode` | `maxUnavailable: 1`, selector `app.kubernetes.io/name: forge-engine-decode` |

Apply the PDB arithmetic with $r = 2$ and $u = 1$: $a = \min(1, 2) = 1$, so a drain moves one decode Pod at a time and the other keeps serving. The workload name `deploy/forge-engine-decode` is how the decode-kill drill finds the Pod to kill. And `--set image.tag=latest` on any of the three stops at validation with `at '/image/tag': 'not' failed`, before Kubernetes sees anything.

## The files and what they must do

| Path | Holds |
|---|---|
| `deploy/helm/<system>-gateway/` | `values.schema.json` copied from the contract; values in the schema's shape; Deployment (1 replica, probes on the health port, Pod and container security context, env: pepper and API key by `secretKeyRef`, `TL_GATEWAY__USAGE_DB` on the claim), Service (NodePort 30080), PodDisruptionBudget (`maxUnavailable: 1`), PersistentVolumeClaim for the ledger, ConfigMap with `runtime.toml` when `runtime` is set (then `--config`; otherwise the tracer flags) |
| `deploy/helm/<system>-engine/` | the same for the engine, one release per `role`, the model under `/artifacts` read-only, the role passed to the engine |
| `deploy/observability/Chart.yaml`, `Chart.lock`, `values.yaml` | the three pinned dependencies; Prometheus on NodePort 30090, Grafana on 30300 with its admin password from a Secret, the default rule selector |

### What a correct solution must handle

Begin with conformance. The charts, schemas, and the stack's three files must be present. Each `values.schema.json` must equal the contract exactly, so every chart that consumes it has one shape, and both charts must pass `helm lint`. The schema must be live: `latest`, a literal pepper, two gateway replicas, and `role=leader` must all fail to render.

The rendered output must then satisfy the policy table, on the gateway and on the engine in each role, so the policy holds for every release. Secrets must appear only by reference: no rendered Secret with data, no literal secret env, the pepper by reference, so nothing secret lands in git or Helm. The gateway must have one replica and no PDB that allows zero disruptions, so drains in the drills finish. The gateway must mount a PVC the chart creates, so the bill survives restarts. Each engine release must get its own names and pass its role to the engine, as in the hand-worked example, which the disaggregated run and the decode-kill drill depend on. `Chart.yaml` and `Chart.lock` must match the pins, so the stack is reproducible and upgrades are explicit. The stack must keep the default selectors, NodePorts 30090 and 30300, and a Grafana password from a Secret, so the SLO rules load.

On the cluster, `kubectl apply --dry-run=server` of the gateway, the engine, and a decode engine must succeed, which proves the API server accepts every object. Finally, the gateway and engine must be available and the release `observability` deployed.

## Building it step by step

### Reasoning from first principles

1. **Bad values must fail before anything is installed.** Helm validates merged values against `values.schema.json` on every `lint`, `template`, `install`, and `upgrade`, so the schema is the earliest possible gate. It is a contract file, copied unchanged, so every consumer of the chart agrees on the shape.
2. **Every number lives in values, every rule in templates.** The templates only arrange values. The policy (probes, limits, non-root, pins, PDB) is therefore visible in `helm template` output, where it can be checked without a cluster.
3. **Everything is named after the release.** Names, selectors, and the PDB all use `{{ .Release.Name }}`, so installing the same chart three times (unified, prefill, decode) gives three independent sets of objects. A role switch inside one release would make them collide.
4. **A secret value never passes through Helm.** Whatever Helm sees ends up in its stored release, so the chart takes `{name, key}` references and the Secret is created out of band.
5. **State that outlives a Pod needs a claim.** The root filesystem is read-only and an `emptyDir` dies with the Pod, so the ledger gets a PersistentVolumeClaim created by the chart, sized in values.
6. **Disruption policy follows from replica count.** With one replica, the only workable PDB is $u = 1$; with two decode replicas the same $u = 1$ keeps one serving during a drain.
7. **The upstream stack must not drift.** Exact dependency versions plus `Chart.lock` make the stack reproducible. Keeping the default rule selector and naming the release `observability` makes `release: observability` the one label rules need.

### The schema at work

The contract schema is long; three of its definitions carry the policy.

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

The tag rule produces the `'not' failed` you saw in the hand-worked example. `secretRef` with `additionalProperties: false` is what makes both `pepper: "s3cr3t"` (a string, not `{name, key}`) and `pepper: {value: ...}` fail. The `envVar` rule reads: an env var either comes from a Secret, or its name must not look secret. The gateway schema also pins `"replicaCount": {"const": 1}`, because limits, cache, ledger, and registry are per process.

The full contract is [`gateway.values.schema.json`](../../../../course/contracts/helm/gateway.values.schema.json) and [`engine.values.schema.json`](../../../../course/contracts/helm/engine.values.schema.json); the charts carry identical copies.

### The gateway chart's production additions

The Deployment from the tracer deploy gains the pepper, the ledger, and the config form:

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

`fsGroup` is the detail that makes the claim usable. The `local-path` volume is created owned by root, and without `fsGroup` the non-root gateway could not create `usage.db` in it. The claim itself is four lines of substance:

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

When `runtime` is set, a ConfigMap renders it with `toToml` as `runtime.toml`, and the Pod template carries `checksum/runtime: {{ include (print $.Template.BasePath "/configmap.yaml") . | sha256sum }}` as an annotation. A ConfigMap change alone does not restart Pods; changing the annotation changes the Pod template, so Kubernetes rolls the Deployment and the new config is actually read.

The full solution is [`forge-gateway/templates/`](../../../../course/solutions/entry/deploy/helm/forge-gateway/templates/deployment.yaml) (`deployment.yaml`, `pvc.yaml`, `configmap.yaml`, `pdb.yaml`) and [`values.yaml`](../../../../course/solutions/entry/deploy/helm/forge-gateway/values.yaml).

### The PodDisruptionBudget

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

The template uses `maxUnavailable` rather than `minAvailable`. With `minAvailable: 1` and one replica, $a = 0$ and drains hang; `maxUnavailable: 1` gives $a = \min(1, r) \geq 1$ for any $r \geq 1$.

### One engine release per role

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

The role travels as a label (for selection and dashboards) and as an environment variable the engine reads, in the `TL_<SECTION>__<KEY>` form of `spec/cli-roles.md`, so the same image serves all three roles. Names come from the release, never from the role, which is why the hand-worked example gets three disjoint sets of objects. The full solution is [`forge-engine/templates/`](../../../../course/solutions/entry/deploy/helm/forge-engine/templates/deployment.yaml) and [`values.yaml`](../../../../course/solutions/entry/deploy/helm/forge-engine/values.yaml).

### The observability umbrella

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

`fullnameOverride` on the collector and Tempo gives them Service names that do not depend on the release name, so the collector config of [Tracing for the serving path](../../systems/04-observability/01-tracing-for-the-serving-path.md) can address `tempo` directly. `charts/` is git-ignored: `Chart.lock` is the pin, and the archives are a cache that `helm dependency build` restores.

The full solution is [`Chart.yaml`](../../../../course/solutions/entry/deploy/observability/Chart.yaml), [`values.yaml`](../../../../course/solutions/entry/deploy/observability/values.yaml), and [`Chart.lock`](../../../../course/solutions/entry/deploy/observability/Chart.lock).

### Why this design

- **A JSON Schema over template-time `fail` calls.** `{{ fail }}` in templates works but spreads the rules across files and runs only when that template branch renders; the schema checks all values, every time, in one place that is also documentation.
- **Policy checked on rendered output over checking `values.yaml`.** What matters is what Kubernetes receives; a default in a template or an override with `--set` would escape a values-only check. In production an admission controller (Kyverno, Pod Security Admission) enforces the same rules at the API server.
- **A PVC over `hostPath` for the ledger.** `hostPath` would also survive restarts on kind, but it bypasses the storage API; the claim works unchanged on any cluster with a default StorageClass.
- **An umbrella chart over three separate `helm install`s.** One release, one lock file, one upgrade; the cost is that upgrading one component means bumping the umbrella.
- **Tradeoff accepted:** the single gateway replica means a drain is a short outage. That is stated (the schema's `const: 1`) rather than hidden behind a PDB that would hang the drain.

## What can go wrong

The failures below share a theme: each one is silent at install time and loud only later, under a drain, a restart, or an alert that never fires.

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| no `values.schema.json`, or a loosened copy | `tag: latest` and a pasted pepper install without a word | the contract schema copied unchanged |
| a probe on the API port, or none | a hung engine keeps receiving traffic; a slow start restarts forever | both probes on the named `health` port, timings from values (as in the tracer engine chart) |
| `maxUnavailable: 0` (or `minAvailable: 1`) on one replica | `kubectl drain` waits forever; the cluster upgrade stalls | `maxUnavailable: 1` |
| the ledger on an `emptyDir` or the root filesystem | usage resets on every restart, or the gateway cannot write at all | a chart-created PVC with `fsGroup` |
| a custom `ruleSelector`, or a release not named `observability` | the SLO alerts are valid YAML and never fire | default selectors, release named `observability` |
| `version: ">=91.0.0"` and no `Chart.lock` | the stack changes under you on the next install | exact versions plus `Chart.lock` |
| one engine release with `role` in a template conditional | prefill and decode share a name; installing one replaces the other | names from the release, role only as label and env |
| `adminPassword: admin` for Grafana | the password is in git and in `helm get values` | `grafana.admin.existingSecret` |

> **Warning:** A PodDisruptionBudget that allows zero disruptions does not make a workload safer. It makes every voluntary disruption impossible, so the first node upgrade becomes an incident.

## Where this leads

The charts run the images of [Dockerfiles for the gateway and the engine](01-dockerfiles-for-gateway-and-engine.md) on the cluster, registry, and NodePorts of [kind cluster, local registry, and port mappings](02-kind-cluster-and-local-registry.md). The production platform installs the gateway, the engines (unified, then prefill and two decode), and the stack, and conformance tests, load tests, and [A decode worker dies mid-stream](../../systems/05-incident-response-and-chaos/01-kill-decode.md) run against them. [The Tilt dev loop](04-tilt-dev-loop.md) reinstalls these charts on every change. The observability chapter of *Systems and Architecture* installs its collector pipeline, ServiceMonitors, rules, and dashboards into this stack, in [Tracing for the serving path](../../systems/04-observability/01-tracing-for-the-serving-path.md), [Metrics and log correlation](../../systems/04-observability/02-metrics-and-log-correlation.md), [SLOs and multi-window burn-rate alerts](../../systems/04-observability/03-slos-and-burn-rate-alerts.md), and [Dashboards as code](../../systems/04-observability/04-dashboards-as-code.md). Later in this chapter, [Durable and worker images and charts](06-durable-and-worker-charts.md) and [The agent image and chart](07-agent-image-and-chart.md) follow the same schema and policy.

The production equivalents of the pieces here:

| Your piece | Production equivalent | What it adds |
|---|---|---|
| the policy test over `helm template` | Kyverno, OPA Gatekeeper, Pod Security Admission | the same rules enforced by the API server for every install, not just one chart's |
| Secrets created by hand | External Secrets Operator, Sealed Secrets | secrets synced from a vault, or encrypted in git |
| the umbrella chart | Argo CD, Flux | git as the source of truth for every release, with drift detection |

### Further reading

- [Helm: schema files](https://helm.sh/docs/topics/charts/#schema-files)
- [Kubernetes: disruptions and PodDisruptionBudgets](https://kubernetes.io/docs/concepts/workloads/pods/disruptions/)
- [kube-prometheus-stack](https://github.com/prometheus-community/helm-charts/tree/main/charts/kube-prometheus-stack)
- [Helm: chart dependencies](https://helm.sh/docs/helm/helm_dependency/)
- [Kyverno](https://kyverno.io/docs/) and [Pod Security Standards](https://kubernetes.io/docs/concepts/security/pod-security-standards/)
- [External Secrets](https://external-secrets.io/)
- [Argo CD](https://argo-cd.readthedocs.io/)

## Key terms

- **values schema**: a `values.schema.json` that Helm uses to validate merged values on every lint, template, install, and upgrade.
- **PodDisruptionBudget (PDB)**: an object that limits how many Pods of a set voluntary evictions may take down at once.
- **`maxUnavailable`**: the PDB field giving the number of Pods that may be unavailable; with $r$ replicas, $\min(u, r)$ may be evicted at once.
- **secret by reference**: a chart that names a Secret and key via `secretKeyRef` instead of holding the value.
- **PersistentVolumeClaim (PVC)**: a request for storage that outlives any one Pod.
- **`fsGroup`**: a Pod security setting that gives the Pod's group ownership of mounted volume files.
- **umbrella chart**: a chart with no templates of its own that pins and configures other charts as dependencies.
- **rule selector**: the label selector Prometheus uses to decide which `PrometheusRule` and `ServiceMonitor` objects to load.

## Summary

- A chart's `values.schema.json` makes Helm refuse bad values before anything is installed (`latest`, a literal pepper, a second gateway replica); the contract defines the schema and the chart carries it unchanged.
- The policy is checked on what Kubernetes receives (`helm template`): probes, requests and limits, a numeric non-root user, no privilege escalation, pinned images, and a PDB for every workload.
- A single-replica gateway's PDB must allow one Pod down; `maxUnavailable: 0` makes every node drain hang.
- Secrets only by reference: the chart names a Secret and a key, and you create the Secret by hand. The usage ledger lives on a PVC, because the root filesystem is read-only and an `emptyDir` dies with the Pod.
- Upstream charts are pinned in `Chart.yaml` and `Chart.lock`, and the stack keeps the default rule selector, so rules labelled `release: observability` load and anything else is silently ignored.

## Review questions

1. Why does `minAvailable: 1` hang a drain of the single gateway Pod while `maxUnavailable: 1` does not?
   <details><summary>Answer</summary>With one replica, <code>minAvailable: 1</code> leaves $a = 1 - 1 = 0$ evictions allowed, so the eviction API refuses forever. <code>maxUnavailable: 1</code> allows $a = \min(1, 1) = 1$: the Pod is evicted and rescheduled, at the cost of a short outage.</details>
2. Installing `forge-engine-prefill` and `forge-engine-decode` from the same chart gives two Deployments. What in the templates guarantees they do not share Pods?
   <details><summary>Answer</summary>Every name and every selector uses <code>{{ .Release.Name }}</code> (the label <code>app.kubernetes.io/name</code>), so each Deployment, Service, and PDB selects only its own release's Pods. The role only sets a label value and an environment variable.</details>
3. Why does the gateway Pod set `fsGroup`?
   <details><summary>Answer</summary>The volume behind the ledger claim is created owned by root. <code>fsGroup</code> makes Kubernetes give the Pod's group ownership of the mounted files, so the non-root gateway (uid and gid 10001) can create and write <code>usage.db</code>.</details>
4. A new `PrometheusRule` is valid YAML and applied, but its alert never fires. What is the first thing to check?
   <details><summary>Answer</summary>Its labels: with the default selector, Prometheus loads only rules labelled <code>release: observability</code>, matching the release name. A missing label, or a release installed under a different name, makes the rule silently ignored.</details>
5. Why does a change to `runtime` in values restart the gateway, given that ConfigMap changes alone do not?
   <details><summary>Answer</summary>The Pod template carries a <code>checksum/runtime</code> annotation computed from the rendered ConfigMap. A different config gives a different checksum, which changes the Pod template and triggers a rollout.</details>
