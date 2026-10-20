# Durable and worker images and charts

The durable orchestration chapter of *AI Engineering*, [Durable Orchestration and Workers](../07-durable-orchestration/), builds a durable server and workers that survive kills on your laptop. On kind they need three things the serving charts never had to think about. The server **has state**: a Deployment Pod writes its write-ahead log (WAL) to the container filesystem, and the first reschedule erases every workflow. The worker's **load is a queue**, not requests: a corpus build enqueues 200 shard tasks in a second, the CPU of two idle workers says nothing about it, and a fixed replica count either wastes the cluster or leaves the queue growing. And the worker's work is **long**: a training step runs for minutes inside a Python child, so the way Kubernetes stops a Pod (SIGTERM, then SIGKILL 30 s later) decides whether a rollout loses a checkpoint.

This section packages both components so they keep the promises of that chapter, from [the append-only event log](../07-durable-orchestration/01-event-log.md) to [the subprocess activity runner](../07-durable-orchestration/09-subprocess-activities.md), when Kubernetes, not a person at a terminal, decides when Pods start and stop. You will build two images (`durable.Dockerfile` and `worker.Dockerfile`), a durable chart whose StatefulSet keeps its WAL on a PersistentVolumeClaim with the WAL quota rendered into `runtime.toml`, a worker chart whose Deployment drains before it is killed, writes `/artifacts`, and is scaled by a KEDA `ScaledObject`, and a small `deploy/keda` chart that pins KEDA.

The charts follow the policy and join the stack of [Helm charts and the observability stack](04-helm-charts-and-observability-stack.md), with the image rules of [Dockerfiles for the gateway and the engine](02-dockerfiles-for-gateway-and-engine.md) and the cluster of [kind cluster, local registry, and port mappings](03-kind-cluster-and-local-registry.md). The images build the entry points of [the workflow service](../07-durable-orchestration/02-workflow-service.md) and [the task protocol and worker pool](../07-durable-orchestration/04-task-protocol-and-workers.md), and the worker image carries the Python units of the subprocess runner. The contract is the values schemas [`durable.values.schema.json`](../../../../course/contracts/helm/durable.values.schema.json) and [`worker.values.schema.json`](../../../../course/contracts/helm/worker.values.schema.json), the KEDA pin in [`helm/observability.md`](../../../../course/contracts/helm/observability.md), the gauge `tl.durable.task_queue.depth` in [`otel/metrics.yaml`](../../../../course/contracts/otel/metrics.yaml), and the `[durable]` and `[worker]` sections of [`config/runtime.schema.json`](../../../../course/contracts/config/runtime.schema.json).

> **Learning objectives**
> - Run a single-writer stateful server as a one-replica StatefulSet with a WAL claim that outlives its Pod.
> - Render an application-level disk quota as an integer and size it below the claim.
> - Choose a termination grace period from a worker's drain and checkpoint budgets.
> - Scale workers on queue depth with KEDA and compute the replica count for a given backlog.
> - Build a worker image that combines a static Go binary with a Python runtime.

## Installing the charts

The solution is installed from the repo root:

```bash
helm dependency update deploy/keda    # writes deploy/keda/Chart.lock
helm upgrade --install keda deploy/keda -n keda --create-namespace
helm upgrade --install forge-durable deploy/helm/forge-durable -n forge
helm upgrade --install forge-worker deploy/helm/forge-worker -n forge
```

KEDA comes first because it provides the `ScaledObject` CRD the worker chart uses. Before reading any chart, though, start with the images.

## Two images, one rule set

Both Dockerfiles keep the rules of [Dockerfiles for the gateway and the engine](02-dockerfiles-for-gateway-and-engine.md): multi-stage, every base pinned by **digest**, a numeric non-root `USER`, an exec-form `ENTRYPOINT` (the process is PID 1 and receives SIGTERM, which the worker needs to drain), a `HEALTHCHECK` in exec form on the health port 9464, and the OCI labels.

The durable image is a static Go binary on alpine, like the gateway. The worker image is different, and it is really two worlds. Its runtime stage is a **Python** base (glibc, so numpy and pyarrow install as wheels) that also carries the static Go worker and the `python/` tree, because the worker execs `{tinyllm} train` and `{corpus} run --stage` as subprocess activities.

## The values contract and the policy

Each chart carries its `values.schema.json` copied unchanged from `contracts/helm/`, so Helm refuses unsafe values before an install: `image.tag=latest`, `testClock=true` (a test-only clock in a cluster), `persistence.enabled=false`, a missing or zero `walMaxBytes`, two durable replicas without Raft, and a worker with no task queue. The policy of [Helm charts and the observability stack](04-helm-charts-and-observability-stack.md) applies to what `helm template` renders: probes, CPU and memory requests and limits, a numeric non-root user without privilege escalation, a pinned image, a PodDisruptionBudget for every workload, and secrets only by reference.

## The durable server: StatefulSet, WAL claim, quota

The durable server's state is its WAL, and each Kubernetes object in its chart exists to protect it:

| Object | Why |
|---|---|
| `StatefulSet`, `replicas: 1`, `podManagementPolicy: OrderedReady` | one writer per WAL; a stable Pod name (`<system>-durable-0`) that Raft peers can address later ([Raft HA](../07-durable-orchestration/10-raft-ha.md)) |
| `volumeClaimTemplates` `wal`, mounted at `/var/lib/durable` | the claim `wal-<system>-durable-0` survives Pod deletion and rescheduling; `[durable].wal_dir` points inside the mount |
| `securityContext.fsGroup: 10001` | a freshly provisioned volume belongs to root; `fsGroup` makes it group-writable for the server's gid |
| `runtime.toml` `wal_max_bytes` | the quota at which appends fail with `RESOURCE_EXHAUSTED` and nothing acknowledged is lost (the event log's guarantee, rehearsed in [the event log disk-full drill](../12-incident-response-and-chaos/12-eventlog-disk-full.md)) |
| Service `NodePort` 7233 to 30733, plus 9464 | workers in the cluster use `<system>-durable:7233`; a training worker on your Mac uses `127.0.0.1:30733` (the port the kind cluster maps) |

kind's local volumes ignore their size, so `walMaxBytes` is the real disk quota, and it must sit below the claim's size. One Helm detail bites here: Helm reads YAML numbers as float64, so `toToml` writes `wal_max_bytes = 1.073741824e+09`, which `runtime.schema.json` (an integer) rejects. You must cast every integer with `int64` in the template.

## The worker: drain time and a writable artifact root

The worker's problem is not state but stopping. Kubernetes stops a Pod by sending SIGTERM to PID 1 and, after `terminationGracePeriodSeconds` (default 30), SIGKILL. The worker on SIGTERM stops polling and lets in-flight activities finish for up to 30 s, and the subprocess runner gives a Python child 30 s between SIGTERM and SIGKILL to checkpoint. Both must fit, so the grace period is at least 60 s; the solution uses 75. The default 30 s kills training between checkpoints on every rollout.

The worker writes checkpoints, shards, and `DONE.json` under `/artifacts`, so it mounts that hostPath **read-write**, whereas the engine mounts it read-only. Its root filesystem stays read-only, with an `emptyDir` at `/tmp`.

## KEDA: scaling on queue depth

The last question is how many workers to run. KEDA is an operator that turns a `ScaledObject` into a HorizontalPodAutoscaler fed by an external metric. Here the metric is the durable server's gauge `tl_durable_task_queue_depth{queue, kind}`, scraped by Prometheus. Let $d$ be the pending activity tasks on the worker's queue (from Prometheus), $t$ the threshold of pending tasks one worker should absorb (4 in the solution), and $r_{\min} = 1$, $r_{\max} = 6$ the bounds. The desired replica count is

$$r = \min(\max(\lceil d/t \rceil, r_{\min}), r_{\max})$$

The `ScaledObject` owns the count. So the Deployment sets no `replicas` (a `helm upgrade` would otherwise reset it), and the chart renders no HPA of its own to compete. `cooldownPeriod` is long (120 s), because removing a worker mid-activity costs a 75 s drain and a redelivery, more than an idle Pod. KEDA itself is pinned in `deploy/keda/Chart.yaml` and `Chart.lock` to the version in `contracts/helm/observability.md`.

## Working the quota, a burst, and a rollout by hand

The solution's values are: claim `2Gi`, `walMaxBytes: 1073741824`, worker threshold 4, replicas 1 to 6, grace 75 s.

**The quota.** $2\,\text{Gi} = 2 \times 2^{30} = 2147483648$ bytes. The quota is $2^{30} = 1073741824$, half of it, which leaves room to raise the quota during an incident (as in the event log disk-full drill) without resizing the claim. `helm template` must render `wal_max_bytes = 1073741824` (an integer), and with `--set walMaxBytes=123456789`, `wal_max_bytes = 123456789`.

**A burst.** `CorpusBuild` enqueues 17 shard tasks on queue `data`. Prometheus scrapes $d = 17$; KEDA computes $\lceil 17/4 \rceil = 5$, within $[1, 6]$, so the worker Deployment goes from 1 to 5 replicas. As the queue drains to $d = 3$, $\lceil 3/4 \rceil = 1$; after the 120 s cooldown KEDA scales back to 1, and each removed Pod gets SIGTERM, drains, and exits within 75 s.

**A rollout.** Run `helm upgrade` of the worker during a training activity. The old Pod gets SIGTERM at $t = 0$ and stops polling; the runner SIGTERMs the Python child, which checkpoints at its next step (say $t = 8$ s) and exits 130. The activity is reported retryable (`WorkerShutdown`), the Pod exits at $t \approx 9$ s, far inside 75 s, and the new Pod resumes from the checkpoint.

These three numbers are what a correct pair of charts must get right; the burst itself is what the durable platform does on kind.

## The files and what they must do

| File | What it holds |
|---|---|
| `deploy/docker/durable.Dockerfile` | static build of `./cmd/durable` onto a digest-pinned alpine, `USER 10001`, ports 7233, 7234, 9464 |
| `deploy/docker/worker.Dockerfile` | static build of `./cmd/worker` copied onto a digest-pinned Python base with numpy, pyarrow, zstandard, and `python/` |
| `deploy/helm/<system>-durable/` | `values.schema.json` (the contract), StatefulSet, Service and headless Service, ConfigMap with `runtime.toml`, PDB |
| `deploy/helm/<system>-worker/` | `values.schema.json`, Deployment, ConfigMap, PDB, `ScaledObject` |
| `deploy/keda/` | `Chart.yaml` with the pinned `keda` dependency, `Chart.lock`, `values.yaml` |

### What a correct solution must handle

Statically, every file above must exist. The Dockerfiles must follow the image rules (digest pins, an exec-form `HEALTHCHECK`, labels), and the worker image must carry `python/` and build `./cmd/worker`. Both schemas must equal the contracts, and both charts must pass `helm lint`. Each unsafe value listed above must fail `helm template`. The rendered charts must satisfy the policy: probes, resources, non-root, pinned images, PDBs, and no literal secrets.

The durable chart must render a one-replica StatefulSet with a claim template mounted where `wal_dir` points, plus `fsGroup`. The WAL quota must render as an integer, follow `--set`, and stay below the claim size, exactly as worked above. The durable Service must expose NodePort 30733 for gRPC 7233, and port 9464. The worker's grace period must be at least 60 s; it must mount `/artifacts` read-write (not as an `emptyDir`) with a read-only root filesystem. The `ScaledObject` must have the right target, bounds, query, and threshold, with no `replicas` in the Deployment and no HPA of the chart's own. `deploy/keda`'s `Chart.yaml` and `Chart.lock` must match the pin.

With Docker, both images must build from the system's entry points, their `USER` must be numeric and not 0, and the worker image must be able to import numpy and `tinyllm.io.activity`. On the cluster, KEDA's CRD and release must be present on kind, both charts must pass `kubectl apply --dry-run=server`, and the releases must be ready: durable with a bound WAL claim, the worker available, and KEDA's HPA in existence.

## Building it step by step

### Reasoning from first principles

1. **State needs an identity that outlives the Pod.** The WAL must be found again by the replacement Pod. A StatefulSet gives a stable Pod name and a claim per Pod from `volumeClaimTemplates` that is never deleted with the Pod. A Deployment with a PVC would also keep the data, but its rollout would run an old and a new Pod at once against one WAL.
2. **One writer per WAL.** So `replicas: 1` (the schema enforces it until Raft), `podManagementPolicy: OrderedReady`, and a PDB of `maxUnavailable: 1` so a drain can still move it.
3. **The real quota is in the application.** kind's local-path volumes ignore their size, so the only enforced limit is the server's `wal_max_bytes`. It must be below the claim size (so the disk never fills first) and rendered as an integer (the schema says so, and Helm's numbers are floats).
4. **Stopping must fit inside the grace period.** A worker drains for up to 30 s, and a Python child checkpoints in up to 30 s after its own SIGTERM; the Pod's grace period must exceed their sum. 75 s gives 15 s of margin.
5. **Load is queue depth, not CPU.** Pending tasks are the demand; CPU measures only what is already running. So the scaler reads `tl_durable_task_queue_depth`, and replicas follow $\lceil d/t \rceil$.
6. **One controller per number.** If KEDA owns the replica count, the Deployment must not set one, or every `helm upgrade` resets it.
7. **The worker image must contain what its activities exec.** Activities run `{tinyllm}` and `{corpus}` as subprocesses, so the runtime base is Python with the pinned wheels, and the static Go worker is copied in.

### The images

The durable image is the gateway image with a different binary and one addition:

```dockerfile
COPY --from=build /out/durable /usr/local/bin/durable
# The WAL lives on the StatefulSet's volume at /var/lib/durable; the
# directory exists and belongs to the runtime user even without a volume.
RUN mkdir -p /var/lib/durable && chown 10001:10001 /var/lib/durable
USER 10001:10001
# 7233: tl.durable.v1 gRPC; 7234: Raft (dur.10); 9464: /healthz, /readyz, /metrics.
EXPOSE 7233 7234 9464
```

The worker image's runtime stage is where the two worlds meet:

```dockerfile
# The runtime is Python (glibc, for numpy and pyarrow wheels); the Go worker
# is a static binary copied in.
FROM python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258
ARG VERSION=0.0.0-dev
ARG REVISION=unknown
# ... OCI labels as in dep.01
# Python dependencies first (they change least), pinned; no pip cache in the layer.
RUN pip install --no-cache-dir "numpy==2.3.3" "pyarrow==21.0.0" "zstandard==0.25.0"
WORKDIR /app
# The units the activities exec ({tinyllm}, {corpus}).
COPY python/ /app/python/
COPY --from=build /out/worker /usr/local/bin/worker
USER 10001:10001
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
EXPOSE 9464
# No wget or curl in python-slim, but there is Python.
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:9464/healthz', timeout=2).status == 200 else 1)"]
ENTRYPOINT ["/usr/local/bin/worker"]
```

`PYTHONDONTWRITEBYTECODE=1` matters because the root filesystem will be read-only. `PYTHONUNBUFFERED=1` makes child output reach the logs line by line, so a crash does not swallow its last message. glibc (Debian) rather than alpine (musl) is what lets numpy and pyarrow install from wheels instead of compiling.

The full solution is [`durable.Dockerfile`](../../../../course/solutions/entry/deploy/docker/durable.Dockerfile) and [`worker.Dockerfile`](../../../../course/solutions/entry/deploy/docker/worker.Dockerfile).

### The durable StatefulSet

```yaml
kind: StatefulSet
metadata:
  name: {{ .Release.Name }}
spec:
  serviceName: {{ .Release.Name }}-headless     # stable DNS for each pod (Raft, later)
  replicas: {{ .Values.replicaCount }}          # the schema allows 1 until dur.10
  # The WAL has one writer: never run the old and the new pod at once.
  podManagementPolicy: OrderedReady
  template:
    metadata:
      annotations:
        checksum/runtime: {{ include (print $.Template.BasePath "/configmap.yaml") . | sha256sum }}
    spec:
      # SIGTERM: stop accepting, fsync, exit; the WAL's torn tail is
      # truncated on the next start anyway (dur.01).
      terminationGracePeriodSeconds: 30
      securityContext:
        runAsNonRoot: {{ .Values.securityContext.runAsNonRoot }}
        runAsUser: {{ .Values.securityContext.runAsUser }}
        {{- with .Values.securityContext.runAsGroup }}
        runAsGroup: {{ . }}
        # The volume is mounted group-writable for this gid, so a non-root
        # server can write its WAL.
        fsGroup: {{ . }}
        {{- end }}
      containers:
        - name: durable
          args: ["--config", "/etc/tinyllm/runtime.toml"]
          # ... OTel env, ports grpc/raft/health, probes (/readyz: WAL replayed)
          volumeMounts:
            - name: wal
              mountPath: /var/lib/durable
            - name: runtime
              mountPath: /etc/tinyllm
              readOnly: true
  volumeClaimTemplates:          # one claim per pod: wal-forge-durable-0
    - metadata:
        name: wal
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: {{ .Values.persistence.size }}   # 2Gi
```

The readiness probe asks `/readyz`, which answers only once the WAL is replayed and gRPC is listening, so workers are never routed to a server still recovering. With one replica, a rolling update is necessarily stop-then-start: the claim is `ReadWriteOnce`, and a second Pod could not mount it anyway.

### The runtime config and the integer cast

```yaml
{{- /* runtime.toml: Helm reads YAML numbers as floats, and toToml would
       write 5.0 where runtime.schema.json wants an integer, so every
       integer key is cast with int64. */ -}}
{{- define "durable.runtime" -}}
{{- $r := default dict (.Values.runtime).durable -}}
{{- $d := dict "grpc_listen" (printf ":%d" (int .Values.service.grpcPort)) "health_listen" (printf ":%d" (int .Values.service.healthPort)) "wal_dir" "/var/lib/durable/wal" "wal_max_bytes" (int64 .Values.walMaxBytes) -}}
{{- range $k := list "visibility_timeout_ms" "dlq_after_attempts" "long_poll_ms" -}}
{{- if hasKey $r $k }}{{ $_ := set $d $k (int64 (get $r $k)) }}{{ end -}}
{{- end -}}
{{- toToml (dict "durable" $d "otel" (dict "endpoint" $.Values.otel.endpoint)) -}}
{{- end -}}
```

The keys the chart must control (`grpc_listen`, `wal_dir` inside the mount, `wal_max_bytes`) are always built from values. Optional integer keys are copied from `runtime.durable` only if present, each through `int64`. `wal_dir` is a subdirectory of the mount, so the mount point itself (which may contain `lost+found` on some volumes) is never the WAL directory. With the default `walMaxBytes: 1073741824`, the ConfigMap holds `wal_max_bytes = 1073741824`, as computed above.

The Service is a NodePort for gRPC (7233 to 30733) plus the health port, and a second, headless Service (`clusterIP: None`) is the StatefulSet's `serviceName`.

The full solution is [`statefulset.yaml`](../../../../course/solutions/entry/deploy/helm/forge-durable/templates/statefulset.yaml), [`_helpers.tpl`](../../../../course/solutions/entry/deploy/helm/forge-durable/templates/_helpers.tpl), [`service.yaml`](../../../../course/solutions/entry/deploy/helm/forge-durable/templates/service.yaml), and [`values.yaml`](../../../../course/solutions/entry/deploy/helm/forge-durable/values.yaml).

### The worker Deployment

```yaml
spec:
  {{- if not .Values.autoscaling.enabled }}
  replicas: {{ .Values.replicaCount }}       # only when KEDA does not own the count
  {{- end }}
  template:
    spec:
      terminationGracePeriodSeconds: {{ .Values.terminationGracePeriodSeconds | default 75 }}
      containers:
        - name: worker
          args: ["--config", "/etc/tinyllm/runtime.toml"]
          env:
            - name: TL_ARTIFACTS
              value: {{ .Values.artifacts.mountPath | quote }}
            # ... TL_CORPUS_ENTRY, OTel env
            # Python writes nothing next to its sources: the root filesystem is read-only.
            - name: PYTHONDONTWRITEBYTECODE
              value: "1"
            - name: HOME
              value: /tmp                    # libraries that cache under ~ write to the emptyDir
          volumeMounts:
            - name: artifacts
              mountPath: {{ .Values.artifacts.mountPath }}   # no readOnly: the worker writes outputs
            - name: tmp
              mountPath: /tmp
          securityContext:
            readOnlyRootFilesystem: {{ .Values.securityContext.readOnlyRootFilesystem | default true }}
      volumes:
        - name: artifacts
          hostPath:
            path: {{ .Values.artifacts.hostPath }}
            type: Directory
        - name: tmp
          emptyDir: {}
```

```yaml
# values.yaml (excerpt)
# A worker stops polling on SIGTERM and lets in-flight activities finish
# (dur.04 drain, 30 s), and a Python child gets 30 s between SIGTERM and
# SIGKILL to checkpoint (dur.09): the pod needs both before Kubernetes kills it.
terminationGracePeriodSeconds: 75
```

The writable surface is exactly two paths: `/artifacts` (outputs that must survive) and `/tmp` (scratch that must not). Everything else is read-only, so a bug that writes elsewhere fails loudly instead of filling the node.

The full solution is [`forge-worker/templates/deployment.yaml`](../../../../course/solutions/entry/deploy/helm/forge-worker/templates/deployment.yaml) and [`values.yaml`](../../../../course/solutions/entry/deploy/helm/forge-worker/values.yaml).

### The ScaledObject and KEDA's pin

```yaml
{{- if .Values.autoscaling.enabled }}
# KEDA owns the replica count: it creates and drives an HPA for this
# Deployment from the Prometheus query below.
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: {{ .Release.Name }}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {{ .Release.Name }}
  minReplicaCount: {{ .Values.autoscaling.minReplicas | default 1 }}   # r_min
  maxReplicaCount: {{ .Values.autoscaling.maxReplicas | default 6 }}   # r_max
  pollingInterval: 15
  # Scale down slowly: a worker removed mid-activity is drained (75 s) and
  # its task redelivered, which costs more than an idle pod.
  cooldownPeriod: 120
  triggers:
    - type: prometheus
      metadata:
        serverAddress: {{ .Values.autoscaling.prometheusAddress | quote }}
        # d: pending activity tasks on this worker's queue
        query: sum(tl_durable_task_queue_depth{queue={{ .Values.autoscaling.queue | default "data" | quote }},kind="activity"})
        threshold: {{ .Values.autoscaling.targetDepth | default 4 | quote }}            # t
{{- end }}
```

`sum(...)` collapses the gauge to one number even if several series match. The threshold is quoted because KEDA's trigger metadata is a map of strings. `kind="activity"` excludes workflow tasks, which are short and do not need more workers.

KEDA itself is an umbrella chart, `deploy/keda`, with one dependency pinned to `keda` 2.21.0 (from `contracts/helm/observability.md`) and a `Chart.lock`. It must be installed before the first `ScaledObject`, whose CRD it provides. In [the Tilt dev loop](05-tilt-dev-loop.md), the worker is installed with `autoscaling.enabled=false` and a fixed count.

The full solution is [`scaledobject.yaml`](../../../../course/solutions/entry/deploy/helm/forge-worker/templates/scaledobject.yaml) and [`keda/Chart.yaml`](../../../../course/solutions/entry/deploy/keda/Chart.yaml).

### Why this design

- **StatefulSet over a Deployment with a PVC.** Both keep the data; only the StatefulSet guarantees one Pod per identity and gives the stable names Raft will need. The cost is slower, ordered rollouts, which a single writer requires anyway.
- **An application quota over relying on volume size.** On kind the volume size is advisory; even on a real CSI driver, failing appends with `RESOURCE_EXHAUSTED` at a known quota is safer than discovering `ENOSPC` mid-write.
- **KEDA on queue depth over an HPA on CPU.** CPU lags demand and is near zero for a worker waiting on I/O; queue depth is the demand itself. The cost is a dependency on Prometheus being up: if it is not, KEDA holds the last count.
- **A long grace period over a preStop hook.** A preStop sleep would delay SIGTERM, not make the drain happen; the worker already drains on SIGTERM, so it needs time, not a hook.
- **Tradeoff accepted:** a single durable replica is a single point of failure; the WAL makes it crash-safe, not highly available. [Raft HA for the durable log](../07-durable-orchestration/10-raft-ha.md) lifts the `replicaCount` limit.

## What can go wrong

Stateful and queue-driven workloads fail in more ways than the serving charts did. Ten mistakes account for most of them:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| the server as a Deployment with an `emptyDir` | every workflow vanishes on the first reschedule | a StatefulSet with a `wal` claim template |
| no `fsGroup` | the non-root server cannot open its WAL on a fresh volume: CrashLoopBackOff | `fsGroup` from `runAsGroup` |
| `toToml` of a float quota | the server rejects `wal_max_bytes = 1.073741824e+09` at start | `int64` on every integer key in `durable.runtime` |
| quota above the volume size | the disk fills before `RESOURCE_EXHAUSTED` trips; the WAL tears | `walMaxBytes` 1 GiB on a 2 GiB claim |
| the default 30 s grace on workers | every rollout and scale-down kills training between checkpoints | `terminationGracePeriodSeconds: 75` |
| `/artifacts` read-only in the worker | every subprocess activity fails writing its first output | the worker mounts `/artifacts` without `readOnly` |
| `replicas` set beside a ScaledObject, or a second HPA | each `helm upgrade` resets the count; two controllers flap | `replicas` rendered only when autoscaling is off; KEDA makes the only HPA |
| scaling on CPU | a 200-task burst waits on two idle-looking workers | a Prometheus trigger on `tl_durable_task_queue_depth` |
| `--test-clock` or `:latest` in a chart | a test seam in production; kind runs whatever it cached | the contract schemas reject `testClock: true` and `latest` |
| a worker image without the Python units | every activity exits 1 with an ImportError, only on kind | the worker image is Python-based with pinned wheels and `COPY python/` |

> **Warning:** The integer-cast bug is especially cruel because `helm install` succeeds. The failure only appears when the server validates its config at start-up and crashloops.

## Where this leads

These charts lean on [Helm charts and the observability stack](04-helm-charts-and-observability-stack.md) for the chart policy, the observability stack, and the Prometheus that KEDA queries, and on [the subprocess activity runner](../07-durable-orchestration/09-subprocess-activities.md) for the units the worker image carries. Going forward, [Observability for the control plane](../11-observability/06-control-plane-observability.md) in this book reads the durable server's queue and DLQ gauges and the worker traces. The durable platform installs these charts on kind and runs its kill loop against them. Three drills target these releases: [the durable server SIGKILLed in a loop](../12-incident-response-and-chaos/03-durable-kill9.md), [a poison task in the dead-letter queue](../12-incident-response-and-chaos/04-poison-task.md), and [event log disk full](../12-incident-response-and-chaos/12-eventlog-disk-full.md). Finally, [The agent image and chart](08-agent-image-and-chart.md) extends the worker chart for agent workers.

The production versions of these pieces:

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| one-replica StatefulSet on local-path | Temporal on Cassandra or PostgreSQL with history shards | stateless frontends, shard ownership over a replicated store | Temporal docs, "Temporal Platform: persistence" |
| `walMaxBytes` as quota | a CSI driver that enforces capacity, volume expansion | real per-volume limits and online growth | Kubernetes "Volume expansion" |
| KEDA on queue depth | Temporal worker autoscaling on schedule-to-start latency | scales on how long tasks wait, not how many | Temporal docs, "Worker performance" |
| hostPath `/artifacts` | object storage (S3, GCS) or a ReadWriteMany volume | many nodes, durable outputs, no node affinity | CSI drivers for S3, `ReadWriteMany` access modes |

### Further reading

- [Kubernetes: StatefulSets](https://kubernetes.io/docs/concepts/workloads/controllers/statefulset/)
- [KEDA: Prometheus scaler](https://keda.sh/docs/2.21/scalers/prometheus/)
- [Kubernetes: Pod termination](https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/#pod-termination)

## Key terms

- **StatefulSet**: a workload controller that gives each Pod a stable name and its own persistent claim, with ordered rollout.
- **`volumeClaimTemplates`**: the StatefulSet field that creates one PersistentVolumeClaim per Pod, kept when the Pod is deleted.
- **WAL quota (`wal_max_bytes`)**: the application-enforced size at which the durable server refuses appends with `RESOURCE_EXHAUSTED`.
- **`terminationGracePeriodSeconds`**: how long Kubernetes waits between SIGTERM and SIGKILL when stopping a Pod.
- **KEDA**: an operator that scales workloads on external metrics by creating and driving an HPA.
- **`ScaledObject`**: the KEDA resource that names a scale target, bounds, and triggers.
- **cooldown period**: how long KEDA waits after demand falls before scaling down.

## Summary

- The durable server's state is its WAL: it runs as a one-replica StatefulSet whose `volumeClaimTemplates` gives the Pod a claim that outlives it, and `fsGroup` lets the non-root server write the fresh volume.
- kind's local volumes ignore their size, so `walMaxBytes`, rendered as an integer `[durable].wal_max_bytes`, is the real disk quota and sits below the claim size.
- A worker gets a grace period long enough to drain (30 s) and let its Python child checkpoint (30 s more); the solution uses 75 s, because the default 30 s kills training between checkpoints.
- KEDA scales workers on queue depth, not CPU, through a `ScaledObject` that owns the replica count: the Deployment sets none and no second HPA competes.
- The worker image is two worlds: a static Go binary copied into a Python base that can import the units its activities exec.

## Review questions

1. Why is `fsGroup` needed for the durable server but not for the engine?
   <details><summary>Answer</summary>The durable server writes to a freshly provisioned volume, which belongs to root; <code>fsGroup</code> makes it group-writable for gid 10001. The engine only reads <code>/artifacts</code> and writes nothing.</details>
2. A teammate removes the `int64` casts. What fails, and where?
   <details><summary>Answer</summary>Helm parses <code>1073741824</code> as a float64, so <code>toToml</code> writes <code>wal_max_bytes = 1.073741824e+09</code>. The server validates its config against <code>runtime.schema.json</code>, which requires an integer, and refuses to start: CrashLoopBackOff after a successful install.</details>
3. With $t = 4$, $r_{\min} = 1$, $r_{\max} = 6$, the queue holds 30 tasks. How many workers run, and what happens to the backlog?
   <details><summary>Answer</summary>$\lceil 30/4 \rceil = 8$, clamped to 6. Six workers run and the queue drains more slowly than one worker per four tasks; the clamp is a cost ceiling, chosen deliberately.</details>
4. Why does the Deployment omit `replicas` when autoscaling is enabled?
   <details><summary>Answer</summary>KEDA's HPA writes the replica count. If the chart also set it, every <code>helm upgrade</code> would reset the count to the chart's value, fighting the autoscaler and causing a burst of scale-downs and scale-ups.</details>
5. Why is the worker's grace period 75 s and not 60 s?
   <details><summary>Answer</summary>60 s is exactly the sum of the two inner budgets (a 30 s drain and a 30 s checkpoint window). Any delay between them, such as process start-up, signal delivery, or the final report to the server, would push past it. 75 s leaves margin so SIGKILL never arrives while a checkpoint is being written.</details>
