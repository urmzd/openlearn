<!-- ol:module dep.06 -->
# Durable and worker images and charts: WAL PVC, StatefulSet, KEDA autoscaling of workers

## Overview

| | |
|---|---|
| **Module** | `dep.06` · practice · ops · Pass 8 · 4 to 6 h |
| **You build** | `deploy/docker/durable.Dockerfile` and `deploy/docker/worker.Dockerfile`; the charts `deploy/helm/<system>-durable` (a StatefulSet whose WAL is on a PersistentVolumeClaim, with the WAL quota rendered into `runtime.toml`) and `deploy/helm/<system>-worker` (a Deployment that drains before it is killed, writes `/artifacts`, and is scaled by a KEDA `ScaledObject`); `deploy/keda` pinning KEDA |
| **Contract** | [`course/contracts/helm/durable.values.schema.json`](../../course/contracts/helm/durable.values.schema.json), [`course/contracts/helm/worker.values.schema.json`](../../course/contracts/helm/worker.values.schema.json), the KEDA pin in [`course/contracts/helm/observability.md`](../../course/contracts/helm/observability.md), the gauge `tl.durable.task_queue.depth` in [`course/contracts/otel/metrics.yaml`](../../course/contracts/otel/metrics.yaml), and `[durable]`/`[worker]` of [`course/contracts/config/runtime.schema.json`](../../course/contracts/config/runtime.schema.json) |
| **Tests** | `course/tests/dep.06/` (`check` runs `artifacts.py`: static, docker, and cluster tiers; section 4) |
| **Needs** | `dep.03` (the chart policy and the stack these charts join, with the images of `dep.01` and the cluster of `dep.02`), `dur.09` (the Python units the worker image carries); your durable server (`dur.02`) and worker (`dur.04`) entry points, which the images build |
| **Used by** | `dep.07` extends the worker chart for agent workers; MS-durable deploys these charts on kind, `obs.05` traces and scrapes them, and drills `durable-kill9`, `poison-task`, and `eventlog-disk-full` run against them |
| **Milestone** | MS-durable (its kind steps install these charts) |
| **Optional depth** | [Kubernetes: StatefulSets](https://kubernetes.io/docs/concepts/workloads/controllers/statefulset/) (free); [KEDA: Prometheus scaler](https://keda.sh/docs/2.21/scalers/prometheus/) (free); [Kubernetes: pod termination](https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/#pod-termination) (free) |

## Key Takeaways

- The durable server's state is its **WAL**. It runs as a one-replica **StatefulSet** whose `volumeClaimTemplates` gives the pod a PersistentVolumeClaim that outlives it; an `fsGroup` lets the non-root server write the fresh volume.
- kind's local volumes ignore their size, so `walMaxBytes` (rendered as an **integer** `[durable].wal_max_bytes`) is the real disk quota, and it sits below the claim's size.
- A worker gets `terminationGracePeriodSeconds` long enough to **drain** (30 s) and let its Python child checkpoint (30 s more); the default 30 s kills training between checkpoints on every rollout.
- **KEDA** scales workers on **queue depth**, not CPU, through a `ScaledObject` that owns the replica count: the Deployment sets none, and no second HPA competes.
- The worker image is two worlds: a static Go binary copied into a **Python** base that can import the units its activities exec.

## How to work this chapter

```bash
ol start dep.06                       # records the start; there are no stubs
ol tests dep.06                       # read the test catalog first
# write the Dockerfiles, both charts, and deploy/keda (section 4), then:
helm dependency update deploy/keda    # writes deploy/keda/Chart.lock
OL_SMOKE=1 ol check dep.06            # static and docker tiers, no cluster
helm upgrade --install keda deploy/keda -n keda --create-namespace
helm upgrade --install forge-durable deploy/helm/forge-durable -n forge
helm upgrade --install forge-worker deploy/helm/forge-worker -n forge
ol check dep.06                       # every tier, against kind-<system>
```

---

## 1. Why now

Pass 8 gave you a durable server and workers that survive kills on your laptop. On kind they need three things your Pass 7 charts never had to think about. The server **has state**: a Deployment pod writes its WAL to the container filesystem, and the first reschedule erases every workflow. The worker's **load is a queue**, not requests: a corpus build enqueues 200 shard tasks in a second, the CPU of two idle workers says nothing about it, and a fixed replica count either wastes the cluster or leaves the queue growing. And the worker's work is **long**: a training step runs for minutes inside a Python child, so the way Kubernetes stops a pod (SIGTERM, then SIGKILL 30 s later) decides whether a rollout loses a checkpoint. This module packages both components so they keep the promises of `dur.01` to `dur.09` when Kubernetes, not you, decides when pods start and stop.

## 2. Principles

### 2.1 Two images, one rule set

Both Dockerfiles keep the `dep.01` rules: multi-stage, every base pinned by **digest**, a numeric non-root `USER`, an exec-form `ENTRYPOINT` (the process is PID 1 and receives SIGTERM, which the worker needs to drain), a `HEALTHCHECK` in exec form on the health port 9464, and the OCI labels. The durable image is a static Go binary on alpine, like the gateway. The worker image is different: its runtime stage is a **Python** base (glibc, so numpy and pyarrow install as wheels) that also carries the static Go worker and your `python/` tree, because the worker execs `{tinyllm} train` and `{corpus} run --stage` as subprocess activities (`dur.09`).

### 2.2 The values contract and the policy

Each chart carries its `values.schema.json` copied unchanged from `contracts/helm/`, so Helm refuses unsafe values before an install: `image.tag=latest`, `testClock=true` (a test-only clock in a cluster), `persistence.enabled=false`, a missing or zero `walMaxBytes`, two durable replicas without Raft, a worker with no task queue. The `dep.03` policy applies to what `helm template` renders: probes, CPU and memory requests and limits, a numeric non-root user without privilege escalation, a pinned image, a PodDisruptionBudget for every workload, and secrets only by reference.

### 2.3 The durable server: StatefulSet, WAL claim, quota

| Object | Why |
|---|---|
| `StatefulSet`, `replicas: 1`, `podManagementPolicy: OrderedReady` | one writer per WAL; a stable pod name (`<system>-durable-0`) that Raft peers can address later (`dur.10`) |
| `volumeClaimTemplates` `wal`, mounted at `/var/lib/durable` | the claim `wal-<system>-durable-0` survives pod deletion and rescheduling; `[durable].wal_dir` points inside the mount |
| `securityContext.fsGroup: 10001` | a freshly provisioned volume belongs to root; `fsGroup` makes it group-writable for the server's gid |
| `runtime.toml` `wal_max_bytes` | the quota at which appends fail with `RESOURCE_EXHAUSTED` and nothing acknowledged is lost (`dur.01`, drill `eventlog-disk-full`) |
| Service `NodePort` 7233 to 30733, plus 9464 | workers in the cluster use `<system>-durable:7233`; a training worker on your Mac uses `127.0.0.1:30733` (the port `dep.02` maps) |

One Helm detail bites here: Helm reads YAML numbers as float64, so `toToml` writes `wal_max_bytes = 1.073741824e+09`, which `runtime.schema.json` (an integer) rejects. Cast every integer with `int64` in the template.

### 2.4 The worker: drain time and a writable artifact root

Kubernetes stops a pod by sending SIGTERM to PID 1 and, after `terminationGracePeriodSeconds` (default 30), SIGKILL. Your worker on SIGTERM stops polling and lets in-flight activities finish for up to 30 s (`dur.04`), and the subprocess runner gives a Python child 30 s between SIGTERM and SIGKILL to checkpoint (`dur.09`). Both must fit: the grace period is at least 60 s (the reference uses 75). The worker writes checkpoints, shards, and `DONE.json` under `/artifacts`, so it mounts that hostPath **read-write** (the engine mounts it read-only), while its root filesystem stays read-only with an `emptyDir` at `/tmp`.

### 2.5 KEDA: scaling on queue depth

KEDA is an operator that turns a `ScaledObject` into a HorizontalPodAutoscaler fed by an external metric. Here the metric is the durable server's gauge `tl_durable_task_queue_depth{queue, kind}`, scraped by Prometheus, and the rule is: replicas $= \lceil \text{depth} / \text{threshold} \rceil$, clamped to `[minReplicaCount, maxReplicaCount]`.

| Symbol | Meaning | Reference value |
|---|---|---|
| $d$ | pending activity tasks on the worker's queue | from Prometheus |
| $t$ | threshold: pending tasks one worker should absorb | 4 |
| $r = \min(\max(\lceil d/t \rceil, r_{\min}), r_{\max})$ | desired replicas | $r_{\min}$ = 1, $r_{\max}$ = 6 |

The `ScaledObject` owns the count, so the Deployment sets no `replicas` (a `helm upgrade` would otherwise reset it) and the chart renders no HPA of its own. `cooldownPeriod` is long (120 s): removing a worker mid-activity costs a 75 s drain and a redelivery, more than an idle pod. KEDA itself is pinned in `deploy/keda/Chart.yaml` and `Chart.lock` to the version in `contracts/helm/observability.md`.

## 3. Worked example by hand

The reference values: claim `2Gi`, `walMaxBytes: 1073741824`, worker threshold 4, replicas 1 to 6, grace 75 s.

**The quota.** $2\,\text{Gi} = 2 \times 2^{30} = 2147483648$ bytes; the quota is $2^{30} = 1073741824$, half of it, which leaves room to raise the quota during an incident (drill `eventlog-disk-full`) without resizing the claim. `helm template` must render `wal_max_bytes = 1073741824` (an integer), and with `--set walMaxBytes=123456789`, `wal_max_bytes = 123456789`.

**A burst.** `CorpusBuild` enqueues 17 shard tasks on queue `data`. Prometheus scrapes $d = 17$; KEDA computes $\lceil 17/4 \rceil = 5$, within $[1, 6]$, so the worker Deployment goes from 1 to 5 replicas. As the queue drains to $d = 3$, $\lceil 3/4 \rceil = 1$; after the 120 s cooldown KEDA scales back to 1, and each removed pod gets SIGTERM, drains, and exits within 75 s.

**A rollout.** `helm upgrade` of the worker during a training activity: the old pod gets SIGTERM at $t = 0$, stops polling, the runner SIGTERMs the Python child, which checkpoints at its next step (say $t = 8$ s) and exits 130; the activity is reported retryable (`WorkerShutdown`), the pod exits at $t \approx 9$ s, far inside 75 s, and the new pod resumes from the checkpoint.

These are `test_wal_quota_is_rendered`, `test_keda_scales_on_queue_depth`, and `test_worker_drains_before_it_is_killed`; the burst itself runs on kind in MS-durable.

## 4. The artifact and its check

| File | What it holds |
|---|---|
| `deploy/docker/durable.Dockerfile` | static build of `./cmd/durable` onto a digest-pinned alpine, `USER 10001`, ports 7233, 7234, 9464 |
| `deploy/docker/worker.Dockerfile` | static build of `./cmd/worker` copied onto a digest-pinned Python base with numpy, pyarrow, zstandard, and `python/` |
| `deploy/helm/<system>-durable/` | `values.schema.json` (the contract), StatefulSet, Service and headless Service, ConfigMap with `runtime.toml`, PDB |
| `deploy/helm/<system>-worker/` | `values.schema.json`, Deployment, ConfigMap, PDB, `ScaledObject` |
| `deploy/keda/` | `Chart.yaml` with the pinned `keda` dependency, `Chart.lock`, `values.yaml` |

### What the tests check

| Test | KIND | Checks |
|---|---|---|
| `test_layout_present` | unit | every file above exists |
| `test_dockerfiles_follow_the_image_rules` | unit | dep.01 rules, digest pins, exec-form HEALTHCHECK, labels; the worker image carries `python/` and builds `./cmd/worker` |
| `test_values_schemas_are_the_contract` | conformance | both schemas equal the contracts |
| `test_charts_lint` | conformance | `helm lint` of both charts |
| `test_schema_rejects_unsafe_values` | boundary | eight unsafe values each fail `helm template` |
| `test_policy_over_rendered_charts` | unit | probes, resources, non-root, pinned images, PDBs, no literal secrets |
| `test_durable_is_a_statefulset_with_a_wal_volume` | unit | one-replica StatefulSet, a claim template mounted where `wal_dir` points, `fsGroup` |
| `test_wal_quota_is_rendered` | unit | section 3's quota as an integer, following `--set`, below the claim size |
| `test_durable_service_ports` | unit | NodePort 30733 for gRPC 7233, port 9464 |
| `test_worker_drains_before_it_is_killed` | boundary | grace period at least 60 s |
| `test_worker_writes_artifacts` | unit | `/artifacts` read-write, not an emptyDir; read-only root filesystem |
| `test_keda_scales_on_queue_depth` | unit | the ScaledObject's target, bounds, query, and threshold; no `replicas`, no HPA |
| `test_keda_is_pinned` | conformance | `deploy/keda` Chart.yaml and Chart.lock match the pin |
| `test_images_build` | unit | both images build from your entry points |
| `test_images_run_as_numeric_non_root` | unit | image `USER` numeric and not 0 |
| `test_worker_image_has_the_python_units` | unit | the worker image imports numpy and `tinyllm.io.activity` |
| `test_keda_installed` | conformance | KEDA's CRD and release on kind |
| `test_server_side_dry_run` | conformance | both charts pass `kubectl apply --dry-run=server` |
| `test_releases_ready` | conformance | durable ready with a bound WAL claim, worker available, KEDA's HPA exists |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. the server as a Deployment with an emptyDir | every workflow vanishes on the first reschedule | `test_durable_is_a_statefulset_with_a_wal_volume` |
| 2. no `fsGroup` | the non-root server cannot open its WAL on a fresh volume: CrashLoopBackOff | `test_durable_is_a_statefulset_with_a_wal_volume` |
| 3. `toToml` of a float quota | the server rejects `wal_max_bytes = 1.073741824e+09` at start | `test_wal_quota_is_rendered` |
| 4. quota above the volume size | the disk fills before `RESOURCE_EXHAUSTED` trips; the WAL tears | `test_wal_quota_is_rendered` |
| 5. the default 30 s grace on workers | every rollout and scale-down kills training between checkpoints | `test_worker_drains_before_it_is_killed` |
| 6. `/artifacts` read-only in the worker | every subprocess activity fails writing its first output | `test_worker_writes_artifacts` |
| 7. `replicas` set beside a ScaledObject, or a second HPA | each `helm upgrade` resets the count; two controllers flap | `test_keda_scales_on_queue_depth` |
| 8. scaling on CPU | a 200-task burst waits on two idle-looking workers | `test_keda_scales_on_queue_depth` |
| 9. `--test-clock` or `:latest` in a chart | a test seam in production; kind runs whatever it cached | `test_schema_rejects_unsafe_values` |
| 10. a worker image without the Python units | every activity exits 1 with an ImportError, only on kind | `test_worker_image_has_the_python_units` |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `dep.03` | the chart policy, the observability stack, and the Prometheus that KEDA queries |
| Back | `dur.09` | the worker image carries the units the subprocess runner execs |
| Forward | `obs.05` | control-plane dashboards read the durable server's queue and DLQ gauges and the worker traces |
| Forward | MS-durable | the kind steps install these charts and run the kill loop against them |
| Forward | `ops.02`, `ops.03`, `ops.11` | drills `durable-kill9`, `poison-task`, `eventlog-disk-full` target these releases |
| Forward | `dep.07` | the agent chart follows the worker chart |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| one-replica StatefulSet on local-path | Temporal on Cassandra or PostgreSQL with history shards | stateless frontends, shard ownership over a replicated store | Temporal docs, "Temporal Platform: persistence" |
| `walMaxBytes` as quota | a CSI driver that enforces capacity, volume expansion | real per-volume limits and online growth | Kubernetes "Volume expansion" |
| KEDA on queue depth | Temporal worker autoscaling on schedule-to-start latency | scales on how long tasks wait, not how many | Temporal docs, "Worker performance" |
| hostPath `/artifacts` | object storage (S3, GCS) or a ReadWriteMany volume | many nodes, durable outputs, no node affinity | CSI drivers for S3, `ReadWriteMany` access modes |
