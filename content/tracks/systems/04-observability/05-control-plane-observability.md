<!-- ol:module obs.05 -->
# Observability for the control plane: workflow and activity spans, queue and DLQ panels

## Overview

| | |
|---|---|
| **Module** | `obs.05` · practice · ops · Pass 8 · 3 to 4 h |
| **You build** | `deploy/observability/dashboards/control-plane.json` (queue depth, dead letters, redeliveries, schedule-to-start, WAL size, training gauges), `deploy/observability/control-plane-monitors.yaml` (scrapes for the durable server, the workers, and the collector's Prometheus exporter), and the worker chart's export settings (`dep.06`): Go over OTLP/gRPC, Python over OTLP/HTTP |
| **Contract** | [`otel/semconv.md`](../../../../course/contracts/otel/semconv.md) (the span tree from `{ctl} train` to `train.step`), [`otel/metrics.yaml`](../../../../course/contracts/otel/metrics.yaml) (`tl.durable.*`, `tl.train.*`), [`helm/observability.md`](../../../../course/contracts/helm/observability.md) (the monitor label, the Python metrics pipeline) |
| **Tests** | `course/tests/obs.05/` (`check` runs `artifacts.py`: static, python, and cluster tiers; section 4) |
| **Needs** | `dep.06` (the charts these panels and monitors watch), `dur.09` (`telemetry.py`, the Python end of the trace), `obs.04` (dashboards as code; its promtool and PromQL helpers are reused); reading: `obs.01` (`go/otelx`, the Go spans), `obs.02` (scraping) |
| **Used by** | no call site (a practice): drills `durable-kill9` and `poison-task` (`ops.02`, `ops.03`) are detected on this dashboard |
| **Milestone** | MS-durable (one trace from `{ctl} train` to `train.step` on kind) |
| **Optional depth** | [Temporal: SDK metrics and schedule-to-start latency](https://docs.temporal.io/references/sdk-metrics) (free); [OTLP specification](https://opentelemetry.io/docs/specs/otlp/) (free); [Prometheus: histograms and summaries](https://prometheus.io/docs/practices/histograms/) (free) |

## Key Takeaways

- One **TrainRun is one trace**: `{ctl} train` starts it, the durable server stores the context in the workflow's first event, every activity task carries it, the worker hands it to Python as `TRACEPARENT`, and `train.step` spans hang under it every 50th step.
- A queue-based system answers three questions on one screen: is work **piling up** (depth per queue), is work being **given up on** (dead letters), are workers **dying** (redeliveries, as a rate).
- **Schedule-to-start** latency is the queue's user-facing latency; its p95 is `histogram_quantile` over bucket **rates** grouped **by `le`**.
- Python cannot be scraped: it **pushes** metrics over OTLP to the collector, whose Prometheus exporter is scraped instead. The Go worker speaks OTLP/**gRPC** on 4317, Python OTLP/**HTTP** on 4318.
- Monitors carry `release: observability`, or Prometheus never sees them.

## How to work this chapter

```bash
ol start obs.05                       # records the start; there are no stubs
ol tests obs.05                       # read the test catalog first
# write the dashboard, the monitors, and the worker chart's TL_PYTHON_OTLP_ENDPOINT (section 4)
OL_SMOKE=1 ol check obs.05            # static and python tiers
kubectl apply -f deploy/observability/control-plane-monitors.yaml
<system> train --spec specs/tiny.json # on kind: makes the trace the cluster tier looks for
ol check obs.05
```

---

## 1. Why now

Your serving path is visible end to end (`obs.01` to `obs.04`), but the control plane you built in Pass 8 is a black box: when a corpus build stalls, nothing says whether the queue is growing, a task is stuck in the dead-letter queue, or workers keep dying and redelivering. The training run is worse: the interesting work happens in a Python child process two hops from the CLI, and without a propagated context its spans either do not exist or start a trace of their own that nobody can connect to the `TrainRun` that caused them. This module makes the control plane observable: the dashboard the drills of Pass 8 are detected on, and one trace from the command you type to the training step.

## 2. Principles

### 2.1 One trace across four processes

| Process | Span | Gets its parent from |
|---|---|---|
| `{ctl} train` | the CLI span | a new root |
| durable server | stores the context in `WorkflowExecutionStarted.trace_context` | the `StartWorkflow` request's metadata |
| worker (workflow task) | `workflow TrainRun` | the stored context, on every replay (`tl.workflow.replay = true` when replaying) |
| worker (activity task) | `activity train` | `ActivityTask.trace_context` |
| Python child | `train.run`, then `train.step` every 50th step, `train.checkpoint` | `TRACEPARENT`, set by the subprocess runner (`dur.09`) |

Two links break most often: the worker not handing the activity's context to the child (Python starts a new root trace), and the child exporting to the wrong port (its spans vanish; `telemetry.py` swallows export errors by design, so nothing says so).

### 2.2 The control-plane dashboard

| Panel | Query shape | Question |
|---|---|---|
| task queue depth by queue | `sum by (queue, kind) (tl_durable_task_queue_depth)` | is work piling up, and where |
| dead-letter queue size | `sum by (queue) (tl_durable_dlq_size)` (red at 1) | is work being given up on |
| redeliveries per second | `sum by (queue) (rate(tl_durable_redeliveries_total[$__rate_interval]))` | are workers dying or leases expiring |
| schedule-to-start p95 | `histogram_quantile(0.95, sum by (le, queue) (rate(..._bucket[$__rate_interval])))` | how long tasks wait for a worker |
| WAL size | `sum(tl_durable_wal_bytes)` against `walMaxBytes` | how close the quota is |
| training loss, tokens per second | `tl_train_loss`, `tl_train_tokens_per_second` | is the run making progress |

| Symbol | Meaning | Type |
|---|---|---|
| `$__rate_interval` | Grafana's rate window: at least four scrape intervals | duration |
| `${datasource}` | the dashboard's Prometheus datasource variable | string |

Every name is in `otel/metrics.yaml`; a panel on anything else is empty when it matters. Depth stays `by (queue)` because one stuck queue disappears in a total; redeliveries is a counter, so only its rate means anything.

### 2.3 Scrapes and pushes

The durable server and the workers serve `/metrics` on the port named `health` (9464), and a `PodMonitor` per component scrapes it. Python children live for minutes and are not addressable, so they **push** OTLP metrics to the collector, whose `prometheus` exporter (port `prom-exporter`, 8889) re-exposes them; a third monitor scrapes that. Every monitor carries `release: observability`, the label kube-prometheus-stack selects on.

## 3. Worked example by hand

**Schedule-to-start p95.** Over the last 5 minutes queue `data` saw 100 tasks with these cumulative bucket counts of `tl_durable_task_schedule_to_start_seconds` (rates scale them all by the same factor, so counts work by hand):

| `le` | 0.1 | 0.5 | 1 | 5 | 10 | +Inf |
|---|---|---|---|---|---|---|
| count | 40 | 70 | 88 | 97 | 100 | 100 |

The 95th task falls in the bucket $(1, 5]$: 88 tasks are below 1 s and 97 below 5 s. Prometheus interpolates linearly inside the bucket: $1 + (5 - 1) \times \frac{95 - 88}{97 - 88} = 1 + 4 \times \frac{7}{9} \approx 4.11$ s. Without `le` in the `by` clause the buckets are summed together and `histogram_quantile` returns NaN; that is `test_schedule_to_start_quantile`.

**The trace.** With `TRACEPARENT=00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01` (the activity span), a 120-step run exports one `train.run` in trace `4bf92f...4736` whose parent is `00f067aa0ba902b7`, and three `train.step` spans (steps 0, 50, 100) whose parent is `train.run`; that is `test_python_span_tree_from_traceparent`.

## 4. The artifact and its check

| File | Holds |
|---|---|
| `deploy/observability/dashboards/control-plane.json` | the six panels of 2.2, `uid` `<system>-control-plane`, datasource `${datasource}` |
| `deploy/observability/control-plane-monitors.yaml` | PodMonitors: component `durable` and `worker` on port `health`, the collector on `prom-exporter`; each `release: observability` |
| `deploy/helm/<system>-worker/values.yaml` (dep.06) | `OTEL_EXPORTER_OTLP_ENDPOINT` on 4317 for the Go worker, `TL_PYTHON_OTLP_ENDPOINT` on 4318 that the worker passes to `Subprocess.OTLPEndpoint` |

### What the tests check

| Test | KIND | Checks |
|---|---|---|
| `test_dashboard_is_portable` | unit | stable uid, no id, the datasource variable on every panel |
| `test_queries_name_contract_metrics` | conformance | every metric in `otel/metrics.yaml` |
| `test_queries_parse` | conformance | promtool parses every query |
| `test_queue_panels` | unit | depth by queue, DLQ size, redeliveries as a rate |
| `test_schedule_to_start_quantile` | boundary | section 3's quantile shape |
| `test_storage_and_training_panels` | unit | WAL size and training loss panels |
| `test_monitors_scrape_the_control_plane` | unit | the three monitors with the release label |
| `test_python_exports_over_http` | boundary | Go on 4317, Python on 4318 |
| `test_python_span_tree_from_traceparent` | conformance | section 3's trace from your `telemetry.py` |
| `test_queries_run_in_prometheus` | conformance | every query evaluates on kind |
| `test_train_trace_in_tempo` | conformance | one trace holds `workflow TrainRun`, `activity train`, `train.run`, `train.step` |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. queue depth summed over queues | a stuck `train` queue hides behind a quiet `data` queue | `test_queue_panels` |
| 2. the raw redeliveries counter on a graph | a line that only rises; a kill loop looks like nothing | `test_queue_panels` |
| 3. `le` dropped from the quantile's `by` | the p95 panel shows NaN | `test_schedule_to_start_quantile` |
| 4. Python pointed at the gRPC port | Python spans and gauges vanish silently | `test_python_exports_over_http` |
| 5. the activity context not handed to the child | `train.run` starts a new trace; the TrainRun trace ends at `activity train` | `test_python_span_tree_from_traceparent`, `test_train_trace_in_tempo` |
| 6. a monitor without `release: observability` | valid YAML, no scrape, empty panels | `test_monitors_scrape_the_control_plane` |
| 7. a panel on a metric outside the contract | an empty panel during the incident | `test_queries_name_contract_metrics` |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `dep.06` | the durable and worker pods these monitors scrape, and the worker's export settings |
| Back | `dur.09` | `telemetry.py` makes the Python spans |
| Back | `obs.04` | dashboards as code, promtool parsing |
| Forward | `ops.02` | `durable-kill9` is detected as a redeliveries spike on this dashboard |
| Forward | `ops.03` | `poison-task` is detected as a dead letter |
| Forward | `ops.11` | `eventlog-disk-full` shows on the WAL size panel |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| queue gauges from the server | Temporal's `schedule_to_start_latency` and task-queue backlog metrics | per-worker poll success rates, sticky-cache hit rates | Temporal "SDK metrics" reference |
| PodMonitors | the OpenTelemetry Operator's target allocator | collectors that discover and shard scrape targets themselves | OpenTelemetry Operator docs |
| head sampling (every 50th step) | tail sampling in the collector | keep whole traces that errored or ran long, drop the rest | `tailsamplingprocessor` |
| one Grafana dashboard | SLOs on queue latency with burn-rate alerts | alert on schedule-to-start budget burn, not on raw depth | OpenSLO, Sloth |
