# Observability for the control plane: workflow and activity spans, queue and DLQ panels

## Overview

| | |
|---|---|
| **Module** | `obs.05` · practice · ops · Pass 8 · 3 to 4 h |
| **Solution** | [`deploy/observability/dashboards/control-plane.json`](../../../../course/solutions/entry/deploy/observability/dashboards/control-plane.json) (`uid: forge-control-plane`: queue depth, dead letters, redeliveries, schedule-to-start, WAL size, training gauges), [`deploy/observability/control-plane-monitors.yaml`](../../../../course/solutions/entry/deploy/observability/control-plane-monitors.yaml) (PodMonitors `forge-durable`, `forge-worker`, `forge-python`), and the worker chart's export settings in [`forge-worker/values.yaml`](../../../../course/solutions/entry/deploy/helm/forge-worker/values.yaml) (`dep.06`): Go over OTLP/gRPC, Python over OTLP/HTTP; the trace hand-off is `Subprocess.childEnv` in [`subprocess.go`](../../../../course/solutions/go/activities/subprocess.go) and `Tracer.from_env` in [`telemetry.py`](../../../../course/solutions/python/tinyllm/io/telemetry.py) |
| **Contract** | [`otel/semconv.md`](../../../../course/contracts/otel/semconv.md) (the span tree from `{ctl} train` to `train.step`), [`otel/metrics.yaml`](../../../../course/contracts/otel/metrics.yaml) (`tl.durable.*`, `tl.train.*`), [`helm/observability.md`](../../../../course/contracts/helm/observability.md) (the monitor label, the Python metrics pipeline) |
| **Needs** | `dep.06` (the charts these panels and monitors watch), `dur.09` (`telemetry.py`, the Python end of the trace), `obs.04` (dashboards as code); reading: `obs.01` (`go/otelx`, the Go spans), `obs.02` (scraping) |
| **Used by** | no call site (a practice): drills `durable-kill9` and `poison-task` (`ops.02`, `ops.03`) are detected on this dashboard; the durable system shows one trace from `{ctl} train` to `train.step` on kind |
| **Optional depth** | [Temporal: SDK metrics and schedule-to-start latency](https://docs.temporal.io/references/sdk-metrics) (free); [OTLP specification](https://opentelemetry.io/docs/specs/otlp/) (free); [Prometheus: histograms and summaries](https://prometheus.io/docs/practices/histograms/) (free) |

## Key Takeaways

- One **TrainRun is one trace**: `{ctl} train` starts it, the durable server stores the context in the workflow's first event, every activity task carries it, the worker hands it to Python as `TRACEPARENT`, and `train.step` spans hang under it every 50th step.
- A queue-based system answers three questions on one screen: is work **piling up** (depth per queue), is work being **given up on** (dead letters), are workers **dying** (redeliveries, as a rate).
- **Schedule-to-start** latency is the queue's user-facing latency; its p95 is `histogram_quantile` over bucket **rates** grouped **by `le`**.
- Python cannot be scraped: it **pushes** metrics over OTLP to the collector, whose Prometheus exporter is scraped instead. The Go worker speaks OTLP/**gRPC** on 4317, Python OTLP/**HTTP** on 4318.
- Monitors carry `release: observability`, or Prometheus never sees them.

## How to read this chapter

Sections 1 to 3 build control-plane observability from first principles and work a schedule-to-start quantile and a Python span tree by hand. Section 4 lists the artifacts and what a correct setup must handle. Section 5 walks through the solution (monitors, dashboard, export settings, and the trace hand-off) and explains why each choice is the right one. Try to predict each step before reading it.

The solution is used like this:

```bash
kubectl apply -f deploy/observability/control-plane-monitors.yaml
<system> train --spec specs/tiny.json # on kind: produces one trace from the CLI to train.step
```

---

## 1. Why now

The serving path is visible end to end (`obs.01` to `obs.04`), but the control plane built in Pass 8 is a black box: when a corpus build stalls, nothing says whether the queue is growing, a task is stuck in the dead-letter queue, or workers keep dying and redelivering. The training run is worse: the interesting work happens in a Python child process two hops from the CLI, and without a propagated context its spans either do not exist or start a trace of their own that nobody can connect to the `TrainRun` that caused them. This module makes the control plane observable: the dashboard the drills of Pass 8 are detected on, and one trace from the command typed at the terminal to the training step.

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

The 95th task falls in the bucket $(1, 5]$: 88 tasks are below 1 s and 97 below 5 s. Prometheus interpolates linearly inside the bucket: $1 + (5 - 1) \times \frac{95 - 88}{97 - 88} = 1 + 4 \times \frac{7}{9} \approx 4.11$ s. Without `le` in the `by` clause the buckets are summed together and `histogram_quantile` returns NaN.

**The trace.** With `TRACEPARENT=00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01` (the activity span), a 120-step run exports one `train.run` in trace `4bf92f...4736` whose parent is `00f067aa0ba902b7`, and three `train.step` spans (steps 0, 50, 100) whose parent is `train.run`.

## 4. The artifacts

| File | Holds |
|---|---|
| `deploy/observability/dashboards/control-plane.json` | the six panels of 2.2, `uid` `<system>-control-plane`, datasource `${datasource}` |
| `deploy/observability/control-plane-monitors.yaml` | PodMonitors: component `durable` and `worker` on port `health`, the collector on `prom-exporter`; each `release: observability` |
| `deploy/helm/<system>-worker/values.yaml` (dep.06) | `OTEL_EXPORTER_OTLP_ENDPOINT` on 4317 for the Go worker, `TL_PYTHON_OTLP_ENDPOINT` on 4318 that the worker passes to `Subprocess.OTLPEndpoint` |

### What a correct solution must handle

| Case | Kind of check | Checks |
|---|---|---|
| The dashboard is portable | unit | stable uid, no id, the datasource variable on every panel |
| Queries name contract metrics | conformance | every metric in `otel/metrics.yaml` |
| Queries parse | conformance | promtool parses every query |
| Queue panels | unit | depth by queue, DLQ size, redeliveries as a rate |
| The schedule-to-start quantile | boundary | section 3's quantile shape |
| Storage and training panels | unit | WAL size and training loss panels |
| Monitors scrape the control plane | unit | the three monitors with the release label |
| Python exports over HTTP | boundary | Go on 4317, Python on 4318 |
| The Python span tree from `TRACEPARENT` | conformance | section 3's trace from `telemetry.py` |
| Queries run in Prometheus (cluster) | conformance | every query evaluates on kind |
| The train trace in Tempo (cluster) | conformance | one trace holds `workflow TrainRun`, `activity train`, `train.run`, `train.step` |

## 5. The solution, step by step

### 5.1 From first principles

1. **A queue system fails in three ways** (section 2.2): work piles up, work is given up on, or work is retried because workers die. Each has one contract metric (depth gauge, DLQ gauge, redeliveries counter), so the queue row has exactly those three panels plus the latency users feel, schedule-to-start.
2. **Keep the dimension that localises the fault.** One stuck queue vanishes in a total, so every queue panel groups `by (queue)`.
3. **Counters only mean something as rates.** Redeliveries is `_total`, so the panel is `rate(...)`, which turns a kill loop into a spike instead of an ever-rising line.
4. **Quantiles need bucket rates grouped by `le`** (section 3); grouping additionally by `queue` gives one p95 line per queue.
5. **What cannot be scraped must push.** Server and workers are long-lived pods with a health port: PodMonitors. Python children are short-lived and unaddressable: they push OTLP to the collector, and a third PodMonitor scrapes the collector's Prometheus exporter.
6. **Two protocols, two ports.** The Go SDK speaks OTLP/gRPC (4317); the hand-written Python exporter posts OTLP/JSON over HTTP (4318). The worker chart therefore carries two endpoints.
7. **One trace needs one unbroken chain of contexts** (section 2.1). The last link is the worker setting `TRACEPARENT` from the activity's context, and Python reading it as its remote parent.

### 5.2 The code, annotated

**The monitors** follow `obs.02`'s conventions; the durable and worker monitors differ only in the component:

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PodMonitor
metadata:
  name: forge-durable
  namespace: observability
  labels:
    release: observability                     # or the stack never loads it
spec:
  namespaceSelector:
    matchNames: [forge]
  selector:
    matchLabels:
      app.kubernetes.io/component: durable     # forge-worker: component worker
  podMetricsEndpoints:
    - port: health
      path: /metrics
      interval: 10s
      relabelings:
        - targetLabel: job
          replacement: forge-durable
---
# The Python subprocesses' OTLP metrics (tl_train_*) appear on the collector's
# Prometheus exporter: they cannot be scraped themselves (D9).
apiVersion: monitoring.coreos.com/v1
kind: PodMonitor
metadata:
  name: forge-python
  namespace: observability
  labels:
    release: observability
spec:
  namespaceSelector:
    matchNames: [observability]                # the collector runs in the stack's namespace
  selector:
    matchLabels:
      app.kubernetes.io/name: opentelemetry-collector
  podMetricsEndpoints:
    - port: prom-exporter                      # 8889, opened in collector.yaml (obs.02)
      path: /metrics
      interval: 15s
      relabelings:
        - targetLabel: job
          replacement: forge-python
```

**The dashboard** reuses `obs.04`'s header (`"id": null`, `uid` `forge-control-plane`, the `datasource` variable, UTC) and has three rows. Its queries:

| Row | Panel | Query |
|---|---|---|
| Queues | Task queue depth by queue (time series) | `sum by (queue, kind) (tl_durable_task_queue_depth)` |
| Queues | Dead-letter queue size (stat, red at 1) | `sum by (queue) (tl_durable_dlq_size)` |
| Queues | Redeliveries per second | `sum by (queue) (rate(tl_durable_redeliveries_total[$__rate_interval]))` |
| Queues | Schedule-to-start p95 | `histogram_quantile(0.95, sum by (le, queue) (rate(tl_durable_task_schedule_to_start_seconds_bucket[$__rate_interval])))` |
| Storage | WAL size (gauge, bytes) | `sum(tl_durable_wal_bytes)` |
| Training | Training loss by workflow | `tl_train_loss` |
| Training | Training tokens per second | `tl_train_tokens_per_second` |

The DLQ stat's thresholds are the one non-obvious field: any dead letter is a task the system gave up on, so the panel turns red at 1, not at some "too many":

```json
"fieldConfig": {"defaults": {"unit": "short",
  "thresholds": {"mode": "absolute", "steps": [
    {"color": "green", "value": null},
    {"color": "red", "value": 1}
  ]}}, "overrides": []}
```

The training panels use `legendFormat` `{{tl_workflow_id}}`, so two concurrent TrainRuns are two lines, not one jagged one.

**Export settings** in the worker chart:

```yaml
otel:
  endpoint: http://otel-collector.observability:4317   # the Go worker: OTLP over gRPC
  serviceName: forge-worker

env:
  # The Python children export OTLP over HTTP with a JSON body (dur.09,
  # telemetry.py): port 4318, not the gRPC 4317 the Go worker uses.
  - name: TL_PYTHON_OTLP_ENDPOINT
    value: http://otel-collector.observability:4318
  - name: TL_PYTHON_SERVICE_NAME
    value: forge-python
```

**The hand-off to Python.** `Subprocess.childEnv` builds the child's environment for one delivery. It first removes any inherited copies (a worker started under its own `TRACEPARENT` must not leak it into an activity), then sets the activity's context and the OTLP endpoint:

```go
	if tp := t.TraceContext["traceparent"]; tp != "" {
		set["TRACEPARENT"] = tp // the activity span: train.run's parent
		if ts := t.TraceContext["tracestate"]; ts != "" {
			set["TRACESTATE"] = ts
		}
	}
	if s.OTLPEndpoint != "" {
		set["OTEL_EXPORTER_OTLP_ENDPOINT"] = s.OTLPEndpoint // must be the OTLP/HTTP (4318) endpoint
	}
	// ...
	drop := map[string]bool{"TRACEPARENT": true, "TRACESTATE": true, "OTEL_EXPORTER_OTLP_ENDPOINT": true}
```

On the Python side, `Tracer.from_env` reads `OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_SERVICE_NAME`, and `TRACEPARENT`. A valid parent fixes the trace id and the sampled flag, so every span the child makes joins the TrainRun's trace:

```python
        # A remote parent decided whether this trace is recorded; a new root
        # trace is.
        self.sampled = parent.sampled if parent is not None else True
        self._trace_id = parent.trace_id if parent is not None else self._new_id(16)
```

Spans are posted to `endpoint + "/v1/traces"` and gauges to `"/v1/metrics"` as JSON; `_post` returns `False` on any `URLError` or `OSError` instead of raising, because telemetry must never fail the training step it observes. That is also why a wrong port is silent, and why section 2.1 calls it the most common broken link. `train.step` spans are sampled with `should_sample_step(step, every=50)`, which is `step % every == 0`: steps 0, 50, 100 of section 3.

Full solution: [`control-plane-monitors.yaml`](../../../../course/solutions/entry/deploy/observability/control-plane-monitors.yaml), [`control-plane.json`](../../../../course/solutions/entry/deploy/observability/dashboards/control-plane.json), [`forge-worker/values.yaml`](../../../../course/solutions/entry/deploy/helm/forge-worker/values.yaml), [`subprocess.go`](../../../../course/solutions/go/activities/subprocess.go), [`telemetry.py`](../../../../course/solutions/python/tinyllm/io/telemetry.py).

### 5.3 Why this solution

- **Cost.** Seven queries over gauges and counters whose label sets are bounded by the number of queues and workflows; three scrape targets per replica set. Python's push is one HTTP request per flush and never blocks training.
- **Why push for Python instead of a scrape endpoint in each child?** A child lives minutes, has no Service, and may exit between two scrapes; its final gauges would be lost. Pushing on flush and shutdown delivers them, and the collector turns them back into scrapeable series.
- **Why `every 50th step` instead of every step?** A long run has hundreds of thousands of steps; one span each would dominate the trace backend for little information. Every 50th keeps the shape of the run; tail sampling (Going further) is the production refinement.
- **Why a gauge for DLQ size and not alert-only?** The drill (`ops.03`) is detected on the dashboard; a stat that turns red at 1 makes the dead letter visible to anyone looking, and the alert rules (`rules/durable.yaml`) page on the same series.

### 5.4 Explain it back

1. Why does the queue depth panel group `by (queue, kind)` instead of summing?
   <details><summary>Answer</summary>A stuck `train` queue with 5 tasks next to a draining `data` queue with 500 is invisible in a total that hovers around 500. Grouping by queue shows the one line that does not move.</details>
2. Why is the Python endpoint port 4318 while the Go worker uses 4317?
   <details><summary>Answer</summary>The Go SDK exports OTLP over gRPC, served on 4317. `telemetry.py` posts OTLP/JSON over HTTP to `/v1/traces` and `/v1/metrics`, which the collector serves on 4318. Pointed at 4317, the HTTP posts fail, and because `_post` swallows errors, Python spans and gauges vanish without a message.</details>
3. What happens to the trace if `childEnv` does not set `TRACEPARENT`?
   <details><summary>Answer</summary>`from_env` finds no parent, so the `Tracer` mints a fresh trace id and `train.run` is a root of a new trace. The TrainRun trace ends at `activity train`, and nothing connects the Python spans to the CLI call.</details>
4. Why must `childEnv` drop an inherited `TRACEPARENT` even when the task carries none?
   <details><summary>Answer</summary>A worker started under a `TRACEPARENT` would otherwise pass its own context to every child, attaching unrelated activities to whatever trace started the worker. Dropping it means a task without a context gives a clean root, not a wrong parent.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. queue depth summed over queues | a stuck `train` queue hides behind a quiet `data` queue | depth is `sum by (queue, kind)` |
| 2. the raw redeliveries counter on a graph | a line that only rises; a kill loop looks like nothing | the panel is `rate(tl_durable_redeliveries_total[$__rate_interval])` |
| 3. `le` dropped from the quantile's `by` | the p95 panel shows NaN | `sum by (le, queue)` inside `histogram_quantile` |
| 4. Python pointed at the gRPC port | Python spans and gauges vanish silently | the chart sets `TL_PYTHON_OTLP_ENDPOINT` to port 4318 |
| 5. the activity context not handed to the child | `train.run` starts a new trace; the TrainRun trace ends at `activity train` | `childEnv` sets `TRACEPARENT` from `ActivityTask.trace_context`; `from_env` reads it |
| 6. a monitor without `release: observability` | valid YAML, no scrape, empty panels | all three monitors carry the label |
| 7. a panel on a metric outside the contract | an empty panel during the incident | every query names a `metrics.yaml` metric |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `dep.06` | the durable and worker pods these monitors scrape, and the worker's export settings |
| Back | `dur.09` | `telemetry.py` makes the Python spans |
| Back | `obs.04` | dashboards as code |
| Forward | `ops.02` | `durable-kill9` is detected as a redeliveries spike on this dashboard |
| Forward | `ops.03` | `poison-task` is detected as a dead letter |
| Forward | `ops.11` | `eventlog-disk-full` shows on the WAL size panel |

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| queue gauges from the server | Temporal's `schedule_to_start_latency` and task-queue backlog metrics | per-worker poll success rates, sticky-cache hit rates | Temporal "SDK metrics" reference |
| PodMonitors | the OpenTelemetry Operator's target allocator | collectors that discover and shard scrape targets themselves | OpenTelemetry Operator docs |
| head sampling (every 50th step) | tail sampling in the collector | keep whole traces that errored or ran long, drop the rest | `tailsamplingprocessor` |
| one Grafana dashboard | SLOs on queue latency with burn-rate alerts | alert on schedule-to-start budget burn, not on raw depth | OpenSLO, Sloth |
