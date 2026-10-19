# Observability for the control plane

The serving path is now visible end to end, from [One trace from gateway to engine](00-one-trace.md) through [Dashboards as code for the serving path](04-dashboards-as-code.md). The control plane, the durable orchestration server and its workers built in *AI Platform Engineering*, is still a black box. When a corpus build stalls, nothing says whether the queue is growing, a task is stuck in the dead-letter queue, or workers keep dying and redelivering. The training run is worse: the interesting work happens in a Python child process two hops from the CLI, and without a propagated context its spans either do not exist or start a trace of their own that nobody can connect to the `TrainRun` that caused them.

This section makes the control plane observable. You will build the dashboard that the control-plane drills are detected on, and connect one trace from the command typed at the terminal to the training step inside the Python child. Nothing new is invented here; the section applies the tracing, scraping, and dashboard rules of the previous sections to a queue-based system with a short-lived subprocess at the end.

It builds on the charts from [Durable and worker images and charts](../../infrastructure/01-containers-kubernetes/06-durable-and-worker-charts.md) in *Infrastructure*, which these panels and monitors watch; on `telemetry.py`, the Python end of the trace, from [Subprocess activities](../../ai-platform-engineering/05-durable-orchestration-and-workers/09-subprocess-activities.md) in *AI Platform Engineering*; and on the dashboards-as-code rules of the previous section. It also reads `go/otelx` from [Tracing for the serving path](01-tracing-for-the-serving-path.md) for the Go spans, and the scraping conventions of [Metrics and log correlation](02-metrics-and-log-correlation.md).

The solution is [`deploy/observability/dashboards/control-plane.json`](../../../../course/solutions/entry/deploy/observability/dashboards/control-plane.json) (`uid: forge-control-plane`: queue depth, dead letters, redeliveries, schedule-to-start, WAL size, training gauges), [`deploy/observability/control-plane-monitors.yaml`](../../../../course/solutions/entry/deploy/observability/control-plane-monitors.yaml) (PodMonitors `forge-durable`, `forge-worker`, `forge-python`), and the worker chart's export settings in [`forge-worker/values.yaml`](../../../../course/solutions/entry/deploy/helm/forge-worker/values.yaml): Go over OTLP/gRPC, Python over OTLP/HTTP. The trace hand-off is `Subprocess.childEnv` in [`subprocess.go`](../../../../course/solutions/go/activities/subprocess.go) and `Tracer.from_env` in [`telemetry.py`](../../../../course/solutions/python/tinyllm/io/telemetry.py). The contracts are [`otel/semconv.md`](../../../../course/contracts/otel/semconv.md) (the span tree from `{ctl} train` to `train.step`), [`otel/metrics.yaml`](../../../../course/contracts/otel/metrics.yaml) (`tl.durable.*`, `tl.train.*`), and [`helm/observability.md`](../../../../course/contracts/helm/observability.md) (the monitor label, the Python metrics pipeline).

> **Learning objectives**
> - Trace one `TrainRun` across four processes, and name the two links that break most often.
> - Build a queue dashboard that answers whether work is piling up, being given up on, or being retried because workers die.
> - Compute a schedule-to-start p95 from bucket counts by hand, and write the `histogram_quantile` query that matches it.
> - Explain why Python children push metrics over OTLP/HTTP while the Go worker speaks OTLP/gRPC, and configure both endpoints.

## One trace across four processes

One TrainRun is one trace. The context passes from process to process, and each hop gets its parent from a different place:

| Process | Span | Gets its parent from |
|---|---|---|
| `{ctl} train` | the CLI span | a new root |
| durable server | stores the context in `WorkflowExecutionStarted.trace_context` | the `StartWorkflow` request's metadata |
| worker (workflow task) | `workflow TrainRun` | the stored context, on every replay (`tl.workflow.replay = true` when replaying) |
| worker (activity task) | `activity train` | `ActivityTask.trace_context` |
| Python child | `train.run`, then `train.step` every 50th step, `train.checkpoint` | `TRACEPARENT`, set by the subprocess runner |

`{ctl} train` starts the trace, the durable server stores the context in the workflow's first event, every activity task carries it, the worker hands it to Python as `TRACEPARENT`, and `train.step` spans hang under it every 50th step.

Two links break most often. The first is the worker not handing the activity's context to the child, so Python starts a new root trace. The second is the child exporting to the wrong port, so its spans vanish; `telemetry.py` swallows export errors by design, so nothing says so.

## The control-plane dashboard

Tracing covers one run; the dashboard covers the whole queue system. A queue-based system answers three questions on one screen: is work **piling up** (depth per queue), is work being **given up on** (dead letters), and are workers **dying** (redeliveries, as a rate). Two more panels cover storage and training progress:

| Panel | Query shape | Question |
|---|---|---|
| task queue depth by queue | `sum by (queue, kind) (tl_durable_task_queue_depth)` | is work piling up, and where |
| dead-letter queue size | `sum by (queue) (tl_durable_dlq_size)` (red at 1) | is work being given up on |
| redeliveries per second | `sum by (queue) (rate(tl_durable_redeliveries_total[$__rate_interval]))` | are workers dying or leases expiring |
| schedule-to-start p95 | `histogram_quantile(0.95, sum by (le, queue) (rate(..._bucket[$__rate_interval])))` | how long tasks wait for a worker |
| WAL size | `sum(tl_durable_wal_bytes)` against `walMaxBytes` | how close the quota is |
| training loss, tokens per second | `tl_train_loss`, `tl_train_tokens_per_second` | is the run making progress |

As before, `$__rate_interval` is Grafana's rate window of at least four scrape intervals, and `${datasource}` is the dashboard's Prometheus datasource variable.

Every name is in `otel/metrics.yaml`; a panel on anything else is empty when it matters. Depth stays `by (queue)` because one stuck queue disappears in a total. Redeliveries is a counter, so only its rate means anything. **Schedule-to-start** latency is the queue's user-facing latency, and its p95 is `histogram_quantile` over bucket **rates** grouped **by `le`**.

## Scrapes and pushes

The dashboard needs its series to reach Prometheus, and here the control plane has two kinds of process. The durable server and the workers serve `/metrics` on the port named `health` (9464), and a `PodMonitor` per component scrapes it. Python children live for minutes and are not addressable, so they **push** OTLP metrics to the collector, whose `prometheus` exporter (port `prom-exporter`, 8889) re-exposes them; a third monitor scrapes that. The Go worker speaks OTLP/**gRPC** on 4317, Python OTLP/**HTTP** on 4318.

Every monitor carries `release: observability`, the label kube-prometheus-stack selects on, or Prometheus never sees it.

## Working it by hand

Start with the schedule-to-start p95. Over the last 5 minutes queue `data` saw 100 tasks with these cumulative bucket counts of `tl_durable_task_schedule_to_start_seconds`. Rates scale them all by the same factor, so counts work by hand:

| `le` | 0.1 | 0.5 | 1 | 5 | 10 | +Inf |
|---|---|---|---|---|---|---|
| count | 40 | 70 | 88 | 97 | 100 | 100 |

The 95th task falls in the bucket $(1, 5]$: 88 tasks are below 1 s and 97 below 5 s. Prometheus interpolates linearly inside the bucket:

$$1 + (5 - 1) \times \frac{95 - 88}{97 - 88} = 1 + 4 \times \frac{7}{9} \approx 4.11 \text{ s}$$

Without `le` in the `by` clause the buckets are summed together and `histogram_quantile` returns NaN.

Now the trace. With `TRACEPARENT=00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01` (the activity span), a 120-step run exports one `train.run` in trace `4bf92f...4736` whose parent is `00f067aa0ba902b7`, and three `train.step` spans (steps 0, 50, 100) whose parent is `train.run`.

## The artifacts

Three files make up the work. `deploy/observability/dashboards/control-plane.json` holds the six panels above, with `uid` `<system>-control-plane` and datasource `${datasource}`. `deploy/observability/control-plane-monitors.yaml` holds the PodMonitors: components `durable` and `worker` on port `health`, and the collector on `prom-exporter`, each labelled `release: observability`. And the worker chart's `deploy/helm/<system>-worker/values.yaml` sets `OTEL_EXPORTER_OTLP_ENDPOINT` on 4317 for the Go worker and `TL_PYTHON_OTLP_ENDPOINT` on 4318, which the worker passes to `Subprocess.OTLPEndpoint`. The solution is used like this:

```bash
kubectl apply -f deploy/observability/control-plane-monitors.yaml
<system> train --spec specs/tiny.json # on kind: produces one trace from the CLI to train.step
```

## Building it step by step

Seven observations fix the solution. A queue system fails in three ways: work piles up, work is given up on, or work is retried because workers die. Each has one contract metric (a depth gauge, a DLQ gauge, a redeliveries counter), so the queue row has exactly those three panels plus the latency users feel, schedule-to-start.

Keep the dimension that localises the fault. One stuck queue vanishes in a total, so every queue panel groups `by (queue)`.

Counters only mean something as rates. Redeliveries is `_total`, so the panel is `rate(...)`, which turns a kill loop into a spike instead of an ever-rising line.

Quantiles need bucket rates grouped by `le`, as the worked example showed; grouping additionally by `queue` gives one p95 line per queue.

What cannot be scraped must push. The server and workers are long-lived pods with a health port, so they get PodMonitors. Python children are short-lived and unaddressable, so they push OTLP to the collector, and a third PodMonitor scrapes the collector's Prometheus exporter.

Two protocols mean two ports. The Go SDK speaks OTLP/gRPC (4317); the hand-written Python exporter posts OTLP/JSON over HTTP (4318). The worker chart therefore carries two endpoints.

One trace needs one unbroken chain of contexts. The last link is the worker setting `TRACEPARENT` from the activity's context, and Python reading it as its remote parent.

### The monitors

The monitors follow the conventions of [Metrics and log correlation](02-metrics-and-log-correlation.md); the durable and worker monitors differ only in the component:

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
    - port: prom-exporter                      # 8889, opened in collector.yaml
      path: /metrics
      interval: 15s
      relabelings:
        - targetLabel: job
          replacement: forge-python
```

### The dashboard

The dashboard reuses the serving dashboard's header (`"id": null`, `uid` `forge-control-plane`, the `datasource` variable, UTC) and has three rows with these queries:

| Row | Panel | Query |
|---|---|---|
| Queues | Task queue depth by queue (time series) | `sum by (queue, kind) (tl_durable_task_queue_depth)` |
| Queues | Dead-letter queue size (stat, red at 1) | `sum by (queue) (tl_durable_dlq_size)` |
| Queues | Redeliveries per second | `sum by (queue) (rate(tl_durable_redeliveries_total[$__rate_interval]))` |
| Queues | Schedule-to-start p95 | `histogram_quantile(0.95, sum by (le, queue) (rate(tl_durable_task_schedule_to_start_seconds_bucket[$__rate_interval])))` |
| Storage | WAL size (gauge, bytes) | `sum(tl_durable_wal_bytes)` |
| Training | Training loss by workflow | `tl_train_loss` |
| Training | Training tokens per second | `tl_train_tokens_per_second` |

The DLQ stat's thresholds are the one non-obvious field. Any dead letter is a task the system gave up on, so the panel turns red at 1, not at some "too many":

```json
"fieldConfig": {"defaults": {"unit": "short",
  "thresholds": {"mode": "absolute", "steps": [
    {"color": "green", "value": null},
    {"color": "red", "value": 1}
  ]}}, "overrides": []}
```

The training panels use `legendFormat` `{{tl_workflow_id}}`, so two concurrent TrainRuns are two lines, not one jagged one.

### Export settings

The worker chart carries both endpoints:

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

### The hand-off to Python

`Subprocess.childEnv` builds the child's environment for one delivery. It first removes any inherited copies, because a worker started under its own `TRACEPARENT` must not leak it into an activity, then sets the activity's context and the OTLP endpoint:

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

Spans are posted to `endpoint + "/v1/traces"` and gauges to `"/v1/metrics"` as JSON. `_post` returns `False` on any `URLError` or `OSError` instead of raising, because telemetry must never fail the training step it observes. That is also why a wrong port is silent, and why it is the most common broken link. `train.step` spans are sampled with `should_sample_step(step, every=50)`, which is `step % every == 0`: steps 0, 50, and 100 in the worked example.

The full solution is in [`control-plane-monitors.yaml`](../../../../course/solutions/entry/deploy/observability/control-plane-monitors.yaml), [`control-plane.json`](../../../../course/solutions/entry/deploy/observability/dashboards/control-plane.json), [`forge-worker/values.yaml`](../../../../course/solutions/entry/deploy/helm/forge-worker/values.yaml), [`subprocess.go`](../../../../course/solutions/go/activities/subprocess.go), and [`telemetry.py`](../../../../course/solutions/python/tinyllm/io/telemetry.py).

### Why this design

The dashboard runs seven queries over gauges and counters whose label sets are bounded by the number of queues and workflows, with three scrape targets per replica set. Python's push is one HTTP request per flush and never blocks training.

Python pushes rather than serving a scrape endpoint in each child because a child lives minutes, has no Service, and may exit between two scrapes; its final gauges would be lost. Pushing on flush and shutdown delivers them, and the collector turns them back into scrapeable series.

Only every 50th step gets a span because a long run has hundreds of thousands of steps, and one span each would dominate the trace backend for little information. Every 50th keeps the shape of the run; tail sampling is the production refinement.

DLQ size is a gauge on the dashboard, not alert-only, because the poison-task drill is detected on the dashboard. A stat that turns red at 1 makes the dead letter visible to anyone looking, and the alert rules (`rules/durable.yaml`) page on the same series.

## What can go wrong

A correct setup is checked as follows. The dashboard must be portable (a stable uid, no id, the datasource variable on every panel); every metric must be in `otel/metrics.yaml`; and `promtool` must parse every query. The queue panels must show depth by queue, DLQ size, and redeliveries as a rate; the schedule-to-start quantile must have the shape worked out by hand; and there must be WAL size and training loss panels. The three monitors must exist with the release label. Go must export on 4317 and Python on 4318, and `telemetry.py` given a `TRACEPARENT` must produce the span tree of the worked example. On kind, every query must evaluate, and Tempo must hold one trace containing `workflow TrainRun`, `activity train`, `train.run`, and `train.step`.

The usual mistakes:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| Queue depth summed over queues | a stuck `train` queue hides behind a quiet `data` queue | depth is `sum by (queue, kind)` |
| The raw redeliveries counter on a graph | a line that only rises; a kill loop looks like nothing | the panel is `rate(tl_durable_redeliveries_total[$__rate_interval])` |
| `le` dropped from the quantile's `by` | the p95 panel shows NaN | `sum by (le, queue)` inside `histogram_quantile` |
| Python pointed at the gRPC port | Python spans and gauges vanish silently | the chart sets `TL_PYTHON_OTLP_ENDPOINT` to port 4318 |
| The activity context not handed to the child | `train.run` starts a new trace; the TrainRun trace ends at `activity train` | `childEnv` sets `TRACEPARENT` from `ActivityTask.trace_context`; `from_env` reads it |
| A monitor without `release: observability` | valid YAML, no scrape, empty panels | all three monitors carry the label |
| A panel on a metric outside the contract | an empty panel during the incident | every query names a `metrics.yaml` metric |

## Where this leads

This section closes the chapter. It watched the durable and worker pods of [Durable and worker images and charts](../../infrastructure/01-containers-kubernetes/06-durable-and-worker-charts.md), used the Python spans that `telemetry.py` makes in [Subprocess activities](../../ai-platform-engineering/05-durable-orchestration-and-workers/09-subprocess-activities.md), and followed the dashboards-as-code rules of [Dashboards as code for the serving path](04-dashboards-as-code.md). The next chapter's drills are detected here: [The durable server is SIGKILLed in a loop during a corpus build](../05-incident-response-and-chaos/02-durable-kill9.md) shows as a redeliveries spike, [A poison task lands in the dead-letter queue](../05-incident-response-and-chaos/03-poison-task.md) as a dead letter, and [Drill: event log disk full](../05-incident-response-and-chaos/11-eventlog-disk-full.md) on the WAL size panel.

Each piece has a production equivalent. Queue gauges from the server correspond to Temporal's `schedule_to_start_latency` and task-queue backlog metrics, which add per-worker poll success rates and sticky-cache hit rates. PodMonitors can give way to the OpenTelemetry Operator's target allocator, with collectors that discover and shard scrape targets themselves. Head sampling of every 50th step gives way to tail sampling in the collector (`tailsamplingprocessor`), which keeps whole traces that errored or ran long and drops the rest. And one Grafana dashboard grows into SLOs on queue latency with burn-rate alerts (OpenSLO, Sloth), alerting on schedule-to-start budget burn rather than on raw depth.

### Further reading

- [Temporal: SDK metrics and schedule-to-start latency](https://docs.temporal.io/references/sdk-metrics) (free)
- [OTLP specification](https://opentelemetry.io/docs/specs/otlp/) (free)
- [Prometheus: histograms and summaries](https://prometheus.io/docs/practices/histograms/) (free)
- OpenTelemetry Operator documentation, on the target allocator

## Key terms

- **schedule-to-start latency**: the time a task waits in its queue before a worker picks it up.
- **dead-letter queue (DLQ)**: where the system parks tasks it has given up retrying.
- **redelivery**: a task handed to a worker again because the previous worker died or its lease expired.
- **`TRACEPARENT`**: the environment variable through which the worker hands the activity's trace context to a Python child.
- **push metrics**: metrics a short-lived process sends over OTLP to the collector instead of serving them for a scrape.

## Summary

- One TrainRun is one trace: the CLI starts it, the durable server stores it in the workflow's first event, every activity task carries it, the worker hands it to Python as `TRACEPARENT`, and `train.step` spans hang under it every 50th step.
- A queue-based system answers three questions on one screen: is work piling up (depth per queue), being given up on (dead letters), or retried because workers die (redeliveries, as a rate).
- Schedule-to-start latency is the queue's user-facing latency; its p95 is `histogram_quantile` over bucket rates grouped by `le`.
- Python cannot be scraped, so it pushes metrics over OTLP to the collector, whose Prometheus exporter is scraped instead; the Go worker uses OTLP/gRPC on 4317, Python OTLP/HTTP on 4318.
- Monitors carry `release: observability`, or Prometheus never sees them.

## Review questions

1. Why does the queue depth panel group `by (queue, kind)` instead of summing?
   <details><summary>Answer</summary>A stuck `train` queue with 5 tasks next to a draining `data` queue with 500 is invisible in a total that hovers around 500. Grouping by queue shows the one line that does not move.</details>
2. Why is the Python endpoint port 4318 while the Go worker uses 4317?
   <details><summary>Answer</summary>The Go SDK exports OTLP over gRPC, served on 4317. `telemetry.py` posts OTLP/JSON over HTTP to `/v1/traces` and `/v1/metrics`, which the collector serves on 4318. Pointed at 4317, the HTTP posts fail, and because `_post` swallows errors, Python spans and gauges vanish without a message.</details>
3. What happens to the trace if `childEnv` does not set `TRACEPARENT`?
   <details><summary>Answer</summary>`from_env` finds no parent, so the `Tracer` mints a fresh trace id and `train.run` is a root of a new trace. The TrainRun trace ends at `activity train`, and nothing connects the Python spans to the CLI call.</details>
4. Why must `childEnv` drop an inherited `TRACEPARENT` even when the task carries none?
   <details><summary>Answer</summary>A worker started under a `TRACEPARENT` would otherwise pass its own context to every child, attaching unrelated activities to whatever trace started the worker. Dropping it means a task without a context gives a clean root, not a wrong parent.</details>
