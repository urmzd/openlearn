# Observability

Monitoring tells you *when* something is wrong. Observability tells you *why*. In a distributed system you cannot predict every failure mode in advance, so the useful property is not a wall of pre-built alerts but the ability to understand the system's internal state from its external outputs: to ask arbitrary new questions, and debug problems you have never seen before, without deploying new code. This chapter builds that ability for a real system, the tracer and then the production serving platform, one signal at a time.

The arc is from one request to the whole fleet. It starts with a single trace that crosses the gateway and the engine, written by hand so that every id and timestamp is visible. It then generalizes tracing into a small kit every Go service shares, adds metrics and logs that join to traces by id, turns those metrics into SLOs and burn-rate alerts, and puts the answers on dashboards kept as code. It ends by applying the same rules to the durable control plane, where the last hop is a short-lived Python process. By the end, an on-call engineer paged at night can go from the alert to the dashboard to the one slow request's trace and logs in a few minutes, which is exactly what the drills of the next chapter test.

> **Learning objectives**
> - Explain the difference between monitoring and observability, and why high-cardinality, high-dimensionality structured events matter.
> - Propagate W3C trace context across HTTP, gRPC, durable workflows, and subprocesses, so one request or one training run is one trace.
> - Serve contract metrics with bounded cardinality, scrape them with Prometheus, and correlate logs with traces by id.
> - Define SLOs, compute error budgets and burn rates, and write multi-window burn-rate alerts that page when they should and only then.
> - Keep dashboards as code that answer an incident's first questions, including latency heatmaps that reveal a second mode.

## Before you begin

This chapter assumes some production experience and the material of [Cloud Native](../03-cloud-native/README.md): containers, Kubernetes, and Helm. The sections instrument a system built elsewhere, so it helps to have read [Tracer deploy](../../infrastructure/01-containers-kubernetes/00-tracer-deploy.md) in *Infrastructure*, [The tracer gateway](../../ai-platform-engineering/12-gateway/00-streaming-proxy.md) in *AI Platform Engineering*, and [Your first endpoint](../../build-an-llm/p10-serving/00-your-first-endpoint.md) in *Build Your Own LLM System*. The primary reference is *Observability Engineering* by Charity Majors, Liz Fong-Jones, and George Miranda (O'Reilly); the free [Google SRE Book](https://sre.google/sre-book/) is the supplementary one.

## Core ideas

**Observability versus monitoring.** The three pillars, logs (events), metrics (aggregated numbers), and traces (request flow across services), are necessary but not sufficient. Two properties decide whether you can debug something new. *High cardinality* means many unique values, such as user ids and request ids; metrics cannot afford them, but events can. *High dimensionality* means many attributes per event, and structured events with rich context are what make ad-hoc queries possible. Observability is structured events plus ad-hoc queries; the ability to ask "what is different about the slow requests?" is the differentiator.

**Instrumentation.** OpenTelemetry is the vendor-neutral standard for traces, metrics, and logs: an SDK, a collector, and exporters. A span is a unit of work with a start time, duration, and attributes, and nested spans form traces. Context propagation carries the trace id across service boundaries in headers (W3C Trace Context). Instrument service boundaries, database calls, external API calls, queue publish and consume, and significant business logic. The practical rule is to instrument at the boundaries first and add internal instrumentation only when debugging requires it.

**Debugging with observability.** The core analysis loop is: notice a symptom, form a hypothesis, query to test it, then refine or pivot. Given slow requests, ask which dimensions differ from fast ones (perhaps all the slow ones hit shard 7). Heatmaps show the latency distribution over time and reveal bimodal distributions that a p99 line hides. Monitoring cannot alert on failure modes you never anticipated, the unknown unknowns; observability lets you explore them.

**SLOs and reliability.** An SLI (service level indicator) is a measured quantity, such as the fraction of requests under 200 ms. An SLO (objective) is the target for it, such as 99.9%. An SLA (agreement) is an SLO with business consequences: refunds or credits. The error budget is $100\% - \text{SLO}$; at 99.9% it is 0.1%, or about 8.7 hours a year. Burn-rate alerts fire when the budget is being consumed too fast, not when a single request fails. An error-budget policy says what happens when the budget is exhausted, typically a deployment freeze and a focus on reliability, which makes reliability a first-class engineering concern.

**Observability in practice.** Every deploy should be observed, and canary analysis compares the new version's metrics with the baseline. Staging environments lie, so teams test in production with dark launches, feature flags, and traffic mirroring. Progressive delivery moves from canary to a percentage rollout to full rollout, with automatic rollback on SLO violation. Chaos engineering deliberately injects failures (Chaos Monkey, Litmus) to verify resilience, and is only safe with good observability already in place.

**Organization and culture.** Observability is a cultural practice, not just a tooling choice. Blameless postmortems focus on systems and processes rather than individuals: what can we change so this class of failure cannot happen again? Incident response uses structured roles (incident commander, communications lead, subject-matter experts) and is practised in game days. Observability-driven development instruments first and builds features second: if you cannot observe it, you cannot operate it. On-call needs a fair rotation, clear escalation paths, and runbooks, and alert fatigue is its enemy.

The practices above come with characteristic anti-patterns:

| Practice | When to use | Anti-pattern |
|----------|-------------|-------------|
| Structured events | Always; replace unstructured logs | Unstructured log grep as primary debugging |
| SLOs | Every user-facing service | Alerting on every error instead of budget burn |
| Distributed tracing | Multi-service architectures | Tracing only within a single service |
| Canary deploys | Every production deployment | Big-bang deployments |
| Error budgets | Balancing reliability and velocity | "Always be more reliable" (no budget = no shipping) |
| Blameless postmortems | Every significant incident | Blame individuals, skip action items |

Different organizations emphasize different parts of this picture:

| Company | Approach | Focus |
|---------|----------|-------|
| Google | SRE (invented it), SLOs, error budgets | Reliability as engineering discipline |
| Netflix | Chaos engineering (Chaos Monkey), progressive delivery | Resilience through failure injection |
| Anthropic | ML system observability, inference SLOs | GPU utilization, latency percentiles |
| Amazon | Well-Architected Framework, COE (Correction of Errors) | Operational excellence |
| Honeycomb | Observability 2.0 (structured events, BubbleUp) | Debugging novel problems |

## In this chapter

1. [One trace from gateway to engine](00-one-trace.md): spans, `traceparent` propagation, and OTLP/HTTP JSON, built by hand into a gateway exporter that can never slow a request, with the result checked in Jaeger.
2. [Tracing for the serving path](01-tracing-for-the-serving-path.md): the `go/otelx` kit that every Go service shares, with HTTP and gRPC propagation, CLIENT spans that cover streamed bodies, `ParentBased` sampling, and batch export.
3. [Metrics and log correlation](02-metrics-and-log-correlation.md): contract histograms with exact bounds, capped labels, PodMonitors, the OpenTelemetry Collector, and JSON logs that grep by trace id.
4. [SLOs and multi-window burn-rate alerts](03-slos-and-burn-rate-alerts.md): error budgets, burn rates, two-window alerts in prod and drill profiles, rendered from one SLO file and tested with `promtool`.
5. [Dashboards as code for the serving path](04-dashboards-as-code.md): a portable Grafana dashboard with one row per incident question, and heatmaps that show what a p95 line hides.
6. [Observability for the control plane](05-control-plane-observability.md): one trace from the CLI to a Python training step, and a queue dashboard for depth, dead letters, redeliveries, and schedule-to-start latency.

## Further reading

- Charity Majors, Liz Fong-Jones, and George Miranda, *Observability Engineering* (O'Reilly)
- [Google SRE Book](https://sre.google/sre-book/) (free) and *The Site Reliability Workbook* (free online)
- DORA metrics, for measuring delivery performance alongside reliability
