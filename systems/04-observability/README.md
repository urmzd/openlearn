# Observability

## Overview

- **Primary reference**: *Observability Engineering* by Charity Majors, Liz Fong-Jones, George Miranda (O'Reilly, recommended)
- **Supplementary**: [Google SRE Book](https://sre.google/sre-book/) (free), DORA metrics
- **Prerequisites**: Production experience, [Cloud Native](../03-cloud-native/)
- **Estimated time**: 2-3 weeks at 8-10 hrs/week

## Key Takeaways

- Observability is NOT monitoring -- it's the ability to ask arbitrary questions about your system without deploying new code
- The three pillars (logs, metrics, traces) are necessary but not sufficient; structured events are the key
- SLOs are contracts with your users; error budgets turn reliability into an engineering resource
- Observability is a cultural practice, not just a tooling choice

---

# Concepts & Techniques

## Core Insight

Monitoring tells you WHEN something is wrong. Observability tells you WHY. In complex distributed systems, you can't predict all failure modes. Observability is the ability to understand internal system state from external outputs -- to debug novel problems you've never seen before.

## 1. Observability vs Monitoring

**Key ideas**:
- **Three pillars**: logs (events), metrics (aggregated numbers), traces (request flow across services). Necessary but insufficient.
- **High cardinality**: many unique values (user IDs, request IDs). Metrics can't handle high cardinality; events can.
- **High dimensionality**: many attributes per event. Structured events with rich context enable ad-hoc querying.
- **Observability = structured events + ad-hoc queries**: the ability to BubbleUp (find what's different about slow requests) is the differentiator.

## 2. Instrumentation

**Key ideas**:
- **OpenTelemetry**: vendor-neutral standard for traces, metrics, logs. SDK + collector + exporters.
- **Spans**: units of work with start time, duration, attributes. Nested spans form traces.
- **Context propagation**: trace ID flows across service boundaries via headers (W3C Trace Context).
- **What to instrument**: service boundaries, database calls, external API calls, queue publish/consume, significant business logic.

**Practical rule**: instrument at the boundaries first; add internal instrumentation only when debugging requires it.

## 3. Debugging with Observability

**Key ideas**:
- **Core analysis loop**: notice a symptom → form hypothesis → query to test → refine or pivot
- **BubbleUp**: given slow requests, what dimensions differ from fast requests? (e.g., all slow requests hit shard 7)
- **Heatmaps**: visualize latency distribution over time; reveal bimodal distributions that p99 metrics hide
- **Unknown-unknowns**: you didn't know this failure mode existed. Monitoring can't alert on what you haven't anticipated; observability lets you explore.

## 4. SLOs & Reliability

**Key definitions**:
- **SLI**: Service Level Indicator -- a measured metric (e.g., % of requests < 200ms)
- **SLO**: Service Level Objective -- target for the SLI (e.g., 99.9% of requests < 200ms)
- **SLA**: Service Level Agreement -- SLO with business consequences (refunds, credits)
- **Error budget**: 100% - SLO = budget for unreliability. If SLO is 99.9%, you have 0.1% error budget (8.7 hours/year)

**Key ideas**:
- **Burn rate alerts**: alert when error budget is being consumed too fast, not when a single request fails
- **Error budget policy**: when budget is exhausted, freeze deployments and focus on reliability. This makes reliability a first-class engineering concern.

## 5. Observability in Practice

**Key ideas**:
- **Deploy and observe**: every deploy should be observed. Canary analysis compares new version metrics against baseline.
- **Testing in production**: dark launches, feature flags, traffic mirroring. Staging environments lie.
- **Progressive delivery**: canary → percentage rollout → full rollout, with automatic rollback on SLO violation.
- **Chaos engineering**: deliberately inject failures (Chaos Monkey, Litmus) to verify resilience. Only do this with good observability in place.

## 6. Organization & Culture

**Key ideas**:
- **Blameless postmortems**: focus on systems and processes, not individuals. "What can we change so this class of failure can't happen again?"
- **Incident response**: structured roles (incident commander, communications lead, subject matter experts). Practice with game days.
- **Observability-driven development**: instrument first, then build features. If you can't observe it, you can't operate it.
- **On-call**: fair rotation, clear escalation paths, runbooks. Alert fatigue is the enemy of effective on-call.

---

## Technique Catalog

| Practice | When to Use | Anti-pattern |
|----------|-------------|-------------|
| Structured events | Always; replace unstructured logs | Unstructured log grep as primary debugging |
| SLOs | Every user-facing service | Alerting on every error instead of budget burn |
| Distributed tracing | Multi-service architectures | Tracing only within a single service |
| Canary deploys | Every production deployment | Big-bang deployments |
| Error budgets | Balancing reliability and velocity | "Always be more reliable" (no budget = no shipping) |
| Blameless postmortems | Every significant incident | Blame individuals, skip action items |

## Company Relevance

| Company | Approach | Focus |
|---------|----------|-------|
| Google | SRE (invented it), SLOs, error budgets | Reliability as engineering discipline |
| Netflix | Chaos engineering (Chaos Monkey), progressive delivery | Resilience through failure injection |
| Anthropic | ML system observability, inference SLOs | GPU utilization, latency percentiles |
| Amazon | Well-Architected Framework, COE (Correction of Errors) | Operational excellence |
| Honeycomb | Observability 2.0 (structured events, BubbleUp) | Debugging novel problems |

## Chapters

<!-- ol:chapters -->
| # | Module | Chapter | Kind | Pass |
|---|---|---|---|---|
| 1 | `obs.00` | [One trace: gateway to engine in Jaeger (tracer)](00-one-trace.md) | practice | 1 |
| 2 | `obs.01` | [Tracing for the serving path](01-tracing-for-the-serving-path.md) | build | 7 |
| 3 | `obs.02` | [Metrics and log correlation](02-metrics-and-log-correlation.md) | practice | 7 |
| 4 | `obs.03` | [SLOs and multi-window burn-rate alerts](03-slos-and-burn-rate-alerts.md) | practice | 7 |
| 5 | `obs.04` | [Dashboards as code for the serving path](04-dashboards-as-code.md) | practice | 7 |
| 6 | `obs.05` | [Observability for the control plane: workflow and activity spans, queue and DLQ panels](05-control-plane-observability.md) | practice | 8 |
<!-- /ol:chapters -->
