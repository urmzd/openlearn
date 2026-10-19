# Observability stack: pinned charts and the rule selector
<!-- chapter: content/tracks/infrastructure/01-containers-kubernetes/03-helm-charts-and-observability-stack.md -->

<!-- modules: dep.00 (Jaeger, Pass 1), dep.02 and dep.03 (Pass 7 stack), dep.06 (KEDA), obs.01 to obs.05 -->

The system deploys upstream charts for tracing, metrics, and dashboards and pins them in `deploy/observability/Chart.lock`. These versions are the ones the course tests against; a newer version is a dependency upgrade (craft.15, drill ops.06), never a silent drift. Versions were taken from each project's official Helm repository index, choosing the newest release published at least two weeks before contract 0.2.0 (2026-10-09).

| Chart | Repository | Version | App version | From |
|---|---|---|---|---|
| `jaeger` | `https://jaegertracing.github.io/helm-charts` | `4.14.0` | `2.21.0` | Pass 1 (dep.00), all-in-one |
| `opentelemetry-collector` | `https://open-telemetry.github.io/opentelemetry-helm-charts` | `0.173.1` | `0.160.0` | Pass 7 (dep.03, obs.01) |
| `kube-prometheus-stack` | `https://prometheus-community.github.io/helm-charts` | `91.5.2` | `v0.94.1` | Pass 7 (dep.03, obs.02) |
| `tempo` | `https://grafana.github.io/helm-charts` | `1.24.4` | `2.9.0` | Pass 7 (obs.01) |
| `keda` | `https://kedacore.github.io/charts` | `2.21.0` | `2.21.0` | Pass 8 (dep.06) |

## Fixed names and ports

| Thing | Value |
|---|---|
| Namespace and Helm release of the stack | `observability` (one release name per chart: `jaeger`, `otel-collector`, `observability` for kube-prometheus-stack, `tempo`, `keda` in namespace `keda`) |
| Collector Service | `otel-collector.observability`, OTLP gRPC `4317`, OTLP/HTTP `4318` (the `[otel].endpoint` of runtime.toml) |
| Jaeger query (Pass 1) | NodePort `30686` |
| Prometheus | NodePort `30090` |
| Grafana | NodePort `30300` |
| Tempo query API | port `3200`, NodePort `30320` |

## The PrometheusRule selector

kube-prometheus-stack's Prometheus loads only the `PrometheusRule` objects that match its rule selector. The course leaves the chart's default (`prometheus.prometheusSpec.ruleSelectorNilUsesHelmValues: true`), which selects rules labelled with the stack's release name. So every rule file the system renders from `slo.yaml` ([`otel/slo.schema.json`](../otel/slo.schema.json)) is a `PrometheusRule` with:

```yaml
metadata:
  labels:
    release: observability
```

The same holds for `ServiceMonitor` objects that scrape the services' health ports (`:9464/metrics`): label `release: observability`. A rule without the label is valid YAML and silently ignored, which is exactly the failure obs.03's check catches.

## Pipelines

- **Traces:** every service exports OTLP to the collector, which sends them to Tempo (Pass 7 on) or Jaeger (Pass 1).
- **Service metrics:** Prometheus scrapes each Go and Rust service's health port (`:9464/metrics`) through a `ServiceMonitor`; those services do not also push metrics over OTLP, so no series exists twice.
- **Python metrics:** training and corpus subprocesses cannot be scraped (D9), so they push OTLP metrics to the collector, whose `prometheus` exporter (scraped by its own `ServiceMonitor`) translates the names to the `prometheus` names of [`otel/metrics.yaml`](../otel/metrics.yaml).
