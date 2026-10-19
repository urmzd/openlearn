# SLOs and multi-window burn-rate alerts

The stack is now deployed, traced ([Tracing for the serving path](02-tracing-for-the-serving-path.md)), and scraped ([Metrics and log correlation](03-metrics-and-log-correlation.md)), and a production system must answer a question none of that answers: is it good enough, and when is it not? The obvious alert, "page when TTFT p95 goes over 500 ms", fires all night at 3 requests a minute and never fires during a slow, steady leak. The incident drills of the next chapter, starting with [A decode worker dies mid-stream](../05-incident-response-and-chaos/02-kill-decode.md), only work if the alerts detect the fault within a time limit and stay quiet the rest of the time. That needs targets everyone agreed on, and alerts derived from them.

This section turns "fast enough" into numbers and those numbers into alerts. You will define three SLOs over the series the gateway already serves, derive burn-rate thresholds from them, and render a pair of Prometheus rule files, one for production and one compressed for drills, from a single SLO file. Then you will prove the rules behave correctly by replaying synthetic traffic through them with `promtool`.

The solution is [`deploy/observability/slo.yaml`](../../../../course/solutions/entry/deploy/observability/slo.yaml) (three SLOs, six alerts), the renderer [`render_slo_rules.py`](../../../../course/solutions/entry/deploy/observability/render_slo_rules.py) (`error_ratio`, `render`, the `PROFILES` table), and the PrometheusRule files it writes, [`rules/slo-prod.yaml`](../../../../course/solutions/entry/deploy/observability/rules/slo-prod.yaml) and [`rules/slo-drill.yaml`](../../../../course/solutions/entry/deploy/observability/rules/slo-drill.yaml). The contracts are [`otel/slo.schema.json`](../../../../course/contracts/otel/slo.schema.json) (the SLO file, the six alert names, both window profiles), [`otel/metrics.yaml`](../../../../course/contracts/otel/metrics.yaml) (the SLIs' series and bucket bounds), and [`helm/observability.md`](../../../../course/contracts/helm/observability.md) (the rule selector).

> **Learning objectives**
> - Define an SLO as an SLI, an objective, and a period, and compute its error budget and burn rate.
> - Derive the 14.4x and 6x burn-rate factors from the fraction of budget an hour or six hours may cost.
> - Explain why each alert needs a long and a short window, and work an outage, a blip, and a recovery through both by hand.
> - Render recording and alerting rules for two window profiles from one SLO file, and keep latency SLIs exact by using bucket bounds.
> - Test alert behaviour, not just syntax, with `promtool test rules`.

## SLI, objective, and error budget

An SLO has three parts: an **SLI** (the fraction of good requests), an **objective** (say 0.99), and a period (30 days). The rest of this section uses a small vocabulary:

| Symbol | Meaning | Type |
|---|---|---|
| $G(w)$, $N(w)$ | good requests and all requests over a window $w$ | counts |
| $\text{SLI}(w) = G(w) / N(w)$ | the fraction of good requests | in $[0, 1]$ |
| $o$ | the **objective**, the SLI promised over the period | 0.95 to 1 (the schema's floor) |
| $\beta = 1 - o$ | the **error budget**: the fraction of requests allowed to be bad | e.g. 0.01 |
| $e(w) = 1 - \text{SLI}(w)$ | the **error ratio** over $w$ | in $[0, 1]$ |
| $B(w) = e(w) / \beta$ | the **burn rate** over $w$ | $\ge 0$; 1 spends the budget exactly over the period |
| $\phi$ | an alert's burn-rate factor (14.4 fast, 6 slow) | constant |
| $L$, $S$ | an alert's long and short windows | durations |

The system has three SLOs, defined in `otel/slo.schema.json`. Each one says what a good request is and which series measure it:

| SLO | Good request | Series (`metrics.yaml`) |
|---|---|---|
| `ttft` | first token within `threshold_ms` | `gen_ai_server_time_to_first_token_seconds_bucket{le=T}` over `_count` |
| `tpot` | each later token within `threshold_ms` | `gen_ai_server_time_per_output_token_seconds_bucket{le=T}` over `_count` |
| `availability` | not a 5xx | 1 minus `http_server_request_duration_seconds_count{http_response_status_code=~"5.."}` over all |

## Burn rate

The burn rate says how many times faster than allowed the budget is being spent. Over a 30-day period at burn rate $B$, the budget lasts $30 / B$ days. At $B = 1$ it lasts exactly the period; at $B = 14.4$ it is gone in about 2 days, and one hour at that rate spends $14.4 \cdot 1 / 720 = 2\%$ of it. That is the reasoning behind the factors: a fast alert fires when an hour has cost 2% of the month's budget, a slow one when six hours cost $6 \cdot 6 / 720 = 5\%$.

Burn rate is a ratio of ratios, so the same rule works at 3 requests a second or 3 000. Alerting on burn rate, not on raw error counts, is what makes one threshold work at any traffic level.

Burn rate is also bounded: $B \le 1/\beta$, because $e \le 1$. With $o = 0.95$, $B$ can never exceed 20, so a 14.4x alert needs 72% of requests to be bad. That is why the solution promises $o = 0.99$ for both latency SLOs: $B$ can reach 100, and 14.4% slow requests are enough to page.

## Two windows per alert

Knowing the threshold, the remaining question is over what window to measure the burn. An alert over one window has to trade detection time against noise. Long windows ignore short spikes but take long to fire and, worse, keep firing for a whole window after the problem is fixed. Short windows react fast and page on every hiccup. The multi-window rule asks both:

$$\text{fire} \iff B(L) > \phi \ \text{ and } \ B(S) > \phi$$

The long window proves the burn is big enough to matter; the short one proves it is still happening. A short spike is vetoed by the long window, and a page clears minutes after a fix because the short window recovers first. The system has two kinds of alert per SLO:

| Alert | $\phi$ | $L$ / $S$ (prod) | $L$ / $S$ (drill) | Severity |
|---|---|---|---|---|
| `<SLO>BudgetBurnFast` | 14.4 | 1 h / 5 min | 5 min / 25 s | page |
| `<SLO>BudgetBurnSlow` | 6 | 6 h / 30 min | 30 min / 150 s | ticket |

$S$ is a twelfth of $L$. The **drill** profile divides both by 12 again, so a drill sees a page within its 300 s budget. Each rule file carries one profile; both may be installed at once, because every series and alert carries an `slo_profile` label.

## Rules as code, and as tests

These conditions become Prometheus **recording rules** (one error ratio per SLO and window, `slo:sli_error:ratio_rate<w>`) and **alerting rules** (the comparison above), wrapped in a `PrometheusRule` object that kube-prometheus-stack loads only with the label `release: observability`. Rules that Prometheus never loaded alert no one.

Loading is necessary, not sufficient. `promtool test rules` evaluates rules against synthetic series with a fake clock, which is how to check behaviour rather than just syntax: generate traffic from `slo.yaml` (its thresholds and objectives), replay it through the rendered rules, and assert which alerts fire when.

## Working it by hand

Start with availability at $o = 0.995$, so $\beta = 0.005$, under the prod profile at 10 requests a second.

**A total outage.** From 10:00 every request fails. After 5 minutes $e(5m) = 1$ and $B(5m) = 1 / 0.005 = 200 > 14.4$. The long window needs $e(1h) > 14.4 \cdot 0.005 = 0.072$: 7.2% of the hour, or $0.072 \cdot 60 = 4.3$ minutes. `AvailabilityBudgetBurnFast` fires at about 10:04:20, plus the rule's `for`.

**A two-minute blip.** From 10:00 to 10:02 every request fails, then all succeed. At 10:02 the four windows read:

| Window | $e$ at 10:02 | $B = e / 0.005$ | Over its $\phi$? |
|---|---|---|---|
| fast short (5 min) | $2/5 = 0.4$ | 80 | yes (14.4) |
| fast long (1 h) | $2/60 = 0.033$ | 6.7 | no (14.4) |
| slow short (30 min) | $2/30 = 0.067$ | 13.3 | yes (6) |
| slow long (6 h) | $2/360 = 0.0056$ | 1.1 | no (6) |

Each short window alone would page; each long window vetoes. Nobody is woken for 2 minutes that cost $2/43\,200 \approx 0.005\%$ of the month. A good noise scenario for `promtool` is this blip on top of a steady burn at $0.5\beta$.

**Recovery.** After the outage of the first example is fixed at 11:00, $e(1h)$ stays above 0.072 until about 11:56, but $e(5m)$ is 0 from 11:05, so the page clears at 11:05, not at noon. A long-window-only alert keeps paging for the hour, which is what a reset scenario catches.

**TTFT at $o = 0.99$.** Here $\beta = 0.01$ and the fast threshold is $e > 0.144$. A fast scenario burns at twice that, $e = 0.288$ ($B = 28.8$), for 75 minutes. A correct rule set pages 70 minutes in, fires nothing from TPOT or availability, and no longer pages 8 minutes after the burn stops. A slow scenario burns at $1.5 \cdot 6 = 9$ times the budget: a ticket, and no page, because $9 < 14.4$.

## The SLO file and the rule shape

The source of truth is `deploy/observability/slo.yaml`, validated against `otel/slo.schema.json`:

```yaml
version: 1
profile: prod
windows:
  fast: {long: 1h, short: 5m, burn_rate: 14.4}
  slow: {long: 6h, short: 30m, burn_rate: 6}
slos:
  ttft: {threshold_ms: 500, objective: 0.99}   # a TTFT bucket bound; 500 ms or stricter
  tpot: {threshold_ms: 50, objective: 0.99}    # 60 ms is not a TPOT bound: 50 (stricter)
  availability: {objective: 0.995, period: 30d}
alerts:
  TTFTBudgetBurnFast: {slo: ttft, window: fast, severity: page}
  # ... the other five, as the schema names them
```

The rules go to `deploy/observability/rules/slo-prod.yaml` and `rules/slo-drill.yaml`. One rendered alert looks like this:

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata: {name: forge-slo-prod, namespace: observability, labels: {release: observability}}
spec:
  groups:
    - name: forge-slo-prod
      interval: 30s
      rules:
        - record: slo:sli_error:ratio_rate5m
          expr: 1 - (sum(rate(gen_ai_server_time_to_first_token_seconds_bucket{job="forge-gateway",le="0.5"}[5m]))
                   / sum(rate(gen_ai_server_time_to_first_token_seconds_count{job="forge-gateway"}[5m])))
          labels: {slo: ttft, slo_profile: prod}
        # ... one per SLO and window
        - alert: TTFTBudgetBurnFast
          expr: slo:sli_error:ratio_rate1h{slo="ttft",slo_profile="prod"} > (14.4 * 0.01)
                and slo:sli_error:ratio_rate5m{slo="ttft",slo_profile="prod"} > (14.4 * 0.01)
          for: 2m
          labels: {severity: page, slo: ttft, slo_profile: prod}
          annotations: {runbook: docs/runbooks/TTFTBudgetBurnFast.md}
```

The rules could be written by hand; the solution renders them from `slo.yaml` with a small script. The synthetic `promtool` series carry `job="<system>-gateway"` (and copies under `<system>-engine`), `namespace`, `tl_engine_role`, `gen_ai_operation_name`, `gen_ai_request_model`, and for HTTP `http_request_method`, `http_route`, and `http_response_status_code`, so rules may select on any of them. Write `le` as Prometheus stores it (`"0.5"`, `"0.05"`, `"1.0"`). Rule groups evaluate every minute or faster in prod and every 10 s or faster in the drill profile. The solution is used like this:

```bash
python3 deploy/observability/render_slo_rules.py --system forge   # writes rules/slo-prod.yaml and rules/slo-drill.yaml
promtool check rules deploy/observability/rules/slo-*.yaml
kubectl apply -f deploy/observability/rules/slo-prod.yaml
```

## Building it step by step

Seven decisions produce the solution. First, choose SLIs where the user is: TTFT, TPOT, and 5xx ratio, all measured at the gateway (`job="<system>-gateway"`), because that is the latency and the error a client sees.

Second, make each latency SLI exact. A latency SLI is exact only at a histogram bound, as the previous section showed, so the threshold must be a bucket bound. 500 ms is a TTFT bound. 60 ms is not a TPOT bound, so the solution picks the next stricter bound, 50 ms.

Third, choose objectives that can page. Since $B \le 1/\beta$, a 14.4x alert must be reachable well before every request is bad, so the latency objectives are 0.99 ($B$ up to 100) and availability is 0.995 ($B$ up to 200).

Fourth, compute one error ratio per SLO and window. The alert expression compares ratios over two windows; computing each ratio once as a recording rule makes the six alerts cheap and lets dashboards reuse the same series.

Fifth, compare the ratio, not the burn rate. Because $B(w) > \phi \iff e(w) > \phi \beta$, the threshold is a constant $\phi(1 - o)$ written into the alert, and no division happens at evaluation.

Sixth, render two profiles from one source. Prod and drill differ only in windows, evaluation interval, and `for`; everything else must be identical, so both are rendered from one `slo.yaml`, and every series and alert carries `slo_profile` so both can be loaded together.

Seventh, remember that rules are only real once loaded: the `PrometheusRule` must carry `release: observability`, the stack's rule selector.

### The SLO file

The SLO file is the single source, and its comments record why each number was chosen:

```yaml
# Latency SLOs are ratio SLIs over the gateway's histograms: the fraction of
# requests whose TTFT (TPOT) is at most threshold_ms. Each threshold is a
# bucket boundary of otel/metrics.yaml, so the SLI is exact: 500 ms for TTFT
# and 50 ms for TPOT (stricter than the 60 ms default, which is not a
# bucket boundary). Objectives are 0.99 for both: at the schema's floor of
# 0.95 the largest possible burn rate is 1 / 0.05 = 20, so a fast-burn
# alert at 14.4x would need 72% of requests to be slow before it could fire.
version: 1
profile: prod
windows:
  fast: {long: 1h, short: 5m, burn_rate: 14.4}
  slow: {long: 6h, short: 30m, burn_rate: 6}
slos:
  ttft: {threshold_ms: 500, objective: 0.99}
  tpot: {threshold_ms: 50, objective: 0.99}
  availability: {objective: 0.995, period: 30d}
alerts:
  TTFTBudgetBurnFast: {slo: ttft, window: fast, severity: page}
  TTFTBudgetBurnSlow: {slo: ttft, window: slow, severity: ticket}
  # ... TPOT and Availability, Fast (page) and Slow (ticket)
```

### The profiles

The profiles live in the renderer, because they are a property of where the rules run, not of the SLO:

```python
PROFILES = {
    "prod": {
        "fast": ("1h", "5m"),
        "slow": ("6h", "30m"),
        "interval": "30s",                      # at most 60 s
        "for": {"fast": "2m", "slow": "15m"},
    },
    # the same ratios with windows compressed 12x; 2m30s is spelled 150s
    "drill": {
        "fast": ("5m", "25s"),
        "slow": ("30m", "150s"),
        "interval": "5s",                       # at most 10 s: a 25 s window needs several evaluations
        "for": {"fast": "10s", "slow": "75s"},
    },
}
```

`for` is a small hold so that a single evaluation over the threshold does not page. It is short relative to the short window, so it adds little to detection time.

### The error ratio

The error ratio is the bad fraction $e(w)$. For latency it is one minus the good fraction read at the threshold's bucket; for availability it is the 5xx fraction directly:

```python
def fmt_le(seconds: float) -> str:
    """The `le` label value Prometheus stores for a bucket bound."""
    return repr(float(seconds))  # 0.5 -> "0.5", 0.05 -> "0.05", 1 -> "1.0"


def error_ratio(slo: str, spec: dict, window: str, system: str) -> str:
    sel = f'job="{system}-gateway"'   # measured at the gateway, where the user is
    if slo == "availability":
        return (
            f'sum(rate({HTTP}_count{{{sel},http_response_status_code=~"5.."}}[{window}]))\n'
            f"/\nsum(rate({HTTP}_count{{{sel}}}[{window}]))"
        )
    metric = TTFT if slo == "ttft" else TPOT
    le = fmt_le(spec["threshold_ms"] / 1000)  # 500 ms -> le="0.5": an exact bound
    return (
        f'1 - (\n  sum(rate({metric}_bucket{{{sel},le="{le}"}}[{window}]))\n'
        f"  /\n  sum(rate({metric}_count{{{sel}}}[{window}]))\n)"
    )
```

The `sum(...)` before the division aggregates across pods, models, and roles, so the SLI is the whole service's fraction, not an average of per-pod fractions, which would weight an idle pod like a busy one.

### Rendering

Rendering writes one recording rule per SLO per window, labelled with `slo` and `slo_profile`, then one alert per entry of `alerts`:

```python
    factors = {k: float(v["burn_rate"]) for k, v in doc["windows"].items()}
    for name, a in doc["alerts"].items():
        slo, kind, sev = a["slo"], a["window"], a["severity"]
        budget = round(1 - float(doc["slos"][slo]["objective"]), 6)  # beta = 1 - o, never o itself
        long_, short = p[kind]
        thr = f"({factors[kind]:g} * {budget:g})"                  # e > phi * beta  <=>  B > phi
        out += [
            f"        - alert: {name}",
            "          expr: |",
            f'            slo:sli_error:ratio_rate{long_}{{slo="{slo}",slo_profile="{profile}"}} > {thr}',
            "            and",                                      # both windows
            f'            slo:sli_error:ratio_rate{short}{{slo="{slo}",slo_profile="{profile}"}} > {thr}',
            f"          for: {p['for'][kind]}",
            # ... labels: severity, slo, slo_profile; annotations: summary and
            # runbook: docs/runbooks/<alert name>.md
        ]
```

`round(..., 6)` strips float noise (`1 - 0.995` is `0.0050000000000000044` in binary floating point) before `:g` prints the budget as `0.005`. The header of each file sets `kind: PrometheusRule`, `labels: release: observability`, and the group's `interval` from the profile.

One rendered drill recording rule, as it appears in `rules/slo-drill.yaml`:

```yaml
        - record: slo:sli_error:ratio_rate25s
          expr: |
            1 - (
              sum(rate(gen_ai_server_time_to_first_token_seconds_bucket{job="forge-gateway",le="0.5"}[25s]))
              /
              sum(rate(gen_ai_server_time_to_first_token_seconds_count{job="forge-gateway"}[25s]))
            )
          labels:
            slo: ttft
            slo_profile: drill
```

The full solution is in [`slo.yaml`](../../../../course/solutions/entry/deploy/observability/slo.yaml), [`render_slo_rules.py`](../../../../course/solutions/entry/deploy/observability/render_slo_rules.py), [`rules/slo-prod.yaml`](../../../../course/solutions/entry/deploy/observability/rules/slo-prod.yaml), and [`rules/slo-drill.yaml`](../../../../course/solutions/entry/deploy/observability/rules/slo-drill.yaml).

### Why this design

Three SLOs times four windows is 12 recording rules per profile, each a `rate` over at most two series families; the six alerts only compare recorded series. Evaluation cost does not grow with the number of alerts.

Rendering beats hand-written YAML because there are twelve near-identical expressions per profile and two profiles. A hand edit that changes a threshold in one place and not another is the classic SLO bug; one source and a renderer make the files consistent by construction. The renderer is stdlib-only (a small parser for exactly this file's shape), so it runs anywhere.

Two windows beat one window with a long `for`, because a long `for` delays detection by the same amount and still keeps firing after a fix as long as the one window stays high. The short window is what makes the alert clear minutes after recovery.

Request-based SLIs are directly computable from the histograms the services already serve. The accepted cost is noise at very low traffic, where one bad request in three is 33%; the fix is discussed at the end of this section.

## What can go wrong

A correct solution is checked case by case. `slo.yaml` must validate against the schema with profile `prod`, because the drills, the production system, and the dashboards read it by these names. Both latency thresholds must be bounds of their histogram, so the SLIs are exact, and no looser than the defaults: TTFT at most 500 ms, TPOT at most 60 ms (stricter is allowed, looser never). Both rule files must be `PrometheusRule` objects labelled `release: observability`, so the stack loads them. Each of the six alert names must appear exactly once per file, with the severity from `slo.yaml` and no extras, because detection is by name and severity decides page or ticket. The four windows of each profile must appear as range selectors, since drills look for the drill windows. Group intervals must be at most 60 s (prod) and 10 s (drill), because short windows need frequent evaluation. And `promtool check rules` must accept both files, so no syntax error reaches the cluster.

Behaviour is tested by replay. For each SLO and both profiles, a fast burn must be quiet before, page during, fire none of the other SLOs' alerts, and clear after, as in the worked example. A slow burn at 9x must fire Slow and not Fast, because tickets are not pages. Noise at $0.5\beta$ with a 2-minute spike must fire nothing, checked at 7 points in time; these are the pages you do not want. Finally, on kind, `/api/v1/rules` must list all six alerts, which proves the selector matched.

The usual mistakes, with their symptoms:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| The `PrometheusRule` lacks `release: observability` | `kubectl get prometheusrule` lists it; Prometheus has no such alert; nothing ever pages | `render` writes the label into every file's metadata |
| One window per alert (only the short one) | a page for every 2-minute blip; people learn to ignore pages | every alert is `long > thr and short > thr` |
| A latency threshold between bounds, or a rule using another bound than `slo.yaml` | the SLI counts the wrong requests; a scenario built from the SLO file does not page | thresholds are bounds (500 ms, 50 ms), and `le` is derived from `threshold_ms` by `fmt_le`, never typed twice |
| The drill profile with prod windows | the drill refuses to start; or the drill page arrives after the drill is over | `PROFILES["drill"]` holds the 12x-compressed windows |
| Only the long window | the page keeps firing for an hour after the fix | the short window clears the alert minutes after recovery |
| Factors swapped, or the budget written as the objective (`14.4 * 0.99`) | a slow burn pages at night, or no burn ever pages | factors come from `windows.<kind>.burn_rate`; the threshold uses `1 - objective` |

## Where this leads

These rules read the histograms and 5xx counts of [Metrics and log correlation](03-metrics-and-log-correlation.md), under the scrape jobs it named. Going forward, [Dashboards as code for the serving path](05-dashboards-as-code.md) shows burn-rate stats and a firing-alerts table that read `slo:sli_error:*` and `ALERTS`. In [A decode worker dies mid-stream](../05-incident-response-and-chaos/02-kill-decode.md), `TTFTBudgetBurnFast` under the drill profile detects the killed decode pod. And the release workflow in [ModelRelease: gates, canary, burn check, and rollback](../../ai-platform-engineering/05-durable-orchestration-and-workers/12-model-release.md), in *AI Platform Engineering*, checks the burn rate during a canary before promoting.

Each piece here has a production equivalent. `slo.yaml` and a render script correspond to Sloth, Pyrra, and OpenSLO, which compile SLO specs to rules and dashboards and produce error-budget reports. Request-based SLIs can give way to window-based SLIs ("good minutes"), which are robust to traffic swings and closer to a user's experience of an outage. Two fixed profiles grow into burn-rate alerts with a low-traffic fallback: a minimum request count so that 1 failure in 3 requests does not page. The `promtool` unit tests belong in rules CI on every change, as in [CI for the platform repo](../../infrastructure/01-containers-kubernetes/06-ci-for-your-repo.md), with recorded production series. And severity labels feed Alertmanager routing, inhibition, and silences: pages to on-call, tickets to a queue, drill pages to a drill channel.

### Further reading

- *The Site Reliability Workbook*, chapters 2 and 5 ("Alerting on SLOs", free online), including "Low-traffic services"
- [Prometheus alerting rules](https://prometheus.io/docs/prometheus/latest/configuration/alerting_rules/) and [unit tests](https://prometheus.io/docs/prometheus/latest/configuration/unit_testing_rules/) (free)
- [Sloth](https://sloth.dev/) and [OpenSLO](https://openslo.com/)
- Alex Hidalgo, *Implementing Service Level Objectives* (O'Reilly), chapter 3
- [Alertmanager](https://prometheus.io/docs/alerting/latest/alertmanager/)

## Key terms

- **SLI**: the fraction of good requests over a window.
- **SLO objective**: the SLI value promised over a period, such as 0.99 over 30 days.
- **error budget**: $1 - \text{objective}$, the fraction of requests allowed to be bad.
- **burn rate**: the error ratio divided by the error budget; 1 spends the budget exactly over the period.
- **multi-window alert**: an alert that fires only when the burn rate exceeds its factor over both a long and a short window.
- **recording rule**: a Prometheus rule that precomputes an expression as a new series.
- **window profile**: a set of alert windows, evaluation interval, and `for` durations; prod and a 12x-compressed drill profile.

## Summary

- An SLO turns "fast enough" into a number: an SLI, an objective, and a period; the error budget is $1 - \text{objective}$ of the requests.
- The burn rate is how many times faster than allowed the budget is being spent; alerting on it makes one threshold work at any traffic level.
- Each alert uses two windows: the long one proves the burn matters, the short one proves it is still happening, so blips are vetoed and pages clear minutes after a fix.
- Fast (14.4x over 1 h and 5 min) pages, slow (6x over 6 h and 30 min) opens a ticket; the drill profile compresses the windows 12x so a drill pages within minutes.
- A latency SLI is exact only at a histogram bound, rules that Prometheus never loaded alert no one, and `promtool test rules` proves the rules fire when they should and only then.

## Review questions

1. Why is the alert threshold `14.4 * 0.01` and not `14.4 * 0.99`?
   <details><summary>Answer</summary>The recorded series is the error ratio $e$; the burn rate is $e / \beta$ with $\beta = 1 - o = 0.01$. $B > 14.4 \iff e > 14.4 \beta = 0.144$. Multiplying by the objective (0.99) gives 14.26, which an error ratio in $[0, 1]$ can never exceed, so nothing would ever page.</details>
2. Why is the TPOT threshold 50 ms when the default target is 60 ms?
   <details><summary>Answer</summary>60 ms is not a bucket bound of the TPOT histogram, so no counter says how many requests were at most 60 ms. The nearest exact bound that is no looser is 50 ms.</details>
3. In the 2-minute blip of the worked example, which condition stops the fast alert from paging?
   <details><summary>Answer</summary>The long window: $e(1h) = 2/60 \approx 0.033$, below $14.4 \cdot 0.005 = 0.072$, while the short window alone ($e = 0.4$) would page.</details>
4. Why does every recording rule and alert carry `slo_profile`?
   <details><summary>Answer</summary>Both profiles record a `ratio_rate5m` series for each SLO (the prod short fast window and the drill long fast window are both 5 minutes). Without the label, the two would be the same series from two rules and collide; with it, both files can be installed at once and routed separately.</details>
5. Why does the error ratio sum across pods before dividing?
   <details><summary>Answer</summary>The SLI is a fraction of the service's requests. Summing numerator and denominator first weights each pod by its traffic; averaging per-pod ratios would let an idle pod with one slow request count as much as a busy one.</details>
