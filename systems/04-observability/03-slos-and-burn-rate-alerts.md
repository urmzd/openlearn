<!-- ol:module obs.03 -->
# SLOs and multi-window burn-rate alerts

## Overview

| | |
|---|---|
| **Module** | `obs.03` · practice · ops · Pass 7 · 4 to 6 h |
| **You build** | `deploy/observability/slo.yaml` (three SLOs, six alerts) and the PrometheusRule files rendered from it, `rules/slo-prod.yaml` and `rules/slo-drill.yaml` |
| **Contract** | [`otel/slo.schema.json`](../../course/contracts/otel/slo.schema.json) (the SLO file, the six alert names, both window profiles), [`otel/metrics.yaml`](../../course/contracts/otel/metrics.yaml) (the SLIs' series and bucket bounds), [`helm/observability.md`](../../course/contracts/helm/observability.md) (the rule selector) |
| **Tests** | `course/tests/obs.03/` (`check` runs `artifacts.py`; `promtool` replays course traffic through your rules; section 4) |
| **Needs** | `obs.02` (the series, scraped as `job="<system>-gateway"`) |
| **Used by** | `obs.04` shows burn rates and firing alerts; `ops.01` is detected by `TTFTBudgetBurnFast`; MS-prod loads the rules |
| **Milestone** | MS-prod (SLO rules loaded; load within SLO) |
| **Optional depth** | *The Site Reliability Workbook*, ch. 2 and 5 ("Alerting on SLOs", free online), [Prometheus alerting rules](https://prometheus.io/docs/prometheus/latest/configuration/alerting_rules/) and [unit tests](https://prometheus.io/docs/prometheus/latest/configuration/unit_testing_rules/) (free) |

## Key Takeaways

- An SLO turns "fast enough" into a number: an **SLI** (fraction of good requests), an **objective** (0.99), and a period (30 days). The **error budget** is $1 - \text{objective}$ of the requests.
- The **burn rate** is how many times faster than allowed the budget is being spent. Alerting on burn rate, not on raw error counts, makes one threshold work at any traffic level.
- **Two windows per alert**: the long one proves the burn is big enough to matter, the short one proves it is still happening. A short spike is vetoed by the long window; a page clears minutes after a fix because the short window recovers first.
- **Fast** (14.4x over 1 h and 5 min) pages, **slow** (6x over 6 h and 30 min) opens a ticket. The **drill** profile compresses the windows 12x so a drill pages within minutes.
- A latency SLI is exact only at a histogram bound, and rules that Prometheus never loaded alert no one: the check proves both, and replays traffic through your rules with `promtool`.

## How to work this chapter

```bash
ol start obs.03                       # records the start; there are no stubs
ol tests obs.03                       # read the test catalog first
# write slo.yaml, render rules/slo-prod.yaml and rules/slo-drill.yaml, then:
OL_SMOKE=1 ol check obs.03            # schema, rules, and the promtool scenarios
kubectl apply -f deploy/observability/rules/slo-prod.yaml
ol check obs.03                       # adds: Prometheus at [deploy].prometheus loaded all six alerts
```

---

## 1. Why now

Your stack is deployed, traced (`obs.01`), and scraped (`obs.02`), and MS-prod asks a question none of that answers: is it good enough, and when is it not? "Alert when TTFT p95 goes over 500 ms" fires all night at 3 requests a minute and never fires during a slow, steady leak. The incident drills from here on (`ops.01` first) are graded on whether **your** alerts detect the fault within a time limit, and whether they stay quiet the rest of the time. You need targets everyone agreed on, and alerts derived from them.

## 2. Principles

### 2.1 SLI, objective, error budget

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

The three SLOs of the system (`otel/slo.schema.json`):

| SLO | Good request | Series (`metrics.yaml`) |
|---|---|---|
| `ttft` | first token within `threshold_ms` | `gen_ai_server_time_to_first_token_seconds_bucket{le=T}` over `_count` |
| `tpot` | each later token within `threshold_ms` | `gen_ai_server_time_per_output_token_seconds_bucket{le=T}` over `_count` |
| `availability` | not a 5xx | 1 minus `http_server_request_duration_seconds_count{http_response_status_code=~"5.."}` over all |

### 2.2 Burn rate

Over a 30-day period at burn rate $B$, the budget lasts $30 / B$ days. At $B = 1$ it lasts exactly the period; at $B = 14.4$ it is gone in about 2 days, and one hour at that rate spends $14.4 \cdot 1 / 720 = 2\%$ of it. That is the reasoning behind the factors: a fast alert fires when an hour has cost 2% of the month's budget, a slow one when six hours cost $6 \cdot 6 / 720 = 5\%$.

Burn rate is a ratio of ratios, so the same rule works at 3 requests a second or 3 000. It is also bounded: $B \le 1/\beta$, because $e \le 1$. With $o = 0.95$, $B$ can never exceed 20, so a 14.4x alert needs 72% of requests to be bad. That is why the reference promises $o = 0.99$ for both latency SLOs: $B$ can reach 100, and 14.4% slow requests are enough to page.

### 2.3 Two windows

An alert over one window has to trade detection time against noise. Long windows ignore short spikes but take long to fire and, worse, keep firing for a whole window after the problem is fixed. Short windows react fast and page on every hiccup. The multi-window rule asks both:

$$\text{fire} \iff B(L) > \phi \ \text{ and } \ B(S) > \phi$$

| Alert | $\phi$ | $L$ / $S$ (prod) | $L$ / $S$ (drill) | Severity |
|---|---|---|---|---|
| `<SLO>BudgetBurnFast` | 14.4 | 1 h / 5 min | 5 min / 25 s | page |
| `<SLO>BudgetBurnSlow` | 6 | 6 h / 30 min | 30 min / 150 s | ticket |

$S$ is a twelfth of $L$. The **drill** profile divides both by 12 again, so `ol drill` sees a page within its 300 s budget. Each rule file carries one profile; both may be installed at once, because every series and alert carries an `slo_profile` label.

### 2.4 Rules as code, and as tests

The rules are Prometheus **recording rules** (one error ratio per SLO and window, `slo:sli_error:ratio_rate<w>`) and **alerting rules** (the comparison above), wrapped in a `PrometheusRule` object that kube-prometheus-stack loads only with the label `release: observability`. `promtool test rules` evaluates rules against synthetic series with a fake clock, which is how the course checks behaviour, not just syntax: the check generates traffic from **your** `slo.yaml` (its thresholds and objectives), replays it through **your** rules, and asserts which alerts fire when.

## 3. Worked example by hand

Availability, $o = 0.995$, so $\beta = 0.005$; prod profile; 10 requests a second.

**A total outage.** From 10:00 every request fails. Then $e(5m) = 1$ after 5 minutes and $B(5m) = 1 / 0.005 = 200 > 14.4$. The long window needs $e(1h) > 14.4 \cdot 0.005 = 0.072$: 7.2% of the hour, $0.072 \cdot 60 = 4.3$ minutes. `AvailabilityBudgetBurnFast` fires at about 10:04:20 (plus the rule's `for`).

**A two-minute blip.** From 10:00 to 10:02 every request fails, then all succeed.

| Window | $e$ at 10:02 | $B = e / 0.005$ | Over its $\phi$? |
|---|---|---|---|
| fast short (5 min) | $2/5 = 0.4$ | 80 | yes (14.4) |
| fast long (1 h) | $2/60 = 0.033$ | 6.7 | no (14.4) |
| slow short (30 min) | $2/30 = 0.067$ | 13.3 | yes (6) |
| slow long (6 h) | $2/360 = 0.0056$ | 1.1 | no (6) |

Each short window alone would page; each long window vetoes. Nobody is woken for 2 minutes that cost $2/43\,200 \approx 0.005\%$ of the month. The course's noise scenario is this blip on top of a steady burn at $0.5\beta$.

**Recovery.** After the outage of the first example is fixed at 11:00, $e(1h)$ stays above 0.072 until about 11:56, but $e(5m)$ is 0 from 11:05: the page clears at 11:05, not at noon. A long-window-only alert keeps paging for the hour, which is the course's reset test.

**TTFT, $o = 0.99$.** $\beta = 0.01$, fast threshold $e > 0.144$. The course's fast scenario burns at twice that, $e = 0.288$ ($B = 28.8$), for 75 minutes; it expects the page 70 minutes in, nothing from TPOT or availability, and no page 8 minutes after the burn stops. Its slow scenario burns at $1.5 \cdot 6 = 9$ times the budget: a ticket, and no page, because $9 < 14.4$.

## 4. The interface

`deploy/observability/slo.yaml` (validated against `otel/slo.schema.json`):

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

`deploy/observability/rules/slo-prod.yaml` and `rules/slo-drill.yaml`, one rendered alert:

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

Write the rules by hand or render them from `slo.yaml` with a small script (the reference does). The synthetic series carry `job="<system>-gateway"` (and copies under `<system>-engine`), `namespace`, `tl_engine_role`, `gen_ai_operation_name`, `gen_ai_request_model`, and for HTTP `http_request_method`, `http_route`, `http_response_status_code`; select on any of them. Write `le` as Prometheus stores it (`"0.5"`, `"0.05"`, `"1.0"`). Rule groups evaluate every minute or faster (prod) and every 10 s or faster (drill).

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_slo_file_matches_the_schema` | conformance | `slo.yaml` validates; profile `prod` | `ol drill`, MS-prod, and `obs.04` read it by these names |
| `test_latency_thresholds_are_bucket_bounds` | boundary | both thresholds are bounds of their histogram | an exact SLI (section 2.1 of `obs.02`) |
| `test_targets_are_no_looser_than_the_course_defaults` | boundary | TTFT at most 500 ms, TPOT at most 60 ms | stricter is allowed, looser never |
| `test_rule_files_are_selected_prometheus_rules` | conformance | `PrometheusRule`, label `release: observability` | the stack loads them |
| `test_rules_define_the_six_alerts` | unit | each name exactly once per file, severity from `slo.yaml`, no extras | detection by name, page or ticket |
| `test_rules_use_the_profile_windows` | unit | the four windows of each profile appear as range selectors | the two-window rule; `ol drill`'s gate |
| `test_rule_groups_evaluate_often_enough` | boundary | group interval at most 60 s (prod), 10 s (drill) | short windows need frequent evaluation |
| `test_promtool_accepts_the_rules` | conformance | `promtool check rules` passes for both files | no syntax error reaches the cluster |
| `test_fast_burn_pages_then_resets` | conformance | per SLO: quiet before, page during, no other SLO's alerts, clear after (section 3), both profiles | the page you want |
| `test_slow_burn_tickets_without_paging` | conformance | per SLO, 9x: Slow fires, Fast does not | tickets are not pages |
| `test_noise_and_short_spikes_stay_quiet` | conformance | $0.5\beta$ with a 2-minute spike: nothing fires, at 7 points | the pages you do not want |
| `test_rules_loaded_in_prometheus` | conformance | on kind: `/api/v1/rules` lists all six alerts | the selector matched |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. The `PrometheusRule` lacks `release: observability` | `kubectl get prometheusrule` lists it; Prometheus has no such alert; nothing ever pages | `test_rule_files_are_selected_prometheus_rules`, `test_rules_loaded_in_prometheus` |
| 2. One window per alert (only the short one) | a page for every 2-minute blip; people learn to ignore pages | `test_noise_and_short_spikes_stay_quiet` |
| 3. A latency threshold between bounds, or a rule using another bound than `slo.yaml` | the SLI counts the wrong requests; the scenario built from your file does not page | `test_latency_thresholds_are_bucket_bounds`, `test_fast_burn_pages_then_resets` |
| 4. The drill profile with prod windows | `ol drill start` refuses; or the drill page arrives after the drill is over | `test_rules_use_the_profile_windows` |
| 5. Only the long window | the page keeps firing for an hour after the fix | `test_fast_burn_pages_then_resets` (the reset assertion) |
| 6. Factors swapped, or the budget written as the objective (`14.4 * 0.99`) | a slow burn pages at night, or no burn ever pages | `test_slow_burn_tickets_without_paging`, `test_fast_burn_pages_then_resets` |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `obs.02` | the histograms and the 5xx counts, under the scrape jobs it named |
| Forward | `obs.04` | burn-rate stats and the firing-alerts table read `slo:sli_error:*` and `ALERTS` |
| Forward | `ops.01` | `TTFTBudgetBurnFast` under the drill profile detects the killed decode pod |
| Forward | `dur.12` | the release workflow checks the burn rate during a canary before promoting |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `slo.yaml` and a render script | Sloth, Pyrra, OpenSLO | SLO specs compiled to rules and dashboards, error-budget reports | [Sloth](https://sloth.dev/), [OpenSLO](https://openslo.com/) |
| request-based SLIs | window-based SLIs ("good minutes") | robust to traffic swings, closer to a user's experience of an outage | *Implementing Service Level Objectives*, ch. 3 |
| two fixed profiles | burn-rate alerts with a low-traffic fallback | a minimum request count so 1 failure in 3 requests does not page | SRE Workbook ch. 5, "Low-traffic services" |
| promtool unit tests | rules CI on every change | the same tests in your CI (`dep.05`), with recorded production series | `promtool test rules` |
| severity labels | Alertmanager routing, inhibition, silences | pages to on-call, tickets to a queue, drill pages to a drill channel | [Alertmanager](https://prometheus.io/docs/alerting/latest/alertmanager/) |
