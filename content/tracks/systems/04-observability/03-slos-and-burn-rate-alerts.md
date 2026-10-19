# SLOs and multi-window burn-rate alerts

## Overview

| | |
|---|---|
| **Module** | `obs.03` · practice · ops · Pass 7 · 4 to 6 h |
| **Solution** | [`deploy/observability/slo.yaml`](../../../../course/solutions/entry/deploy/observability/slo.yaml) (three SLOs, six alerts), the renderer [`render_slo_rules.py`](../../../../course/solutions/entry/deploy/observability/render_slo_rules.py) (`error_ratio`, `render`, the `PROFILES` table), and the PrometheusRule files it writes, [`rules/slo-prod.yaml`](../../../../course/solutions/entry/deploy/observability/rules/slo-prod.yaml) and [`rules/slo-drill.yaml`](../../../../course/solutions/entry/deploy/observability/rules/slo-drill.yaml) |
| **Contract** | [`otel/slo.schema.json`](../../../../course/contracts/otel/slo.schema.json) (the SLO file, the six alert names, both window profiles), [`otel/metrics.yaml`](../../../../course/contracts/otel/metrics.yaml) (the SLIs' series and bucket bounds), [`helm/observability.md`](../../../../course/contracts/helm/observability.md) (the rule selector) |
| **Needs** | `obs.02` (the series, scraped as `job="<system>-gateway"`) |
| **Used by** | `obs.04` shows burn rates and firing alerts; `ops.01` is detected by `TTFTBudgetBurnFast`; the production system loads the rules and keeps its load within SLO |
| **Optional depth** | *The Site Reliability Workbook*, ch. 2 and 5 ("Alerting on SLOs", free online), [Prometheus alerting rules](https://prometheus.io/docs/prometheus/latest/configuration/alerting_rules/) and [unit tests](https://prometheus.io/docs/prometheus/latest/configuration/unit_testing_rules/) (free) |

## Key Takeaways

- An SLO turns "fast enough" into a number: an **SLI** (fraction of good requests), an **objective** (0.99), and a period (30 days). The **error budget** is $1 - \text{objective}$ of the requests.
- The **burn rate** is how many times faster than allowed the budget is being spent. Alerting on burn rate, not on raw error counts, makes one threshold work at any traffic level.
- **Two windows per alert**: the long one proves the burn is big enough to matter, the short one proves it is still happening. A short spike is vetoed by the long window; a page clears minutes after a fix because the short window recovers first.
- **Fast** (14.4x over 1 h and 5 min) pages, **slow** (6x over 6 h and 30 min) opens a ticket. The **drill** profile compresses the windows 12x so a drill pages within minutes.
- A latency SLI is exact only at a histogram bound, and rules that Prometheus never loaded alert no one. `promtool test rules` replays synthetic traffic through the rules to prove they fire when they should and only then.

## How to read this chapter

Sections 1 to 3 build SLOs and burn-rate alerting from first principles and work an outage, a blip, and a recovery by hand. Section 4 fixes the SLO file and the rule shape. Section 5 walks through the solution (`slo.yaml` and the script that renders it into rules) and explains why each choice is the right one. Try to predict each step before reading it.

The solution is used like this:

```bash
python3 deploy/observability/render_slo_rules.py --system forge   # writes rules/slo-prod.yaml and rules/slo-drill.yaml
promtool check rules deploy/observability/rules/slo-*.yaml
kubectl apply -f deploy/observability/rules/slo-prod.yaml
```

---

## 1. Why now

The stack is deployed, traced (`obs.01`), and scraped (`obs.02`), and a production system must answer a question none of that answers: is it good enough, and when is it not? "Alert when TTFT p95 goes over 500 ms" fires all night at 3 requests a minute and never fires during a slow, steady leak. The incident drills from here on (`ops.01` first) only work if the alerts detect the fault within a time limit and stay quiet the rest of the time. That needs targets everyone agreed on, and alerts derived from them.

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

Burn rate is a ratio of ratios, so the same rule works at 3 requests a second or 3 000. It is also bounded: $B \le 1/\beta$, because $e \le 1$. With $o = 0.95$, $B$ can never exceed 20, so a 14.4x alert needs 72% of requests to be bad. That is why the solution promises $o = 0.99$ for both latency SLOs: $B$ can reach 100, and 14.4% slow requests are enough to page.

### 2.3 Two windows

An alert over one window has to trade detection time against noise. Long windows ignore short spikes but take long to fire and, worse, keep firing for a whole window after the problem is fixed. Short windows react fast and page on every hiccup. The multi-window rule asks both:

$$\text{fire} \iff B(L) > \phi \ \text{ and } \ B(S) > \phi$$

| Alert | $\phi$ | $L$ / $S$ (prod) | $L$ / $S$ (drill) | Severity |
|---|---|---|---|---|
| `<SLO>BudgetBurnFast` | 14.4 | 1 h / 5 min | 5 min / 25 s | page |
| `<SLO>BudgetBurnSlow` | 6 | 6 h / 30 min | 30 min / 150 s | ticket |

$S$ is a twelfth of $L$. The **drill** profile divides both by 12 again, so a drill sees a page within its 300 s budget. Each rule file carries one profile; both may be installed at once, because every series and alert carries an `slo_profile` label.

### 2.4 Rules as code, and as tests

The rules are Prometheus **recording rules** (one error ratio per SLO and window, `slo:sli_error:ratio_rate<w>`) and **alerting rules** (the comparison above), wrapped in a `PrometheusRule` object that kube-prometheus-stack loads only with the label `release: observability`. `promtool test rules` evaluates rules against synthetic series with a fake clock, which is how to check behaviour, not just syntax: generate traffic from `slo.yaml` (its thresholds and objectives), replay it through the rendered rules, and assert which alerts fire when.

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

Each short window alone would page; each long window vetoes. Nobody is woken for 2 minutes that cost $2/43\,200 \approx 0.005\%$ of the month. A good noise scenario for `promtool` is this blip on top of a steady burn at $0.5\beta$.

**Recovery.** After the outage of the first example is fixed at 11:00, $e(1h)$ stays above 0.072 until about 11:56, but $e(5m)$ is 0 from 11:05: the page clears at 11:05, not at noon. A long-window-only alert keeps paging for the hour, which is what a reset scenario catches.

**TTFT, $o = 0.99$.** $\beta = 0.01$, fast threshold $e > 0.144$. A fast scenario burns at twice that, $e = 0.288$ ($B = 28.8$), for 75 minutes; a correct rule set pages 70 minutes in, fires nothing from TPOT or availability, and no longer pages 8 minutes after the burn stops. A slow scenario burns at $1.5 \cdot 6 = 9$ times the budget: a ticket, and no page, because $9 < 14.4$.

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

The rules can be written by hand or rendered from `slo.yaml` with a small script (the solution does, section 5). The synthetic `promtool` series carry `job="<system>-gateway"` (and copies under `<system>-engine`), `namespace`, `tl_engine_role`, `gen_ai_operation_name`, `gen_ai_request_model`, and for HTTP `http_request_method`, `http_route`, `http_response_status_code`; select on any of them. Write `le` as Prometheus stores it (`"0.5"`, `"0.05"`, `"1.0"`). Rule groups evaluate every minute or faster (prod) and every 10 s or faster (drill).

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| The SLO file matches the schema | conformance | `slo.yaml` validates; profile `prod` | drills, the production system, and `obs.04` read it by these names |
| Latency thresholds are bucket bounds | boundary | both thresholds are bounds of their histogram | an exact SLI (section 2.1 of `obs.02`) |
| Targets no looser than the defaults | boundary | TTFT at most 500 ms, TPOT at most 60 ms | stricter is allowed, looser never |
| Rule files are selected PrometheusRules | conformance | `PrometheusRule`, label `release: observability` | the stack loads them |
| The six alerts | unit | each name exactly once per file, severity from `slo.yaml`, no extras | detection by name, page or ticket |
| Profile windows | unit | the four windows of each profile appear as range selectors | the two-window rule; drills look for the drill windows |
| Groups evaluate often enough | boundary | group interval at most 60 s (prod), 10 s (drill) | short windows need frequent evaluation |
| `promtool check rules` accepts both files | conformance | no syntax errors | no syntax error reaches the cluster |
| A fast burn pages, then resets | conformance | per SLO: quiet before, page during, no other SLO's alerts, clear after (section 3), both profiles | the page you want |
| A slow burn tickets without paging | conformance | per SLO, 9x: Slow fires, Fast does not | tickets are not pages |
| Noise and short spikes stay quiet | conformance | $0.5\beta$ with a 2-minute spike: nothing fires, at 7 points | the pages you do not want |
| Rules loaded in Prometheus | conformance | on kind: `/api/v1/rules` lists all six alerts | the selector matched |

## 5. The solution, step by step

### 5.1 From first principles

1. **Choose SLIs where the user is.** TTFT, TPOT, and 5xx ratio, all measured at the gateway (`job="<system>-gateway"`), because that is the latency and the error a client sees.
2. **Make each latency SLI exact.** The threshold must be a bucket bound (`obs.02`, section 2.1). 500 ms is a TTFT bound. 60 ms is not a TPOT bound, so the solution picks the next stricter bound, 50 ms.
3. **Choose objectives that can page.** Section 2.2 shows $B \le 1/\beta$. A 14.4x alert must be reachable well before every request is bad, so the latency objectives are 0.99 ($B$ up to 100) and availability 0.995 ($B$ up to 200).
4. **One error ratio per SLO and window.** The alert expression compares ratios over two windows; computing each ratio once as a recording rule makes the six alerts cheap and lets dashboards (`obs.04`) reuse the same series.
5. **Compare the ratio, not the burn rate.** $B(w) > \phi \iff e(w) > \phi \beta$, so the threshold is a constant $\phi(1 - o)$ written into the alert, and no division happens at evaluation.
6. **Two profiles from one source.** Prod and drill differ only in windows, evaluation interval, and `for`; everything else must be identical, so both are rendered from one `slo.yaml`, and every series and alert carries `slo_profile` so both can be loaded together.
7. **Rules are only real once loaded.** The `PrometheusRule` must carry `release: observability`, the stack's rule selector.

### 5.2 The code, annotated

**The SLO file** is the single source. The comments record why each number was chosen:

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

**The profiles** live in the renderer, because they are a property of where the rules run, not of the SLO:

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

`for` is a small hold so that a single evaluation over the threshold does not page; it is short relative to the short window, so it adds little to detection time.

**The error ratio** is the bad fraction $e(w)$. For latency it is one minus the good fraction read at the threshold's bucket; for availability it is the 5xx fraction directly:

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

`sum(...)` before the division aggregates across pods, models, and roles, so the SLI is the whole service's fraction, not an average of per-pod fractions (which would weight an idle pod like a busy one).

**Rendering** writes one recording rule per SLO per window, labelled with `slo` and `slo_profile`, then one alert per entry of `alerts`:

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
            "            and",                                      # both windows (section 2.3)
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

Full solution: [`slo.yaml`](../../../../course/solutions/entry/deploy/observability/slo.yaml), [`render_slo_rules.py`](../../../../course/solutions/entry/deploy/observability/render_slo_rules.py), [`rules/slo-prod.yaml`](../../../../course/solutions/entry/deploy/observability/rules/slo-prod.yaml), [`rules/slo-drill.yaml`](../../../../course/solutions/entry/deploy/observability/rules/slo-drill.yaml).

### 5.3 Why this solution

- **Cost.** 3 SLOs times 4 windows is 12 recording rules per profile, each a `rate` over at most two series families; the six alerts only compare recorded series. Evaluation cost does not grow with the number of alerts.
- **Why render instead of writing YAML by hand?** Twelve near-identical expressions per profile, two profiles: a hand edit that changes a threshold in one place and not another is the classic SLO bug. One source and a renderer make the files consistent by construction. The renderer is stdlib-only (a small parser for exactly this file's shape), so it runs anywhere.
- **Why two windows and not one with a long `for`?** A long `for` delays detection by the same amount and still keeps firing after a fix as long as the one window stays high; the short window is what makes the alert clear minutes after recovery.
- **Why request-based SLIs?** They are directly computable from the histograms `obs.02` serves. The accepted cost is noise at very low traffic (one bad request in three is 33%); Going further lists the fix.

### 5.4 Explain it back

1. Why is the alert threshold `14.4 * 0.01` and not `14.4 * 0.99`?
   <details><summary>Answer</summary>The recorded series is the error ratio $e$; the burn rate is $e / \beta$ with $\beta = 1 - o = 0.01$. $B > 14.4 \iff e > 14.4 \beta = 0.144$. Multiplying by the objective (0.99) gives 14.26, which an error ratio in $[0, 1]$ can never exceed, so nothing would ever page.</details>
2. Why is the TPOT threshold 50 ms when the default target is 60 ms?
   <details><summary>Answer</summary>60 ms is not a bucket bound of the TPOT histogram, so no counter says how many requests were at most 60 ms. The nearest exact bound that is no looser is 50 ms.</details>
3. In the 2-minute blip of section 3, which condition stops the fast alert from paging?
   <details><summary>Answer</summary>The long window: $e(1h) = 2/60 \approx 0.033$, below $14.4 \cdot 0.005 = 0.072$, while the short window alone ($e = 0.4$) would page.</details>
4. Why does every recording rule and alert carry `slo_profile`?
   <details><summary>Answer</summary>Both profiles record a `ratio_rate5m` series for each SLO (the prod short fast window and the drill long fast window are both 5 minutes). Without the label, the two would be the same series from two rules and collide; with it, both files can be installed at once and routed separately.</details>
5. Why does the error ratio sum across pods before dividing?
   <details><summary>Answer</summary>The SLI is a fraction of the service's requests. Summing numerator and denominator first weights each pod by its traffic; averaging per-pod ratios would let an idle pod with one slow request count as much as a busy one.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. The `PrometheusRule` lacks `release: observability` | `kubectl get prometheusrule` lists it; Prometheus has no such alert; nothing ever pages | `render` writes the label into every file's metadata |
| 2. One window per alert (only the short one) | a page for every 2-minute blip; people learn to ignore pages | every alert is `long > thr and short > thr` |
| 3. A latency threshold between bounds, or a rule using another bound than `slo.yaml` | the SLI counts the wrong requests; a scenario built from the SLO file does not page | thresholds are bounds (500 ms, 50 ms), and `le` is derived from `threshold_ms` by `fmt_le`, never typed twice |
| 4. The drill profile with prod windows | the drill refuses to start; or the drill page arrives after the drill is over | `PROFILES["drill"]` holds the 12x-compressed windows |
| 5. Only the long window | the page keeps firing for an hour after the fix | the short window clears the alert minutes after recovery |
| 6. Factors swapped, or the budget written as the objective (`14.4 * 0.99`) | a slow burn pages at night, or no burn ever pages | factors come from `windows.<kind>.burn_rate`; the threshold uses `1 - objective` |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `obs.02` | the histograms and the 5xx counts, under the scrape jobs it named |
| Forward | `obs.04` | burn-rate stats and the firing-alerts table read `slo:sli_error:*` and `ALERTS` |
| Forward | `ops.01` | `TTFTBudgetBurnFast` under the drill profile detects the killed decode pod |
| Forward | `dur.12` | the release workflow checks the burn rate during a canary before promoting |

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `slo.yaml` and a render script | Sloth, Pyrra, OpenSLO | SLO specs compiled to rules and dashboards, error-budget reports | [Sloth](https://sloth.dev/), [OpenSLO](https://openslo.com/) |
| request-based SLIs | window-based SLIs ("good minutes") | robust to traffic swings, closer to a user's experience of an outage | *Implementing Service Level Objectives*, ch. 3 |
| two fixed profiles | burn-rate alerts with a low-traffic fallback | a minimum request count so 1 failure in 3 requests does not page | SRE Workbook ch. 5, "Low-traffic services" |
| promtool unit tests | rules CI on every change | the same tests in CI (`dep.05`), with recorded production series | `promtool test rules` |
| severity labels | Alertmanager routing, inhibition, silences | pages to on-call, tickets to a queue, drill pages to a drill channel | [Alertmanager](https://prometheus.io/docs/alerting/latest/alertmanager/) |
