<!-- ol:module ops.01 -->
# A decode worker dies mid-stream

## Overview

| | |
|---|---|
| **Module** | `ops.01` · drill · ops, docs · Pass 7 · 2 to 3 h |
| **You build** | `docs/runbooks/TTFTBudgetBurnFast.md` (Symptoms, Diagnosis, Mitigation: the runbook your alert links to) and `docs/postmortems/<date>-kill-decode.md`, and you keep the service within its SLO while a decode pod dies under load |
| **Contract** | the drill spec `course/drills/kill-decode/drill.toml` (what is injected, detected, and checked: section 4); [`otel/slo.schema.json`](../../../../course/contracts/otel/slo.schema.json) (the `drill` profile) |
| **Tests** | graded by `ol drill end`; no course tests to read (section 4 lists every check) |
| **Needs** | [`obs.03`](../04-observability/03-slos-and-burn-rate-alerts.md) (the alert that detects it); you triage what `dep.00` and the Pass 7 charts deployed, the gateway's failover (`gw.05`), with the dashboard of [`obs.04`](../04-observability/04-dashboards-as-code.md) |
| **Used by** | no call site: a drill. MS-prod requires it within budget; MS-ops reruns it |
| **Milestone** | MS-prod (drill `ops.01` within the error budget) |
| **Optional depth** | *Site Reliability Engineering*, ch. 13 Emergency Response and ch. 22 Addressing Cascading Failures ([free](https://sre.google/sre-book/table-of-contents/)); *Chaos Engineering* (Rosenthal and Jones), ch. 3 |

## Key Takeaways

- Losing one of two decode replicas halves capacity until the replacement is Ready: the survivor queues, TTFT climbs, and the **TTFT** budget burns even though almost nothing fails outright.
- The gateway's job is the **commit rule**: before the first byte a failed request is retried on another engine; after it, the stream ends with an SSE `error` event, never truncated and never spliced with another engine's tokens.
- The **drill profile** compresses the burn-rate windows 12x (5 min and 25 s) so the page arrives within the drill's 300 s budget; under prod windows the same incident would not page at all.
- Mitigation is mostly **waiting correctly**: confirm the replacement is coming (`rollout status`), add capacity if it is not enough, and verify with the SLO, not with a feeling.
- The postmortem's action items are about **capacity during replacement**: replicas, disruption budgets, warm-up time.

## How to work this chapter

```bash
kubectl apply -f deploy/observability/rules/slo-drill.yaml   # the drill profile's alerts (obs.03)
ol drill list                         # kill-decode is ops.01
ol drill start kill-decode            # starts your loadgen, kills a decode pod 30 s in, prints only the page
# ... dashboard, kubectl, gateway logs, the runbook ... restore and hold the SLO ...
ol drill end                          # grades: detected in time, resolved, evidence, runbook, postmortem
ol drill reset                        # stops the loadgen; always run it after `end`
```

---

## 1. Why now

Your platform now runs a prefill engine and two decode engines behind a gateway that routes by prefix affinity and fails over between them (`gw.05`), and you have SLOs with burn-rate alerts (`obs.03`) and a dashboard (`obs.04`). None of that has been tested by an actual failure. Pods die for ordinary reasons every week: a node drain, an eviction, an OOM kill, a crash in new code. This drill deletes a decode pod while your loadgen sends 20 requests a second, and asks three questions with numbers attached: did the alert page in time, did users get clean errors instead of corrupted streams, and did the service return within the SLO without you breaking anything else.

## 2. Principles

### 2.1 What a lost replica does

| Symbol | Meaning | Type |
|---|---|---|
| $\lambda$ | arrival rate of requests | requests per second (20 here) |
| $\mu$ | the rate one decode replica serves within the TTFT threshold | requests per second |
| $n$ | Ready decode replicas | 2, then 1, then 2 |
| $\rho = \lambda / (n \mu)$ | utilization | ratio |
| $t_r$ | time for the replacement to become Ready (schedule, start, load the model, warm up) | seconds |

With $n = 2$ and $\rho = 0.6$, each replica runs at 60%. When one dies, $\rho = 1.2$ for the survivor: more arrives than it can serve, so its queue grows by $\lambda - \mu$ every second until $t_r$ has passed, and every queued request's TTFT includes its wait. That is a TTFT burn, not an availability burn: almost every request still gets a 200.

### 2.2 Before and after the first byte

A request in flight on the killed pod is in one of two states:

| State | What the gateway can do | What the client sees |
|---|---|---|
| no byte sent to the client yet | retry on another decode replica (`gw.05` failover) | a slower first token |
| bytes already streamed | nothing safe: the other replica does not have this sequence's KV cache | `data: {"error": {"code": "upstream_failed", ...}}` then the end of the stream |

Splicing (continuing the stream from another replica) would send tokens generated from a different state: text that reads plausibly and is wrong. Truncation (closing the connection without an `error` event) looks to the client like a complete answer. Both are failures the drill checks for in the loadgen report.

### 2.3 Detection under the drill profile

From `obs.03`: the fast alert fires when the error ratio exceeds $14.4 \beta$ over both the long and the short window. With $\beta = 0.01$ (TTFT, $o = 0.99$) that is 14.4% slow requests. The prod windows (1 h and 5 min) would need 14.4% of a whole **hour** to be slow; a 3-minute event cannot do that. The drill profile's 5-minute long window can, which is why `ol drill` runs every drill with `slo_profile = "drill"`.

### 2.4 Mitigate, then verify with the SLO

Kubernetes replaces the pod by itself (the Deployment's ReplicaSet wants 2). The responder's job is to confirm that is happening, to add capacity if the survivor cannot hold the SLO for $t_r$, and to declare the incident over only when the resolve checks hold: TTFT p95 under 500 ms for 5 minutes, the 5xx ratio under 0.5% for 2 minutes, and the API smoke cases passing through the gateway.

## 3. Worked example by hand

A model timeline (the reference postmortem has the same shape). $\lambda = 20$, $\mu = 12$, $n = 2$, so $\rho = 0.83$ before the kill.

| Time | Event | Numbers |
|---|---|---|
| 10:00:00 | `ol drill start kill-decode --seed 7`; loadgen at 20 rps | baseline TTFT p95 about 180 ms |
| 10:00:30 | decode pod `x2kq` deleted, grace period 0 | 11 streams on it end with an SSE `error` event; new requests fail over |
| 10:00:30 to 10:03:10 | the survivor serves alone: $\rho = 20 / 12 = 1.67$ | the queue grows by about 8 requests a second at first |
| 10:01:00 | 30 s after the kill, about a third of requests wait more than 500 ms | $e(25s) \approx 0.33 > 0.144$: the short window is over |
| 10:02:31 | $e(5m)$ crosses 0.144 (the long window catches up), plus `for: 10s` | **`TTFTBudgetBurnFast` fires**: 121 s after the kill, inside the 300 s budget |
| 10:03:10 | the replacement is Ready ($t_r = 160$ s) | the queue drains in about 20 s |
| 10:04:05 | the short window has been healthy for 25 s | the alert resolves |
| 10:09:30 | `ol drill end` | p95 under 0.5 s held 300 s; 5xx 0.1% held 120 s; smoke cases pass |

Check the detection time by hand: from 10:00:30, a third of each second's requests are slow. The 5-minute window holds 30 s of baseline and $t$ seconds of incident, so $e(5m) \approx 0.33\, t / 300$, which exceeds 0.144 when $t > 131$ s, about 10:02:41 for a steady third. The model timeline's 10:02:31 has the slow share rising above a third as the queue grows; either way it is well inside 300 s. Under prod windows, $e(1h) \approx 0.33 \cdot 160 / 3600 = 0.015$: no page, which is the point of the drill profile.

The availability SLO does not move: failover keeps 5xx near zero, and the 11 broken streams were already 200 responses when they failed. That gap is an action item (an SLI that counts streams ending in `error`).

## 4. Inject, detect, mitigate, verify

| Step | What `ol drill` does | What you do |
|---|---|---|
| inject | starts `{loadgen} --target {deploy.gateway_url} --rate 20 --duration 900s --seed 1`; 30 s later deletes a random Ready pod of `{deploy.services.decode}` with `--grace-period=0` (seeded; revealed at `end`) | nothing: the page is your first signal |
| detect | at `end`, reads the `ALERTS` series: `TTFTBudgetBurnFast` must have fired within 300 s of the kill | acknowledge, open the dashboard (`obs.04`) and your runbook |
| mitigate | | diagnose with the runbook; confirm the replacement; scale up if the survivor cannot hold the SLO; never restart the gateway "to be sure" |
| verify | at `end`: the decode rollout is complete; TTFT p95 under 0.5 s held 300 s; 5xx ratio under 0.005 held 120 s; `openapi:v1:gateway:smoke` passes; a trace in the window spans `<system>-gateway` and `<system>-engine` | write the runbook and the postmortem |

`ol drill start` refuses to run unless your rules under `deploy/observability/` define `TTFTBudgetBurnFast` with the drill profile's windows (`obs.03`). Your `system.toml` needs `[deploy].services.decode` (for example `deploy/<system>-engine-decode`), `[deploy].prometheus`, and `[deploy].traces`, and `[entry].loadgen`.

### What `ol drill end` checks

| Check | Passes when |
|---|---|
| detected | `TTFTBudgetBurnFast` fired within `within_s = 300` of the injection (TTD is reported) |
| resolved | each `[[resolve.check]]` held for its `hold_s` at the end of the run (TTM is reported) |
| trace evidence | a trace in the window has `gateway.proxy` from `<system>-gateway` and spans from `<system>-engine` |
| runbook | `docs/runbooks/TTFTBudgetBurnFast.md` has Symptoms, Diagnosis, Mitigation |
| postmortem | `docs/postmortems/<date>-kill-decode.md` has Summary, Impact, Timeline, Root cause, Detection, Resolution, Action items |
| time limit | 45 minutes from start to end (not graded for the CI responder) |

A runbook starts like this; yours names your system and your commands:

## Symptoms

`TTFTBudgetBurnFast` pages; the TTFT heatmap grows a band above 500 ms; queue depth rises on some engine pods; a few streams end with an `error` event.

## Diagnosis

`kubectl -n <system> get pods -l app.kubernetes.io/name=<system>-engine -o wide`, then queue depth per pod, then the gateway's failover lines for one `trace_id`.

## Mitigation

Confirm the replacement (`kubectl rollout status`), scale the decode Deployment if the survivors stay saturated, and verify with the burn rate under 1.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. Only the prod rules installed | the page never comes; `ol drill start` refuses (no drill windows) | ol drill start (its rule gate); ol drill end, check detected |
| 2. The gateway retries after the first byte | streams with text from two engines that read plausibly and are wrong | gw.05's fault tests (after the first byte a request fails, never spliced); your postmortem's Impact |
| 3. Restarting the gateway to "clear" it | every in-flight stream on every engine dies; the 5xx check fails its hold | resolve check 3 (5xx ratio) |
| 4. Declaring victory when the pod is Running, not Ready | TTFT still high; the hold timer restarts | resolve checks 1 and 2 |
| 5. A postmortem that blames the kill | no action item about capacity during replacement; the next node drain repeats the incident | ol drill end, check postmortem; the self review against course/rubrics/postmortem.md |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `obs.03` | `TTFTBudgetBurnFast` and the drill profile detect the fault |
| Back | `obs.04` | the dashboard is where diagnosis starts |
| Forward | `ops.09` | the same capacity reasoning for one tenant flooding the gateway |
| Forward | `ops.04` | a rolling KV format migration under the same load and SLO checks |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `ol drill` with one injector | Chaos Mesh, LitmusChaos, Gremlin | scheduled experiments, blast-radius limits, many fault types | [Chaos Mesh](https://chaos-mesh.org/) |
| a fixed replica count | HPA or KEDA on queue depth, PodDisruptionBudgets | capacity that follows load; evictions that never take two pods | [PDB](https://kubernetes.io/docs/tasks/run-application/configure-pdb/) |
| fail before first byte, error after | request hedging, KV cache migration between replicas | resume a sequence elsewhere without splicing | vLLM and SGLang disaggregation designs |
| a game day you run alone | team game days with an incident commander | roles, communication, the human side | *SRE Workbook*, ch. 9 Incident Response |
