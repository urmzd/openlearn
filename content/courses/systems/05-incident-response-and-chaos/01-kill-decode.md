# A decode worker dies mid-stream

## Overview

| | |
|---|---|
| **Module** | `ops.01` · drill · ops, docs · Pass 7 · 2 to 3 h |
| **Solution** | a worked incident (section 5): keeping the service within its SLO while a decode pod dies under load; the model runbook [`docs/runbooks/TTFTBudgetBurnFast.md`](../../../../course/solutions/docs/runbooks/TTFTBudgetBurnFast.md) (the one the alert links to) and the model postmortem [`docs/postmortems/2026-10-09-kill-decode.md`](../../../../course/solutions/docs/postmortems/2026-10-09-kill-decode.md) |
| **Contract** | the fault, detection, and resolve conditions in section 4; [`otel/slo.schema.json`](../../../../course/contracts/otel/slo.schema.json) (the `drill` profile) |
| **Needs** | [`obs.03`](../04-observability/03-slos-and-burn-rate-alerts.md) (the alert that detects it); the incident is in what `dep.00` and the Pass 7 charts deployed, the gateway's failover (`gw.05`), with the dashboard of [`obs.04`](../04-observability/04-dashboards-as-code.md) |
| **Used by** | no call site: a drill. The production system is expected to survive it within the error budget, and the operations pass reruns it |
| **Optional depth** | *Site Reliability Engineering*, ch. 13 Emergency Response and ch. 22 Addressing Cascading Failures ([free](https://sre.google/sre-book/table-of-contents/)); *Chaos Engineering* (Rosenthal and Jones), ch. 3 |

## Key Takeaways

- Losing one of two decode replicas halves capacity until the replacement is Ready: the survivor queues, TTFT climbs, and the **TTFT** budget burns even though almost nothing fails outright.
- The gateway's job is the **commit rule**: before the first byte a failed request is retried on another engine; after it, the stream ends with an SSE `error` event, never truncated and never spliced with another engine's tokens.
- The **drill profile** compresses the burn-rate windows 12x (5 min and 25 s) so the page arrives within the drill's 300 s budget; under prod windows the same incident would not page at all.
- Mitigation is mostly **waiting correctly**: confirm the replacement is coming (`rollout status`), add capacity if it is not enough, and verify with the SLO, not with a feeling.
- The postmortem's action items are about **capacity during replacement**: replicas, disruption budgets, warm-up time.

## How to read this chapter

This chapter is a worked incident. Sections 1 to 3 build the capacity, streaming, and detection reasoning from first principles and work the incident's timeline by hand. Section 4 states what is injected, how it is detected, and what "resolved" means. Section 5 walks through the response from page to postmortem, with the model runbook and postmortem, and explains why each step is the right one. Before reading each diagnosis step, predict what its output will show.

The drill needs the drill profile's alerts loaded (`obs.03`):

```bash
kubectl apply -f deploy/observability/rules/slo-drill.yaml
```

---

## 1. Why now

The platform now runs a prefill engine and two decode engines behind a gateway that routes by prefix affinity and fails over between them (`gw.05`), with SLOs and burn-rate alerts (`obs.03`) and a dashboard (`obs.04`). None of that has been tested by an actual failure. Pods die for ordinary reasons every week: a node drain, an eviction, an OOM kill, a crash in new code. This drill deletes a decode pod while the loadgen sends 20 requests a second, and asks three questions with numbers attached: did the alert page in time, did users get clean errors instead of corrupted streams, and did the service return within the SLO without the responder breaking anything else.

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

Splicing (continuing the stream from another replica) would send tokens generated from a different state: text that reads plausibly and is wrong. Truncation (closing the connection without an `error` event) looks to the client like a complete answer. Both are failures, and the loadgen report shows whether either happened.

### 2.3 Detection under the drill profile

From `obs.03`: the fast alert fires when the error ratio exceeds $14.4 \beta$ over both the long and the short window. With $\beta = 0.01$ (TTFT, $o = 0.99$) that is 14.4% slow requests. The prod windows (1 h and 5 min) would need 14.4% of a whole **hour** to be slow; a 3-minute event cannot do that. The drill profile's 5-minute long window can, which is why every drill runs under `slo_profile = "drill"`.

### 2.4 Mitigate, then verify with the SLO

Kubernetes replaces the pod by itself (the Deployment's ReplicaSet wants 2). The responder's job is to confirm that is happening, to add capacity if the survivor cannot hold the SLO for $t_r$, and to declare the incident over only when the resolve checks hold: TTFT p95 under 500 ms for 5 minutes, the 5xx ratio under 0.5% for 2 minutes, and the API smoke cases passing through the gateway.

## 3. Worked example by hand

A model timeline (the model postmortem in section 5.5 has the same numbers). $\lambda = 20$, $\mu = 12$, $n = 2$, so $\rho = 0.83$ before the kill.

| Time | Event | Numbers |
|---|---|---|
| 10:00:00 | the drill starts (seed 7); loadgen at 20 rps | baseline TTFT p95 about 180 ms |
| 10:00:30 | decode pod `x2kq` deleted, grace period 0 | 11 streams on it end with an SSE `error` event; new requests fail over |
| 10:00:30 to 10:03:10 | the survivor serves alone: $\rho = 20 / 12 = 1.67$ | the queue grows by about 8 requests a second at first |
| 10:01:00 | 30 s after the kill, about a third of requests wait more than 500 ms | $e(25s) \approx 0.33 > 0.144$: the short window is over |
| 10:02:31 | $e(5m)$ crosses 0.144 (the long window catches up), plus `for: 10s` | **`TTFTBudgetBurnFast` fires**: 121 s after the kill, inside the 300 s budget |
| 10:03:10 | the replacement is Ready ($t_r = 160$ s) | the queue drains in about 20 s |
| 10:04:05 | the short window has been healthy for 25 s | the alert resolves |
| 10:09:30 | resolve conditions verified | p95 under 0.5 s held 300 s; 5xx 0.1% held 120 s; smoke cases pass |

Check the detection time by hand: from 10:00:30, a third of each second's requests are slow. The 5-minute window holds 30 s of baseline and $t$ seconds of incident, so $e(5m) \approx 0.33\, t / 300$, which exceeds 0.144 when $t > 131$ s, about 10:02:41 for a steady third. The model timeline's 10:02:31 has the slow share rising above a third as the queue grows; either way it is well inside 300 s. Under prod windows, $e(1h) \approx 0.33 \cdot 160 / 3600 = 0.015$: no page, which is the point of the drill profile.

The availability SLO does not move: failover keeps 5xx near zero, and the 11 broken streams were already 200 responses when they failed. That gap is an action item (an SLI that counts streams ending in `error`).

## 4. The incident: what is injected, detected, and resolved

| Step | What happens |
|---|---|
| load | the loadgen (`load.01`) runs through the gateway NodePort: `--target <gateway_url> --rate 20 --duration 900s --seed 1` |
| inject | 30 s into the load (a baseline first), one random Ready pod of the decode Deployment (for example `deploy/<system>-engine-decode`) is deleted with `--grace-period=0`; which one is chosen from a seed |
| detect | `TTFTBudgetBurnFast` (drill profile, [`rules/slo-drill.yaml`](../../../../course/solutions/entry/deploy/observability/rules/slo-drill.yaml)) must fire within 300 s of the kill |
| page | *TTFTBudgetBurnFast (drill profile): time to first token is burning the error budget; some users report streams that stop mid-sentence* |

**Resolved means** all of these hold at the end of the run:

| Condition | Holds for |
|---|---|
| the decode rollout is complete (every replica updated and available) | within 120 s |
| `histogram_quantile(0.95, sum(rate(gen_ai_server_time_to_first_token_seconds_bucket{job="<system>-gateway"}[5m])) by (le)) < 0.5` | 300 s |
| `sum(rate(http_server_request_duration_seconds_count{job="<system>-gateway",http_response_status_code=~"5.."}[5m])) / sum(rate(http_server_request_duration_seconds_count{job="<system>-gateway"}[5m])) < 0.005` | 120 s |
| the v1 gateway API smoke cases pass through the gateway | at the end |
| a trace in the window has `gateway.proxy` from `<system>-gateway` and spans from `<system>-engine` | evidence for the timeline |
| `docs/runbooks/TTFTBudgetBurnFast.md` (Symptoms, Diagnosis, Mitigation) and `docs/postmortems/<date>-kill-decode.md` (the seven sections) exist | at the end |
| all within 45 minutes | |

The runbook path is the one the alert's `runbook` annotation links to (`obs.03`), so the page leads straight to it.

## 5. The solution, step by step

### 5.1 From first principles

1. **The page is a TTFT burn, so the question is "why are first tokens late?"** Late first tokens with almost no errors mean requests are waiting, not failing. Waiting means demand exceeds capacity somewhere (section 2.1).
2. **Find the lost capacity.** A pod that is seconds old or `READY 0/1`, or a `Killing` event, names the replica that went away. Its peers now carry $\lambda$ alone; with $
ho > 1$ their queues must grow.
3. **Confirm with the queue, not the latency.** Queue depth rising on the survivors with unchanged load distinguishes lost capacity from a slow model or a bad release.
4. **Check that the gateway is doing its part** (section 2.2): failover lines before the first byte, `upstream_failed` error events after it, and no spliced or truncated streams. One `trace_id` from a failed line shows which engine the request reached.
5. **Rule out the other common cause**, a release: a new decode revision at the start of the burn points at the rollout instead.
6. **Mitigation follows from $t_r$.** Kubernetes is already replacing the pod. If the survivors can hold for $t_r$, the right action is to watch the rollout; if not, add a replica now. Never restart the gateway: that kills every in-flight stream on every engine.
7. **The incident is over when the SLO says so**, not when the pod is Running: the burn under 1, the alert resolved, the resolve conditions held.
8. **The root cause is capacity during replacement**, not the kill. Pods die every week; the action items must make one loss survivable.

### 5.2 Diagnosis, annotated

```bash
# 1. Which replicas are serving, and did one restart? A pod with AGE of
#    seconds, READY 0/1, or a Killing event is the one that went away.
kubectl -n <system> get pods -l app.kubernetes.io/name=<system>-engine -o wide
kubectl -n <system> get events --sort-by=.lastTimestamp | tail -20
```

```promql
# 2. Is the remaining capacity saturated? A growing queue on the survivors
#    with the load unchanged means lost capacity, not a slow model.
sum by (pod) (tl_engine_queue_depth)
sum by (pod) (tl_engine_active_sequences)
```

```bash
# 3. Is the gateway routing around the hole? Failover before the first byte,
#    stream errors after it. Open one trace_id from a failed line in Tempo.
kubectl -n <system> logs deploy/<system>-gateway --since=15m | grep -E '"failover"|"upstream_failed"'

# 4. Rule out a release: a new revision at the start of the burn points at it.
kubectl -n <system> rollout history deploy/<system>-engine-decode
```

Step 2's two queries are the "Queue depth and running sequences" panel of the `obs.04` dashboard, so in practice diagnosis starts there and the TTFT heatmap beside it: the band above 500 ms appears at the kill.

### 5.3 Mitigation, annotated

```bash
# A lost replica: Kubernetes replaces it. Watch it come back; the burn stops
# when the new pod is Ready and the queue drains.
kubectl -n <system> rollout status deploy/<system>-engine-decode

# If the survivors stay saturated, add capacity now (and return to the chart
# value afterwards with helm upgrade).
kubectl -n <system> scale deploy/<system>-engine-decode --replicas=3

# If a rollout caused it instead:
kubectl -n <system> rollout undo deploy/<system>-engine-decode
```

Confirm with the SLO: the TTFT burn stat under 1, the alert resolved (the short window clears first, section 2.3 of `obs.03`), and the v1 smoke cases passing through `http://127.0.0.1:30080`.

### 5.4 The model runbook

The runbook opens with what the alert *means*, so the reader can reason when the steps do not fit: "over both its long and its short window, more than 14.4 times the allowed share of requests waited longer than the TTFT threshold for their first token. At that rate the 30-day error budget is gone in about two days." Its Symptoms name three views of the same fault:

```markdown
## Symptoms

- The page `TTFTBudgetBurnFast` (`severity: page`), with `slo_profile` `prod`
  (or `drill` during a drill).
- The serving dashboard (obs.04): the TTFT heatmap grows a band above the
  threshold, the TTFT burn stat turns red, queue depth rises on some engine
  pods.
- Users: slow starts; if a decode pod died, some streams ended early with an
  SSE `error` event (`"code": "upstream_failed"`), never with spliced text.
```

Diagnosis is the four steps of 5.2 and Mitigation the three branches of 5.3. Full runbook: [`TTFTBudgetBurnFast.md`](../../../../course/solutions/docs/runbooks/TTFTBudgetBurnFast.md).

### 5.5 The model postmortem

The numbers in section 3 are the model postmortem's. Its root cause is a property of the system, with the number that proves it:

```markdown
## Root cause

The decode tier has two replicas and no headroom: one replica's load is
more than half of what the other can serve within the TTFT threshold. When a
pod disappears, the replacement needs about 2.5 minutes (image already on the
node, model load and warm-up) before readiness, and for that time the
survivor is over capacity. The deletion itself is ordinary (node drain,
eviction, crash); the gap is capacity during replacement.
```

Its Detection section records a blind spot as well as the alert that fired: mid-stream errors are a 200 status with an `error` event, so the 5xx SLI does not see them. The action items each close one gap:

| Action | Owner | Due |
|---|---|---|
| Run three decode replicas (or set the HPA floor to 3) so one loss stays under the TTFT threshold | serving | next sprint |
| Add a PodDisruptionBudget `minAvailable: 2` for decode so voluntary evictions never take two at once | serving | next sprint |
| Count streams that end with an `error` event in an SLI (`tl_gateway_stream_errors_total`) so mid-stream failures are visible to the availability SLO | observability | next quarter |
| Cache model weights on the node (hostPath under /artifacts) to cut the replacement's warm-up | platform | next quarter |

Full postmortem: [`2026-10-09-kill-decode.md`](../../../../course/solutions/docs/postmortems/2026-10-09-kill-decode.md).

### 5.6 Why this solution

- **Waiting is the mitigation** because Kubernetes already does the right thing; the responder's value is confirming it and adding capacity if $t_r$ is too long. Every manual action during an incident carries its own risk, so the runbook makes "watch `rollout status`" an explicit step rather than a non-action.
- **Three replicas, not a faster replacement alone.** With $\mu = 12$ and $\lambda = 20$, two survivors give $
ho = 0.83$: within threshold. Cutting $t_r$ shortens the burn but does not remove it; headroom removes it.
- **Error after the first byte, never splice.** A truncated or spliced stream looks like a correct answer and is wrong; an explicit error is honest and lets the client retry.

### 5.7 Explain it back

1. Why does this incident burn the TTFT budget but not the availability budget?
   <details><summary>Answer</summary>Failover before the first byte turns would-be failures into slower successes, and the streams that broke after the first byte had already returned 200. Almost every request is a 200; many are late. The 5xx SLI cannot see either effect.</details>
2. Why would the same incident not page under the prod profile?
   <details><summary>Answer</summary>The fast alert needs $e(1h) > 0.144$. A third of requests slow for 160 s gives $e(1h) \approx 0.33 \cdot 160 / 3600 \approx 0.015$, an order of magnitude short. The drill profile's 5-minute long window can be dominated by a 2-minute event.</details>
3. Why must the gateway not continue a broken stream on the other decode replica?
   <details><summary>Answer</summary>The other replica does not hold this sequence's KV cache, so the tokens it generates come from a different state. The client would receive text that reads plausibly and is wrong, with no signal that anything happened.</details>
4. Why is "the pod is Running" not the end of the incident?
   <details><summary>Answer</summary>Running means the container started; Ready means it passed readiness after loading the model. Until then it takes no traffic and the survivor's queue still grows. The incident ends when the SLO conditions hold, which is measured, not inferred.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. Only the prod rules installed | the page never comes | the drill profile's rules are applied before the drill (How to read this chapter) |
| 2. The gateway retries after the first byte | streams with text from two engines that read plausibly and are wrong | `gw.05`'s commit rule: after the first byte, an SSE `error` event ends the stream |
| 3. Restarting the gateway to "clear" it | every in-flight stream on every engine dies; the 5xx ratio jumps | the runbook's mitigation never touches the gateway |
| 4. Declaring victory when the pod is Running, not Ready | TTFT still high; the incident is not over | verification uses `rollout status` and the SLO, held for 300 s |
| 5. A postmortem that blames the kill | no action item about capacity during replacement; the next node drain repeats the incident | the model root cause is capacity during replacement, with replicas, a PDB, and warm-up as action items |

## 7. Where it's used next
## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `obs.03` | `TTFTBudgetBurnFast` and the drill profile detect the fault |
| Back | `obs.04` | the dashboard is where diagnosis starts |
| Forward | `ops.09` | the same capacity reasoning for one tenant flooding the gateway |
| Forward | `ops.04` | a rolling KV format migration under the same load and SLO checks |

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| a scripted drill with one injector | Chaos Mesh, LitmusChaos, Gremlin | scheduled experiments, blast-radius limits, many fault types | [Chaos Mesh](https://chaos-mesh.org/) |
| a fixed replica count | HPA or KEDA on queue depth, PodDisruptionBudgets | capacity that follows load; evictions that never take two pods | [PDB](https://kubernetes.io/docs/tasks/run-application/configure-pdb/) |
| fail before first byte, error after | request hedging, KV cache migration between replicas | resume a sequence elsewhere without splicing | vLLM and SGLang disaggregation designs |
| a game day run alone | team game days with an incident commander | roles, communication, the human side | *SRE Workbook*, ch. 9 Incident Response |
