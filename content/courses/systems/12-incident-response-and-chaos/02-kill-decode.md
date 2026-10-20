# A decode worker dies mid-stream

The platform now runs a prefill engine and two decode engines behind a gateway that routes by prefix affinity and fails over between them, as built in [Routing, cascades, and failover](../../ai-engineering/25-ai-gateway/04-routing.md) in *AI Engineering*. It has SLOs and burn-rate alerts from [SLOs and multi-window burn-rate alerts](../11-observability/04-slos-and-burn-rate-alerts.md) and a dashboard from [Dashboards as code for the serving path](../11-observability/05-dashboards-as-code.md), all deployed by the charts that grew out of [Tracer deploy](../10-containers-kubernetes/01-tracer-deploy.md) in this book. None of that has been tested by an actual failure.

Pods die for ordinary reasons every week: a node drain, an eviction, an OOM kill, a crash in new code. This drill deletes a decode pod while the load generator sends 20 requests a second, and asks three questions with numbers attached. Did the alert page in time? Did users get clean errors instead of corrupted streams? Did the service return within the SLO without the responder breaking anything else? The production system is expected to survive this within its error budget, and the drill is worth rerunning whenever the serving tier changes.

Like [First drill: the engine crashloops](01-first-drill.md), this section is a worked incident. It builds the capacity, streaming, and detection reasoning from first principles, works the incident's timeline by hand, states what is injected and what "resolved" means, and then walks through the response from page to postmortem. The model runbook is [`docs/runbooks/TTFTBudgetBurnFast.md`](../../../../course/solutions/docs/runbooks/TTFTBudgetBurnFast.md), the one the alert links to; the model postmortem is [`docs/postmortems/2026-10-09-kill-decode.md`](../../../../course/solutions/docs/postmortems/2026-10-09-kill-decode.md); and the SLO contract is [`otel/slo.schema.json`](../../../../course/contracts/otel/slo.schema.json), whose `drill` profile this drill uses. Before you run it, load the drill profile's alerts:

```bash
kubectl apply -f deploy/observability/rules/slo-drill.yaml
```

> **Learning objectives**
> - Explain, with utilization numbers, why losing one of two decode replicas burns the TTFT budget while availability barely moves.
> - State the gateway's commit rule and why a stream must end with an error event rather than be spliced or truncated.
> - Compute when a burn-rate alert fires under the drill profile, and show why the prod profile would not page.
> - Diagnose lost capacity from pods, events, queue depth, and gateway logs, and choose between waiting, scaling, and rolling back.
> - Write a postmortem whose root cause is capacity during replacement.

## What a lost replica does

To reason about capacity, name the quantities:

| Symbol | Meaning | Type |
|---|---|---|
| $\lambda$ | arrival rate of requests | requests per second (20 here) |
| $\mu$ | the rate one decode replica serves within the TTFT threshold | requests per second |
| $n$ | Ready decode replicas | 2, then 1, then 2 |
| $\rho = \lambda / (n \mu)$ | utilization | ratio |
| $t_r$ | time for the replacement to become Ready (schedule, start, load the model, warm up) | seconds |

With $n = 2$ and $\rho = 0.6$, each replica runs at 60%. When one dies, $\rho = 1.2$ for the survivor: more arrives than it can serve, so its queue grows by $\lambda - \mu$ every second until $t_r$ has passed, and every queued request's TTFT includes its wait. That is a TTFT burn, not an availability burn: almost every request still gets a 200.

## Before and after the first byte

Lost capacity explains the slow requests. The requests that were in flight on the killed pod are a separate problem, and each is in one of two states:

| State | What the gateway can do | What the client sees |
|---|---|---|
| no byte sent to the client yet | retry on another decode replica (the gateway's failover) | a slower first token |
| bytes already streamed | nothing safe: the other replica does not have this sequence's KV cache | `data: {"error": {"code": "upstream_failed", ...}}` then the end of the stream |

This is the gateway's **commit rule**: before the first byte a failed request is retried on another engine; after it, the stream ends with an SSE `error` event. The two tempting alternatives are both failures. Splicing (continuing the stream from another replica) would send tokens generated from a different state: text that reads plausibly and is wrong. Truncation (closing the connection without an `error` event) looks to the client like a complete answer. The load generator's report shows whether either happened.

## Detection under the drill profile

The page has to arrive in time, and that depends on the alert windows. As derived in [SLOs and multi-window burn-rate alerts](../11-observability/04-slos-and-burn-rate-alerts.md), the fast alert fires when the error ratio exceeds $14.4 \beta$ over both the long and the short window. With $\beta = 0.01$ (TTFT, objective $o = 0.99$) that is 14.4% slow requests. The prod windows (1 h and 5 min) would need 14.4% of a whole *hour* to be slow, and a 3-minute event cannot do that. The **drill profile** compresses the windows 12x, to 5 minutes and 25 seconds, and its 5-minute long window can be dominated by a short event. That is why every drill runs under `slo_profile = "drill"`.

## Mitigate, then verify with the SLO

Kubernetes replaces the pod by itself, because the Deployment's ReplicaSet wants 2. Mitigation is therefore mostly waiting correctly. The responder's job is to confirm that the replacement is happening, to add capacity if the survivor cannot hold the SLO for $t_r$, and to declare the incident over only when the resolve checks hold: TTFT p95 under 500 ms for 5 minutes, the 5xx ratio under 0.5% for 2 minutes, and the API smoke cases passing through the gateway.

## Working the timeline by hand

Here is a model timeline; the model postmortem has the same numbers. Take $\lambda = 20$, $\mu = 12$, $n = 2$, so $\rho = 0.83$ before the kill.

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

Check the detection time by hand. From 10:00:30, a third of each second's requests are slow. The 5-minute window holds 30 s of baseline and $t$ seconds of incident, so $e(5m) \approx 0.33\, t / 300$, which exceeds 0.144 when $t > 131$ s, about 10:02:41 for a steady third. The model timeline's 10:02:31 has the slow share rising above a third as the queue grows; either way it is well inside 300 s. Under prod windows, $e(1h) \approx 0.33 \cdot 160 / 3600 = 0.015$: no page, which is the point of the drill profile.

The availability SLO does not move. Failover keeps 5xx near zero, and the 11 broken streams were already 200 responses when they failed. That gap becomes an action item: an SLI that counts streams ending in `error`.

## The incident: what is injected, detected, and resolved

The drill runs in four steps:

| Step | What happens |
|---|---|
| load | the load generator of [An open-loop load generator](../../ai-engineering/23-model-serving/11-load-generator.md) runs through the gateway NodePort: `--target <gateway_url> --rate 20 --duration 900s --seed 1` |
| inject | 30 s into the load (a baseline first), one random Ready pod of the decode Deployment (for example `deploy/<system>-engine-decode`) is deleted with `--grace-period=0`; which one is chosen from a seed |
| detect | `TTFTBudgetBurnFast` (drill profile, [`rules/slo-drill.yaml`](../../../../course/solutions/entry/deploy/observability/rules/slo-drill.yaml)) must fire within 300 s of the kill |
| page | *TTFTBudgetBurnFast (drill profile): time to first token is burning the error budget; some users report streams that stop mid-sentence* |

The incident is resolved when all of these hold at the end of the run:

| Condition | Holds for |
|---|---|
| the decode rollout is complete (every replica updated and available) | within 120 s |
| `histogram_quantile(0.95, sum(rate(gen_ai_server_time_to_first_token_seconds_bucket{job="<system>-gateway"}[5m])) by (le)) < 0.5` | 300 s |
| `sum(rate(http_server_request_duration_seconds_count{job="<system>-gateway",http_response_status_code=~"5.."}[5m])) / sum(rate(http_server_request_duration_seconds_count{job="<system>-gateway"}[5m])) < 0.005` | 120 s |
| the v1 gateway API smoke cases pass through the gateway | at the end |
| a trace in the window has `gateway.proxy` from `<system>-gateway` and spans from `<system>-engine` | evidence for the timeline |
| `docs/runbooks/TTFTBudgetBurnFast.md` (Symptoms, Diagnosis, Mitigation) and `docs/postmortems/<date>-kill-decode.md` (the seven sections) exist | at the end |
| all within 45 minutes | |

The runbook path is the one the alert's `runbook` annotation links to, so the page leads straight to it.

## Working the incident step by step

### Reasoning from first principles

The page is a TTFT burn, so the question is "why are first tokens late?" Late first tokens with almost no errors mean requests are waiting, not failing, and waiting means demand exceeds capacity somewhere. So find the lost capacity. A pod that is seconds old or `READY 0/1`, or a `Killing` event, names the replica that went away. Its peers now carry $\lambda$ alone; with $\rho > 1$ their queues must grow.

Confirm with the queue, not the latency. Queue depth rising on the survivors with unchanged load distinguishes lost capacity from a slow model or a bad release. Then check that the gateway is doing its part: failover lines before the first byte, `upstream_failed` error events after it, and no spliced or truncated streams. One `trace_id` from a failed line shows which engine the request reached. Rule out the other common cause, a release: a new decode revision at the start of the burn points at the rollout instead.

Mitigation follows from $t_r$. Kubernetes is already replacing the pod. If the survivors can hold for $t_r$, the right action is to watch the rollout; if not, add a replica now. Never restart the gateway: that kills every in-flight stream on every engine. The incident is over when the SLO says so, not when the pod is Running: the burn under 1, the alert resolved, the resolve conditions held. And the root cause is capacity during replacement, not the kill. Pods die every week; the action items must make one loss survivable.

### Diagnosis

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

Step 2's two queries are the "Queue depth and running sequences" panel of the serving dashboard, so in practice diagnosis starts there, with the TTFT heatmap beside it: the band above 500 ms appears at the kill.

### Mitigation

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

Confirm with the SLO: the TTFT burn stat under 1, the alert resolved (the short window clears first), and the v1 smoke cases passing through `http://127.0.0.1:30080`.

### The model runbook

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

Diagnosis is the four steps above and Mitigation the three branches. The full runbook is [`TTFTBudgetBurnFast.md`](../../../../course/solutions/docs/runbooks/TTFTBudgetBurnFast.md).

### The model postmortem

The numbers in the worked timeline are the model postmortem's. Its root cause is a property of the system, with the number that proves it:

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

The full postmortem is [`2026-10-09-kill-decode.md`](../../../../course/solutions/docs/postmortems/2026-10-09-kill-decode.md).

### Why this solution

Waiting is the mitigation because Kubernetes already does the right thing; the responder's value is confirming it and adding capacity if $t_r$ is too long. Every manual action during an incident carries its own risk, so the runbook makes "watch `rollout status`" an explicit step rather than a non-action.

The fix is three replicas, not a faster replacement alone. With $\mu = 12$ and $\lambda = 20$, two survivors give $\rho = 0.83$: within threshold. Cutting $t_r$ shortens the burn but does not remove it; headroom removes it.

And the gateway errors after the first byte, never splices. A truncated or spliced stream looks like a correct answer and is wrong; an explicit error is honest and lets the client retry.

## What can go wrong

Each of these pitfalls either hides the incident or makes it worse:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| Only the prod rules installed | the page never comes | the drill profile's rules are applied before the drill |
| The gateway retries after the first byte | streams with text from two engines that read plausibly and are wrong | the gateway's commit rule: after the first byte, an SSE `error` event ends the stream |
| Restarting the gateway to "clear" it | every in-flight stream on every engine dies; the 5xx ratio jumps | the runbook's mitigation never touches the gateway |
| Declaring victory when the pod is Running, not Ready | TTFT still high; the incident is not over | verification uses `rollout status` and the SLO, held for 300 s |
| A postmortem that blames the kill | no action item about capacity during replacement; the next node drain repeats the incident | the model root cause is capacity during replacement, with replicas, a PDB, and warm-up as action items |

> **Warning:** Restarting the gateway feels like a reset, but the gateway holds every in-flight stream on every engine. It turns a TTFT burn into an availability outage.

## Where this leads

This drill exercised the alert and drill profile of [SLOs and multi-window burn-rate alerts](../11-observability/04-slos-and-burn-rate-alerts.md) and the dashboard of [Dashboards as code for the serving path](../11-observability/05-dashboards-as-code.md), where the diagnosis started. The same capacity reasoning returns in [A noisy neighbor](10-noisy-neighbor.md), where one tenant floods the gateway, and the same load and SLO checks frame [The KV format v2 migration](05-kv-v2-migration.md), a rolling KV format migration.

### Further reading

Each simple piece here has a production equivalent:

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| a scripted drill with one injector | Chaos Mesh, LitmusChaos, Gremlin | scheduled experiments, blast-radius limits, many fault types | [Chaos Mesh](https://chaos-mesh.org/) |
| a fixed replica count | HPA or KEDA on queue depth, PodDisruptionBudgets | capacity that follows load; evictions that never take two pods | [PDB](https://kubernetes.io/docs/tasks/run-application/configure-pdb/) |
| fail before first byte, error after | request hedging, KV cache migration between replicas | resume a sequence elsewhere without splicing | vLLM and SGLang disaggregation designs |
| a game day run alone | team game days with an incident commander | roles, communication, the human side | *SRE Workbook*, ch. 9 Incident Response |

- *Site Reliability Engineering* (Google), ch. 13 Emergency Response and ch. 22 Addressing Cascading Failures ([free](https://sre.google/sre-book/table-of-contents/)).
- *Chaos Engineering* (Rosenthal and Jones), ch. 3.

## Key terms

- **Utilization ($\rho$)**: arrival rate divided by total service rate; above 1, queues grow without bound.
- **Commit rule**: before the first byte, retry elsewhere; after it, end the stream with an SSE `error` event.
- **Splicing**: continuing a stream from another replica, which produces plausible but wrong text.
- **Drill profile**: burn-rate windows compressed 12x (5 min and 25 s) so a short drill can page.
- **Capacity during replacement**: the headroom the remaining replicas have while a lost one is rescheduled and warmed up.
- **PodDisruptionBudget**: a limit on how many pods voluntary evictions may take at once.

## Summary

- Losing one of two decode replicas halves capacity until the replacement is Ready: the survivor queues, TTFT climbs, and the TTFT budget burns even though almost nothing fails outright.
- The gateway's commit rule retries before the first byte and ends with an SSE `error` event after it, never truncating and never splicing another engine's tokens.
- The drill profile compresses the burn-rate windows 12x so the page arrives within the drill's 300 s budget; under prod windows the same incident would not page at all.
- Mitigation is mostly waiting correctly: confirm the replacement with `rollout status`, add capacity if it is not enough, and verify with the SLO.
- The postmortem's action items are about capacity during replacement: replicas, disruption budgets, warm-up time, and an SLI for mid-stream errors.

## Review questions

1. Why does this incident burn the TTFT budget but not the availability budget?
   <details><summary>Answer</summary>Failover before the first byte turns would-be failures into slower successes, and the streams that broke after the first byte had already returned 200. Almost every request is a 200; many are late. The 5xx SLI cannot see either effect.</details>
2. Why would the same incident not page under the prod profile?
   <details><summary>Answer</summary>The fast alert needs $e(1h) > 0.144$. A third of requests slow for 160 s gives $e(1h) \approx 0.33 \cdot 160 / 3600 \approx 0.015$, an order of magnitude short. The drill profile's 5-minute long window can be dominated by a 2-minute event.</details>
3. Why must the gateway not continue a broken stream on the other decode replica?
   <details><summary>Answer</summary>The other replica does not hold this sequence's KV cache, so the tokens it generates come from a different state. The client would receive text that reads plausibly and is wrong, with no signal that anything happened.</details>
4. Why is "the pod is Running" not the end of the incident?
   <details><summary>Answer</summary>Running means the container started; Ready means it passed readiness after loading the model. Until then it takes no traffic and the survivor's queue still grows. The incident ends when the SLO conditions hold, which is measured, not inferred.</details>
5. With $\lambda = 20$ and $\mu = 12$, what is the survivor's utilization after the kill, and how does a third replica change the picture?
   <details><summary>Answer</summary>One survivor gives $\rho = 20/12 \approx 1.67$, so its queue grows by about 8 requests a second. With three replicas, losing one leaves two, $\rho = 20/24 \approx 0.83$, which stays within the threshold for the whole replacement time.</details>
