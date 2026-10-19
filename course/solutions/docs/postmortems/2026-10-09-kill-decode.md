# Postmortem: a decode worker died mid-stream

Drill `ops.01`. Blameless: this document is about the system and the process,
not about who typed the command. A model postmortem: the times and counts
show the shape of this incident; yours come from your loadgen report and
your dashboard.

## Summary

Under a steady 20 requests per second, one of the two decode engine pods was
deleted with no grace period. Its 11 in-flight streams ended with an SSE
`error` event. For 2 minutes 40 seconds the surviving decode pod carried the
whole load, its queue grew, and time to first token rose above 500 ms for a
third of requests, which fired `TTFTBudgetBurnFast` (drill profile) 2 minutes
after the kill. The replacement pod became Ready, the queue drained, and the
alert resolved without manual action.

## Impact

- 11 streams ended early with an `error` event (none truncated, none spliced).
- About 1 100 requests over 160 s waited longer than 500 ms for their first
  token; none failed before the first byte, because the gateway failed them
  over to the surviving replica.
- The 5xx ratio stayed at 0.1%, under the availability budget.
- With the prod profile, the same burn would not have paged: the 1 h window
  would have absorbed 160 s. The drill profile exists to see it in minutes.

## Timeline

All times UTC.

| Time | Event |
|---|---|
| 10:00:00 | The game day starts; the loadgen starts at 20 rps |
| 10:00:30 | Decode pod `forge-engine-decode-6c9f...-x2kq` deleted (`--grace-period=0`) |
| 10:00:30 | 11 streams on it end with an SSE `error` event; new requests fail over |
| 10:00:45 | Queue depth on the surviving decode pod passes 30 |
| 10:02:31 | `TTFTBudgetBurnFast` fires (drill profile: 5m and 25s windows over 14.4 x 1%) |
| 10:02:40 | Runbook step 1: a decode pod 2 minutes old, `READY 0/1`, loading the model |
| 10:03:10 | The replacement pod is Ready; the queue drains in 20 s |
| 10:04:05 | The short window clears; the alert resolves |
| 10:09:30 | Resolve checks: TTFT p95 under 500 ms held 300 s and 5xx under 0.5% held 120 s on the dashboard; a chat completion through the gateway answers 200, plain and streamed |

## Root cause

The decode tier has two replicas and no headroom: one replica's load is
more than half of what the other can serve within the TTFT threshold. When a
pod disappears, the replacement needs about 2.5 minutes (image already on the
node, model load and warm-up) before readiness, and for that time the
survivor is over capacity. The deletion itself is ordinary (node drain,
eviction, crash); the gap is capacity during replacement.

## Detection

`TTFTBudgetBurnFast` fired 121 s after the kill, inside the drill's 300 s
budget. The fast window's long half (5 m in the drill profile) took most of
that time to cross 14.4 x the 1% budget; the short window confirmed the burn
was still happening. No availability alert fired: the gateway's failover kept
errors before the first byte at zero, and mid-stream errors are a 200 status
with an `error` event, which the 5xx SLI does not see.

## Resolution

No manual action: Kubernetes replaced the pod, readiness gated traffic until
the model was loaded, and the gateway routed to it from its next heartbeat.
Verified with the resolve checks above and the TTFT heatmap returning
to one band.

## Action items

| Action | Owner | Due |
|---|---|---|
| Run three decode replicas (or set the HPA floor to 3) so one loss stays under the TTFT threshold | serving | next sprint |
| Add a PodDisruptionBudget `minAvailable: 2` for decode so voluntary evictions never take two at once | serving | next sprint |
| Count streams that end with an `error` event in an SLI (`tl_gateway_stream_errors_total`) so mid-stream failures are visible to the availability SLO | observability | next quarter |
| Cache model weights on the node (hostPath under /artifacts) to cut the replacement's warm-up | platform | next quarter |
