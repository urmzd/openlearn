# Runbook: TTFTBudgetBurnFast

Owner: the serving path (`<system>-gateway`, `<system>-engine`). Linked from
the alert's `runbook` annotation (obs.03). Written after drill `ops.01`.
The alert means: over both its long and its short window, more than 14.4
times the allowed share of requests waited longer than the TTFT threshold for
their first token. At that rate the 30-day error budget is gone in about two
days.

## Symptoms

- The page `TTFTBudgetBurnFast` (`severity: page`), with `slo_profile` `prod`
  (or `drill` during a drill).
- The serving dashboard (obs.04): the TTFT heatmap grows a band above the
  threshold, the TTFT burn stat turns red, queue depth rises on some engine
  pods.
- Users: slow starts; if a decode pod died, some streams ended early with an
  SSE `error` event (`"code": "upstream_failed"`), never with spliced text.

## Diagnosis

1. Which replicas are serving, and did one restart?

   ```bash
   kubectl -n <system> get pods -l app.kubernetes.io/name=<system>-engine -o wide
   kubectl -n <system> get events --sort-by=.lastTimestamp | tail -20
   ```

   A pod with `AGE` of seconds, `READY 0/1`, or a `Killing` event is the
   replica that went away. Its peers are carrying its share.

2. Is the remaining capacity saturated? Queue depth and running sequences
   per pod:

   ```promql
   sum by (pod) (tl_engine_queue_depth)
   sum by (pod) (tl_engine_active_sequences)
   ```

   A growing queue on the survivors with the load unchanged means lost
   capacity, not a slow model.

3. Is the gateway routing around the hole? In the gateway logs, failover
   before the first byte and the stream errors after it:

   ```bash
   kubectl -n <system> logs deploy/<system>-gateway --since=15m | grep -E '"failover"|"upstream_failed"'
   ```

   Take one `trace_id` from a failed line and open it in Grafana (Tempo):
   the trace shows which engine the request reached and where it ended.

4. Rule out a release: `kubectl -n <system> rollout history deploy/<system>-engine-decode`.
   A new revision at the start of the burn points at the rollout instead.

## Mitigation

- A lost replica: Kubernetes replaces it on its own. Watch it come back:
  `kubectl -n <system> rollout status deploy/<system>-engine-decode`. The
  burn stops when the new pod is Ready and the queue drains.
- If the survivors stay saturated, add capacity now:
  `kubectl -n <system> scale deploy/<system>-engine-decode --replicas=3`
  (back to the chart value afterwards with `helm upgrade`).
- If a rollout caused it: `kubectl -n <system> rollout undo deploy/<system>-engine-decode`.
- Confirm: the TTFT burn stat under 1, the alert resolved (the short window
  clears first), and a chat completion through the gateway answers 200, plain
  and streamed: `curl -s -H "Authorization: Bearer $TL_API_KEY" -H 'Content-Type: application/json' -d '{"model":"<model>","messages":[{"role":"user","content":"hi"}]}' http://127.0.0.1:30080/v1/chat/completions`.
