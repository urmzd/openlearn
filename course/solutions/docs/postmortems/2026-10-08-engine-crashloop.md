# Postmortem: engine crashloop after a bad rollout

Drill `ops.00`. Blameless: this document is about the system and the process,
not about who typed the command.

## Summary

A change to the engine Deployment pointed `--model-dir` at `/missing`. The new
engine pod exited at startup on every attempt, Kubernetes restarted it with
growing back-off, and the gateway answered every completion with
`503 no_capacity` for 14 minutes. Rolling the Deployment back restored service.

## Impact

- 14 minutes with no completions served; every request through the gateway
  failed fast with `503 no_capacity`.
- No data was lost: the engine holds no state between requests.
- The old pod was replaced before the new one became ready, so there was no
  healthy replica to fall back on.

## Timeline

All times UTC.

| Time | Event |
|---|---|
| 14:02 | The engine Deployment is patched; its args now name `/missing` |
| 14:02 | The new pod starts, exits with code 1, and is restarted |
| 14:03 | Pager symptom: completions through the gateway fail with 503 |
| 14:05 | `kubectl get pods` shows the engine in `CrashLoopBackOff`, restarts 3 |
| 14:08 | `kubectl logs --previous` shows `cannot read /missing/config.json` |
| 14:10 | `kubectl rollout history` shows revision 4 changed the args |
| 14:14 | `kubectl rollout undo`; the rollout completes; `/readyz` is 200 again |
| 14:16 | A chat completion through the gateway (`curl -s -o /dev/null -w '%{http_code}' .../v1/chat/completions`) answers 200 again |

## Root cause

The engine resolves `--model-dir` once, at startup, and exits when
`config.json` is missing. The Deployment was changed outside Helm, so nothing
rendered or reviewed the new args before they reached the cluster. The
Deployment's rolling update removed the only healthy pod before its
replacement was ready, because the engine has one replica and no readiness
gate that would have held the old pod.

## Detection

Manual. Pass 1 has no Prometheus and no alert, so the first signal was a user
seeing 503s. The time to detect was about one minute only because the drill
paged at once. In production, nothing would have paged.

## Resolution

`kubectl rollout undo deploy/<system>-engine` restored the previous args. The
chart was then re-applied with `helm upgrade --install`, so the live object
matches the chart again.

## Action items

| Action | Type | Owner | Status |
|---|---|---|---|
| Write `docs/runbooks/engine-crashloop.md` | mitigate | engine | done |
| Make the engine log the resolved model path and exit with a clear message | detect | engine | done |
| Set `maxUnavailable: 0` on the engine rollout so the old pod stays until the new one is Ready | prevent | deploy | open |
| Alert on gateway `/readyz` failing for 1 minute (arrives with Prometheus in Pass 7) | detect | observability | open |
| Change the cluster only through `helm upgrade`, never `kubectl edit` or `patch` | prevent | team | open |
