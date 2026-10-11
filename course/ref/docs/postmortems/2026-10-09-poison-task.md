# Postmortem: a poison task landed in the dead-letter queue

Drill `ops.03`. Blameless: this document is about the system and the
process, not about who typed the command. A model postmortem: the times and
counts show the shape of this incident; yours come from your dashboard,
`{ctl} wf dlq list`, and `ol drill end`.

## Summary

A worker rollout shipped a fault-injection setting to production:
`TL_FAILPOINTS=dur/activity/poison/shard-00001=panic`. The next CorpusBuild,
`drill-poison`, ran every shard task normally except `shard-00001`, which
panicked on each of its 5 attempts and was dead-lettered.
`DurableDeadLetters` paged 30 s later. The responder found the setting in the
worker Deployment's environment, removed it with a rollout, redrove the one
dead letter, and the build completed 9 minutes after it started.

## Impact

- CorpusBuild `drill-poison` was stuck for 6 min 20 s, from the dead letter
  to the redrive.
- 1 task dead-lettered after 5 attempts; no other task failed.
- No data was lost or duplicated: the panicking attempts never wrote outputs,
  and the redriven attempt produced the same shard a clean run does.

## Timeline

All times UTC.

| Time | Event |
|---|---|
| 10:00:00 | `ol drill start poison-task`: the worker Deployment gains `TL_FAILPOINTS`, a rollout starts |
| 10:00:40 | rollout complete; `{ctl} data build ... --id drill-poison` starts |
| 10:01:30 | `shard-00001` attempt 1 panics (`Panic`, retryable); the other shards complete |
| 10:01:31 to 10:01:46 | attempts 2 to 5 after backoff of 1, 2, 4, and 8 s; each panics |
| 10:01:46 | `shard-00001` moves to the dead-letter queue of `data` |
| 10:02:16 | `DurableDeadLetters` fires (one dead letter for 15 s, plus the scrape) |
| 10:03:00 | runbook step 1: `wf dlq list data` shows `shard-00001`, 5 attempts, `Panic: failpoint dur/activity/poison/shard-00001` |
| 10:04:30 | runbook step 2: the worker's environment has `TL_FAILPOINTS`; `rollout history` shows the revision that added it |
| 10:05:10 | `kubectl set env deploy/<system>-worker TL_FAILPOINTS-`; rollout complete at 10:05:50 |
| 10:06:05 | `wf dlq redrive data <task id>`; the attempt succeeds |
| 10:09:00 | CorpusBuild completes; the alert resolves |
| 10:11:00 | `ol drill end` |

## Root cause

A configuration meant for tests (a failpoint that panics one activity)
reached the production worker Deployment. Nothing stopped it: the chart
accepts any environment variable, and the rollout had no check for fault or
debug settings. The retry policy did what it should with a deterministic
failure: five attempts, then the dead-letter queue, so the problem stayed
contained to one task instead of retrying forever.

## Detection

`DurableDeadLetters` fired 30 s after the dead letter. Before that, five
`Panic` failures in 16 s were visible only in `wf describe`: no alert covers
a high activity failure rate, which would have paged a minute earlier.

## Resolution

The fix was the configuration, not the task: remove `TL_FAILPOINTS` from the
worker Deployment, wait for the rollout, then redrive exactly the dead
letter that failed. Redriving first would have produced five more panics and
a second dead letter.

## Action items

| Action | Type | Owner |
|---|---|---|
| Reject `TL_FAILPOINTS` in the worker chart's values schema outside test profiles (a policy test over `helm template`) | prevent | platform |
| Alert on the activity failure rate per type, not only on dead letters | detect | control plane |
| Add "what changed?" (`rollout history`, environment diff) as step 2 of every control-plane runbook | process | on-call |
| Show `last_failure` in the dead-letter panel so the page carries the cause | detect | observability |
