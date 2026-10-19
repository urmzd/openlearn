# Postmortem: the durable server was SIGKILLed in a loop during a corpus build

Drill `ops.02`. Blameless: this document is about the system and the
process, not about who typed the command. A model postmortem: the times and
counts show the shape of this incident; yours come from your dashboard,
`{ctl} wf describe`, and `ol drill end`.

## Summary

While CorpusBuild `drill-kill9` cleaned the MS-corpus fixture, the durable
server's pod `<system>-durable-0` was killed with SIGKILL five times, 30 s
apart. Each time the StatefulSet recreated it on the same WAL claim, the
server truncated a torn last record and recovered its log, and the
activities whose leases had been live were delivered again after their
visibility timeout. `DurableRedeliveriesSpike` paged 80 s after the first
kill. No acknowledged event was lost, no activity's effect happened twice,
and the build completed with the same 169 documents, 3 shards, and token
counts as a clean run, 3 minutes 40 seconds later than a clean run.

## Impact

- The build took 6 min 10 s instead of 2 min 30 s.
- 14 activity tasks were delivered twice and 3 three times; each rerun found
  its work directory and resumed or re-emitted its `DONE.json`, so the
  shards and the ledger hold every row once (`{corpus} ledger verify`
  passed).
- No workflow failed and nothing was dead-lettered.
- New workflow starts failed with `UNAVAILABLE` for about 8 s per restart
  (40 s in total); the CLI's retry with backoff hid most of it.

## Timeline

All times UTC.

| Time | Event |
|---|---|
| 10:00:00 | `{ctl} data build ... --id drill-kill9` starts CorpusBuild |
| 10:00:05 | `ol drill start durable-kill9` |
| 10:00:25 | first SIGKILL of `<system>-durable-0` (fetch and filter done, dedup running) |
| 10:00:33 | the pod is Ready again: recovery truncated 1 torn record, replayed 412 records |
| 10:00:55 to 10:02:25 | four more kills, 30 s apart |
| 10:01:45 | `DurableRedeliveriesSpike` fires for queue `data` (6 redeliveries in 2 min, held 20 s) |
| 10:01:50 | runbook step 1: `RESTARTS 3` on `<system>-durable-0`, `Killing` events, no OOM |
| 10:02:00 | runbook step 2: each start logged a torn-tail truncation, no `ErrCorrupt` |
| 10:02:33 | the last restart; no more kills |
| 10:04:40 | the alert resolves: no redelivery in the last 2 minutes |
| 10:06:10 | CorpusBuild completes |
| 10:07:30 | the same start returns the finished run with the clean-run numbers; the ledger verifies |
| 10:09:00 | `ol drill end` |

## Root cause

The durable server is a single replica, so every kill is a full outage of
the control plane for the time it takes to start and recover the log (about
8 s here). Activities whose worker could not reach the server to heartbeat
or complete kept their leases until the visibility timeout (30 s), and were
then delivered again. That is the design working: leases make a lost worker
or server safe, at the cost of redoing work. The drill's kills are the
trigger; the exposure is one replica and a 30 s visibility timeout on
activities that do not checkpoint.

## Detection

`DurableRedeliveriesSpike` fired 80 s after the first kill: the leases cut
by the first kill expired 30 s later (3 redeliveries), the second kill's 30 s
after that took the 2-minute increase past 5, and the alert's `for` added
20 s. The dashboard's
redeliveries rate and the durable pod's restart count told the story in one
screen. No alert fired on the restarts themselves; kube-state-metrics has
them, and an alert on durable restarts would have paged 40 s earlier.

## Resolution

Nothing had to be changed: the StatefulSet restarted the server each time and
the workers reconnected with backoff. The responder confirmed the server and
workers were Ready, waited for the queues to drain, re-ran the same start
(`started: false`, `status: COMPLETED`, 169 documents, 3 shards, 75164 and
6868 tokens), and verified the ledger.

## Action items

| Action | Type | Owner |
|---|---|---|
| Alert on durable server restarts (`kube_pod_container_status_restarts_total`), not only on their effect | detect | control plane |
| Heartbeat from the long corpus stages so a redelivery resumes instead of redoing a stage | mitigate | data pipeline |
| Run the durable server with three replicas (dur.10, optional) so one kill is not an outage | prevent | control plane |
| Make the CLI's start retry visible (a log line per retry) so users see an outage, not a slow command | detect | CLI |
