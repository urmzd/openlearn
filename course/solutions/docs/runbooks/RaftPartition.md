# Runbook: RaftPartition

Owner: the durable control-plane on-call. This procedure requires the optional
three-replica Raft deployment from `dur.10` and `MS-durable-ha`.

## Symptoms and initial evidence

A partition separates a durable replica from its peers. Requests through a
minority can fail or time out, leadership can change, and replication can lag.
Record the incident start, affected workflows, replica identities, current term,
leader, commit index, and the last acknowledged workflow event before changing
network rules.

```sh
kubectl -n forge get pods -o wide
kubectl -n forge get statefulset forge-durable
kubectl -n forge get events --sort-by=.lastTimestamp
kubectl -n forge logs forge-durable-0 --since=10m
kubectl -n forge logs forge-durable-1 --since=10m
kubectl -n forge logs forge-durable-2 --since=10m
```

Check peer DNS, connectivity on the Raft port, NetworkPolicies, node health,
and storage errors. Compare all three replicas' terms and commit positions;
a process being alive alone does not establish quorum.

## Containment

1. Identify the connected majority. Route control-plane requests to that group
   and pause new submissions if a majority cannot be established.
2. Preserve every replica's WAL and persistent volume. Keep the configured
   quorum and membership intact while diagnosing the partition.
3. Isolate a follower first in a drill. Then isolate the current leader from
   quorum and record that the minority rejects writes rather than acknowledging
   uncommitted events. Track each client request's idempotency key and response.

## Recovery

1. Remove the injected network rule or repair the failed link. Restore peer
   connectivity before considering a membership change.
2. Wait for one leader to hold a stable term and for followers to catch up with
   the committed log. Inspect all replica logs for storage corruption or gaps.
3. Read the affected workflow histories through the client and compare them
   with the recorded successful acknowledgements. Reissue uncertain requests
   with the same idempotency keys so recovery cannot duplicate their effects.
4. Submit a small canary workflow, wait for completion, and read it through
   each replica endpoint. Confirm the workers drain the outstanding queue.

## Success criteria and escalation

Attach the client history, partition schedule, per-replica logs, and the
linearizability check result to the drill report. Every acknowledged write must
remain present in order after recovery; minority writes must have failed, and
replicas must agree on the committed history. Record convergence time and the
final workflow result before closing the incident.

Escalate to the durable implementation owner if quorum remains unstable,
committed entries disagree, an acknowledged event is missing, or a WAL reports
corruption. Preserve the affected volumes and collected evidence for analysis.
