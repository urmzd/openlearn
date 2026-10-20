# A Raft partition (optional)

A network partition can create competing Raft leadership and threaten durable ordering. This drill applies to the three-replica durable system of [Raft high availability for the durable log](../07-durable-orchestration/10-raft-ha.md) in *Systems Engineering*, and is optional because that section is. If you built it, this is where you find out whether the safety properties hold on a real cluster, not only in the test harness.

The single-replica drills, ending with [The event log disk fills up](12-eventlog-disk-full.md), showed a log that never loses acknowledged work on one server. Raft extends that promise to three: an entry is committed only when a majority has it, so a minority partition, including an isolated leader, cannot acknowledge writes; they must fail or time out. After healing, replicas converge on one leader with a stable term, and every acknowledged write is present, in order, with no gaps.

The section states the principles and the drill scenario, lists what a correct runbook must cover, and derives the procedure. The solution is the model runbook [`docs/runbooks/RaftPartition.md`](../../../../course/solutions/docs/runbooks/RaftPartition.md). Try to write the procedure yourself before you read the solution.

> **Learning objectives**
> - Explain why only the majority side of a partition can commit, and what that means for an isolated leader.
> - Record the evidence (term, leader, commit index, last acknowledged event per client) that a linearizability check needs.
> - Make client uncertainty after a timeout safe with idempotent retries.
> - Heal before any membership change, and verify that every acknowledged write survives in order.

## Principles

Use a deterministic partition harness, preserve quorum rules, and verify linearizable reads and acknowledged writes. Heal the network before promoting a replacement: promoting one during a partition risks two majorities' worth of confusion and lost commits.

## The drill scenario

Partition one follower, then isolate the leader from quorum. Check that minority writes are rejected; heal links and confirm committed entries converge without gaps.

## What a correct runbook must cover

A runbook for a partition is complete when it answers five questions:

| Case | What the runbook must contain | Why it matters |
|---|---|---|
| Initial evidence | start time, replica identities, term, leader, commit index, last acknowledged event per client | the baseline for the linearizability check |
| Partition schedule | isolate a follower first, then the leader from quorum | exercises both cases |
| Minority behaviour | writes through the minority fail rather than acknowledging | the safety property |
| Recovery | heal links; one stable leader; followers caught up; no membership change first | convergence |
| Verification | every acknowledged write present in order; canary workflow completes; linearizability check result attached | done means proven |

## Deriving the procedure

### Reasoning from first principles

What does Raft guarantee? An entry is committed only once a majority (2 of 3) stores it, and a leader needs a majority to stay leader. So a partition can only make progress on the majority side.

Isolating a follower leaves a majority with the leader: writes continue, and the follower falls behind and catches up after healing.

Isolating the leader leaves it in a minority. It cannot commit, so writes through it must fail or time out, while the majority elects a new leader in a higher term. Recording that minority writes were rejected is the drill's key evidence.

Clients may be uncertain. A write that timed out may or may not have committed; reissuing it with the same idempotency key makes the retry safe.

Heal, then verify. After connectivity returns, the old leader steps down on seeing the higher term, and logs converge. Every acknowledged write must be present in order. Changing membership before healing could remove the replicas that hold committed entries.

### The model runbook

The solution's runbook, [`RaftPartition.md`](../../../../course/solutions/docs/runbooks/RaftPartition.md), follows that chain step by step. Before touching the network, record term, leader, commit index, and the last acknowledged event per client. Contain by routing to the connected majority and preserving every replica's WAL. Isolate a follower, then the leader, and record that the minority rejects writes. Heal before any membership change, and wait for a stable leader and caught-up followers. Compare the workflow histories with the recorded acknowledgements and reissue uncertain requests with the same idempotency keys. Finish with a canary workflow read through each replica. Its success criteria:

```markdown
Every acknowledged write must
remain present in order after recovery; minority writes must have failed, and
replicas must agree on the committed history. Record convergence time and the
final workflow result before closing the incident.
```

### Why this solution

Majority quorum makes split-brain commits impossible: two disjoint majorities cannot exist in three replicas. Idempotent retries turn the client's uncertainty after a timeout into a safe operation. And promoting a new replica during the partition is exactly the wrong move: membership changes need the cluster's agreement, and doing one under a partition can leave committed entries only on replicas that are being removed.

## What can go wrong

Three mistakes break the guarantees the drill is meant to prove:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| Changing membership before healing the network | committed entries live only on replicas being removed; history diverges | recovery heals links and waits for a stable leader first |
| Treating a live process as a healthy replica | a minority replica keeps accepting client connections and timing out | diagnosis compares terms and commit positions across all three replicas |
| Retrying uncertain writes with new keys | duplicate workflow effects after recovery | uncertain requests are reissued with the same idempotency keys |

## Where this leads

The replicated log tested here is the one built in [Raft high availability for the durable log](../07-durable-orchestration/10-raft-ha.md), and this drill closes the chapter's sequence after [The event log disk fills up](12-eventlog-disk-full.md). To compare this inject with the other drills and reuse the same containment and verification fields, see the [chapter opening](README.md).

## Key terms

- **Network partition**: a split that leaves some replicas unable to reach others.
- **Majority quorum**: the 2 of 3 replicas that must store an entry before it is committed.
- **Term**: Raft's election epoch; a leader that sees a higher term steps down.
- **Linearizability check**: verification that every acknowledged write is present, in order, as if executed on a single copy.
- **Membership change**: adding or removing replicas, which must not happen while the cluster is partitioned.

## Summary

- Raft commits an entry only when a majority has it, so a minority partition, including an isolated leader, cannot acknowledge writes.
- Isolating a follower leaves writes flowing; isolating the leader forces a new election in a higher term on the majority side.
- Timed-out writes are uncertain; reissuing them with the same idempotency key makes retries safe.
- Heal before any membership change, then verify one stable leader and every acknowledged write present in order.

## Review questions

1. Why must an isolated leader reject writes?
   <details><summary>Answer</summary>It cannot reach a majority, so it cannot commit; acknowledging would promise durability Raft cannot deliver, and the majority side may elect a new leader whose log overrides uncommitted entries.</details>
2. How does the old leader learn it is no longer leader after healing?
   <details><summary>Answer</summary>It receives messages carrying a higher term from the new leader and steps down to follower, then its log is brought in line with the committed history.</details>
3. Why reissue uncertain requests with the same idempotency key?
   <details><summary>Answer</summary>A timed-out write may have committed on the majority. Reusing the key returns the existing result instead of applying the effect twice.</details>
