# Drill: Raft partition (optional)

## Overview

| | |
|---|---|
| **Module** | `ops.12` · drill · ops, docs · Pass 11 · 1 to 2 h |
| **Solution** | the model runbook [`docs/runbooks/RaftPartition.md`](../../../../course/solutions/docs/runbooks/RaftPartition.md), with the reasoning behind each step in section 5 |
| **Needs** | none |
| **Used by** | no code call site: an optional drill for the three-replica durable system (`dur.10`) |

## Key Takeaways

- Raft commits an entry only when a majority has it, so a minority partition, including an isolated leader, cannot acknowledge writes: they must fail or time out.
- After healing, replicas converge on one leader with a stable term, and every acknowledged write is present, in order, with no gaps.
- Heal the network before any membership change: promoting a replacement during a partition risks two majorities' worth of confusion and lost commits.

## How to read this chapter

This chapter is a worked drill. Sections 1 to 3 state the risk, the principles, and the drill scenario. Section 4 lists what a correct runbook must cover. Section 5 derives the procedure from first principles, gives the model runbook, and explains why each step is the right one. Try to write the procedure yourself from sections 2 and 3 before reading section 5.

---

## 1. Why now

A network partition can create competing Raft leadership and threaten durable ordering. The drill is optional because it requires the HA module.

## 2. Principles

Use a deterministic partition harness, preserve quorum rules, and verify linearizable reads and acknowledged writes. Heal the network before promoting a replacement.

## 3. Worked example

Partition one follower, then isolate the leader from quorum. Check that minority writes are rejected; heal links and confirm committed entries converge without gaps.

## 4. What a correct runbook must cover

| Case | What the runbook must contain | Why it matters |
|---|---|---|
| Initial evidence | start time, replica identities, term, leader, commit index, last acknowledged event per client | the baseline for the linearizability check |
| Partition schedule | isolate a follower first, then the leader from quorum | exercises both cases |
| Minority behaviour | writes through the minority fail rather than acknowledging | the safety property |
| Recovery | heal links; one stable leader; followers caught up; no membership change first | convergence |
| Verification | every acknowledged write present in order; canary workflow completes; linearizability check result attached | done means proven |

## 5. The solution, step by step

### 5.1 From first principles

1. **What does Raft guarantee?** An entry is committed only once a majority (2 of 3) stores it, and a leader needs a majority to stay leader. So a partition can only make progress on the majority side.
2. **Isolating a follower** leaves a majority with the leader: writes continue, the follower falls behind and catches up after healing.
3. **Isolating the leader** leaves it in a minority: it cannot commit, so writes through it must fail or time out, while the majority elects a new leader in a higher term. Recording that minority writes were rejected is the drill's key evidence.
4. **Clients may be uncertain.** A write that timed out may or may not have committed; reissuing it with the same idempotency key makes the retry safe.
5. **Heal, then verify.** After connectivity returns, the old leader steps down on seeing the higher term, and logs converge. Every acknowledged write must be present in order; changing membership before healing could remove the replicas that hold committed entries.

### 5.2 The model runbook

The solution's runbook, [`RaftPartition.md`](../../../../course/solutions/docs/runbooks/RaftPartition.md), follows the chain above: record term, leader, commit index, and the last acknowledged event per client before touching the network; contain by routing to the connected majority and preserving every replica's WAL; isolate a follower, then the leader, and record that the minority rejects writes; heal before any membership change; wait for a stable leader and caught-up followers; compare the workflow histories with the recorded acknowledgements and reissue uncertain requests with the same idempotency keys; finish with a canary workflow read through each replica. Its success criteria:

```markdown
Every acknowledged write must
remain present in order after recovery; minority writes must have failed, and
replicas must agree on the committed history. Record convergence time and the
final workflow result before closing the incident.
```

### 5.3 Why this solution

- **Majority quorum** makes split-brain commits impossible: two disjoint majorities cannot exist in three replicas.
- **Idempotent retries** turn the client's uncertainty after a timeout into a safe operation.
- **Why not promote a new replica during the partition?** Membership changes need the cluster's agreement; doing one under a partition can leave committed entries only on replicas that are being removed.

### 5.4 Explain it back

1. Why must an isolated leader reject writes?
   <details><summary>Answer</summary>It cannot reach a majority, so it cannot commit; acknowledging would promise durability Raft cannot deliver, and the majority side may elect a new leader whose log overrides uncommitted entries.</details>
2. How does the old leader learn it is no longer leader after healing?
   <details><summary>Answer</summary>It receives messages carrying a higher term from the new leader and steps down to follower, then its log is brought in line with the committed history.</details>
3. Why reissue uncertain requests with the same idempotency key?
   <details><summary>Answer</summary>A timed-out write may have committed on the majority. Reusing the key returns the existing result instead of applying the effect twice.</details>

## 6. Pitfalls

| # | Pitfall | Symptom | How the solution avoids it |
|---|---|---|---|
| 1 | Changing membership before healing the network | committed entries live only on replicas being removed; history diverges | recovery heals links and waits for a stable leader first |
| 2 | Treating a live process as a healthy replica | a minority replica keeps accepting client connections and timing out | diagnosis compares terms and commit positions across all three replicas |
| 3 | Retrying uncertain writes with new keys | duplicate workflow effects after recovery | uncertain requests are reissued with the same idempotency keys |

## 7. Where it's used next

| Direction | Module | Connection |
|---|---|---|
| Back | `ops.11` | A prerequisite: the contract or evidence this runbook relies on. |
| Back | `dur.10` | A prerequisite: the contract or evidence this runbook relies on. |

## Going further

| Resource | What to inspect |
|---|---|
| [Incident drill catalog](README.md) | Compare this inject with the adjacent failure drills and use the same containment and verification fields. |
