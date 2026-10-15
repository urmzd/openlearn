<!-- ol:module dur.10 -->
# Raft HA for the durable log (optional)

## Overview

| | |
|---|---|
| **Module** | `dur.10` · side · Go · Pass 8 · 8 to 12 h (optional) |
| **You build** | `go/durable/raft/raft.go`: a Raft node as a deterministic state machine: `New`, `Tick`, `Step`, `Propose`, `ReadMessages`, `ReadCommitted`, `Status`, with the `Storage` it saves through and the `Message` it sends |
| **Contract** | the package API in section 4 (no Go interface file yet); the wire form [`course/contracts/proto/tl/raft/v1/raft.proto`](../../../../course/contracts/proto/tl/raft/v1/raft.proto), which your server's composition root adapts `Message` to; each entry carries one framed WAL record of [`formats/wal.md`](../../../../course/contracts/formats/wal.md) |
| **Tests** | `course/tests/go/dur_10/`: hand-driven scenarios (every message delivered when the test says) and a seeded fault simulation with a linearizability check; section 4 lists them |
| **Needs** | nothing you must have built: the node treats entries as bytes. Reading: `lang.06` (Go), the single-node log of `dur.01` (what gets replicated), the election demo in [`infrastructure/02-messaging-and-queueing/code/raft_election.go`](../../infrastructure/02-messaging-and-queueing/code/raft_election.go) |
| **Used by** | no module: your `{durable} --replicas 3` entry point runs three nodes over gRPC `tl.raft.v1`, and the optional drill `ops.12` partitions its leader |
| **Milestone** | MS-durable-ha (optional): the durable kill loop against three replicas, leader kills and partitions, a linearizable history |
| **Optional depth** | Ongaro and Ousterhout, [*In Search of an Understandable Consensus Algorithm*](https://raft.github.io/raft.pdf) (free), figure 2 is the whole protocol; Ongaro's [thesis](https://github.com/ongardie/dissertation) ch. 3 and 6 (free); Herlihy and Wing, *Linearizability* (1990); [Porcupine](https://github.com/anishathalye/porcupine) (a linearizability checker); Kleppmann, *Designing Data-Intensive Applications*, ch. 9 |

## Key Takeaways

- One durable server is one disk: when it dies, workflows stop. **Three replicas that agree on one log** keep serving while any one of them is down, because every decision needs only a **majority** (2 of 3), and any two majorities share a member.
- Raft splits agreement into **leader election** (one leader per term, voters only pick a candidate whose log is at least as up to date as theirs) and **log replication** (the leader's log wins; a follower accepts entries only after the entry before them matches).
- An entry is **committed** when the leader of its term has it on a majority. Entries of earlier terms are never committed by counting copies (**figure 8**): they commit when a later entry of the current term does, which is why a new leader appends a **no-op**.
- Everything a node promises (its term, its vote, its log) is **saved before** the message that depends on it is sent. A node that forgets its vote in a restart can elect two leaders in one term.
- Writing the node as a **deterministic state machine** (ticks in, messages out, no goroutines, no clock) is what makes it testable: a seed names a fault schedule, and a failing schedule replays exactly.

## How to work this chapter

```bash
ol start dur.10         # stubs go/durable/raft/raft.go
ol tests dur.10         # the test catalog: scenarios first, then the simulation
ol check dur.10         # go test over the course tests; the exit code is the verdict
ol diff  dur.10         # after passing: your code against the reference
```

Then, if you want the milestone, run three of them in your durable server (section 4, "Wiring it into the server") and `ol milestone MS-durable-ha`.

---

## 1. Why now

Pass 8 built a durable execution engine on one write-ahead log (`dur.01`): every workflow event and queue transition is a record, fsynced before it is acknowledged. That log survives a crash of the process, but not the loss of the machine or its disk, and while the one server restarts, nothing runs. This optional module removes that single point of failure the way etcd, Consul, CockroachDB, and Kafka's KRaft controllers do: three servers keep identical copies of the log and agree, record by record, on what it contains. The work is in this one package; the server gains a mode, `--replicas 3`, in which an append is acknowledged only after Raft commits it.

## 2. Principles

### 2.1 Majorities, terms, and roles

| Symbol | Meaning | Type |
|---|---|---|
| $N$ | number of members (servers) | 3 here |
| $q = \lfloor N/2 \rfloor + 1$ | quorum: the size of a majority | 2 of 3, 3 of 5 |
| $t$ | a **term**: a number that only grows; each term has at most one leader | `uint64`, starts at 0 |
| $\text{vote}$ | the member this node voted for in its current term; 0 = none | `uint64` |
| $\log[i] = (i, t_i, d_i)$ | the $i$-th entry: its index (1-based), the term of the leader that created it, its data | `Entry` |
| $\text{last}$ | the last entry's index and term; $(0, 0)$ for an empty log | |
| $\text{commit}$ | the highest index known committed | `uint64` |
| $E$, $H$ | election timeout base and heartbeat interval, in ticks, $0 < H < E$ | `int` |

Two majorities of the same $N$ members always intersect: $q + q > N$. Every rule below leans on that one fact: if a majority voted for A in term $t$, no other majority can vote for B in term $t$; if a majority holds entry $i$, any future majority contains a member that holds it.

Every node is a **follower**, a **candidate**, or the **leader**. Followers only answer. A follower that hears nothing from a leader for its election timeout becomes a candidate and asks for votes; a candidate with a majority of votes becomes leader for that term. Every message carries the sender's term, and one rule applies to all of them: **a message with a higher term makes the receiver adopt that term and become a follower** (with no vote yet in it). A message with a lower term is stale: a request is refused with the receiver's term (so the sender learns it is behind), a reply is dropped.

### 2.2 Elections

A follower's timeout is drawn uniformly from $[E, 2E)$ ticks and drawn again every time it is reset (on a valid AppendEntries, or when it grants a vote). Random timeouts make it likely that one node times out first and wins before another starts: that is the whole liveness argument.

To campaign, a node increments its term, votes for itself, **saves** $(t, \text{vote})$, and sends RequestVote$(t, \text{last index}, \text{last term})$ to every peer. A voter grants when both hold:

1. it has not voted in this term, or it voted for this same candidate (a retry after a lost reply);
2. the candidate's log is **at least as up to date** as its own: the candidate's last term is greater, or the last terms are equal and the candidate's last index is at least the voter's.

The grant is saved before the reply leaves. Rule 2 is the **election restriction**: a committed entry is on a majority, the winner needs a majority of votes, the two majorities share a voter, and that voter only votes for a log at least as up to date as its own, so the winner holds every committed entry. Comparing the last **term** first matters: a long log from an old term can hold entries that a newer leader overwrote.

### 2.3 Heartbeats

The leader sends AppendEntries to every peer every $H$ ticks, with entries when the peer lags and empty otherwise (a **heartbeat**). Each one resets the follower's election timer. A candidate that receives AppendEntries of its own term learns that someone else won that term and becomes a follower.

### 2.4 Replication and the consistency check

The leader keeps, for each peer $p$: $\text{next}[p]$, the next index to send (initially its last index + 1), and $\text{match}[p]$, the highest index known to be on $p$ (initially 0). AppendEntries carries $(\text{prev index}, \text{prev term})$, the entry just before the new ones, plus the entries and the leader's commit index. The follower:

1. refuses if it has no entry at prev index with prev term. It answers with a **hint**: if its log is shorter, its last index + 1; otherwise the first index of the term it holds at prev index (so the leader skips the whole conflicting term in one round trip);
2. otherwise walks the entries: one it already holds with the same term stays (**never truncate a matching entry**: the message may be an old, shorter copy); at the first one it lacks, or holds with a different term, it deletes that entry and everything after it and appends the rest;
3. sets $\text{commit} = \min(\text{leader commit}, \text{prev index} + \text{len(entries)})$ if that is larger: only entries this message verified;
4. answers success with $\text{match} = \text{prev index} + \text{len(entries)}$.

On a refusal the leader lowers $\text{next}[p]$ (to the hint, never below $\text{match}[p] + 1$) and retries; on success it raises $\text{match}[p]$. The **log matching property** follows by induction: if two logs hold an entry with the same index and term, they are identical up to it, because each AppendEntries extends only a log that agreed at prev.

### 2.5 The commit rule and figure 8

The leader sets commit to the largest $i$ such that a majority has $\text{match} \ge i$ (itself included) **and** $t_i$ is the leader's current term. Entries before $i$ commit with it.

Why not count copies of an older entry? Figure 8 of the paper, in three nodes (section 3 replays it): a leader of term 4 holds $(2, t_2)$, written in term 2; node 3 holds $(2, t_3)$. When node 2 also gets $(2, t_2)$, it is on a majority, yet node 3, whose last term 3 beats 2, can still win an election with node 2's vote and overwrite index 2 everywhere. Only once an entry of the current term 4 is on a majority can no node with a conflicting index 2 win any more (its last term would be at most 3 < 4). So a new leader appends one **no-op** entry of its own term at once: without it, the entries a crashed leader left behind would wait for the next client write to commit.

### 2.6 What must be on disk, and when

| Saved | Before sending |
|---|---|
| the new term (vote cleared) | any reply in that term |
| the vote | the vote reply, or the RequestVotes of its own campaign |
| appended entries | the success reply (follower), or counting itself toward a commit (leader) |

The commit index and the roles are volatile: after a restart a node is a follower with commit 0, and committed entries come out of `ReadCommitted` again from index 1, so the applier must tolerate a replay. A write that fails stops the node: it may already have promised something it cannot remember.

### 2.7 Linearizability: what the tests demand of the whole

Clients see a history of operations, each with an **invocation** and a **response** time. The history is **linearizable** (Herlihy and Wing) if every operation can be given one point in time between its invocation and its response such that, in the order of those points, each result is what the sequential object would return. For an append-only log whose appends return their position, the $k$-th point must be the append that returned position $k$, so the order is forced and the check is one pass: walk the appends by position, putting each point as early as its invocation and the previous point allow, and fail if a point falls after its response. General checkers (Wing and Gong's search, Porcupine) explore orders when results do not force one.

## 3. Worked example by hand

Three nodes, empty logs, $H = 3$, timeouts fixed at 10, 15, and 20 ticks (the tests fix them so the run is the same every time). Every node ticks once per row.

| Tick or event | What happens | State after |
|---|---|---|
| ticks 1 to 9 | nothing: no timeout reached | all followers, term 0 |
| tick 10 | node 1 times out: term 1, votes for itself, saves $(1, 1)$, sends RequestVote$(t{=}1, \text{last}{=}(0, 0))$ to 2 and 3 | node 1 candidate, term 1 |
| node 2 gets the request | term 1 is new: adopt it; no vote yet, and $(0,0)$ is as up to date as its own $(0,0)$: grant, save $(1, 1)$, reset its timer | node 2 follower, term 1, vote 1 |
| node 1 gets the grant | 2 votes of 3 = $q$: **leader of term 1**; appends the no-op $(1, t_1)$; next = 1 for both peers | node 1: last (1, 1), commit 0 |
| AppendEntries to 2 and 3 | prev $(0, 0)$, entries [$(1, t_1, \text{no-op})$], commit 0 | |
| node 3's late request | it grants too (same reasons); node 1, already leader, ignores the extra vote | |
| node 2 accepts | prev 0 always matches; appends, answers success, match 1 | |
| node 1 counts | match: self 1, node 2 1: 2 of 3 hold index 1, of term 1 = current: **commit 1** | `ReadCommitted` = [no-op] |
| `Propose("x")` | index 2, term 1; AppendEntries to 2: prev $(1, t_1)$, entries [$(2, t_1, \text{x})$], commit 1 | |
| node 2 accepts, match 2 | 2 of 3 hold index 2: **commit 2**; the client's `x` is acknowledged | node 2 still has commit 1 |
| 3 ticks later | heartbeat with commit 2 | followers commit 2 |

Now figure 8, with $\text{MaxEntries} = 1$ so each message carries one entry. Saved state: node 1 $[(1,t_1), (2,t_2)]$, node 2 $[(1,t_1)]$, node 3 $[(1,t_1), (2,t_3)]$, all in term 3; node 3 is cut off.

1. Node 1 times out: term 4. Node 2 grants (last $(2, t_2)$ beats its $(1, t_1)$). Node 1 leads term 4 and appends its no-op $(3, t_4)$.
2. To node 2: prev $(2, t_2)$: refused, hint 2 (its log ends at 1). Next: prev $(1, t_1)$, entries [$(2, t_2)$]: accepted, match 2.
3. Count: index 3 has term 4 but only node 1 holds it; index 2 is on 2 of 3 nodes but has term 2, not 4. **Commit stays 0.** (Had node 1 committed index 2 here and crashed, node 3 with last term 3 could win term 5 with node 2's vote and replace index 2 with $(2, t_3)$: a committed entry lost.)
4. Next message: entries [$(3, t_4)$], accepted, match 3. Index 3 is on a majority and of term 4: **commit 3**, and indexes 1 and 2 with it.

## 4. The interface

```go
// go/durable/raft/raft.go
type Role int // Follower, Candidate, Leader
type Entry struct { Index, Term uint64; Data []byte } // empty Data = the leader's no-op
type MsgType int // MsgVote, MsgVoteResp, MsgApp, MsgAppResp
type Message struct {
	Type MsgType; From, To, Term uint64
	LastLogIndex, LastLogTerm uint64                    // MsgVote
	Granted bool                                        // MsgVoteResp
	PrevLogIndex, PrevLogTerm uint64; Entries []Entry; Commit uint64 // MsgApp
	Success bool; MatchIndex, ConflictIndex uint64      // MsgAppResp
}
type HardState struct { Term, Vote uint64 }
type Storage interface {
	Load() (HardState, []Entry, error)
	SaveHardState(HardState) error
	Append(entries []Entry) error // replaces every stored entry from entries[0].Index on
}
type Config struct {
	ID uint64; Peers []uint64           // Peers includes ID
	ElectionTicks, HeartbeatTicks int   // E and H
	Timeout func() int                  // nil: uniform [E, 2E) from Seed
	Seed uint64; MaxEntries int         // MaxEntries per MsgApp, 0 = 64
}
var ErrNotLeader, ErrEmptyProposal error
type NotLeaderError struct{ Leader uint64 } // wraps ErrNotLeader; Leader 0 = unknown
type Status struct { ID, Term, Vote uint64; Role Role; Leader, Commit, LastIndex, LastTerm uint64 }

func New(cfg Config, st Storage) (*Node, error)
func (n *Node) Tick() error
func (n *Node) Step(m Message) error
func (n *Node) Propose(data []byte) (index, term uint64, err error)
func (n *Node) ReadMessages() []Message // queued since the last call, in order
func (n *Node) ReadCommitted() []Entry  // committed since the last call, in index order
func (n *Node) Status() Status
```

The node never blocks, sleeps, or spawns: `Tick`, `Step`, and `Propose` change state and queue messages; the caller drains them. A proposal is acknowledged when an entry with **its index and its term** comes out of `ReadCommitted`; a different entry at that index means it was lost in a leader change and may be retried.

**Wiring it into the server** (your entry point, D16; not graded by `ol check`). With `--replicas 3` your durable server runs one `Node`, ticks it from a `time.Ticker` (for example 10 ms ticks, $E = 15$, $H = 5$), serves `tl.raft.v1.Raft` on `[durable].raft` (`:7234` in Kubernetes) by turning each RPC into a `Step` and each queued `Message` into an outgoing RPC, implements `Storage` on its data directory (`HardState` in a small file replaced by rename, entries in their own segment files), and applies committed entries in order into its `dur.01` log, skipping a record whose WAL seq it already holds. Every client RPC of `tl.durable.v1` that appends first goes through `Propose` and waits for its index and term to commit; on a follower it fails with `UNAVAILABLE` and the trailer `tl-raft-leader: <host:port>` from `NotLeaderError.Leader`.

### What the tests check

The scenarios deliver every message by hand, so each is a paper exercise; the simulation draws its faults from the course's frozen PCG32, so a failing seed replays exactly.

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `TestHandExampleElectionAndFirstCommit` | unit | section 3, message by message: the campaign, the saved vote, the no-op, commit 1, then `x` at index 2 and the followers' commit from the next heartbeat | the shape every later test builds on |
| `TestVoteRequiresUpToDateLog` | boundary | the five orders of (last term, last index) in section 2.2 | a leader never lacks a committed entry |
| `TestOneVotePerTermSurvivesRestart` | fault | one vote per term, saved before the reply, remembered across a restart; the same candidate may ask again | two leaders in one term are impossible |
| `TestMajorityOfThreeNeedsTwo` | boundary | an isolated node stays a candidate; an isolated leader commits nothing | a partitioned minority acknowledges nothing |
| `TestCandidateStepsDownForSameTermLeader` | unit | a candidate that lost term 1 follows the term-1 leader at its first AppendEntries and never disrupts it | elections settle instead of cascading |
| `TestHigherTermReplyMakesLeaderStepDown` | unit | a reply from term 7 turns a leader into a follower of term 7 with no leader, saved | a deposed leader stops at once |
| `TestStableLeaderNoSpuriousElections` | unit | 400 quiet ticks keep one term and one leader | heartbeats and timer resets work; no election storm |
| `TestAppendConsistencyCheckAndHint` | unit | refusal hints: past the end, last index + 1; inside a conflicting term, its first index | repairs take one round trip per term |
| `TestAppendReplacesOnlyTheConflictingSuffix` | unit | entries before the conflict stay; the leader's replace the rest, in memory and in `Storage` | followers converge to the leader's log |
| `TestStaleShortAppendDoesNotTruncate` | fault | a reordered old AppendEntries truncates nothing and reports match 2 | reordering never loses counted entries |
| `TestFollowerCommitStopsAtVerifiedEntries` | unit | commit = min(leader commit, last entry this message verified) | a follower never applies an entry it has not matched |
| `TestLaggingFollowerIsRepaired` | unit | a follower with a divergent term-2 suffix ends with the leader's exact log and commit | the log matching property in practice |
| `TestFigure8OldTermEntryNotCommittedByCounting` | unit | section 3's figure 8: commit stays 0 with $(2, t_2)$ on a majority, becomes 3 when the term-4 no-op is | the one subtle safety rule |
| `TestNewLeaderCommitsEarlierTermsThroughItsNoop` | fault | a new leader commits a crashed leader's entries with no client write | workflows blocked on those records resume |
| `TestProposeOnFollowerReturnsNotLeader` | unit | `*NotLeaderError` naming the leader, wrapping `ErrNotLeader`; empty data refused | the server can redirect clients |
| `TestStorageFailureSendsNothing` | fault | a failed vote save sends no RequestVote and stops the node | no promise the node cannot keep |
| `TestPCG32MatchesSpec` | golden | the simulation's generator is `spec/pcg32.md`'s | a seed names one fault schedule |
| `TestSimulatedFaultsKeepSafetyAndLinearizability` | fault, property | 40 seeds of drops, delays, reordering, partitions (the leader included), and crash-restarts: one leader per term, no index applied twice differently, recovery after the heal, every acknowledged append at its acknowledged position, a linearizable history | the HA promise as a whole |
| `TestLeaderKillAckedAppendsSurvive` | fault | 20 acknowledged appends, the leader killed: the new leader has all 20 in order; the old one restarts and catches up | `MS-durable-ha`'s kill loop, in miniature |
| `TestLinearizabilityCheckerRejectsStaleOrder` | unit | the checker rejects a real-time violation and a gap, and accepts overlapping ops in either order | the checker itself is checked |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. Voting without the up-to-date check, or comparing indexes only | a leader without committed entries overwrites them | `TestVoteRequiresUpToDateLog`, `TestSimulatedFaultsKeepSafetyAndLinearizability` (mutants `s01`, `s02`) |
| 2. Voting twice in a term, or forgetting the vote in a restart | two leaders in one term | `TestOneVotePerTermSurvivesRestart` (mutants `s03`, `s04`) |
| 3. Truncating on every AppendEntries | a reordered old message deletes entries the leader counted | `TestStaleShortAppendDoesNotTruncate` (mutant `s05`) |
| 4. Following the leader's commit past what this message verified | a follower applies its own stale entries | `TestFollowerCommitStopsAtVerifiedEntries` (mutant `s06`) |
| 5. Committing an old term's entry by counting copies | figure 8: an acknowledged write disappears after two leader changes | `TestFigure8OldTermEntryNotCommittedByCounting` (mutant `s07`) |
| 6. Quorum as `N/2` instead of `N/2 + 1` | an isolated node elects itself and acknowledges alone | `TestMajorityOfThreeNeedsTwo` (mutant `s08`) |
| 7. Ignoring a higher term in a reply | a deposed leader keeps sending in a dead term | `TestHigherTermReplyMakesLeaderStepDown` (mutant `s09`) |
| 8. No heartbeats, or no timer reset on AppendEntries | an election every timeout; writes stall | `TestStableLeaderNoSpuriousElections` (mutants `s10`, `s15`) |
| 9. `next` that never backs off, or backs off one entry at a time with a useless hint | a lagging follower is never repaired, or slowly | `TestLaggingFollowerIsRepaired`, `TestAppendConsistencyCheckAndHint` (mutants `s11`, `s18`) |
| 10. Accepting proposals on a follower | a client is acknowledged for a write no leader has | `TestProposeOnFollowerReturnsNotLeader` (mutant `s13`) |
| 11. Sending after a failed write, or acknowledging unsaved entries | a restart forgets a vote or an acknowledged entry | `TestStorageFailureSendsNothing`, `TestLeaderKillAckedAppendsSurvive` (mutants `s14`, `s16`) |
| 12. No no-op at the start of a term | a crashed leader's entries stay uncommitted until the next write | `TestNewLeaderCommitsEarlierTermsThroughItsNoop` (mutant `s17`) |
| 13. A candidate that ignores a same-term leader | it times out again and drags everyone into a new term | `TestCandidateStepsDownForSameTermLeader` (mutant `s12`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `dur.01` | the log whose records the entries carry; your server applies committed entries into it |
| Back | `lang.06` | Go, interfaces, and `testing` |
| Forward | MS-durable-ha | the durable kill loop against `{durable} --replicas 3` with leader kills and partitions, and a linearizability check over the history |
| Forward | `ops.12` | the optional drill that partitions your leader on kind |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| a ticked state machine with `ReadMessages` | etcd/raft's `Ready` and `Advance` | batching, pipelining, flow control | [etcd-io/raft](https://github.com/etcd-io/raft) |
| elections that can disrupt a leader | pre-vote and check-quorum | a rejoining node cannot depose a healthy leader | thesis, section 9.6 |
| reads through the log | ReadIndex and leader leases | linearizable reads without a log write | thesis, section 6.4 |
| a log that grows forever | snapshots, `InstallSnapshot`, compaction | bounded storage and fast catch-up | paper, section 7 |
| a fixed member list | joint consensus, single-server changes | adding and removing replicas safely | thesis, chapter 4 |
| one linear check per history | Porcupine, Jepsen's Knossos and Elle | general models, partial histories, real clusters | [jepsen.io](https://jepsen.io/) |
