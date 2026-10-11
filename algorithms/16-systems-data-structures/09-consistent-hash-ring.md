<!-- ol:module ds.09 -->
# Consistent hash ring with bounded loads

## Overview

| | |
|---|---|
| **Module** | `ds.09` · build · Go · Pass 7 · 3 to 4 h |
| **You build** | `go/ds/ring/ring.go`: `FNV1a64`, `Mix64`, `Hash64`, `VnodeLabel`, `New`, `Add`, `Remove`, `Len`, `Nodes`, `Get`, `Capacity`, `GetBounded` |
| **Contract** | no interface file yet: the API is section 4 of this chapter, held by the course tests and the parity suite `ring.hash` ([`course/conformance/parity/ring.hash.toml`](../../course/conformance/parity/ring.hash.toml)) |
| **Tests** | `course/tests/go/ds_09/` (what they check: section 4) · your own tests in `go/ds/ring/ring_learner_test.go`, rung R2, graded by mutation (threshold 0.60) |
| **Needs** | reading: [`lang.06` Go](../../software-craftsmanship/12-language-and-tool-primers/06-go.md), `S-M07d` [balls into bins](../../math/07-probability-statistics/93-problem-set-d.md) |
| **Used by** | `gw.05` affinity routing keys the ring by the first block of the prompt |
| **Milestone** | MS-gateway |
| **Optional depth** | Karger et al., *Consistent Hashing and Random Trees* (STOC 1997); Mirrokni, Thorup, Zadimoghaddam, *Consistent Hashing with Bounded Loads* (SODA 2018) |

## Key Takeaways

- A ring places every node at many pseudo-random **positions** (virtual nodes) and gives a key to the first position at or clockwise after the key's own hash; past the top it wraps (`TestHandExample`).
- Adding a node moves only the keys that land in its new arcs, about $K/(N+1)$ of $K$ keys, and every moved key goes to the new node; removing a node moves only its own keys (`TestAddMovesOnlyToNewNode`, `TestRemoveMovesOnlyItsKeys`).
- **Bounded loads** cap every node at $\lceil c \cdot (m+1)/n \rceil$: a key whose owner is full walks clockwise to the next node below the cap, so a hot prefix spills over instead of melting one replica (`TestBoundedNeverExceedsCapacity`).
- Short labels such as `engine-0#7` make bare FNV-1a cluster; finalizing with the SplitMix64 mix spreads them, and ties between equal positions go to the smaller node name, so every gateway replica computes the same ring (`TestKeysSpreadByArcLength`, `TestCollisionsBreakTiesByName`).

## How to work this chapter

```bash
ol start ds.09          # writes go/ds/ring/ring.go with stub bodies (and go/go.mod if absent)
ol tests ds.09          # read the test catalog first
ol check ds.09          # course tests, then your tests graded by mutation
ol parity ring.hash     # your ring against the golden key-to-node map
ol diff  ds.09          # after passing: your code against the reference
```

Write the hashes first and check them on the vectors of section 4, then `Add`/`Get` on the worked example, then `GetBounded`.

---

## 1. Why now

Your gateway is about to talk to more than one engine (`gw.05`). Every engine keeps a prefix cache (`L8.4`): when two requests start with the same system prompt, the second one skips recomputing that prefix, but only if it lands on the **same** engine. Round robin scatters them, so each engine recomputes every prefix. A plain `hash(prefix) mod N` keeps them together until an engine dies or joins: then $N$ changes and almost every prefix moves, emptying every cache at once. Routing also has to survive a hot prefix: if half the traffic shares one system prompt, the engine that owns it saturates while the others idle. This module builds the data structure that solves all three: a consistent hash ring with bounded loads.

## 2. Principles

### 2.1 Positions on a ring

| Symbol | Meaning | Type |
|---|---|---|
| $h(x)$ | the ring's hash of a byte string, a position in $[0, 2^{64})$ | `uint64` |
| $N$ | number of member nodes (engines) | integer |
| $V$ | virtual nodes per member (`vnodes`) | integer, 160 in `gw.05` |
| $K$ | number of keys (requests) | integer |
| $p_j$ | the fraction of the ring that node $j$ owns (its arc share) | real, $\sum_j p_j = 1$ |

Treat $[0, 2^{64})$ as a circle. Node `engine-a` with $V$ virtual nodes sits at the $V$ positions $h(\texttt{engine-a\#0}), \dots, h(\texttt{engine-a\#}(V-1))$. A key $x$ belongs to the node of the **first position $\ge h(x)$**; if $h(x)$ is past the largest position, it wraps around to the smallest. Node $j$ therefore owns the arcs that end at its positions, and if keys hash uniformly, the expected fraction of keys it receives is its arc share $p_j$. With $V = 1$ the arcs are very uneven; with $V$ positions per node the shares concentrate around $1/N$ (their spread shrinks roughly like $1/\sqrt{V}$).

**The hash.** The ring needs a hash that any language can reproduce exactly (the parity suite compares your Go ring with a Python reference), so it starts from FNV-1a 64, the same function as the KV block hash: start at $14695981039346656037$, and for each byte xor it in, then multiply by $1099511628211$ modulo $2^{64}$. FNV-1a is weak on short inputs that differ only in their last bytes: `engine-0#7` and `engine-0#8` differ in one byte, which the final multiply spreads over only part of the word, and the positions cluster (measured on 8 nodes and 160 vnodes: shares from 3.6% to 22%). So the ring finalizes it with the SplitMix64 mix you met in `M06.3`:

$$z \leftarrow (z \oplus (z \gg 30)) \cdot \texttt{0xbf58476d1ce4e5b9},\quad z \leftarrow (z \oplus (z \gg 27)) \cdot \texttt{0x94d049bb133111eb},\quad z \leftarrow z \oplus (z \gg 31).$$

`Hash64(b) = Mix64(FNV1a64(b))` is the default hash for both labels and keys.

**Ties.** Two positions can be equal (a collision, or a deliberately coarse test hash). The ring orders points by (position, node name, copy index), so the result never depends on the order in which nodes were added: two gateway replicas that heard the heartbeats in a different order still agree on every key.

### 2.2 What moves when membership changes

Add node $N+1$. Its $V$ new positions cut existing arcs; a key moves exactly when its hash falls into one of the new node's arcs, and then it moves **to the new node**. No key moves between two old nodes. The expected number of moved keys is $K \cdot p_{N+1}$, about $K/(N+1)$. Compare `hash mod N`: going from 8 to 9 buckets keeps a key only when $h \bmod 8 = h \bmod 9$, which holds for about $1/9$ of keys, so $8/9$ of the caches go cold. Removing a node is the mirror image: only its keys move, each to the next position clockwise.

### 2.3 Bounded loads

Let $m$ be the number of keys already placed (for `gw.05`: requests in flight plus queue depth, summed over nodes) and $c \ge 1$ a slack factor (`affinity_load_factor`, default 1.25). The **capacity** for placing one more key is

$$\mathrm{cap} = \left\lceil c \cdot \frac{m+1}{n} \right\rceil.$$

To place key $x$, walk clockwise from $h(x)$ and take the **first distinct node whose load is below cap**. Because the loads sum to $m$ and $n \cdot \mathrm{cap} \ge c(m+1) > m$, some node is below cap, so the walk always ends. The $+1$ counts the key being placed: without it, an empty ring has $\mathrm{cap} = 0$ and no node qualifies. The guarantee, placing $K$ keys one by one: every node ends with at most $\lceil c K / n \rceil$. This is the balls-into-bins bound of `S-M07d` turned into a rule. When loads are all zero, bounded and plain lookup agree, so affinity is kept until it actually costs something.

## 3. Worked example by hand

Three nodes `a`, `b`, `c`, two virtual nodes each, and a hash given as a table so every number is checkable on paper:

| Label | `a#0` | `b#0` | `c#0` | `a#1` | `b#1` | `c#1` |
|---|---|---|---|---|---|---|
| position | 10 | 30 | 50 | 60 | 80 | 95 |

Sorted, the ring reads `a 10, b 30, c 50, a 60, b 80, c 95`, then wraps to `a 10`.

| Key | $h$ | First point $\ge h$ | Owner |
|---|---|---|---|
| `k1` | 5 | `a#0` at 10 | a |
| `k2` | 55 | `a#1` at 60 | a |
| `k3` | 85 | `c#1` at 95 | c |
| `k4` | 97 | none, wrap to `a#0` at 10 | a |
| `k5` | 30 | `b#0` at 30 (equal counts) | b |

Now bounded loads with $c = 1.25$ and current loads `a = 2`, `b = 0`, `c = 1`: $m = 3$, $n = 3$, so $\mathrm{cap} = \lceil 1.25 \cdot 4 / 3 \rceil = \lceil 1.667 \rceil = 2$. Key `k1`'s owner `a` has load 2, not below 2, so the walk continues clockwise from `a#0` at 10 to `b#0` at 30: `b` has load 0, so `GetBounded(k1) = b`. Key `k3`'s owner `c` has load 1 < 2 and keeps it.

These are `TestHandExample` and `TestBoundedSkipsFullNodeHand`.

## 4. The interface

```go
package ring // import "tinyllm/ds/ring"

func FNV1a64(b []byte) uint64           // FNV-1a 64
func Mix64(z uint64) uint64             // SplitMix64 finalizer
func Hash64(b []byte) uint64            // Mix64(FNV1a64(b)): the default hash
func VnodeLabel(node string, idx int) []byte // "<node>#<idx>", idx from 0

type Ring struct{ /* unexported */ }
func New(vnodes int, h func([]byte) uint64) *Ring   // h nil = Hash64; vnodes < 1 panics
func (r *Ring) Add(node string)                      // idempotent
func (r *Ring) Remove(node string)                   // a non-member is a no-op
func (r *Ring) Len() int
func (r *Ring) Nodes() []string                      // sorted
func (r *Ring) Get(key []byte) (node string, ok bool)    // ok false on an empty ring
func Capacity(total, n int, c float64) int               // ceil(c*(total+1)/n); 0 when n <= 0
func (r *Ring) GetBounded(key []byte, load func(node string) int, c float64) (node string, ok bool) // c < 1 acts as 1
```

Hash vectors: `FNV1a64("") = 0xcbf29ce484222325`, `FNV1a64("a") = 0xaf63dc4c8601ec8c`, `FNV1a64("foobar") = 0x85944171f73967e8`. The ring is not safe for concurrent use; `gw.05` holds its own lock around it. Use only the standard library.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `TestHandExample` | unit | section 3: owners of `k1` to `k5`, the wrap, a key exactly on a position | the definition `gw.05` and the parity suite share |
| `TestBoundedSkipsFullNodeHand` | unit | section 3: `k1` skips the full `a` for `b`; `k3` stays | the spill rule that protects a hot engine |
| `TestCapacity` | unit | $\lceil c(m+1)/n \rceil$ on six cases, including $n = 0$ | the bound every placement uses |
| `TestFNV1a64Vectors` | unit | FNV-1a vectors, `Mix64`, `Hash64`, `VnodeLabel` | Python, Go, and the oracle place nodes identically |
| `TestEmptyRing` | boundary | `ok = false` from both lookups, before and after the last Remove | a gateway with no healthy worker answers 503, not a panic |
| `TestNewRejectsZeroVnodes` | boundary | `New(0, nil)` panics | a member with no position would never get a key |
| `TestAddRemoveIdempotent` | unit | a repeated Add and a stranger's Remove change no key; `Len`, sorted `Nodes` | heartbeats re-announce workers every 2 s |
| `TestCollisionsBreakTiesByName` | boundary | equal positions go to the smaller name in either join order | replicas agree on the ring |
| `TestInsertionOrderIrrelevant` | property | 2000 keys map the same for two join orders | the same |
| `TestAddMovesOnlyToNewNode` | property, statistical | moved keys all go to the new node; their count is within 4 sd of $K p_{\text{new}}$; $p_{\text{new}}$ near $1/9$ | caches stay warm when an engine joins |
| `TestRemoveMovesOnlyItsKeys` | property | only the removed node's keys move | caches stay warm when an engine dies |
| `TestKeysSpreadByArcLength` | statistical | key counts fit the arc shares: chi-square below 24.32 (7 df, $p = 10^{-3}$) | the hash spreads keys uniformly |
| `TestBoundedNeverExceedsCapacity` | property | distinct keys, one hot key, three hot keys, $c \in \{1, 1.25, 2\}$: no node above $\lceil cK/n \rceil$ | the guarantee `gw.05` relies on |
| `TestBoundedEqualsGetWhenUnloaded` | property | with zero loads bounded and plain lookup agree | affinity is kept until it costs something |
| `TestBoundedClampsCBelowOne` | boundary | $c = 0.5$ places exactly like $c = 1$ | a misconfigured factor cannot make placement impossible |
| `TestGoldenKeyToNodeMap` | golden, conformance | every case of `parity/ring_hash.json` (plain, removals, bounded with load feedback) | `ol parity ring.hash` |

Your own tests (rung R2) go in `go/ds/ring/ring_learner_test.go` as `package ring_test`. Name them after what they prove: the owner is the first point clockwise and wraps; an empty ring says nobody; Remove moves only its keys; the capacity formula; bounded placement respects capacity with a hot key; $c < 1$ acts as 1; ties go to the smaller name; the hash is FNV-1a then SplitMix. They are graded against the reference with one planted bug each; 60% must be caught.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. no wrap: a key past the last position gets the last node | keys near the top all pile on one node | `TestHandExample` (mutant `s01`) |
| 2. `>` instead of `>=` in the search | a key exactly on a position goes to the next node | `TestHandExample` (mutant `s02`) |
| 3. capacity without the $+1$, `<=` instead of `<`, or dividing by vnodes instead of nodes | an empty ring has capacity 0, or a node ends one key over the bound | `TestCapacity`, `TestBoundedNeverExceedsCapacity`, `TestBoundedClampsCBelowOne` (mutants `s03`, `s04`, `s10`, `s11`) |
| 4. sorting by position only | two replicas that heard workers in a different order route a colliding key differently | `TestCollisionsBreakTiesByName` (mutant `s05`) |
| 5. Remove that forgets the member but keeps its positions | keys still route to a dead engine | `TestRemoveMovesOnlyItsKeys`, `TestEmptyRing` (mutant `s06`) |
| 6. FNV-1 instead of FNV-1a, labels from 1, or no finalizer | positions differ from the Python reference; with bare FNV the shares range from 4% to 22% | `TestFNV1a64Vectors`, `TestKeysSpreadByArcLength`, `TestGoldenKeyToNodeMap` (mutants `s07`, `s08`, `s14`) |
| 7. indexing an empty slice, or no `vnodes` guard | a panic when the last engine is evicted | `TestEmptyRing`, `TestNewRejectsZeroVnodes` (mutants `s12`, `s13`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `lang.06` | slices, maps, `sort.Search`, closures for the load function |
| Back | `S-M07d` | balls into bins: why $\lceil c \cdot \text{avg} \rceil$ is achievable and what $c$ buys |
| Forward | `gw.05` | `route_policy = "affinity"` keys `GetBounded` by the first prompt block, with load = in flight + queue depth |

If you skip this module, `ol check gw.05` reports `needs ds.09: build it, or pass --ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `Ring` with virtual nodes | Envoy `ring_hash` load balancer | minimum ring size, per-host weights, hash policies on headers | [Envoy load balancers](https://www.envoyproxy.io/docs/envoy/latest/intro/arch_overview/upstream/load_balancing/load_balancers) (free) |
| `GetBounded` | HAProxy `hash-balance-factor`, Google Cloud Pub/Sub | the bounded-loads rule in production since 2017 | [Vimeo engineering: Improving load balancing with a new consistent-hashing algorithm](https://medium.com/vimeo-engineering-blog/improving-load-balancing-with-a-new-consistent-hashing-algorithm-9f1bd75709ed) (free) |
| ring lookup per request | Maglev hashing, jump consistent hash | O(1) lookups from a precomputed table; no memory for vnodes | [Maglev (NSDI 2016)](https://research.google/pubs/maglev-a-fast-and-reliable-software-network-load-balancer/) (free) |
| prefix affinity | llm-d inference scheduler, SGLang router | KV-cache-aware scoring instead of a hash of the prefix | [llm-d](https://github.com/llm-d/llm-d) (free) |
