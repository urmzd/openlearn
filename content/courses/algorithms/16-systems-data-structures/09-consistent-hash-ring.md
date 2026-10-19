# Consistent hash ring with bounded loads

## Overview

| | |
|---|---|
| **Module** | `ds.09` · build · Go · Pass 7 · 1.5 to 2 h reading |
| **Solution** | [`go/ds/ring/ring.go`](../../../../course/solutions/go/ds/ring/ring.go): `FNV1a64`, `Mix64`, `Hash64`, `VnodeLabel`, `New`, `Add`, `Remove`, `Len`, `Nodes`, `Get`, `Capacity`, `GetBounded`, with the unexported `point`, `less`, `first` · model tests: [`ring_learner_test.go`](../../../../course/solutions/learner-tests/ds.09/go/ds/ring/ring_learner_test.go) |
| **Contract** | there is no separate interface file: the API is section 4 of this chapter, and the hash vectors and tie rule there are what every implementation (Go, and the Python reference that generated the golden key-to-node map) must reproduce |
| **Needs** | reading: [`lang.06` Go](../../programming-languages/02-language-and-tool-primers/06-go.md), `S-M07d` [balls into bins](../../mathematics-for-computing/07-probability-statistics/93-problem-set-d.md) |
| **Used by** | `gw.05` affinity routing keys the ring by the first block of the prompt; it is what lets the multi-engine gateway keep prefix caches warm |
| **Optional depth** | Karger et al., *Consistent Hashing and Random Trees* (STOC 1997); Mirrokni, Thorup, Zadimoghaddam, *Consistent Hashing with Bounded Loads* (SODA 2018) |

## Key Takeaways

- A ring places every node at many pseudo-random **positions** (virtual nodes) and gives a key to the first position at or clockwise after the key's own hash; past the top it wraps.
- Adding a node moves only the keys that land in its new arcs, about $K/(N+1)$ of $K$ keys, and every moved key goes to the new node; removing a node moves only its own keys.
- **Bounded loads** cap every node at $\lceil c \cdot (m+1)/n \rceil$: a key whose owner is full walks clockwise to the next node below the cap, so a hot prefix spills over instead of melting one replica.
- Short labels such as `engine-0#7` make bare FNV-1a cluster; finalizing with the SplitMix64 mix spreads them, and ties between equal positions go to the smaller node name, so every gateway replica computes the same ring.

## How to read this chapter

Sections 1 to 3 build the ring, its hash, and the bounded-loads rule from first principles and work five keys and one spill by hand. Section 4 fixes the interface and lists what a correct ring must handle. Section 5 walks through [`ring.go`](../../../../course/solutions/go/ds/ring/ring.go) in the order it was designed: the hashes (checked against the vectors of section 4), then `Add`/`Get` on the worked example, then `GetBounded`. Try to predict each step before reading it.

---

## 1. Why now

The gateway is about to talk to more than one engine (`gw.05`). Every engine keeps a prefix cache (`L8.4`): when two requests start with the same system prompt, the second one skips recomputing that prefix, but only if it lands on the **same** engine. Round robin scatters them, so each engine recomputes every prefix. A plain `hash(prefix) mod N` keeps them together until an engine dies or joins: then $N$ changes and almost every prefix moves, emptying every cache at once. Routing also has to survive a hot prefix: if half the traffic shares one system prompt, the engine that owns it saturates while the others idle. This module builds the data structure that solves all three: a consistent hash ring with bounded loads.

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

**The hash.** The ring needs a hash that any language can reproduce exactly (the Go ring must place every key exactly as a Python reference does), so it starts from FNV-1a 64, the same function as the KV block hash: start at $14695981039346656037$, and for each byte xor it in, then multiply by $1099511628211$ modulo $2^{64}$. FNV-1a is weak on short inputs that differ only in their last bytes: `engine-0#7` and `engine-0#8` differ in one byte, which the final multiply spreads over only part of the word, and the positions cluster (measured on 8 nodes and 160 vnodes: shares from 3.6% to 22%). So the ring finalizes it with the SplitMix64 mix of `M06.3`:

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

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| The hand example | unit | section 3: owners of `k1` to `k5`, the wrap, a key exactly on a position | the definition `gw.05` and the Python reference share |
| The hand example's bounded spill | unit | section 3: `k1` skips the full `a` for `b`; `k3` stays | the spill rule that protects a hot engine |
| Capacity | unit | $\lceil c(m+1)/n \rceil$ on six cases, including $n = 0$ | the bound every placement uses |
| Hash vectors | unit | FNV-1a vectors, `Mix64`, `Hash64`, `VnodeLabel` | Python and Go place nodes identically |
| An empty ring | boundary | `ok = false` from both lookups, before and after the last Remove | a gateway with no healthy worker answers 503, not a panic |
| Zero vnodes | boundary | `New(0, nil)` panics | a member with no position would never get a key |
| Add and Remove are idempotent | unit | a repeated Add and a stranger's Remove change no key; `Len`, sorted `Nodes` | heartbeats re-announce workers every 2 s |
| Collisions break ties by name | boundary | equal positions go to the smaller name in either join order | replicas agree on the ring |
| Insertion order is irrelevant | property | 2000 keys map the same for two join orders | the same |
| Add moves keys only to the new node | property, statistical | moved keys all go to the new node; their count is within 4 sd of $K p_{\text{new}}$; $p_{\text{new}}$ near $1/9$ | caches stay warm when an engine joins |
| Remove moves only its keys | property | only the removed node's keys move | caches stay warm when an engine dies |
| Keys spread by arc length | statistical | key counts fit the arc shares: chi-square below 24.32 (7 df, $p = 10^{-3}$) | the hash spreads keys uniformly |
| Bounded never exceeds capacity | property | distinct keys, one hot key, three hot keys, $c \in \{1, 1.25, 2\}$: no node above $\lceil cK/n \rceil$ | the guarantee `gw.05` relies on |
| Bounded equals Get when unloaded | property | with zero loads bounded and plain lookup agree | affinity is kept until it costs something |
| Bounded clamps $c$ below one | boundary | $c = 0.5$ places exactly like $c = 1$ | a misconfigured factor cannot make placement impossible |
| Golden key-to-node map | golden | every case of a golden map generated by the Python reference (plain, removals, bounded with load feedback) | any implementation of the ring routes identically |

A model test suite in `package ring_test` (the owner is the first point clockwise and wraps; an empty ring says nobody; Remove moves only its keys; the capacity formula; bounded placement respects capacity with a hot key; $c < 1$ acts as 1; ties go to the smaller name; the hash is FNV-1a then SplitMix) is in [`ring_learner_test.go`](../../../../course/solutions/learner-tests/ds.09/go/ds/ring/ring_learner_test.go).

## 5. The solution, step by step

### 5.1 From first principles

1. **Membership changes must move as few keys as possible.** `hash mod N` ties every key to $N$. Placing nodes and keys in one hash space and giving each key to the next node clockwise ties a key only to its neighbourhood: adding a node steals only the arcs its positions cut.
2. **One position per node gives very uneven arcs,** so each node gets $V$ positions (virtual nodes) and its share concentrates near $1/N$. The data structure is then a sorted array of points, and a lookup is a binary search for the first point at or after $h(x)$, wrapping to index 0.
3. **Every gateway replica must compute the same ring,** whatever order it heard about nodes. The hash is fixed (FNV-1a for portability, finalized with SplitMix64 because bare FNV-1a clusters on labels that differ in their last byte), the labels are fixed (`node#idx`, idx from 0), and equal positions are ordered by (node name, copy index).
4. **A hot key must not overload its owner.** Placing one more key when $m$ are placed, no node should exceed $\lceil c(m+1)/n \rceil$. A key whose owner is at the cap walks clockwise to the next *distinct* node below it. Since loads sum to $m < n \cdot \mathrm{cap}$, such a node exists for $c \ge 1$, so the walk terminates within $n$ distinct nodes. In the worked example, cap is 2, `a` holds 2, so `k1` moves on to `b`.
5. **When nobody is loaded the cap does not bind,** so bounded lookup equals plain lookup and affinity is kept until it costs something.

### 5.2 The code, annotated

The hashes. Each is one or two lines and matches the formulas of section 2.1; Go's `uint64` multiply is already mod $2^{64}$.

```go
const (
	fnvOffset uint64 = 14695981039346656037
	fnvPrime  uint64 = 1099511628211
)

func FNV1a64(b []byte) uint64 {
	h := fnvOffset
	for _, c := range b {
		h ^= uint64(c) // xor first: FNV-1a, not FNV-1
		h *= fnvPrime
	}
	return h
}

func Mix64(z uint64) uint64 {
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return z ^ (z >> 31)
}

func Hash64(b []byte) uint64 { return Mix64(FNV1a64(b)) } // spreads FNV's clustered outputs

// "engine-a#0", "engine-a#1", ...: copies numbered from 0, in decimal.
func VnodeLabel(node string, idx int) []byte {
	return []byte(node + "#" + strconv.Itoa(idx))
}
```

The ring's state is a sorted slice of points and a set of members. `less` is the total order that makes the slice independent of insertion order.

```go
type point struct {
	pos  uint64
	node string
	idx  int
}

type Ring struct {
	vnodes int
	h      Hash
	points []point         // sorted by (pos, node, idx)
	nodes  map[string]bool // the members
}

func less(a, b point) bool {
	if a.pos != b.pos {
		return a.pos < b.pos
	}
	if a.node != b.node {
		return a.node < b.node // a collision goes to the smaller name, on every replica
	}
	return a.idx < b.idx
}
```

Membership. `Add` is idempotent through the member set (heartbeats repeat), and re-sorts with the total order. `Remove` must drop the points as well as the member, or keys would still route to a dead engine; it filters in place.

```go
func (r *Ring) Add(node string) {
	if r.nodes[node] {
		return // re-announced: nothing changes
	}
	r.nodes[node] = true
	for i := 0; i < r.vnodes; i++ {
		r.points = append(r.points, point{pos: r.h(VnodeLabel(node, i)), node: node, idx: i})
	}
	sort.Slice(r.points, func(i, j int) bool { return less(r.points[i], r.points[j]) })
}

func (r *Ring) Remove(node string) {
	if !r.nodes[node] {
		return
	}
	delete(r.nodes, node)
	kept := r.points[:0] // reuse the backing array; order is preserved, so no re-sort
	for _, p := range r.points {
		if p.node != node {
			kept = append(kept, p)
		}
	}
	r.points = kept
}
```

Lookup. `sort.Search` finds the first point with `pos >= x`; `>=` (not `>`) is what makes a key exactly on a position belong to that position, as `k5` does in the worked example. Falling off the end wraps to 0.

```go
func (r *Ring) first(x uint64) int {
	i := sort.Search(len(r.points), func(i int) bool { return r.points[i].pos >= x })
	if i == len(r.points) {
		return 0 // past the largest position: wrap around
	}
	return i
}

func (r *Ring) Get(key []byte) (node string, ok bool) {
	if len(r.points) == 0 {
		return "", false // no healthy worker: the caller answers 503
	}
	return r.points[r.first(r.h(key))].node, true
}
```

Bounded loads. The cap divides by members, not points, and counts the key being placed. The walk skips points of nodes already checked, and stops once every member has been seen.

```go
func Capacity(total, n int, c float64) int {
	if n <= 0 {
		return 0
	}
	return int(math.Ceil(c * float64(total+1) / float64(n))) // +1: the key being placed
}

func (r *Ring) GetBounded(key []byte, load func(node string) int, c float64) (node string, ok bool) {
	if len(r.points) == 0 {
		return "", false
	}
	if c < 1 {
		c = 1 // below 1 no node may be below the cap: placement would be impossible
	}
	total := 0
	for n := range r.nodes {
		total += load(n)
	}
	limit := Capacity(total, len(r.nodes), c)
	start := r.first(r.h(key))
	seen := make(map[string]bool, len(r.nodes))
	for k := 0; k < len(r.points) && len(seen) < len(r.nodes); k++ {
		p := r.points[(start+k)%len(r.points)] // clockwise, wrapping
		if seen[p.node] {
			continue // a node is judged once, at its first point on the walk
		}
		seen[p.node] = true
		if load(p.node) < limit { // strictly below: placing makes it at most the cap
			return p.node, true
		}
	}
	// Unreachable for c >= 1 (some node is below the average); keep the
	// plain owner rather than failing a request.
	return r.points[start].node, true
}
```

Full solution: [`ring.go`](../../../../course/solutions/go/ds/ring/ring.go).

### 5.3 Why this solution

- **Cost.** `Get` is $O(\log(NV))$ (binary search over $NV$ points); `GetBounded` adds $O(N)$ to sum the loads and walks at most until $N$ distinct nodes are seen, typically a few points. `Add` is $O(NV \log(NV))$ for the re-sort and `Remove` is $O(NV)$; both are rare (membership changes) next to lookups. Memory is $NV$ points.
- **Against `hash mod N`.** O(1) lookup, but a membership change remaps about $(N-1)/N$ of keys and empties every prefix cache at once.
- **Against rendezvous (highest random weight) hashing.** Same minimal movement and no vnodes, but $O(N)$ hashes per lookup; with bounded loads it also needs a full ranking per key.
- **Against jump consistent hash or Maglev.** O(1) lookups from a table, but jump hash only supports removing the last bucket, and Maglev's table must be rebuilt on every change; neither has a natural clockwise walk for the bounded-loads spill.
- **Against inserting points with a binary-search insert instead of re-sorting.** Asymptotically better for `Add`, but `sort.Slice` over a few thousand points is microseconds and keeps the code obviously correct.
- **Tradeoffs accepted.** The ring is not safe for concurrent use (`gw.05` holds its own lock). `GetBounded` calls `load` once per member to sum loads, so the load function must be cheap. The final fallback returns the plain owner instead of an error, which only matters if a caller passes inconsistent loads.

### 5.4 Explain it back

**Why does the capacity formula have $m + 1$ and not $m$?**

<details><summary>Answer</summary>

The cap is for placing one more key. With $m$, an empty ring would have $\mathrm{cap} = 0$ and no node would be "below cap", so nothing could be placed. With $m + 1$, $n \cdot \mathrm{cap} \ge c(m+1) > m$ for $c \ge 1$, so the loads (which sum to $m$) cannot all be at the cap: some node is below it and the walk ends.

</details>

**Why are points ordered by (position, node, index) and not just by position?**

<details><summary>Answer</summary>

`sort.Slice` is not stable, and even a stable sort would keep whatever order nodes were added in. Two gateway replicas that heard heartbeats in different orders would then disagree on which node owns a key that hashes to a collided position, and the same prefix would go to different engines depending on which replica handled it. A total order on points makes the ring a function of the member set alone.

</details>

**Why does `GetBounded` skip points of nodes it has already seen?**

<details><summary>Answer</summary>

A node's load does not change between its points; if it was at the cap at its first point on the walk, it is at the cap at all of them. Skipping keeps the walk to at most $N$ distinct checks and makes "the next node clockwise" mean the next *different* node, which is the rule section 2.3 states.

</details>

**What exactly moves when a ninth node joins an eight-node ring, and why does nothing move between old nodes?**

<details><summary>Answer</summary>

Only keys whose hash falls in one of the new node's arcs, the stretches just before each of its $V$ new positions, move, and they move to the new node. For every other key, the first point at or after its hash is unchanged, because the new points are not between that hash and its old owner's point. That is about $K/9$ keys, against about $8K/9$ for `hash mod N` going from 8 to 9.

</details>

**Why finalize FNV-1a with `Mix64` instead of using FNV-1a directly?**

<details><summary>Answer</summary>

Labels like `engine-0#7` and `engine-0#8` differ in one late byte, and FNV-1a's last multiply spreads that difference over only part of the word, so the positions of one node's vnodes cluster. Measured on 8 nodes with 160 vnodes, bare FNV gives shares from 3.6% to 22%. SplitMix64 makes every input bit affect every output bit, so the positions spread and shares concentrate near $1/N$.

</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. no wrap: a key past the last position gets the last node | keys near the top all pile on one node | `first` returns 0 when the search falls off the end |
| 2. `>` instead of `>=` in the search | a key exactly on a position goes to the next node | `first` searches for `pos >= x` |
| 3. capacity without the $+1$, `<=` instead of `<`, or dividing by vnodes instead of nodes | an empty ring has capacity 0, or a node ends one key over the bound | `Capacity` uses `total+1` and divides by `len(r.nodes)`; `GetBounded` requires `load < limit` and clamps `c` to 1 |
| 4. sorting by position only | two replicas that heard workers in a different order route a colliding key differently | `less` breaks ties by node name, then copy index |
| 5. Remove that forgets the member but keeps its positions | keys still route to a dead engine | `Remove` filters the node's points out of `r.points` |
| 6. FNV-1 instead of FNV-1a, labels from 1, or no finalizer | positions differ from the Python reference; with bare FNV the shares range from 4% to 22% | `FNV1a64` xors before multiplying, `VnodeLabel` counts from 0, `Hash64` applies `Mix64` |
| 7. indexing an empty slice, or no `vnodes` guard | a panic when the last engine is evicted | `Get` and `GetBounded` return `ok = false` on an empty ring; `New` panics on `vnodes < 1` |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `lang.06` | slices, maps, `sort.Search`, closures for the load function |
| Back | `S-M07d` | balls into bins: why $\lceil c \cdot \text{avg} \rceil$ is achievable and what $c$ buys |
| Forward | `gw.05` | `route_policy = "affinity"` keys `GetBounded` by the first prompt block, with load = in flight + queue depth |

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `Ring` with virtual nodes | Envoy `ring_hash` load balancer | minimum ring size, per-host weights, hash policies on headers | [Envoy load balancers](https://www.envoyproxy.io/docs/envoy/latest/intro/arch_overview/upstream/load_balancing/load_balancers) (free) |
| `GetBounded` | HAProxy `hash-balance-factor`, Google Cloud Pub/Sub | the bounded-loads rule in production since 2017 | [Vimeo engineering: Improving load balancing with a new consistent-hashing algorithm](https://medium.com/vimeo-engineering-blog/improving-load-balancing-with-a-new-consistent-hashing-algorithm-9f1bd75709ed) (free) |
| ring lookup per request | Maglev hashing, jump consistent hash | O(1) lookups from a precomputed table; no memory for vnodes | [Maglev (NSDI 2016)](https://research.google/pubs/maglev-a-fast-and-reliable-software-network-load-balancer/) (free) |
| prefix affinity | llm-d inference scheduler, SGLang router | KV-cache-aware scoring instead of a hash of the prefix | [llm-d](https://github.com/llm-d/llm-d) (free) |
