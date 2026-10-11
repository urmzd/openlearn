<!-- ol:module S-M06a -->
# Discrete math 2 problem set, part a: DAGs, modular arithmetic, hashing

## Overview

| | |
|---|---|
| **Module** | `S-M06a` · solve · none · Pass 2 · 4 to 5 h |
| **You build** | answers in `solve/S-M06a.toml` (23 checked by SymPy) and 5 proofs in `solve/S-M06a/qN.md` (self-graded against their rubrics) |
| **Contract** | none: a pen and paper set |
| **Tests** | `course/solve/S-M06a/key.toml` (hidden): typed answers plus reject canaries; the problems are in `course/solve/S-M06a/problems.md` and in section 4 |
| **Needs** | `S-M05` (proof methods, induction, counting, bijections). Reading: the [Discrete Math 2 topic](README.md), graphs and number theory sections |
| **Used by** | no call site (a solve set). Do it before `M06.1` (iterative toposort for backward; q20 is its correctness proof) and `M06.3` (PCG32, SplitMix64, FNV-1a: wrapping arithmetic, rotations, and the odd multiplier of q22). Part b, `S-M06b` in Pass 3, covers trees, birthday bounds, MinHash, and Bloom filters |
| **Milestone** | `MS-P2` (the Pass 2 gate runs `ol check` on every solve part of the pass) |
| **Optional depth** | Hammack, *Book of Proof*, ch. 11 and 12; Cormen et al., *Introduction to Algorithms*, ch. 11 (hashing) and 22.4 (topological sort); O'Neill, "PCG: A Family of Simple Fast Space-Efficient Statistically Good Algorithms" (2014), sections 4 and 6 |

## Key Takeaways

- A computation graph is a DAG; a topological order lists every node after its inputs, it is rarely unique, and backward walks it in reverse (q2, q3, q8).
- Recursion depth equals the longest input chain, so a recursive traversal of a $10^5$-node chain dies at Python's limit of 1000; the iterative DFS of q20 does not (q9).
- Fixed-width integers are arithmetic modulo $2^w$: addition wraps, rotation moves bits around the word, and $x \mapsto ax \bmod 2^w$ permutes the words exactly when $a$ is odd (q12, q13, q22, q23).
- FNV-1a XORs a byte in, then multiplies; swapping the two steps gives FNV-1, a different hash (q14).
- With $n$ keys in $m$ buckets the expected number of colliding pairs is $\binom{n}{2}/m$, which is why collisions appear long before the table is full (q17).

## How to work this chapter

```bash
ol start S-M06a             # writes solve/S-M06a.toml and one file per proof
ol check S-M06a             # SymPy checks the answers, then asks each proof rubric (y/n)
ol check S-M06a --regrade   # ask the rubrics again after you change a proof
```

---

## 1. Why now

Your bigram model in Pass 1 needed no graph: its gradient was a count. In Pass 2 you write an autograd engine (`L0.1`), and backward is only correct if every node's gradient is complete before it is passed on to that node's inputs, which means visiting the graph in reverse topological order. `M06.1` builds that order with an iterative depth-first search, because a recursive one crashes on the $10^5$-node chains that long sequences produce. In the same pass `M06.3` builds PCG32, the random generator every later module draws from, and FNV-1a, the hash that `rt.04` uses to name KV-cache blocks. Both are arithmetic on 32- and 64-bit words that silently wrap. This set makes you fluent in directed acyclic graphs, modular arithmetic, and the basic probability of hashing before you write that code.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $G = (V, E)$ | a directed graph: nodes $V$, edges $E \subseteq V \times V$ | |
| $u \to v$ | an edge; in an autograd graph, $u$ is an input of $v$ | |
| $\deg^-(v), \deg^+(v)$ | in-degree (edges into $v$), out-degree (edges out of $v$) | integer |
| $A$ | adjacency matrix, $A_{ij} = 1$ when $i \to j$ | `int[n][n]` |
| $a \bmod m$ | remainder of $a$ on division by $m$, in $\{0, \dots, m-1\}$ | integer |
| $a \equiv b \pmod m$ | $m$ divides $a - b$ | relation |
| $\gcd(a, m)$ | greatest common divisor | integer |
| $\oplus$ | bitwise XOR | `uint64` |
| $n, m$ | number of keys, number of buckets | integer |
| $\alpha = n/m$ | load factor | real |

### 2.1 Directed graphs and DAGs

A **directed graph** is a set of nodes and a set of ordered pairs of nodes, the edges. A **path** is a sequence of edges $v_0 \to v_1 \to \dots \to v_k$, and a **cycle** is a path of length at least 1 that returns to its start. A graph with no cycle is a **directed acyclic graph (DAG)**. Every expression a program evaluates is a DAG: the nodes are values, and $u \to v$ when $u$ is an input of the operation that makes $v$. A value used twice ($b$ in $ab + b$) has out-degree 2, which is why a gradient must be summed over consumers.

A **topological order** lists all nodes so that every edge goes forward: $u \to v$ implies $u$ is listed before $v$. A graph has one exactly when it is acyclic (q19 one way; q18 gives the other). Topological orders are rarely unique: any two nodes not connected by a path can be swapped. That is why the `M06.1` tests check the property "every edge goes forward", never one specific list.

Two algorithms produce one. **Kahn's algorithm** keeps the in-degree of every node, repeatedly outputs a node of in-degree 0, and decrements the in-degrees of the nodes it points to. **Depth-first search** outputs a node after all of its inputs have been output (post-order). Written recursively, DFS uses one stack frame per node on the current path, so its depth is the length of the longest input chain, and Python raises `RecursionError` past 1000 frames. Written with an explicit stack (q20), its depth is bounded only by memory.

Paths also count. Entry $(i, j)$ of $A^k$ is the number of paths of length exactly $k$ from $i$ to $j$, because $(A^k)_{ij} = \sum_{l} (A^{k-1})_{il} A_{lj}$ extends each path of length $k - 1$ by one edge. A DAG on $n$ nodes has no path longer than $n - 1$ edges, so $A^n = 0$. In reverse mode, the derivative of an output with respect to an input is a sum over all paths between them of the product of the local derivatives along each path (`M04.2`'s chain rule, organized by the graph).

### 2.2 Modular arithmetic

**Division with remainder**: for integers $a$ and $m \ge 1$ there are unique integers $q$ and $r$ with $a = qm + r$ and $0 \le r < m$; write $a \bmod m = r$. For negative $a$ the remainder is still in $\{0, \dots, m - 1\}$: $-17 = -4 \cdot 5 + 3$. (C's `%` truncates toward zero and returns $-2$; Python's `%` returns 3.) Two integers are **congruent modulo $m$**, $a \equiv b$, when $m$ divides $a - b$; congruence is an equivalence relation, and it respects $+$ and $\times$ (q21), so you may reduce after every step.

The **multiplicative inverse** of $a$ modulo $m$ is an $x$ with $ax \equiv 1$; it exists exactly when $\gcd(a, m) = 1$. Then multiplication by $a$ permutes $\{0, \dots, m-1\}$ (q22); when $\gcd(a, m) = g > 1$, it maps everything onto multiples of $g$ and loses states (q23).

### 2.3 Machine words

A `uint32_t` holds $w = 32$ bits and its arithmetic is arithmetic modulo $2^{32}$: $2^{32} - 1 + 2$ wraps to 1. C, Rust, and Go all expose this (C on unsigned types, Rust with `wrapping_add` and `wrapping_mul`, Go on `uint32` and `uint64`), and Python emulates it with `& 0xFFFFFFFF`. A **logical shift** right by $r$ drops the low $r$ bits and fills with zeros. A **rotation** right by $r$ moves bit $i$ to bit $(i - r) \bmod w$, so no bit is lost: in C, `(x >> r) | (x << ((32 - r) & 31))`. PCG32 advances a 64-bit LCG state $s \leftarrow (as + c) \bmod 2^{64}$ with odd $a$ and odd $c$, then outputs a 32-bit word built with a XOR-shift and a rotation by the state's top 5 bits (`course/contracts/spec/pcg32.md`).

**FNV-1a 64** hashes bytes $c_1, \dots, c_k$ by starting at the offset basis $h_0 = 14695981039346656037$ and setting $h \leftarrow ((h \oplus c_i) \cdot p) \bmod 2^{64}$ with the prime $p = 1099511628211$. The order matters: FNV-1 multiplies first and XORs second, and gives a different value.

### 2.4 Hashing and load factor

A **hash table** with $m$ buckets stores a key $x$ in bucket $h(x)$. **Chaining** keeps a list per bucket. The **load factor** $\alpha = n/m$ is the average list length. Under **simple uniform hashing** (each key lands in each bucket with probability $1/m$, independently), an unsuccessful lookup computes the hash and scans one whole chain, $1 + \alpha$ probes on average. A **universal family** such as $h_{a,b}(x) = ((ax + b) \bmod p) \bmod m$ with a prime $p$ larger than any key and random $a \ne 0$, $b$ makes any two keys collide with probability at most about $1/m$, whatever the keys.

Collisions are common long before the table fills. Each of the $\binom{n}{2}$ pairs of keys collides with probability $1/m$, and expectation is linear, so the expected number of colliding pairs is $\binom{n}{2}/m = \frac{n(n-1)}{2m}$: with $m = 365$ and $n = 23$ it is already about $0.69$. Part b (`S-M06b`) turns this into the birthday bound.

## 3. Worked example by hand

This is a sibling of q3 and q22, not one of the graded problems.

**Count the topological orders** of the graph with edges $a \to c$, $b \to c$, $c \to d$ on nodes $\{a, b, c, d\}$.

$c$ needs $a$ and $b$ before it, and $d$ needs $c$, so $d$ is last and $c$ is third: positions 3 and 4 are forced. Positions 1 and 2 hold $a$ and $b$ in either order. Total: 2 orders, $(a, b, c, d)$ and $(b, a, c, d)$. In general, count by cases on which source comes first, as in q3; never assume the order is unique.

**Claim.** $x \mapsto 5x \bmod 8$ is a bijection of $\{0, \dots, 7\}$, and $x \mapsto 4x \bmod 8$ is not.

*Method: direct computation, then the general reason.* The images under $5x \bmod 8$ of $0, 1, \dots, 7$ are $0, 5, 2, 7, 4, 1, 6, 3$: all eight values, each once, so it is a bijection. The reason is $\gcd(5, 8) = 1$, and $5 \cdot 5 = 25 \equiv 1$, so multiplying by 5 is undone by multiplying by 5 again. Under $4x \bmod 8$ the images are $0, 4, 0, 4, \dots$: two values, because $\gcd(4, 8) = 4$ and every image is a multiple of 4. A multiplier sharing a factor with the modulus collapses states. That is the content of q22 and q23, and why an LCG on 64-bit words uses an odd multiplier.

## 4. The problem set

Write each answer in `solve/S-M06a.toml`; lettered parts are their own tables:

```toml
[q4]
answer = "[0, 0, 2, 1, 2]"
[q7.a]
answer = "[[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]]"
[q20]
proof = "S-M06a/q20.md"
```

Numbers are exact integers or fractions (`3/4`, `2^31 + 2^30`); no hexadecimal, so convert to decimal.

<!-- ol:problems S-M06a -->

### Graphs and DAGs

**q1.** In $G_1$: (a) what is the in-degree of node 5? (b) What is the out-degree of node 2? `[number]`

**q2.** (a) Is $(1, 2, 3, 4, 5)$ a topological order of $G_1$? (b) Is $(2, 4, 3, 1, 5)$? `[bool]`

**q3.** How many topological orders does $G_1$ have? `[number]`

**q4.** Kahn's algorithm starts from the in-degree of every node. Give the in-degrees of nodes $1, 2, 3, 4, 5$ of $G_1$ in that order. `[vector]`

**q5.** How many directed paths lead from node 2 to node 5 in $G_1$? (In reverse mode, the gradient of node 5 with respect to node 2 is a sum with one term per path.) `[number]`

**q6.** Is the graph with edges $1 \to 2$, $2 \to 3$, $3 \to 1$, $3 \to 4$ a DAG? `[bool]`

**q7.** The chain $1 \to 2 \to 3 \to 4$ has adjacency matrix $A$ with $A_{ij} = 1$ when $i \to j$ is an edge. Entry $(i, j)$ of $A^k$ counts the paths of length $k$ from $i$ to $j$. Give (a) $A^2$ and (b) $A^3$. `[matrix]`

**q8.** The autograd graph of $y = ab + b$ has nodes $a$, $b$, $t = ab$, and $y = t + b$. Backward must visit a node only after every node that consumes it. Which order is valid for backward? `[choice]`
(a) $y, t, a, b$ (b) $a, b, t, y$ (c) $t, y, a, b$ (d) $b, y, t, a$

**q9.** A recursive depth-first search that starts at the last node of a chain of $10^5$ nodes and recurses into each node's input makes how many nested calls at its deepest point, counting the first call? (Python's default recursion limit is 1000.) `[number]`

### Modular arithmetic and hashing

Here $a \bmod m$ is the remainder in $\{0, 1, \dots, m - 1\}$, also for negative $a$.

**q10.** (a) $17 \bmod 5$. (b) $-17 \bmod 5$. `[number]`

**q11.** Give the multiplicative inverse of 3 modulo 11: the $x \in \{0, \dots, 10\}$ with $3x \bmod 11 = 1$. `[number]`

**q12.** A `uint32_t` adds modulo $2^{32}$. What is $(4294967295 + 2) \bmod 2^{32}$? `[number]`

**q13.** PCG32's output step rotates a 32-bit word right. Rotating $x$ right by $r$ bits moves bit $i$ to bit $(i - r) \bmod 32$. Give the rotation of $x = 2147483649$ (bits 31 and 0 set) right by 1 bit, as a decimal number. `[number]`

**q14.** FNV-1a 64 starts at $h = 14695981039346656037$ and, for each byte $c$, sets $h \leftarrow ((h \oplus c) \cdot 1099511628211) \bmod 2^{64}$, where $\oplus$ is bitwise XOR. Give the FNV-1a 64 hash of the one-byte string `a` (byte 97), as a decimal number. `[number]`

**q15.** A hash table with chaining has $m = 16$ buckets and holds $n = 12$ keys. (a) Give its load factor $\alpha = n/m$. (b) Under simple uniform hashing, an unsuccessful search hashes once and then scans a whole chain; its expected cost is $1 + \alpha$ probes. Give it. `[number]`

**q16.** For the universal hash $h(x) = ((ax + b) \bmod p) \bmod m$ with $p = 17$, $a = 3$, $b = 5$, $m = 8$, give $h(10)$. `[number]`

**q17.** $n$ keys are hashed independently and uniformly into $m$ buckets. What is the expected number of pairs of keys that share a bucket? `[expr in n, m]`

### Proofs

**q18.** Prove that every finite, nonempty directed acyclic graph has a node with in-degree 0. `[proof]`

**q19.** Prove that a directed graph that has a topological order has no directed cycle. `[proof]`

**q20.** `M06.1` orders the autograd graph with an iterative depth-first search. Each node $v$ has a finite list of inputs, `inputs(v)`, and the graph reachable from the root is acyclic:

```
order = []; seen = {root}; stack = [(root, iterator over inputs(root))]
while stack is not empty:
    (v, it) = top of stack
    if it yields an input u that is not in seen:
        add u to seen; push (u, iterator over inputs(u))
    else if it is exhausted:
        pop (v, it); append v to order
return order
```

Prove that every node reachable from the root appears in `order` exactly once, and after all of its inputs. (Backward then walks `order` in reverse.) `[proof]`

**q21.** Prove that $(a + b) \bmod m = ((a \bmod m) + (b \bmod m)) \bmod m$ for all integers $a, b$ and every integer $m \ge 1$. `[proof]`

**q22.** Prove that if $\gcd(a, m) = 1$, then $x \mapsto ax \bmod m$ is a bijection of $\{0, 1, \dots, m - 1\}$. (This is why the LCG step $x \mapsto (ax + c) \bmod 2^{64}$ inside PCG32 never merges two states: its multiplier $a$ is odd.) `[proof]`

**q23.** Is $x \mapsto 6x \bmod 16$ a bijection of $\{0, 1, \dots, 15\}$? `[bool]`

<!-- /ol:problems -->

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| Testing for one specific topological order | a correct `toposort` fails on a valid order | q3 (canary: 1) |
| Swapping in-degree and out-degree | Kahn's algorithm starts from sinks and emits nothing | q4 (canary: out-degrees) |
| Backward in forward order, or a shared node before one of its consumers | a gradient used before every consumer added to it | q8 (canaries b and d) |
| Recursing over a long chain | `RecursionError` at depth 1000 on a long sequence | q9 (canary 1000) |
| Using C's truncating `%` for a negative operand | a negative bucket index | q10 (canary -2) |
| Forgetting to wrap at the word size | a Python port of PCG32 diverges from C after one step | q12 (canary 4294967297) |
| Shifting instead of rotating | PCG32 output loses high bits | q13 (canary 1073741824) |
| Multiplying before XOR in FNV-1a | block hashes that disagree with every other implementation | q14 (canary: the FNV-1 value) |
| A multiplier sharing a factor with the modulus | the generator or hash collapses onto a fraction of its states | q22 rubric, q23 |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `S-M05` | induction, pigeonhole, and the injective-and-surjective test behind q18 to q22 |
| Forward | `M06.1` | `toposort` for backward, iterative, checked on a $10^5$-deep chain; q20 is its correctness proof |
| Forward | `M06.3` | PCG32, SplitMix64, FNV-1a, and `universal_hash` in Python and C |
| Forward | `L0.1` | backward walks the topological order in reverse and sums gradients over consumers |
| Forward | `rt.04` | KV blocks are named by chained FNV-1a 64 hashes (DESIGN D13) |
| Forward | `S-M06b` | trees, birthday bounds, MinHash, and Bloom filters build on q15 to q17 |
