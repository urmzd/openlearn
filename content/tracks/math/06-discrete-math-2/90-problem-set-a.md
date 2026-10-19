# Discrete math 2 problem set, part a: DAGs, modular arithmetic, hashing

## Overview

| | |
|---|---|
| **Module** | `S-M06a` · solve · none · Pass 2 · 4 to 5 h |
| **Solution** | a worked answer after every problem in section 4: 18 problems with exact answers (23 parts) and 5 proofs (model proofs also in [`course/solutions/solve/S-M06a/`](../../../../course/solutions/solve/S-M06a/)) |
| **Contract** | none: a pen and paper set |
| **Needs** | `S-M05` (proof methods, induction, counting, bijections). Reading: the [Discrete Math 2 topic](README.md), graphs and number theory sections |
| **Used by** | no call site (a solve set). Do it before `M06.1` (iterative toposort for backward; q20 is its correctness proof) and `M06.3` (PCG32, SplitMix64, FNV-1a: wrapping arithmetic, rotations, and the odd multiplier of q22). Part b, `S-M06b` in Pass 3, covers trees, birthday bounds, MinHash, and Bloom filters |
| **Optional depth** | Hammack, *Book of Proof*, ch. 11 and 12; Cormen et al., *Introduction to Algorithms*, ch. 11 (hashing) and 22.4 (topological sort); O'Neill, "PCG: A Family of Simple Fast Space-Efficient Statistically Good Algorithms" (2014), sections 4 and 6 |

## Key Takeaways

- A computation graph is a DAG; a topological order lists every node after its inputs, it is rarely unique, and backward walks it in reverse (q2, q3, q8).
- Recursion depth equals the longest input chain, so a recursive traversal of a $10^5$-node chain dies at Python's limit of 1000; the iterative DFS of q20 does not (q9).
- Fixed-width integers are arithmetic modulo $2^w$: addition wraps, rotation moves bits around the word, and $x \mapsto ax \bmod 2^w$ permutes the words exactly when $a$ is odd (q12, q13, q22, q23).
- FNV-1a XORs a byte in, then multiplies; swapping the two steps gives FNV-1, a different hash (q14).
- With $n$ keys in $m$ buckets the expected number of colliding pairs is $\binom{n}{2}/m$, which is why collisions appear long before the table is full (q17).

## How to read this chapter

Sections 1 to 3 review graphs, modular arithmetic, and hashing and work two examples by hand. Section 4 is the problem set: every problem is followed by a collapsed worked answer with the final result or a complete model proof, and the common wrong answers and why they are wrong. Try each problem before opening its answer.

---

## 1. Why now

The bigram model of Pass 1 needed no graph: its gradient was a count. In Pass 2 the system gains an autograd engine (`L0.1`), and backward is only correct if every node's gradient is complete before it is passed on to that node's inputs, which means visiting the graph in reverse topological order. `M06.1` builds that order with an iterative depth-first search, because a recursive one crashes on the $10^5$-node chains that long sequences produce. In the same pass `M06.3` builds PCG32, the random generator every later module draws from, and FNV-1a, the hash that `rt.04` uses to name KV-cache blocks. Both are arithmetic on 32- and 64-bit words that silently wrap. This set builds fluency in directed acyclic graphs, modular arithmetic, and the basic probability of hashing before reading that code.

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

A **topological order** lists all nodes so that every edge goes forward: $u \to v$ implies $u$ is listed before $v$. A graph has one exactly when it is acyclic (q19 one way; q18 gives the other). Topological orders are rarely unique: any two nodes not connected by a path can be swapped. That is why a correct `M06.1` solution is judged by the property "every edge goes forward", never by one specific list.

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

This is a sibling of q3 and q22, not one of the problems of the set.

**Count the topological orders** of the graph with edges $a \to c$, $b \to c$, $c \to d$ on nodes $\{a, b, c, d\}$.

$c$ needs $a$ and $b$ before it, and $d$ needs $c$, so $d$ is last and $c$ is third: positions 3 and 4 are forced. Positions 1 and 2 hold $a$ and $b$ in either order. Total: 2 orders, $(a, b, c, d)$ and $(b, a, c, d)$. In general, count by cases on which source comes first, as in q3; never assume the order is unique.

**Claim.** $x \mapsto 5x \bmod 8$ is a bijection of $\{0, \dots, 7\}$, and $x \mapsto 4x \bmod 8$ is not.

*Method: direct computation, then the general reason.* The images under $5x \bmod 8$ of $0, 1, \dots, 7$ are $0, 5, 2, 7, 4, 1, 6, 3$: all eight values, each once, so it is a bijection. The reason is $\gcd(5, 8) = 1$, and $5 \cdot 5 = 25 \equiv 1$, so multiplying by 5 is undone by multiplying by 5 again. Under $4x \bmod 8$ the images are $0, 4, 0, 4, \dots$: two values, because $\gcd(4, 8) = 4$ and every image is a multiple of 4. A multiplier sharing a factor with the modulus collapses states. That is the content of q22 and q23, and why an LCG on 64-bit words uses an odd multiplier.

## 4. The problem set, with worked answers

Answers are exact integers or fractions ($3/4$, $2^{31} + 2^{30}$), written in decimal. Lettered parts share one answer block. The tag after each problem says what kind of answer is expected.

Throughout, $G_1$ is the directed graph on nodes $\{1, 2, 3, 4, 5\}$ with edges $1 \to 3$, $2 \to 3$, $3 \to 5$, $2 \to 4$, $4 \to 5$. An edge $u \to v$ means "$u$ is an input of $v$", so $u$ must be computed before $v$.

### Graphs and DAGs

**q1.** In $G_1$: (a) what is the in-degree of node 5? (b) What is the out-degree of node 2? `[number]`

<details><summary>Answer</summary>

**(a) $2$, (b) $2$.** (a) Edges into 5: $3 \to 5$ and $4 \to 5$. (b) Edges out of 2: $2 \to 3$ and $2 \to 4$. In-degree drives Kahn's algorithm; out-degree is how many consumers a gradient sums over. A common mistake is 0 for either part, which reads the other degree (node 5 has out-degree 0, node 2 in-degree 0).

</details>

**q2.** (a) Is $(1, 2, 3, 4, 5)$ a topological order of $G_1$? (b) Is $(2, 4, 3, 1, 5)$? `[bool]`

<details><summary>Answer</summary>

**(a) True, (b) false.** Check each edge goes forward. (a) $1 \to 3$, $2 \to 3$, $2 \to 4$, $3 \to 5$, $4 \to 5$ all go from an earlier to a later position. (b) $1 \to 3$ goes backward: 3 is at position 3 and 1 at position 4. One backward edge is enough to reject an order, and checking the edge property is how a correct `M06.1` output is recognized.

</details>

**q3.** How many topological orders does $G_1$ have? `[number]`

<details><summary>Answer</summary>

**$5$.** Node 5 needs both 3 and 4, so it is last. Among $\{1, 2, 3, 4\}$, node 3 must follow 1 and 2, and 4 must follow 2. By cases on the first node, which must be a source (1 or 2): starting with 1, then 2 is forced (3 and 4 both wait on 2), then 3 and 4 in either order: 2 orders. Starting with 2, the remaining $\{1, 3, 4\}$ need only 1 before 3: the orders $1, 3, 4$; $1, 4, 3$; $4, 1, 3$: 3 orders. Total 5. Common mistakes: 1, which assumes the order is unique, and $5! = 120$, which counts every permutation.

</details>

**q4.** Kahn's algorithm starts from the in-degree of every node. Give the in-degrees of nodes $1, 2, 3, 4, 5$ of $G_1$ in that order. `[vector]`

<details><summary>Answer</summary>

**$(0, 0, 2, 1, 2)$.** Count incoming edges: nodes 1 and 2 have none, 3 has two (from 1 and 2), 4 has one (from 2), 5 has two (from 3 and 4). Kahn's algorithm starts from the zeros, 1 and 2. A common mistake is $(1, 2, 1, 1, 0)$, the out-degrees, which would start from the sink 5.

</details>

**q5.** How many directed paths lead from node 2 to node 5 in $G_1$? (In reverse mode, the gradient of node 5 with respect to node 2 is a sum with one term per path.) `[number]`

<details><summary>Answer</summary>

**$2$.** $2 \to 3 \to 5$ and $2 \to 4 \to 5$. The chain rule gives $\frac{\partial v_5}{\partial v_2}$ as the sum of two products of local derivatives, one per path, which is why reverse mode adds gradients from every consumer.

</details>

**q6.** Is the graph with edges $1 \to 2$, $2 \to 3$, $3 \to 1$, $3 \to 4$ a DAG? `[bool]`

<details><summary>Answer</summary>

**False.** $1 \to 2 \to 3 \to 1$ is a directed cycle. The extra edge $3 \to 4$ does not matter: one cycle is enough.

</details>

**q7.** The chain $1 \to 2 \to 3 \to 4$ has adjacency matrix $A$ with $A_{ij} = 1$ when $i \to j$ is an edge. Entry $(i, j)$ of $A^k$ counts the paths of length $k$ from $i$ to $j$. Give (a) $A^2$ and (b) $A^3$. `[matrix]`

<details><summary>Answer</summary>

**(a) $A^2 = \begin{pmatrix} 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \end{pmatrix}$, (b) $A^3 = \begin{pmatrix} 0 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \end{pmatrix}$.** The paths of length 2 are $1 \to 2 \to 3$ and $2 \to 3 \to 4$, so $(A^2)_{13} = (A^2)_{24} = 1$. The only path of length 3 is $1 \to 2 \to 3 \to 4$. $A^4 = 0$: a DAG's adjacency matrix is nilpotent. Common mistakes: $A$ itself for (a), $(A^\top)^2$ (paths reversed), and the zero matrix for (b), which is $A^4$.

</details>

**q8.** The autograd graph of $y = ab + b$ has nodes $a$, $b$, $t = ab$, and $y = t + b$. Backward must visit a node only after every node that consumes it. Which order is valid for backward? `[choice]`
(a) $y, t, a, b$ (b) $a, b, t, y$ (c) $t, y, a, b$ (d) $b, y, t, a$

<details><summary>Answer</summary>

**(a).** Backward visits a node only after all of its consumers. $y$ consumes $t$ and $b$; $t$ consumes $a$ and $b$. So $y$ first, then $t$; $b$ must come after both $y$ and $t$. Order (a), $y, t, a, b$, satisfies this. Common mistakes: (b) is the forward order; (d) visits $b$ before its consumer $t$, so $b$'s gradient would be passed on before $t$'s contribution is added; (c) visits $t$ before its consumer $y$.

</details>

**q9.** A recursive depth-first search that starts at the last node of a chain of $10^5$ nodes and recurses into each node's input makes how many nested calls at its deepest point, counting the first call? (Python's default recursion limit is 1000.) `[number]`

<details><summary>Answer</summary>

**$100{,}000$.** The search starts at the last node and recurses into its input, one call per node down the chain, so the deepest point has all $10^5$ calls on the stack, 100 times Python's limit. This is why `M06.1` uses an explicit stack. Common mistakes: 1000 is the limit, not the depth; 17 $\approx \log_2 10^5$ would be the depth of a balanced tree, not a chain.

</details>

### Modular arithmetic and hashing

Here $a \bmod m$ is the remainder in $\{0, 1, \dots, m - 1\}$, also for negative $a$.

**q10.** (a) $17 \bmod 5$. (b) $-17 \bmod 5$. `[number]`

<details><summary>Answer</summary>

**(a) $2$, (b) $3$.** (a) $17 = 3 \cdot 5 + 2$. (b) $-17 = -4 \cdot 5 + 3$, and the remainder must be in $\{0, \dots, 4\}$. A common mistake is $-2$, which is C's truncating `%` ($-17 = -3 \cdot 5 - 2$); Python's `%` gives 3. Answering 2 for (b) drops the sign.

</details>

**q11.** Give the multiplicative inverse of 3 modulo 11: the $x \in \{0, \dots, 10\}$ with $3x \bmod 11 = 1$. `[number]`

<details><summary>Answer</summary>

**$4$.** $3 \cdot 4 = 12 = 11 + 1$. The inverse exists because $\gcd(3, 11) = 1$. A common mistake is $1/3$, the rational inverse, which is not an element of $\{0, \dots, 10\}$.

</details>

**q12.** A `uint32_t` adds modulo $2^{32}$. What is $(4294967295 + 2) \bmod 2^{32}$? `[number]`

<details><summary>Answer</summary>

**$1$.** $4294967295 = 2^{32} - 1$, so the sum is $2^{32} + 1 \equiv 1$. A common mistake is 4294967297, which does not wrap: that is what a Python port of PCG32 produces without `& 0xFFFFFFFF`.

</details>

**q13.** PCG32's output step rotates a 32-bit word right. Rotating $x$ right by $r$ bits moves bit $i$ to bit $(i - r) \bmod 32$. Give the rotation of $x = 2147483649$ (bits 31 and 0 set) right by 1 bit, as a decimal number. `[number]`

<details><summary>Answer</summary>

**$3{,}221{,}225{,}472$.** Bit 31 moves to bit 30 and bit 0 moves to bit $(0 - 1) \bmod 32 = 31$, so the result has bits 31 and 30 set: $2^{31} + 2^{30}$. This rotation is the last step of PCG32's XSH-RR output ([`pcg32.md`](../../../../course/contracts/spec/pcg32.md)). Common mistakes: $1{,}073{,}741{,}824 = 2^{30}$ is a logical shift, which loses the low bit; 3 rotates left.

</details>

**q14.** FNV-1a 64 starts at $h = 14695981039346656037$ and, for each byte $c$, sets $h \leftarrow ((h \oplus c) \cdot 1099511628211) \bmod 2^{64}$, where $\oplus$ is bitwise XOR. Give the FNV-1a 64 hash of the one-byte string `a` (byte 97), as a decimal number. `[number]`

<details><summary>Answer</summary>

**$12{,}638{,}187{,}200{,}555{,}641{,}996$.** XOR first: the offset basis is `0xcbf29ce484222325`, and $\texttt{0x25} \oplus \texttt{0x61} = \texttt{0x44}$, so $h \oplus 97 = 14695981039346656068$. Then multiply by the prime and reduce modulo $2^{64}$: $(14695981039346656068 \cdot 1099511628211) \bmod 2^{64} = 12638187200555641996$ (`0xaf63dc4c8601ec8c`). A common mistake is 12638153115695167422, the FNV-1 value (multiply, then XOR): the KV block hashes of `rt.04` would then disagree with every other implementation.

</details>

**q15.** A hash table with chaining has $m = 16$ buckets and holds $n = 12$ keys. (a) Give its load factor $\alpha = n/m$. (b) Under simple uniform hashing, an unsuccessful search hashes once and then scans a whole chain; its expected cost is $1 + \alpha$ probes. Give it. `[number]`

<details><summary>Answer</summary>

**(a) $3/4$, (b) $7/4$.** (a) $\alpha = 12/16$. (b) $1 + \alpha = 7/4$: one probe for the hash and on average $\alpha$ for the chain. Common mistakes: $4/3 = m/n$ for (a), and $3/4$ for (b), which forgets the hash computation itself.

</details>

**q16.** For the universal hash $h(x) = ((ax + b) \bmod p) \bmod m$ with $p = 17$, $a = 3$, $b = 5$, $m = 8$, give $h(10)$. `[number]`

<details><summary>Answer</summary>

**$1$.** $3 \cdot 10 + 5 = 35$, $35 \bmod 17 = 1$, $1 \bmod 8 = 1$. A common mistake is 3, which is $35 \bmod 8$ and skips the reduction modulo $p$, so the family is no longer universal.

</details>

**q17.** $n$ keys are hashed independently and uniformly into $m$ buckets. What is the expected number of pairs of keys that share a bucket? `[expr in n, m]`

<details><summary>Answer</summary>

**$\frac{n(n - 1)}{2m}$.** Let $X_{ij}$ be 1 when keys $i < j$ share a bucket; $\Pr[X_{ij} = 1] = 1/m$. By linearity of expectation, $\mathbb{E}\left[\sum_{i<j} X_{ij}\right] = \binom{n}{2} \cdot \frac{1}{m}$, with no independence needed between pairs. Common mistakes: $n/m$ is the load factor, and $n^2/m$ counts ordered pairs including each key with itself.

</details>

### Proofs

**q18.** Prove that every finite, nonempty directed acyclic graph has a node with in-degree 0. `[proof]`

<details><summary>Answer</summary>

Claim: every finite, nonempty directed acyclic graph $G$ (a DAG) has a node with in-degree 0.

Method: contradiction, with the pigeonhole principle.

Let $G$ have $n \ge 1$ nodes and suppose every node has in-degree at least 1, so every node has at least one input. Start at any node $v_0$, pick an input $v_1$ of it (an edge $v_1 \to v_0$), then an input $v_2$ of $v_1$, and so on: each step is possible because every node has an input. After $n$ steps we have $n + 1$ nodes $v_0, v_1, \dots, v_n$ with edges $v_{i+1} \to v_i$.

There are only $n$ distinct nodes, so by pigeonhole $v_i = v_j$ for some $i < j$. Then $v_j \to v_{j-1} \to \dots \to v_i = v_j$ is a directed cycle, contradicting that $G$ is acyclic.

So some node has in-degree 0. (Kahn's algorithm relies on this at every step: removing a source from a DAG leaves a DAG, which again has a source.)

Model proof also in [`q18.md`](../../../../course/solutions/solve/S-M06a/q18.md).

</details>

**q19.** Prove that a directed graph that has a topological order has no directed cycle. `[proof]`

<details><summary>Answer</summary>

Claim: if a directed graph $G$ has a topological order, then $G$ has no directed cycle.

Definitions: a topological order is a listing of all nodes, with position $\mathrm{pos}(v)$, such that every edge $u \to v$ has $\mathrm{pos}(u) < \mathrm{pos}(v)$. A directed cycle is a sequence of edges $v_1 \to v_2 \to \dots \to v_k \to v_1$ with $k \ge 1$.

Method: contradiction.

Suppose $G$ has a topological order and a directed cycle $v_1 \to v_2 \to \dots \to v_k \to v_1$. Applying the order property to each edge of the cycle gives $\mathrm{pos}(v_1) < \mathrm{pos}(v_2) < \dots < \mathrm{pos}(v_k) < \mathrm{pos}(v_1)$, and by transitivity of $<$, $\mathrm{pos}(v_1) < \mathrm{pos}(v_1)$, which is false.

So $G$ has no directed cycle. (With q18 the converse also holds: repeatedly removing a node of in-degree 0 lists any DAG in topological order.)

Model proof also in [`q19.md`](../../../../course/solutions/solve/S-M06a/q19.md).

</details>

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

<details><summary>Answer</summary>

Claim: for a root whose reachable graph is finite and acyclic, the iterative DFS appends every reachable node to `order` exactly once, and each node after all of its inputs.

Method: three invariants of the loop, then acyclicity.

(1) At most once. A node is pushed only when it is not in `seen`, and it is added to `seen` at that moment (the root at the start), so each node is pushed at most once. A node is appended only when its own stack entry is popped, so it is appended at most once.

(2) Every reachable node is pushed. Induction on $d(u)$, the length of the shortest input path from the root to $u$. For $d = 0$, $u$ is the root, pushed at the start. If $u$ is an input of a node $v$ with $d(v) = d(u) - 1$, then $v$ was pushed by the hypothesis. The loop only pops $v$ once its iterator is exhausted, so before that the iterator yielded $u$; at that moment $u$ was either not seen (and is pushed now) or already seen (and was pushed earlier). The loop ends only with an empty stack; each iteration either pushes a node (at most $n$ times in total) or advances an iterator or pops, so it terminates, and every pushed node is eventually popped and appended. With (1), every reachable node is appended exactly once.

(3) Inputs first. First a stack invariant: if the stack holds $s_0, s_1, \dots, s_t$ from bottom to top, then each $s_{i+1}$ is an input of $s_i$, because a node is pushed only when the iterator of the node then on top yields it. Write $x \to y$ when $x$ is an input of $y$; the invariant says $s_t \to s_{t-1} \to \dots \to s_0$.

Now let $v$ be appended. Its iterator yielded every input $u$ of $v$, each at a moment when $v$ was on top. If $u$ was not seen then, it was pushed on top of $v$; the stack is last in, first out, so $u$ was popped and appended before $v$. If $u$ was already seen, it was pushed earlier, and either it has already been appended (fine), or it is still on the stack below $v$: $u = s_i$ with $v = s_t$, $i < t$. In that last case the invariant gives a path $v = s_t \to \dots \to s_i = u$, and $u \to v$ closes it into a directed cycle, which contradicts acyclicity. ($u = v$ is impossible too, since $v \to v$ would be a cycle.)

So every input of $v$ is appended before $v$. Walking `order` in reverse visits every node before its inputs, which is the order backward needs.

Model proof also in [`q20.md`](../../../../course/solutions/solve/S-M06a/q20.md).

</details>

**q21.** Prove that $(a + b) \bmod m = ((a \bmod m) + (b \bmod m)) \bmod m$ for all integers $a, b$ and every integer $m \ge 1$. `[proof]`

<details><summary>Answer</summary>

Claim: for all integers $a, b$ and every integer $m \ge 1$, $(a + b) \bmod m = ((a \bmod m) + (b \bmod m)) \bmod m$.

Definitions: by division with remainder, every integer $x$ is $x = qm + r$ for unique integers $q$ and $r$ with $0 \le r < m$; $x \bmod m$ is that $r$. Uniqueness implies: if $x - y$ is a multiple of $m$, then $x \bmod m = y \bmod m$ (writing $y = qm + r$ gives $x = (q + k)m + r$ with the same $r$).

Method: direct.

Write $a = q_1 m + r_1$ and $b = q_2 m + r_2$ with $r_1 = a \bmod m$ and $r_2 = b \bmod m$. Then $(a + b) - (r_1 + r_2) = (q_1 + q_2) m$, a multiple of $m$. By the uniqueness remark, $(a + b) \bmod m = (r_1 + r_2) \bmod m$, which is the claim. This is why a hash or an RNG may reduce after every step instead of once at the end, and why a `uint64_t` that wraps computes the right value modulo $2^{64}$.

Model proof also in [`q21.md`](../../../../course/solutions/solve/S-M06a/q21.md).

</details>

**q22.** Prove that if $\gcd(a, m) = 1$, then $x \mapsto ax \bmod m$ is a bijection of $\{0, 1, \dots, m - 1\}$. (This is why the LCG step $x \mapsto (ax + c) \bmod 2^{64}$ inside PCG32 never merges two states: its multiplier $a$ is odd.) `[proof]`

<details><summary>Answer</summary>

Claim: if $\gcd(a, m) = 1$, then $f(x) = ax \bmod m$ is a bijection of $Z_m = \{0, 1, \dots, m - 1\}$.

Method: show $f$ is injective; an injective map from a finite set to itself is a bijection, because its image has as many elements as the set itself.

Suppose $f(x) = f(y)$ for $x, y \in Z_m$. Then $ax$ and $ay$ have the same remainder, so $m$ divides $ax - ay = a(x - y)$.

Since $\gcd(a, m) = 1$, Bezout's identity gives integers $s, t$ with $sa + tm = 1$. Multiply by $(x - y)$: $x - y = s \cdot a(x - y) + t m (x - y)$. Both terms on the right are multiples of $m$, so $m$ divides $x - y$ (this is Euclid's lemma).

But $x, y \in \{0, \dots, m - 1\}$, so $\lvert x - y \rvert < m$, and the only multiple of $m$ in that range is 0. So $x = y$, $f$ is injective, and therefore a bijection of $Z_m$.

For $m = 2^{64}$ this says any odd multiplier permutes the 64-bit states: PCG32's LCG step never sends two states to one. q23 shows what goes wrong when $\gcd(a, m) > 1$.

Model proof also in [`q22.md`](../../../../course/solutions/solve/S-M06a/q22.md).

</details>

**q23.** Is $x \mapsto 6x \bmod 16$ a bijection of $\{0, 1, \dots, 15\}$? `[bool]`

<details><summary>Answer</summary>

**False.** $\gcd(6, 16) = 2$, so every image $6x \bmod 16$ is even: only the 8 values $0, 2, \dots, 14$ are reached, and for example $6 \cdot 0 \equiv 6 \cdot 8 \equiv 0 \pmod{16}$. Two inputs share each image. This is q22's hypothesis failing, and why an LCG's multiplier must be odd.

</details>

## 5. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| Testing for one specific topological order | a correct `toposort` fails on a valid order | checks that every edge goes forward instead of comparing with one list (q2, q3) |
| Swapping in-degree and out-degree | Kahn's algorithm starts from sinks and emits nothing | counts edges *into* each node (q4) |
| Backward in forward order, or a shared node before one of its consumers | a gradient used before every consumer added to it | visits a node only after all of its consumers: reverse topological order (q8) |
| Recursing over a long chain | `RecursionError` at depth 1000 on a long sequence | uses an explicit stack, whose depth is bounded only by memory (q9, q20) |
| Using C's truncating `%` for a negative operand | a negative bucket index | takes the remainder in $\{0, \dots, m - 1\}$, as Python's `%` does (q10) |
| Forgetting to wrap at the word size | a Python port of PCG32 diverges from C after one step | reduces modulo $2^w$ after every operation (q12, q21) |
| Shifting instead of rotating | PCG32 output loses high bits | ORs the shifted-out bits back in at the top (q13) |
| Multiplying before XOR in FNV-1a | block hashes that disagree with every other implementation | XORs the byte in first, then multiplies (q14) |
| A multiplier sharing a factor with the modulus | the generator or hash collapses onto a fraction of its states | checks $\gcd(a, m) = 1$; for $m = 2^w$, uses an odd multiplier (q22, q23) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `S-M05` | induction, pigeonhole, and the injective-and-surjective test behind q18 to q22 |
| Forward | `M06.1` | `toposort` for backward, iterative so that a $10^5$-deep chain works; q20 is its correctness proof |
| Forward | `M06.3` | PCG32, SplitMix64, FNV-1a, and `universal_hash` in Python and C |
| Forward | `L0.1` | backward walks the topological order in reverse and sums gradients over consumers |
| Forward | `rt.04` | KV blocks are named by chained FNV-1a 64 hashes |
| Forward | `S-M06b` | trees, birthday bounds, MinHash, and Bloom filters build on q15 to q17 |
