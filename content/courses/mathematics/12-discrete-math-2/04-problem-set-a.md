# Discrete math 2 problem set, part a: DAGs, modular arithmetic, hashing

The bigram model at the start of *AI Engineering* needed no graph: its gradient was a count. Once the system gains an autograd engine, in [Tensor, broadcasting backward, and no_grad](../../ai-engineering/06-numerics-and-autograd/02-tensor-and-broadcasting-backward.md), backward is only correct if every node's gradient is complete before it is passed on to that node's inputs, which means visiting the graph in reverse topological order. [Graphs, DAGs, and an iterative topological sort](01-graphs-dags-and-topological-sort.md) builds that order with an iterative depth-first search, because a recursive one crashes on the $10^5$-node chains that long sequences produce.

At the same point, [Modular arithmetic, hashing, and PCG32 in Python](03-modular-arithmetic-hashing-and-pcg32.md) builds PCG32, the random generator every later section draws from, and FNV-1a, the hash that the [Paged KV block pool](../../ai-engineering/21-inference/08-paged-kv-block-pool.md) uses to name KV-cache blocks. Both are arithmetic on 32- and 64-bit words that silently wrap. This problem set builds fluency in directed acyclic graphs, modular arithmetic, and the basic probability of hashing before you read that code. It is a pen-and-paper set with no code to write: 18 problems with exact answers (23 parts in all) and 5 proofs. Problem 20 is the correctness proof of the topological sort, and Problem 22 explains the odd multiplier inside PCG32.

You will need the proof methods, induction, counting, and bijections of the [Discrete Math 1 problem set](../11-discrete-math-1/03-problem-set/README.md), and the graph and number-theory material of [the chapter opening](README.md). Every problem is followed by a collapsed worked answer with the final result or a complete model proof, together with the common wrong answers and why they are wrong. Try each problem before opening its answer.

> **Learning objectives**
> - Recognize a valid topological order, count the orders of a small DAG, and explain why the order is rarely unique.
> - Relate recursion depth to the longest input chain, and prove an iterative DFS correct.
> - Compute remainders, inverses, wrapping sums, and rotations on machine words, and prove when $x \mapsto ax \bmod m$ is a bijection.
> - Compute FNV-1a and a universal hash by hand, and the expected number of colliding pairs in a hash table.

## Directed graphs and DAGs

The notation is shared by the review and the problems.

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

A **directed graph** is a set of nodes and a set of ordered pairs of nodes, the edges. A **path** is a sequence of edges $v_0 \to v_1 \to \dots \to v_k$, and a **cycle** is a path of length at least 1 that returns to its start. A graph with no cycle is a **directed acyclic graph (DAG)**. Every expression a program evaluates is a DAG: the nodes are values, and $u \to v$ when $u$ is an input of the operation that makes $v$. A value used twice ($b$ in $ab + b$) has out-degree 2, which is why a gradient must be summed over consumers. Note that this set draws edges from input to consumer, the forward direction, while the topological-sort section draws them from a node to its parents; the two are the same graph with the arrows reversed.

A **topological order** lists all nodes so that every edge goes forward: $u \to v$ implies $u$ is listed before $v$. A graph has one exactly when it is acyclic (Problem 19 proves one direction; Problem 18 gives the other). Topological orders are rarely unique: any two nodes not connected by a path can be swapped. That is why an arbitrary topological sort is judged by the property "every edge goes forward", never by one specific list. (The implementation in the topological-sort section additionally fixes one particular order, for bit-identical gradients.)

Two algorithms produce one. **Kahn's algorithm** keeps the in-degree of every node, repeatedly outputs a node of in-degree 0, and decrements the in-degrees of the nodes it points to. **Depth-first search** outputs a node after all of its inputs have been output (post-order). Written recursively, DFS uses one stack frame per node on the current path, so its depth is the length of the longest input chain, and Python raises `RecursionError` past 1000 frames. Written with an explicit stack (Problem 20), its depth is bounded only by memory.

Paths also count. Entry $(i, j)$ of $A^k$ is the number of paths of length exactly $k$ from $i$ to $j$, because $(A^k)_{ij} = \sum_{l} (A^{k-1})_{il} A_{lj}$ extends each path of length $k - 1$ by one edge. A DAG on $n$ nodes has no path longer than $n - 1$ edges, so $A^n = 0$. In reverse mode, the derivative of an output with respect to an input is a sum over all paths between them of the product of the local derivatives along each path: the chain rule of [Jacobians, the multivariable chain rule, and numeric JVP/VJP](../10-calculus-3/02-jacobians-and-the-chain-rule.md), organized by the graph.

## Modular arithmetic

Graphs fix the order of computation; modular arithmetic describes what each step of a generator or hash computes. **Division with remainder** says that for integers $a$ and $m \ge 1$ there are unique integers $q$ and $r$ with $a = qm + r$ and $0 \le r < m$; write $a \bmod m = r$. For negative $a$ the remainder is still in $\{0, \dots, m - 1\}$: $-17 = -4 \cdot 5 + 3$. (C's `%` truncates toward zero and returns $-2$; Python's `%` returns 3.) Two integers are **congruent modulo $m$**, $a \equiv b$, when $m$ divides $a - b$. Congruence is an equivalence relation, and it respects $+$ and $\times$ (Problem 21), so you may reduce after every step.

The **multiplicative inverse** of $a$ modulo $m$ is an $x$ with $ax \equiv 1$; it exists exactly when $\gcd(a, m) = 1$. Then multiplication by $a$ permutes $\{0, \dots, m-1\}$ (Problem 22); when $\gcd(a, m) = g > 1$, it maps everything onto multiples of $g$ and loses states (Problem 23).

## Machine words

A `uint32_t` holds $w = 32$ bits and its arithmetic is arithmetic modulo $2^{32}$: $2^{32} - 1 + 2$ wraps to 1. C, Rust, and Go all expose this (C on unsigned types, Rust with `wrapping_add` and `wrapping_mul`, Go on `uint32` and `uint64`), and Python emulates it with `& 0xFFFFFFFF`. A **logical shift** right by $r$ drops the low $r$ bits and fills with zeros. A **rotation** right by $r$ moves bit $i$ to bit $(i - r) \bmod w$, so no bit is lost: in C, `(x >> r) | (x << ((32 - r) & 31))`. PCG32 advances a 64-bit LCG state $s \leftarrow (as + c) \bmod 2^{64}$ with odd $a$ and odd $c$, then outputs a 32-bit word built with a XOR-shift and a rotation by the state's top 5 bits ([`course/contracts/spec/pcg32.md`](../../../../course/contracts/spec/pcg32.md)).

**FNV-1a 64** hashes bytes $c_1, \dots, c_k$ by starting at the offset basis $h_0 = 14695981039346656037$ and setting $h \leftarrow ((h \oplus c_i) \cdot p) \bmod 2^{64}$ with the prime $p = 1099511628211$. The order matters: FNV-1 multiplies first and XORs second, and gives a different value.

## Hashing and load factor

A **hash table** with $m$ buckets stores a key $x$ in bucket $h(x)$. **Chaining** keeps a list per bucket. The **load factor** $\alpha = n/m$ is the average list length. Under **simple uniform hashing** (each key lands in each bucket with probability $1/m$, independently), an unsuccessful lookup computes the hash and scans one whole chain, $1 + \alpha$ probes on average. A **universal family** such as $h_{a,b}(x) = ((ax + b) \bmod p) \bmod m$ with a prime $p$ larger than any key and random $a \ne 0$, $b$ makes any two keys collide with probability at most about $1/m$, whatever the keys.

Collisions are common long before the table fills. Each of the $\binom{n}{2}$ pairs of keys collides with probability $1/m$, and expectation is linear, so the expected number of colliding pairs is $\binom{n}{2}/m = \frac{n(n-1)}{2m}$. With $m = 365$ and $n = 23$ it is already about $0.69$. [Part b of the problem set](05-problem-set-b.md) turns this into the birthday bound.

## Working two examples by hand

The two examples here are siblings of Problems 3 and 22, not problems of the set. First, count the topological orders of the graph with edges $a \to c$, $b \to c$, $c \to d$ on nodes $\{a, b, c, d\}$.

$c$ needs $a$ and $b$ before it, and $d$ needs $c$, so $d$ is last and $c$ is third: positions 3 and 4 are forced. Positions 1 and 2 hold $a$ and $b$ in either order. That gives 2 orders, $(a, b, c, d)$ and $(b, a, c, d)$. In general, count by cases on which source comes first, as in Problem 3; never assume the order is unique.

Second, claim: $x \mapsto 5x \bmod 8$ is a bijection of $\{0, \dots, 7\}$, and $x \mapsto 4x \bmod 8$ is not. The method is direct computation, then the general reason. The images under $5x \bmod 8$ of $0, 1, \dots, 7$ are $0, 5, 2, 7, 4, 1, 6, 3$: all eight values, each once, so it is a bijection. The reason is $\gcd(5, 8) = 1$, and $5 \cdot 5 = 25 \equiv 1$, so multiplying by 5 is undone by multiplying by 5 again. Under $4x \bmod 8$ the images are $0, 4, 0, 4, \dots$: two values, because $\gcd(4, 8) = 4$ and every image is a multiple of 4. A multiplier sharing a factor with the modulus collapses states. That is the content of Problems 22 and 23, and the reason an LCG on 64-bit words uses an odd multiplier.

## How the problems work

Answers are exact integers or fractions ($3/4$, $2^{31} + 2^{30}$), written in decimal. Lettered parts share one answer block. The tag after each problem says what kind of answer is expected.

Throughout, $G_1$ is the directed graph on nodes $\{1, 2, 3, 4, 5\}$ with edges $1 \to 3$, $2 \to 3$, $3 \to 5$, $2 \to 4$, $4 \to 5$. An edge $u \to v$ means "$u$ is an input of $v$", so $u$ must be computed before $v$.

## Problems on graphs and DAGs

The first nine problems work on $G_1$ and two small companions. They build up from degrees and orders to counting paths and the recursion depth that motivates an iterative traversal.

**Problem 1.** In $G_1$: (a) what is the in-degree of node 5? (b) What is the out-degree of node 2? `[number]`

<details><summary>Answer</summary>

**(a) $2$, (b) $2$.** (a) Edges into 5: $3 \to 5$ and $4 \to 5$. (b) Edges out of 2: $2 \to 3$ and $2 \to 4$. In-degree drives Kahn's algorithm; out-degree is how many consumers a gradient sums over. A common mistake is 0 for either part, which reads the other degree (node 5 has out-degree 0, node 2 in-degree 0).

</details>

**Problem 2.** (a) Is $(1, 2, 3, 4, 5)$ a topological order of $G_1$? (b) Is $(2, 4, 3, 1, 5)$? `[bool]`

<details><summary>Answer</summary>

**(a) True, (b) false.** Check each edge goes forward. (a) $1 \to 3$, $2 \to 3$, $2 \to 4$, $3 \to 5$, $4 \to 5$ all go from an earlier to a later position. (b) $1 \to 3$ goes backward: 3 is at position 3 and 1 at position 4. One backward edge is enough to reject an order, and checking the edge property is how a correct topological sort is recognized.

</details>

**Problem 3.** How many topological orders does $G_1$ have? `[number]`

<details><summary>Answer</summary>

**$5$.** Node 5 needs both 3 and 4, so it is last. Among $\{1, 2, 3, 4\}$, node 3 must follow 1 and 2, and 4 must follow 2. By cases on the first node, which must be a source (1 or 2): starting with 1, then 2 is forced (3 and 4 both wait on 2), then 3 and 4 in either order: 2 orders. Starting with 2, the remaining $\{1, 3, 4\}$ need only 1 before 3: the orders $1, 3, 4$; $1, 4, 3$; $4, 1, 3$: 3 orders. Total 5. Common mistakes: 1, which assumes the order is unique, and $5! = 120$, which counts every permutation.

</details>

**Problem 4.** Kahn's algorithm starts from the in-degree of every node. Give the in-degrees of nodes $1, 2, 3, 4, 5$ of $G_1$ in that order. `[vector]`

<details><summary>Answer</summary>

**$(0, 0, 2, 1, 2)$.** Count incoming edges: nodes 1 and 2 have none, 3 has two (from 1 and 2), 4 has one (from 2), 5 has two (from 3 and 4). Kahn's algorithm starts from the zeros, 1 and 2. A common mistake is $(1, 2, 1, 1, 0)$, the out-degrees, which would start from the sink 5.

</details>

**Problem 5.** How many directed paths lead from node 2 to node 5 in $G_1$? (In reverse mode, the gradient of node 5 with respect to node 2 is a sum with one term per path.) `[number]`

<details><summary>Answer</summary>

**$2$.** $2 \to 3 \to 5$ and $2 \to 4 \to 5$. The chain rule gives $\frac{\partial v_5}{\partial v_2}$ as the sum of two products of local derivatives, one per path, which is why reverse mode adds gradients from every consumer.

</details>

**Problem 6.** Is the graph with edges $1 \to 2$, $2 \to 3$, $3 \to 1$, $3 \to 4$ a DAG? `[bool]`

<details><summary>Answer</summary>

**False.** $1 \to 2 \to 3 \to 1$ is a directed cycle. The extra edge $3 \to 4$ does not matter: one cycle is enough.

</details>

**Problem 7.** The chain $1 \to 2 \to 3 \to 4$ has adjacency matrix $A$ with $A_{ij} = 1$ when $i \to j$ is an edge. Entry $(i, j)$ of $A^k$ counts the paths of length $k$ from $i$ to $j$. Give (a) $A^2$ and (b) $A^3$. `[matrix]`

<details><summary>Answer</summary>

**(a) $A^2 = \begin{pmatrix} 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \end{pmatrix}$, (b) $A^3 = \begin{pmatrix} 0 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \end{pmatrix}$.** The paths of length 2 are $1 \to 2 \to 3$ and $2 \to 3 \to 4$, so $(A^2)_{13} = (A^2)_{24} = 1$. The only path of length 3 is $1 \to 2 \to 3 \to 4$. $A^4 = 0$: a DAG's adjacency matrix is nilpotent. Common mistakes: $A$ itself for (a), $(A^\top)^2$ (paths reversed), and the zero matrix for (b), which is $A^4$.

</details>

**Problem 8.** The autograd graph of $y = ab + b$ has nodes $a$, $b$, $t = ab$, and $y = t + b$. Backward must visit a node only after every node that consumes it. Which order is valid for backward? `[choice]`
(a) $y, t, a, b$ (b) $a, b, t, y$ (c) $t, y, a, b$ (d) $b, y, t, a$

<details><summary>Answer</summary>

**(a).** Backward visits a node only after all of its consumers. $y$ consumes $t$ and $b$; $t$ consumes $a$ and $b$. So $y$ first, then $t$; $b$ must come after both $y$ and $t$. Order (a), $y, t, a, b$, satisfies this. Common mistakes: (b) is the forward order; (d) visits $b$ before its consumer $t$, so $b$'s gradient would be passed on before $t$'s contribution is added; (c) visits $t$ before its consumer $y$.

</details>

**Problem 9.** A recursive depth-first search that starts at the last node of a chain of $10^5$ nodes and recurses into each node's input makes how many nested calls at its deepest point, counting the first call? (Python's default recursion limit is 1000.) `[number]`

<details><summary>Answer</summary>

**$100{,}000$.** The search starts at the last node and recurses into its input, one call per node down the chain, so the deepest point has all $10^5$ calls on the stack, 100 times Python's limit. This is why the topological sort in [Graphs, DAGs, and an iterative topological sort](01-graphs-dags-and-topological-sort.md) uses an explicit stack. Common mistakes: 1000 is the limit, not the depth; 17 $\approx \log_2 10^5$ would be the depth of a balanced tree, not a chain.

</details>

## Problems on modular arithmetic and hashing

The next eight problems move from remainders and inverses to the word-level operations of PCG32 and FNV-1a, and finish with hash-table load and collisions. Here $a \bmod m$ is the remainder in $\{0, 1, \dots, m - 1\}$, also for negative $a$.

**Problem 10.** (a) $17 \bmod 5$. (b) $-17 \bmod 5$. `[number]`

<details><summary>Answer</summary>

**(a) $2$, (b) $3$.** (a) $17 = 3 \cdot 5 + 2$. (b) $-17 = -4 \cdot 5 + 3$, and the remainder must be in $\{0, \dots, 4\}$. A common mistake is $-2$, which is C's truncating `%` ($-17 = -3 \cdot 5 - 2$); Python's `%` gives 3. Answering 2 for (b) drops the sign.

</details>

**Problem 11.** Give the multiplicative inverse of 3 modulo 11: the $x \in \{0, \dots, 10\}$ with $3x \bmod 11 = 1$. `[number]`

<details><summary>Answer</summary>

**$4$.** $3 \cdot 4 = 12 = 11 + 1$. The inverse exists because $\gcd(3, 11) = 1$. A common mistake is $1/3$, the rational inverse, which is not an element of $\{0, \dots, 10\}$.

</details>

**Problem 12.** A `uint32_t` adds modulo $2^{32}$. What is $(4294967295 + 2) \bmod 2^{32}$? `[number]`

<details><summary>Answer</summary>

**$1$.** $4294967295 = 2^{32} - 1$, so the sum is $2^{32} + 1 \equiv 1$. A common mistake is 4294967297, which does not wrap: that is what a Python port of PCG32 produces without `& 0xFFFFFFFF`.

</details>

**Problem 13.** PCG32's output step rotates a 32-bit word right. Rotating $x$ right by $r$ bits moves bit $i$ to bit $(i - r) \bmod 32$. Give the rotation of $x = 2147483649$ (bits 31 and 0 set) right by 1 bit, as a decimal number. `[number]`

<details><summary>Answer</summary>

**$3{,}221{,}225{,}472$.** Bit 31 moves to bit 30 and bit 0 moves to bit $(0 - 1) \bmod 32 = 31$, so the result has bits 31 and 30 set: $2^{31} + 2^{30}$. This rotation is the last step of PCG32's XSH-RR output ([`pcg32.md`](../../../../course/contracts/spec/pcg32.md)). Common mistakes: $1{,}073{,}741{,}824 = 2^{30}$ is a logical shift, which loses the low bit; 3 rotates left.

</details>

**Problem 14.** FNV-1a 64 starts at $h = 14695981039346656037$ and, for each byte $c$, sets $h \leftarrow ((h \oplus c) \cdot 1099511628211) \bmod 2^{64}$, where $\oplus$ is bitwise XOR. Give the FNV-1a 64 hash of the one-byte string `a` (byte 97), as a decimal number. `[number]`

<details><summary>Answer</summary>

**$12{,}638{,}187{,}200{,}555{,}641{,}996$.** XOR first: the offset basis is `0xcbf29ce484222325`, and $\texttt{0x25} \oplus \texttt{0x61} = \texttt{0x44}$, so $h \oplus 97 = 14695981039346656068$. Then multiply by the prime and reduce modulo $2^{64}$: $(14695981039346656068 \cdot 1099511628211) \bmod 2^{64} = 12638187200555641996$ (`0xaf63dc4c8601ec8c`). A common mistake is 12638153115695167422, the FNV-1 value (multiply, then XOR): the KV block hashes of the [Paged KV block pool](../../ai-engineering/21-inference/08-paged-kv-block-pool.md) would then disagree with every other implementation.

</details>

**Problem 15.** A hash table with chaining has $m = 16$ buckets and holds $n = 12$ keys. (a) Give its load factor $\alpha = n/m$. (b) Under simple uniform hashing, an unsuccessful search hashes once and then scans a whole chain; its expected cost is $1 + \alpha$ probes. Give it. `[number]`

<details><summary>Answer</summary>

**(a) $3/4$, (b) $7/4$.** (a) $\alpha = 12/16$. (b) $1 + \alpha = 7/4$: one probe for the hash and on average $\alpha$ for the chain. Common mistakes: $4/3 = m/n$ for (a), and $3/4$ for (b), which forgets the hash computation itself.

</details>

**Problem 16.** For the universal hash $h(x) = ((ax + b) \bmod p) \bmod m$ with $p = 17$, $a = 3$, $b = 5$, $m = 8$, give $h(10)$. `[number]`

<details><summary>Answer</summary>

**$1$.** $3 \cdot 10 + 5 = 35$, $35 \bmod 17 = 1$, $1 \bmod 8 = 1$. A common mistake is 3, which is $35 \bmod 8$ and skips the reduction modulo $p$, so the family is no longer universal.

</details>

**Problem 17.** $n$ keys are hashed independently and uniformly into $m$ buckets. What is the expected number of pairs of keys that share a bucket? `[expr in n, m]`

<details><summary>Answer</summary>

**$\frac{n(n - 1)}{2m}$.** Let $X_{ij}$ be 1 when keys $i < j$ share a bucket; $\Pr[X_{ij} = 1] = 1/m$. By linearity of expectation, $\mathbb{E}\left[\sum_{i<j} X_{ij}\right] = \binom{n}{2} \cdot \frac{1}{m}$, with no independence needed between pairs. Common mistakes: $n/m$ is the load factor, and $n^2/m$ counts ordered pairs including each key with itself.

</details>

## Proofs

The last six problems are proofs, plus one closing check. Problems 18 and 19 together prove that a topological order exists exactly when the graph is acyclic; Problem 20 proves the iterative DFS correct; Problems 21 to 23 justify reducing after every step and the odd multiplier of an LCG.

**Problem 18.** Prove that every finite, nonempty directed acyclic graph has a node with in-degree 0. `[proof]`

<details><summary>Answer</summary>

Claim: every finite, nonempty directed acyclic graph $G$ (a DAG) has a node with in-degree 0.

Method: contradiction, with the pigeonhole principle.

Let $G$ have $n \ge 1$ nodes and suppose every node has in-degree at least 1, so every node has at least one input. Start at any node $v_0$, pick an input $v_1$ of it (an edge $v_1 \to v_0$), then an input $v_2$ of $v_1$, and so on: each step is possible because every node has an input. After $n$ steps we have $n + 1$ nodes $v_0, v_1, \dots, v_n$ with edges $v_{i+1} \to v_i$.

There are only $n$ distinct nodes, so by pigeonhole $v_i = v_j$ for some $i < j$. Then $v_j \to v_{j-1} \to \dots \to v_i = v_j$ is a directed cycle, contradicting that $G$ is acyclic.

So some node has in-degree 0. (Kahn's algorithm relies on this at every step: removing a source from a DAG leaves a DAG, which again has a source.)

Model proof also in [`q18.md`](../../../../course/solutions/solve/S-M06a/q18.md).

</details>

**Problem 19.** Prove that a directed graph that has a topological order has no directed cycle. `[proof]`

<details><summary>Answer</summary>

Claim: if a directed graph $G$ has a topological order, then $G$ has no directed cycle.

Definitions: a topological order is a listing of all nodes, with position $\mathrm{pos}(v)$, such that every edge $u \to v$ has $\mathrm{pos}(u) < \mathrm{pos}(v)$. A directed cycle is a sequence of edges $v_1 \to v_2 \to \dots \to v_k \to v_1$ with $k \ge 1$.

Method: contradiction.

Suppose $G$ has a topological order and a directed cycle $v_1 \to v_2 \to \dots \to v_k \to v_1$. Applying the order property to each edge of the cycle gives $\mathrm{pos}(v_1) < \mathrm{pos}(v_2) < \dots < \mathrm{pos}(v_k) < \mathrm{pos}(v_1)$, and by transitivity of $<$, $\mathrm{pos}(v_1) < \mathrm{pos}(v_1)$, which is false.

So $G$ has no directed cycle. (With Problem 18 the converse also holds: repeatedly removing a node of in-degree 0 lists any DAG in topological order.)

Model proof also in [`q19.md`](../../../../course/solutions/solve/S-M06a/q19.md).

</details>

**Problem 20.** The section [Graphs, DAGs, and an iterative topological sort](01-graphs-dags-and-topological-sort.md) orders the autograd graph with an iterative depth-first search. Each node $v$ has a finite list of inputs, `inputs(v)`, and the graph reachable from the root is acyclic:

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

**Problem 21.** Prove that $(a + b) \bmod m = ((a \bmod m) + (b \bmod m)) \bmod m$ for all integers $a, b$ and every integer $m \ge 1$. `[proof]`

<details><summary>Answer</summary>

Claim: for all integers $a, b$ and every integer $m \ge 1$, $(a + b) \bmod m = ((a \bmod m) + (b \bmod m)) \bmod m$.

Definitions: by division with remainder, every integer $x$ is $x = qm + r$ for unique integers $q$ and $r$ with $0 \le r < m$; $x \bmod m$ is that $r$. Uniqueness implies: if $x - y$ is a multiple of $m$, then $x \bmod m = y \bmod m$ (writing $y = qm + r$ gives $x = (q + k)m + r$ with the same $r$).

Method: direct.

Write $a = q_1 m + r_1$ and $b = q_2 m + r_2$ with $r_1 = a \bmod m$ and $r_2 = b \bmod m$. Then $(a + b) - (r_1 + r_2) = (q_1 + q_2) m$, a multiple of $m$. By the uniqueness remark, $(a + b) \bmod m = (r_1 + r_2) \bmod m$, which is the claim. This is why a hash or an RNG may reduce after every step instead of once at the end, and why a `uint64_t` that wraps computes the right value modulo $2^{64}$.

Model proof also in [`q21.md`](../../../../course/solutions/solve/S-M06a/q21.md).

</details>

**Problem 22.** Prove that if $\gcd(a, m) = 1$, then $x \mapsto ax \bmod m$ is a bijection of $\{0, 1, \dots, m - 1\}$. (This is why the LCG step $x \mapsto (ax + c) \bmod 2^{64}$ inside PCG32 never merges two states: its multiplier $a$ is odd.) `[proof]`

<details><summary>Answer</summary>

Claim: if $\gcd(a, m) = 1$, then $f(x) = ax \bmod m$ is a bijection of $Z_m = \{0, 1, \dots, m - 1\}$.

Method: show $f$ is injective; an injective map from a finite set to itself is a bijection, because its image has as many elements as the set itself.

Suppose $f(x) = f(y)$ for $x, y \in Z_m$. Then $ax$ and $ay$ have the same remainder, so $m$ divides $ax - ay = a(x - y)$.

Since $\gcd(a, m) = 1$, Bezout's identity gives integers $s, t$ with $sa + tm = 1$. Multiply by $(x - y)$: $x - y = s \cdot a(x - y) + t m (x - y)$. Both terms on the right are multiples of $m$, so $m$ divides $x - y$ (this is Euclid's lemma).

But $x, y \in \{0, \dots, m - 1\}$, so $\lvert x - y \rvert < m$, and the only multiple of $m$ in that range is 0. So $x = y$, $f$ is injective, and therefore a bijection of $Z_m$.

For $m = 2^{64}$ this says any odd multiplier permutes the 64-bit states: PCG32's LCG step never sends two states to one. Problem 23 shows what goes wrong when $\gcd(a, m) > 1$.

Model proof also in [`q22.md`](../../../../course/solutions/solve/S-M06a/q22.md).

</details>

**Problem 23.** Is $x \mapsto 6x \bmod 16$ a bijection of $\{0, 1, \dots, 15\}$? `[bool]`

<details><summary>Answer</summary>

**False.** $\gcd(6, 16) = 2$, so every image $6x \bmod 16$ is even: only the 8 values $0, 2, \dots, 14$ are reached, and for example $6 \cdot 0 \equiv 6 \cdot 8 \equiv 0 \pmod{16}$. Two inputs share each image. This is the hypothesis of Problem 22 failing, and why an LCG's multiplier must be odd.

</details>

## What can go wrong

The problems were chosen so that each common mistake shows up as a wrong answer. The table collects those mistakes, what they look like in code, and the habit that avoids them.

| Pitfall | Symptom | How to avoid it |
|---|---|---|
| Testing for one specific topological order | a correct `toposort` fails on a valid order | check that every edge goes forward instead of comparing with one list (Problems 2, 3) |
| Swapping in-degree and out-degree | Kahn's algorithm starts from sinks and emits nothing | count edges *into* each node (Problem 4) |
| Backward in forward order, or a shared node before one of its consumers | a gradient used before every consumer added to it | visit a node only after all of its consumers: reverse topological order (Problem 8) |
| Recursing over a long chain | `RecursionError` at depth 1000 on a long sequence | use an explicit stack, whose depth is bounded only by memory (Problems 9, 20) |
| Using C's truncating `%` for a negative operand | a negative bucket index | take the remainder in $\{0, \dots, m - 1\}$, as Python's `%` does (Problem 10) |
| Forgetting to wrap at the word size | a Python port of PCG32 diverges from C after one step | reduce modulo $2^w$ after every operation (Problems 12, 21) |
| Shifting instead of rotating | PCG32 output loses high bits | OR the shifted-out bits back in at the top (Problem 13) |
| Multiplying before XOR in FNV-1a | block hashes that disagree with every other implementation | XOR the byte in first, then multiply (Problem 14) |
| A multiplier sharing a factor with the modulus | the generator or hash collapses onto a fraction of its states | check $\gcd(a, m) = 1$; for $m = 2^w$, use an odd multiplier (Problems 22, 23) |

## Where this leads

Looking back, Problems 18 to 22 rest on the induction, pigeonhole, and injective-and-surjective arguments of the [Discrete Math 1 problem set](../11-discrete-math-1/03-problem-set/README.md). Looking forward, [Graphs, DAGs, and an iterative topological sort](01-graphs-dags-and-topological-sort.md) implements `toposort` for backward, iterative so that a $10^5$-deep chain works, and Problem 20 is its correctness proof. [Modular arithmetic, hashing, and PCG32 in Python](03-modular-arithmetic-hashing-and-pcg32.md) implements PCG32, SplitMix64, FNV-1a, and `universal_hash`. In *AI Engineering*, [Tensor, broadcasting backward, and no_grad](../../ai-engineering/06-numerics-and-autograd/02-tensor-and-broadcasting-backward.md) walks the topological order in reverse and sums gradients over consumers, and the [Paged KV block pool](../../ai-engineering/21-inference/08-paged-kv-block-pool.md) names KV blocks by chained FNV-1a 64 hashes. [Part b of the problem set](05-problem-set-b.md) builds trees, birthday bounds, MinHash, and Bloom filters on Problems 15 to 17.

### Further reading

- Hammack, *Book of Proof*, chapters 11 and 12.
- Cormen et al., *Introduction to Algorithms*, chapter 11 (hashing) and section 22.4 (topological sort).
- O'Neill, "PCG: A Family of Simple Fast Space-Efficient Statistically Good Algorithms" (2014), sections 4 and 6.

## Key terms

- **DAG**: a directed graph with no directed cycle; every computation graph is one.
- **In-degree, out-degree**: the number of edges into and out of a node.
- **Topological order**: a listing of the nodes in which every edge goes forward.
- **Kahn's algorithm**: repeatedly output a node of in-degree 0 and decrement its successors' in-degrees.
- **Congruence modulo $m$**: $a \equiv b$ when $m$ divides $a - b$.
- **Multiplicative inverse**: an $x$ with $ax \equiv 1 \pmod m$, which exists exactly when $\gcd(a, m) = 1$.
- **Rotation**: a shift that moves the bits falling off one end back in at the other.
- **Load factor**: $\alpha = n/m$, the average chain length of a hash table.

## Summary

- A computation graph is a DAG; a topological order lists every node after its inputs, it is rarely unique, and backward walks it in reverse (Problems 2, 3, 8).
- Recursion depth equals the longest input chain, so a recursive traversal of a $10^5$-node chain dies at Python's limit of 1000; the iterative DFS of Problem 20 does not (Problem 9).
- Fixed-width integers are arithmetic modulo $2^w$: addition wraps, rotation moves bits around the word, and $x \mapsto ax \bmod 2^w$ permutes the words exactly when $a$ is odd (Problems 12, 13, 22, 23).
- FNV-1a XORs a byte in, then multiplies; swapping the two steps gives FNV-1, a different hash (Problem 14).
- With $n$ keys in $m$ buckets the expected number of colliding pairs is $\binom{n}{2}/m$, which is why collisions appear long before the table is full (Problem 17).
- Model proofs for Problems 18 to 22 are also in [`course/solutions/solve/S-M06a/`](../../../../course/solutions/solve/S-M06a/).
