# S-M06a problems: graphs and DAGs, modular arithmetic and hashing, proofs

Answer every question in `solve/S-M06a.toml` (written by `ol start S-M06a`).
The tag after each question is its answer type: `[number]` is an exact value,
`[expr]` a formula in the named variables, `[vector]` a list `[0, 0, 2]`,
`[matrix]` a list of rows, `[bool]` is `true` or `false`, `[choice]` one
letter, and `[proof]` a file `solve/S-M06a/qN.md` graded against its rubric.

Throughout, $G_1$ is the directed graph on nodes $\{1, 2, 3, 4, 5\}$ with edges
$1 \to 3$, $2 \to 3$, $3 \to 5$, $2 \to 4$, $4 \to 5$. An edge $u \to v$ means
"$u$ is an input of $v$", so $u$ must be computed before $v$.

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
