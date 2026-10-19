# Discrete Math 2

This chapter covers the discrete structures that turn up most directly in algorithms and systems: graphs, number theory, and advanced counting. Graph theory is the language of networks, dependencies, and state machines, and it is the single most widely used discrete structure in computer science. Number theory underpins cryptography, hashing, and random number generation. Recurrence relations are the bridge between recursive algorithms and closed-form running times. Together these topics form the mathematical core of a computer science education.

The chapter moves from structure to arithmetic to probability. It opens with directed graphs and the topological order that every backward pass in an autograd engine depends on, built as an iterative depth-first search. Trees and tries follow, the structure behind every tokenizer's "which vocabulary entries start here?" question. Then modular arithmetic becomes concrete in PCG32, the random generator the rest of the books draw from, and in the FNV-1a and universal hash functions. Two problem sets close the chapter: the first drills DAGs, modular arithmetic, and hashing by hand; the second derives the numbers you must choose for tries, hash tables, MinHash and LSH, Bloom filters, and divide-and-conquer recurrences.

> **Learning objectives**
> - Define graphs, DAGs, trees, and tries, prove their basic counting facts, and implement an iterative topological sort and a trie.
> - Work fluently in $\mathbb{Z}_n$ and on fixed-width machine words, and explain why PCG32, FNV-1a, and universal hashing are built the way they are.
> - Predict collision rates with birthday bounds, size a Bloom filter, and read an LSH S-curve.
> - Solve linear recurrences and divide-and-conquer recurrences, and use inclusion-exclusion and combinatorial proofs.

## Before you begin

You need the sets, relations, counting, induction, and bijections of [Discrete Math 1](../05-discrete-math-1/README.md), and its [problem set](../05-discrete-math-1/90-problem-set.md) is good preparation for the proofs here. The companion textbook is *Discrete Mathematics: An Open Introduction* by Oscar Levin (https://discrete.openmathbooks.org, CC BY-SA 4.0). Levin's text is conversational and example-rich; draw every graph by hand, and check any counting formula on small cases before trusting it.

## In this chapter

1. [Graphs, DAGs, and an iterative topological sort](01-graphs-dags-and-topological-sort.md) proves that a topological order exists exactly for DAGs and builds `toposort`, the reverse depth-first post-order that autograd's backward pass walks, iterative and keyed by identity.
2. [Trees and tries, longest-prefix match](02-trees-and-tries-longest-prefix-match.md) counts the nodes of a trie and builds longest-prefix match, the lookup behind WordPiece, the Unigram lattice, and added-token matching.
3. [Modular arithmetic, hashing, and PCG32 in Python](03-modular-arithmetic-hashing-and-pcg32.md) derives PCG32 from a linear congruential generator, with bit-exact uniforms, unbiased bounded integers, SplitMix64 sub-streams, FNV-1a, and universal hashing.
4. [Discrete math 2 problem set, part a: DAGs, modular arithmetic, hashing](90-problem-set-a.md) works topological orders, machine-word arithmetic, and hash-table load by hand, including the correctness proof of the iterative DFS.
5. [Discrete math 2 problem set, part b: trees, birthday bounds, MinHash and LSH, Bloom filters, recurrences](91-problem-set-b.md) derives the formulas that set the parameters of tries, hash tables, near-duplicate detection, Bloom filters, and recursive algorithms.

## The mathematics at a glance

The sections above apply the theory to specific code. This survey collects the classical definitions and theorems of the chapter in one place, following Levin, so you can see the whole field the sections draw on.

### Graph theory

A **graph** is $G = (V, E)$, a set of vertices $V$ and a set of edges $E$ (pairs of vertices). The **degree** $\deg(v)$ is the number of edges incident to $v$. A **path** is a sequence of distinct vertices in which consecutive vertices are adjacent, and a **cycle** is a path that starts and ends at the same vertex. A graph is **connected** when there is a path between every pair of vertices. An **isomorphism** is a bijection $f: V(G) \to V(H)$ that preserves adjacency. A **planar graph** can be drawn in the plane without edge crossings. The **chromatic number** $\chi(G)$ is the minimum number of colours needed so that no two adjacent vertices share a colour. An **Euler path** uses every edge exactly once (an Euler circuit is a closed one); a **Hamilton path** visits every vertex exactly once (a Hamilton circuit is a closed one). A **tree** is a connected acyclic graph, and a **bipartite graph** is one whose vertices split into two sets so that every edge goes between the sets. Levin covers all of this in chapter 4, sections 4.1 to 4.7.

The key theorems each have a short intuition:

- **Handshaking lemma.** The sum of all degrees is $2\lvert E \rvert$, because each edge contributes 1 to the degree of each endpoint and so is counted twice.
- **Euler's formula for planar graphs.** A connected planar graph has $V - E + F = 2$, where $F$ is the number of faces. This topological invariant constrains planar graphs; its corollary $E \le 3V - 6$ shows that $K_5$ and $K_{3,3}$ are not planar.
- **Euler path and circuit criterion.** A connected graph has an Euler circuit if and only if every vertex has even degree, and an Euler path (not a circuit) if and only if exactly two vertices have odd degree. To traverse every edge, you must be able to leave every vertex as many times as you enter it.
- **Tree characterizations.** For a graph $G$ on $n$ vertices the following are equivalent: $G$ is a tree; $G$ is connected with $n - 1$ edges; $G$ is acyclic with $n - 1$ edges; there is a unique path between any two vertices. Trees are minimally connected: removing any edge disconnects them.
- **Four colour theorem.** Every planar graph can be properly coloured with at most 4 colours, so any map can be coloured with 4 colours without adjacent regions sharing one. The proof requires computer verification.
- **Five colour theorem.** Every planar graph can be properly coloured with 5 colours. This one is provable without a computer: Euler's formula gives a vertex of degree at most 5, and induction does the rest.

Three worked examples show the theorems in use. Does $K_4$, the complete graph on 4 vertices, have an Euler circuit? Every vertex has degree 3, which is odd, so no Euler circuit exists; and since more than two vertices (all four) have odd degree, there is no Euler path either.

Is $K_{2,3}$ planar? $V = 5$ and $E = 6$, and $E \le 3V - 6 = 9$ passes. But $K_{2,3}$ is bipartite, so it has no triangles, which gives the stricter bound $E \le 2V - 4 = 6$. It passes with equality, so the formula alone does not rule it out, and in fact $K_{2,3}$ is planar (draw it).

Finally, prove that every tree with $n \ge 2$ vertices has at least two leaves (vertices of degree 1). A tree has $n - 1$ edges, so the sum of degrees is $2(n - 1)$. If at most one vertex had degree 1, the remaining $n - 1$ vertices would have degree at least 2, giving a sum of at least $2(n - 1) + 1 > 2(n - 1)$, a contradiction.

### Number theory

Number theory supplies the arithmetic. $a \mid b$, read "$a$ divides $b$", means $b = ak$ for some integer $k$. The **greatest common divisor** $\gcd(a, b)$ is the largest $d$ with $d \mid a$ and $d \mid b$. $a$ is **congruent** to $b$ modulo $n$ when $n \mid (a - b)$, and **modular arithmetic** is arithmetic in $\mathbb{Z}_n = \{0, 1, \dots, n-1\}$ with operations taken mod $n$. A **multiplicative inverse** $a^{-1} \bmod n$ exists if and only if $\gcd(a, n) = 1$. Levin treats this in supplementary material; Hammack's *Book of Proof*, chapters 1 to 7, applies proof technique to number theory.

The key theorems:

- **Division algorithm.** For any integers $a$ and $d > 0$ there are unique $q$ and $r$ with $a = dq + r$ and $0 \le r < d$: every integer lands in a unique "box" relative to $d$.
- **Euclidean algorithm.** $\gcd(a, b) = \gcd(b, a \bmod b)$, terminating when the remainder is 0. The GCD does not change when you subtract one number from the other, so taking the remainder does many subtractions at once.
- **Bezout's identity.** $\gcd(a, b) = ax + by$ for some integers $x, y$, found by the extended Euclidean algorithm: the GCD can always be built as a linear combination of $a$ and $b$.
- **Fermat's little theorem.** If $p$ is prime and $p \nmid a$, then $a^{p-1} \equiv 1 \pmod p$: with a prime modulus, exponentiation is cyclic with period dividing $p - 1$.
- **RSA.** Choose primes $p, q$. Let $n = pq$ and $\varphi(n) = (p-1)(q-1)$. Choose $e$ with $\gcd(e, \varphi(n)) = 1$ and compute $d = e^{-1} \bmod \varphi(n)$. The public key is $(n, e)$ and the private key is $d$. Encrypt with $c = m^e \bmod n$ and decrypt with $m = c^d \bmod n$. Euler's theorem guarantees $m^{ed} \equiv m \pmod n$, so encryption and decryption are inverses.

As a worked example, compute $\gcd(252, 105)$ with the Euclidean algorithm: $252 = 2 \cdot 105 + 42$, $105 = 2 \cdot 42 + 21$, $42 = 2 \cdot 21 + 0$, so $\gcd(252, 105) = 21$. The extended version works backwards: $21 = 105 - 2 \cdot 42 = 105 - 2(252 - 2 \cdot 105) = 5 \cdot 105 - 2 \cdot 252$, so $21 = (-2) \cdot 252 + 5 \cdot 105$.

To compute $3^{100} \bmod 7$, use Fermat's little theorem: $3^6 \equiv 1 \pmod 7$, and $100 = 6 \cdot 16 + 4$, so $3^{100} = (3^6)^{16} \cdot 3^4 \equiv 1^{16} \cdot 81 \equiv 4 \pmod 7$.

### Advanced counting

The counting tools extend those of the previous chapter. The **inclusion-exclusion principle** is $\lvert A_1 \cup \dots \cup A_n \rvert = \sum \lvert A_i \rvert - \sum \lvert A_i \cap A_j \rvert + \sum \lvert A_i \cap A_j \cap A_k \rvert - \dots$: alternately add and subtract to correct for overcounting. The **pigeonhole principle** says that if $n + 1$ objects go into $n$ boxes, some box holds at least 2; more generally, if more than $kn$ objects go into $n$ boxes, some box holds at least $k + 1$. A **recurrence relation** defines $a_n$ in terms of earlier terms, as in $a_n = 2a_{n-1} + 1$. For $a_n = c_1 a_{n-1} + c_2 a_{n-2}$ the **characteristic equation** is $r^2 = c_1 r + c_2$, and its roots give the closed form. A **generating function** $A(x) = \sum a_n x^n$ encodes a sequence as a power series. Levin covers these in chapter 1, sections 1.1 to 1.7, and chapter 2, sections 2.1 to 2.4.

The key results:

- **Solving linear recurrences.** If $r^2 - c_1 r - c_2 = 0$ has distinct roots $r_1, r_2$, then $a_n = A r_1^n + B r_2^n$, with constants from the initial conditions; with a repeated root $r$, $a_n = (A + Bn) r^n$. The recurrence behaves like a linear ODE, and its solution is a combination of geometric sequences.
- **Master theorem.** For $T(n) = aT(n/b) + f(n)$, compare $f(n)$ with $n^{\log_b a}$. If $f$ grows slower, $T = \Theta(n^{\log_b a})$; if they match, $T = \Theta(n^{\log_b a} \log n)$; if $f$ grows faster, $T = \Theta(f(n))$. The recursion tree spends most of its work at the leaves, evenly across levels, or at the root.
- **Derangements.** The number of permutations with no fixed point is $D_n = n! \sum_{k=0}^{n} (-1)^k / k!$, approximately $n!/e$. By inclusion-exclusion, subtract the permutations fixing at least one element, add back those fixing at least two, and so on.

Three worked examples. Solve $a_n = 5a_{n-1} - 6a_{n-2}$ with $a_0 = 1$, $a_1 = 4$: the characteristic equation $r^2 - 5r + 6 = 0$ has roots 2 and 3, so $a_n = A \cdot 2^n + B \cdot 3^n$. From $a_0 = 1$, $A + B = 1$; from $a_1 = 4$, $2A + 3B = 4$. Solving gives $B = 2$ and $A = -1$, so $a_n = -2^n + 2 \cdot 3^n$.

How many integers from 1 to 1000 are divisible by 2, 3, or 5? $\lvert A_2 \rvert = 500$, $\lvert A_3 \rvert = 333$, $\lvert A_5 \rvert = 200$, $\lvert A_2 \cap A_3 \rvert = 166$, $\lvert A_2 \cap A_5 \rvert = 100$, $\lvert A_3 \cap A_5 \rvert = 66$, and $\lvert A_2 \cap A_3 \cap A_5 \rvert = 33$. By inclusion-exclusion, $500 + 333 + 200 - 166 - 100 - 66 + 33 = 734$.

And by pigeonhole, in any group of 13 people at least two share a birth month: 13 people, 12 months.

### Combinatorial proofs

The last technique replaces algebra with counting. A **combinatorial proof** proves an identity by showing that both sides count the same set in two different ways; **double counting** is the general principle that two ways of counting the same quantity yield an equality. The method is to identify a set $S$ that one side counts, then find a different way to count $S$ that yields the other side. No algebraic manipulation is needed. Levin covers this in chapter 2, sections 2.1 and 2.2, with supplementary material from Hammack.

Three classical identities fall out this way:

- **Vandermonde's identity.** $\binom{m+n}{r} = \sum_{k=0}^{r} \binom{m}{k}\binom{n}{r-k}$. Choose $r$ people from a group of $m$ men and $n$ women: the left side counts directly, and the right side splits by choosing $k$ men and $r - k$ women.
- **Hockey stick identity.** $\binom{r}{r} + \binom{r+1}{r} + \dots + \binom{n}{r} = \binom{n+1}{r+1}$. To choose $r + 1$ elements from $\{1, \dots, n+1\}$, let the largest chosen element be $j + 1$ (where $j \ge r$), then choose the remaining $r$ from $\{1, \dots, j\}$, giving $\binom{j}{r}$. Summing over $j$ from $r$ to $n$ gives the left side.
- **Row sums of Pascal's triangle.** $\sum_{k=0}^{n} \binom{n}{k} = 2^n$. The left side counts the subsets of an $n$-element set by size; the right side counts all subsets directly, with 2 choices per element.

Two more worked examples. To prove $\binom{n}{k} = \binom{n}{n-k}$: $\binom{n}{k}$ counts the ways to choose $k$ items from $n$, and for each such choice the unchosen items form a set of size $n - k$. That is a bijection between $k$-element and $(n-k)$-element subsets, so the counts agree. To prove $\binom{2n}{2} = 2\binom{n}{2} + n^2$: the left side chooses 2 people from $2n$, with $n$ in group A and $n$ in group B. The right side splits the choice into both from A ($\binom{n}{2}$), both from B ($\binom{n}{2}$), or one from each ($n \cdot n$), for a total of $2\binom{n}{2} + n^2$.

### Techniques and where they apply

The table pairs each technique with when to reach for it.

| Technique | When to use | Key formula or idea |
|-----------|-------------|------------------|
| BFS/DFS on graphs | explore connectivity, find paths | systematic vertex traversal |
| Euler's formula | analyse planar graphs | $V - E + F = 2$ |
| Degree-based arguments | prove existence of certain vertices | handshaking: sum of degrees $= 2\lvert E \rvert$ |
| Euclidean algorithm | compute a GCD efficiently | $\gcd(a,b) = \gcd(b, a \bmod b)$ |
| Modular exponentiation | compute $a^k \bmod n$ efficiently | repeated squaring |
| Characteristic equation | solve linear recurrences | factor $r^k - c_1 r^{k-1} - \dots = 0$ |
| Generating functions | solve counting problems and recurrences | encode a sequence as a power series |
| Inclusion-exclusion | count a union of overlapping sets | alternately add and subtract intersections |
| Pigeonhole principle | prove existence of collisions or repeats | more items than containers |
| Combinatorial proof | prove identities without algebra | count the same set two ways |

These ideas reach far into computing. Graph traversal gives shortest paths, cycle detection, and topological sort; graph colouring drives register allocation in compilers and scheduling; planar graphs appear in VLSI circuit layout and map rendering. Trees are the data structures (binary search trees, heaps, tries) and the spanning trees of networks. Euler and Hamilton paths show up in network routing and in DNA sequencing by Eulerian assembly. Modular arithmetic is the basis of hashing, checksums, and random number generators, and RSA and number theory underpin public-key cryptography and digital signatures. Recurrences give the running times of merge sort and quicksort, the master theorem classifies divide-and-conquer algorithms at a glance, and generating functions power Knuth-style analysis of algorithms. Many of these are developed in *Algorithms* and *Systems and Architecture*.

## Further reading

- Oscar Levin, *Discrete Mathematics: An Open Introduction* (https://discrete.openmathbooks.org). Essential exercises: chapter 4, exercises 4.1.1 to 4.1.8, 4.2.1 to 4.2.6, 4.3.1 to 4.3.8, and 4.4.1 to 4.4.5 (graphs); chapter 1, exercises 1.1.1 to 1.1.8, 1.2.1 to 1.2.6, 1.3.1 to 1.3.8, and 1.4.1 to 1.4.8; chapter 2, exercises 2.4.1 to 2.4.8 (counting and recurrences). Challenge exercises: 4.7.1 to 4.7.6, 1.7.1 to 1.7.6, 2.4.9 to 2.4.14, and the combinatorial proofs 2.1.6 to 2.1.12.
- Richard Hammack, *Book of Proof*: chapters 4 to 7 review exercises, and exercises 1 to 8 of the number-theory material in chapters 4 to 6, with chapter 6's advanced exercises as a challenge.
- As a challenge project, implement RSA by hand for small primes, such as $p = 5$ and $q = 11$.
