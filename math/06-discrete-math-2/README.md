# Discrete Math 2

## Overview
- **Textbook**: *Discrete Mathematics: An Open Introduction* by Oscar Levin -- https://discrete.openmathbooks.org (CC BY-SA 4.0)
- **Prerequisites**: [Discrete Math 1](../05-discrete-math-1/)
- **Estimated time**: 4 weeks at 10-12 hrs/week

## Key Takeaways
- Master graph theory -- the single most widely used discrete structure in computer science
- Understand modular arithmetic and the number-theoretic foundations of cryptography
- Solve recurrence relations that describe the running time of recursive algorithms
- Use advanced counting techniques (inclusion-exclusion, generating functions) to analyze complex problems
- Develop comfort with combinatorial proofs as an elegant alternative to algebraic manipulation

## How to Study
- Read each section and work through inline examples; Levin's text is conversational and example-rich
- Draw graphs by hand for every graph theory problem -- visualization is essential
- For recurrences, practice both the characteristic equation method and the Master Theorem
- Verify counting results with small cases before trusting a formula
- Connect graph theory to real algorithms (BFS, DFS, Dijkstra) early for motivation

---

# Concepts & Techniques

## Core Insight

Discrete Math 2 covers the structures that appear most directly in algorithms and systems:
graphs, number theory, and advanced counting. Graph theory provides the language for networks,
dependencies, and state machines. Number theory underpins cryptography and hashing. Recurrence
relations are the bridge between recursive algorithms and closed-form running times. Together,
these topics form the mathematical core of a CS education.

## 1. Graph Theory

**Textbook sections**: Ch 4, Sections 4.1-4.7

**Key definitions**:
- **Graph**: G = (V, E) where V is a set of vertices and E is a set of edges (pairs of vertices)
- **Degree**: deg(v) = number of edges incident to v
- **Path**: A sequence of distinct vertices where consecutive vertices are adjacent
- **Cycle**: A path that starts and ends at the same vertex
- **Connected**: There is a path between every pair of vertices
- **Isomorphism**: A bijection f: V(G) -> V(H) that preserves adjacency
- **Planar graph**: A graph that can be drawn in the plane without edge crossings
- **Chromatic number**: chi(G) = minimum number of colors needed so no two adjacent vertices share a color
- **Euler path/circuit**: A walk that uses every edge exactly once (circuit = closed)
- **Hamilton path/circuit**: A path that visits every vertex exactly once (circuit = closed)
- **Tree**: A connected acyclic graph
- **Bipartite graph**: Vertices can be split into two sets so every edge goes between the sets

**Key theorems**:
- **Handshaking Lemma**: Sum of all degrees = 2|E|. *Intuition*: each edge contributes 1 to the degree of each endpoint, so it is counted twice.
- **Euler's Formula (Planar Graphs)**: For a connected planar graph, V - E + F = 2 where F is the number of faces. *Intuition*: this topological invariant constrains planar graphs. Corollary: E <= 3V - 6 (so K_5 and K_{3,3} are not planar).
- **Euler Path/Circuit Criterion**: A connected graph has an Euler circuit iff every vertex has even degree. It has an Euler path (not circuit) iff exactly two vertices have odd degree. *Intuition*: to traverse every edge, you must be able to "leave" every vertex as many times as you "enter" it.
- **Tree Characterizations**: For a graph G with n vertices, the following are equivalent: G is a tree; G is connected with n-1 edges; G is acyclic with n-1 edges; there is a unique path between any two vertices. *Intuition*: trees are "minimally connected" -- removing any edge disconnects them.
- **Four Color Theorem**: Every planar graph can be properly colored with at most 4 colors. *Intuition*: any map can be colored with 4 colors so no adjacent regions share a color. (Proof requires computer verification.)
- **Five Color Theorem**: Every planar graph can be properly colored with 5 colors. *Intuition*: provable without computer; uses Euler's formula to find a vertex of degree <= 5, then induction.

**Worked example**:
> Does K_4 (complete graph on 4 vertices) have an Euler circuit? Degrees: every vertex has degree 3 (odd). Since not all degrees are even, no Euler circuit exists. Does it have an Euler path? More than two vertices have odd degree (all four do), so no Euler path either.

> Is K_{2,3} planar? V = 5, E = 6. Check: E <= 3V - 6 = 9. Passes. But K_{2,3} is bipartite, so no triangles, giving the stricter bound E <= 2V - 4 = 6. It passes with equality, so the formula alone does not rule it out. In fact K_{2,3} is planar (draw it).

> Prove every tree with n >= 2 vertices has at least two leaves (vertices of degree 1). A tree has n-1 edges, so the sum of degrees is 2(n-1). If at most one vertex had degree 1, the remaining n-1 vertices would have degree >= 2, giving sum >= 2(n-1) + 1 > 2(n-1), contradiction.

**Essential problems**: Levin Ch 4, Exercises: #4.1.1-4.1.8, #4.2.1-4.2.6, #4.3.1-4.3.8, #4.4.1-4.4.5
**Challenge problems**: Levin Ch 4, Exercises: #4.7.1-4.7.6

## 2. Number Theory

**Textbook sections**: Levin supplementary; also Hammack Ch 1-7 for proof technique applied to number theory

**Key definitions**:
- **Divisibility**: a | b means b = ak for some integer k
- **Greatest Common Divisor**: gcd(a,b) = largest d such that d | a and d | b
- **Congruence**: a is congruent to b (mod n) means n | (a - b)
- **Modular arithmetic**: Arithmetic in Z_n = {0, 1, ..., n-1} with operations mod n
- **Multiplicative inverse mod n**: a^(-1) mod n exists iff gcd(a,n) = 1

**Key theorems**:
- **Division Algorithm**: For any integers a and d > 0, there exist unique q and r with a = dq + r and 0 <= r < d. *Intuition*: every integer can be placed in a unique "box" relative to d.
- **Euclidean Algorithm**: gcd(a,b) = gcd(b, a mod b), terminating when the remainder is 0. *Intuition*: the GCD does not change when you subtract one number from the other -- so take the remainder instead for speed.
- **Bezout's Identity**: gcd(a,b) = ax + by for some integers x, y (found by the Extended Euclidean Algorithm). *Intuition*: the GCD can always be "built" as a linear combination of a and b.
- **Fermat's Little Theorem**: If p is prime and p does not divide a, then a^(p-1) is congruent to 1 (mod p). *Intuition*: in modular arithmetic with a prime modulus, exponentiation is cyclic with period dividing p-1.
- **RSA**: Choose primes p, q. Let n = pq, phi(n) = (p-1)(q-1). Choose e with gcd(e, phi(n)) = 1. Compute d = e^(-1) mod phi(n). Public key: (n,e). Private key: d. Encrypt: c = m^e mod n. Decrypt: m = c^d mod n. *Intuition*: Euler's theorem guarantees m^(ed) = m mod n, so encryption and decryption are inverses.

**Worked example**:
> Compute gcd(252, 105) using the Euclidean Algorithm. 252 = 2*105 + 42. 105 = 2*42 + 21. 42 = 2*21 + 0. So gcd(252,105) = 21.

> Extended: work backwards. 21 = 105 - 2*42 = 105 - 2*(252 - 2*105) = 5*105 - 2*252. So gcd = 21 = (-2)*252 + 5*105.

> Compute 3^100 mod 7. By Fermat's Little Theorem: 3^6 = 1 mod 7. 100 = 6*16 + 4. So 3^100 = (3^6)^16 * 3^4 = 1^16 * 81 = 81 mod 7 = 4.

**Essential problems**: Levin supplementary exercises on modular arithmetic; Hammack Ch 4-6 number theory exercises: #1-8 from each
**Challenge problems**: Implement RSA by hand for small primes (e.g., p=5, q=11); Hammack Ch 6 advanced exercises

## 3. Advanced Counting

**Textbook sections**: Ch 1, Sections 1.1-1.7; Ch 2, Sections 2.1-2.4

**Key definitions**:
- **Inclusion-Exclusion Principle**: |A_1 union ... union A_n| = Sum |A_i| - Sum |A_i intersect A_j| + Sum |A_i intersect A_j intersect A_k| - ... *Intuition*: alternately add and subtract to correct for overcounting.
- **Pigeonhole Principle**: If n+1 objects are placed in n boxes, at least one box has >= 2 objects. Generalized: if more than kn objects go into n boxes, some box has >= k+1 objects. *Intuition*: you cannot spread more items than slots without doubling up.
- **Recurrence relation**: An equation defining a_n in terms of earlier terms, e.g., a_n = 2*a_(n-1) + 1
- **Characteristic equation**: For a_n = c_1*a_(n-1) + c_2*a_(n-2), solve r^2 = c_1*r + c_2; roots give the closed form
- **Generating function**: A(x) = Sum a_n * x^n; encodes a sequence as a power series

**Key theorems**:
- **Solving Linear Recurrences**: If the characteristic equation r^2 - c_1*r - c_2 = 0 has distinct roots r_1, r_2, then a_n = A*r_1^n + B*r_2^n (constants from initial conditions). If repeated root r, then a_n = (A + Bn)*r^n. *Intuition*: the recurrence behaves like a linear ODE -- the solution is a combination of exponentials (geometric sequences).
- **Master Theorem**: For T(n) = aT(n/b) + f(n): compare f(n) with n^(log_b(a)). If f grows slower: T = Theta(n^(log_b(a))). If same: T = Theta(n^(log_b(a)) * log n). If faster: T = Theta(f(n)). *Intuition*: the recursion tree either spends most work at the leaves, distributes evenly, or spends most work at the root.
- **Derangements**: D_n = n! * Sum from k=0 to n of (-1)^k / k! (the number of permutations with no fixed points). D_n is approximately n!/e. *Intuition*: by inclusion-exclusion, subtract permutations fixing at least one element, add back those fixing at least two, etc.

**Worked example**:
> Solve a_n = 5*a_(n-1) - 6*a_(n-2) with a_0 = 1, a_1 = 4. Characteristic equation: r^2 - 5r + 6 = 0, roots r = 2, r = 3. General solution: a_n = A*2^n + B*3^n. From a_0 = 1: A + B = 1. From a_1 = 4: 2A + 3B = 4. Solving: B = 2, A = -1. So a_n = -2^n + 2*3^n.

> How many integers from 1 to 1000 are divisible by 2 or 3 or 5? |A_2| = 500, |A_3| = 333, |A_5| = 200, |A_2 intersect A_3| = 166, |A_2 intersect A_5| = 100, |A_3 intersect A_5| = 66, |A_2 intersect A_3 intersect A_5| = 33. By inclusion-exclusion: 500 + 333 + 200 - 166 - 100 - 66 + 33 = 734.

> Pigeonhole: In any group of 13 people, at least two share a birth month (13 people, 12 months).

**Essential problems**: Levin Ch 1, Exercises: #1.1.1-1.1.8, #1.2.1-1.2.6, #1.4.1-1.4.8; Ch 2, Exercises: #2.4.1-2.4.8
**Challenge problems**: Levin Ch 1, Exercises: #1.7.1-1.7.6; Ch 2, Exercises: #2.4.9-2.4.14

## 4. Additional Proof Techniques

**Textbook sections**: Ch 2, Sections 2.1-2.2; supplementary from Hammack

**Key definitions**:
- **Combinatorial proof**: Proving an identity by showing both sides count the same set in two different ways
- **Double counting**: Two ways of counting the same quantity yield an equality

**Key technique**:
- Identify a set S that one side of the identity counts. Find a different way to count S that yields the other side. No algebraic manipulation needed.

**Key theorems (proved combinatorially)**:
- **Vandermonde's Identity**: C(m+n, r) = Sum from k=0 to r of C(m,k)*C(n,r-k). *Combinatorial proof*: choose r people from a group of m men and n women. The left side counts directly. The right side splits by choosing k men and r-k women.
- **Hockey Stick Identity**: C(r,r) + C(r+1,r) + ... + C(n,r) = C(n+1,r+1). *Combinatorial proof*: to choose r+1 elements from {1,...,n+1}, let the largest chosen element be j+1 (where j >= r). Then choose the remaining r from {1,...,j}, giving C(j,r). Summing over j from r to n gives the left side.
- **Sum of Row of Pascal's Triangle**: Sum from k=0 to n of C(n,k) = 2^n. *Combinatorial proof*: the left side counts subsets of an n-element set by size. The right side counts all subsets directly (2 choices per element).

**Worked example**:
> Prove C(n,k) = C(n,n-k) combinatorially. C(n,k) counts ways to choose k items from n. For each such choice, the unchosen items form a set of size n-k. This is a bijection between k-element subsets and (n-k)-element subsets, so C(n,k) = C(n,n-k).

> Prove C(2n,2) = 2*C(n,2) + n^2 combinatorially. Left side: choose 2 people from 2n (n in group A, n in group B). Right side: both from A (C(n,2)), both from B (C(n,2)), or one from each (n*n). Total: 2*C(n,2) + n^2.

**Essential problems**: Levin Ch 1, Exercises: #1.3.1-1.3.8, #1.4.1-1.4.6; Hammack Ch 4-7 review exercises
**Challenge problems**: Levin Ch 2, Exercises on combinatorial proofs: #2.1.6-2.1.12

---

## Technique Catalog

| Technique | When to Use | Key Formula/Idea |
|-----------|-------------|------------------|
| BFS/DFS on graphs | Explore connectivity, find paths | Systematic vertex traversal |
| Euler's Formula | Analyze planar graphs | V - E + F = 2 |
| Degree-based arguments | Prove existence of certain vertices | Handshaking: sum of degrees = 2|E| |
| Euclidean Algorithm | Compute GCD efficiently | gcd(a,b) = gcd(b, a mod b) |
| Modular exponentiation | Compute a^k mod n efficiently | Repeated squaring |
| Characteristic equation | Solve linear recurrences | Factor r^k - c_1*r^(k-1) - ... = 0 |
| Generating functions | Solve counting problems, recurrences | Encode sequence as power series |
| Inclusion-Exclusion | Count union of overlapping sets | Alternate add/subtract intersections |
| Pigeonhole Principle | Prove existence of collisions/repeats | More items than containers |
| Combinatorial proof | Prove identities without algebra | Count the same set two ways |

## Connections to CS & Algorithms

| Math Concept | CS Application | Repo Link |
|-------------|----------------|-----------|
| Graph traversal (BFS/DFS) | Shortest path, cycle detection, topological sort | algorithms track |
| Graph coloring | Register allocation in compilers, scheduling | systems track |
| Planar graphs | VLSI circuit layout, map rendering | systems track |
| Trees | Data structures (BST, heaps, tries), spanning trees | algorithms track |
| Euler/Hamilton paths | Network routing, DNA sequencing (Eulerian assembly) | algorithms track |
| Modular arithmetic | Hashing, checksums, random number generators | systems track |
| RSA / number theory | Public-key cryptography, digital signatures | systems track |
| Recurrence relations | Running time of merge sort, quicksort, etc. | algorithms track |
| Master Theorem | Quick classification of divide-and-conquer complexity | algorithms track |
| Generating functions | Analysis of algorithms (Knuth-style) | algorithms track |

## Company Relevance

| Company | How This Appears | Difficulty |
|---------|-----------------|------------|
| Google/Meta | Graph problems dominate coding interviews (BFS, DFS, shortest path) | Medium-Hard |
| Amazon | Tree and graph traversal for logistics, dependency graphs | Medium |
| Security (Cloudflare, CrowdStrike) | RSA, modular arithmetic, cryptographic protocols | Hard |
| Compiler teams (Apple, LLVM) | Graph coloring for register allocation | Hard |
| Networking (Cisco, Juniper) | Routing algorithms use graph theory directly | Medium |
| Any SWE role | Recurrence analysis for understanding algorithm complexity | Medium |

## Chapters

<!-- ol:chapters -->
| # | Module | Chapter | Kind | Pass |
|---|---|---|---|---|
| 1 | `M06.1` | [Graphs, DAGs, and an iterative topological sort](01-graphs-dags-and-topological-sort.md) | build | 2 |
| 2 | `M06.2` | [Trees and tries, longest-prefix match](02-trees-and-tries-longest-prefix-match.md) | build | 3 |
| 3 | `M06.3` | [Modular arithmetic, hashing, and PCG32 in Python](03-modular-arithmetic-hashing-and-pcg32.md) | build | 2 |
| 4 | `S-M06a` | [Discrete math 2 problem set, part a: DAGs, modular arithmetic, hashing](90-problem-set-a.md) | solve | 2 |
| 5 | `S-M06b` | [Solve set: trees, birthday bounds, Jaccard/MinHash/LSH S-curve, Bloom FP rate, recurrences](91-problem-set-b.md) | solve | 3 |
<!-- /ol:chapters -->
