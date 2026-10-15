# Competitive Programming

Advanced algorithmic problem-solving techniques beyond standard interview prep. Builds the speed and depth needed for ICPC, Codeforces, and quantitative trading interviews.

## Overview

- **Primary textbook**: *Competitive Programmer's Handbook* by Antti Laaksonen -- [Free PDF](https://cses.fi/book/book.pdf)
- **Problem sets**: [CSES Problem Set](https://cses.fi/problemset/) (300 curated problems), [Codeforces](https://codeforces.com/), [AtCoder](https://atcoder.jp/)
- **Prerequisites**: [Algorithms track](../algorithms/) (especially topics 01-11)
- **Estimated time**: 6-8 weeks at 8-10 hrs/week (ongoing practice after)

## Key Takeaways

- Competitive programming demands both correctness and speed -- O(n log n) isn't always enough; constant factors matter
- Most hard problems combine 2-3 known techniques; the skill is recognizing which ones
- Implementation speed comes from having templates and patterns internalized, not from typing fast
- The gap between "knowing the algorithm" and "solving the problem" is pattern recognition

## How to Study

- Work through the CSES Problem Set in order -- it's deliberately sequenced
- Read the handbook chapter before attempting problems in that category
- Time yourself: 30 min per problem initially, reduce to 15 min as you improve
- Upsolve every problem you can't solve -- read editorials, implement, then re-solve next week

---

# Concepts & Techniques

## Core Insight

Every competitive programming problem reduces to: (1) identify the mathematical structure, (2) pick the right data structure, (3) implement it cleanly in under 30 minutes. The handbook covers techniques rarely seen in interview prep but essential for hard problems and quant interviews.

## 1. Number Theory

**Handbook sections**: Ch 21-22

**Key ideas**:
- **Modular arithmetic**: operations under mod, modular inverse via Fermat's little theorem (a^(p-2) mod p)
- **Sieve of Eratosthenes**: O(n log log n) prime generation; linear sieve for multiplicative functions
- **GCD & Extended Euclidean**: ax + by = gcd(a,b); Bezout's identity
- **Chinese Remainder Theorem**: solving systems of modular equations
- **Euler's totient**: counting coprime integers; application to RSA

**Essential problems**: CSES -- Exponentiation, Counting Divisors, Common Divisors, Sum of Divisors
**Challenge problems**: Codeforces -- Necklace of Beads, GCD Table

**Connections**: [Discrete Math 2](../math/06-discrete-math-2/) (number theory), cryptography, quant interview math puzzles

## 2. Advanced Graph Algorithms

**Handbook sections**: Ch 15-20

**Key ideas**:
- **Shortest paths**: Dijkstra (non-negative), Bellman-Ford (negative edges), Floyd-Warshall (all-pairs), SPFA
- **Minimum spanning trees**: Kruskal's (union-find), Prim's (priority queue)
- **Strongly connected components**: Kosaraju's, Tarjan's algorithms; 2-SAT reduction
- **Network flow**: Ford-Fulkerson, Edmonds-Karp (O(VE^2)), Dinic's (O(V^2 E)); min-cut max-flow theorem
- **Bipartite matching**: Hungarian algorithm, Hopcroft-Karp
- **Euler tours & paths**: Hierholzer's algorithm; necessary conditions

**Essential problems**: CSES -- Shortest Routes I/II, Road Reparation, Planets and Kingdoms, Download Speed
**Challenge problems**: CSES -- Police Chase, School Dance, Distinct Routes

**Connections**: [Graphs](../algorithms/06-graphs/) for basics; network flow appears in quant trading (optimal execution)

## 3. Segment Trees & Range Queries

**Handbook sections**: Ch 9, Ch 28

**Key ideas**:
- **Static range queries**: prefix sums (1D, 2D), sparse table (O(1) RMQ after O(n log n) build)
- **Segment tree**: point update + range query in O(log n); build in O(n)
- **Lazy propagation**: range updates in O(log n); deferred updates
- **Persistent segment tree**: version history, O(log n) per operation
- **Fenwick tree (BIT)**: simpler alternative for prefix operations; O(log n) update/query
- **Merge sort tree**: range order statistics

**Essential problems**: CSES -- Static Range Sum, Dynamic Range Sum, Range Minimum Queries, Range Update Queries
**Challenge problems**: CSES -- Salary Queries, Prefix Sum Queries, Pizzeria Queries

**Connections**: Database indexing, computational geometry, interval scheduling

## 4. String Algorithms

**Handbook sections**: Ch 26

**Key ideas**:
- **Hashing**: polynomial rolling hash; Rabin-Karp; double hashing to avoid collisions
- **KMP**: failure function, O(n+m) pattern matching
- **Z-algorithm**: Z-array construction, pattern matching alternative to KMP
- **Trie**: prefix tree, Aho-Corasick for multiple pattern matching
- **Suffix array**: O(n log n) construction; LCP array; substring queries
- **Suffix automaton**: DAG of all substrings; O(n) construction

**Essential problems**: CSES -- String Matching, Finding Borders, Finding Periods, Minimal Rotation
**Challenge problems**: CSES -- Substring Order I/II, Repeating Substring

**Connections**: [Information Theory](../math/11-information-theory/) (compression), bioinformatics (sequence alignment)

## 5. Geometry

**Handbook sections**: Ch 29-30

**Key ideas**:
- **Cross product**: orientation test (left/right/collinear), area of triangle/polygon
- **Convex hull**: Andrew's monotone chain O(n log n), Graham scan
- **Line intersection**: parametric intersection, sweep line
- **Closest pair of points**: divide-and-conquer O(n log n)
- **Polygon operations**: point-in-polygon (ray casting), area (shoelace formula)

**Essential problems**: CSES -- Point Location Test, Line Segment Intersection, Polygon Area, Convex Hull
**Challenge problems**: CSES -- Point in Polygon, Minimum Euclidean Distance

**Connections**: [Calculus 3](../math/04-calculus-3/) (vectors), computer graphics, robotics path planning

## 6. Advanced Dynamic Programming

**Handbook sections**: Ch 10, Ch 24-25

**Key ideas**:
- **Bitmask DP**: subset enumeration, Hamiltonian path, assignment problem; O(2^n * n)
- **Digit DP**: counting numbers with constraints up to N
- **DP on trees**: rerooting technique, tree diameter, subtree queries
- **Convex hull trick**: optimizing DP with linear functions; Li Chao tree
- **Divide-and-conquer optimization**: when opt[i][j] <= opt[i][j+1]; reduces O(kn^2) to O(kn log n)
- **Knuth's optimization**: when cost satisfies quadrilateral inequality

**Essential problems**: CSES -- Elevator Rides, Counting Tilings, Hamiltonian Flights
**Challenge problems**: CSES -- Money Sums, Removal Game, Two Sets II

**Connections**: [Dynamic Programming](../algorithms/07-dynamic-programming/) for foundations

## 7. Game Theory

**Handbook sections**: Ch 25

**Key ideas**:
- **Nim**: XOR of pile sizes determines winner; Sprague-Grundy theorem
- **Sprague-Grundy**: every impartial game is equivalent to a Nim heap; Grundy values
- **Minimax**: optimal play in two-player zero-sum games; alpha-beta pruning
- **Game graphs**: position → state, move → edge; winning/losing position classification

**Essential problems**: CSES -- Stick Game, Nim Game, Stair Game, Grundy's Game
**Challenge problems**: CSES -- Another Game, Nim Game II

**Connections**: [Reinforcement Learning](../ml/03-reinforcement-learning/) (game-playing agents), quant interview puzzles

## 8. Fast Fourier Transform (FFT) & Polynomial Arithmetic

**Handbook sections**: Ch 24 (Number Theory applications)

**Key ideas**:
- **FFT**: O(n log n) polynomial multiplication via DFT; Cooley-Tukey algorithm
- **NTT**: Number Theoretic Transform -- FFT over finite fields (exact integer arithmetic)
- **Applications**: large number multiplication, string matching with wildcards, convolution, generating function evaluation
- **Polynomial division**: modular inverse of polynomials

**Essential problems**: CSES -- Polynomial Multiplication
**Challenge problems**: Codeforces -- Convolution problems

**Connections**: Signal processing, [Information Theory](../math/11-information-theory/)

---

## Technique Catalog

| Technique | Time Complexity | When to Use |
|-----------|----------------|-------------|
| Prefix sums | O(n) build, O(1) query | Static range sum queries |
| Segment tree | O(n) build, O(log n) ops | Dynamic range queries with updates |
| Fenwick tree | O(n) build, O(log n) ops | Simpler prefix sum with updates |
| Sparse table | O(n log n) build, O(1) query | Static RMQ (idempotent operations) |
| Union-Find | O(alpha(n)) per op | Dynamic connectivity, MST (Kruskal) |
| KMP/Z-algo | O(n + m) | Single pattern matching |
| Suffix array | O(n log n) build | All substring queries |
| FFT/NTT | O(n log n) | Polynomial multiplication, convolution |
| Convex hull trick | O(n) / O(n log n) | DP optimization with linear cost |
| Sprague-Grundy | Game-dependent | Impartial combinatorial games |

## Progression Path

| Level | Rating (CF) | Focus |
|-------|------------|-------|
| Beginner | 800-1200 | Implementation, basic math, sorting/searching |
| Intermediate | 1200-1600 | Standard DP, graph BFS/DFS, binary search on answer |
| Advanced | 1600-2000 | Segment trees, advanced DP, number theory |
| Expert | 2000-2400 | Flows, FFT, advanced data structures, geometry |
| Master | 2400+ | Combining techniques, novel reductions, research-level |

## Company Relevance

| Company | How This Appears | Difficulty |
|---------|-----------------|------------|
| Jane Street | Probability puzzles, game theory, optimization under constraints | Expert |
| Citadel | Fast algorithms, mathematical problem-solving | Advanced-Expert |
| Two Sigma | Algorithmic puzzles, data structure design | Advanced |
| HRT | Low-latency thinking, bit manipulation, cache-friendly algorithms | Advanced |
| Google | Hard Leetcode-style with DP/graph twists | Advanced |
| DeepMind | Research-flavored algorithm design | Expert |
