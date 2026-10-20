# Competitive Programming

This book covers algorithmic problem-solving beyond standard interview preparation: the speed and depth needed for ICPC, Codeforces, and quantitative trading interviews. It is a guided companion to a free textbook and a curated problem set rather than a set of chapters of its own. For each topic it tells you what to read, which ideas matter, which problems to solve first, and where the topic connects to the rest of the library.

It is written for readers who have finished the fundamentals and want to go further. Competitive programming demands both correctness and speed: O(n log n) is not always enough, and constant factors matter. Most hard problems combine two or three known techniques, and the skill is recognizing which ones. Implementation speed comes from having templates and patterns internalized, not from typing fast, and the gap between knowing an algorithm and solving a problem with it is pattern recognition. By the end you will be able to identify the structure of a hard problem, pick the right technique, and implement it cleanly under time pressure.

## The core insight

Every competitive programming problem reduces to three steps: identify the mathematical structure, pick the right data structure, and implement it cleanly in under 30 minutes. The topics below are the structures and techniques that appear most often in hard problems and quant interviews, and that are rarely covered in interview preparation.

## Prerequisites and materials

You should be comfortable with the material in [Programming and Algorithms](README.md), especially its first eleven chapters. The primary textbook is Antti Laaksonen's *Competitive Programmer's Handbook*, available as a [free PDF](https://cses.fi/book/book.pdf). The problems come from the [CSES Problem Set](https://cses.fi/problemset/), 300 curated problems, with harder practice on [Codeforces](https://codeforces.com/) and [AtCoder](https://atcoder.jp/).

## How to work through the book

Work through the CSES Problem Set in order; it is deliberately sequenced. Read the handbook chapter for a topic before attempting problems in that category. Time yourself, starting at 30 minutes per problem and working down to 15 as you improve. And upsolve every problem you cannot solve: read the editorial, implement the solution, then solve it again from scratch a week later. Practice continues well after the last topic.

## Number theory

Handbook chapters 21 and 22 cover the number theory that contest problems lean on. Modular arithmetic comes first: operations under a modulus, and the modular inverse via Fermat's little theorem, $a^{p-2} \bmod p$. The sieve of Eratosthenes generates primes in $O(n \log \log n)$, and a linear sieve computes multiplicative functions. The extended Euclidean algorithm solves $ax + by = \gcd(a, b)$, which is Bezout's identity. The Chinese remainder theorem solves systems of modular equations, and Euler's totient counts coprime integers, with RSA as its best-known application.

Start with the CSES problems Exponentiation, Counting Divisors, Common Divisors, and Sum of Divisors. For a challenge, try Necklace of Beads and GCD Table on Codeforces. The theory behind this topic is [Discrete math 2](../mathematics/12-discrete-math-2/README.md) in *Mathematics*, and the same ideas reach into cryptography and quant interview puzzles.

## Advanced graph algorithms

Handbook chapters 15 to 20 cover graphs beyond traversal. For shortest paths, Dijkstra handles non-negative weights, Bellman-Ford handles negative edges, Floyd-Warshall computes all pairs, and SPFA is a queue-based variant of Bellman-Ford. Minimum spanning trees come from Kruskal's algorithm with union-find or Prim's with a priority queue. Strongly connected components come from Kosaraju's or Tarjan's algorithm, and 2-SAT reduces to them. Network flow runs from Ford-Fulkerson through Edmonds-Karp, $O(VE^2)$, to Dinic's, $O(V^2 E)$, with the min-cut max-flow theorem tying flows to cuts. Bipartite matching uses the Hungarian algorithm or Hopcroft-Karp, and Euler tours and paths use Hierholzer's algorithm once their necessary conditions hold.

Start with Shortest Routes I and II, Road Reparation, Planets and Kingdoms, and Download Speed on CSES; the challenge problems are Police Chase, School Dance, and Distinct Routes. The basics are in [Graphs](08-graphs/README.md), and network flow appears in quant trading in problems of optimal execution.

## Segment trees and range queries

Handbook chapters 9 and 28 cover range queries. Static queries need only prefix sums, in one or two dimensions, or a sparse table, which answers range-minimum queries in $O(1)$ after an $O(n \log n)$ build. A segment tree supports a point update and a range query in $O(\log n)$ and builds in $O(n)$. Lazy propagation extends it to range updates in $O(\log n)$ by deferring work, and a persistent segment tree keeps version history at $O(\log n)$ per operation. A Fenwick tree (binary indexed tree) is a simpler alternative for prefix operations, with $O(\log n)$ update and query, and a merge sort tree answers range order statistics.

Start with Static Range Sum, Dynamic Range Sum, Range Minimum Queries, and Range Update Queries; the challenges are Salary Queries, Prefix Sum Queries, and Pizzeria Queries. The same structures show up in database indexing, computational geometry, and interval scheduling.

## String algorithms

Handbook chapter 26 covers strings. Polynomial rolling hashes power Rabin-Karp, and double hashing avoids collisions. KMP uses a failure function to match a pattern in $O(n + m)$, and the Z-algorithm builds a Z-array as an alternative. A trie is a prefix tree, and Aho-Corasick extends it to many patterns at once. A suffix array builds in $O(n \log n)$ and, with its LCP array, answers substring queries; a suffix automaton is a DAG of all substrings built in $O(n)$.

Start with String Matching, Finding Borders, Finding Periods, and Minimal Rotation; the challenges are Substring Order I and II and Repeating Substring. The connections run to compression, in [Information theory](../mathematics/17-information-theory/README.md), and to sequence alignment in bioinformatics.

## Geometry

Handbook chapters 29 and 30 cover computational geometry. The cross product is the workhorse: it gives the orientation test (left, right, or collinear) and the area of a triangle or polygon. The convex hull comes from Andrew's monotone chain in $O(n \log n)$ or a Graham scan. Line intersection uses a parametric form or a sweep line, the closest pair of points uses divide and conquer in $O(n \log n)$, and polygon operations include point-in-polygon by ray casting and area by the shoelace formula.

Start with Point Location Test, Line Segment Intersection, Polygon Area, and Convex Hull; the challenges are Point in Polygon and Minimum Euclidean Distance. The vector background is in [Calculus 3](../mathematics/10-calculus-3/README.md), and the same methods drive computer graphics and robot path planning.

## Advanced dynamic programming

Handbook chapters 10, 24, and 25 cover dynamic programming beyond the standard patterns. Bitmask DP enumerates subsets for problems such as the Hamiltonian path and the assignment problem, in $O(2^n \cdot n)$. Digit DP counts numbers up to $N$ that satisfy constraints. DP on trees uses the rerooting technique for problems such as the tree diameter and subtree queries. The convex hull trick, and its Li Chao tree variant, optimizes DPs whose transitions are linear functions. Divide-and-conquer optimization applies when $\text{opt}[i][j] \le \text{opt}[i][j+1]$ and reduces $O(kn^2)$ to $O(kn \log n)$, and Knuth's optimization applies when the cost satisfies the quadrangle inequality.

Start with Elevator Rides, Counting Tilings, and Hamiltonian Flights; the challenges are Money Sums, Removal Game, and Two Sets II. The foundations are in [Dynamic programming](09-dynamic-programming/README.md).

## Game theory

Handbook chapter 25 covers games. In Nim, the XOR of the pile sizes determines the winner. The Sprague-Grundy theorem generalizes this: every impartial game is equivalent to a Nim heap whose size is the game's Grundy value. Minimax gives optimal play in two-player zero-sum games, with alpha-beta pruning to cut the search. And any game can be modeled as a graph, positions as states and moves as edges, in which each position is classified as winning or losing.

Start with Stick Game, Nim Game, Stair Game, and Grundy's Game; the challenges are Another Game and Nim Game II. The ideas connect to game-playing agents in [Reinforcement learning](../ai-engineering/04-reinforcement-learning/README.md) and to quant interview puzzles.

## The fast Fourier transform and polynomial arithmetic

The handbook treats the FFT among its number-theory applications in chapter 24. The FFT multiplies polynomials in $O(n \log n)$ through the discrete Fourier transform, using the Cooley-Tukey algorithm. The number-theoretic transform (NTT) is the FFT over a finite field, which keeps the arithmetic exact for integers. Applications include multiplying large numbers, string matching with wildcards, convolution, and evaluating generating functions, and polynomial division uses the modular inverse of a polynomial.

Start with Polynomial Multiplication on CSES, then convolution problems on Codeforces. The FFT connects to signal processing and to [Information theory](../mathematics/17-information-theory/README.md).

## Techniques at a glance

The table collects the main techniques with their costs, for quick recall during practice:

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

## Measuring progress

Codeforces ratings give a rough map of what to focus on at each stage:

| Level | Rating (CF) | Focus |
|-------|------------|-------|
| Beginner | 800-1200 | Implementation, basic math, sorting/searching |
| Intermediate | 1200-1600 | Standard DP, graph BFS/DFS, binary search on answer |
| Advanced | 1600-2000 | Segment trees, advanced DP, number theory |
| Expert | 2000-2400 | Flows, FFT, advanced data structures, geometry |
| Master | 2400+ | Combining techniques, novel reductions, research-level |

## Where it shows up

These skills are tested directly in some interviews. Jane Street asks probability puzzles, game theory, and optimization under constraints at an expert level. Citadel looks for fast algorithms and mathematical problem-solving at an advanced to expert level. Two Sigma asks algorithmic puzzles and data-structure design, and HRT probes low-latency thinking, bit manipulation, and cache-friendly algorithms, both at an advanced level. Google asks hard LeetCode-style problems with DP or graph twists, and DeepMind asks research-flavored algorithm design at an expert level.

## Key terms

- **Segment tree**: a binary tree over array ranges in which each node stores an aggregate, such as the sum or minimum, of its range; it supports point updates and range queries in $O(\log n)$ after an $O(n)$ build.
- **Lazy propagation**: deferring range updates in a segment tree by storing pending changes at internal nodes and pushing them to the children only when a later operation descends through, which makes range updates $O(\log n)$.
- **Fenwick tree / binary indexed tree (BIT)**: an array-based structure for prefix sums with $O(\log n)$ point updates and prefix queries, smaller and simpler than a segment tree.
- **Sparse table**: precomputed answers for every range whose length is a power of two, built in $O(n \log n)$, that answer idempotent range queries such as range minimum in $O(1)$ on a static array.
- **Knuth-Morris-Pratt algorithm (KMP)**: string matching in $O(n + m)$ that uses a failure function over the pattern to know, after a mismatch, how far the pattern can shift without re-reading text.
- **Suffix array**: the starting positions of a string's suffixes in sorted order, built in $O(n \log n)$; with its LCP (longest common prefix) array it answers substring queries.
- **Sprague-Grundy theorem**: every impartial game is equivalent to a Nim heap whose size is the game's Grundy value, so a sum of games is a win for the player to move exactly when the XOR of the Grundy values is nonzero.
- **Number-theoretic transform (NTT)**: the fast Fourier transform carried out in arithmetic modulo a suitable prime, which multiplies integer polynomials in $O(n \log n)$ with exact results and no rounding.
- **Convex hull trick**: a DP optimization for transitions that minimize or maximize over a set of linear functions, keeping only the lines on their lower or upper envelope so each transition is a fast query.
