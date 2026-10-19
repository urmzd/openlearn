# Algorithms

This book is for programmers who can already write working code and now want to choose the right algorithm on purpose: to look at a problem, recognize the pattern underneath it, and explain why one approach runs in linear time while the obvious one runs in quadratic time. It is organized around the patterns that recur in technical interviews and in everyday systems work, and each chapter pairs a short explanation of its patterns with complete, runnable solutions you can trace line by line.

By the end you will be able to reduce nested loops to hash-map lookups and pointer sweeps, phrase a search as a binary search over an invariant, traverse and order graphs, set up a dynamic program by naming its state and transition, argue that a greedy choice is safe, prune a backtracking search, and reason about the hash maps, heaps, radix trees, and probabilistic filters that real systems are built on. More than memorizing solutions, you will be able to say why the optimal one is optimal.

## How the book is organized

The first eleven chapters, plus the chapter on probabilistic structures, are the pattern chapters. Each one opens with the core insight of its pattern family and then works through classic problems whose solutions sit beside the text. The final chapter on systems data structures is different in kind: it is a sequence of full build sections, each implementing one structure that the inference and serving system in *Build an LLM* depends on. Chapter numbers 12 to 14 are intentionally unused; they belonged to retired topics, and the numbering is kept so that existing links stay in order.

[Arrays and hashing](01-arrays-hashing/README.md) starts where almost every efficient solution starts: trading memory for time with a hash map or set. Complement lookup ([Two Sum](01-arrays-hashing/two-sum.js)), frequency counting ([Contains Duplicate](01-arrays-hashing/contains-duplicate.js)), using indices as a hash ([Missing Number](01-arrays-hashing/missing-number.js)), and prefix and suffix products ([Product of Array Except Self](01-arrays-hashing/product-of-array-expect-self.js)) each turn an $O(n^2)$ scan into $O(n)$.

[Two pointers and sliding window](02-two-pointers-sliding-window/README.md) shows how sorted order or a monotone window lets two indices replace a nested loop. You will reason about pointer-movement invariants in [3Sum](02-two-pointers-sliding-window/3sum.js) and [Container With Most Water](02-two-pointers-sliding-window/container-with-most-water.js).

[Binary search](03-binary-search/README.md) generalizes the familiar exact-match search into boundary finding over any monotone predicate, including rotated arrays ([Find Minimum in Rotated Sorted Array](03-binary-search/find-minimum-in-rotated-sorted-array.js), [Search in Rotated Sorted Array](03-binary-search/search-in-rotated-sorted-array.js)) and the patience-sorting approach to the [longest increasing subsequence](03-binary-search/liss.js).

[Linked lists](04-linked-lists/README.md) covers the runner technique, dummy heads, and reversal, with digit-by-digit arithmetic and carry handling in [Add Two Numbers](04-linked-lists/add-two-numbers.js).

[Trees](05-trees/README.md) covers depth-first and breadth-first traversal, path problems, and binary-search-tree properties, culminating in a self-balancing [AVL tree with a range query](05-trees/example-7.py) written in Python.

[Graphs](06-graphs/README.md) applies DFS and BFS to grids and adjacency lists, then turns to dependencies: [Number of Islands](06-graphs/number-of-islands.js), [Clone Graph](06-graphs/clone-graph.js), topological sort and cycle detection in [Course Schedule](06-graphs/course-schedule.js), multi-source search in [Pacific Atlantic Water Flow](06-graphs/pacific-atlantic-water-flow.ts), and a C solution to the [Corgi Conundrum](06-graphs/charles-and-the-corgi-conundrum.c).

[Dynamic programming](07-dynamic-programming/README.md) is the longest chapter. It moves from Fibonacci-style recurrences ([Climbing Stairs](07-dynamic-programming/climbing-stairs.js)) and take-or-skip choices ([House Robber](07-dynamic-programming/house-robber.js), [House Robber II](07-dynamic-programming/rob-houses-pt-2.js)) through unbounded knapsack ([Coin Change](07-dynamic-programming/change-coin.js)), grid DP ([Unique Paths](07-dynamic-programming/unique-paths.js)), partition DP ([Decode Ways](07-dynamic-programming/decode-ways.js), [Word Break](07-dynamic-programming/word-break.js)), running min and max tracking ([Maximum Subarray](07-dynamic-programming/maximum-subarray.js), [Maximum Product Subarray](07-dynamic-programming/maximum-product-subarray.js)), and two-string DP ([Longest Common Subsequence](07-dynamic-programming/lcs.js)). Worked examples in Python and C then cover [0/1 knapsack](07-dynamic-programming/example-1.py), [edit distance](07-dynamic-programming/example-2.py), [Kadane's algorithm](07-dynamic-programming/example-3.py), [DP on a graph](07-dynamic-programming/example-4.py), the [optimal binary search tree](07-dynamic-programming/example-5.c) with interval DP, [probabilistic transitions](07-dynamic-programming/example-6.py), and [sequence alignment](07-dynamic-programming/example-8.c) with reconstruction.

[Greedy algorithms](08-greedy/README.md) is about proving that a local choice leads to a global optimum, by an exchange argument or by tracking a running extreme. Its problems are [Best Time to Buy and Sell Stock](08-greedy/best-time-to-buy-and-sell-stock.js), [Jump Game](08-greedy/jump-game.js), three Python examples ([1](08-greedy/example-1.py), [2](08-greedy/example-2.py), [3](08-greedy/example-3.py)), and a Java implementation of [Huffman coding](08-greedy/huffman-coding/) built on a priority queue.

[Backtracking](09-backtracking/README.md) teaches the choose, explore, unchoose loop and how pruning and constraint propagation keep it tractable, through [Combination Sum](09-backtracking/combination-sum.js) and [map coloring](09-backtracking/example-2.py).

[Math and bit manipulation](10-math-bit/README.md) replaces loops with constant-time bitwise transformations and pays close attention to JavaScript's 32-bit signed semantics: [Counting Bits](10-math-bit/count-number-of-bits.js), [Number of 1 Bits](10-math-bit/number-of-1-bits.js), [Reverse Bits](10-math-bit/reverse-bits.js), and [Sum of Two Integers](10-math-bit/sum-of-two-integers.js) without the `+` operator.

[Recursion and divide-and-conquer](11-recursion-divide-conquer/README.md) is about base cases and input reduction, and about when splitting a problem improves its asymptotics, with a [tree path](11-recursion-divide-conquer/example-1.py) problem and a [divide-and-conquer maximum subarray](11-recursion-divide-conquer/example-3.py).

[Probabilistic data structures](15-probabilistic-structures/README.md) trades exactness for large space savings with bounded error: Bloom filters, HyperLogLog, and skip lists, with a [Bloom filter](15-probabilistic-structures/bloom.py) implementation in Python.

[Systems data structures](16-systems-data-structures/README.md) builds, in C, Rust, and Go, the structures that a production LLM server relies on: a growable array, Swiss-table and Robin Hood hash maps, an intrusive list with LRU eviction, binary heaps for top-k selection and lazy deletion, a radix tree over token ids, a Bloom filter, and a consistent hash ring with bounded loads. Each section sits just before the part of *Build an LLM* that first calls it.

## Prerequisites and related books

You should be comfortable writing loops, functions, and recursion in at least one language, and able to read JavaScript and Python; a few solutions are in C, TypeScript, and Java. Several chapters lean on discrete mathematics, probability, and linear algebra, which are covered in [*Mathematics for Computing*](../mathematics-for-computing/README.md). The systems chapter additionally expects the [C](../programming-languages/02-language-and-tool-primers/03-c.md), [Rust](../programming-languages/02-language-and-tool-primers/04-rust.md), and [Go](../programming-languages/02-language-and-tool-primers/06-go.md) primers from *Programming Languages*.

Three other books pick up where this one leaves off. [*Competitive Programming*](../competitive-programming/README.md) goes further into contest techniques. [*Interview Preparation*](../interview-preparation/README.md) collects company-specific guides covering focus areas, question styles, and preparation strategies. [*Build an LLM*](../build-an-llm/README.md) is where the systems data structures are put to work. For standalone exercises outside any book, see the repository's [practice directory](../../../practice/).

## Conventions

Solutions live next to the chapter text as source files, one problem per file, so every problem title in this book links straight to its code. The JavaScript files are solution functions: run them with Node by adding a small driver, or paste them into an online judge. The Python and C files run directly; for example, `python3 05-trees/example-7.py` optionally followed by integers as input values. Complexity is written in big-O notation with inline math such as $O(n \log n)$. Chapter openings begin with the pattern's core insight; the systems chapter follows the full section format used across these books, with learning objectives, annotated listings, call-outs such as "> **Warning:**" for pitfalls, and review questions whose answers are folded under a disclosure triangle.

## Further reading

These four references support the whole book. Chapter openings cite the relevant CLRS chapters where they apply.

- Cormen, Leiserson, Rivest, and Stein, *Introduction to Algorithms* (CLRS), 4th ed., MIT Press. The comprehensive reference for every topic here, with formal proofs and correctness arguments.
- Antti Laaksonen, *Competitive Programmer's Handbook* ([free PDF](https://cses.fi/book/book.pdf)). Concise implementations, contest techniques, and advanced topics such as segment trees and the FFT.
- *Codeless Data Structures and Algorithms*. Conceptual understanding and visual intuition without code.
- Steven Skiena, *The Algorithm Design Manual*. War stories and practical advice on choosing an algorithm.

For a paced route through the book, see the [study plan](study-plan.md).
