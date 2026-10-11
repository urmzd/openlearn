# Algorithm Mastery

Interview-style algorithms organized by priority, with specialized tracks.

> **Math Foundations** -- Many topics here build on discrete math, probability, and linear algebra. See the [Math track](../math/) for prerequisite material.

See [Study Plan](../STUDY-PLAN.md) for the 21-day algorithm study plan.

## Core References

| Book | Access | Use For |
|------|--------|---------|
| *Introduction to Algorithms* (CLRS), 4th ed. -- Cormen, Leiserson, Rivest, Stein | Owned (MIT Press) | Comprehensive reference for all algorithm topics; formal proofs, correctness arguments |
| *Competitive Programmer's Handbook* -- Antti Laaksonen | [Free PDF](https://cses.fi/book/book.pdf) | Concise implementations, contest techniques, advanced topics (segment trees, FFT) |
| *Codeless Data Structures and Algorithms* | Owned | Conceptual understanding, visual intuition |
| *Algorithm Design Manual* -- Skiena | Recommended | War stories, practical algorithm selection |

CLRS chapter references are noted in each topic README where applicable. For competition-focused study, see the [Competitive Programming](../competitive-programming/) track.

## Topics

| # | Topic | Problems | Lang | README |
|---|-------|----------|------|--------|
| 01 | [Arrays & Hashing](01-arrays-hashing/) | [Two Sum](01-arrays-hashing/two-sum.js), [Contains Duplicate](01-arrays-hashing/contains-duplicate.js), [Product of Array Except Self](01-arrays-hashing/product-of-array-expect-self.js), [Missing Number](01-arrays-hashing/missing-number.js) | JS | [Patterns](01-arrays-hashing/README.md) |
| 02 | [Two Pointers & Sliding Window](02-two-pointers-sliding-window/) | [3Sum](02-two-pointers-sliding-window/3sum.js), [Container With Most Water](02-two-pointers-sliding-window/container-with-most-water.js) | JS | [Patterns](02-two-pointers-sliding-window/README.md) |
| 03 | [Binary Search](03-binary-search/) | [Find Min in Rotated Array](03-binary-search/find-minimum-in-rotated-sorted-array.js), [Search in Rotated Array](03-binary-search/search-in-rotated-sorted-array.js), [LISS](03-binary-search/liss.js) | JS | [Patterns](03-binary-search/README.md) |
| 04 | [Linked Lists](04-linked-lists/) | [Add Two Numbers](04-linked-lists/add-two-numbers.js) | JS | [Patterns](04-linked-lists/README.md) |
| 05 | [Trees](05-trees/) | [AVL Tree + Range Query](05-trees/example-7.py) | Python | [Patterns](05-trees/README.md) |
| 06 | [Graphs](06-graphs/) | [Clone Graph](06-graphs/clone-graph.js), [Course Schedule](06-graphs/course-schedule.js), [Number of Islands](06-graphs/number-of-islands.js), [Pacific Atlantic Water Flow](06-graphs/pacific-atlantic-water-flow.ts), [Corgi Conundrum](06-graphs/charles-and-the-corgi-conundrum.c) | JS/TS/C | [Patterns](06-graphs/README.md) |
| 07 | [Dynamic Programming](07-dynamic-programming/) | [Climbing Stairs](07-dynamic-programming/climbing-stairs.js), [House Robber](07-dynamic-programming/house-robber.js), [House Robber II](07-dynamic-programming/rob-houses-pt-2.js), [Coin Change](07-dynamic-programming/change-coin.js), [Unique Paths](07-dynamic-programming/unique-paths.js), [Decode Ways](07-dynamic-programming/decode-ways.js), [Word Break](07-dynamic-programming/word-break.js), [Max Subarray](07-dynamic-programming/maximum-subarray.js), [Max Product Subarray](07-dynamic-programming/maximum-product-subarray.js), [LCS](07-dynamic-programming/lcs.js), [Knapsack](07-dynamic-programming/example-1.py), [Edit Distance](07-dynamic-programming/example-2.py), [Kadane](07-dynamic-programming/example-3.py), [DP on Graph](07-dynamic-programming/example-4.py), [Optimal BST](07-dynamic-programming/example-5.c), [Probabilistic Transitions](07-dynamic-programming/example-6.py), [Alignment](07-dynamic-programming/example-8.c) | JS/Python/C | [Patterns](07-dynamic-programming/README.md) |
| 08 | [Greedy](08-greedy/) | [Best Time to Buy/Sell Stock](08-greedy/best-time-to-buy-and-sell-stock.js), [Jump Game](08-greedy/jump-game.js), [Example 1](08-greedy/example-1.py), [Example 2](08-greedy/example-2.py), [Example 3](08-greedy/example-3.py), [Huffman Coding](08-greedy/huffman-coding/) | JS/Python/Java | [Patterns](08-greedy/README.md) |
| 09 | [Backtracking](09-backtracking/) | [Combination Sum](09-backtracking/combination-sum.js), [Map Coloring](09-backtracking/example-2.py) | JS/Python | [Patterns](09-backtracking/README.md) |
| 10 | [Math & Bit Manipulation](10-math-bit/) | [Count Bits](10-math-bit/count-number-of-bits.js), [Number of 1 Bits](10-math-bit/number-of-1-bits.js), [Reverse Bits](10-math-bit/reverse-bits.js), [Sum of Two Integers](10-math-bit/sum-of-two-integers.js) | JS | [Patterns](10-math-bit/README.md) |
| 11 | [Recursion & Divide-and-Conquer](11-recursion-divide-conquer/) | [Tree Path](11-recursion-divide-conquer/example-1.py), [Max Subarray D&C](11-recursion-divide-conquer/example-3.py) | Python | [Patterns](11-recursion-divide-conquer/README.md) |
| 15 | [Probabilistic Structures](15-probabilistic-structures/) | [Bloom Filter](15-probabilistic-structures/bloom.py) | Python | [Patterns](15-probabilistic-structures/README.md) |
| 16 | [Systems Data Structures](16-systems-data-structures/) | Course modules `ds.01` to `ds.09`: Swiss and Robin Hood hash maps, heaps and top-k, LRU, radix tree, Bloom filter, bounded-load consistent hashing | C/Rust/Go | [Patterns](16-systems-data-structures/README.md) |

Topics 12 to 14 were retired; the numbers stay reserved so links keep their order. Topics 01 to 11 and 15 are the interview-pattern chapters (off the course spine, used by the interview drills); topic 16 hosts course modules.

## Interviews

Company-specific interview guides covering focus areas, question styles, and preparation strategies.

See [`interviews/README.md`](../interviews/README.md) for the full breakdown.

## Practice

Standalone implementations for hands-on practice.

See [`practice/`](../practice/) for exercises.
