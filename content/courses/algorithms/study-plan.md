# Study plan

This plan paces the pattern chapters of *Algorithms* over seventeen days of two to three hours each, in three weeks. It is a suggestion, not a gate: if a pattern is already familiar, skim its chapter opening and move on; if one is new, give it an extra day. The order matters more than the calendar, because later patterns reuse earlier ones. Dynamic programming leans on recursion and hash maps, graph search leans on queues and sets, and greedy proofs make more sense once you have seen a DP that the greedy choice would get wrong.

## A daily rhythm

Each day works best with the same four movements. Start by reading the chapter opening for the day's pattern, about fifteen minutes, before you touch any code, so that you know what invariant or state you are looking for. Then spend sixty to ninety minutes on the problems: give each one a genuine twenty-minute attempt before you open the solution, because the attempt is what makes the solution memorable. Spend about thirty minutes reviewing and annotating the solution, tracing it by hand on a small example and writing down the edge cases it handles. Finish with fifteen minutes of spaced review, re-solving one problem from an earlier day without looking at its code.

## Week 1: foundations

The first week builds the toolkit that every later chapter assumes. Day 1 is [arrays and hashing](01-arrays-hashing/README.md): complement lookup, frequency counting, and using indices as a hash, through [Two Sum](01-arrays-hashing/two-sum.js), [Contains Duplicate](01-arrays-hashing/contains-duplicate.js), [Missing Number](01-arrays-hashing/missing-number.js), and [Product of Array Except Self](01-arrays-hashing/product-of-array-expect-self.js).

Day 2 turns to [two pointers and sliding windows](02-two-pointers-sliding-window/README.md), where opposite-end pointers, fast and slow pointers, and windows that expand and shrink replace nested loops. Work [3Sum](02-two-pointers-sliding-window/3sum.js) and [Container With Most Water](02-two-pointers-sliding-window/container-with-most-water.js), paying attention to why each pointer move is safe.

Day 3 is [binary search](03-binary-search/README.md) as invariant-based search and as search over an answer space: [Find Minimum in Rotated Sorted Array](03-binary-search/find-minimum-in-rotated-sorted-array.js), [Search in Rotated Sorted Array](03-binary-search/search-in-rotated-sorted-array.js), and the [longest increasing subsequence](03-binary-search/liss.js).

Day 4 covers [linked lists](04-linked-lists/README.md), the runner technique, dummy heads, and reversal, with [Add Two Numbers](04-linked-lists/add-two-numbers.js). Day 5 moves to [trees](05-trees/README.md): DFS and BFS traversal, path problems, and BST properties, practiced on the [AVL tree with a range query](05-trees/example-7.py).

The week ends with two days on [graphs](06-graphs/README.md). Day 6 is traversal, DFS and BFS on grids and graph copying, with [Number of Islands](06-graphs/number-of-islands.js) and [Clone Graph](06-graphs/clone-graph.js). Day 7 is dependencies: topological sort in [Course Schedule](06-graphs/course-schedule.js) and multi-source BFS in [Pacific Atlantic Water Flow](06-graphs/pacific-atlantic-water-flow.ts).

## Week 2: optimization and search

The second week is mostly [dynamic programming](07-dynamic-programming/README.md), taken in four steps of increasing difficulty. Day 8 covers the one-dimensional basics: Fibonacci-style recurrences, take-or-skip choices, and unbounded knapsack, through [Climbing Stairs](07-dynamic-programming/climbing-stairs.js), [House Robber](07-dynamic-programming/house-robber.js), [House Robber II](07-dynamic-programming/rob-houses-pt-2.js), and [Coin Change](07-dynamic-programming/change-coin.js). Day 9 moves to sequences: partition DP, two-string DP, and Kadane's algorithm, with [Decode Ways](07-dynamic-programming/decode-ways.js), [Word Break](07-dynamic-programming/word-break.js), [Longest Common Subsequence](07-dynamic-programming/lcs.js), and [Maximum Subarray](07-dynamic-programming/maximum-subarray.js).

Day 10 is the advanced tier: grid DP, tracking both a running minimum and maximum, and 0/1 knapsack, in [Maximum Product Subarray](07-dynamic-programming/maximum-product-subarray.js), [Unique Paths](07-dynamic-programming/unique-paths.js), [0/1 knapsack](07-dynamic-programming/example-1.py), and [edit distance](07-dynamic-programming/example-2.py). Day 11 takes on the hard variants, interval DP, Knuth's optimization, and solution reconstruction, in [Kadane's algorithm](07-dynamic-programming/example-3.py), [DP on a graph](07-dynamic-programming/example-4.py), the [optimal binary search tree](07-dynamic-programming/example-5.c), and [sequence alignment](07-dynamic-programming/example-8.c).

Day 12 is [greedy algorithms](08-greedy/README.md): the greedy-choice property, the exchange argument, and priority queues, with [Best Time to Buy and Sell Stock](08-greedy/best-time-to-buy-and-sell-stock.js), [Jump Game](08-greedy/jump-game.js), [Huffman coding](08-greedy/huffman-coding), and the three Python examples starting with [example 1](08-greedy/example-1.py). Day 13 is [backtracking](09-backtracking/README.md), choose, explore, unchoose, with pruning and constraint propagation, through [Combination Sum](09-backtracking/combination-sum.js) and [map coloring](09-backtracking/example-2.py).

Day 14 closes the week with two short chapters. From [math and bit manipulation](10-math-bit/README.md), work [Counting Bits](10-math-bit/count-number-of-bits.js), [Number of 1 Bits](10-math-bit/number-of-1-bits.js), [Reverse Bits](10-math-bit/reverse-bits.js), and [Sum of Two Integers](10-math-bit/sum-of-two-integers.js) without `+`. From [recursion and divide-and-conquer](11-recursion-divide-conquer/README.md), work the [tree path](11-recursion-divide-conquer/example-1.py) problem and the [divide-and-conquer maximum subarray](11-recursion-divide-conquer/example-3.py).

## Week 3: specialized structures and interview practice

The last three days widen the view. Day 15 is [probabilistic data structures](15-probabilistic-structures/README.md): Bloom filters, HyperLogLog, and skip lists, with the [Bloom filter](15-probabilistic-structures/bloom.py) implementation. On day 16, pick two or three companies from [*Interview Preparation*](../interview-preparation/README.md), read their guides, and compare them against the [concepts shared across companies](../interview-preparation/shared-concepts/README.md). Day 17 is a mock interview day: revisit one problem from each of chapters 01 to 11 under timed conditions, forty-five minutes each, and use the chapter openings as your quick reference between rounds.

The [systems data structures](16-systems-data-structures/README.md) chapter is not part of this seventeen-day route. Its sections are best read alongside *Build Your Own LLM System*, each one just before the part of that book that first uses the structure.
