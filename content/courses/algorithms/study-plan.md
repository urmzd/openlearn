# Algorithm Mastery (17 days)

~2-3 hours per day. Each day targets specific patterns and problems.

### Week 1: Foundations

| Day | Focus | Problems | Patterns |
|-----|-------|----------|----------|
| **1** | Arrays & Hashing | [Two Sum](../../tracks/algorithms/01-arrays-hashing/two-sum.js), [Contains Duplicate](../../tracks/algorithms/01-arrays-hashing/contains-duplicate.js), [Missing Number](../../tracks/algorithms/01-arrays-hashing/missing-number.js), [Product Except Self](../../tracks/algorithms/01-arrays-hashing/product-of-array-expect-self.js) | [Hash map complement lookup, frequency counting, index-as-hash](../../tracks/algorithms/01-arrays-hashing/README.md) |
| **2** | Two Pointers & Sliding Window | [3Sum](../../tracks/algorithms/02-two-pointers-sliding-window/3sum.js), [Container With Most Water](../../tracks/algorithms/02-two-pointers-sliding-window/container-with-most-water.js) | [Opposite-end pointers, fast/slow pointers, window expand/shrink](../../tracks/algorithms/02-two-pointers-sliding-window/README.md) |
| **3** | Binary Search | [Find Min in Rotated Array](../../tracks/algorithms/03-binary-search/find-minimum-in-rotated-sorted-array.js), [Search in Rotated Array](../../tracks/algorithms/03-binary-search/search-in-rotated-sorted-array.js), [LISS](../../tracks/algorithms/03-binary-search/liss.js) | [Invariant-based search, search on answer space](../../tracks/algorithms/03-binary-search/README.md) |
| **4** | Linked Lists | [Add Two Numbers](../../tracks/algorithms/04-linked-lists/add-two-numbers.js) | [Runner technique, dummy head, reversal](../../tracks/algorithms/04-linked-lists/README.md) |
| **5** | Trees | [AVL + Range Query](../../tracks/algorithms/05-trees/example-7.py) | [DFS/BFS traversal, path problems, BST properties](../../tracks/algorithms/05-trees/README.md) |
| **6** | Graphs (traversal) | [Number of Islands](../../tracks/algorithms/06-graphs/number-of-islands.js), [Clone Graph](../../tracks/algorithms/06-graphs/clone-graph.js) | [DFS/BFS on grids, graph copying](../../tracks/algorithms/06-graphs/README.md) |
| **7** | Graphs (dependencies) | [Course Schedule](../../tracks/algorithms/06-graphs/course-schedule.js), [Pacific Atlantic](../../tracks/algorithms/06-graphs/pacific-atlantic-water-flow.ts) | [Topological sort, multi-source BFS](../../tracks/algorithms/06-graphs/README.md) |

### Week 2: Optimization & Search

| Day | Focus | Problems | Patterns |
|-----|-------|----------|----------|
| **8** | DP: 1D basics | [Climbing Stairs](../../tracks/algorithms/07-dynamic-programming/climbing-stairs.js), [House Robber](../../tracks/algorithms/07-dynamic-programming/house-robber.js), [House Robber II](../../tracks/algorithms/07-dynamic-programming/rob-houses-pt-2.js), [Coin Change](../../tracks/algorithms/07-dynamic-programming/change-coin.js) | [Fibonacci-style, take/skip, unbounded knapsack](../../tracks/algorithms/07-dynamic-programming/README.md) |
| **9** | DP: sequences | [Decode Ways](../../tracks/algorithms/07-dynamic-programming/decode-ways.js), [Word Break](../../tracks/algorithms/07-dynamic-programming/word-break.js), [LCS](../../tracks/algorithms/07-dynamic-programming/lcs.js), [Max Subarray](../../tracks/algorithms/07-dynamic-programming/maximum-subarray.js) | [Partition DP, two-string DP, Kadane's](../../tracks/algorithms/07-dynamic-programming/README.md) |
| **10** | DP: advanced | [Max Product Subarray](../../tracks/algorithms/07-dynamic-programming/maximum-product-subarray.js), [Unique Paths](../../tracks/algorithms/07-dynamic-programming/unique-paths.js), [Knapsack](../../tracks/algorithms/07-dynamic-programming/example-1.py), [Edit Distance](../../tracks/algorithms/07-dynamic-programming/example-2.py) | [Grid DP, min/max tracking, 0/1 knapsack](../../tracks/algorithms/07-dynamic-programming/README.md) |
| **11** | DP: hard variants | [Kadane](../../tracks/algorithms/07-dynamic-programming/example-3.py), [DP on Graph](../../tracks/algorithms/07-dynamic-programming/example-4.py), [Optimal BST](../../tracks/algorithms/07-dynamic-programming/example-5.c), [Alignment](../../tracks/algorithms/07-dynamic-programming/example-8.c) | [Interval DP, Knuth optimization, reconstruction](../../tracks/algorithms/07-dynamic-programming/README.md) |
| **12** | Greedy | [Buy/Sell Stock](../../tracks/algorithms/08-greedy/best-time-to-buy-and-sell-stock.js), [Jump Game](../../tracks/algorithms/08-greedy/jump-game.js), [Huffman Coding](../../tracks/algorithms/08-greedy/huffman-coding), [Examples 1-3](../../tracks/algorithms/08-greedy/example-1.py) | [Greedy choice property, exchange argument, priority queues](../../tracks/algorithms/08-greedy/README.md) |
| **13** | Backtracking & Search | [Combination Sum](../../tracks/algorithms/09-backtracking/combination-sum.js), [Map Coloring](../../tracks/algorithms/09-backtracking/example-2.py) | [Choice/explore/unchoose, pruning, constraint propagation](../../tracks/algorithms/09-backtracking/README.md) |
| **14** | Math, Bits & Recursion | [Count Bits](../../tracks/algorithms/10-math-bit/count-number-of-bits.js), [1 Bits](../../tracks/algorithms/10-math-bit/number-of-1-bits.js), [Reverse Bits](../../tracks/algorithms/10-math-bit/reverse-bits.js), [Sum Without +](../../tracks/algorithms/10-math-bit/sum-of-two-integers.js), [Tree Path](../../tracks/algorithms/11-recursion-divide-conquer/example-1.py), [Max Subarray D&C](../../tracks/algorithms/11-recursion-divide-conquer/example-3.py) | [Bit tricks, divide & conquer](../../tracks/algorithms/10-math-bit/README.md), [Recursion patterns](../../tracks/algorithms/11-recursion-divide-conquer/README.md) |

### Week 3: Specialized Tracks & Interview Prep

| Day | Focus | Problems | Patterns |
|-----|-------|----------|----------|
| **15** | Probabilistic structures | [Bloom Filter](../../tracks/algorithms/15-probabilistic-structures/bloom.py) | [Bloom filters, HyperLogLog, skip lists](../../tracks/algorithms/15-probabilistic-structures/README.md) |
| **16** | Company-targeted review | Pick 2-3 companies from [interviews/](../../interviews/README.md) and study their guides | [Shared concepts across companies](../../interviews/shared-concepts/README.md) |
| **17** | Mock interview day | Revisit 1 problem from each of topics 01-11 under timed conditions (45 min each) | Review all pattern READMEs as quick reference |

### Daily Routine (Algorithms)

1. **Read patterns first** (15 min) -- read the topic README before touching code
2. **Solve problems** (60-90 min) -- attempt each problem for 20 min before looking at the solution
3. **Review and annotate** (30 min) -- understand the solution, note edge cases, trace through examples
4. **Spaced review** (15 min) -- re-solve one problem from a previous day without looking at code

---
