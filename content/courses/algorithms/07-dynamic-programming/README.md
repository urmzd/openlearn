# Dynamic programming

Dynamic programming is recursion plus memoization: **don't solve the same subproblem twice.** A naive recursion for Fibonacci, edit distance, or knapsack recomputes the same subproblems an exponential number of times; storing each answer the first time it is computed makes the whole computation polynomial. The idea is simple. The hard part is identifying the **state** (what defines a subproblem) and the **transition** (how subproblems relate). If you can write the recurrence, you can write the DP.

This is the longest chapter in the book because DP is less one pattern than a family of them, distinguished by the shape of the state. It moves from one-dimensional state over a sequence, through grids and pairs of sequences, to knapsack, intervals, state machines, bitmasks, and trees. Each pattern comes with a template; the chapter then gives a five-step method for approaching any DP problem and walks through seventeen solutions in JavaScript, Python, and C.

> **Learning objectives**
> - Define the state and transition of a DP problem and identify its base cases and evaluation order.
> - Choose between top-down memoization and bottom-up tables, and reduce space with rolling arrays.
> - Apply the linear, grid, two-sequence, knapsack, interval, state-machine, bitmask, and tree DP templates.
> - Explain why 0/1 knapsack iterates capacity backwards and unbounded knapsack forwards.
> - Reconstruct an optimal solution, not just its value, from a DP table.

## Before you begin

You should be comfortable with recursion, as in [Recursion and divide-and-conquer](../11-recursion-divide-conquer/README.md), and with tree traversal from [Trees](../05-trees/README.md) for the last pattern. Some examples read graph structure, covered in [Graphs](../06-graphs/README.md), and one uses probabilities; [Probability and statistics](../../mathematics-for-computing/08-probability-statistics/README.md) in *Mathematics for Computing* covers the background.

## Linear DP

When the answer at position $i$ depends on the answers at a few earlier positions in a sequence, the state is one index:

```
dp[i] = best answer using elements 0..i
dp[i] = max(dp[i-1] + something, dp[i-2] + something_else, ...)
```

Climbing Stairs, House Robber, Maximum Subarray, Decode Ways, Coin Change, and Word Break are all linear DPs. Because each step usually looks back only one or two positions, the table can often be replaced by a few variables.

The most important one-dimensional DP is Kadane's algorithm for the maximum subarray. The state is the best sum of a subarray *ending here*: either extend the previous one or start fresh at the current element.

```python
max_ending_here = max_so_far = nums[0]
for x in nums[1:]:
    max_ending_here = max(x, max_ending_here + x)
    max_so_far = max(max_so_far, max_ending_here)
```

## Grid DP

On a matrix where you move right or down, the state is a cell and the transition comes from the cells above and to the left:

```
dp[i][j] = best answer reaching cell (i, j)
dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
```

Unique Paths, Minimum Path Sum, Maximal Square, and Dungeon Game are grid DPs. Since each row depends only on the previous one, a single row of storage is enough.

## Two-sequence DP

When the problem relates two strings or arrays, the state is a pair of prefixes:

```
dp[i][j] = answer for first[0..i-1] and second[0..j-1]
# Transitions usually involve:
#   match: dp[i-1][j-1] + something
#   skip first: dp[i-1][j] + something
#   skip second: dp[i][j-1] + something
```

Longest Common Subsequence, Edit Distance, Regular Expression Matching, Wildcard Matching, and Interleaving String all take this form. Edit distance is the quintessential two-dimensional DP: each cell is the cheapest of a deletion, an insertion, or a replacement (free when the characters match).

```python
dp[i][j] = min(
    dp[i-1][j] + 1,        # delete
    dp[i][j-1] + 1,        # insert
    dp[i-1][j-1] + (0 if a[i]==b[j] else 1)  # replace
)
```

## Knapsack variants

To select items under a constraint such as weight or capacity while maximizing value, the state is the items considered so far and the capacity left. In the 0/1 knapsack, each item is used at most once, so each cell either skips item $i$ or takes it from the previous row:

```
dp[i][w] = max value using items 0..i with capacity w
dp[i][w] = max(dp[i-1][w], dp[i-1][w - weight[i]] + value[i])
```

To save space, keep a single row and iterate capacity **backwards**, so that `dp[w - weight]` still holds the previous row's value when you read it. In the unbounded knapsack, items can be reused, so you want the current row's value, and you iterate capacity **forwards**:

```
dp[w] = max value with capacity w (items reusable)
for each item: dp[w] = max(dp[w], dp[w - weight] + value)
```

0/1 Knapsack, Coin Change (unbounded), Partition Equal Subset Sum, and Target Sum are knapsack problems.

## Interval DP

When the optimal strategy is over a contiguous range $[i, j]$ and you must try every split point, fill the table by increasing interval length so that both halves of every split are already solved:

```
for length in range(2, n+1):
    for i in range(n - length + 1):
        j = i + length - 1
        for k in range(i, j):
            dp[i][j] = min(dp[i][j], dp[i][k] + dp[k+1][j] + cost(i, j))
```

The three nested loops make this $O(n^3)$. Matrix Chain Multiplication, Burst Balloons, Minimum Cost Tree From Leaf Values, and the optimal binary search tree are interval DPs.

## State-machine DP

Some problems have several states at each position: holding a stock, having just sold, resting during a cooldown. Keep one value per state and update them together from the previous position's values:

```
held = -prices[0]    # holding a stock
sold = 0             # just sold
rest = 0             # cooldown / idle
for price in prices[1:]:
    held, sold, rest = max(held, rest - price), held + price, max(rest, sold)
```

The simultaneous assignment matters: every new value must be computed from the old ones. This template solves Best Time to Buy and Sell Stock with Cooldown, and the same idea covers Best Time to Buy and Sell Stock I through IV and With Transaction Fee.

## Bitmask DP

When there are at most about 20 items and the state must record which subset has been used, encode the subset as the bits of an integer:

```
dp[mask] = best answer using the subset of items encoded in mask
for mask in range(1 << n):
    for i in range(n):
        if mask & (1 << i):
            dp[mask] = min(dp[mask], dp[mask ^ (1 << i)] + cost(i, mask))
```

There are $2^n$ masks, which is why $n$ must be small. The Travelling Salesman Problem, Find the Shortest Superstring, Can I Win, and Partition to K Equal Sum Subsets are bitmask DPs. [Math and bit manipulation](../10-math-bit/README.md) covers the bit operations.

## DP on trees

Finally, for optimization over a tree, each node returns a small tuple of answers for its subtree, typically one for "this node included" and one for "excluded":

```python
def dp(node):
    if not node: return base
    left = dp(node.left)
    right = dp(node.right)
    include_node = f(node.val, left[EXCLUDE], right[EXCLUDE])
    exclude_node = g(best(left), best(right))
    return (include_node, exclude_node)
```

House Robber III, Binary Tree Maximum Path Sum, and Binary Tree Cameras use this pattern, which [Trees](../05-trees/README.md) introduced.

## How to approach any DP problem

Whatever the pattern, the same five steps get you from a problem statement to code:

1. **Define the state.** What information do I need to make the optimal decision?
2. **Write the recurrence.** How does `dp[current]` relate to `dp[smaller]`?
3. **Identify the base cases.** What is the answer for the smallest subproblems?
4. **Determine the order.** Top-down with memoization, or bottom-up? If bottom-up, which dimensions iterate first?
5. **Optimize space.** Can I drop a dimension? Two dimensions often become one by reusing the previous row or a rolling array.

Interviewers favor different variants. As a rough guide:

| Company | Favorite variant | Difficulty |
|---------|-----------------|------------|
| Google | interval DP, bitmask DP | Hard |
| Meta | linear DP, two-sequence DP | Medium-Hard |
| Amazon | knapsack, coin change | Medium |
| Two Sigma | DP-heavy algorithmic rounds | Hard |
| Citadel | complex state-machine DP | Hard |
| Jane Street | probability DP, game-theory DP | Hard |

## In this chapter

The JavaScript files are solution functions; run them with Node and a small driver, or in an online judge. The Python and C examples are standalone programs: run them with `python3 <file>`, or compile with `cc <file> -o <bin>` and run the binary.

The JavaScript solutions, in roughly increasing difficulty:

- [Climbing Stairs](climbing-stairs.js): the Fibonacci recurrence $f(n) = f(n-1) + f(n-2)$, kept in two rolling variables after the base cases $1$ and $2$.
- [House Robber](house-robber.js): take-or-skip, where each house either adds its value to the best total two houses back or is skipped. Written as a tail recursion that carries the last two totals.
- [House Robber II](rob-houses-pt-2.js): the houses form a circle, so the first and last cannot both be robbed. It runs the linear solution twice, once without the last house and once without the first, and takes the larger.
- [Coin Change](change-coin.js): unbounded knapsack, bottom-up. `dp[a]` is the fewest coins making amount `a`, initialized to the impossible value `amount + 1`, and the answer is `-1` if it is never improved.
- [Unique Paths](unique-paths.js): grid DP in a single row; each cell accumulates the paths from the cell to its left and the one above.
- [Decode Ways](decode-ways.js): partition DP over a digit string. A nonzero digit continues every decoding so far, and a valid two-digit code (10 to 26) adds the decodings from two positions back.
- [Word Break](word-break.js): top-down partition DP with a memo by start index, trying only prefixes up to the longest dictionary word.
- [Maximum Subarray](maximum-subarray.js): Kadane's algorithm.
- [Maximum Product Subarray](maximum-product-subarray.js): Kadane's idea extended to products by tracking both the running maximum and the running minimum, since multiplying by a negative number swaps them.
- [Longest Common Subsequence](lcs.js): two-string DP using two alternating rows, so space is linear in the length of one string.

The Python and C worked examples:

- [0/1 knapsack with squashing](example-1.py): extends the binary knapsack with a third choice per item, packing it "squashed" at half its volume (rounded up) for half its profit, and reconstructs the chosen items from the table. Its docstring argues correctness from the standard knapsack: the extension only adds choices, so the profit can never be worse.
- [Edit distance and global alignment](example-2.py): fills the edit-distance table and recovers an alignment of the two strings by tracing back through change, insert, and delete moves, printing gaps as `-`.
- [Kadane's algorithm](example-3.py): the maximum subarray in Python, with a short correctness argument in its docstring.
- [DP on a graph](example-4.py): the longest monochromatic path, computed with a DFS whose memo records, for each node, the longest path of its color through its neighbors.
- [Optimal binary search tree](example-5.c): interval DP in C over key access frequencies, keeping a cost table and a root table; `main` runs it on the frequencies `{5, 3, 4, 1, 1, 2, 1}`.
- [Probabilistic transitions](example-6.py): a DP over probabilities. Starting from town `A` and riding a fixed sequence of bus companies, each of which goes to one of its destinations uniformly at random, it computes the probability of ending in each town by pushing probability mass forward one trip at a time.
- [Sequence alignment with reconstruction](example-8.c): alignment in C in $O(mn)$ time, recovering the aligned strings by keeping pointers to shared character nodes rather than copying strings. The comments explain how restricting the table to cells with $|i - j| \le k$ gives $O((m+n)k)$. Given no input it aligns `AGATACATCA` with `GATTAGATACAT`.
