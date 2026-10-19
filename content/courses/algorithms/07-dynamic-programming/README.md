# 07-dynamic-programming

## Summary
- Contains: `change-coin.js`, `climbing-stairs.js`, `decode-ways.js`, `house-robber.js`, `lcs.js`,
  `maximum-product-subarray.js`, `maximum-subarray.js`, `rob-houses-pt-2.js`, `unique-paths.js`,
  `word-break.js`, plus examples in Python and C (`example-1.py`, `example-2.py`, `example-3.py`,
  `example-4.py`, `example-5.c`, `example-6.py`, `example-8.c`).
- Mixed-language DP practice covering 1D/2D state and sequence problems.

## Key takeaways
- Define state clearly, then decide between top-down memoization or bottom-up tables.
- Space can often be reduced by reusing previous rows or rolling arrays.

## How to run
- JS files are solution functions (run with Node + a driver or an online judge).
- Python/C examples are standalone; run with `python3 <file>` or compile with `cc <file> -o <bin>`.

---

# Dynamic Programming Patterns

## Core Insight

DP = **recursion + memoization** = "don't solve the same subproblem twice." The hard part is identifying the **state** (what defines a subproblem) and the **transition** (how subproblems relate). If you can write the recurrence, you can write the DP.

## Pattern 1: Linear DP (1D State)

**When to use**: Answer depends on previous element(s) in a sequence.

**Template**:
```
dp[i] = best answer using elements 0..i
dp[i] = max(dp[i-1] + something, dp[i-2] + something_else, ...)
```

**Interview problems**: Climbing Stairs, House Robber, Maximum Subarray (Kadane), Decode Ways, Coin Change, Word Break

**Kadane's (Maximum Subarray) — the most important 1D DP**:
```python
max_ending_here = max_so_far = nums[0]
for x in nums[1:]:
    max_ending_here = max(x, max_ending_here + x)
    max_so_far = max(max_so_far, max_ending_here)
```

## Pattern 2: Grid DP (2D State, Spatial)

**When to use**: Problems on a matrix, moving right/down.

**Template**:
```
dp[i][j] = best answer reaching cell (i, j)
dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
```

**Interview problems**: Unique Paths, Minimum Path Sum, Maximal Square, Dungeon Game

## Pattern 3: Two-Sequence DP

**When to use**: Two strings/arrays, find some relationship between them.

**Template**:
```
dp[i][j] = answer for first[0..i-1] and second[0..j-1]
# Transitions usually involve:
#   match: dp[i-1][j-1] + something
#   skip first: dp[i-1][j] + something
#   skip second: dp[i][j-1] + something
```

**Interview problems**: Longest Common Subsequence, Edit Distance, Regular Expression Matching, Wildcard Matching, Interleaving String

**Edit Distance — the quintessential 2D DP**:
```python
dp[i][j] = min(
    dp[i-1][j] + 1,        # delete
    dp[i][j-1] + 1,        # insert
    dp[i-1][j-1] + (0 if a[i]==b[j] else 1)  # replace
)
```

## Pattern 4: Knapsack Variants

**When to use**: Select items with constraints (weight, capacity) to optimize value.

**0/1 Knapsack**:
```
dp[i][w] = max value using items 0..i with capacity w
dp[i][w] = max(dp[i-1][w], dp[i-1][w - weight[i]] + value[i])
```

Space optimization: single row, iterate capacity **backwards**.

**Unbounded Knapsack**:
```
dp[w] = max value with capacity w (items reusable)
for each item: dp[w] = max(dp[w], dp[w - weight] + value)
```
Iterate capacity **forwards**.

**Interview problems**: 0/1 Knapsack, Coin Change (unbounded), Partition Equal Subset Sum, Target Sum

## Pattern 5: Interval DP

**When to use**: Optimal strategy over a contiguous range [i, j], where you try all split points.

**Template**:
```
for length in range(2, n+1):
    for i in range(n - length + 1):
        j = i + length - 1
        for k in range(i, j):
            dp[i][j] = min(dp[i][j], dp[i][k] + dp[k+1][j] + cost(i, j))
```

**Interview problems**: Matrix Chain Multiplication, Burst Balloons, Minimum Cost Tree From Leaf Values, Optimal BST

## Pattern 6: State Machine DP

**When to use**: Multiple states at each position (holding stock, cooldown, etc.).

**Template (Buy/Sell Stock with Cooldown)**:
```
held = -prices[0]    # holding a stock
sold = 0             # just sold
rest = 0             # cooldown / idle
for price in prices[1:]:
    held, sold, rest = max(held, rest - price), held + price, max(rest, sold)
```

**Interview problems**: Best Time to Buy and Sell Stock (I-IV), With Cooldown, With Transaction Fee

## Pattern 7: Bitmask DP

**When to use**: n ≤ 20 items, need to track which subset has been used.

**Template**:
```
dp[mask] = best answer using the subset of items encoded in mask
for mask in range(1 << n):
    for i in range(n):
        if mask & (1 << i):
            dp[mask] = min(dp[mask], dp[mask ^ (1 << i)] + cost(i, mask))
```

**Interview problems**: Travelling Salesman, Shortest Superstring, Can I Win, Partition to K Equal Sum Subsets

## Pattern 8: DP on Trees

**When to use**: Optimization problem on a tree structure.

**Template**:
```python
def dp(node):
    if not node: return base
    left = dp(node.left)
    right = dp(node.right)
    include_node = f(node.val, left[EXCLUDE], right[EXCLUDE])
    exclude_node = g(best(left), best(right))
    return (include_node, exclude_node)
```

**Interview problems**: House Robber III, Binary Tree Maximum Path Sum, Binary Tree Cameras

## Company Targeting

| Company | Favorite Variant | Difficulty |
|---------|-----------------|------------|
| Google | Interval DP, bitmask DP | Hard |
| Meta | Linear DP, 2-sequence DP | Medium-Hard |
| Amazon | Knapsack, coin change | Medium |
| Two Sigma | DP-heavy algorithmic rounds | Hard |
| Citadel | Complex state machine DP | Hard |
| Jane Street | Probability DP, game theory DP | Hard |

## How to Approach Any DP Problem

1. **Define the state**: What information do I need to make the optimal decision?
2. **Write the recurrence**: How does dp[current] relate to dp[smaller]?
3. **Identify base cases**: What's the answer for the smallest subproblems?
4. **Determine order**: Bottom-up? Which dimensions iterate first?
5. **Optimize space**: Can I drop a dimension? (Often 2D → 1D)
