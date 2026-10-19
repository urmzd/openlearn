# Backtracking

Some problems ask for every subset, every ordering, or any assignment that satisfies a set of constraints. There is no shortcut around exploring the possibilities, but there is a disciplined way to explore them. Backtracking is **DFS on the decision tree** with pruning: at each node you make a choice, recurse, and then undo the choice, so one mutable partial solution serves the whole search. The key to performance is **pruning early**: never explore a branch that cannot lead to a valid solution.

This chapter presents the choose, explore, unchoose loop in five forms (subsets, permutations, combination sums, board search, and constraint satisfaction), then collects the pruning strategies that keep the search tractable and the complexity of each problem type.

> **Learning objectives**
> - Write the choose, explore, unchoose loop for subsets, permutations, and combination sums, including inputs with duplicates.
> - Search a grid by marking cells visited in place and restoring them on return.
> - Frame a constraint satisfaction problem as variables, domains, and a consistency check.
> - Apply pruning: sorting, duplicate skipping, bounds, constraint propagation, and symmetry breaking.
> - Estimate a backtracking search's cost from its branching factor and depth.

## Before you begin

You should be comfortable with recursion ([Recursion and divide-and-conquer](../11-recursion-divide-conquer/README.md)) and with DFS on grids ([Graphs](../06-graphs/README.md)). Counting arguments for subsets and permutations are covered in [Counting params, FLOPs, and KV bytes](../../mathematics-for-computing/06-discrete-math-1/01-counting-params-flops-kv-bytes.md) in *Mathematics for Computing*.

## Subsets

To generate the power set, record every partial result and extend it only with elements after the last one chosen, so each subset is produced once:

```python
def backtrack(start, current):
    result.append(current[:])  # add every partial result
    for i in range(start, len(nums)):
        current.append(nums[i])
        backtrack(i + 1, current)
        current.pop()

result = []
backtrack(0, [])
```

Note the copy `current[:]`: the list itself keeps changing as the search continues. If the input has duplicates, sort it first and skip `nums[i] == nums[i-1]` when `i > start`, which keeps equal elements from starting the same branch twice. Subsets and Subsets II are the standard problems.

## Permutations

To generate all orderings, every unused element is a candidate at every position, and a result is recorded only when the ordering is complete:

```python
def backtrack(current):
    if len(current) == len(nums):
        result.append(current[:])
        return
    for num in nums:
        if num in current:  # or use a visited set
            continue
        current.append(num)
        backtrack(current)
        current.pop()
```

With duplicates, sort, track usage with a visited array instead of membership, and skip `nums[i]` when `nums[i] == nums[i-1] and not visited[i-1]`, so equal values are always used in a fixed order. Permutations and Permutations II are the standard problems.

## Combination sums

To find combinations that sum to a target, subtract each choice from the remaining target. Passing `i` rather than `i + 1` to the recursive call lets a candidate be reused, and on sorted input the loop can stop as soon as a candidate exceeds what is left:

```python
def backtrack(start, target, current):
    if target == 0:
        result.append(current[:])
        return
    for i in range(start, len(candidates)):
        if candidates[i] > target:
            break  # prune (requires sorted input)
        current.append(candidates[i])
        backtrack(i, target - candidates[i], current)  # i, not i+1, allows reuse
        current.pop()
```

Combination Sum, Combination Sum II, and Combination Sum III are variations on this template.

## Board and grid search

Word Search, Sudoku, and N-Queens search a board. For Word Search, mark a cell visited by overwriting it, explore its neighbors for the next letter, and restore it on the way back so other paths can use it:

```python
def backtrack(r, c, idx):
    if idx == len(word):
        return True
    if r < 0 or r >= rows or c < 0 or c >= cols:
        return False
    if board[r][c] != word[idx]:
        return False
    temp = board[r][c]
    board[r][c] = '#'  # mark visited
    found = any(backtrack(r+dr, c+dc, idx+1) for dr, dc in dirs)
    board[r][c] = temp  # restore
    return found
```

Word Search, Word Search II (which searches for many words at once with a trie), N-Queens, and Sudoku Solver are the standard problems.

## Constraint satisfaction

The most general form assigns values to variables subject to constraints, as in graph coloring or Sudoku. Try each value in a variable's domain, keep it only if it is consistent with the assignments so far, and undo it if the rest of the search fails:

```python
def backtrack(variable_index):
    if variable_index == n:
        return True  # all assigned
    for value in domain[variable_index]:
        if is_consistent(variable_index, value):
            assign(variable_index, value)
            if backtrack(variable_index + 1):
                return True
            unassign(variable_index, value)
    return False  # no valid assignment
```

N-Queens, Sudoku Solver, Graph Coloring, and Map Coloring all fit this frame.

## Pruning strategies

Every backtracking search is exponential in the worst case; pruning is what makes real inputs fast. Five strategies cover most problems:

1. **Sort the input**, which enables early termination when the remaining values are too large or too small.
2. **Skip duplicates** with `if i > start and nums[i] == nums[i-1]: continue`.
3. **Check bounds**: if a partial solution already exceeds the target, stop.
4. **Propagate constraints**: shrink the domains of unassigned variables before recursing, as AC-3 does for Sudoku.
5. **Break symmetry**: fix the first element to avoid exploring symmetric solutions.

## How expensive is the search?

The cost of a backtracking search is roughly its branching factor raised to its depth. The common problem types compare like this:

| Problem type | Branching | Depth | Complexity |
|-------------|-----------|-------|------------|
| subsets | 2 (include or exclude) | $n$ | $O(2^n)$ |
| permutations | $n$ down to 1 | $n$ | $O(n!)$ |
| combination sum | variable | target / min | exponential |
| N-Queens | $\le n$ | $n$ | $O(n!)$ worst |
| Sudoku | $\le 9$ | 81 | $O(9^{81})$ worst |

The Sudoku bound shows why pruning matters: no one runs the worst case, because constraint propagation removes almost every branch.

Interviewers favor different variants. As a rough guide:

| Company | Favorite variant | Difficulty |
|---------|-----------------|------------|
| Google | N-Queens, Sudoku, Word Search II | Hard |
| Meta | permutations, subsets | Medium |
| Amazon | combination sum variants | Medium |
| Palantir | constraint satisfaction | Hard |
| Anthropic | search with pruning heuristics | Medium-Hard |

## In this chapter

- [Combination Sum](combination-sum.js): this solution is to the counting variant, Combination Sum IV, which asks how many ordered sequences of `nums` sum to `target`. Because only the count is needed, it memoizes on the remaining target (`cache[0] = 1`) instead of building each combination, which turns the exponential search into a DP: for each target, sum the counts for `target - num` over every `num` that fits. It shows where backtracking ends and [dynamic programming](../07-dynamic-programming/README.md) begins. It is a solution function; run it with Node and a small driver, or in an online judge.
- [Map coloring](example-2.py): counts the ways to color a map of the six inhabited continents' regions with four colors so that no two neighbors share a color. It enumerates colorings of the eight "separator" regions that lie on borders between continents, and for each consistent choice counts each continent's interior colorings independently and multiplies the counts, which is far cheaper than searching all regions at once. Run it with `python3 09-backtracking/example-2.py`.
