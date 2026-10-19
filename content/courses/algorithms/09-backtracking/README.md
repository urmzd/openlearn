# 09-backtracking

## Summary
- Contains: `combination-sum.js`, `example-2.py`.
- Backtracking exercises in JS and Python.

## Key takeaways
- Use recursion with pruning to explore combinations efficiently.
- Backtrack by undoing mutations or using fresh state per call.

## How to run
- JS file is a solution function (run with Node + a driver or an online judge).
- Python example is runnable with `python3 09-backtracking/example-2.py`.

---

# Backtracking Patterns

## Core Insight

Backtracking is **DFS on the decision tree** with pruning. At each node, you make a choice, recurse, then undo the choice. The key to performance is **pruning early** — don't explore branches that can't lead to valid solutions.

## Pattern 1: Subsets / Power Set

**When to use**: Generate all subsets of a set.

**Template**:
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

**With duplicates**: Sort first, skip `nums[i] == nums[i-1]` when `i > start`.

**Interview problems**: Subsets, Subsets II

## Pattern 2: Permutations

**When to use**: Generate all orderings of elements.

**Template**:
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

**With duplicates**: Sort, use a visited array, skip `nums[i] == nums[i-1] and not visited[i-1]`.

**Interview problems**: Permutations, Permutations II

## Pattern 3: Combination Sum

**When to use**: Find combinations that sum to a target.

**Template**:
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

**Interview problems**: Combination Sum, Combination Sum II, Combination Sum III

## Pattern 4: Board/Grid Search

**When to use**: Word search in grid, Sudoku solver, N-Queens.

**Template (Word Search)**:
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

**Interview problems**: Word Search, Word Search II (use Trie), N-Queens, Sudoku Solver

## Pattern 5: Constraint Satisfaction

**When to use**: Assign values to variables subject to constraints (graph coloring, Sudoku).

**Template**:
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

**Interview problems**: N-Queens, Sudoku Solver, Graph Coloring, Map Coloring

## Pruning Strategies

1. **Sort the input**: Enables early termination when remaining values are too large/small
2. **Skip duplicates**: `if i > start and nums[i] == nums[i-1]: continue`
3. **Bound checking**: If partial solution already exceeds target, stop
4. **Constraint propagation**: Reduce domains before recursing (Sudoku AC-3)
5. **Symmetry breaking**: Fix the first element to avoid symmetric solutions

## Company Targeting

| Company | Favorite Variant | Difficulty |
|---------|-----------------|------------|
| Google | N-Queens, Sudoku, Word Search II | Hard |
| Meta | Permutations, subsets | Medium |
| Amazon | Combination sum variants | Medium |
| Palantir | Constraint satisfaction | Hard |
| Anthropic | Search with pruning heuristics | Medium-Hard |

## Time Complexity

| Problem Type | Branching | Depth | Complexity |
|-------------|-----------|-------|------------|
| Subsets | 2 (include/exclude) | n | O(2^n) |
| Permutations | n down to 1 | n | O(n!) |
| Combination sum | variable | target/min | Exponential |
| N-Queens | ≤ n | n | O(n!) worst |
| Sudoku | ≤ 9 | 81 | O(9^81) worst |
