# 05-trees

## Summary
- Contains: `example-7.py`.
- Python implementation of tree logic (balanced BST operations and a sample computation).

## Key takeaways
- Rotations and balance factors keep trees height-balanced.
- Encapsulating comparison logic helps reuse tree structures.

## How to run
- `python3 05-trees/example-7.py` optionally followed by integers as input values.

---

# Tree Patterns

## Core Insight

Tree problems are almost always solved with **recursion** (DFS) or **level-order traversal** (BFS). The key question: **does the answer come from combining subtree answers (bottom-up), or from passing state downward (top-down)?**

## Pattern 1: Bottom-Up DFS (Post-Order)

**When to use**: The answer at a node depends on answers from its children. This is the most common tree pattern.

**Template**:
```python
def solve(node):
    if not node:
        return base_case
    left = solve(node.left)
    right = solve(node.right)
    return combine(left, right, node.val)
```

**Interview problems**: Maximum Depth, Diameter of Binary Tree, Balanced Binary Tree, Maximum Path Sum, Subtree of Another Tree

## Pattern 2: Top-Down DFS (Pre-Order)

**When to use**: You need to pass information from parent to children (path sum, root-to-leaf constraints).

**Template**:
```python
def solve(node, state):
    if not node:
        return
    if is_leaf(node):
        process(state)
        return
    solve(node.left, updated_state)
    solve(node.right, updated_state)
```

**Interview problems**: Path Sum, Path Sum II, Sum Root to Leaf Numbers, Binary Tree Paths

## Pattern 3: Level-Order BFS

**When to use**: Need to process level by level, find shortest path, or connect nodes at the same level.

**Template**:
```python
from collections import deque
queue = deque([root])
while queue:
    level_size = len(queue)
    for _ in range(level_size):
        node = queue.popleft()
        process(node)
        if node.left: queue.append(node.left)
        if node.right: queue.append(node.right)
```

**Interview problems**: Level Order Traversal, Zigzag Level Order, Right Side View, Minimum Depth

## Pattern 4: BST Properties

**When to use**: The tree is a BST — exploit `left < root < right` invariant.

**Template (validate BST)**:
```python
def is_valid(node, lo=-inf, hi=inf):
    if not node:
        return True
    if node.val <= lo or node.val >= hi:
        return False
    return is_valid(node.left, lo, node.val) and \
           is_valid(node.right, node.val, hi)
```

**Template (in-order traversal gives sorted order)**:
```python
def inorder(node):
    if not node: return
    inorder(node.left)
    process(node.val)  # values come in sorted order
    inorder(node.right)
```

**Interview problems**: Validate BST, Kth Smallest in BST, LCA of BST, Convert Sorted Array to BST

## Pattern 5: Lowest Common Ancestor (LCA)

**When to use**: Find the deepest node that is an ancestor of both target nodes.

**Template (general binary tree)**:
```python
def lca(root, p, q):
    if not root or root == p or root == q:
        return root
    left = lca(root.left, p, q)
    right = lca(root.right, p, q)
    if left and right:
        return root       # p and q are in different subtrees
    return left or right   # both in the same subtree
```

**For BST**: If both values < root, go left. Both > root, go right. Otherwise, root is the LCA.

**Interview problems**: LCA of Binary Tree, LCA of BST

## Pattern 6: Serialization / Construction

**When to use**: Build a tree from traversal orders, or serialize/deserialize.

**Key facts**:
- **Preorder + Inorder** → unique tree
- **Postorder + Inorder** → unique tree
- **Preorder + Postorder** → unique tree only if full binary tree
- **Level-order + Inorder** → unique tree

**Interview problems**: Construct Binary Tree from Preorder and Inorder, Serialize and Deserialize Binary Tree

## Pattern 7: Tree DP

**When to use**: Optimization over tree structure (max path sum, house robber on tree).

**Template**:
```python
def solve(node):
    if not node:
        return (val_if_included_base, val_if_excluded_base)
    left = solve(node.left)
    right = solve(node.right)
    include = node.val + left[1] + right[1]  # can't include children
    exclude = max(left) + max(right)          # take best of children
    return (include, exclude)
```

**Interview problems**: House Robber III, Binary Tree Maximum Path Sum, Binary Tree Cameras

## Company Targeting

| Company | Favorite Variant | Difficulty |
|---------|-----------------|------------|
| Google | Serialize/deserialize, tree DP | Hard |
| Meta | LCA, iterative traversals | Medium |
| Amazon | BST operations, validate BST | Medium |
| Apple | AVL/Red-Black tree concepts | Medium-Hard |
| Anthropic | Tree as system state (tries, radix) | Medium-Hard |

## Recursion Debugging Tip

When your tree recursion isn't working:
1. Verify base case (null node)
2. Verify leaf case (no children)
3. Check: are you combining subtree results correctly?
4. Draw the recursion tree for a 3-node example
