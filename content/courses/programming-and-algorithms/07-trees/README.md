# Trees

Tree problems are almost always solved with **recursion** (depth-first search) or **level-order traversal** (breadth-first search). What decides the shape of a solution is one question: **does the answer come from combining subtree answers (bottom-up), or from passing state downward (top-down)?** Once you know which, the code nearly writes itself, because the recursion mirrors the tree.

This chapter builds a toolkit of seven patterns: bottom-up and top-down DFS, level-order BFS, the binary-search-tree invariant, lowest common ancestors, construction from traversals, and dynamic programming on trees. It ends with a self-balancing AVL tree in Python that answers range queries, and with a short routine for debugging tree recursion when it goes wrong.

> **Learning objectives**
> - Decide whether a tree problem is bottom-up or top-down and write the matching recursion.
> - Traverse a tree level by level with a queue.
> - Validate a binary search tree with bounds, and use in-order traversal to visit values in sorted order.
> - Find a lowest common ancestor in a general binary tree and in a BST.
> - State which pairs of traversals determine a tree uniquely, and solve optimization problems with tree DP.
> - Explain how rotations and balance factors keep an AVL tree height-balanced.

## Before you begin

You need to be comfortable with recursion and with queues; [Recursion and divide-and-conquer](../13-recursion-divide-conquer/README.md) covers base cases and input reduction if you want a refresher first. [Trees and tries: longest prefix match](../../mathematics/12-discrete-math-2/02-trees-and-tries-longest-prefix-match.md) in *Mathematics* gives the discrete-math view of trees.

## Bottom-up DFS (post-order)

The most common tree pattern applies when the answer at a node depends on the answers from its children. Recurse into both subtrees first, then combine their results with the node's own value:

```python
def solve(node):
    if not node:
        return base_case
    left = solve(node.left)
    right = solve(node.right)
    return combine(left, right, node.val)
```

Maximum Depth of Binary Tree, Diameter of Binary Tree, Balanced Binary Tree, Binary Tree Maximum Path Sum, and Subtree of Another Tree are all bottom-up.

## Top-down DFS (pre-order)

When information has to flow from parent to children, such as a running path sum or a constraint on root-to-leaf paths, pass it as an argument and process it at the leaves:

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

Path Sum, Path Sum II, Sum Root to Leaf Numbers, and Binary Tree Paths use this form.

## Level-order BFS

When you need to process the tree level by level, find the shallowest node, or connect nodes on the same level, use a queue. Recording the queue's length at the start of each round tells you exactly how many nodes belong to the current level:

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

Binary Tree Level Order Traversal, Zigzag Level Order Traversal, Binary Tree Right Side View, and Minimum Depth of Binary Tree are the standard problems.

## Binary search tree properties

If the tree is a BST, exploit the `left < root < right` invariant. Validating a BST is a top-down problem: each node must lie strictly between bounds inherited from its ancestors, not merely between its own children:

```python
def is_valid(node, lo=-inf, hi=inf):
    if not node:
        return True
    if node.val <= lo or node.val >= hi:
        return False
    return is_valid(node.left, lo, node.val) and \
           is_valid(node.right, node.val, hi)
```

The other key fact is that an in-order traversal visits a BST's values in sorted order:

```python
def inorder(node):
    if not node: return
    inorder(node.left)
    process(node.val)  # values come in sorted order
    inorder(node.right)
```

Validate Binary Search Tree, Kth Smallest Element in a BST, Lowest Common Ancestor of a BST, and Convert Sorted Array to BST all rest on these two facts.

## Lowest common ancestor

The lowest common ancestor of two nodes is the deepest node that has both as descendants. In a general binary tree, search both subtrees; if each returns a hit, the current node is where the paths split:

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

In a BST it is simpler: if both values are less than the root, go left; if both are greater, go right; otherwise the root is the LCA. Lowest Common Ancestor of a Binary Tree and of a BST are the two classic problems.

## Construction and serialization

Some problems ask you to build a tree from its traversal orders, or to serialize a tree and read it back. Which traversals determine a tree is a fact worth memorizing:

- preorder and inorder give a unique tree;
- postorder and inorder give a unique tree;
- preorder and postorder give a unique tree only for a full binary tree;
- level order and inorder give a unique tree.

Construct Binary Tree from Preorder and Inorder Traversal and Serialize and Deserialize Binary Tree are the standard problems.

## Tree DP

Finally, optimization over tree structure (the maximum path sum, House Robber on a tree) is dynamic programming in which each subtree returns more than one value. For House Robber III, each node returns the best total if it is included and the best if it is excluded; including a node forbids including its children:

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

House Robber III, Binary Tree Maximum Path Sum, and Binary Tree Cameras use this pattern. [Dynamic programming](../09-dynamic-programming/README.md) develops the general method.

## Debugging tree recursion

When a tree recursion is not working, check four things in order:

1. Verify the base case (a null node).
2. Verify the leaf case (a node with no children).
3. Check that you are combining subtree results correctly.
4. Draw the recursion tree for a three-node example.

Interviewers favor different variants. As a rough guide:

| Company | Favorite variant | Difficulty |
|---------|-----------------|------------|
| Google | serialize and deserialize, tree DP | Hard |
| Meta | LCA, iterative traversals | Medium |
| Amazon | BST operations, validate BST | Medium |
| Apple | AVL and red-black tree concepts | Medium-Hard |
| Anthropic | trees as system state (tries, radix trees) | Medium-Hard |

## Key terms

- **Bottom-up DFS**: a tree recursion that solves both subtrees first and then combines their results with the node's own value (post-order); it fits problems where a node's answer depends on its children's answers, such as height, diameter, and balance.
- **Top-down DFS**: a tree recursion that passes state, such as a running path sum, from parent to children as an argument (pre-order) and processes it at the leaves; it fits problems where information flows down from the root.
- **Level-order traversal**: visiting a tree level by level with a queue, reading the queue's length at the start of each round to know how many nodes belong to the current level; it is breadth-first search on a tree.
- **Binary search tree (BST)**: a binary tree in which every node's value is greater than every value in its left subtree and less than every value in its right subtree, so a search follows one root-to-leaf path and an in-order traversal yields the values in sorted order.
- **In-order traversal**: visiting a binary tree's left subtree, then the node, then the right subtree; on a binary search tree it produces the values in sorted order.
- **Lowest common ancestor (LCA)**: the deepest node that has both of two given nodes as descendants, counting a node as its own descendant; found by searching both subtrees in a general binary tree, or by following the values left or right in a BST.
- **Tree DP**: dynamic programming over a tree in which each subtree returns a small tuple of answers, such as the best total with and without its root, and the parent combines its children's tuples; it solves House Robber III and maximum path sum in $O(n)$.
- **AVL tree**: a self-balancing binary search tree that keeps the heights of every node's two subtrees within one of each other by rotating after insertions, which guarantees $O(\log n)$ height and operations.
- **Balance factor**: the height of a node's left subtree minus the height of its right subtree; an AVL tree keeps it in $\{-1, 0, 1\}$, and its sign pattern after an insertion decides between a single and a double rotation.

## In this chapter

- [AVL tree with range queries](example-7.py): a self-balancing binary search tree in Python. Each node tracks its height; after every insertion, the balance factor decides between a single rotation (left or right) and a double rotation, keeping the tree height-balanced. The comparison logic is passed in as a `comparator` with configurable keys, so the same tree can order tuples by any field. The program uses the tree to find the longest subsequence of its input in which each element is at least the previous one and at most 3 more: for each value it queries the best chain ending in the range `[S[i] - 3, S[i]]`, inserts the extended chain, and reconstructs the answer through each node's link to its predecessor. The comments document two known limitations: inserting duplicate values loses them, and `range_max` runs in $O(n)$ rather than the $O(\log n)$ an augmented tree would give. Run it with `python3 05-trees/example-7.py`, optionally followed by integers to use as input; without arguments it uses `[1, 9, 2, 3, 7, 6, 5]`.

The systems chapter builds a production tree over token ids in [the radix tree](../15-systems-data-structures/07-radix-tree.md).
