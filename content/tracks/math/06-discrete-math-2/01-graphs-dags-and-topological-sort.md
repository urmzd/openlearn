<!-- ol:module M06.1 -->
# Graphs, DAGs, and an iterative topological sort

## Overview

| | |
|---|---|
| **Module** | `M06.1` · build · Python · Pass 2 · 2 to 3 h |
| **You build** | `python/tinyllm/autograd/graph.py`: `toposort(root, parents)`, the order in which backward visits a computation graph |
| **Contract** | [`course/contracts/py/tinyllm/autograd/graph.pyi`](../../../../course/contracts/py/tinyllm/autograd/graph.pyi) |
| **Tests** | `course/tests/M06.1/test_graph.py` (what they check: section 4) |
| **Needs** | nothing to build first. Reading: [Discrete Math 1](../05-discrete-math-1/) (sets, relations, induction) |
| **Used by** | `M08.2` scalar reverse-mode autodiff walks this order · later `L0.1` (`Tensor.backward`) and `L8.7` (the automata of constrained decoding) |
| **Milestone** | `MS-P2` (the foundations gate) |
| **Optional depth** | Cormen, Leiserson, Rivest, and Stein, *Introduction to Algorithms*, sections 20.3 and 20.4 (depth-first search, topological sort); Sedgewick and Wayne, *Algorithms*, section 4.2; Kahn, "Topological sorting of large networks" (1962) |

## Key Takeaways

- A **topological order** of a directed graph lists every vertex before the vertices it points to; it exists exactly when the graph has **no directed cycle** (a DAG) (`test_cycle_raises`).
- Backward needs every node **before its inputs**: a value's gradient is complete only after every consumer has added its share. The **reverse of a depth-first post-order** is such an order (`test_shared_input_comes_after_every_consumer`).
- Breadth-first order and pre-order look right on small graphs and are wrong on graphs of uneven depth (`test_uneven_depth_defeats_breadth_first`).
- One **visited mark** per node makes the walk linear in the size of the graph instead of exponential in its depth (`test_each_node_once_and_parents_called_once`).
- The walk must be **iterative** and keyed by **identity**: a 10^5-deep chain breaks a recursive one, and tensors cannot go in a set (`test_deep_chain_no_recursion_error`, `test_nodes_with_elementwise_eq`).

## How to work this chapter

```bash
ol start M06.1              # stubs python/tinyllm/autograd/graph.py into your repo
ol tests M06.1              # read the test catalog first: rung R0, you write no tests here
ol check M06.1              # exit code is the verdict
ol diff  M06.1              # after passing: your code against the reference
```

---

## 1. Why now

Pass 2 builds automatic differentiation: first on scalars (`M08.2`), then on tensors (`L0.1`). A loss is computed from the parameters through hundreds of operations, and backward has to push the derivative of the loss back through all of them to every parameter. Each operation can only pass a gradient to its inputs once its own gradient is complete, and a value used twice (a weight shared by every time step, an `x` in `x * y + x`) receives contributions from several places. Visit the nodes in the wrong order and a shared weight is updated with half its gradient: the loss goes down a little, then stalls, and nothing crashes. Visit them recursively and the first recurrent model you unroll over 1000 tokens dies with `RecursionError`. This module defines graphs and DAGs from the beginning and builds the one function that fixes the order: `toposort(root, parents)`.

## 2. Principles

| Symbol | Meaning | Type |
|---|---|---|
| $G = (V, E)$ | a directed graph: a set of vertices $V$ and a set of edges $E$ | |
| $u \to v$ | an edge from $u$ to $v$, an element $(u, v) \in E$ | |
| $n = \lvert V \rvert$, $m = \lvert E \rvert$ | number of vertices and of edges | `int` |
| $r$ | the root: the loss, where backward starts | a node |
| $\mathrm{parents}(v)$ | the values $v$ was computed from: the edges $v \to p$ | iterable of nodes |
| $\pi$ | an order of the vertices, a list in which each appears once | `list` |
| $\pi(v)$ | the position of $v$ in $\pi$ | `int` |

### 2.1 Directed graphs

A **directed graph** is a set of vertices $V$ and a set of ordered pairs $E \subseteq V \times V$ called edges. The edge $(u, v)$ is written $u \to v$: it leaves $u$ and enters $v$. A **path** is a sequence $v_0 \to v_1 \to \dots \to v_k$ of edges; a **cycle** is a path with $k \ge 1$ that ends where it started. A vertex $v$ is **reachable** from $r$ when some path leads from $r$ to $v$. In code, a graph is usually stored as **adjacency lists**: for each vertex, the list of vertices its edges enter. `parents` is exactly that: a function from a node to the list of its out-neighbours.

### 2.2 DAGs and topological orders

A **directed acyclic graph** (DAG) is a directed graph with no cycle. A **topological order** is a list $\pi$ of the vertices in which every edge points forward: $u \to v$ implies $\pi(u) < \pi(v)$.

*A graph has a topological order if and only if it is a DAG.* If there were a cycle $v_0 \to v_1 \to \dots \to v_0$, the order would need $\pi(v_0) < \pi(v_1) < \dots < \pi(v_0)$, which is impossible. Conversely, a finite DAG has a vertex with no incoming edge (follow edges backwards from any vertex; without a cycle you must stop, and where you stop nothing points in). Put that vertex first, delete it, and the rest is still a DAG, so by induction on $n$ it has a topological order too. That proof is also an algorithm (Kahn's): repeatedly take a vertex with no remaining incoming edge.

### 2.3 Computation graphs and the backward order

A **computation graph** has one node per value. The edges go from each value to the values it was computed from, $v \to p$ for every $p \in \mathrm{parents}(v)$: $L = m + s$ has edges $L \to m$ and $L \to s$. The loss $r = L$ reaches every node that matters.

Backward applies the chain rule. If $p$ is used by consumers $v_1, \dots, v_k$, then

$$\frac{\partial L}{\partial p} = \sum_{i=1}^{k} \frac{\partial L}{\partial v_i} \frac{\partial v_i}{\partial p},$$

so $p$ may pass its gradient on only after every consumer $v_i$ has contributed. In edge terms: every edge $v \to p$ needs $v$ processed before $p$. That is precisely a topological order of the graph with edges $v \to p$, starting at the root. (Seen from the forward computation, where inputs come first, it is the *reverse* topological order; same thing, opposite arrows.)

### 2.4 Depth-first search and the post-order

**Depth-first search** (DFS) from $r$ follows one edge as deep as it can go before trying the next. Each vertex is in one of three states: **white** (not seen), **gray** (entered, still exploring its edges, on the stack), **black** (finished). The **post-order** lists vertices in the order they turn black.

*Claim: in the post-order, every vertex comes after all the vertices it points to.* Take an edge $v \to p$ and look at the moment DFS examines it, while $v$ is gray. If $p$ is white, DFS enters $p$ and finishes it before it can finish $v$. If $p$ is black, it already finished. If $p$ is gray, $p$ is an ancestor of $v$ on the current path, so there is a path $p \to \dots \to v$ and with the edge $v \to p$ a cycle: in a DAG this cannot happen, and meeting a gray vertex is exactly how DFS detects a cycle. In both remaining cases $p$ turns black before $v$.

Reversing the post-order therefore puts every vertex before the vertices it points to: the root first, each node before its parents. Every vertex is entered once (the colour test), and each edge is examined once, so the whole walk costs $O(n + m)$. Without the black state, a vertex reached along two paths is explored twice, and a stack of $d$ diamonds (each node with two parents that share one parent below) has $2^d$ paths.

### 2.5 Iterative DFS

Recursion keeps the gray path on the call stack, and CPython stops at about 1000 nested calls (`sys.getrecursionlimit()`). A 1000-step unrolled recurrence is a path 1000 nodes long. The iterative version keeps the path on an explicit list of pairs `(node, iterator over its parents)`:

1. Push `(r, iter(parents(r)))` and mark `r` gray.
2. Look at the top pair `(v, it)`. Take the next parent `p` from `it`:
   white: mark it gray, push `(p, iter(parents(p)))`, and go back to 2 (descend);
   gray: a cycle, raise;
   black: skip it and take the next parent.
3. When `it` is exhausted: pop, mark `v` black, append `v` to the post-order.

Keeping the iterator on the stack is what makes this the same traversal as the recursive one: when a child finishes, the parent resumes exactly where it left off. `parents(v)` is called once per node, when the node is pushed.

### 2.6 Identity, not equality

Tensors overload `==` to compare elementwise and return an array, and a Python class that defines `__eq__` without `__hash__` becomes unhashable. Two different tensors can also hold equal values and still be different nodes. So visited state is a dictionary keyed by `id(node)`, the object's identity. An `id` is only unique among objects that are alive; every node we record stays alive because it sits on the stack or in the output list.

### 2.7 One fixed order

A DAG usually has many topological orders. `toposort` returns one specific order: the reverse post-order of the DFS that visits parents in the order `parents` lists them. That matters because backward *adds* contributions in this order, and floating-point addition is not associative: a different valid order changes the last bits of the gradients, and two runs of the same seed stop being bit-identical (P11). Fixing the order here keeps `L0.5`'s "same seed, same run" promise.

## 3. Worked example by hand

$L = (x \cdot y) + (x + c)$ with intermediate nodes $m = x \cdot y$ and $s = x + c$:

```
parents(L) = [m, s]      parents(m) = [x, y]      parents(s) = [x, c]
parents(x) = parents(y) = parents(c) = []
```

The iterative DFS, one row per step (stack written bottom to top, with what each iterator has left):

| Step | Top of stack | Next parent | Action | Post-order so far |
|---|---|---|---|---|
| 1 | `L` [m, s] | `m` (white) | push `m` | |
| 2 | `m` [x, y] | `x` (white) | push `x` | |
| 3 | `x` [] | none | pop, emit `x` | x |
| 4 | `m` [y] | `y` (white) | push `y` | x |
| 5 | `y` [] | none | pop, emit `y` | x, y |
| 6 | `m` [] | none | pop, emit `m` | x, y, m |
| 7 | `L` [s] | `s` (white) | push `s` | x, y, m |
| 8 | `s` [x, c] | `x` (black) | skip | x, y, m |
| 9 | `s` [c] | `c` (white) | push `c` | x, y, m |
| 10 | `c` [] | none | pop, emit `c` | x, y, m, c |
| 11 | `s` [] | none | pop, emit `s` | x, y, m, c, s |
| 12 | `L` [] | none | pop, emit `L` | x, y, m, c, s, L |

Reversed: **`[L, s, c, m, y, x]`**. Check every edge: `L` before `m` and `s`; `m` before `x` and `y`; `s` before `x` and `c`. This is `test_hand_example_exact_order`.

Now run backward in that order with $x = 2$, $y = 3$, $c = 1$ (so $m = 6$, $s = 3$, $L = 9$), starting from $\partial L / \partial L = 1$:

| Visit | Gradient complete | Passes on |
|---|---|---|
| `L` | 1 | to `s`: 1, to `m`: 1 |
| `s` | 1 | to `x`: 1, to `c`: 1 |
| `c` | 1 | |
| `m` | 1 | to `x`: $y = 3$, to `y`: $x = 2$ |
| `y` | 2 | |
| `x` | 1 + 3 = **4** | |

$\partial L / \partial x = y + 1 = 4$: correct. A **pre-order** (emit on entry) gives `[L, m, x, y, s, c]`: `x` is visited right after `m`, holding only 3, and passes 3 on. That is the bug `test_shared_input_comes_after_every_consumer` exists for.

## 4. The interface

```python
def toposort(root: T, parents: Callable[[T], Iterable[T]]) -> list[T]:
    """Every node reachable from root, each once, root first, every node before
    each of its parents: the reverse DFS post-order, parents taken in the order
    `parents` yields them. Keyed by id(); iterative; parents called once per node.
    ValueError when a cycle is reachable from root."""
```

`M08.2` and `L0.1` call it as `toposort(loss, lambda v: v._parents)` and then walk the list, each node adding its local derivatives into its parents' gradients.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_exact_order` | unit, smoke | section 3's order exactly | bit-identical gradients across runs |
| `test_shared_input_comes_after_every_consumer` | property | `x` after both `m` and `s` | complete gradients for shared weights |
| `test_uneven_depth_defeats_breadth_first` | boundary | a parent that is also a grandparent comes last | breadth-first order is not enough |
| `test_each_node_once_and_parents_called_once` | property | 12 stacked diamonds: 37 nodes, `parents` once each | linear time on deep networks |
| `test_repeated_parent_appears_once` | boundary | `x * x` lists `x` twice, the order once | squares and self-products |
| `test_nodes_with_elementwise_eq` | boundary | nodes whose `==` raises and whose hash is `None` | tensors as nodes in `L0.1` |
| `test_deep_chain_no_recursion_error` | boundary | a chain of 100 001 nodes | long unrolled recurrences in `L3.1` |
| `test_single_node` | boundary, smoke | a leaf as the root | a parameter used directly as the loss |
| `test_cycle_raises` | boundary | `ValueError` on a reachable cycle | automata in `L8.7` |
| `test_random_dags_valid_and_match_graphlib_on_cycles` | property, differential | 200 random DAGs valid; with a back edge added, raises exactly when the stdlib `graphlib` finds a cycle | an independent implementation agrees |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| a recursive DFS | `RecursionError` once the graph is about 1000 nodes deep | `test_deep_chain_no_recursion_error` (mutant `s01`) |
| emitting a node when it is first entered (pre-order) | a shared input passes on a partial gradient | `test_shared_input_comes_after_every_consumer` (mutant `s02`) |
| putting nodes in a set or using them as dict keys | `TypeError: unhashable type` on the first tensor | `test_nodes_with_elementwise_eq` (mutant `s03`) |
| no mark for finished nodes | duplicates in the order, exponential time on diamonds | `test_each_node_once_and_parents_called_once` (mutant `s04`) |
| breadth-first order | invalid as soon as one input sits at two depths | `test_uneven_depth_defeats_breadth_first` (mutant `s05`) |
| treating a gray node like a black one | a cyclic automaton gets a silently wrong order | `test_cycle_raises` (mutant `s06`) |
| visiting parents in reverse | still valid, but a different order and different gradient bits | `test_hand_example_exact_order` (mutant `s07`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | [Discrete Math 1](../05-discrete-math-1/) | relations, induction, and proof by contradiction (reading) |
| Forward | `M08.2` | scalar reverse-mode autodiff: `backward` walks `toposort(loss, parents)` |
| Forward | `L0.1` | `Tensor.backward` does the same over tensors, accumulating gradients in this order |
| Forward | `L8.7` | regex and JSON-schema automata are graphs; reachability and cycles come from this traversal |
| Forward | `M06.2` | tries are trees, the simplest DAGs (reading) |

If you skip this module, `ol check M08.2` stops with `BLOCKED ... needs M06.1`: build it, or pass `--ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `toposort` over `parents` | PyTorch autograd engine | counts each node's dependencies first, then runs a ready queue (Kahn's algorithm) across devices and threads | `torch/csrc/autograd/engine.cpp`, `compute_dependencies` |
| one fixed order | JAX | a traced program (a jaxpr) is already a topologically ordered list, so the transpose walks it backwards | `jax/_src/interpreters/ad.py`, `backward_pass` |
| cycle check | Python `graphlib.TopologicalSorter` | Kahn's algorithm with incremental `get_ready`/`done` for parallel schedulers | CPython `Lib/graphlib.py` |
| DAG of tasks | workflow engines (Airflow, Temporal child workflows) | the same order schedules jobs whose inputs are other jobs' outputs | your `dur.*` modules |
