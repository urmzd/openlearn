"""Course tests for M06.1: toposort (tinyllm/autograd/graph.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/M06.1), and the chapter section it comes from.

The chapter's worked example (section 3) is the graph of
    L = (x * y) + (x + c)      named nodes: L, m = x * y, s = x + c, x, y, c
with parents(L) = [m, s], parents(m) = [x, y], parents(s) = [x, c].
"""

from __future__ import annotations

import graphlib
import os

import pytest
from _lib.pcg32 import PCG32
from tinyllm.autograd.graph import toposort


class Node:
    """A graph node that behaves like a tensor in the one way that matters
    here: `==` is elementwise, so Python makes it unhashable."""

    def __init__(self, name: str, parents: list["Node"] | None = None) -> None:
        self.name = name
        self.parents = parents or []

    def __eq__(self, other):  # like numpy: an array, never a bool
        raise TypeError("Node == Node is elementwise; compare identities with `is`")

    __hash__ = None  # what defining __eq__ does to a class anyway

    def __repr__(self) -> str:
        return self.name


def names(order: list[Node]) -> list[str]:
    return [n.name for n in order]


def hand_graph() -> Node:
    x, y, c = Node("x"), Node("y"), Node("c")
    m = Node("m", [x, y])
    s = Node("s", [x, c])
    return Node("L", [m, s])


def get_parents(n: Node) -> list[Node]:
    return n.parents


def assert_valid(order: list[Node], root: Node) -> None:
    """Root first, each node once, every node before each of its parents."""
    pos = {id(n): i for i, n in enumerate(order)}
    assert len(pos) == len(order), "a node appears twice"
    assert order[0] is root, "the root must come first"
    for n in order:
        for p in n.parents:
            assert id(p) in pos, f"parent {p} of {n} is missing"
            assert pos[id(n)] < pos[id(p)], f"{n} must come before its parent {p}"


def random_dag(rng: PCG32, n: int, max_parents: int) -> list[Node]:
    """Node i draws its parents from nodes with a larger index, so the graph
    is acyclic by construction; node 0 is the root."""
    nodes = [Node(f"n{i}") for i in range(n)]
    for i in range(n - 1):
        k = rng.below(max_parents + 1)
        nodes[i].parents = [nodes[i + 1 + rng.below(n - 1 - i)] for _ in range(k)]
    return nodes


def test_hand_example_exact_order():
    # WHY: the chapter's worked example. The order is fixed, not just valid:
    #      L0.1 accumulates gradients in this order, and float addition is not
    #      associative, so another valid order changes the low bits of every
    #      gradient. Post-order of a DFS that takes parents in list order is
    #      x, y, m, c, s, L; reversed, L, s, c, m, y, x.
    # KIND: unit, smoke
    # CATCHES: s02, s05, s07, m01
    # CHAPTER: M06.1 section 3
    assert names(toposort(hand_graph(), get_parents)) == ["L", "s", "c", "m", "y", "x"]


def test_shared_input_comes_after_every_consumer():
    # WHY: x feeds both m and s. Its gradient is d L/d x = (from m) + (from s),
    #      so backward may only reach x after BOTH have passed their share.
    #      Emitting x on the first visit (pre-order) hands x a half-finished
    #      gradient.
    # KIND: property
    # CATCHES: s02, m02
    # CHAPTER: M06.1 section 2.3
    root = hand_graph()
    order = toposort(root, get_parents)
    assert_valid(order, root)
    assert names(order).index("x") > max(
        names(order).index("m"), names(order).index("s")
    )


def test_uneven_depth_defeats_breadth_first():
    # WHY: breadth-first order looks right on balanced graphs. Here x is a
    #      direct parent of L AND sits three levels down through a -> b, so
    #      BFS emits x right after L, before b, whose parent it is.
    # KIND: boundary
    # CATCHES: s05
    # CHAPTER: M06.1 section 5, Pitfalls
    x = Node("x")
    b = Node("b", [x])
    a = Node("a", [b])
    L = Node("L", [x, a])
    order = toposort(L, get_parents)
    assert_valid(order, L)
    assert names(order) == ["L", "a", "b", "x"]


def test_each_node_once_and_parents_called_once():
    # WHY: a chain of diamonds doubles the number of paths at every level:
    #      12 levels already have 2^12 paths from root to leaf, and a deep
    #      network has hundreds. Without a visited mark the walk is
    #      exponential; with it, each node is entered once and `parents`
    #      (which can be expensive) is called once per node.
    # KIND: property
    # CATCHES: s04, m02
    # CHAPTER: M06.1 section 2.4
    calls: dict[int, int] = {}

    def counting(n: Node) -> list[Node]:
        calls[id(n)] = calls.get(id(n), 0) + 1
        return n.parents

    bottom = Node("leaf")
    top = bottom
    for i in range(12):
        left, right = Node(f"l{i}", [top]), Node(f"r{i}", [top])
        top = Node(f"j{i}", [left, right])
    order = toposort(top, counting)
    assert len(order) == 1 + 3 * 12
    assert_valid(order, top)
    assert set(calls.values()) == {1}, "parents() called more than once for some node"


def test_repeated_parent_appears_once():
    # WHY: y = x * x lists x twice in its parents. It is still one node, and
    #      must appear once in the order (L0.1 then adds both contributions).
    # KIND: boundary
    # CATCHES: s04
    # CHAPTER: M06.1 section 2.4
    x = Node("x")
    y = Node("y", [x, x])
    assert names(toposort(y, get_parents)) == ["y", "x"]


def test_nodes_with_elementwise_eq():
    # WHY: tensors overload == elementwise, which makes them unhashable, and
    #      two different tensors can hold equal values. Visited state must be
    #      keyed by identity (id), never by putting nodes in a set or a dict.
    # KIND: boundary
    # CATCHES: s03
    # CHAPTER: M06.1 section 5, Pitfalls
    order = toposort(hand_graph(), get_parents)
    assert len(order) == 6


def test_deep_chain_no_recursion_error():
    # WHY: a model unrolled over a long sequence (L3.1 truncated BPTT, a
    #      10^5-step loop in a test) gives a computation graph 10^5 nodes
    #      deep. A recursive DFS hits Python's recursion limit near 1000.
    # KIND: boundary
    # CATCHES: s01
    # CHAPTER: M06.1 section 2.5
    leaf = Node("leaf")
    node = leaf
    for i in range(100_000):
        node = Node("c", [node])
    order = toposort(node, get_parents)
    assert len(order) == 100_001
    assert order[0] is node and order[-1] is leaf


def test_single_node():
    # WHY: a loss that is a leaf (a parameter used as the loss directly) has
    #      an empty graph; backward must still visit it.
    # KIND: boundary, smoke
    # CATCHES: s03
    # CHAPTER: M06.1 section 4
    x = Node("x")
    order = toposort(x, get_parents)
    assert len(order) == 1 and order[0] is x


def test_cycle_raises():
    # WHY: a computation graph cannot have a cycle, but L8.7 builds automata
    #      that can; a silent wrong order there is worse than an error.
    #      A node met again while still on the stack closes a cycle.
    # KIND: boundary
    # CATCHES: s06
    # CHAPTER: M06.1 section 2.2
    a = Node("a")
    b = Node("b", [a])
    c = Node("c", [b])
    a.parents = [c]
    with pytest.raises(ValueError):
        toposort(Node("root", [a]), get_parents)


def test_random_dags_valid_and_match_graphlib_on_cycles():
    # WHY: 200 random DAGs from the frozen PCG32: every order is valid. Then
    #      the same graphs with one back edge added: toposort raises exactly
    #      when the stdlib graphlib (an independent implementation) finds a
    #      cycle reachable from the root.
    # KIND: property, differential
    # CATCHES: s02, s04, s05, s06, m02
    # CHAPTER: M06.1 section 2
    rng = PCG32(int(os.environ.get("OL_SEED", "0")), seq=61)
    for _ in range(200):
        n = 2 + rng.below(25)
        nodes = random_dag(rng, n, max_parents=3)
        order = toposort(nodes[0], get_parents)
        assert_valid(order, nodes[0])
        # add an edge from a later node back to an earlier one
        i, j = rng.below(n), rng.below(n)
        lo, hi = min(i, j), max(i, j)
        nodes[hi].parents = nodes[hi].parents + [nodes[lo]]
        reach: dict[int, Node] = {}
        todo = [nodes[0]]
        while todo:
            v = todo.pop()
            if id(v) not in reach:
                reach[id(v)] = v
                todo += v.parents
        ts = graphlib.TopologicalSorter(
            {k: [id(p) for p in v.parents] for k, v in reach.items()}
        )
        try:
            ts.prepare()
            has_cycle = False
        except graphlib.CycleError:
            has_cycle = True
        if has_cycle:
            with pytest.raises(ValueError):
                toposort(nodes[0], get_parents)
        else:
            assert_valid(toposort(nodes[0], get_parents), nodes[0])
