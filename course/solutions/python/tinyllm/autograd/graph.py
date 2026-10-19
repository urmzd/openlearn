"""tinyllm.autograd.graph (M06.1): the order backward walks a computation graph.

A computation graph is a DAG: every node is a value and `parents(node)` lists
the values it was computed from. Backward starts at the loss (the root) and
must reach a node only after every node that used it, so the node's gradient
is complete before it is passed on. That order is a reverse topological order.

    order = toposort(loss, lambda t: t._parents)   # loss first, leaves last

Contract: contracts/py/tinyllm/autograd/graph.pyi.
"""

from __future__ import annotations

from typing import Callable, Iterable, Iterator, TypeVar

T = TypeVar("T")

_ON_STACK = 1  # entered, some parents still being explored ("gray")
_DONE = 2  # every parent emitted, node emitted ("black")


def toposort(root: T, parents: Callable[[T], Iterable[T]]) -> list[T]:
    """Root first, every node before its parents; the reverse of the
    depth-first post-order over `parents`. Iterative, identity-keyed."""
    # state is keyed by id(node): tensors overload == elementwise and are
    # unhashable, so a set of nodes would fail. Every node we key on stays
    # alive (it sits on the stack or in `post`), so no id is ever reused.
    state: dict[int, int] = {id(root): _ON_STACK}
    # An explicit stack of (node, iterator over its parents) replaces the
    # call stack of a recursive DFS: a 10^5-deep chain would overflow Python's
    # recursion limit (about 1000 frames).
    stack: list[tuple[T, Iterator[T]]] = [(root, iter(parents(root)))]
    post: list[T] = []
    while stack:
        node, pending = stack[-1]
        for p in pending:
            s = state.get(id(p))
            if s is None:
                state[id(p)] = _ON_STACK
                stack.append((p, iter(parents(p))))
                break  # descend first; resume this node's iterator later
            if s == _ON_STACK:
                raise ValueError("toposort: the graph has a cycle reachable from root")
            # _DONE: already emitted through another path (a shared input)
        else:
            # Every parent is emitted, so emitting the node now keeps
            # "parents before node" in the post-order.
            stack.pop()
            state[id(node)] = _DONE
            post.append(node)
    post.reverse()  # node before its parents, root first
    return post
