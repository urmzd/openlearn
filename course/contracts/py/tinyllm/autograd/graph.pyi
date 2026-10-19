# contracts/py/tinyllm/autograd/graph.pyi (M06.1)
# chapter: content/courses/mathematics-for-computing/07-discrete-math-2/01-graphs-dags-and-topological-sort.md
#
# A computation graph is a directed acyclic graph (DAG): each node is a value,
# and `parents(node)` lists the nodes it was computed from (an op's inputs).
# Backward visits the graph from the loss toward the leaves, and a node's
# gradient is complete only after every node that consumed it has passed its
# share back. toposort returns exactly that order.
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")

def toposort(root: T, parents: Callable[[T], Iterable[T]]) -> list[T]:
    """Every node reachable from root through `parents`, each exactly once,
    root first, and every node before each of its parents (a reverse
    topological order of the computation, ready for backward).

    The order is fixed, not just valid: it is the reverse of the depth-first
    post-order that starts at root and visits each node's parents in the
    order `parents` yields them. Nodes are told apart by identity (`id`),
    never by `==` or `hash`, so values with an elementwise `__eq__` (tensors)
    work. `parents` is called exactly once per reachable node. The traversal
    is iterative: a chain 10^5 nodes deep raises no RecursionError.
    ValueError when a cycle is reachable from root."""
