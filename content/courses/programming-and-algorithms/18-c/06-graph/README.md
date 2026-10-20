# Graph as an adjacency list

A surprising number of problems are graph problems in disguise: build systems order tasks by their dependencies, routers find shortest hops, garbage collectors trace reachability. This section builds a directed graph in which each vertex owns a growable array of outgoing edges, and then the three traversals that most graph problems reduce to: breadth-first search for hop distances, depth-first search for reachability, and topological sort for dependency order.

None of the three is long, and each has one bug that survives every small test. BFS that marks too late floods its queue. DFS that recurses overflows the stack on a long path. Topological sort that counts neighbours instead of edges reports a cycle that is not there. You will see each one and the single line that prevents it. The per-vertex edge arrays grow exactly like the [dynamic array](../01-dynamic-array/README.md), and the FIFO and stack disciplines are the only other machinery you need.

> **Learning objectives**
> - Represent a sparse directed graph with per-vertex adjacency arrays in O(V + E) space.
> - Implement BFS that enqueues each vertex at most once, and explain why it yields shortest hop counts only for unit weights.
> - Convert recursive DFS into an iterative one with resumable stack frames.
> - Implement Kahn's topological sort and use its emitted count to detect cycles, including with parallel edges.

## The interface

[`graph.h`](../../../../../practice/build/systems/c/06-graph/graph.h) declares the interface, [`graph.c`](../../../../../practice/build/systems/c/06-graph/graph.c) is the solution, and [`main.c`](../../../../../practice/build/systems/c/06-graph/main.c) is the test suite.

| Function | Does |
|----------|------|
| `graph_init(g, n)` | `n` vertices, no edges |
| `graph_add_edge(g, from, to)` | One directed edge. Duplicates allowed |
| `graph_degree(g, v)` | Out-degree |
| `graph_bfs(g, src, dist)` | Hop counts, `GRAPH_UNREACHABLE` where no path exists |
| `graph_dfs(g, src, order, &count)` | First-visit order, iteratively |
| `graph_toposort(g, order)` | Topological order, or -1 if cyclic |

Undirected graphs are built by adding both directions. The tests do that explicitly rather than having the API hide it, because forgetting the reverse edge is one of the most common graph bugs and hiding it teaches nothing. DFS must survive a 100,000-vertex path, so it cannot recurse.

To build and run the tests from this directory:

```bash
cc -std=c11 -Wall -Wextra -O2 main.c graph.c -o graph && ./graph
# ok  c/06-graph  200173 checks passed
```

## Breadth-first search and when to mark

Mark vertices when you enqueue them, not when you dequeue them. This is the BFS bug that survives every small test. If you only mark on dequeue, a vertex with three unprocessed predecessors gets pushed three times before it is first examined. The traversal still terminates and still reports correct distances, but the queue can hold far more than $n$ entries, which overflows the fixed-size queue this implementation allocates and turns a dense graph quadratic.

BFS gives shortest paths only because every edge costs the same. The moment edges carry weights, the first time you reach a vertex is no longer the cheapest time, and you need Dijkstra's priority queue instead. `test_bfs_takes_the_short_route` adds the shortcut edge *last* specifically so that an implementation which overwrites an already-known distance reports 3 instead of 1.

## Depth-first search without recursion

DFS must be iterative here. Recursive DFS is prettier, but its stack depth is the length of the longest path, so a path graph of any real size overflows. The iterative version needs an explicit stack whose frames carry *how many neighbours have been consumed*, because that is the state the call stack was holding for you.

## Topological order and cycles

Kahn's algorithm gets cycle detection for free. A vertex inside a cycle never reaches in-degree zero, so it never enters the queue and never comes out. If fewer than $n$ vertices are emitted, the remainder are in or behind cycles. There is no colouring, no separate pass, and no recursion.

The subtlety is that in-degree counts edges, not neighbours. Three parallel edges from 0 to 1 give vertex 1 an in-degree of 3, so processing vertex 0 must decrement three times. Decrementing once per distinct neighbour leaves the count stuck above zero and reports a cycle in a graph that plainly has none. `test_toposort_with_parallel_edges` exists solely to catch this.

## Deriving the design

The whole implementation follows from six observations.

Store what the traversals read. Every traversal asks one question: "what are `v`'s outgoing edges?". An array of $n$ adjacency lists answers it in O(degree), with total space O(V + E). An adjacency matrix would cost O(V²) space and O(V) per question.

Each list grows like the [dynamic array](../01-dynamic-array/README.md). Doubling capacity makes `graph_add_edge` O(1) amortised. Zero-initialising the array of lists (`calloc`) makes every list start as `{NULL, 0, 0}`, so the first `realloc` acts as `malloc`.

BFS explores in order of distance. A FIFO queue processes all vertices at distance $d$ before any at $d + 1$. With unit weights, the first time a vertex is discovered is therefore along a shortest path, so its distance is set once and never revised.

Discovery is the moment to mark. Setting `dist[w]` when `w` is enqueued both records the answer and prevents a second enqueue. Every vertex enters the queue at most once, so an $n$-slot array suffices as the queue.

Iterative DFS needs a frame per vertex, not just a vertex. Recursive DFS resumes a vertex's neighbour loop where it left off. An explicit stack must hold the same state: the vertex plus the index of the next edge to try.

A DAG always has a vertex with no incoming edges. Emit it, delete its edges (decrement each target's in-degree once per edge), and repeat. If the graph has a cycle, the vertices on it never reach in-degree zero, so fewer than $n$ are emitted: cycle detection is the count at the end.

## Building it step by step

Construction. `calloc` is what makes the lazy per-list allocation work:

```c
int graph_init(Graph *g, size_t n) {
  g->n = n;
  g->adj = NULL;
  if (n == 0) return 0;

  /* calloc, not malloc: every AdjList must start {NULL, 0, 0}, or the first
   * graph_add_edge reallocs a garbage pointer. */
  g->adj = calloc(n, sizeof(AdjList));
  if (!g->adj) return -1;
  return 0;
}

int graph_add_edge(Graph *g, size_t from, size_t to) {
  if (from >= g->n || to >= g->n) return -1;   /* reject out-of-range ids */

  AdjList *a = &g->adj[from];
  if (a->len == a->cap) {
    size_t cap = a->cap ? a->cap * 2 : 4;
    if (cap > SIZE_MAX / sizeof(size_t)) return -1;
    size_t *grown = realloc(a->to, cap * sizeof(size_t));
    if (!grown) return -1;   /* old list intact */
    a->to = grown;
    a->cap = cap;
  }
  a->to[a->len++] = to;
  return 0;
}
```

BFS. `dist` doubles as the visited set:

```c
  for (size_t i = 0; i < g->n; i++) dist[i] = GRAPH_UNREACHABLE;
  /* ... */
  /* Every vertex is enqueued at most once, so n slots suffice and the queue
   * never wraps: head and tail only move forward. */
  size_t *queue = malloc(g->n * sizeof(size_t));
  if (!queue) return -1;
  size_t head = 0, tail = 0;

  dist[src] = 0;
  queue[tail++] = src;

  while (head < tail) {
    size_t v = queue[head++];
    const AdjList *a = &g->adj[v];
    for (size_t i = 0; i < a->len; i++) {
      size_t w = a->to[i];
      /* Mark on ENQUEUE. Marking on dequeue would let w be pushed once per
       * predecessor and overflow this n-slot queue. */
      if (dist[w] == GRAPH_UNREACHABLE) {
        dist[w] = dist[v] + 1;
        queue[tail++] = w;
      }
    }
  }

  free(queue);
```

Iterative DFS with resumable frames:

```c
  typedef struct { size_t v, next_edge; } Frame;
  Frame *stack = malloc(g->n * sizeof(Frame));   /* each vertex pushed once */
  if (!stack) { free(seen); return -1; }
  size_t top = 0;

  seen[src] = 1;
  order[(*count)++] = src;
  stack[top++] = (Frame){src, 0};

  while (top > 0) {
    Frame *f = &stack[top - 1];          /* peek, do not pop */
    const AdjList *a = &g->adj[f->v];

    if (f->next_edge >= a->len) { top--; continue; }   /* v is finished */

    size_t w = a->to[f->next_edge++];    /* advance this frame's cursor */
    if (seen[w]) continue;

    /* First visit: record it and descend, exactly where the recursive
     * version would make its call. */
    seen[w] = 1;
    order[(*count)++] = w;
    stack[top++] = (Frame){w, 0};
  }
```

Because a vertex is pushed only on its first visit, the stack never holds more
than `n` frames, and neighbours are explored in insertion order, matching the
recursive formulation.

Kahn's algorithm:

```c
  size_t *indeg = calloc(g->n, sizeof(size_t));
  if (!indeg) return -1;
  for (size_t v = 0; v < g->n; v++) {
    for (size_t i = 0; i < g->adj[v].len; i++) indeg[g->adj[v].to[i]]++;
  }
  /* ... queue every vertex with in-degree 0 ... */

  size_t emitted = 0;
  while (head < tail) {
    size_t v = queue[head++];
    order[emitted++] = v;
    for (size_t i = 0; i < g->adj[v].len; i++) {
      size_t w = g->adj[v].to[i];
      /* One decrement per EDGE. A parallel edge counted twice above must be
       * paid back twice, or w never reaches zero. */
      if (--indeg[w] == 0) queue[tail++] = w;
    }
  }
  /* ... free ... */

  /* A vertex on a cycle never reaches in-degree 0, so it is never emitted. */
  return emitted == g->n ? 0 : -1;
```

The full solution is in [`graph.c`](../../../../../practice/build/systems/c/06-graph/graph.c).

## Why this design

BFS, DFS, and toposort are each O(V + E) time and O(V) extra space. `graph_add_edge` is O(1) amortised, and the graph itself is O(V + E).

Adjacency lists beat a matrix for this workload. A matrix answers "is there an edge u→v?" in O(1) but costs O(V²) memory and makes every traversal O(V²). Most real graphs are sparse, so lists win.

The explicit stack beats recursion because recursion is limited by the thread's stack (often 8 MB, sometimes far less), so its depth is a hidden, input-dependent limit. The heap stack's size is bounded by $n$ and allocated up front.

Kahn's algorithm is chosen over DFS-based toposort. DFS post-order also yields a topological order and detects cycles with three colours, but needs recursion or another resumable-frame stack. Kahn's algorithm is iterative by nature and reports a cycle with a single count.

The accepted trade is that duplicate edges are stored, not merged. That keeps `add_edge` O(1) and pushes deduplication to callers who need it, at the cost that in-degree logic must count edges.

## What can go wrong

Each traversal has its characteristic failure, and the tests are built to expose each one.

| Mistake | Consequence | Caught by |
|---------|-------------|-----------|
| Marking on dequeue in BFS | A vertex is enqueued once per predecessor and overflows the $n$-slot queue | Dense-graph tests |
| Overwriting an already-set distance | The shortcut added last is ignored; reports 3 instead of 1 | `test_bfs_takes_the_short_route` |
| Recursive DFS | Stack overflow on a 100,000-vertex path | The long-path DFS test |
| Decrementing in-degree once per neighbour | Parallel edges leave counts above zero; a false cycle | `test_toposort_with_parallel_edges` |
| `malloc` instead of `calloc` for the lists | The first `add_edge` reallocs a garbage pointer | Sanitizer runs |
| Forgetting the reverse edge for an undirected graph | Half the graph is unreachable | Tests build both directions explicitly |

## Where this leads

The natural next step is Dijkstra, using the [binary heap](../04-binary-heap/README.md), and it immediately runs into the awkward part: you need `decrease_key`, and the array-backed heap has no way to find an element without a side index. After that, Tarjan's strongly connected components, which needs a recursive formulation or a considerably more intricate iterative one.

## Key terms

- **Adjacency list**: a per-vertex list of outgoing edges; O(V + E) space for the whole graph.
- **Breadth-first search (BFS)**: traversal in order of hop distance using a FIFO queue.
- **Depth-first search (DFS)**: traversal that follows one path as far as possible before backtracking, using a stack.
- **In-degree**: the number of edges entering a vertex, counting parallel edges separately.
- **Topological order**: an ordering of a DAG's vertices in which every edge points forward.
- **Kahn's algorithm**: topological sort by repeatedly emitting vertices whose in-degree has dropped to zero.

## Summary

- Each vertex owns a doubling array of targets, so adding an edge is O(1) amortised and the graph is O(V + E).
- BFS marks on enqueue, so each vertex enters the queue once and an $n$-slot array suffices; distances are shortest only for unit weights.
- Iterative DFS keeps a frame of (vertex, next edge index), reproducing what the recursive call stack held.
- Kahn's algorithm decrements in-degree once per edge, and fewer than $n$ emitted vertices means a cycle.

## Review questions

1. Why can the BFS queue be a plain array of `n` slots with no wrap-around?

   <details><summary>Answer</summary>

   Each vertex is enqueued at most once, because it is marked (its `dist` set)
   at the moment it is enqueued. So at most `n` pushes happen in total, and
   `tail` never exceeds `n`.

   </details>

2. Why does BFS never need to update a distance it has already set?

   <details><summary>Answer</summary>

   The FIFO queue processes vertices in non-decreasing order of distance. The
   first time `w` is discovered, it is from a vertex at the smallest distance
   that reaches it, so `dist[v] + 1` is already minimal. That relies on every
   edge having the same weight.

   </details>

3. What state does each DFS `Frame` hold, and why is the vertex alone not
   enough?

   <details><summary>Answer</summary>

   The vertex and the index of its next unexplored edge. When the traversal
   returns to a vertex after finishing a child, it must continue with the next
   neighbour rather than starting over; the recursive version kept that loop
   index in the call frame.

   </details>

4. A graph has vertices 0, 1, 2 with edges 0→1, 1→2, 2→1. What does
   `graph_toposort` return, and how many vertices are emitted?

   <details><summary>Answer</summary>

   It returns -1. Vertex 0 has in-degree 0 and is emitted; that drops vertex 1
   from 2 to 1. Vertices 1 and 2 each keep an incoming edge from the other, so
   neither reaches zero. One vertex is emitted, fewer than 3.

   </details>
