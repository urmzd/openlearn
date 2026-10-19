# C++ 09: Graph algorithms with concepts

**Concepts:** C++20 concepts, ranges, generic algorithms
**Difficulty:** ⭐⭐⭐

BFS, DFS, shortest path, and Dijkstra written against a *concept* rather than a
class. The point is not the graph algorithms; it is that the same code runs on
three unrelated types that share no base class and do not know about each
other. The solution is the header [`graph.hpp`](graph.hpp);
[`main.cpp`](main.cpp) is the test suite and defines the three graph types.

## The problem

| Concept | Requires |
|---------|----------|
| `Graph<G>` | `G::vertex_type`, hashable and equality-comparable, plus `g.neighbours(v)` returning an input range |
| `WeightedGraph<G>` | `Graph<G>` and `g.weight(a, b)` convertible to `double` |

| Algorithm | Constrained on | Returns |
|-----------|----------------|---------|
| `bfs(g, src)` | `Graph` | hop counts, as `unordered_map<V, size_t>` |
| `dfs_order(g, src)` | `Graph` | first-visit order |
| `shortest_path(g, src, dst)` | `Graph` | `optional<vector<V>>` |
| `shortest_distances(g, src)` (Dijkstra) | `WeightedGraph` | `unordered_map<V, double>` |

The test types are an adjacency list, a 300×300 grid whose neighbours are
computed on demand, and a weighted graph with string vertices.

## Build and run

From this directory:

```bash
c++ -std=c++20 -Wall -Wextra -O2 main.cpp -o graph && ./graph
# ok  cpp/09-graph-concepts  45 checks passed
```

## What to notice

**The concept moves the error to the call site.** An unconstrained template
that fails deep inside its body produces a page of instantiation backtrace.
`template <Graph G>` rejects the call itself and says which requirement was not
met. The tests assert this with `static_assert(!Graph<NotAGraph>)`.

**The grid is the case that justifies the whole approach.** `Grid` stores no
adjacency structure at all: `neighbours` computes the four cardinal directions
on demand. A 300×300 grid has 90,000 vertices and around 360,000 edges that
never exist in memory. With a concept, `Grid` simply satisfies it.

**`WeightedGraph` refining `Graph` is subsumption, and it does real work.**
Dijkstra is callable only on graphs that report weights, so passing an
unweighted one is a clean overload-resolution failure. Concepts also order
overloads by specificity, which is what makes constrained templates compose.

**BFS and Dijkstra answer different questions, and the test makes them
disagree on purpose.** In the weighted graph, `start → b` is one hop but costs
5, while `start → a → b` is two hops and costs 2. BFS reports 1, Dijkstra
reports 2.0, and both are right.

**Lazy deletion beats decrease-key in practice.** `std::priority_queue` has no
way to update an entry's priority, so this pushes a new entry and skips stale
ones on pop. The [C binary heap exercise](../../c/04-binary-heap/) discusses the
other side of that trade.

**Mark on enqueue, not on dequeue.** `try_emplace` returning whether it
inserted does the check and the mark in one lookup. This is the same bug as in
the [C version](../../c/06-graph/), and the idiomatic C++ spelling makes it
harder to get wrong.

**Watch for macro commas again.** `CHECK(dist[Cell{0, 0}] == 0, "...")` does not
compile: the preprocessor reads the comma inside the braced initialiser as an
argument separator. Extra parentheses around the condition fix it.

## The solution, step by step

### From first principles

1. **Write down what the algorithms actually use.** BFS needs: a vertex type it
   can copy, compare, and hash (to keep a visited map), and a way to list a
   vertex's neighbours. Nothing else. That list *is* the `Graph` concept.
2. **Ask for the weakest range that works.** The algorithms only iterate
   neighbours once, front to back, so `std::ranges::input_range` is enough. That
   lets a type return a vector, a view, or anything else iterable.
3. **Dijkstra needs one more thing**, edge weights, so `WeightedGraph` is
   `Graph` plus `weight(v, w)`. Building it on `Graph` lets the compiler know
   every weighted graph is a graph.
4. **Sparse or implicit graphs mean hash maps, not arrays.** Vertices are
   arbitrary types (cells, strings), not `0..n-1`, so distances and the visited
   set live in `unordered_map` / `unordered_set` keyed by vertex.
5. **BFS: mark on enqueue** with a single `try_emplace`, as in the C version.
6. **Dijkstra with lazy deletion.** Push `(distance, vertex)` whenever a shorter
   distance is found; when a popped entry is worse than the recorded distance,
   it is stale and skipped. Correct for non-negative weights.

### The code, annotated

The concepts:

```cpp
template <typename G>
concept Graph = requires(const G& g, typename G::vertex_type v) {
  typename G::vertex_type;
  requires std::equality_comparable<typename G::vertex_type>;
  // Hashable, because the algorithms keep unordered_map/set of vertices.
  requires requires(typename G::vertex_type x) { std::hash<typename G::vertex_type>{}(x); };
  // Any input range: a vector, or a range computed on demand.
  { g.neighbours(v) } -> std::ranges::input_range;
};

template <typename G>
concept WeightedGraph = Graph<G> && requires(const G& g, typename G::vertex_type v) {
  { g.weight(v, v) } -> std::convertible_to<double>;
};
```

BFS:

```cpp
template <Graph G>
auto bfs(const G& g, typename G::vertex_type src)
    -> std::unordered_map<typename G::vertex_type, std::size_t> {
  using V = typename G::vertex_type;
  std::unordered_map<V, std::size_t> dist;   // doubles as the visited set
  std::queue<V> frontier;

  dist[src] = 0;
  frontier.push(src);

  while (!frontier.empty()) {
    V v = frontier.front();
    frontier.pop();
    for (const V& w : g.neighbours(v)) {
      // Insert-if-absent and test in one lookup: marks on ENQUEUE.
      if (dist.try_emplace(w, dist[v] + 1).second) frontier.push(w);
    }
  }
  return dist;
}
```

Iterative DFS. It pushes neighbours in reverse so the first neighbour is popped
first, matching recursive order, and skips already-visited vertices when popped:

```cpp
  std::vector<V> stack{src};
  while (!stack.empty()) {
    V v = stack.back();
    stack.pop_back();
    if (!seen.insert(v).second) continue;  // a vertex can be pushed more than once
    order.push_back(v);

    std::vector<V> ns;                       // collect: input ranges are one-pass
    for (const V& w : g.neighbours(v)) ns.push_back(w);
    for (auto it = ns.rbegin(); it != ns.rend(); ++it) {
      if (!seen.contains(*it)) stack.push_back(*it);
    }
  }
```

Neighbours are copied into `ns` because an input range cannot be iterated
backwards. Unlike the C version's frame-per-vertex stack, this one may hold a
vertex several times (once per unvisited predecessor), so it is bounded by
edges rather than vertices.

`shortest_path` is BFS with a `parent` map; when it discovers `dst`, it walks
the parents back to `src` and reverses. Dijkstra:

```cpp
template <WeightedGraph G>
auto shortest_distances(const G& g, typename G::vertex_type src)
    -> std::unordered_map<typename G::vertex_type, double> {
  using V = typename G::vertex_type;
  using Entry = std::pair<double, V>;
  // Min-heap on distance.
  std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> pq;
  std::unordered_map<V, double> dist;

  dist[src] = 0.0;
  pq.emplace(0.0, src);

  while (!pq.empty()) {
    auto [d, v] = pq.top();
    pq.pop();

    // Stale entry: v was improved after this was pushed.
    if (auto it = dist.find(v); it != dist.end() && d > it->second) continue;

    for (const V& w : g.neighbours(v)) {
      double nd = d + g.weight(v, w);
      auto it = dist.find(w);
      if (it == dist.end() || nd < it->second) {   // relax
        dist[w] = nd;
        pq.emplace(nd, w);                         // push again, no decrease-key
      }
    }
  }
  return dist;
}
```

Full solution: [`graph.hpp`](graph.hpp).

### Why this solution

- **Cost.** BFS and `shortest_path` O(V + E) expected (hash operations are O(1)
  on average). DFS O(V + E). Dijkstra O(E log E) with lazy deletion, since the
  queue can hold one entry per relaxation.
- **Concepts versus a base class.** A virtual `neighbours()` would force a
  common return type and a vtable call per vertex, and `Grid` would have to fit
  that interface. A concept accepts whatever each type naturally provides.
- **Concepts versus unconstrained templates.** Same generated code; the
  difference is entirely in error messages and in being able to overload and
  detect (`static_assert(!Graph<NotAGraph>)`).
- **Lazy deletion versus an indexed heap.** An indexed heap supports
  decrease-key and keeps the queue at V entries, but needs a position map
  updated on every swap. Lazy deletion trades a larger queue for far simpler
  code, and is what most production implementations do.
- **Accepted trade.** Hash maps instead of arrays cost constant factors on dense
  integer graphs, and Dijkstra is only correct for non-negative weights.

### Explain it back

1. Why does the `Graph` concept require hashing?

   <details><summary>Answer</summary>

   The algorithms track distances and visited vertices in `unordered_map` and
   `unordered_set`. Without a `std::hash` specialisation they would fail to
   compile deep inside the body; requiring it in the concept reports the
   problem at the call site.

   </details>

2. Why is `input_range` the right requirement for `neighbours`?

   <details><summary>Answer</summary>

   Every algorithm walks a vertex's neighbours once, forwards. Asking for more
   (random access, sized) would exclude cheap computed ranges for no benefit.
   That is also why `dfs_order` copies them into a vector before reversing.

   </details>

3. In Dijkstra, why is `d > it->second` enough to identify a stale entry?

   <details><summary>Answer</summary>

   Whenever a shorter distance to `v` is found, `dist[v]` is lowered and a new
   entry pushed. An older entry for `v` therefore carries a larger distance than
   the recorded one. The entry whose distance equals `dist[v]` is the current one
   and is processed.

   </details>

4. Why do BFS and Dijkstra give different answers for `b` in the weighted test,
   and which is "right"?

   <details><summary>Answer</summary>

   BFS counts edges: `start → b` is one hop. Dijkstra sums weights: going via
   `a` costs 2, less than the direct edge's 5. Both are right for the question
   each answers; BFS is a shortest-path algorithm only when all weights are
   equal.

   </details>

## Extending it

Add A\* by constraining on a `Heuristic` concept, and notice you now need to
express a relationship *between* two template parameters rather than a property
of one. Then try making `neighbours` return a lazy `std::ranges` view instead of
a `std::vector`, which removes the per-vertex allocation the grid currently
pays and is the reason the concept asks only for `input_range`.
