// graph.hpp - graph algorithms written against a concept, not a class.
//
// The reference implementation. main.cpp tests it.
//
// The point is not the graph. It is that `bfs` here works on ANY type
// satisfying the Graph concept: an adjacency list, a grid where neighbours are
// computed rather than stored, or a lazily generated state space. None of them
// share a base class, and none of them know about each other.
//
// Before concepts you would write this as an unconstrained template and get a
// page of instantiation errors when a type did not fit. A concept moves the
// error to the call site and states what was actually missing.
#pragma once

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <optional>
#include <queue>
#include <ranges>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// What it means to be a graph: vertices are hashable and comparable, and you
// can ask any vertex for a range of its neighbours.
template <typename G>
concept Graph = requires(const G& g, typename G::vertex_type v) {
  typename G::vertex_type;
  requires std::equality_comparable<typename G::vertex_type>;
  requires requires(typename G::vertex_type x) { std::hash<typename G::vertex_type>{}(x); };
  { g.neighbours(v) } -> std::ranges::input_range;
};

// A weighted graph additionally reports an edge cost.
template <typename G>
concept WeightedGraph = Graph<G> && requires(const G& g, typename G::vertex_type v) {
  { g.weight(v, v) } -> std::convertible_to<double>;
};

// ---- algorithms ------------------------------------------------------------

// Shortest hop counts from `src`. Works on any Graph.
template <Graph G>
auto bfs(const G& g, typename G::vertex_type src)
    -> std::unordered_map<typename G::vertex_type, std::size_t> {
  using V = typename G::vertex_type;
  std::unordered_map<V, std::size_t> dist;
  std::queue<V> frontier;

  dist[src] = 0;
  frontier.push(src);

  while (!frontier.empty()) {
    V v = frontier.front();
    frontier.pop();
    for (const V& w : g.neighbours(v)) {
      // Insert-and-check in one lookup. Marking on ENQUEUE rather than on
      // dequeue is what stops a vertex being queued once per predecessor.
      if (dist.try_emplace(w, dist[v] + 1).second) frontier.push(w);
    }
  }
  return dist;
}

// Vertices in first-visit order, iteratively so that a long path cannot
// overflow the stack.
template <Graph G>
auto dfs_order(const G& g, typename G::vertex_type src)
    -> std::vector<typename G::vertex_type> {
  using V = typename G::vertex_type;
  std::vector<V> order;
  std::unordered_set<V> seen;
  std::vector<V> stack{src};

  while (!stack.empty()) {
    V v = stack.back();
    stack.pop_back();
    if (!seen.insert(v).second) continue;  // already visited
    order.push_back(v);

    // Push neighbours reversed, so that the first neighbour is popped first
    // and the traversal matches the recursive formulation's order.
    std::vector<V> ns;
    for (const V& w : g.neighbours(v)) ns.push_back(w);
    for (auto it = ns.rbegin(); it != ns.rend(); ++it) {
      if (!seen.contains(*it)) stack.push_back(*it);
    }
  }
  return order;
}

// A path from src to dst, or nullopt when unreachable.
template <Graph G>
auto shortest_path(const G& g, typename G::vertex_type src, typename G::vertex_type dst)
    -> std::optional<std::vector<typename G::vertex_type>> {
  using V = typename G::vertex_type;
  if (src == dst) return std::vector<V>{src};

  std::unordered_map<V, V> parent;
  std::unordered_set<V> seen{src};
  std::queue<V> frontier;
  frontier.push(src);

  while (!frontier.empty()) {
    V v = frontier.front();
    frontier.pop();
    for (const V& w : g.neighbours(v)) {
      if (!seen.insert(w).second) continue;
      parent[w] = v;
      if (w == dst) {
        // Walk the parent chain back and reverse it.
        std::vector<V> path{dst};
        V cur = dst;
        while (cur != src) {
          cur = parent[cur];
          path.push_back(cur);
        }
        std::reverse(path.begin(), path.end());
        return path;
      }
      frontier.push(w);
    }
  }
  return std::nullopt;
}

// Dijkstra, available only for graphs that also report weights. The constraint
// is what makes calling this on an unweighted graph a clear error rather than
// a template instantiation failure deep inside the body.
template <WeightedGraph G>
auto shortest_distances(const G& g, typename G::vertex_type src)
    -> std::unordered_map<typename G::vertex_type, double> {
  using V = typename G::vertex_type;
  using Entry = std::pair<double, V>;
  std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> pq;
  std::unordered_map<V, double> dist;

  dist[src] = 0.0;
  pq.emplace(0.0, src);

  while (!pq.empty()) {
    auto [d, v] = pq.top();
    pq.pop();

    // Lazy deletion: the queue may hold stale entries for a vertex that has
    // since been improved. Skipping them is far simpler than implementing
    // decrease-key, and costs only a larger queue.
    if (auto it = dist.find(v); it != dist.end() && d > it->second) continue;

    for (const V& w : g.neighbours(v)) {
      double nd = d + g.weight(v, w);
      auto it = dist.find(w);
      if (it == dist.end() || nd < it->second) {
        dist[w] = nd;
        pq.emplace(nd, w);
      }
    }
  }
  return dist;
}
