// main.cpp - the test suite for the reference implementation. Exits 0 when
// every check holds.
//
// Three unrelated graph types, no common base class, one set of algorithms.
#include "graph.hpp"

#include <cstdio>
#include <cstdlib>
#include <ranges>
#include <string>

static int checks = 0;
#define CHECK(cond, what)                                                     \
  do {                                                                        \
    ++checks;                                                                 \
    if (!(cond)) {                                                            \
      std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, (what));   \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

// ---- graph 1: a stored adjacency list --------------------------------------

class AdjacencyList {
 public:
  using vertex_type = int;

  void add_edge(int from, int to) { adj_[from].push_back(to); }
  void add_undirected(int a, int b) {
    add_edge(a, b);
    add_edge(b, a);
  }

  const std::vector<int>& neighbours(int v) const {
    static const std::vector<int> none;
    auto it = adj_.find(v);
    return it == adj_.end() ? none : it->second;
  }

 private:
  std::unordered_map<int, std::vector<int>> adj_;
};

// ---- graph 2: a grid whose neighbours are COMPUTED, never stored ------------
//
// There is no adjacency structure here at all. A million-cell grid costs
// nothing to represent, which is the case that makes a concept worth more than
// a base class.

struct Cell {
  int r, c;
  bool operator==(const Cell& o) const { return r == o.r && c == o.c; }
};

template <>
struct std::hash<Cell> {
  std::size_t operator()(const Cell& p) const noexcept {
    return std::hash<long long>{}(static_cast<long long>(p.r) * 1000003 + p.c);
  }
};

class Grid {
 public:
  using vertex_type = Cell;

  Grid(int rows, int cols, std::vector<Cell> walls = {})
      : rows_(rows), cols_(cols), walls_(walls.begin(), walls.end()) {}

  std::vector<Cell> neighbours(Cell v) const {
    std::vector<Cell> out;
    const int dr[] = {-1, 1, 0, 0};
    const int dc[] = {0, 0, -1, 1};
    for (int i = 0; i < 4; ++i) {
      Cell n{v.r + dr[i], v.c + dc[i]};
      if (n.r < 0 || n.c < 0 || n.r >= rows_ || n.c >= cols_) continue;
      if (walls_.contains(n)) continue;
      out.push_back(n);
    }
    return out;
  }

 private:
  int rows_, cols_;
  std::unordered_set<Cell> walls_;
};

// ---- graph 3: weighted, over strings ---------------------------------------

class WeightedNamed {
 public:
  using vertex_type = std::string;

  void add_edge(const std::string& a, const std::string& b, double w) {
    adj_[a].push_back(b);
    weights_[a + "\x1f" + b] = w;
  }

  const std::vector<std::string>& neighbours(const std::string& v) const {
    static const std::vector<std::string> none;
    auto it = adj_.find(v);
    return it == adj_.end() ? none : it->second;
  }

  double weight(const std::string& a, const std::string& b) const {
    return weights_.at(a + "\x1f" + b);
  }

 private:
  std::unordered_map<std::string, std::vector<std::string>> adj_;
  std::unordered_map<std::string, double> weights_;
};

// ---- the concepts must classify these correctly ----------------------------

static_assert(Graph<AdjacencyList>, "an adjacency list is a Graph");
static_assert(Graph<Grid>, "a computed grid is a Graph");
static_assert(Graph<WeightedNamed>, "a weighted graph is also a Graph");

static_assert(!WeightedGraph<AdjacencyList>, "an unweighted graph is not a WeightedGraph");
static_assert(!WeightedGraph<Grid>, "nor is a grid");
static_assert(WeightedGraph<WeightedNamed>, "but the weighted one is");

// A type missing neighbours() must not satisfy the concept.
struct NotAGraph {
  using vertex_type = int;
};
static_assert(!Graph<NotAGraph>, "a type without neighbours() is not a Graph");

// Nor one missing vertex_type.
struct AlsoNotAGraph {
  std::vector<int> neighbours(int) const { return {}; }
};
static_assert(!Graph<AlsoNotAGraph>, "a type without vertex_type is not a Graph");

// The algorithms must be callable only with satisfying types.
template <typename G, typename = void>
struct bfs_callable : std::false_type {};
template <typename G>
struct bfs_callable<G, std::void_t<decltype(bfs(std::declval<const G&>(),
                                               std::declval<typename G::vertex_type>()))>>
    : std::true_type {};

static_assert(bfs_callable<AdjacencyList>::value, "bfs works on an adjacency list");
static_assert(!bfs_callable<NotAGraph>::value, "bfs must reject a non-graph");

// ---- tests -----------------------------------------------------------------

static void test_adjacency_list() {
  AdjacencyList g;
  g.add_undirected(0, 1);
  g.add_undirected(0, 2);
  g.add_undirected(1, 3);
  g.add_undirected(3, 4);

  auto dist = bfs(g, 0);
  CHECK(dist[0] == 0, "source is at distance 0");
  CHECK(dist[1] == 1, "one hop");
  CHECK(dist[2] == 1, "one hop");
  CHECK(dist[3] == 2, "two hops");
  CHECK(dist[4] == 3, "three hops");
  CHECK(!dist.contains(99), "an absent vertex is not reported");

  auto order = dfs_order(g, 0);
  CHECK(order.size() == 5, "dfs visits every reachable vertex");
  CHECK(order[0] == 0, "starting at the source");
  CHECK(order[1] == 1, "descending into the first neighbour");
  CHECK(order[2] == 3, "and going deep before wide");

  auto path = shortest_path(g, 0, 4);
  CHECK(path.has_value(), "a path exists");
  CHECK(path->size() == 4, "and has four vertices");
  CHECK(path->front() == 0 && path->back() == 4, "from source to destination");

  CHECK(!shortest_path(g, 0, 99).has_value(), "an unreachable destination gives nullopt");
  auto trivial = shortest_path(g, 2, 2);
  CHECK(trivial.has_value() && trivial->size() == 1, "src == dst is a one-vertex path");
}

// The same algorithms, on a graph whose edges do not exist in memory.
static void test_computed_grid() {
  Grid g(5, 5);
  auto dist = bfs(g, Cell{0, 0});
  CHECK(dist.size() == 25, "bfs reached every cell");
  CHECK((dist[Cell{0, 0}] == 0), "source");
  CHECK((dist[Cell{4, 4}] == 8), "Manhattan distance across an open grid");
  CHECK((dist[Cell{0, 4}] == 4), "along an edge");

  // A wall down the middle with one gap forces a detour.
  std::vector<Cell> walls;
  for (int r = 0; r < 4; ++r) walls.push_back(Cell{r, 2});
  Grid blocked(5, 5, walls);

  auto d2 = bfs(blocked, Cell{0, 0});
  CHECK((d2[Cell{0, 4}] == 12), "the path must go around the wall");
  CHECK((!d2.contains(Cell{0, 2})), "a wall cell is never reached");

  auto path = shortest_path(blocked, Cell{0, 0}, Cell{0, 4});
  CHECK(path.has_value(), "a path around the wall exists");
  CHECK(path->size() == 13, "and its length matches the distance");

  // Every consecutive pair in the path must be adjacent.
  for (std::size_t i = 1; i < path->size(); ++i) {
    const Cell& a = (*path)[i - 1];
    const Cell& b = (*path)[i];
    int d = std::abs(a.r - b.r) + std::abs(a.c - b.c);
    CHECK(d == 1, "consecutive path cells are adjacent");
  }

  // Fully walled off.
  std::vector<Cell> full;
  for (int r = 0; r < 5; ++r) full.push_back(Cell{r, 2});
  Grid sealed(5, 5, full);
  CHECK(!shortest_path(sealed, Cell{0, 0}, Cell{0, 4}).has_value(),
        "a sealed wall makes the far side unreachable");
}

// A large computed graph: no adjacency structure is ever built.
static void test_large_computed_grid() {
  Grid g(300, 300);
  auto dist = bfs(g, Cell{0, 0});
  CHECK(dist.size() == 90000, "bfs covered a 300x300 grid");
  CHECK((dist[Cell{299, 299}] == 598), "the far corner is 598 hops away");
}

static void test_weighted() {
  WeightedNamed g;
  g.add_edge("start", "a", 1.0);
  g.add_edge("start", "b", 5.0);
  g.add_edge("a", "b", 1.0);
  g.add_edge("b", "end", 1.0);
  g.add_edge("a", "end", 10.0);

  auto dist = shortest_distances(g, "start");
  CHECK(dist["start"] == 0.0, "source at zero");
  CHECK(dist["a"] == 1.0, "direct edge");
  // start -> a -> b costs 2, beating the direct edge of 5.
  CHECK(dist["b"] == 2.0, "dijkstra takes the cheaper two-hop route");
  CHECK(dist["end"] == 3.0, "and continues to the cheapest total");

  // BFS on the same graph counts HOPS, not weight, and so disagrees. That
  // disagreement is the whole reason Dijkstra exists.
  auto hops = bfs(g, "start");
  CHECK(hops["b"] == 1, "bfs sees b as one hop away");
  CHECK(hops["end"] == 2, "and end as two");
  CHECK(dist["end"] != static_cast<double>(hops["end"]),
        "hop count and weighted distance are different questions");
}

int main() {
  test_adjacency_list();
  test_computed_grid();
  test_large_computed_grid();
  test_weighted();
  std::printf("ok  cpp/09-graph-concepts  %d checks passed\n", checks);
  return 0;
}
