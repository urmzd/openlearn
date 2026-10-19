/* main.c - the test suite for the reference implementation. Exits 0 when every
 * check holds. */
#include "graph.h"

#include <stdio.h>
#include <stdlib.h>

static int checks = 0;
#define CHECK(cond, what)                                                      \
  do {                                                                         \
    checks++;                                                                  \
    if (!(cond)) {                                                             \
      fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, (what));         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static void add_undirected(Graph *g, size_t a, size_t b) {
  CHECK(graph_add_edge(g, a, b) == 0, "edge added");
  CHECK(graph_add_edge(g, b, a) == 0, "reverse edge added");
}

static void test_construction(void) {
  Graph g;
  CHECK(graph_init(&g, 5) == 0, "init succeeds");
  CHECK(g.n == 5, "vertex count is recorded");
  for (size_t v = 0; v < 5; v++) {
    CHECK(graph_degree(&g, v) == 0, "a fresh vertex has no edges");
  }

  CHECK(graph_add_edge(&g, 0, 1) == 0, "edge added");
  CHECK(graph_add_edge(&g, 0, 2) == 0, "edge added");
  CHECK(graph_degree(&g, 0) == 2, "out-degree counts outgoing edges");
  CHECK(graph_degree(&g, 1) == 0, "a directed edge does not touch the target");

  /* Growth past the initial capacity must not lose edges. */
  for (size_t i = 0; i < 100; i++) {
    CHECK(graph_add_edge(&g, 3, i % 5) == 0, "edge added during growth");
  }
  CHECK(graph_degree(&g, 3) == 100, "every edge survived reallocation");

  CHECK(graph_add_edge(&g, 5, 0) == -1, "out-of-range source is rejected");
  CHECK(graph_add_edge(&g, 0, 5) == -1, "out-of-range target is rejected");

  graph_free(&g);
  CHECK(g.adj == NULL && g.n == 0, "free returns the zero state");
  graph_free(&g); /* must not double-free */
}

static void test_empty_graph(void) {
  Graph g;
  CHECK(graph_init(&g, 0) == 0, "a graph with no vertices is legal");
  size_t order[1];
  CHECK(graph_toposort(&g, order) == 0, "toposort of nothing succeeds");
  graph_free(&g);
}

/*  0 -- 1 -- 3
 *  |    |
 *  2    4 -- 5        6 (isolated) */
static void test_bfs_distances(void) {
  Graph g;
  size_t dist[7];
  CHECK(graph_init(&g, 7) == 0, "init succeeds");
  add_undirected(&g, 0, 1);
  add_undirected(&g, 0, 2);
  add_undirected(&g, 1, 3);
  add_undirected(&g, 1, 4);
  add_undirected(&g, 4, 5);

  CHECK(graph_bfs(&g, 0, dist) == 0, "bfs succeeds");
  CHECK(dist[0] == 0, "the source is at distance 0");
  CHECK(dist[1] == 1, "a neighbour is at distance 1");
  CHECK(dist[2] == 1, "the other neighbour is at distance 1");
  CHECK(dist[3] == 2, "two hops away");
  CHECK(dist[4] == 2, "two hops away");
  CHECK(dist[5] == 3, "three hops away");
  CHECK(dist[6] == GRAPH_UNREACHABLE, "an isolated vertex is unreachable");

  CHECK(graph_bfs(&g, 6, dist) == 0, "bfs from an isolated vertex succeeds");
  CHECK(dist[6] == 0, "it can still reach itself");
  CHECK(dist[0] == GRAPH_UNREACHABLE, "and nothing else");

  CHECK(graph_bfs(&g, 7, dist) == -1, "out-of-range source is rejected");
  graph_free(&g);
}

/* BFS must find the SHORTEST path, not the first one it stumbles into. Here
 * vertex 3 is reachable in one hop directly and in three hops the long way
 * round; a traversal that overwrites an already-set distance reports 3. */
static void test_bfs_takes_the_short_route(void) {
  Graph g;
  size_t dist[4];
  CHECK(graph_init(&g, 4) == 0, "init succeeds");
  graph_add_edge(&g, 0, 1);
  graph_add_edge(&g, 1, 2);
  graph_add_edge(&g, 2, 3);
  graph_add_edge(&g, 0, 3); /* the shortcut, added last */

  CHECK(graph_bfs(&g, 0, dist) == 0, "bfs succeeds");
  CHECK(dist[3] == 1, "bfs reports the shortest distance, not the first found");
  graph_free(&g);
}

/* A long path: 100000 vertices in a line. A recursive DFS overflows the stack
 * here on most systems, which is why the interface demands an iterative one. */
static void test_dfs_survives_a_very_long_path(void) {
  enum { N = 100000 };
  Graph g;
  CHECK(graph_init(&g, N) == 0, "init succeeds");
  for (size_t i = 0; i + 1 < N; i++) {
    CHECK(graph_add_edge(&g, i, i + 1) == 0, "edge added");
  }

  size_t *order = malloc(N * sizeof *order);
  size_t count = 0;
  CHECK(graph_dfs(&g, 0, order, &count) == 0, "dfs succeeds on a very deep graph");
  CHECK(count == N, "dfs reached every vertex");
  for (size_t i = 0; i < N; i++) {
    CHECK(order[i] == i, "a path graph is visited in order");
  }

  size_t *dist = malloc(N * sizeof *dist);
  CHECK(graph_bfs(&g, 0, dist) == 0, "bfs succeeds too");
  CHECK(dist[N - 1] == N - 1, "the far end is n-1 hops away");
  free(order);
  free(dist);
  graph_free(&g);
}

static void test_dfs_order_and_reachability(void) {
  Graph g;
  size_t order[6], count = 0;
  CHECK(graph_init(&g, 6) == 0, "init succeeds");
  /* 0 -> 1 -> 3, 0 -> 2, and 4 -> 5 in a separate component. */
  graph_add_edge(&g, 0, 1);
  graph_add_edge(&g, 0, 2);
  graph_add_edge(&g, 1, 3);
  graph_add_edge(&g, 4, 5);

  CHECK(graph_dfs(&g, 0, order, &count) == 0, "dfs succeeds");
  CHECK(count == 4, "dfs visits only the reachable component");
  CHECK(order[0] == 0, "dfs starts at the source");
  /* Neighbours in insertion order means 1 before 2, and depth-first means 3
   * comes before 2. */
  CHECK(order[1] == 1, "dfs descends into the first neighbour");
  CHECK(order[2] == 3, "and goes deep before wide");
  CHECK(order[3] == 2, "only then does it take the second neighbour");

  /* Each vertex appears exactly once even when several paths lead to it. */
  graph_add_edge(&g, 2, 3);
  graph_add_edge(&g, 3, 0);
  CHECK(graph_dfs(&g, 0, order, &count) == 0, "dfs succeeds with cycles present");
  CHECK(count == 4, "cycles do not cause repeats or hangs");
  graph_free(&g);
}

static void test_toposort(void) {
  Graph g;
  size_t order[6];
  /* A classic dependency DAG: 5,4 -> 2,0,1,3 */
  CHECK(graph_init(&g, 6) == 0, "init succeeds");
  graph_add_edge(&g, 5, 2);
  graph_add_edge(&g, 5, 0);
  graph_add_edge(&g, 4, 0);
  graph_add_edge(&g, 4, 1);
  graph_add_edge(&g, 2, 3);
  graph_add_edge(&g, 3, 1);

  CHECK(graph_toposort(&g, order) == 0, "toposort of a DAG succeeds");

  /* Verify the defining property directly rather than pinning one valid
   * answer: several orders are correct, and the algorithm may pick any. */
  size_t pos[6];
  for (size_t i = 0; i < 6; i++) pos[order[i]] = i;
  for (size_t v = 0; v < 6; v++) {
    for (size_t i = 0; i < g.adj[v].len; i++) {
      CHECK(pos[v] < pos[g.adj[v].to[i]], "every edge points forward in the order");
    }
  }
  graph_free(&g);
}

static void test_toposort_detects_cycles(void) {
  Graph g;
  size_t order[3];
  CHECK(graph_init(&g, 3) == 0, "init succeeds");
  graph_add_edge(&g, 0, 1);
  graph_add_edge(&g, 1, 2);
  CHECK(graph_toposort(&g, order) == 0, "a chain is a DAG");

  graph_add_edge(&g, 2, 0); /* closes the loop */
  CHECK(graph_toposort(&g, order) == -1, "a cycle is reported");
  graph_free(&g);

  /* A self-loop is the smallest cycle there is. */
  CHECK(graph_init(&g, 2) == 0, "init succeeds");
  graph_add_edge(&g, 0, 0);
  CHECK(graph_toposort(&g, order) == -1, "a self-loop is a cycle");
  graph_free(&g);
}

/* Parallel edges make in-degree larger than the number of distinct
 * predecessors. Decrementing once per neighbour instead of once per edge
 * leaves the count stuck above zero and reports a phantom cycle. */
static void test_toposort_with_parallel_edges(void) {
  Graph g;
  size_t order[3];
  CHECK(graph_init(&g, 3) == 0, "init succeeds");
  graph_add_edge(&g, 0, 1);
  graph_add_edge(&g, 0, 1); /* same edge again */
  graph_add_edge(&g, 0, 1); /* and again */
  graph_add_edge(&g, 1, 2);

  CHECK(graph_degree(&g, 0) == 3, "parallel edges are all stored");
  CHECK(graph_toposort(&g, order) == 0, "parallel edges are not a cycle");
  CHECK(order[0] == 0 && order[1] == 1 && order[2] == 2, "and the order is still correct");
  graph_free(&g);
}

int main(void) {
  test_construction();
  test_empty_graph();
  test_bfs_distances();
  test_bfs_takes_the_short_route();
  test_dfs_survives_a_very_long_path();
  test_dfs_order_and_reachability();
  test_toposort();
  test_toposort_detects_cycles();
  test_toposort_with_parallel_edges();
  printf("ok  c/06-graph  %d checks passed\n", checks);
  return 0;
}
