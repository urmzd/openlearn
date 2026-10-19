/* graph.h - a directed graph stored as adjacency lists.
 *
 * The interface. graph.c implements it and main.c tests it.
 *
 * Each vertex owns a growable array of outgoing edges. Undirected graphs are
 * built by adding both directions, which the tests do explicitly.
 */
#ifndef GRAPH_H
#define GRAPH_H

#include <stddef.h>

#define GRAPH_UNREACHABLE ((size_t)-1)

typedef struct {
  size_t *to;   /* destination vertex of each outgoing edge */
  size_t len;
  size_t cap;
} AdjList;

typedef struct {
  AdjList *adj; /* one per vertex */
  size_t n;
} Graph;

/* Build a graph with `n` vertices and no edges. 0 on success, -1 on
 * allocation failure. */
int graph_init(Graph *g, size_t n);

/* Add a directed edge from -> to. Duplicate edges are allowed and are stored
 * twice: deduplicating is the caller's business. 0 on success, -1 on
 * allocation failure. */
int graph_add_edge(Graph *g, size_t from, size_t to);

/* Number of outgoing edges from `v`. */
size_t graph_degree(const Graph *g, size_t v);

/* Release everything. Safe to call twice. */
void graph_free(Graph *g);

/* Breadth-first search from `src`.
 *
 * Writes into `dist` (length g->n) the number of edges on a shortest path from
 * src to each vertex, or GRAPH_UNREACHABLE for vertices with no path. Returns
 * 0 on success, -1 on allocation failure.
 *
 * BFS gives shortest paths only because every edge has the same weight. The
 * moment edges carry different costs this is wrong and you need Dijkstra. */
int graph_bfs(const Graph *g, size_t src, size_t *dist);

/* Depth-first search from `src`, iterative rather than recursive so that a
 * long path cannot overflow the stack.
 *
 * Writes into `order` the vertices in the order they were first visited, and
 * into *count how many were reached. `order` must have room for g->n.
 * Neighbours are explored in the order they were added. */
int graph_dfs(const Graph *g, size_t src, size_t *order, size_t *count);

/* Topological order of a DAG, written into `order` (length g->n).
 *
 * Returns 0 on success, or -1 if the graph has a cycle, in which case `order`
 * holds nothing meaningful. Uses Kahn's algorithm, so detecting the cycle is
 * free: if fewer than n vertices come out, the rest are in one. */
int graph_toposort(const Graph *g, size_t *order);

#endif /* GRAPH_H */
