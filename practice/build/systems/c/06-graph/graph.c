/* graph.c - the reference implementation of the graph and its traversals. */
#include "graph.h"

#include <stdint.h>
#include <stdlib.h>

int graph_init(Graph *g, size_t n) {
  g->n = n;
  g->adj = NULL;
  if (n == 0) return 0;

  /* calloc, not malloc: every AdjList must start zeroed, or the first
   * graph_add_edge reallocs a garbage pointer. */
  g->adj = calloc(n, sizeof(AdjList));
  if (!g->adj) return -1;
  return 0;
}

int graph_add_edge(Graph *g, size_t from, size_t to) {
  if (from >= g->n || to >= g->n) return -1;

  AdjList *a = &g->adj[from];
  if (a->len == a->cap) {
    size_t cap = a->cap ? a->cap * 2 : 4;
    if (cap > SIZE_MAX / sizeof(size_t)) return -1;
    size_t *grown = realloc(a->to, cap * sizeof(size_t));
    if (!grown) return -1;
    a->to = grown;
    a->cap = cap;
  }
  a->to[a->len++] = to;
  return 0;
}

size_t graph_degree(const Graph *g, size_t v) {
  if (v >= g->n) return 0;
  return g->adj[v].len;
}

void graph_free(Graph *g) {
  for (size_t i = 0; i < g->n; i++) free(g->adj[i].to);
  free(g->adj);
  g->adj = NULL;
  g->n = 0;
}

int graph_bfs(const Graph *g, size_t src, size_t *dist) {
  if (src >= g->n) return -1;

  for (size_t i = 0; i < g->n; i++) dist[i] = GRAPH_UNREACHABLE;
  if (g->n == 0) return 0;

  /* A plain array used as a ring-free FIFO. Every vertex is enqueued at most
   * once, so n slots is always enough and the queue never wraps. */
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
      /* Mark on ENQUEUE, not on dequeue. Marking when a vertex comes off the
       * queue lets the same vertex be pushed several times before it is first
       * processed, which is quadratic on a dense graph and can overflow this
       * fixed-size queue. */
      if (dist[w] == GRAPH_UNREACHABLE) {
        dist[w] = dist[v] + 1;
        queue[tail++] = w;
      }
    }
  }

  free(queue);
  return 0;
}

int graph_dfs(const Graph *g, size_t src, size_t *order, size_t *count) {
  *count = 0;
  if (src >= g->n) return -1;

  char *seen = calloc(g->n, 1);
  if (!seen) return -1;

  /* An explicit stack, because recursion depth here is the length of the
   * longest path: a 100k-vertex path graph would blow the call stack.
   *
   * Each entry is a vertex plus how many of its neighbours have been consumed,
   * which is what lets the traversal resume where it left off. */
  typedef struct { size_t v, next_edge; } Frame;
  Frame *stack = malloc(g->n * sizeof(Frame));
  if (!stack) { free(seen); return -1; }
  size_t top = 0;

  seen[src] = 1;
  order[(*count)++] = src;
  stack[top++] = (Frame){src, 0};

  while (top > 0) {
    Frame *f = &stack[top - 1];
    const AdjList *a = &g->adj[f->v];

    if (f->next_edge >= a->len) { top--; continue; }

    size_t w = a->to[f->next_edge++];
    if (seen[w]) continue;

    seen[w] = 1;
    order[(*count)++] = w;
    stack[top++] = (Frame){w, 0};
  }

  free(stack);
  free(seen);
  return 0;
}

int graph_toposort(const Graph *g, size_t *order) {
  if (g->n == 0) return 0;

  size_t *indeg = calloc(g->n, sizeof(size_t));
  if (!indeg) return -1;
  for (size_t v = 0; v < g->n; v++) {
    for (size_t i = 0; i < g->adj[v].len; i++) indeg[g->adj[v].to[i]]++;
  }

  size_t *queue = malloc(g->n * sizeof(size_t));
  if (!queue) { free(indeg); return -1; }
  size_t head = 0, tail = 0;

  for (size_t v = 0; v < g->n; v++) {
    if (indeg[v] == 0) queue[tail++] = v;
  }

  size_t emitted = 0;
  while (head < tail) {
    size_t v = queue[head++];
    order[emitted++] = v;
    for (size_t i = 0; i < g->adj[v].len; i++) {
      size_t w = g->adj[v].to[i];
      /* Decrement per EDGE, not per distinct neighbour. A parallel edge v->w
       * contributes 2 to w's in-degree, so it must be paid back twice or w
       * never reaches zero. */
      if (--indeg[w] == 0) queue[tail++] = w;
    }
  }

  free(queue);
  free(indeg);

  /* Kahn's algorithm gives cycle detection for nothing: a vertex inside a
   * cycle never reaches in-degree zero, so it is never emitted. */
  return emitted == g->n ? 0 : -1;
}
