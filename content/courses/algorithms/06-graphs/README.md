# Graphs

Graph problems boil down to two questions: **how do I traverse, and what do I track?** Breadth-first search gives shortest paths in unweighted graphs. Depth-first search explores connected components and detects cycles. Most "hard" graph problems turn out to be BFS or DFS with clever state encoding: the nodes are board positions or words, the edges are moves or one-letter changes, and the visited set records more than a node id.

This chapter builds the traversal toolkit first (BFS for shortest paths, DFS for flood fill), then turns to dependencies with topological sort and cycle detection, weighted shortest paths with Dijkstra's algorithm, dynamic connectivity with union-find, and multi-source BFS. It closes with a comparison of graph representations and five solutions in JavaScript, TypeScript, and C.

> **Learning objectives**
> - Find shortest paths in unweighted graphs with BFS and count connected regions with DFS flood fill.
> - Order a DAG with Kahn's algorithm and detect cycles in directed and undirected graphs.
> - Compute weighted shortest paths with Dijkstra's algorithm and explain its stale-entry check.
> - Implement union-find with path compression and union by rank.
> - Choose a graph representation by density and by the operations you need.

## Before you begin

You should know queues, stacks, and recursion, and be comfortable with the tree traversals in [Trees](../05-trees/README.md), since a tree is a graph without cycles. [Graphs, DAGs, and topological sort](../../mathematics-for-computing/07-discrete-math-2/01-graphs-dags-and-topological-sort.md) in *Mathematics for Computing* covers the definitions and proofs. Dijkstra's algorithm uses a priority queue; [the binary heap for top-k](../13-systems-data-structures/04-binary-heap-top-k.md) builds one.

## BFS for shortest unweighted paths

To find the minimum number of steps, moves, or transformations, explore outward in rings with a queue. Mark a node visited when you enqueue it, not when you dequeue it, or the same node is queued many times:

```python
from collections import deque
def bfs(start, target):
    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        node, dist = queue.popleft()
        if node == target:
            return dist
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    return -1
```

Shortest Path in Binary Matrix, Word Ladder, Rotting Oranges, and Open the Lock are BFS problems in disguise.

## DFS for connected components and flood fill

To count islands or connected regions, or to fill or mark a region, start a DFS from every unvisited cell and let it mark everything it reaches. Each start is one component:

```python
def dfs(r, c):
    if r < 0 or r >= rows or c < 0 or c >= cols:
        return
    if grid[r][c] != target:
        return
    grid[r][c] = visited_marker  # mark in-place
    for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
        dfs(r + dr, c + dc)

count = 0
for r in range(rows):
    for c in range(cols):
        if grid[r][c] == target:
            dfs(r, c)
            count += 1
```

Marking the grid in place doubles as the visited set. Number of Islands, Flood Fill, Surrounded Regions, and Pacific Atlantic Water Flow are the standard problems.

## Topological sort

Dependencies, prerequisites, and build orders are all directed acyclic graphs that must be processed in dependency order. Kahn's algorithm counts each node's incoming edges, starts from the nodes with none, and releases a node once all its prerequisites have been processed. If some nodes are never released, the graph has a cycle:

```python
from collections import deque
indegree = {u: 0 for u in graph}
for u in graph:
    for v in graph[u]:
        indegree[v] += 1
queue = deque([u for u in indegree if indegree[u] == 0])
order = []
while queue:
    u = queue.popleft()
    order.append(u)
    for v in graph[u]:
        indegree[v] -= 1
        if indegree[v] == 0:
            queue.append(v)
if len(order) != len(graph):
    return "cycle detected"
```

Course Schedule, Course Schedule II, Alien Dictionary, and Task Scheduler use it.

## Cycle detection

Cycles can also be found directly with DFS, and the method depends on whether edges have a direction. In a directed graph, track three states per node: unvisited, in progress, and visited. An edge back to an in-progress node is a back edge, and a back edge means a cycle. In an undirected graph, every edge looks like it leads back, so the rule is different: reaching a neighbor that is already visited and is not the node you came from means a cycle. Course Schedule and the Detect Cycle problems for directed and undirected graphs exercise both.

## Dijkstra's algorithm for weighted shortest paths

With non-negative edge weights, BFS's rings become a priority queue ordered by distance. Each pop settles the closest unsettled node. Because a node can be pushed several times with improving distances, skip any popped entry whose distance is stale:

```python
import heapq
def dijkstra(graph, start):
    dist = {start: 0}
    heap = [(0, start)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, float('inf')):
            continue
        for v, w in graph[u]:
            nd = d + w
            if nd < dist.get(v, float('inf')):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist
```

Network Delay Time, Cheapest Flights Within K Stops, Path With Minimum Effort, and Swim in Rising Water are the standard problems. Negative weights break the algorithm, because a settled node could later be reached more cheaply.

## Union-find

For dynamic connectivity ("are these two nodes connected?" as edges keep arriving), a disjoint-set structure answers each query in nearly constant amortized time. Path compression flattens the trees during `find`, and union by rank keeps them shallow:

```python
parent = list(range(n))
rank = [0] * n

def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]  # path compression
        x = parent[x]
    return x

def union(x, y):
    px, py = find(x), find(y)
    if px == py: return False
    if rank[px] < rank[py]: px, py = py, px
    parent[py] = px
    if rank[px] == rank[py]: rank[px] += 1
    return True
```

`union` returning `False` means the two nodes were already connected, which is exactly how Redundant Connection finds the extra edge. Number of Connected Components in an Undirected Graph, Accounts Merge, and Smallest String With Swaps also use it.

## Multi-source BFS

When there are several starting points at once (every rotten orange, every gate), initialize the queue with all of them and then run a standard BFS. The distances that come out are distances to the nearest source. Rotting Oranges, 01 Matrix, Walls and Gates, and Pacific Atlantic Water Flow (which starts from every cell on each ocean's border) use this pattern.

## Representing a graph

How you store the graph decides what is cheap. The four common choices:

| Format | Best for | Space |
|--------|----------|-------|
| adjacency list | sparse graphs (most interviews) | $O(V + E)$ |
| adjacency matrix | dense graphs, quick edge lookup | $O(V^2)$ |
| edge list | union-find problems | $O(E)$ |
| implicit graph | grid and matrix problems | $O(1)$ extra |

Interviewers favor different variants. As a rough guide:

| Company | Favorite variant | Difficulty |
|---------|-----------------|------------|
| Google | Dijkstra variants, topological sort | Hard |
| Meta | BFS and DFS on grids, clone graph | Medium |
| Amazon | course schedule (topological sort) | Medium |
| Anthropic | graphs as distributed-system models | Hard |
| NVIDIA | parallel BFS, graph partitioning | Hard |
| Palantir | entity relationship graphs | Medium-Hard |

## In this chapter

The JavaScript files are solution functions; run them with Node and a small driver, or in an online judge. The TypeScript code can be run with `ts-node`, or compiled with `tsc` and run with Node.

- [Number of Islands](number-of-islands.js): DFS flood fill on a grid of `"1"` and `"0"` cells. Each unvisited land cell starts a DFS that sinks its whole island to `"0"`, so the number of starts is the number of islands. The function is written with TypeScript type annotations, so run it through `ts-node` or `tsc`.
- [Clone Graph](clone-graph.js): a DFS that copies each node once, keeping a map from value to clone so that cycles and shared neighbors point at the same copy instead of recursing forever.
- [Course Schedule](course-schedule.js): Kahn's algorithm. It builds an adjacency map from each prerequisite to the courses it unlocks and an in-degree count per course, processes courses with no remaining prerequisites, and reports that all courses can be finished only if every course with prerequisites is eventually released.
- [Pacific Atlantic Water Flow](pacific-atlantic-water-flow.ts): reverses the direction of flow and runs multi-source DFS uphill from each ocean's border cells, one visited set per ocean. A cell that ends up in both sets can drain to both oceans.
- [Charles and the Corgi Conundrum](charles-and-the-corgi-conundrum.c): a graph problem stated in C, left for you to implement. Each corgi's "social reach" is $1$ plus the degree of each of its friends, and the program prints the average over all corgis per test case to three decimals. In the sample, the path $1 - 2 - 3$ gives every corgi a reach of $3$, so the output is `Pond #1: 3.000`. Summing over all corgis, each vertex $v$ contributes its degree once per neighbor, so the total is $n + \sum_v \deg(v)^2$, computable from an array of degrees in $O(n + m)$.
