# 06-graphs

## Summary
- Contains: `clone-graph.js`, `course-schedule.js`, `number-of-islands.js`, `pacific-atlantic-water-flow.ts`.
- Graph traversal and topological reasoning problems in JS/TS.

## Key takeaways
- DFS/BFS patterns recur across connectivity and reachability tasks.
- Cycle detection/topo order is central to scheduling constraints.

## How to run
- JS files are solution functions (run with Node + a driver or an online judge).
- The TS file can be run with `ts-node` or compiled with `tsc` and run with Node.

---

# Graph Patterns

## Core Insight

Graph problems boil down to: **how do I traverse, and what do I track?** BFS gives shortest unweighted paths. DFS explores connected components and detects cycles. Most "hard" graph problems are just BFS/DFS with clever state encoding.

## Pattern 1: BFS — Shortest Path (Unweighted)

**When to use**: Find minimum steps/moves/transformations.

**Template**:
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

**Interview problems**: Shortest Path in Binary Matrix, Word Ladder, Rotting Oranges, Open the Lock

## Pattern 2: DFS — Connected Components & Flood Fill

**When to use**: Count islands, connected regions, or fill/mark regions.

**Template**:
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

**Interview problems**: Number of Islands, Flood Fill, Surrounded Regions, Pacific Atlantic Water Flow

## Pattern 3: Topological Sort (DAG Processing)

**When to use**: Dependencies, prerequisites, build order — any DAG where you process nodes in dependency order.

**Template (Kahn's — BFS)**:
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

**Interview problems**: Course Schedule, Course Schedule II, Alien Dictionary, Task Scheduler

## Pattern 4: Dijkstra — Weighted Shortest Path

**When to use**: Weighted graph, non-negative edge weights.

**Template**:
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

**Interview problems**: Network Delay Time, Cheapest Flights Within K Stops, Path with Minimum Effort, Swim in Rising Water

## Pattern 5: Union-Find (Disjoint Set Union)

**When to use**: Dynamic connectivity — "are these two nodes connected?" with merge operations.

**Template**:
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

**Interview problems**: Number of Connected Components, Redundant Connection, Accounts Merge, Smallest String With Swaps

## Pattern 6: Cycle Detection

**DFS (directed graph)**: Track three states: unvisited, in-progress, visited. A back edge to an in-progress node = cycle.

**DFS (undirected graph)**: If you visit a neighbor that's already visited and isn't the parent, there's a cycle.

**Interview problems**: Course Schedule, Detect Cycle in Directed/Undirected Graph

## Pattern 7: Multi-Source BFS

**When to use**: Multiple starting points simultaneously (rotting oranges, walls and gates).

**Template**: Initialize queue with ALL sources, then standard BFS.

**Interview problems**: Rotting Oranges, 01 Matrix, Walls and Gates, Pacific Atlantic Water Flow

## Company Targeting

| Company | Favorite Variant | Difficulty |
|---------|-----------------|------------|
| Google | Dijkstra variants, topological sort | Hard |
| Meta | BFS/DFS on grids, clone graph | Medium |
| Amazon | Course schedule (topo sort) | Medium |
| Anthropic | Graph as distributed system model | Hard |
| NVIDIA | Parallel BFS, graph partitioning | Hard |
| Palantir | Entity relationship graphs | Medium-Hard |

## Graph Representation Cheat Sheet

| Format | Best For | Space |
|--------|----------|-------|
| Adjacency List | Sparse graphs (most interviews) | O(V + E) |
| Adjacency Matrix | Dense graphs, quick edge lookup | O(V²) |
| Edge List | Union-Find problems | O(E) |
| Implicit Graph | Grid/matrix problems | O(1) extra |
