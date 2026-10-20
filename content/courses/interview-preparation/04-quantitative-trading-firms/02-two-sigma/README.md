# Two Sigma

Two Sigma is one of the most prestigious quantitative hedge funds, managing roughly \$60B+ in assets with a technology-first approach to trading. It uses machine learning, distributed systems, and massive data processing to find and act on signals in markets, so its software engineers spend much of their time building data infrastructure: pipelines, real-time processing, and storage for financial time series. The interviews reflect that. Compared with other quant firms, they lean toward traditional computer science fundamentals, heavy on dynamic programming and graphs, combined with data-system design and a round of practical probability.

This chapter takes you through the process in order. After the pipeline and compensation, it sets out the themes that run through every round. It then works the coding round in depth, with two dynamic programming solutions, two graph solutions, and a time-series data structure, each explained so you can reproduce and defend it. The system design section covers market data, backtesting, and real-time risk. The quantitative section works three representative problems by hand, and the chapter ends with the behavioral round and a preparation plan.

> **Learning objectives**
> - Describe Two Sigma's interview process and how it differs from other quant firms.
> - Solve and explain the reference problems: counting longest increasing subsequences, minimum-cost grid paths with obstacles, finding bridges with Tarjan's algorithm, cheapest flights with at most $k$ stops, and a time-series store with binary-search range queries.
> - Design a market data system, a backtesting platform, or a real-time risk system, naming the correctness properties each must preserve.
> - Work conditional-probability, expected-value, and estimation questions out loud.

## Before you begin

The coding round assumes you already know dynamic programming and graph search; the [Programming and Algorithms](../../../programming-and-algorithms/README.md) book covers both from first principles. The quantitative round uses conditional probability and expectation, which [Mathematics](../../../mathematics/README.md) covers. For the design round, read the [Distributed Systems](../../01-shared-concepts/01-distributed-systems.md) chapter of this book and, if you want more depth on pipelines and storage, the [Data Engineering](../../../data-engineering/README.md) book.

## In this chapter

- [The interview process](#the-interview-process) and [Compensation](#compensation): six rounds over three to six weeks, and a bonus-heavy pay structure.
- [What Two Sigma is looking for](#what-two-sigma-is-looking-for): five themes, from strong fundamentals to collaboration with researchers.
- [The coding rounds](#the-coding-rounds): dynamic programming, graph algorithms, and a time-series database.
- [The system design round](#the-system-design-round): market data, backtesting, and real-time risk.
- [The quantitative round](#the-quantitative-round): probability, expected value, and estimation, worked by hand.
- [The behavioral round](#the-behavioral-round) and [Preparing well](#preparing-well): culture, common questions, and where to spend your time.

## The interview process

Expect three to six weeks and six rounds after the recruiter screen:

| Round | Format | Duration | Focus |
|-------|--------|----------|-------|
| Recruiter Screen | Phone | 30 min | Background, motivation |
| Phone Screen 1 | Coding (HackerRank) | 60 min | DS&A, medium-hard |
| Phone Screen 2 | Technical | 60 min | Systems + math reasoning |
| Onsite 1 | Coding | 60-90 min | Algorithms, DP, graph problems |
| Onsite 2 | System Design | 60-90 min | Low-latency data systems |
| Onsite 3 | Quantitative / Math | 60 min | Probability, statistics |
| Onsite 4 | Behavioral | 45 min | Culture fit, intellectual curiosity |

Remote interviews run over Google Meet or Microsoft Teams, and the meeting links arrive the evening before.

## Compensation

Pay is base salary plus a discretionary bonus, and the bonus dominates: it can be two to five times base, so a great year dramatically raises total compensation. Reported totals are roughly \$300-400K for a junior engineer in year one and \$500K-1M+ for a senior engineer.

## What Two Sigma is looking for

Five themes run through the process. Two Sigma is a technology-first hedge fund that trades with ML, distributed systems, and massive data processing, so it hires engineers who can build at scale. It expects strong CS fundamentals; the interviews are more traditional data structures and algorithms than Jane Street's, which lean toward functional programming and mathematics, so expect DP, graphs, and systems problems. Much of the engineering work is data infrastructure: pipelines, real-time processing, and storage systems. Quantitative thinking still matters; software engineering interviews are less math-heavy than those for quant researchers, but probability and estimation are part of the loop. And the culture is collaborative, with engineers working closely alongside researchers and modelers.

## The coding rounds

The coding problems are medium-hard. Dynamic programming is heavily favored, graph algorithms appear often, and some problems are systems-oriented: designing a data structure for a particular access pattern. Python, Java, or C++ are the preferred languages. The listings below cover each of the three families.

### Dynamic programming

The first problem asks not for the length of the longest increasing subsequence but for how many longest increasing subsequences there are. The standard $O(n^2)$ LIS recurrence extends naturally: alongside `lengths[i]`, the length of the longest increasing subsequence ending at index `i`, keep `counts[i]`, the number of subsequences that achieve that length. For each earlier index `j` with `nums[j] < nums[i]`, extending from `j` gives length `lengths[j] + 1`. If that beats the best so far, it replaces both the length and the count; if it ties, its count is added. The answer sums the counts of every index whose length equals the overall maximum.

The second problem finds the minimum-cost path from the top-left to the bottom-right of a grid, moving only right or down, where `-1` marks an obstacle and every other value is the cost of entering that cell. Each cell's best cost is its own cost plus the cheaper of the cell above and the cell to the left, skipping obstacles and unreachable neighbors (those still at infinity). Filling the table row by row guarantees both neighbors are final before they are used. The answer is the bottom-right entry, or `-1` if it was never reached or the start is blocked.

```python
def longest_increasing_subsequence_count(nums: list[int]) -> int:
    """Count the number of longest increasing subsequences."""
    if not nums:
        return 0
    n = len(nums)
    lengths = [1] * n  # LIS length ending at i
    counts = [1] * n   # Number of LIS of that length ending at i

    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                if lengths[j] + 1 > lengths[i]:
                    lengths[i] = lengths[j] + 1
                    counts[i] = counts[j]
                elif lengths[j] + 1 == lengths[i]:
                    counts[i] += counts[j]

    max_len = max(lengths)
    return sum(c for l, c in zip(lengths, counts) if l == max_len)


def min_cost_path_with_obstacles(grid: list[list[int]]) -> int:
    """Find minimum cost path from top-left to bottom-right.
    -1 = obstacle, otherwise cell value is cost."""
    rows, cols = len(grid), len(grid[0])
    INF = float('inf')
    dp = [[INF] * cols for _ in range(rows)]

    if grid[0][0] == -1:
        return -1
    dp[0][0] = grid[0][0]

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == -1:
                continue
            if r > 0 and dp[r-1][c] != INF:
                dp[r][c] = min(dp[r][c], dp[r-1][c] + grid[r][c])
            if c > 0 and dp[r][c-1] != INF:
                dp[r][c] = min(dp[r][c], dp[r][c-1] + grid[r][c])

    return dp[rows-1][cols-1] if dp[rows-1][cols-1] != INF else -1
```

Both run in polynomial time: the LIS count is $O(n^2)$ time and $O(n)$ space, and the grid is $O(RC)$. Worth mentioning in the interview: strict `<` means equal values do not extend a subsequence, and the grid solution would need a different algorithm (Dijkstra) if moves in all four directions were allowed, because the row-by-row order would no longer finalize neighbors in time.

### Graph algorithms

The first graph problem finds every critical connection, or bridge: an edge whose removal disconnects the graph. Tarjan's algorithm finds them all in one depth-first search, in $O(V + E)$ time. It records each vertex's discovery time `disc[u]` and a low-link value `low[u]`, the earliest discovery time reachable from `u`'s DFS subtree using at most one back edge. After exploring a child `v`, if `low[v] > disc[u]` then nothing in `v`'s subtree can reach `u` or any of its ancestors except through the edge `(u, v)`, so that edge is a bridge. A visited neighbor that is not the parent is a back edge, and it lowers `low[u]` to that neighbor's discovery time.

The second finds the cheapest flight route from `src` to `dst` using at most `k` stops, which is at most `k + 1` edges. A modified Bellman-Ford runs exactly `k + 1` rounds of relaxation. The important detail is `new_prices = prices[:]`: every relaxation in a round reads from the previous round's prices, so each round extends paths by exactly one edge. Relaxing in place would let a single round chain several edges together and silently exceed the stop limit.

```python
from collections import defaultdict, deque

def find_critical_connections(n: int, connections: list[list[int]]) -> list[list[int]]:
    """Find all bridges in an undirected graph (Tarjan's algorithm)."""
    graph = defaultdict(list)
    for u, v in connections:
        graph[u].append(v)
        graph[v].append(u)

    disc = [0] * n
    low = [0] * n
    visited = [False] * n
    bridges = []
    timer = [1]

    def dfs(u, parent):
        visited[u] = True
        disc[u] = low[u] = timer[0]
        timer[0] += 1

        for v in graph[u]:
            if not visited[v]:
                dfs(v, u)
                low[u] = min(low[u], low[v])
                if low[v] > disc[u]:
                    bridges.append([u, v])
            elif v != parent:
                low[u] = min(low[u], disc[v])

    for i in range(n):
        if not visited[i]:
            dfs(i, -1)

    return bridges


def shortest_path_with_k_stops(n: int, flights: list, src: int, dst: int, k: int) -> int:
    """Cheapest flight with at most k stops (modified Bellman-Ford)."""
    INF = float('inf')
    prices = [INF] * n
    prices[src] = 0

    for _ in range(k + 1):
        new_prices = prices[:]
        for u, v, cost in flights:
            if prices[u] != INF and prices[u] + cost < new_prices[v]:
                new_prices[v] = prices[u] + cost
        prices = new_prices

    return prices[dst] if prices[dst] != INF else -1
```

The flight search costs $O(k \cdot E)$. Two edge cases are worth raising about the bridge finder. Skipping the parent by vertex means two parallel edges between the same pair would be misreported as a bridge; tracking the parent edge's index instead fixes that. And Python's default recursion limit of 1,000 will overflow on a long path graph, so for large inputs you either raise the limit or rewrite the DFS iteratively.

### Data structure design

The systems-flavored problem is a small time-series database: insert `(timestamp, value)` points per metric, return all points in a time range, and aggregate them. The design keeps each metric's points in a list sorted by timestamp. Insertion binary-searches for the insertion point, which tolerates out-of-order arrivals, and range queries use two binary searches: `_bisect_left` finds the first point at or after `start`, and `_bisect_right` finds the first point after `end`, so the slice between them is exactly the closed range $[\text{start}, \text{end}]$.

```python
import heapq
from collections import defaultdict

class TimeSeriesDB:
    """Time series database optimized for range queries and aggregations."""

    def __init__(self):
        self.series: dict[str, list[tuple[int, float]]] = defaultdict(list)

    def insert(self, metric: str, timestamp: int, value: float):
        """Insert a data point. Assumes roughly time-ordered insertion."""
        series = self.series[metric]
        # Binary search for insertion point (handle out-of-order)
        lo, hi = 0, len(series)
        while lo < hi:
            mid = (lo + hi) // 2
            if series[mid][0] < timestamp:
                lo = mid + 1
            else:
                hi = mid
        series.insert(lo, (timestamp, value))

    def query_range(self, metric: str, start: int, end: int) -> list[tuple[int, float]]:
        """Return all points in [start, end]."""
        series = self.series.get(metric, [])
        # Binary search for start
        lo = self._bisect_left(series, start)
        hi = self._bisect_right(series, end)
        return series[lo:hi]

    def aggregate(self, metric: str, start: int, end: int, func: str) -> float | None:
        """Aggregate values in range. func: 'avg', 'sum', 'min', 'max', 'count'."""
        points = self.query_range(metric, start, end)
        if not points:
            return None
        values = [v for _, v in points]
        if func == "avg":
            return sum(values) / len(values)
        elif func == "sum":
            return sum(values)
        elif func == "min":
            return min(values)
        elif func == "max":
            return max(values)
        elif func == "count":
            return len(values)

    def _bisect_left(self, series, timestamp):
        lo, hi = 0, len(series)
        while lo < hi:
            mid = (lo + hi) // 2
            if series[mid][0] < timestamp:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def _bisect_right(self, series, timestamp):
        lo, hi = 0, len(series)
        while lo < hi:
            mid = (lo + hi) // 2
            if series[mid][0] <= timestamp:
                lo = mid + 1
            else:
                hi = mid
        return lo
```

Range queries cost $O(\log n + m)$ for $m$ returned points, and aggregation adds $O(m)$. The weak spot is insertion: finding the position is $O(\log n)$, but `list.insert` shifts every later element, so an out-of-order point costs $O(n)$. That is why the docstring assumes roughly time-ordered arrivals, where the insertion point is at or near the end. Good follow-ups to anticipate: precomputed prefix sums or a segment tree to answer `sum` and `avg` in $O(\log n)$, time-bucketed chunks to bound the insertion cost, and explicit rejection of an unknown `func` (the listing silently returns `None`). The `heapq` import is unused; you could use it for a top-$k$ query if asked.

## The system design round

System design at Two Sigma is heavy on data infrastructure. Expect prompts about real-time market data processing (ingesting, normalizing, storing, and serving financial data), time-series databases, low-latency pipelines that process events in single-digit milliseconds, and research platforms that let quant researchers backtest strategies efficiently.

### Designing a market data system

Exchange feeds arrive through feed handlers, are normalized into a common format, deduplicated and sequenced, and then fan out to an in-memory real-time store, a compressed columnar historical store, and an analytics engine. Those in turn serve live dashboards, backtesting, and signal generation:

```
[Exchange Feeds] --> [Feed Handlers] --> [Normalizer]
                                              |
                                        [Dedup / Sequencer]
                                              |
                              +---------------+---------------+
                              |               |               |
                        [Real-time Store]  [Historical Store]  [Analytics Engine]
                        (in-memory)        (columnar, compressed)
                              |               |               |
                        [Live Dashboard]  [Backtesting]    [Signal Generation]
```

The constraints shape every box. Market data must be processed in microseconds, at volumes of millions of messages per second across exchanges. Messages must be processed in the correct sequence, which is why the sequencer sits before the fan-out. Historical data must be replayable for backtesting. And exchanges add new fields, so the schema has to evolve without breaking consumers.

### Designing a backtesting platform

A backtest runs a strategy definition over historical data in a simulation engine, which feeds an order simulator, a P&L calculator, risk metrics, and finally a results dashboard:

```
[Strategy Definition] --> [Historical Data] --> [Simulation Engine]
                                                       |
                                                 [Order Simulator]
                                                       |
                                                 [P&L Calculator]
                                                       |
                                                 [Risk Metrics]
                                                       |
                                                 [Results Dashboard]
```

The hard requirements are about trustworthiness. Point-in-time correctness means the strategy sees only data that was available at each simulated moment; any leak of future data makes results look better than reality. Slippage modeling simulates realistic execution, since you cannot always trade at the price you want. Parallelization lets thousands of strategy variants run concurrently. And reproducibility means the same inputs always produce the same outputs.

### Designing a real-time risk system

A risk system tracks positions and marks them to market in real time across the whole portfolio. It computes Value at Risk (VaR) with Monte Carlo simulation, monitors limits and alerts when exposure crosses a threshold, tracks cross-asset correlations as they shift, and runs stress tests: what-if scenarios such as a market crash or a liquidity crisis.

## The quantitative round

The quantitative round is lighter than a quant researcher's interview but still substantial. It covers basic probability and statistics, expected-value calculations, combinatorics, and estimation (Fermi) questions. The three problems below show the expected level; work each one out loud.

### Conditional probability

*Two dice are rolled. Given that the sum is at least 7, what is the probability that it is exactly 9?*

Restrict the sample space to the condition. Of the 36 equally likely outcomes, those with sum at least 7 are:

```
Sum >= 7 outcomes: (1,6),(2,5),(2,6),(3,4),(3,5),(3,6),(4,3),(4,4),(4,5),(4,6),
                   (5,2),(5,3),(5,4),(5,5),(5,6),(6,1),(6,2),(6,3),(6,4),(6,5),(6,6)
= 21 outcomes

Sum = 9 outcomes: (3,6),(4,5),(5,4),(6,3)
= 4 outcomes

P(9 | >= 7) = 4/21 ≈ 0.190
```

A quicker count: sums 7 through 12 occur in $6+5+4+3+2+1 = 21$ ways, and a sum of 9 in 4 of them, so $P(9 \mid \ge 7) = 4/21 \approx 0.190$.

### Expected value

*You draw cards from a standard deck until you get an ace. How many cards do you expect to draw?*

Use symmetry. The 4 aces divide the 48 non-aces into 5 groups (before the first ace, between consecutive aces, and after the last), and by symmetry each group has the same expected size, $48/5$. You draw the whole first group and then the first ace:

$$E[\text{draws}] = \frac{48}{5} + 1 = \frac{53}{5} = 10.6.$$

### Estimation

*How much data does a major stock exchange produce per day?*

State your assumptions and multiply:

```
NYSE: ~6.5 hours of trading per day
Symbols: ~3,000 listed
Messages per symbol: ~10,000 per day (quotes, trades, updates)
Message size: ~100 bytes

Data per day ≈ 3,000 × 10,000 × 100 bytes = 3 GB

But: Market data includes derivatives, options = 10x more symbols
Total: ~30-50 GB/day raw data (compressed: ~5-10 GB)
```

The interviewer cares less about the final number than about clear assumptions and the instinct to revisit them, as in the step that notices derivatives and options multiply the symbol count.

## The behavioral round

Two Sigma's culture values intellectual curiosity (people who love learning), collaboration (engineers work closely with quant researchers), rigor (decisions are data-driven, not opinion-driven), and humility, since markets are humbling and ego gets in the way. Prepare specific stories for the common questions:

- "Why Two Sigma over a tech company?"
- "Tell me about the most complex system you've built."
- "Describe a time you had to explain a technical concept to a non-technical person."
- "What's a hard problem you solved with data?"
- "How do you stay current with technology trends?"

## Preparing well

Put dynamic programming first: Two Sigma loves it, and working 30-40 DP problems of varying difficulty is a reasonable target. Then cover graph algorithms: BFS, DFS, shortest paths, topological sort, and bridges and articulation points. Learn to think in time series, meaning how to store and query time-ordered data efficiently. Refresh probability basics: Bayes' rule, expected value, variance, and the common distributions. For the design round, practice data pipelines that handle millions of events per second, and understand why latency matters in trading and how to minimize it. Finally, read what Two Sigma publishes in its [articles](https://www.twosigma.com/articles/); its engineering content is excellent and gives you material for "Why Two Sigma?"

## Key terms

- **Dynamic programming**: solving a problem by combining solutions to overlapping subproblems stored in a table.
- **Bridge (critical connection)**: an edge whose removal increases the number of connected components.
- **Low-link value**: in Tarjan's algorithm, the earliest discovery time reachable from a vertex's DFS subtree through at most one back edge.
- **Bellman-Ford**: a shortest-path algorithm that relaxes every edge in rounds; limiting the rounds limits the number of edges per path.
- **Point-in-time correctness**: the backtesting rule that a strategy may only use data available at each simulated moment.
- **Value at Risk (VaR)**: a loss threshold that the portfolio is not expected to exceed over a period at a given confidence level.
- **Slippage**: the difference between the expected price of a trade and the price actually obtained.

## Summary

- Two Sigma's six rounds emphasize traditional CS fundamentals, data-infrastructure design, and practical probability, more than the functional and mathematical focus of some other quant firms.
- DP and graphs dominate coding: know LIS counting, grid DP, Tarjan's bridges, and round-limited Bellman-Ford, and be able to explain each invariant.
- Design questions revolve around market data, backtesting, and risk, with ordering, replay, point-in-time correctness, and reproducibility as the key properties.
- Quantitative questions reward restricting the sample space, using symmetry, and stating estimation assumptions clearly.
- Compensation is bonus-heavy, and the culture prizes curiosity, collaboration, rigor, and humility.

## Review questions

1. In the LIS-count solution, why does a tie add `counts[j]` to `counts[i]` while an improvement replaces it?
   <details><summary>Answer</summary>`counts[i]` counts subsequences of the current best length ending at `i`. A longer extension makes all previously counted subsequences non-maximal, so the count resets to the number of ways to reach `j`. A tie gives additional distinct subsequences of the same best length, so their count is added.</details>
2. Why does the cheapest-flights solution copy `prices` at the start of each round?
   <details><summary>Answer</summary>Each round must extend paths by exactly one edge. Reading from the previous round's prices while writing to the copy ensures that; relaxing in place could chain several edges in one round and return a path with more than `k` stops.</details>
3. State the bridge condition in Tarjan's algorithm and explain it.
   <details><summary>Answer</summary>Tree edge `(u, v)` is a bridge if `low[v] > disc[u]`. It means no vertex in `v`'s DFS subtree has a back edge to `u` or any ancestor of `u`, so removing `(u, v)` cuts the subtree off from the rest of the graph.</details>
4. What is the cost of inserting an out-of-order point into the `TimeSeriesDB`, and how would you improve it?
   <details><summary>Answer</summary>The binary search is $O(\log n)$, but `list.insert` shifts later elements, so the insert is $O(n)$. Splitting each series into time-bucketed chunks, or using a balanced tree or skip list, bounds the cost; in practice, mostly ordered arrivals keep inserts near the end and cheap.</details>
5. Why is point-in-time correctness the most important property of a backtesting platform?
   <details><summary>Answer</summary>If a strategy can see data that was not yet available at the simulated time (look-ahead bias), the backtest overstates performance, and capital gets allocated to a strategy that cannot work in live trading.</details>
6. Draw cards until the first ace. Explain the symmetry argument for $E = 10.6$.
   <details><summary>Answer</summary>The 4 aces split the 48 non-aces into 5 gaps, and by symmetry each gap has expected size $48/5 = 9.6$. You draw the entire first gap and then the ace, so $9.6 + 1 = 10.6$.</details>

## Further reading

- [Interviewing for Software Engineering - Two Sigma](https://www.twosigma.com/careers/interviewing-at-two-sigma/interviewing-for-software-engineering/)
- [Two Sigma Software Engineer Interview Guide - InterviewQuery](https://www.interviewquery.com/interview-guides/two-sigma-software-engineer)
- [Two Sigma Articles](https://www.twosigma.com/articles/)
- [Glassdoor - Two Sigma SWE Interview Questions](https://www.glassdoor.com/Interview/Two-Sigma-Software-Engineer-Interview-Questions-EI_IE241045.0,9_KO10,27.htm)
- [DE Shaw vs Two Sigma vs Citadel - Wall Street Oasis](https://www.wallstreetoasis.com/forum/hedge-fund/de-shaw-vs-two-sigma-vs-citadel-offer-advice)
- [Top Quant Trading Firms Tier List - QuantVPS](https://www.quantvps.com/blog/top-quant-trading-firms)
- Community reports on Blind and r/cscareerquestions
