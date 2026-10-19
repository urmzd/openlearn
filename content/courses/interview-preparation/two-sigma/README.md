# Two Sigma Software Engineer Interview Guide

Comprehensive preparation for Two Sigma SWE roles. Two Sigma is one of the most prestigious quantitative hedge funds, managing ~$60B+ in assets with a technology-first approach.

## Interview Process Overview

Timeline: **3-6 weeks**, **5-6 rounds**

| Round | Format | Duration | Focus |
|-------|--------|----------|-------|
| Recruiter Screen | Phone | 30 min | Background, motivation |
| Phone Screen 1 | Coding (HackerRank) | 60 min | DS&A, medium-hard |
| Phone Screen 2 | Technical | 60 min | Systems + math reasoning |
| Onsite 1 | Coding | 60-90 min | Algorithms, DP, graph problems |
| Onsite 2 | System Design | 60-90 min | Low-latency data systems |
| Onsite 3 | Quantitative / Math | 60 min | Probability, statistics |
| Onsite 4 | Behavioral | 45 min | Culture fit, intellectual curiosity |

Remote interviews are conducted via Google Meet or Microsoft Teams. Meeting links are sent the evening prior.

## Compensation

- **Junior SWE**: ~$300-400K total comp (Year 1)
- **Senior SWE**: ~$500K-1M+ total comp
- **Comp structure**: Base salary + discretionary bonus (bonus can be 2-5x base)
- Bonus-heavy: A great year can dramatically increase total comp

## Key Themes

1. **Technology-first hedge fund** -- Two Sigma uses ML, distributed systems, and massive data processing for trading. They hire engineers who can build at scale.
2. **Strong CS fundamentals** -- More traditional DS&A than Jane Street (which leans functional/math). Expect DP, graphs, and systems problems.
3. **Data infrastructure** -- Much of the engineering work is data pipelines, real-time processing, and storage systems.
4. **Quantitative thinking** -- While less math-heavy than quant researcher roles, SWE interviews still include probability and estimation.
5. **Collaborative culture** -- Two Sigma emphasizes collaboration between engineers, researchers, and modelers.

## Coding Rounds

### What to Expect

- Medium-hard algorithmic problems
- Dynamic programming is heavily favored
- Graph algorithms appear frequently
- Systems-oriented coding (data structures for specific access patterns)
- Language: Python, Java, or C++ preferred

### Reported Problem Types

#### Dynamic Programming (Very Common)

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

#### Graph Problems

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

#### Data Structure Design

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

## System Design Round

### Two Sigma-Specific Focus

System design at Two Sigma is data-infrastructure-heavy:

- **Real-time market data processing** -- Ingest, normalize, store, and serve financial data
- **Time series databases** -- Efficient storage and querying of financial time series
- **Low-latency data pipelines** -- Process events in single-digit milliseconds
- **Research platform** -- Enable quant researchers to backtest strategies efficiently

### Common Topics

#### Design a Market Data System

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

Key considerations:
- **Latency**: Market data must be processed in microseconds
- **Volume**: Millions of messages per second across exchanges
- **Ordering**: Messages must be processed in correct sequence
- **Replay**: Must be able to replay historical data for backtesting
- **Schema evolution**: New data fields from exchanges

#### Design a Backtesting Platform

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

- **Point-in-time correctness**: Strategy must only see data available at that moment
- **Slippage modeling**: Simulate realistic execution (can't always get the price you want)
- **Parallelization**: Run thousands of strategy variants concurrently
- **Reproducibility**: Same inputs must produce same outputs

#### Design a Real-Time Risk System

- **Position tracking**: Real-time mark-to-market across all positions
- **VaR (Value at Risk)**: Monte Carlo simulation for risk metrics
- **Limit monitoring**: Alert when exposure exceeds thresholds
- **Correlation**: Track cross-asset correlations in real-time
- **Stress testing**: What-if scenarios (market crash, liquidity crisis)

## Quantitative / Math Round

### What to Expect

Lighter than quant researcher interviews but still substantial:

- Basic probability and statistics
- Expected value calculations
- Combinatorics
- Estimation / Fermi questions

### Sample Problems

#### Probability

**"Two dice are rolled. Given that the sum is at least 7, what's the probability it's exactly 9?"**

```
Sum >= 7 outcomes: (1,6),(2,5),(2,6),(3,4),(3,5),(3,6),(4,3),(4,4),(4,5),(4,6),
                   (5,2),(5,3),(5,4),(5,5),(5,6),(6,1),(6,2),(6,3),(6,4),(6,5),(6,6)
= 21 outcomes

Sum = 9 outcomes: (3,6),(4,5),(5,4),(6,3)
= 4 outcomes

P(9 | >= 7) = 4/21 ≈ 0.190
```

#### Expected Value

**"You draw cards from a standard deck until you get an ace. How many cards do you expect to draw?"**

By symmetry, the 4 aces divide the 52 cards into 5 groups. Expected size of first group = 48/5 non-aces, then 1 ace.

E[draws] = 48/5 + 1 = 53/5 = 10.6

#### Estimation

**"How much data does a major stock exchange produce per day?"**

```
NYSE: ~6.5 hours of trading per day
Symbols: ~3,000 listed
Messages per symbol: ~10,000 per day (quotes, trades, updates)
Message size: ~100 bytes

Data per day ≈ 3,000 × 10,000 × 100 bytes = 3 GB

But: Market data includes derivatives, options = 10x more symbols
Total: ~30-50 GB/day raw data (compressed: ~5-10 GB)
```

## Behavioral Round

### Two Sigma Culture

- **Intellectual curiosity** -- They want people who love learning
- **Collaboration** -- Engineers work closely with quant researchers
- **Rigor** -- Decisions are data-driven, not opinion-driven
- **Humility** -- Markets are humbling. Check your ego.

### Common Questions

- "Why Two Sigma over a tech company?"
- "Tell me about the most complex system you've built."
- "Describe a time you had to explain a technical concept to a non-technical person."
- "What's a hard problem you solved with data?"
- "How do you stay current with technology trends?"

## Preparation Tips

1. **DP mastery** -- Two Sigma loves dynamic programming. Practice 30-40 DP problems of varying difficulty.
2. **Graph algorithms** -- BFS, DFS, shortest path, topological sort, bridges/articulation points.
3. **Time series thinking** -- Understand efficient storage and querying of time-ordered data.
4. **Probability basics** -- Bayes, expected value, variance, basic distributions.
5. **Systems at scale** -- Design data pipelines that handle millions of events per second.
6. **Low-latency patterns** -- Understand why latency matters in trading and how to minimize it.
7. **Read about Two Sigma** -- twosigma.com/articles. They publish excellent engineering content.

## Sources

- [Interviewing for Software Engineering - Two Sigma](https://www.twosigma.com/careers/interviewing-at-two-sigma/interviewing-for-software-engineering/)
- [Two Sigma Software Engineer Interview Guide - InterviewQuery](https://www.interviewquery.com/interview-guides/two-sigma-software-engineer)
- [Two Sigma Articles](https://www.twosigma.com/articles/)
- [Glassdoor - Two Sigma SWE Interview Questions](https://www.glassdoor.com/Interview/Two-Sigma-Software-Engineer-Interview-Questions-EI_IE241045.0,9_KO10,27.htm)
- [DE Shaw vs Two Sigma vs Citadel - Wall Street Oasis](https://www.wallstreetoasis.com/forum/hedge-fund/de-shaw-vs-two-sigma-vs-citadel-offer-advice)
- [Top Quant Trading Firms Tier List - QuantVPS](https://www.quantvps.com/blog/top-quant-trading-firms)
- Blind, r/cscareerquestions (community reports)
