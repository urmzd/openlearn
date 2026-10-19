# Citadel / Citadel Securities Software Engineer Interview Guide

Comprehensive preparation for Citadel (hedge fund) and Citadel Securities (market maker) SWE roles. Known for some of the **hardest algorithmic interviews** in finance.

## Interview Process Overview

Timeline: **3-5 weeks**, **4-5 rounds**

| Round | Format | Duration | Focus |
|-------|--------|----------|-------|
| Online Assessment | HackerRank | 90 min | 3 LC medium-hard problems |
| Phone Screen | Technical | 45-60 min | Algorithms + systems |
| Onsite 1 | Coding | 60 min | Hard algorithms |
| Onsite 2 | System Design | 60 min | Low-latency, financial systems |
| Onsite 3 | Behavioral + Technical | 45 min | Culture, problem-solving approach |

Citadel Securities (market making) is more systems/C++-focused. Citadel (hedge fund) is more data/ML-focused.

## Compensation

- **Junior SWE**: ~$250-400K total comp (Year 1)
- **Senior SWE**: ~$500K-1.5M+ total comp
- **Bonus**: Can be 3-10x base salary in good years
- Citadel Securities tends to pay slightly higher than Citadel hedge fund side

## Key Themes

1. **Algorithmic intensity** -- Citadel's coding rounds are among the hardest. Expect LC hard level, especially string/DP/math problems.
2. **C++ for Citadel Securities** -- Market making systems are C++. Deep language knowledge expected.
3. **Python for Citadel hedge fund** -- Data pipelines, ML, research tools.
4. **Low-latency obsession** -- Microsecond-level performance. Every allocation, every cache miss matters.
5. **Mathematical rigor** -- More quantitative than most tech companies.

## Online Assessment

### Format

- **Platform**: HackerRank
- **Duration**: 90 minutes
- **Problems**: 3 problems, typically medium-hard to hard difficulty
- **Language**: Most languages accepted (C++, Python, Java common)

### Problem Difficulty

Citadel's OA is significantly harder than most companies. Expect:

- String manipulation with complex constraints
- DP on strings/sequences
- Graph problems with optimization
- Mathematical/number theory problems

### Example Problem Types

#### String DP

```python
def min_insertions_palindrome(s: str) -> int:
    """Minimum insertions to make string a palindrome.
    Equivalent to len(s) - LCS(s, reverse(s))."""
    n = len(s)
    rev = s[::-1]

    # LCS DP
    dp = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if s[i-1] == rev[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return n - dp[n][n]
```

#### Combinatorial Optimization

```python
def max_profit_job_scheduling(jobs: list[tuple[int, int, int]]) -> int:
    """Maximum profit from non-overlapping jobs.
    jobs: [(start, end, profit), ...]"""
    jobs.sort(key=lambda x: x[1])  # Sort by end time
    n = len(jobs)
    dp = [0] * (n + 1)

    for i in range(1, n + 1):
        start, end, profit = jobs[i-1]
        # Binary search for latest non-conflicting job
        lo, hi = 0, i - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if jobs[mid-1][1] <= start:
                lo = mid
            else:
                hi = mid - 1

        dp[i] = max(dp[i-1], dp[lo] + profit)

    return dp[n]
```

#### Number Theory

```python
def count_primes_in_range(left: int, right: int) -> int:
    """Count primes in [left, right] using segmented sieve."""
    import math

    limit = int(math.isqrt(right)) + 1
    # Simple sieve up to sqrt(right)
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False

    small_primes = [i for i in range(2, limit + 1) if is_prime[i]]

    # Segmented sieve for [left, right]
    segment = [True] * (right - left + 1)
    if left <= 1:
        segment[1 - left] = False
    if left == 0:
        segment[0] = False

    for p in small_primes:
        start = max(p * p, ((left + p - 1) // p) * p)
        for j in range(start, right + 1, p):
            segment[j - left] = False

    return sum(segment)
```

## Coding Rounds (Onsite)

### What to Expect

- **Hard** problems. Genuinely hard, not "hard with a trick."
- May involve multiple algorithmic techniques combined
- Clean, efficient code expected
- Complexity analysis required
- C++ specific: May test template metaprogramming, move semantics, memory management

### C++ Specific Topics (Citadel Securities)

```cpp
// Topics they may test:

// 1. Move semantics
class OrderBook {
    std::vector<Order> orders_;
public:
    void add_order(Order&& order) {
        orders_.push_back(std::move(order));
    }
};

// 2. Custom allocators / memory pools
template<typename T, size_t BlockSize = 4096>
class PoolAllocator {
    // Pre-allocate blocks, avoid malloc in hot path
};

// 3. Lock-free patterns
std::atomic<int> sequence_number{0};
int next_seq() {
    return sequence_number.fetch_add(1, std::memory_order_relaxed);
}

// 4. Cache-friendly data structures
// Struct of Arrays (SoA) vs Array of Structs (AoS)
// SoA is better for iterating over single fields
struct MarketData_SoA {
    std::vector<double> prices;
    std::vector<int> volumes;
    std::vector<int64_t> timestamps;
};

// 5. SIMD-friendly code
// Align data, avoid branches, use contiguous memory
```

## System Design Round

### Common Topics

#### Design an Order Matching Engine

```
[Order Gateway] --> [Pre-trade Risk Check] --> [Order Book]
                                                    |
                                             [Match Engine]
                                                    |
                                        [Trade Confirmation]
                                                    |
                              +----------+----------+----------+
                              |          |          |          |
                         [Execution]  [Clearing]  [Market Data]  [Risk Update]
```

- **Price-time priority**: Orders matched by best price, then earliest timestamp
- **Latency target**: < 1 microsecond for matching
- **Lock-free matching**: SPSC queues, no mutex in hot path
- **Deterministic**: Same input sequence must produce same output
- **Multicast output**: Trade and market data published via multicast

#### Design a Real-Time P&L System

- **Position tracking**: Real-time inventory across all instruments
- **Mark-to-market**: Continuously revalue positions with latest prices
- **Greeks**: Compute option sensitivities (delta, gamma, vega, theta)
- **Aggregation**: Roll up P&L by desk, strategy, asset class
- **Alerting**: Trigger alerts on P&L thresholds

#### Design a Market Data Infrastructure

- **Ingestion**: Multiple exchange feeds (10M+ messages/sec aggregate)
- **Normalization**: Convert exchange-specific formats to unified schema
- **Conflation**: During bursts, send only latest quote (configurable per consumer)
- **Distribution**: Multicast for co-located systems, TCP for remote
- **Storage**: Tick database for historical replay (petabytes)

### Performance Concepts

| Concept | Why It Matters |
|---------|---------------|
| Cache line alignment | Avoid false sharing between cores |
| Branch prediction | Minimize unpredictable branches in hot paths |
| Memory-mapped I/O | Bypass kernel for shared memory communication |
| Kernel bypass (DPDK) | Avoid kernel network stack overhead |
| CPU pinning | Dedicate cores to critical threads |
| NUMA awareness | Allocate memory close to the CPU using it |
| Lock-free queues | Avoid mutex overhead in producer-consumer |
| Huge pages | Reduce TLB misses for large data structures |

## Preparation Tips

1. **LC hard problems** -- Practice 50+ hard problems. Citadel draws from the hardest pool.
2. **DP mastery** -- String DP, interval DP, bitmask DP, tree DP. Know them all.
3. **C++ depth** (for Securities) -- Modern C++ (17/20), templates, move semantics, smart pointers, atomics.
4. **Low-latency patterns** -- Lock-free programming, memory pools, cache-friendly design.
5. **Financial math** -- Black-Scholes basics, Greeks, order book mechanics.
6. **Practice under pressure** -- Time yourself strictly. Citadel's OA is 30 min per hard problem.
7. **Competitive programming** -- Citadel recruits heavily from CP backgrounds. Codeforces/USACO experience helps.

## Sources

- [Glassdoor - Citadel Software Developer Interview Questions](https://www.glassdoor.com/Interview/Citadel-Software-Developer-Interview-Questions-EI_IE14937.0,7_KO8,26.htm)
- [Top Quant Trading Firms Tier List - QuantVPS](https://www.quantvps.com/blog/top-quant-trading-firms)
- [Quant Firm Tier List - WallStreetQuants](https://www.thewallstreetquants.com/firm-list)
- [DE Shaw vs Two Sigma vs Citadel - Wall Street Oasis](https://www.wallstreetoasis.com/forum/hedge-fund/de-shaw-vs-two-sigma-vs-citadel-offer-advice)
- [Quant Funds Revealed: Careers, Salaries & Recruiting - M&I](https://mergersandinquisitions.com/quant-funds/)
- Blind, r/cscareerquestions (community reports)
