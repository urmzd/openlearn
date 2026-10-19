# Interviewing at Citadel and Citadel Securities

Citadel runs some of the hardest algorithmic interviews in finance. Two related firms hire under the name: Citadel, the hedge fund, and Citadel Securities, the market maker. Their loops share a reputation for genuinely hard coding problems, the kind that combine several techniques rather than hinge on one trick, and they diverge in emphasis. Citadel Securities builds market-making systems in C++ and cares about microseconds, while the hedge fund leans toward Python, data pipelines, and ML research tooling. This chapter prepares you for both.

It starts with the process and compensation, then the themes that run through the loop. The core of the chapter is the online assessment, with three worked problems that represent its three main families (string dynamic programming, combinatorial optimization, and number theory), followed by the onsite coding rounds and the C++ topics Citadel Securities tests. It then covers low-latency system design (an order matching engine, a real-time P&L system, and market data infrastructure) and the performance concepts behind them, and ends with a preparation plan.

> **Learning objectives**
> - Describe the Citadel and Citadel Securities loops and how their emphases differ.
> - Solve minimum insertions to a palindrome via longest common subsequence, weighted job scheduling via DP with binary search, and prime counting in a range with a segmented sieve.
> - Explain the C++ topics tested for low-latency roles: move semantics, memory pools, atomics, data layout, and SIMD-friendly code.
> - Design an order matching engine, a real-time P&L system, and a market data pipeline under microsecond constraints.
> - Explain the hardware-level performance techniques (cache alignment, kernel bypass, CPU pinning, NUMA, huge pages) and why each matters.

## Before you begin

This chapter assumes strong algorithms preparation; work through hard dynamic programming before you start. For Citadel Securities roles, you need modern C++ (17 or 20). The lock-free and atomic ideas build on [Concurrency patterns](../anthropic/concepts/concurrency-patterns.md), and the other quantitative-firm chapters in this book, such as [Jane Street](../jane-street/README.md) and [Hudson River Trading](../hrt/README.md), cover overlapping ground on market making and low latency.

## The interview process

The process takes three to five weeks and has four or five rounds:

| Round | Format | Duration | Focus |
|-------|--------|----------|-------|
| Online Assessment | HackerRank | 90 min | 3 LC medium-hard problems |
| Phone Screen | Technical | 45-60 min | Algorithms + systems |
| Onsite 1 | Coding | 60 min | Hard algorithms |
| Onsite 2 | System Design | 60 min | Low-latency, financial systems |
| Onsite 3 | Behavioral + Technical | 45 min | Culture, problem-solving approach |

Citadel Securities, the market maker, is more focused on systems and C++. Citadel, the hedge fund, is more focused on data and ML.

## Compensation

Junior engineers report about $250-400K in total compensation in the first year, and senior engineers about $500K to $1.5M or more. The bonus can be three to ten times base salary in good years. Citadel Securities tends to pay slightly more than the hedge fund side.

## What Citadel is looking for

Five themes run through the loop. **Algorithmic intensity** comes first: the coding rounds are among the hardest anywhere, at LeetCode hard level, with string, DP, and math problems especially common. **C++ for Citadel Securities**: market-making systems are written in C++, and deep language knowledge is expected. **Python for the hedge fund**, where the work is data pipelines, ML, and research tools. A **low-latency obsession**, where performance is measured in microseconds and every allocation and cache miss matters. And **mathematical rigor**: the interviews are more quantitative than at most tech companies.

## The online assessment

The assessment is 90 minutes on HackerRank, with three problems at medium-hard to hard difficulty, in most languages (C++, Python, and Java are common). It is significantly harder than most companies' assessments, and with three problems in 90 minutes you have about 30 minutes for each. Expect string manipulation with complex constraints, DP on strings and sequences, graph problems with an optimization twist, and math or number theory. Three representative problems follow.

### Minimum insertions to make a palindrome

Given a string, find the minimum number of characters to insert to make it a palindrome. The key insight is a reduction. The characters you do *not* need to insert form the longest palindromic subsequence, and every other character needs a mirror inserted. So the answer is $n$ minus the length of the longest palindromic subsequence. And the longest palindromic subsequence of `s` is exactly the longest common subsequence of `s` and its reverse.

That turns the problem into textbook LCS DP: `dp[i][j]` is the LCS length of the first `i` characters of `s` and the first `j` characters of the reverse. Matching characters extend the diagonal; otherwise you take the better of dropping one character from either string.

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

This runs in $O(n^2)$ time and space. For `"mbadm"`, the longest palindromic subsequence is `"mam"` (or `"mbm"`), length 3, so two insertions are needed. If space is a concern, the DP needs only the previous row, which brings space down to $O(n)$.

### Weighted job scheduling

Given jobs as `(start, end, profit)` triples, choose non-overlapping jobs to maximize total profit. This is the classic combinatorial optimization problem, and it yields to DP once the jobs are sorted by end time.

Let `dp[i]` be the best profit using the first `i` jobs in end-time order. For job `i` there are two choices: skip it, keeping `dp[i-1]`; or take it, earning its profit plus the best profit from jobs that end by its start. Because jobs are sorted by end time, those compatible jobs form a prefix, and binary search finds the longest such prefix in $O(\log n)$.

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

The binary search finds the largest `lo` such that the `lo`-th job (one-indexed) ends no later than the current job starts; `lo = 0` means no earlier job is compatible. The upper-middle choice `mid = (lo + hi + 1) // 2` guarantees progress when `lo = mid`. The comparison `<=` treats a job that ends exactly when another starts as compatible; say so, because the problem statement may define it either way. The total cost is $O(n \log n)$.

### Counting primes in a range

Count the primes in $[\text{left}, \text{right}]$, where `right` can be large but the range itself fits in memory. A plain sieve up to `right` would use too much memory, so use a **segmented sieve**. Every composite number in the range has a prime factor no larger than $\sqrt{\text{right}}$. So first sieve the small primes up to $\sqrt{\text{right}}$, then use each small prime $p$ to cross off its multiples within the segment, starting at the larger of $p^2$ and the first multiple of $p$ that is at least `left`. Starting at $p^2$ ensures that $p$ itself is never crossed off when it lies in the range.

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

The expression `((left + p - 1) // p) * p` is the ceiling of `left / p` times `p`, the first multiple of `p` at or above `left`. The special cases mark 0 and 1 as non-prime when they fall in the segment. Time is about $O(\sqrt{R} \log \log R + (R - L) \log \log R)$, and memory is $O(\sqrt{R} + (R - L))$.

> **Warning:** The guard `if left <= 1: segment[1 - left] = False` assumes 1 is inside the segment. For the degenerate input `left = 0, right = 0`, the segment has one slot and index 1 is out of range. Handle `right < 2` up front by returning 0, and mention it: edge cases like this are what separate a hard-problem pass from a fail.

## The onsite coding rounds

The onsite coding problems are hard: genuinely hard, not "hard with a trick," and they may combine several algorithmic techniques. You are expected to write clean, efficient code and to give a complexity analysis. For C++ roles you may also be tested on template metaprogramming, move semantics, and memory management.

### C++ topics for Citadel Securities

Citadel Securities interviewers probe the C++ features that matter in a hot path. The listing below sketches five of them:

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

Be ready to explain each. **Move semantics** let `add_order` take ownership of an order's resources instead of copying them; note that inside the function `order` is an lvalue, so `std::move` is needed again to move it into the vector. A **memory pool** pre-allocates blocks so that the hot path never calls `malloc`, whose latency is unpredictable. A **relaxed atomic** `fetch_add` is enough for a unique sequence number, because uniqueness needs atomicity but not ordering with respect to other memory operations; if other threads must also see data written before the increment, you need acquire-release ordering. **Struct-of-arrays** layout keeps each field contiguous, so a loop that reads only prices streams through cache lines full of prices rather than dragging volumes and timestamps along. And **SIMD-friendly** code is aligned, branch-free, and contiguous, so the compiler can vectorize it.

## The system design round

The design round is about financial systems under extreme latency constraints. Three prompts are commonly reported.

### An order matching engine

The matching engine is the heart of an exchange or a market maker's internal crossing system. Orders pass a risk check, rest in or match against the order book, and the resulting trades fan out to downstream systems:

```text
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

Five properties define a good design. Matching uses **price-time priority**: the best price first, then the earliest timestamp at that price. The **latency target** is under one microsecond for matching. Matching is **lock-free**, using single-producer single-consumer queues with no mutex in the hot path. The engine is **deterministic**: the same input sequence must produce the same output, which makes replay and recovery possible. And trades and market data are published by **multicast**.

### A real-time P&L system

For a real-time profit-and-loss system, cover **position tracking** (real-time inventory across all instruments), **mark-to-market** (continuously revaluing positions at the latest prices), the **Greeks** (option sensitivities: delta, gamma, vega, theta), **aggregation** (rolling P&L up by desk, strategy, and asset class), and **alerting** on P&L thresholds.

### Market data infrastructure

For market data, cover **ingestion** from multiple exchange feeds, at more than 10 million messages per second in aggregate; **normalization** of exchange-specific formats into a unified schema; **conflation**, which during bursts sends a consumer only the latest quote, configurable per consumer; **distribution** by multicast to co-located systems and by TCP to remote ones; and **storage** in a tick database for historical replay, at petabyte scale.

### Performance concepts

Every one of these designs depends on hardware-level performance techniques. Know what each does and why it matters:

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

## Preparing well

Work 50 or more LeetCode hard problems, because Citadel draws from the hardest pool. Master DP in all its forms: string, interval, bitmask, and tree DP. For Securities roles, build C++ depth in modern C++ (17 and 20): templates, move semantics, smart pointers, and atomics. Learn the low-latency patterns: lock-free programming, memory pools, and cache-friendly design. Learn the financial math basics: Black-Scholes, the Greeks, and order book mechanics. Practice under strict time pressure, at about 30 minutes per hard problem as in the assessment. And consider competitive programming: Citadel recruits heavily from that background, and Codeforces or USACO experience helps.

## Key terms

- **Market maker**: a firm that continuously quotes buy and sell prices and profits from the spread; Citadel Securities is one.
- **Longest common subsequence (LCS)**: the longest sequence appearing in order, not necessarily contiguously, in two strings.
- **Segmented sieve**: a prime sieve over a range $[L, R]$ that uses only primes up to $\sqrt{R}$.
- **Price-time priority**: the matching rule that fills the best price first and, at equal prices, the earliest order first.
- **Conflation**: replacing a burst of quote updates with only the latest one for consumers that cannot keep up.
- **False sharing**: cores contending for a cache line because unrelated variables they write share it.

## Summary

- Citadel's coding bar is LeetCode hard, especially string DP, combinatorial optimization, and number theory, at about 30 minutes per problem in the assessment.
- Minimum insertions to a palindrome is $n - \text{LCS}(s, \text{reverse}(s))$; weighted job scheduling is DP over end-sorted jobs with binary search; prime counting in a range uses a segmented sieve.
- Citadel Securities tests C++ for the hot path: moves, pools, atomics with the right memory order, struct-of-arrays layout, and vectorizable code.
- Matching engines need price-time priority, sub-microsecond lock-free matching, determinism, and multicast output.
- Low latency comes from the hardware up: cache alignment, branch prediction, kernel bypass, pinning, NUMA, and huge pages.

## Review questions

1. Why does the minimum number of insertions to make `s` a palindrome equal `len(s) - LCS(s, reverse(s))`?
   <details><summary>Answer</summary>The longest palindromic subsequence can stay in place, and every other character needs a matching character inserted on the opposite side. The longest palindromic subsequence of <code>s</code> is the LCS of <code>s</code> and its reverse.</details>
2. In weighted job scheduling, why must jobs be sorted by end time rather than start time?
   <details><summary>Answer</summary>Sorting by end time makes the jobs compatible with job <code>i</code> (those ending by its start) a prefix of the sorted list, so <code>dp[lo]</code> over that prefix is well defined and binary search can find it.</details>
3. In the segmented sieve, why does crossing off start at `max(p * p, first multiple of p ≥ left)`?
   <details><summary>Answer</summary>The first multiple at or above <code>left</code> is where the segment begins, and any smaller multiple of <code>p</code> below $p^2$ has a smaller prime factor that already crosses it off. Starting no lower than $p^2$ also keeps <code>p</code> itself from being marked composite when it lies in the range.</details>
4. When is `memory_order_relaxed` enough for an atomic sequence counter, and when is it not?
   <details><summary>Answer</summary>It is enough when you only need each caller to get a unique number. If other threads must also observe data written before the increment, the increment must carry release semantics and the reads acquire semantics.</details>
5. Why must a matching engine be deterministic?
   <details><summary>Answer</summary>Replaying the same input sequence must reproduce the same trades, which enables recovery after a crash, hot standbys that stay in sync, and auditing.</details>
6. A market data consumer falls behind during a burst. What does conflation do, and what does the consumer lose?
   <details><summary>Answer</summary>Conflation sends only the latest quote per instrument instead of every intermediate update. The consumer stays current but loses the intermediate price path, which is acceptable for some consumers and not for others, hence the per-consumer setting.</details>

## Further reading

- [Glassdoor: Citadel software developer interview questions](https://www.glassdoor.com/Interview/Citadel-Software-Developer-Interview-Questions-EI_IE14937.0,7_KO8,26.htm)
- [Top quant trading firms tier list](https://www.quantvps.com/blog/top-quant-trading-firms) (QuantVPS)
- [Quant firm tier list](https://www.thewallstreetquants.com/firm-list) (WallStreetQuants)
- [DE Shaw vs Two Sigma vs Citadel](https://www.wallstreetoasis.com/forum/hedge-fund/de-shaw-vs-two-sigma-vs-citadel-offer-advice) (Wall Street Oasis)
- [Quant funds revealed: careers, salaries, and recruiting](https://mergersandinquisitions.com/quant-funds/) (Mergers & Inquisitions)
- Community reports on Blind and r/cscareerquestions.
