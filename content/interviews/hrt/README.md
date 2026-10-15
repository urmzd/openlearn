# Hudson River Trading (HRT) Software Engineer Interview Guide

Comprehensive preparation for HRT SWE roles. HRT is one of the most technically demanding quantitative trading firms, with a tiny engineering team and an obsession with nanosecond-level performance.

## Interview Process Overview

Timeline: **3-6 weeks**, **4-5 rounds**

| Round | Format | Duration | Focus |
|-------|--------|----------|-------|
| Online Assessment | HackerRank | 90 min | Hard algorithmic problems |
| Phone Screen | Technical | 60 min | Systems + algorithms |
| Onsite 1 | Coding | 90 min | Hard algorithms, C++ depth |
| Onsite 2 | Systems / Low-Latency | 90 min | Performance, architecture |
| Onsite 3 | Behavioral + Technical | 45 min | Culture fit, problem-solving |

HRT has **very few SWE openings** relative to applicants, making it one of the most selective employers in the world.

## Compensation

- **Junior SWE**: ~$400-600K total comp (Year 1)
- **Senior SWE**: ~$700K-2M+
- Heavy bonus component, tied to firm performance
- HRT is consistently among the highest-paying employers globally

## Key Themes

1. **Nanoseconds matter** -- HRT competes on speed. Their systems are measured in nanoseconds, not milliseconds.
2. **C++ is mandatory** -- Almost all production code is C++. Deep C++ knowledge is non-negotiable.
3. **Hardware awareness** -- You need to think about CPU caches, memory buses, network cards, and FPGAs.
4. **Tiny team, huge impact** -- Small engineering team means every hire has outsized impact.
5. **FPGA and kernel bypass** -- HRT uses FPGAs for ultra-low-latency market data processing.

## What Makes HRT Different From Other Quant Firms

| Aspect | HRT | Jane Street | Two Sigma | Citadel |
|--------|-----|-------------|-----------|---------|
| Primary language | C++ | OCaml | Python/Java | C++/Python |
| Latency target | Nanoseconds | Microseconds | Milliseconds | Microseconds |
| Hardware focus | FPGA, custom NICs | Standard | Standard | Custom hardware |
| Team size | Very small | Medium | Large | Large |
| Trading style | Market making, HFT | Market making | Systematic | Multi-strategy |

## Coding Rounds

### What to Expect

- **C++ required** for most positions
- Problems are algorithmically hard AND require efficient implementation
- Memory management, bit manipulation, cache-friendly design matter
- May include systems-level coding (network protocols, binary parsing)

### C++ Deep Knowledge Expected

```cpp
// 1. Bit manipulation for performance
inline int count_set_bits(uint64_t n) {
    return __builtin_popcountll(n);  // Hardware instruction
}

inline int find_lowest_set_bit(uint64_t n) {
    return __builtin_ctzll(n);  // Count trailing zeros
}

// 2. Cache-friendly iteration
// BAD: Column-major access on row-major array
for (int col = 0; col < N; col++)
    for (int row = 0; row < N; row++)
        sum += matrix[row][col];  // Cache miss every access

// GOOD: Row-major access
for (int row = 0; row < N; row++)
    for (int col = 0; col < N; col++)
        sum += matrix[row][col];  // Sequential, cache-friendly

// 3. Avoiding allocations in hot path
class MessageParser {
    // Pre-allocate buffers
    char buffer_[65536] __attribute__((aligned(64)));
    size_t pos_ = 0;

public:
    // Parse without allocation
    std::string_view next_field(char delimiter) {
        size_t start = pos_;
        while (pos_ < sizeof(buffer_) && buffer_[pos_] != delimiter)
            pos_++;
        return std::string_view(buffer_ + start, pos_ - start);
    }
};

// 4. Lock-free single-producer single-consumer queue
template<typename T, size_t N>
class SPSCQueue {
    static_assert((N & (N - 1)) == 0, "N must be power of 2");

    alignas(64) std::atomic<size_t> head_{0};
    alignas(64) std::atomic<size_t> tail_{0};
    T buffer_[N];

public:
    bool push(const T& item) {
        size_t tail = tail_.load(std::memory_order_relaxed);
        size_t next = (tail + 1) & (N - 1);
        if (next == head_.load(std::memory_order_acquire))
            return false;  // Full
        buffer_[tail] = item;
        tail_.store(next, std::memory_order_release);
        return true;
    }

    bool pop(T& item) {
        size_t head = head_.load(std::memory_order_relaxed);
        if (head == tail_.load(std::memory_order_acquire))
            return false;  // Empty
        item = buffer_[head];
        head_.store((head + 1) & (N - 1), std::memory_order_release);
        return true;
    }
};

// 5. Compile-time computation
template<int N>
struct Fibonacci {
    static constexpr int value = Fibonacci<N-1>::value + Fibonacci<N-2>::value;
};
template<> struct Fibonacci<0> { static constexpr int value = 0; };
template<> struct Fibonacci<1> { static constexpr int value = 1; };
// Fibonacci<10>::value computed at compile time
```

### Algorithmic Problems

```cpp
// Efficient string matching (may need to process billions of messages)
class AhoCorasick {
    struct Node {
        int children[26] = {};
        int fail = 0;
        int output = -1;
    };

    std::vector<Node> trie;

public:
    AhoCorasick() : trie(1) {}

    void add_pattern(const std::string& pattern, int id) {
        int node = 0;
        for (char c : pattern) {
            int ch = c - 'a';
            if (!trie[node].children[ch]) {
                trie[node].children[ch] = trie.size();
                trie.emplace_back();
            }
            node = trie[node].children[ch];
        }
        trie[node].output = id;
    }

    void build() {
        std::queue<int> q;
        for (int c = 0; c < 26; c++) {
            if (trie[0].children[c]) {
                q.push(trie[0].children[c]);
            }
        }
        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (int c = 0; c < 26; c++) {
                int v = trie[u].children[c];
                if (v) {
                    trie[v].fail = trie[trie[u].fail].children[c];
                    if (trie[trie[v].fail].output >= 0)
                        trie[v].output = trie[trie[v].fail].output;
                    q.push(v);
                } else {
                    trie[u].children[c] = trie[trie[u].fail].children[c];
                }
            }
        }
    }

    std::vector<std::pair<int, int>> search(const std::string& text) {
        std::vector<std::pair<int, int>> matches;  // (position, pattern_id)
        int node = 0;
        for (int i = 0; i < text.size(); i++) {
            node = trie[node].children[text[i] - 'a'];
            if (trie[node].output >= 0)
                matches.emplace_back(i, trie[node].output);
        }
        return matches;
    }
};
```

## Systems / Low-Latency Round

### What to Expect

This round tests your understanding of building systems where every nanosecond counts.

### Topics

#### Network Stack Optimization

```
Standard path:  NIC -> Kernel -> Socket -> User space  (~10-50 μs)
Kernel bypass:  NIC -> User space (DPDK/RDMA)          (~1-5 μs)
FPGA:           NIC -> FPGA logic                       (~100-500 ns)
```

#### Memory Hierarchy Optimization

```
Register:    ~0.3 ns    (fastest, limited)
L1 Cache:    ~1 ns      (32-64 KB per core)
L2 Cache:    ~4 ns      (256 KB - 1 MB per core)
L3 Cache:    ~10-40 ns  (shared, 10-100 MB)
DRAM:        ~100 ns    (GBs, main memory)
```

**Key principle**: Data structure design should maximize L1/L2 cache hits.

- Prefer arrays over linked lists (contiguous memory)
- Keep hot data small and together
- Avoid pointer chasing
- Use struct-of-arrays (SoA) for column-oriented access

#### FPGA Concepts

- FPGAs process market data in hardware, before it reaches the CPU
- Sub-microsecond parsing and decision-making
- Deterministic latency (no jitter from OS scheduling)
- Used for market data feed parsing, order entry, simple strategies

### Design Questions

- "Design a network packet parser that processes 10M packets/second"
- "Design a time-series database optimized for sequential writes and random reads"
- "Design a system that detects arbitrage opportunities across exchanges in < 1 microsecond"
- "How would you minimize jitter in a trading system?"

#### Jitter Minimization Checklist

```
1. CPU isolation: isolcpus, nohz_full, rcu_nocbs
2. Disable hyperthreading on latency-critical cores
3. Lock pages in memory (mlockall)
4. Use huge pages (reduce TLB misses)
5. Disable CPU frequency scaling (set performance governor)
6. Pin threads to cores (pthread_setaffinity_np)
7. Pre-fault stack and heap memory
8. Avoid syscalls in hot path
9. Use busy-polling instead of interrupts
10. Disable NUMA balancing
```

## Preparation Tips

1. **C++ mastery** -- This is non-negotiable. Know C++17/20, templates, move semantics, atomics, memory model.
2. **Low-latency systems** -- Read "Trading and Exchanges" by Larry Harris. Understand market microstructure.
3. **Cache-friendly programming** -- Practice writing code that minimizes cache misses. Profile with `perf`.
4. **Lock-free data structures** -- SPSC queues, lock-free stacks, hazard pointers.
5. **Competitive programming** -- HRT recruits from CP backgrounds. Strong Codeforces rating helps.
6. **Networking fundamentals** -- TCP vs UDP, multicast, kernel bypass (DPDK/io_uring).
7. **Bit manipulation** -- Fast, branchless operations. Know intrinsics (`__builtin_*`).
8. **Read HRT's blog** -- hudsonrivertrading.com/hrtbeat. They share engineering insights.

## Sources

- [HRT Engineering Blog (HRTbeat)](https://www.hudsonrivertrading.com/hrtbeat/)
- [HRT Careers](https://www.hudsonrivertrading.com/careers/)
- [Top Quant Trading Firms Tier List - QuantVPS](https://www.quantvps.com/blog/top-quant-trading-firms)
- [Quant Firm Tier List - WallStreetQuants](https://www.thewallstreetquants.com/firm-list)
- [Quant Funds Revealed - M&I](https://mergersandinquisitions.com/quant-funds/)
- Glassdoor, Blind, r/cscareerquestions (community reports)
