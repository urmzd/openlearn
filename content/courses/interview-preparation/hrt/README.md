# Hudson River Trading

Hudson River Trading (HRT) is one of the most technically demanding quantitative trading firms in the world. It runs a tiny engineering team with an obsession with nanosecond-level performance, and it has very few software engineering openings relative to the number of applicants, which makes it one of the most selective employers anywhere. If you interview there, expect hard algorithms written in efficient C++, and a systems round in which the unit of discussion is the cache line and the network card rather than the service.

This chapter prepares you for HRT SWE roles. It describes the process and how HRT differs from other quant firms, then works through the C++ fluency the coding rounds expect (bit manipulation, cache-friendly loops, allocation-free parsing, a lock-free queue, compile-time computation, and Aho-Corasick string matching), and finally the low-latency systems round: the network stack, the memory hierarchy, FPGAs, and jitter.

> **Learning objectives**
> - Describe the HRT interview loop and how its focus differs from Jane Street, Two Sigma, and Citadel.
> - Write and explain a lock-free single-producer single-consumer queue, including its memory orderings.
> - Explain why loop order and data layout change performance, using the memory hierarchy's latencies.
> - Implement Aho-Corasick multi-pattern matching and state its complexity.
> - List the operating-system and hardware techniques that reduce jitter in a trading system.

## Before you begin

You need solid C++ (C++17/20, templates, move semantics, atomics, and the memory model) and data structures at competitive-programming level. The [Jane Street](../jane-street/README.md) chapter covers market making and probability, which HRT shares as a domain, and [Distributed systems](../shared-concepts/distributed-systems.md) covers the networking background.

## The interview process

The process takes three to six weeks across four or five rounds:

| Round | Format | Duration | Focus |
|-------|--------|----------|-------|
| Online Assessment | HackerRank | 90 min | Hard algorithmic problems |
| Phone Screen | Technical | 60 min | Systems + algorithms |
| Onsite 1 | Coding | 90 min | Hard algorithms, C++ depth |
| Onsite 2 | Systems / Low-Latency | 90 min | Performance, architecture |
| Onsite 3 | Behavioral + Technical | 45 min | Culture fit, problem-solving |

HRT is consistently among the highest-paying employers globally. A junior SWE can expect about $400-600K in total compensation in the first year, and a senior SWE about $700K-2M or more. The bonus is a heavy component and is tied to firm performance.

## What HRT looks for

Everything at HRT follows from the fact that it competes on speed: its systems are measured in nanoseconds, not milliseconds. That is why C++ is mandatory. Almost all production code is C++, and deep C++ knowledge is non-negotiable. It is also why hardware awareness matters: you need to think about CPU caches, memory buses, network cards, and FPGAs, and HRT uses FPGAs and kernel bypass for ultra-low-latency market data processing. Finally, the team is tiny, so every hire has outsized impact and the bar is set accordingly.

It helps to see how this differs from the other quant firms covered in this book:

| Aspect | HRT | Jane Street | Two Sigma | Citadel |
|--------|-----|-------------|-----------|---------|
| Primary language | C++ | OCaml | Python/Java | C++/Python |
| Latency target | Nanoseconds | Microseconds | Milliseconds | Microseconds |
| Hardware focus | FPGA, custom NICs | Standard | Standard | Custom hardware |
| Team size | Very small | Medium | Large | Large |
| Trading style | Market making, HFT | Market making | Systematic | Multi-strategy |

## Coding rounds

C++ is required for most positions. The problems are algorithmically hard and also require an efficient implementation, so memory management, bit manipulation, and cache-friendly design matter as much as the asymptotic complexity. Some rounds include systems-level coding such as network protocols or binary parsing.

### The C++ you are expected to know

The listing below collects five idioms that interviewers probe. First, bit manipulation through compiler intrinsics, which compile to single hardware instructions. Second, cache-friendly iteration: C++ arrays are row-major, so iterating column by column jumps a whole row's width between accesses and misses the cache on nearly every one, while iterating row by row walks memory sequentially. Third, avoiding allocation in the hot path by parsing into a preallocated, cache-line-aligned buffer and returning `std::string_view`s into it rather than new strings. Fourth, a lock-free single-producer single-consumer (SPSC) queue. Fifth, compile-time computation with templates.

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

The SPSC queue is the piece you are most likely to be asked to explain line by line. Requiring `N` to be a power of two lets the index wrap with a mask, `& (N - 1)`, instead of a division. The head and tail indices are each aligned to 64 bytes so that they sit on separate cache lines; otherwise the producer and consumer would fight over one line (false sharing). Each side owns one index and only reads the other's. The producer reads its own `tail_` with relaxed ordering, since nobody else writes it, then checks fullness by acquiring `head_`. It writes the slot and only then publishes the new tail with a release store, so that a consumer which acquires that tail is guaranteed to see the written item. `pop` mirrors this exactly. One slot is always left empty, which is how the queue tells full from empty without a separate count, so its capacity is `N - 1`. Note also that `next_field` in the parser stops at the delimiter without consuming it; a real parser would advance `pos_` past it before the next call.

### Algorithmic problems

The algorithmic problems lean towards what trading systems actually do, such as processing billions of messages. Multi-pattern string matching is a good example. Aho-Corasick builds a trie of all the patterns, then adds failure links, computed breadth-first, that say where to continue matching when the next character does not extend the current prefix. The `build` step also fills in missing children with the failure transition, turning the trie into a full automaton, so `search` takes exactly one transition per character of text.

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

After `build`, searching costs $O(n)$ in the text length plus the number of matches reported, independent of how many patterns there are. Two simplifications are worth naming if asked: the alphabet is lowercase `a` to `z`, and each node keeps a single `output`, so when several patterns end at the same position (one a suffix of another) only one is reported. A full implementation follows an output-link chain to report them all.

## The systems and low-latency round

This round tests whether you understand how to build systems where every nanosecond counts. It starts with the network.

### The network stack

A packet that goes through the kernel's network stack is slow by HRT standards. Bypassing the kernel saves roughly an order of magnitude, and handling the packet in an FPGA on the NIC saves another:

```
Standard path:  NIC -> Kernel -> Socket -> User space  (~10-50 μs)
Kernel bypass:  NIC -> User space (DPDK/RDMA)          (~1-5 μs)
FPGA:           NIC -> FPGA logic                       (~100-500 ns)
```

### The memory hierarchy

Once data is in user space, the memory hierarchy dominates. These are the latencies to keep in your head:

```
Register:    ~0.3 ns    (fastest, limited)
L1 Cache:    ~1 ns      (32-64 KB per core)
L2 Cache:    ~4 ns      (256 KB - 1 MB per core)
L3 Cache:    ~10-40 ns  (shared, 10-100 MB)
DRAM:        ~100 ns    (GBs, main memory)
```

A DRAM access costs a hundred L1 hits, so the key principle is that data structure design should maximise L1 and L2 cache hits. In practice that means preferring arrays over linked lists for contiguous memory, keeping hot data small and together, avoiding pointer chasing, and using a struct-of-arrays (SoA) layout when access is column-oriented.

### FPGAs

FPGAs take this one step further by processing market data in hardware before it ever reaches the CPU. They give sub-microsecond parsing and decision-making, and deterministic latency, with no jitter from OS scheduling. HRT uses them for market data feed parsing, order entry, and simple strategies.

### Design questions

Typical prompts include "Design a network packet parser that processes 10M packets/second", "Design a time-series database optimized for sequential writes and random reads", "Design a system that detects arbitrage opportunities across exchanges in < 1 microsecond", and "How would you minimize jitter in a trading system?"

The jitter question has a well-known checklist of answers, each removing a source of unpredictable delay:

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

Be ready to say why each helps. Isolating cores and pinning threads keeps the scheduler and other work off the critical path. Locking and pre-faulting memory removes page faults, and huge pages reduce TLB misses. A fixed frequency governor removes clock ramping. Busy-polling and avoiding syscalls trade CPU for predictable latency, and disabling NUMA balancing stops the kernel from migrating pages under you.

## Preparing

C++ mastery comes first and is non-negotiable: C++17/20, templates, move semantics, atomics, and the memory model. For the domain, read *Trading and Exchanges* by Larry Harris to understand market microstructure. Practise cache-friendly programming and profile your code with `perf`. Implement lock-free data structures: SPSC queues, lock-free stacks, and hazard pointers. HRT recruits from competitive programming, and a strong Codeforces rating helps. Brush up on networking fundamentals (TCP versus UDP, multicast, and kernel bypass with DPDK or io_uring) and on fast, branchless bit manipulation with intrinsics such as `__builtin_*`. Finally, read HRT's engineering blog, HRTbeat, where the firm shares engineering insights.

## Summary

- HRT competes on nanoseconds, so C++ depth and hardware awareness are prerequisites, not extras.
- Coding rounds judge both algorithmic difficulty and implementation efficiency: allocation, layout, and bits.
- A correct SPSC queue rests on separate cache lines, acquire/release pairs, and one empty slot to tell full from empty.
- The memory hierarchy spans about 0.3 ns to 100 ns; design data structures for L1 and L2 hits.
- Kernel bypass, FPGAs, and the jitter checklist are the vocabulary of the systems round.

## Further reading

- [HRT Engineering Blog (HRTbeat)](https://www.hudsonrivertrading.com/hrtbeat/)
- [HRT Careers](https://www.hudsonrivertrading.com/careers/)
- [Top Quant Trading Firms Tier List (QuantVPS)](https://www.quantvps.com/blog/top-quant-trading-firms)
- [Quant Firm Tier List (WallStreetQuants)](https://www.thewallstreetquants.com/firm-list)
- [Quant Funds Revealed (M&I)](https://mergersandinquisitions.com/quant-funds/)
- Larry Harris, *Trading and Exchanges*.
- Community reports on Glassdoor, Blind, and r/cscareerquestions.
