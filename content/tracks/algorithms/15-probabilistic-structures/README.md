# Probabilistic Data Structures Patterns

## Core Insight

Probabilistic structures trade **exactness for massive space savings**. They give approximate answers with bounded error probabilities — perfect for large-scale systems where exact answers are too expensive.

## Pattern 1: Bloom Filter

**What**: Set membership test. "Definitely not in set" or "probably in set."

**How**: k hash functions, m-bit array. Insert: set bits at h1(x), h2(x), ..., hk(x). Query: check all k bits.

```python
class BloomFilter:
    def __init__(self, size, num_hashes):
        self.bits = [False] * size
        self.size = size
        self.num_hashes = num_hashes

    def _hashes(self, item):
        # Double hashing: h(i) = h1 + i*h2
        h1 = hash(item) % self.size
        h2 = hash(repr(item)) % self.size
        return [(h1 + i * h2) % self.size for i in range(self.num_hashes)]

    def add(self, item):
        for h in self._hashes(item):
            self.bits[h] = True

    def might_contain(self, item):
        return all(self.bits[h] for h in self._hashes(item))
```

**False positive rate**: `(1 - e^(-kn/m))^k` where n = items inserted.

**Optimal k**: `k = (m/n) * ln(2)`

**Use cases**: Spell checkers, cache lookups, database query optimization, network routers.

## Pattern 2: Count-Min Sketch

**What**: Frequency estimation. Answers "how many times has X appeared?" with bounded overcount.

**How**: d hash functions, d × w counter matrix. Increment all d counters on insert. Query returns minimum across d counters.

**Error bound**: Overestimates by at most ε*N with probability 1-δ, where w = ⌈e/ε⌉, d = ⌈ln(1/δ)⌉.

**Use cases**: Network traffic monitoring, trending topics, heavy hitters detection.

## Pattern 3: HyperLogLog

**What**: Cardinality estimation (count distinct). Estimates unique elements using O(log log n) space.

**How**: Hash each element, count leading zeros. The maximum number of leading zeros estimates log2 of cardinality.

**Practical accuracy**: ~2% error with 1.5KB of memory for billions of unique elements.

**Use cases**: Unique visitor counting, database APPROX_COUNT_DISTINCT, Redis PFCOUNT.

## Pattern 4: Skip List

**What**: Probabilistic alternative to balanced BSTs. Expected O(log n) search/insert/delete.

**How**: Multi-level linked list. Each element is promoted to the next level with probability 1/2. Search starts at top level, drops down when the next element is too large.

**Advantages over BSTs**: Simpler to implement, naturally lock-free, used in Redis sorted sets and LevelDB.

## Company Targeting

| Company | Focus | Difficulty |
|---------|-------|------------|
| Google | Bloom filters in BigTable, HyperLogLog | Medium-Hard |
| Netflix | Streaming cardinality estimation | Medium |
| Databricks | Approximate query processing | Medium-Hard |
| Amazon | DynamoDB bloom filters | Medium |
| Stripe | Rate limiting with count-min sketch | Medium |

## Space Comparison

| Structure | Space | False Positives | Deletions | Use |
|-----------|-------|----------------|-----------|-----|
| Hash Set | O(n) | None | Yes | Exact membership |
| Bloom Filter | O(1) per element | Yes (~1%) | No* | Approximate membership |
| Counting Bloom | O(1) per element | Yes | Yes | Membership + delete |
| Count-Min Sketch | O(1/ε * log(1/δ)) | Overcounts | With negative counts | Frequency estimation |
| HyperLogLog | O(log log n) | ±2% | No | Cardinality |

*Cuckoo filters support deletion and have better space efficiency for low false-positive rates.
