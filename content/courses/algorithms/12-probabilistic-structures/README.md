# Probabilistic data structures

An exact set of a billion keys costs gigabytes; an exact count of distinct visitors needs to remember every visitor. Probabilistic data structures trade **exactness for massive space savings**: they give approximate answers with bounded error probabilities, which is exactly right for large-scale systems where exact answers are too expensive. A Bloom filter answers membership in about ten bits per element with a 1% false-positive rate; HyperLogLog counts billions of distinct items to within about 2% in 1.5 KB.

This chapter introduces four such structures: the Bloom filter for membership, the count-min sketch for frequencies, HyperLogLog for cardinality, and the skip list as a randomized alternative to balanced trees. For each it gives the mechanism, the error bound, and where it is used, and it ends with a space comparison against an exact hash set.

> **Learning objectives**
> - Explain why a Bloom filter has false positives but no false negatives, and compute its false-positive rate and optimal number of hash functions.
> - Describe how a count-min sketch bounds its overcount and how to size it for a given error and confidence.
> - Explain how HyperLogLog estimates cardinality from leading zeros.
> - Describe the skip list and why it achieves expected $O(\log n)$ operations.
> - Choose between exact and probabilistic structures by space, error, and deletion support.

## Before you begin

You need hash functions and hash sets from [Arrays and hashing](../01-arrays-hashing/README.md), linked lists from [Linked lists](../04-linked-lists/README.md) for the skip list, and basic probability; [Probability and statistics](../../mathematics-for-computing/08-probability-statistics/README.md) in *Mathematics for Computing* covers what is needed. A production Bloom filter is built in full in [the Bloom filter](../13-systems-data-structures/08-bloom-filter.md) section of the systems chapter.

## Bloom filters

A Bloom filter tests set membership with two possible answers: "definitely not in the set" or "probably in the set." It is an $m$-bit array with $k$ hash functions. To insert $x$, set the bits at $h_1(x), h_2(x), \dots, h_k(x)$; to query, check all $k$ bits. A bit is never cleared, so an inserted element always tests positive (no false negatives), but an absent element can find all its bits set by other insertions (a false positive).

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

Double hashing derives all $k$ positions from two hash values, which is as good as $k$ independent hashes in practice. After $n$ insertions, the false-positive rate is

$$p \approx \left(1 - e^{-kn/m}\right)^k,$$

and it is minimized by choosing $k = \frac{m}{n} \ln 2$. Bloom filters show up in spell checkers, cache lookups (skip the disk read when the key is definitely absent), database query optimization, and network routers.

## Count-min sketches

A count-min sketch estimates frequencies: "how many times has $x$ appeared?", with a bounded overcount. It is a $d \times w$ matrix of counters with $d$ hash functions, one per row. An insertion increments one counter in each row; a query returns the minimum of the $d$ counters for $x$. Collisions only ever add to a counter, so every row overestimates, and the minimum is the least-polluted estimate.

With $N$ total insertions, choosing $w = \lceil e/\varepsilon \rceil$ and $d = \lceil \ln(1/\delta) \rceil$ guarantees that the estimate exceeds the true count by at most $\varepsilon N$ with probability at least $1 - \delta$. Network traffic monitoring, trending topics, and heavy-hitter detection all use it.

## HyperLogLog

HyperLogLog estimates cardinality (the number of distinct elements) in $O(\log \log n)$ space. Hash each element and count the leading zeros of the hash. A hash with $r$ leading zeros turns up about once in $2^r$ distinct elements, so the maximum number of leading zeros seen estimates $\log_2$ of the cardinality. Real implementations split the hashes into many buckets and average their estimates harmonically to reduce the variance.

In practice it reaches about 2% error with 1.5 KB of memory for billions of unique elements. Unique-visitor counting, databases' `APPROX_COUNT_DISTINCT`, and Redis's `PFCOUNT` are all HyperLogLog.

## Skip lists

A skip list is a probabilistic alternative to a balanced binary search tree, with expected $O(\log n)$ search, insertion, and deletion. It is a multi-level linked list: every element is on the bottom level, and each element is promoted to the next level up with probability $1/2$, so each level has about half the elements of the one below. A search starts at the top level and moves right until the next element is too large, then drops down a level.

Compared with balanced trees, skip lists are simpler to implement, need no rotations, and lend themselves to lock-free concurrent versions. Redis sorted sets and LevelDB's memtable use them.

## Choosing a structure

The structures differ in space, the kind of error they make, and whether they support deletion. Compared with an exact hash set:

| Structure | Space | False positives | Deletions | Use |
|-----------|-------|----------------|-----------|-----|
| hash set | $O(n)$ | none | yes | exact membership |
| Bloom filter | $O(1)$ per element | yes (about 1%) | no | approximate membership |
| counting Bloom filter | $O(1)$ per element | yes | yes | membership with deletion |
| count-min sketch | $O(\frac{1}{\varepsilon} \log \frac{1}{\delta})$ | overcounts | with negative counts | frequency estimation |
| HyperLogLog | $O(\log \log n)$ | about ±2% | no | cardinality |

A plain Bloom filter cannot delete, because clearing a bit could remove other elements too; a counting Bloom filter replaces bits with small counters to allow it. Cuckoo filters also support deletion and are more space-efficient at low false-positive rates.

Interviewers and production systems use these structures in different ways. As a rough guide:

| Company | Focus | Difficulty |
|---------|-------|------------|
| Google | Bloom filters in Bigtable, HyperLogLog | Medium-Hard |
| Netflix | streaming cardinality estimation | Medium |
| Databricks | approximate query processing | Medium-Hard |
| Amazon | Bloom filters in DynamoDB | Medium |
| Stripe | rate limiting with count-min sketch | Medium |

## In this chapter

- [Bloom filter event validator](bloom.py): an early, small program that checks a CSV catalog of event ids. A Bloom filter sized for 4,980,000 items at a 1% false-positive rate flags ids that have probably been seen before, and a running comparison flags ids that are out of order; both lists are printed at the end. It uses the third-party `bloom_filter` package and reads `catalog_incorrect.csv` from the current directory. Because the filter can report false positives, an id it flags as a duplicate is only probably one; a second exact check would confirm it.
