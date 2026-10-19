# Near-duplicate dedup: MinHash, LSH, union-find, and decontamination

After [exact dedup](03-exact-dedup-bloom-and-sort-merge.md) the corpus has no exact copies left, but it is still full of almost-copies: the same story reposted with one word changed, a page scraped twice with a different footer, a template filled with other names. Exact hashes see two different documents. A model trained on them memorizes the repeated text and spends its capacity on it. Worse, some of those almost-copies are your evaluation stories: train on them and your validation loss in [TinyStories, owned end to end](../../build-an-llm/capstones/01-tinystories.md) measures recall, not learning.

Telling that two texts are 95% the same by comparing every pair is hopeless: at a million documents that is half a trillion comparisons. This section finds near duplicates in roughly linear time with MinHash signatures and locality-sensitive hashing (LSH), merges them into clusters with union-find, and then removes anything that overlaps your protected eval and validation sets. That last step, decontamination, is what makes benchmark numbers mean something.

You will need the `Doc` type from [the filter stages](02-extract-normalize-and-quality-filters.md), the `PCG32` generator and FNV-1a hash from [Modular arithmetic, hashing, and PCG32](../../mathematics-for-computing/06-discrete-math-2/03-modular-arithmetic-hashing-and-pcg32.md) in *Mathematics for Computing*, and, for practice by hand, the Jaccard and S-curve problems in [the discrete math problem set](../../mathematics-for-computing/06-discrete-math-2/91-problem-set-b.md). The solution is [`python/corpus/minhash.py`](../../../../course/solutions/python/corpus/minhash.py), its contract is [`minhash.pyi`](../../../../course/contracts/py/corpus/minhash.pyi), and the shard column it feeds is described in [`formats/corpus-shard.md`](../../../../course/contracts/formats/corpus-shard.md).

> **Learning objectives**
> - Compute word shingles, Jaccard similarity, and a MinHash estimate by hand, and explain why the estimate is unbiased.
> - Derive the LSH S-curve $p(s) = 1 - (1 - s^r)^b$ and its threshold $(1/b)^{1/r}$.
> - Cluster confirmed pairs with a union-find whose result does not depend on input order or worker count.
> - Decontaminate a corpus against protected eval sets with word 13-grams.

## Notation

The section uses the following symbols throughout:

| Symbol | Meaning | Type / shape |
|---|---|---|
| $k$ | shingle length in words (default 5) | `int` |
| $S(d)$ | the set of word $k$-shingles of document $d$ | `set[str]` |
| $J(A, B) = \lvert A \cap B \rvert / \lvert A \cup B \rvert$ | Jaccard similarity of two sets, in $[0, 1]$ | `float` |
| $P = 2^{31} - 1$ | a prime (a Mersenne prime) | `int` |
| $x(s) = \mathrm{fnv1a64}(s) \bmod P$ | the integer a shingle hashes to | `int` in $[0, P)$ |
| $h_i(x) = (a_i x + b_i) \bmod P$ | the $i$-th hash function, $1 \le a_i < P$, $0 \le b_i < P$ | `int` |
| $n$ | number of hash functions (`num_perm`, default 128) | `int` |
| $\mathrm{sig}_i(A) = \min_{s \in A} h_i(x(s))$ | entry $i$ of the MinHash signature | `uint64[n]` |
| $\hat J = \frac{1}{n}\sum_i [\mathrm{sig}_i(A) = \mathrm{sig}_i(B)]$ | the estimate | `float` |
| $b, r$ | bands and rows per band, $b \cdot r = n$ (16 and 8) | `int` |
| $s$ | the true similarity of a pair | `float` |
| $p(s) = 1 - (1 - s^r)^b$ | probability that a pair at similarity $s$ becomes a candidate | `float` |
| $t$ | the confirmation threshold on $\hat J$ (0.8) | `float` |

## Shingles and Jaccard similarity

The first step turns "how similar are these texts?" into "how much do these sets overlap?". Lower-case the text and split it into word runs (`re.findall(r"\w+", text.lower())`, so punctuation and case never matter). A $k$-shingle is $k$ consecutive words joined by one space; "the cat sat on the mat" has the 3-shingles "the cat sat", "cat sat on", "sat on the", "on the mat". A text shorter than $k$ words is one shingle (all its words), so a three-word line can still match its copy. A text with no words has no shingles and is nobody's near duplicate.

$J(A, B)$ is the size of the overlap divided by the size of the union: 1 for equal sets, 0 for disjoint ones. One changed word in a 150-word story changes at most $k$ shingles, so $J$ stays above 0.9, while two unrelated stories share almost none.

## MinHash

Jaccard needs both full sets. To compare documents cheaply you want a fixed-size summary that still estimates $J$. Pick a random hash function $h$ and look at the shingle of $A \cup B$ with the smallest hash. It is equally likely to be any element of the union, and the minimum of $A$ equals the minimum of $B$ exactly when that element is in both. So

$$\Pr[\min_{A} h = \min_{B} h] = \frac{\lvert A \cap B \rvert}{\lvert A \cup B \rvert} = J(A, B).$$

With $n$ independent hash functions, each equal entry is a coin that lands heads with probability $J$, and $\hat J$ is the fraction of heads: an unbiased estimate with standard deviation $\sqrt{J(1-J)/n}$, about 0.035 at $J = 0.8$ and $n = 128$. The share of equal entries in two signatures is therefore an unbiased estimate of the Jaccard similarity of their shingle sets, and the signature has $n$ numbers whatever the document's length.

### The hash family, made exact

A random permutation of all shingles is too expensive, so each $h_i$ is a Carter-Wegman universal hash $(a_i x + b_i) \bmod P$ over $x(s)$, the FNV-1a hash of the shingle's UTF-8 bytes reduced mod $P$. The parameters come from PCG32: `rng = PCG32(seed)`, then for $i = 0, 1, \dots$ draw $a_i = 1 + $ `rng.below(P - 1)` and $b_i = $ `rng.below(P)`, interleaved. $a_i$ must be nonzero, because $a = 0$ maps every shingle to $b$ and that entry always agrees.

Because $a_i, x < 2^{31}$, the product $a_i x + b_i < 2^{63}$ fits in `uint64`, so numpy computes every entry exactly, and every implementation (Python, the Go port used by [CorpusBuild](09-corpusbuild-as-a-durable-workflow.md), any other) produces the same signature for the same seed. FNV-1a runs byte by byte, but every shingle runs the same steps, so the loop can run over byte positions with all shingles at once: numpy's `uint64` multiply wraps modulo $2^{64}$, which is exactly FNV's arithmetic.

## LSH banding

Signatures are small, but comparing every pair of them is still quadratic. LSH avoids that. Cut each signature into $b$ bands of $r$ consecutive rows and hash each band, with its band index, into a bucket. Two documents become **candidates** when they agree on all $r$ rows of at least one band. A band agrees with probability $s^r$, so it disagrees with probability $1 - s^r$, all $b$ bands disagree with probability $(1 - s^r)^b$, and

$$p(s) = 1 - (1 - s^r)^b.$$

That is an S-curve: near 0 for small $s$, near 1 for large $s$, steepest near $(1/b)^{1/r}$. Pairs above that point become candidates, pairs below rarely do, and no pair is compared to all the others. With $b = 16, r = 8$ the steep part is around 0.7, below the 0.8 we want, so true near duplicates almost never slip through and the false candidates are filtered next.

## Confirming and clustering with union-find

A candidate pair is kept when $\hat J \ge t$. Near-duplication is not transitive ($x \sim y$ and $y \sim z$ do not imply $x \sim z$), but a cluster should hold the whole chain, so confirmed pairs are merged with a **union-find** (disjoint-set forest). `find(x)` follows parent pointers to the root, and `union(x, y)` points one root at the other. Choose the root as the **smallest id** (string order) and the result is the same whatever order the pairs arrive in. `near_dedup` tags every document with its root (`meta["minhash_cluster"]`) and, with `drop=True`, keeps only the roots.

### Process-parallel, worker-invariant

Signatures are independent per document, so `signatures(..., workers=N)` splits the texts into $N$ contiguous chunks and computes them on a `ProcessPoolExecutor`. Every chunk uses the same `(num_perm, seed)` and the chunks come back in order, so the array is identical for every $N$. The pool uses the `spawn` start method (the same on every OS).

> **Warning:** `spawn` re-imports the program's `__main__`. The CLI's entry needs `if __name__ == "__main__":`, or each spawned worker re-runs the CLI.

## Decontamination

The last idea is simpler and matters most for honest numbers. A protected text is any eval or validation example you will report a number on. Its word $n$-grams ($n = 13$, the GPT-3 rule) form a set, and a training document is contaminated when any of its word 13-grams is in that set. Thirteen words are long enough that a shared run is almost never a coincidence and short enough to catch a copied paragraph with edits around it.

In `.jsonl` protected files every string value counts except the keys `case_id`, `id`, `tags`, and `scorer_args`: they are labels, not text the model is tested on. A protected text shorter than 13 words contributes no 13-gram, so if your eval items are that short, lower $n$ for that suite.

## Working it by hand

**Shingles and Jaccard.** Let $A$ = "the cat sat on the mat today", $B$ = "the cat sat on the mat again", $k = 3$:

| | 3-shingles |
|---|---|
| $A$ | the cat sat · cat sat on · sat on the · on the mat · the mat today |
| $B$ | the cat sat · cat sat on · sat on the · on the mat · the mat again |

Four are shared out of six distinct in total, so $J = 4/6 = 0.667$.

**MinHash with three toy hash functions.** Give the six distinct shingles ids $1$ to $6$ in the order above, with "the mat today" $= 5$ and "the mat again" $= 6$, so $A = \{1,2,3,4,5\}$ and $B = \{1,2,3,4,6\}$. Use $P = 7$:

| $x$ | 1 | 2 | 3 | 4 | 5 | 6 | min over $A$ | min over $B$ | equal? |
|---|---|---|---|---|---|---|---|---|---|
| $h_1 = (2x + 1) \bmod 7$ | 3 | 5 | 0 | 2 | 4 | 6 | 0 | 0 | yes |
| $h_2 = (3x + 2) \bmod 7$ | 5 | 1 | 4 | 0 | 3 | 6 | 0 | 0 | yes |
| $h_3 = (5x + 3) \bmod 7$ | 1 | 6 | 4 | 2 | 0 | 5 | 0 | 1 | no |

So $\hat J = 2/3$. The third function's minimum over $A$ landed on shingle 5, which $B$ lacks, so the entries differ. Three functions give a noisy estimate; 128 give a standard deviation near 0.04. This toy is the idea behind the real formula with $P = 2^{31} - 1$ in the implementation below.

**The S-curve.** With $b = 16$, $r = 8$, the threshold is $(1/16)^{1/8} = 2^{-4/8} = 2^{-1/2} = 0.7071$. At $s = 0.8$: $0.8^8 = 0.16777$, $1 - 0.16777 = 0.83223$, $0.83223^{16} = 0.05295$, so $p = 0.9470$. At $s = 0.5$: $0.5^8 = 0.0039063$, $0.9960938^{16} = 0.93930$, so $p = 0.0607$. A pair at 0.8 is a candidate 95 times in 100; a pair at 0.5 about 6 times in 100, and then its estimate (near 0.5) fails the 0.8 check.

**Union-find on a chain.** The confirmed pairs arrive as $(y, z)$ then $(x, y)$. `union(y, z)` finds roots $y$ and $z$; the smaller is $y$, so `parent[z] = y`. `union(x, y)` finds roots $x$ and $y$; the smaller is $x$, so `parent[y] = x`. Now `find(z)` follows $z \to y \to x$: one cluster $\{x, y, z\}$ with root $x$, and `near_dedup` keeps only $x$.

**Decontamination.** A protected story contains "Tom found a shiny red shell by the river and carried it home to show his sister." A training document says "... and then TOM FOUND A SHINY, RED shell by the river and carried it home to show ..." Its words lower-cased are the same run, so the 13-gram "tom found a shiny red shell by the river and carried it home" is shared and the document is dropped. Had it copied only 12 of those words, it would stay.

## Designing the interface

```python
# python/corpus/minhash.py (the full contract is contracts/py/corpus/minhash.pyi)
MERSENNE31: int                                            # 2**31 - 1
def words(text: str) -> list[str]
def shingles(text: str, k: int = 5) -> set[str]
def jaccard(a: AbstractSet[str], b: AbstractSet[str]) -> float
def shingle_hashes(shingles: Iterable[str]) -> NDArray     # uint64 [n]
def perm_params(num_perm: int, seed: int) -> tuple[NDArray, NDArray]
def minhash(shingles: Iterable[str], num_perm: int = 128, seed: int = 0) -> NDArray
def estimate(sig_a: NDArray, sig_b: NDArray) -> float
class LSH:            # __init__(bands=16, rows=8), threshold, probability(s), insert(key, sig), candidates()
class UnionFind:      # find(x), union(x, y), groups()
def signatures(texts, *, num_perm=128, seed=0, k=5, workers=1) -> NDArray    # [len(texts), num_perm]
def clusters(docs: Mapping[str, str], *, num_perm=128, bands=16, threshold=0.8, k=5, seed=0, workers=1) -> list[set[str]]
def near_dedup(docs: Iterable[Doc], *, ..., drop: bool = True) -> Iterator[Doc]
def protected_ngrams(paths: Sequence[Path], n: int = 13) -> set[str]
def contaminated(text: str, grams: AbstractSet[str], n: int = 13) -> bool
def decontaminate(docs: Iterable[Doc], protected: Sequence[Path], n: int = 13) -> Iterator[Doc]
```

`near_dedup` must see every document before it can cluster, so it reads its whole input, unlike the streaming filter stages; `decontaminate` streams. Both are stages, composed after `exact_dedup`.

## Building it step by step

Eight requirements shape the code.

1. **Similarity of texts becomes similarity of sets.** Lower-cased word runs remove case and punctuation; $k$-shingles turn "one changed word" into "at most $k$ changed elements". Jaccard on those sets is the quantity to threshold.
2. **Jaccard needs both sets; a fixed-size summary is needed instead.** The min-hash argument says one random hash's minimum agrees with probability exactly $J$, so $n$ independent minima give an unbiased estimate with standard deviation $\sqrt{J(1-J)/n}$, independent of document length.
3. **"Random hash" must be reproducible across processes, runs, and languages.** That rules out Python's salted `hash()` and forces a specified family: FNV-1a over UTF-8 bytes reduced mod $P = 2^{31}-1$, then $(a_i x + b_i) \bmod P$ with parameters drawn from a seeded PCG32. Keeping $a_i, x, b_i < 2^{31}$ keeps $a_i x + b_i < 2^{63}$, so `uint64` arithmetic is exact and vectorizable.
4. **Comparing all signature pairs is still quadratic.** Banding with $b \cdot r = n$ hashes each band to a bucket; only bucket-mates are compared. The band index must be part of the bucket key, otherwise equal rows in different bands collide.
5. **LSH is probabilistic in both directions, so its pairs are candidates.** The S-curve's steep point sits at $0.707$, below $t = 0.8$, so true pairs are rarely missed; the estimate check $\hat J \ge t$ then rejects the false candidates.
6. **Near-duplication is not transitive, but the decision "keep one of these" must be.** Union-find makes a chain one cluster. Choosing the smallest id as the root (not union by size, not argument order) makes the clusters and the survivor independent of the order pairs arrive in.
7. **Parallelism must not change the answer.** Every chunk recomputes the same `(a, b)` from the same seed, chunks are contiguous, and `pool.map` returns them in order, so the stacked array is identical for every worker count.
8. **Decontamination is a set-membership test, streamed.** Build the protected 13-gram set once; then each training document is checked in one pass and kept or dropped, and nothing needs the whole corpus.

### Shingles

```python
_WORD = re.compile(r"\w+")

def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())        # case and punctuation never matter

def shingles(text: str, k: int = 5) -> set[str]:
    if k < 1:
        raise ValueError(f"shingle length k must be >= 1, got {k}")
    w = words(text)
    if not w:
        return set()                          # no words: nobody's near duplicate
    if len(w) < k:
        return {" ".join(w)}                  # a short text is one shingle, so it can match its copy
    return {" ".join(w[i : i + k]) for i in range(len(w) - k + 1)}
```

### FNV-1a for many shingles at once

The byte grid is padded with zeros, and the `live` mask makes sure padding never enters a shorter shingle's hash:

```python
def shingle_hashes(shingles: Iterable[str]) -> NDArray:
    data = [s.encode("utf-8") for s in shingles]
    if not data:
        return np.zeros(0, dtype=np.uint64)
    lengths = np.fromiter((len(b) for b in data), dtype=np.int64, count=len(data))
    width = int(lengths.max())
    grid = np.zeros((len(data), width), dtype=np.uint64)
    for i, b in enumerate(data):
        grid[i, : len(b)] = np.frombuffer(b, dtype=np.uint8)
    h = np.full(len(data), FNV_OFFSET, dtype=np.uint64)
    prime = np.uint64(FNV_PRIME)
    for j in range(width):                    # one step per byte position, all shingles together
        live = lengths > j                    # shingles that still have a byte j
        h[live] = (h[live] ^ grid[live, j]) * prime   # uint64 multiply wraps mod 2^64, as FNV does
    return h % np.uint64(MERSENNE31)          # x(s) < 2^31, so a*x + b fits in 63 bits
```

### Hash parameters and the signature

```python
def perm_params(num_perm: int, seed: int) -> tuple[NDArray, NDArray]:
    rng = PCG32(seed)
    a = np.empty(num_perm, dtype=np.uint64)
    b = np.empty(num_perm, dtype=np.uint64)
    for i in range(num_perm):
        a[i] = 1 + rng.below(MERSENNE31 - 1)  # a in [1, P): a = 0 would make the entry constant
        b[i] = rng.below(MERSENNE31)          # interleaved draws: a0, b0, a1, b1, ...
    return a, b

def _signature(x: NDArray, a: NDArray, b: NDArray) -> NDArray:
    if x.size == 0:
        return np.full(a.shape[0], MERSENNE31, dtype=np.uint64)   # sentinel: above every real value
    h = (a[:, None] * x[None, :] + b[:, None]) % np.uint64(MERSENNE31)   # [num_perm, shingles]
    return h.min(axis=1)

def estimate(sig_a: NDArray, sig_b: NDArray) -> float:
    # ... shape check ...
    return float(np.mean(sig_a == sig_b))     # position by position, not as sets
```

### LSH banding

The key `(band, bytes)` keeps bands apart, and an all-sentinel signature is recorded but never bucketed:

```python
    def insert(self, key: str, sig: NDArray) -> None:
        sig = np.asarray(sig, dtype=np.uint64)
        if sig.shape != (self.bands * self.rows,):
            raise ValueError(...)
        if key in self._keys:
            raise ValueError(f"key {key!r} was inserted before")
        self._keys.add(key)
        if bool(np.all(sig == MERSENNE31)):
            return  # an empty text has no shingles: nobody's near duplicate
        for band in range(self.bands):
            part = sig[band * self.rows : (band + 1) * self.rows]
            self._buckets.setdefault((band, part.tobytes()), []).append(key)

    def candidates(self) -> list[tuple[str, str]]:
        pairs: set[tuple[str, str]] = set()   # a pair sharing several bands is still one pair
        for keys in self._buckets.values():
            if len(keys) < 2:
                continue
            ks = sorted(keys)
            for i in range(len(ks)):
                for j in range(i + 1, len(ks)):
                    pairs.add((ks[i], ks[j]))  # smaller key first
        return sorted(pairs)                  # deterministic order for the unions
```

### Union-find with the smallest root

```python
    def find(self, x: str) -> str:
        parent = self._parent
        if x not in parent:
            parent[x] = x
            return x
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:              # path compression: point every node on the path at root
            parent[x], x = root, parent[x]
        return root

    def union(self, x: str, y: str) -> str:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return rx
        if ry < rx:
            rx, ry = ry, rx
        self._parent[ry] = rx                 # the smaller root wins, whatever the argument order
        return rx
```

Because every root is the minimum of its set, the final partition and its roots are the same for any sequence of unions over the same pairs.

### Signatures in parallel, then clusters

```python
def signatures(texts, *, num_perm=128, seed=0, k=5, workers=1) -> NDArray:
    # ...
    if workers == 1 or len(texts) < 2:
        return _signature_chunk((texts, num_perm, seed, k))
    size = -(-len(texts) // workers)          # ceiling division: contiguous chunks
    chunks = [(texts[i : i + size], num_perm, seed, k) for i in range(0, len(texts), size)]
    ctx = multiprocessing.get_context("spawn")   # same start method on every OS
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        parts = list(pool.map(_signature_chunk, chunks))   # map returns results in chunk order
    return np.vstack(parts)

def clusters(docs, *, num_perm=128, bands=16, threshold=0.8, k=5, seed=0, workers=1):
    if bands < 1 or num_perm % bands:
        raise ValueError(f"bands ({bands}) must divide num_perm ({num_perm})")
    keys = sorted(docs)                       # input order cannot leak into the result
    sigs = signatures([docs[x] for x in keys], num_perm=num_perm, seed=seed, k=k, workers=workers)
    lsh = LSH(bands, num_perm // bands)
    row = {}
    for i, key in enumerate(keys):
        lsh.insert(key, sigs[i])
        row[key] = i
    uf = UnionFind()
    for x, y in lsh.candidates():
        if estimate(sigs[row[x]], sigs[row[y]]) >= threshold:   # confirm before merging
            uf.union(x, y)
    return uf.groups()
```

`_signature_chunk` calls `perm_params(num_perm, seed)` itself, so every worker uses the same hash family rather than a per-chunk seed.

### The stages

`near_dedup` must see everything before clustering, so it holds the documents (`list(docs)`), then yields in input order:

```python
    for d in items:
        r = root.get(d.id, d.id)              # a singleton is its own cluster
        if drop and r != d.id:
            continue                          # keep only each cluster's smallest id
        yield dataclasses.replace(d, meta={**d.meta, "minhash_cluster": r})
```

Decontamination streams:

```python
def contaminated(text: str, grams: AbstractSet[str], n: int = 13) -> bool:
    w = words(text)                           # same word rule: case and punctuation ignored
    return any(" ".join(w[i : i + n]) in grams for i in range(len(w) - n + 1))   # + 1: the last window counts

def decontaminate(docs, protected, n=13):
    grams = protected_ngrams(protected, n)    # built once
    for d in docs:                            # one document at a time
        if not contaminated(d.text, grams, n):
            yield d
```

`protected_ngrams` reads `.jsonl` files line by line, collecting every string value recursively except under `case_id`, `id`, `tags`, and `scorer_args`, raises on a line that is not a JSON object, and treats any other file as one text. The full solution is [`minhash.py`](../../../../course/solutions/python/corpus/minhash.py).

### Why this design

Signatures cost $O(n \cdot \lvert S(d) \rvert)$ per document, so $O(n \cdot T)$ in total for $T$ words. Banding is $O(b)$ per document, and the confirm step is proportional to the number of candidate pairs, which LSH keeps near linear for a corpus without huge clusters. Memory is $8n$ bytes per signature (1 KB at $n = 128$) plus the buckets. All-pairs Jaccard would be $O(D^2)$ comparisons.

Exact Jaccard on candidates would be slightly more accurate than $\hat J$, but it needs the shingle sets kept around for every document; the signature estimate is within about 0.035 at $J = 0.8$ and costs nothing extra. A greedy "drop it if it resembles a kept document" rule is order-dependent and breaks chains: whether $z$ survives depends on whether $y$ was seen first. Union-find with minimum roots is order-free. Union by size or rank gives better worst-case bounds but makes the survivor depend on union order; with path compression alone the forest stays shallow enough at this scale, and determinism matters more. For contamination, substring search misses copies with changed case or punctuation and is slower, while a set of normalized 13-grams catches both in one hash lookup per window.

Two tradeoffs are accepted. `near_dedup` is a barrier that holds the documents in memory (the stream after exact dedup is smaller), and a near duplicate at true similarity just above 0.8 is missed about 5% of the time.

## What a correct solution must handle

The cases cover the definitions, the statistics, and determinism:

| Case | What must hold | Why it matters |
|---|---|---|
| The hand example | the shingles above, $J = 4/6$, threshold 0.7071, $p(0.8) = 0.9470$, $p(0.5) = 0.0607$ | the definitions are exactly the ones above |
| The signature formula | signatures recomputed from FNV-1a and interleaved PCG32 draws | every run, resumed or ported, clusters alike |
| Words and short texts | case and punctuation ignored; fewer than $k$ words is one shingle | short documents still match their copies |
| Empty texts | wordless texts: all-$P$ signature, $J = 0$, no cluster | empty pages do not collapse into one survivor |
| An unbiased estimator | mean of $\hat J$ over 40 fresh pairs at $J = 1/3$ within $\lvert z \rvert < 3.29$ | the estimate means what it says |
| The S-curve | candidate rates at 13 similarities fit $p(s)$ (chi-square), 50% point within 0.05 of $(1/b)^{1/r}$ | the threshold you configure is the one you get |
| Bands kept apart | equal rows in different bands are not a candidate | false candidates cost time and, unconfirmed, data |
| Candidate pairs | one pair per key pair, smaller first, sorted; bad inserts raise | deterministic union order |
| Union-find roots | every permutation of the unions gives the same groups and roots | the survivor is stable |
| Planted clusters | planted clusters found exactly for 3 seeds; borderline pairs stay apart | the confirm step does its job |
| A chain | $x \sim y \sim z$, $x \not\sim z$: one cluster, keep $x$ | clusters are transitive |
| Worker count | 1 and 3 workers give identical signatures and clusters | the corpus hash must not depend on worker count |
| Input order | forward and reversed input keep the same documents | stages upstream may reorder |
| Tags, order, meta | `minhash_cluster` tags, `drop` flag, order and meta kept, repeated ids raise | the shard writer builds its column from the tag |
| Contamination | 13-word spans anywhere (upper-cased, punctuated) contaminate; 12 words, ids, and tags do not | benchmark numbers stay honest |
| A span at the very end | the last window of a document is checked | an off-by-one hides the commonest copy |
| Protected files | JSON Lines string values minus id-like keys; plain files whole; bad lines raise | the right text is protected |
| Streaming decontamination | reads 5 of an endless stream without running ahead | the stage runs on any corpus size |
| After exact dedup | near dedup drops exactly the planted near duplicates | the two dedup counts add up in the manifest |

The reference suite [`test_minhash_props.py`](../../../../course/solutions/learner-tests/data.04/python/tests/data-04-minhash/test_minhash_props.py) states the same laws as properties: the hand example; the signature formula; for any 1 to $k-1$ words, one shingle; wordless texts never cluster; the estimate is unbiased over fresh pairs; equal values in different bands, or in only $r - 1$ rows of a band, are no candidate; union-find roots are the minimum of each group for any union sequence (Hypothesis); a chain is one cluster; candidates below the threshold are not merged; workers do not change signatures; tags, order, and meta survive `near_dedup`; a protected 13-gram anywhere (any case, any punctuation) contaminates and 12 words do not; id-like keys are not protected; `decontaminate` streams. Each one pins down one of the pitfalls below.

## What can go wrong

Short and empty texts trip up naive code in opposite directions. Returning no shingles for a text shorter than $k$ means short documents never match their copies, so `shingles` returns `{" ".join(w)}` when `len(w) < k`. Indexing empty signatures puts every wordless document into one bucket, and all but one are dropped; `LSH.insert` skips all-sentinel signatures.

The estimator and banding have their own traps. Comparing signature values as sets instead of position by position gives biased estimates and missed clusters; the solution uses `np.mean(sig_a == sig_b)`. A bucket key without the band index makes equal rows in different bands into false candidates; buckets are keyed by `(band, part.tobytes())`. Merging every LSH candidate without the estimate check merges borderline pairs (0.5 to 0.6) a quarter of the time; `estimate(...) >= threshold` runs before `union`.

> **Warning:** Determinism breaks in several quiet ways. Python's `hash()` is salted per process, so signatures change between runs and workers; use FNV-1a over UTF-8 bytes. A seed per worker chunk, or chunks collected as they finish, changes the output hash with `--workers`; every chunk uses the same seed and `pool.map` keeps order. Union by size or by argument order makes the kept document depend on which pair came first; `union` always makes the smaller root the root. And "drop it if it resembles a kept document" instead of union-find keeps both ends of a chain.

Decontamination has four pitfalls of its own. Writing `range(len(w) - n)` instead of `range(len(w) - n + 1)` lets a document ending with the protected span pass. Substring matching on raw text lets punctuation and case changes hide copies; the solution uses normalized word 13-grams in a set. Treating `case_id`, `tags`, or `scorer_args` as protected text means labels, not eval text, start dropping documents; `_strings` skips `_SKIP_KEYS`. And `list(docs)` in `decontaminate` holds the whole corpus in memory, which is why it is a generator over `docs`.

Finally, with no `if __name__ == "__main__":` in the CLI entry, running with `--workers 4` makes each spawned worker re-run the CLI, and the pool dies with "bootstrapping phase". The corpus CLI ([`entry/python/corpus/__main__.py`](../../../../course/solutions/entry/python/corpus/__main__.py)) ends with the `__main__` guard.

## Where this leads

These two stages share the `Doc` and `Stage` shape from [the filter stages](02-extract-normalize-and-quality-filters.md), and run after [exact dedup](03-exact-dedup-bloom-and-sort-merge.md), so near dedup counts only near copies. Next, [the PII scrub](05-pii-scrub.md) runs on the survivors and keeps `minhash_cluster`; [the shard writer](06-parquet-shards-and-manifest.md) turns it into the smallest row index of each cluster and puts `near_dropped` and `decontaminated` into the manifest; and [CorpusBuild](09-corpusbuild-as-a-durable-workflow.md) runs this stage as the `dedup_near` activity, with an output hash that must not depend on the worker count.

### Further reading

- Broder, *On the Resemblance and Containment of Documents* (1997): the origin of MinHash.
- Leskovec, Rajaraman, and Ullman, [*Mining of Massive Datasets*](http://www.mmds.org/), chapter 3.
- Lee et al., [*Deduplicating Training Data Makes Language Models Better*](https://arxiv.org/abs/2107.06499).
- Brown et al., [*Language Models are Few-Shot Learners*](https://arxiv.org/abs/2005.14165), appendix C: 13-gram contamination; Touvron et al. (2023), appendix A.6: Llama 2's token-level contamination scores.
- datatrove `MinhashDedupSignature`, `MinhashDedupBuckets`, and `MinhashDedupCluster` (`src/datatrove/pipeline/dedup/minhash.py`): signatures written to disk per shard, buckets sorted and merged across machines (14 bands of 8 in FineWeb), and union-find over billions of pairs streamed from bucket files.
- Dolma's dedup (`allenai/dolma`) and text-dedup (`ChenghaoMou/text-dedup`): suffix arrays for exact substring dedup, Bloom filters for paragraph dedup.

## Key terms

- **Shingle**: $k$ consecutive lower-cased words joined by one space.
- **Jaccard similarity**: the size of two sets' intersection divided by the size of their union.
- **MinHash signature**: the minima of $n$ hash functions over a document's shingles; equal entries estimate Jaccard.
- **LSH banding**: splitting signatures into $b$ bands of $r$ rows so only documents sharing a whole band are compared.
- **Union-find**: a disjoint-set forest that merges confirmed pairs into clusters; here rooted at the smallest id.
- **Decontamination**: dropping training documents that share a word 13-gram with protected eval or validation text.

## Summary

- The share of equal entries in two MinHash signatures is an unbiased estimate of the Jaccard similarity of their shingle sets.
- Banding the signature into $b$ bands of $r$ rows turns that estimate into an S-curve: pairs above about $(1/b)^{1/r}$ become candidates, pairs below rarely do, and no pair is compared to all the others.
- Candidates are confirmed by their estimate and merged by union-find, whose root is the smallest id, so the clusters and the survivor never depend on input order or worker count.
- Decontamination drops every document that shares a word 13-gram with a protected eval or validation text; that is what makes benchmark numbers mean something.

## Review questions

1. Why must the LSH bucket key include the band index?
   <details><summary>Answer</summary>Without it, band 3 of one document and band 7 of another collide whenever their rows happen to be equal, which is not evidence of similarity: the rows come from different hash functions. Those false candidates cost confirm time and, if the estimate check were skipped, real data.</details>
2. Why is $a_i$ drawn as `1 + below(P - 1)` rather than `below(P)`?
   <details><summary>Answer</summary>With $a_i = 0$, $h_i(x) = b_i$ for every shingle, so that signature entry is the same for every non-empty document and always "agrees", biasing $\hat J$ upward. The draw keeps $a_i \in [1, P)$.</details>
3. Why does `UnionFind.union` attach the larger root under the smaller one instead of using union by size?
   <details><summary>Answer</summary>The cluster's root is the document `near_dedup` keeps. If the root depended on sizes or argument order, a reordered input or different candidate order could keep a different document. With the smallest id always the root, the result is a function of the set of confirmed pairs only.</details>
4. Why does `signatures` produce the same array for 1 and 3 workers?
   <details><summary>Answer</summary>Each chunk regenerates the same `(a, b)` from `(num_perm, seed)`, the chunks are contiguous slices in order, and `pool.map` returns results in submission order, so `np.vstack` reassembles the same rows. A seed per chunk, or collecting results as they complete, would break this.</details>
5. What does `range(len(w) - n + 1)` in `contaminated` protect against?
   <details><summary>Answer</summary>An off-by-one. With `range(len(w) - n)` the last window is never checked, so a document whose final 13 words are a copied eval span (the commonest shape of a pasted example) would pass.</details>
