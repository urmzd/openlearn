# Exact dedup: paragraph hashes, a Bloom screen, and a sort-merge confirm

After [the quality filters](02-extract-normalize-and-quality-filters.md) every document is clean, English, and plausibly prose, and many of them say the same thing. Web pages share navigation and license footers; dataset mirrors repeat whole files; story collections reuse an opening paragraph. A model trained on that sees the repeated text again and again. Lee et al. found that deduplicating C4 removed text that models otherwise reproduced verbatim, and that the deduplicated models trained faster to the same loss. Repeated rare strings are also the ones a model memorizes, which matters for privacy (see [Privacy and PII policy](../../responsible-ai/02-privacy-and-pii/01-privacy-and-pii-policy.md) in *Responsible AI*).

The obvious fix, a Python set of every paragraph seen, costs about 90 bytes per paragraph (a 32-byte digest as a `bytes` object is 65 bytes, plus its slot in the set), so a billion paragraphs need 90 GB. A compact Bloom screen does the screening in about 10 bytes each. This section builds exact dedup from that screen, and shows what the Bloom filter's false positives mean for correctness: with a sort-merge confirm behind it, nothing.

You will use the `Doc` type and `compose` from the previous section, the FNV-1a and SplitMix64 hashes from [Modular arithmetic, hashing, and PCG32](../../mathematics-for-computing/06-discrete-math-2/03-modular-arithmetic-hashing-and-pcg32.md) in *Mathematics for Computing*, and the false-positive rate and optimal $k$ worked in [the discrete math problem set on Bloom filters and LSH](../../mathematics-for-computing/06-discrete-math-2/91-problem-set-b.md). The data structure itself is covered in [Bloom filter](../../algorithms/16-systems-data-structures/08-bloom-filter.md) in *Algorithms*; the Python screen here follows the shared bit-sizing rules in [`formats/bloom.md`](../../../../course/contracts/formats/bloom.md), and the standalone Rust Bloom module from that chapter does not cross into this stage. The solution is [`python/corpus/dedup.py`](../../../../course/solutions/python/corpus/dedup.py) (`paragraphs`, `paragraph_hash`, `bloom_rate`, `exact_dedup`, `STATS_KEYS`), with its contract in [`dedup.pyi`](../../../../course/contracts/py/corpus/dedup.pyi).

> **Learning objectives**
> - Split normalized text into paragraphs and explain why paragraphs are the unit of exact dedup.
> - Size a Bloom filter from a bytes-per-item budget and compute its false-positive rate.
> - Explain why a Bloom hit may only nominate a candidate, and confirm candidates exactly with a sort-merge.
> - Implement dedup as a four-pass stage over a disk spool that cleans up after itself.

## Paragraphs and their hashes

Start with the unit of comparison. A **paragraph** is a piece of the text between blank lines, where a blank line is empty or holds only spaces and tabs. Each piece is stripped of white space at both ends, and empty pieces are skipped. A single line break stays inside a paragraph. After normalization, paragraphs are separated by exactly `\n\n`.

Comparing paragraphs by their text would mean storing the text. Instead compare their **SHA-256 digests**: `paragraph_hash(p) = sha256(p.encode("utf-8")).digest()`, 32 bytes. Equal paragraphs have equal digests; unequal ones collide with probability about $2^{-256}$, which for our purposes is never. A paragraph's **repeat** is any occurrence whose digest occurred earlier in the input, in an earlier document or earlier in the same one. Exact dedup keeps the *first* occurrence of every paragraph, in input order, and removes the later ones; a document that loses every paragraph is dropped.

Why paragraphs and not whole documents? Duplicates are rarely whole documents: a copied story with a new title, or a page with a shared footer, differs as a document and repeats as paragraphs. Why not sentences or substrings? Shorter units remove ordinary phrases ("Once upon a time, there was a little girl.") from unrelated stories, and finding repeated substrings of any length needs a suffix array (Lee et al.), a different tool. Paragraphs are the CCNet compromise.

## The Bloom filter as a screen

Digests are smaller than text but a set of them is still 90 bytes each. A Bloom filter is a bit array of $m$ bits and $k$ hash functions. Inserting an item sets its $k$ bits; asking about an item answers "present" when all $k$ bits are set. The notation:

| Symbol | Meaning | Type |
|---|---|---|
| $N$ | paragraphs in the input (the items the filter will hold) | integer |
| $m$ | bits in the filter | integer |
| $k$ | hash functions (bits per item) | integer |
| $B$ | `bloom_bytes_per_item`: filter bytes per paragraph, so $m = 8BN$ | float, $> 0$ |
| $p$ | the false-positive rate: the chance that a never-inserted item is reported present | float in $(0, 1)$ |

Two facts drive the design. First, there are **no false negatives**: an inserted item's bits stay set, so the filter never says "new" about something it has seen. Second, there are **false positives**: with the optimal $k = (m/N)\ln 2$, the rate is $p = e^{-(m/N)(\ln 2)^2}$. Solving for the memory per item gives $m/N = -\ln p / (\ln 2)^2$. `formats/bloom.md` sizes a filter that way, so for a budget of $B$ bytes per item ($m/N = 8B$ bits) you ask for

$$p = e^{-8B(\ln 2)^2}, \qquad m = \left\lceil \frac{-N \ln p}{(\ln 2)^2} \right\rceil \approx 8BN .$$

`bloom_rate(B)` computes that $p$. $B = 10$ gives $p \approx 2.0 \times 10^{-17}$; $B = 1$ gives $p \approx 0.021$; $B = 0.25$ gives $p \approx 0.38$.

The screen pass hashes every paragraph in input order and asks the filter. If it answers "present", the digest becomes a **candidate**; otherwise it is inserted. By the first fact, every digest that occurs twice or more becomes a candidate at its second occurrence. By the second, some digests that occur once become candidates too: the false positives. So the candidates are a **superset** of the repeated digests, and a small one: the repeated digests plus about $pN$ false positives. A Bloom filter answers "definitely new" or "maybe seen" in about 10 bytes per item, a tenth of what a set of hashes costs, but its "maybe" is only a candidate.

## Confirming by sort-merge

The candidates still have to be sorted into real repeats and false alarms. The confirm pass reads the input again and records an **occurrence** $(h, i, j)$ for every paragraph whose digest $h$ is a candidate, where $i$ is the document's index and $j$ the paragraph's index within it. Sorting the occurrences makes equal digests adjacent, and within one digest the order is input order (by $i$, then $j$). Then the pass **merges**, walking runs of equal $h$. A run of length 1 is a false positive: that paragraph is unique, so keep it. A run of length $r \ge 2$ is a confirmed repeat: keep its first occurrence and mark the other $r - 1$ for removal.

Sorting is what makes this scale: an external sort handles candidate lists larger than memory, which a hash map of counts does not. Here the candidates fit in a Python list. Because the confirm is exact, no unique paragraph is ever dropped, whatever the filter size.

## Four passes over a spool

There is one practical obstacle. The stage consumes an iterator, which can be read once, but the algorithm needs the input three more times. So it **spools**: the first pass writes every document as one JSON line to a temporary file and counts $N$, and the screen, confirm, and emit passes reread that file. Memory holds the filter, the candidate set, and the candidate occurrences, never the documents.

The cost is that nothing is emitted until the input is exhausted: dedup is a barrier in the otherwise streaming pipeline. The spool lives in a `tempfile.TemporaryDirectory` (under `spool_dir` when given), which is removed when the generator finishes *or is closed early*, because the spool is a full copy of the corpus.

The emit pass yields each document with its marked paragraphs removed: unchanged (the same `Doc`, exact text and meta) when it lost nothing, with the remaining paragraphs joined by `\n\n` when it lost some, and not at all when it lost every paragraph. Output order is input order. With `stats={}` passed in, the stage fills `docs_in`, `paragraphs`, `bloom_bits` ($m$), `candidates`, `false_positives`, `duplicate_paragraphs`, `docs_dropped`, and `docs_out` once the output is exhausted; the corpus manifest's `dedup.exact_dropped` and the datasheet report them.

## Working it by hand

Take three documents and $B = 10$:

| Doc | Text |
|---|---|
| A (`s:0`) | `The cat sat.` (P1), blank line, `The dog ran.` (P2) |
| B (`s:1`) | `The dog ran.` (P2), blank line, `A bird sang.` (P3) |
| C (`s:2`) | `The cat sat.` (P1) |

**Spool.** There are $N = 5$ paragraphs. The filter has $m = \lceil 5 \cdot 80.0 \rceil = 400$ bits and $k = \text{round}(400/5 \cdot \ln 2) = 55$ hashes. The digests begin `h(P1) = 84549cfa...`, `h(P2) = 0ea960d6...`, `h(P3) = c2c47b92...`.

**Screen.** In input order, the filter sees:

| Occurrence $(i, j)$ | Paragraph | Filter says | Action |
|---|---|---|---|
| (0, 0) | P1 | new | insert |
| (0, 1) | P2 | new | insert |
| (1, 0) | P2 | present | candidate `h(P2)` |
| (1, 1) | P3 | new | insert |
| (2, 0) | P1 | present | candidate `h(P1)` |

**Confirm.** The occurrences of the two candidates are $(h(P1), 0, 0)$, $(h(P2), 0, 1)$, $(h(P2), 1, 0)$, $(h(P1), 2, 0)$. Sorted (`0e...` before `84...`) they are $(h(P2), 0, 1)$, $(h(P2), 1, 0)$, $(h(P1), 0, 0)$, $(h(P1), 2, 0)$. There are two runs of length 2, so remove $(1, 0)$ and $(2, 0)$. No run has length 1, so there are no false positives.

**Emit.** A loses nothing and comes out unchanged. B loses P2 and comes out as `A bird sang.`. C loses its only paragraph and is dropped. The stats: 3 documents in, 5 paragraphs, 400 bits, 2 candidates, 0 false positives, 2 duplicate paragraphs, 1 document dropped, 2 out.

With $B = 0.25$ the filter would have $\lceil 5 \cdot 2 \rceil = 10$ bits and might report P3 as present on first sight. P3 would become a candidate, its run in the confirm pass would have length 1, and it would be kept: the output is identical, and only `false_positives` grows.

## Designing the interface

```python
STATS_KEYS: tuple[str, ...]
def paragraphs(text: str) -> list[str]: ...
def paragraph_hash(p: str) -> bytes: ...
def bloom_rate(bloom_bytes_per_item: float) -> float: ...
def exact_dedup(docs: Iterable[Doc], bloom_bytes_per_item: float = 10, *,
                spool_dir: str | Path | None = None,
                stats: dict[str, Any] | None = None) -> Iterator[Doc]: ...
```

In a pipeline, bind the options: `compose(build_filters(cfg["filters"]), functools.partial(exact_dedup, bloom_bytes_per_item=cfg["dedup"]["bloom_bytes_per_item"]))`. The private Python filter is sized for `max(N, 1)` at `bloom_rate(B)`; use its `insert` and `contains`, and its `m` is the bit count reported in stats.

## Building it step by step

The reasoning chain runs from the definition of "first" to the spool.

1. **"Keep the first occurrence" is a property of input order**, so the algorithm must see occurrences in order and identify each by position $(i, j)$: document index, paragraph index. Keeping the first makes the result stable when sources are appended later.
2. **Comparing text means storing text; comparing digests does not.** SHA-256 turns each paragraph into 32 bytes with collisions at $2^{-256}$, so "equal digest" can stand for "equal paragraph".
3. **A set of digests is exact but costs about 90 bytes each in Python.** A Bloom filter costs $B$ bytes each and never forgets an insert (no false negatives), so every true repeat is caught at its second occurrence. Its only error is the other direction: it may call a new paragraph "seen".
4. **So a Bloom hit can only nominate, never decide.** The candidates are a superset of the repeated digests. A second, exact pass over just the candidates removes the false positives; because candidates are few (repeats plus about $pN$), that pass is cheap.
5. **Exact confirmation needs equal digests together.** Sorting the occurrences $(h, i, j)$ makes runs of equal $h$ adjacent and, inside a run, puts them in input order. A run of length 1 is a false positive; in a longer run, everything after the first element is a repeat. Sorting also scales to an external sort when the list outgrows memory.
6. **The algorithm reads the input four times but the stage gets a one-shot iterator.** So the first pass spools it to disk, and memory holds only the filter, the candidates, and their occurrences. The spool is a corpus-sized copy, so it lives in a `TemporaryDirectory` that the generator's `with` block removes on exhaustion or `close()`.
7. **Size the filter by what it holds: paragraphs, not documents.** The first pass counts $N$, and `bloom_rate(B)` turns the config's byte budget into the target $p$.

### Paragraphs, digests, and sizing

The helpers in [`dedup.py`](../../../../course/solutions/python/corpus/dedup.py) are short:

```python
_BLANK = re.compile(r"\n[ \t]*\n")   # a blank line may hold spaces or tabs

def paragraphs(text: str) -> list[str]:
    return [p.strip() for p in _BLANK.split(text) if p.strip()]   # empty pieces skipped

def paragraph_hash(p: str) -> bytes:
    return hashlib.sha256(p.encode("utf-8")).digest()

def bloom_rate(bloom_bytes_per_item: float) -> float:
    if not bloom_bytes_per_item > 0:
        raise ValueError("bloom_bytes_per_item must be positive")
    # m/n = 8B bits per item at the optimal k gives p = exp(-(m/n)(ln 2)^2)
    return math.exp(-8.0 * bloom_bytes_per_item * math.log(2) ** 2)
```

### The private filter

The filter follows `formats/bloom.md`: $m$ from $N$ and $p$, $k$ rounded from $(m/N)\ln 2$, and double hashing with FNV-1a and SplitMix64:

```python
class _Bloom:
    def __init__(self, n: int, rate: float) -> None:
        if n < 1 or not 0 < rate < 1:
            raise ValueError("Bloom needs n >= 1 and 0 < rate < 1")
        self.m = math.ceil(-n * math.log(rate) / math.log(2) ** 2)
        self.k = max(1, math.floor(self.m / n * math.log(2) + 0.5))   # round half up
        self.bits = bytearray((self.m + 7) // 8)

    def _positions(self, item: bytes):
        h1 = fnv1a64(item)
        h2 = splitmix64(h1) | 1          # odd step: the k positions do not collapse onto a short cycle
        for i in range(self.k):
            yield ((h1 + i * h2) & ((1 << 64) - 1)) % self.m   # Kirsch-Mitzenmacher, mod 2^64 then mod m
```

### The stage: spool, screen, confirm, emit

The whole body sits inside the `TemporaryDirectory` block, so the spool goes away however the generator ends.

```python
def exact_dedup(docs, bloom_bytes_per_item=10, *, spool_dir=None, stats=None):
    rate = bloom_rate(bloom_bytes_per_item)            # fails fast on B <= 0, before reading input
    with tempfile.TemporaryDirectory(prefix="dedup-", dir=spool_dir) as td:
        spool = Path(td) / "spool.jsonl"
        n_docs = n_paras = 0
        with spool.open("w", encoding="utf-8") as fh:
            for doc in docs:                           # pass 1: the only read of the real input
                fh.write(_doc_line(doc) + "\n")
                n_docs += 1
                n_paras += len(paragraphs(doc.text))   # N counts paragraphs, not documents

        # Screen: one Bloom filter sized for every paragraph of the input.
        bloom = _Bloom(max(n_paras, 1), rate)          # max(.., 1): an empty input still gets a filter
        candidates: set[bytes] = set()
        for doc in _read_spool(spool):
            for p in paragraphs(doc.text):
                h = paragraph_hash(p)
                if bloom.contains(h):
                    candidates.add(h)                  # maybe seen: nominate, do not drop
                else:
                    bloom.insert(h)

        # Confirm: sort-merge the occurrences of candidate hashes.
        occ: list[tuple[bytes, int, int]] = []
        for i, doc in enumerate(_read_spool(spool)):
            for j, p in enumerate(paragraphs(doc.text)):
                h = paragraph_hash(p)
                if h in candidates:
                    occ.append((h, i, j))
        occ.sort()                                     # equal h adjacent; within h, input order
        drop: set[tuple[int, int]] = set()
        false_pos = 0
        k = 0
        while k < len(occ):
            end = k
            while end + 1 < len(occ) and occ[end + 1][0] == occ[k][0]:
                end += 1                               # [k, end] is one run of equal digests
            if end == k:
                false_pos += 1  # the filter said "seen", but this hash occurs once
            for _, i, j in occ[k + 1 : end + 1]:
                drop.add((i, j))  # every occurrence after the first
            k = end + 1

        # Emit, in input order.
        dropped_docs = out_docs = 0
        for i, doc in enumerate(_read_spool(spool)):
            paras = paragraphs(doc.text)
            keep = [p for j, p in enumerate(paras) if (i, j) not in drop]
            if not keep:
                dropped_docs += 1
                continue
            out_docs += 1
            if len(keep) == len(paras):
                yield doc                              # untouched: exact original text, not a rebuild
            else:
                yield dataclasses.replace(doc, text="\n\n".join(keep))
        if stats is not None:
            stats.update(docs_in=n_docs, paragraphs=n_paras, bloom_bits=bloom.m,
                         candidates=len(candidates), false_positives=false_pos,
                         duplicate_paragraphs=len(drop), docs_dropped=dropped_docs,
                         docs_out=out_docs)            # filled only once the output is exhausted
```

In the solution file the `bloom = _Bloom(...)` line is indented one level deeper, so it runs just before the spool file is closed; the behaviour is the same. The full solution is [`dedup.py`](../../../../course/solutions/python/corpus/dedup.py).

### Why this design

The cost is four linear passes over the spool, each hashing every paragraph: $O(T)$ for $T$ bytes of text. Memory is $8BN$ bits for the filter plus $O(R + pN)$ for the candidates and their occurrences, where $R$ is the number of repeated digests; the sort is $O(C \log C)$ in the number of candidate occurrences $C$. Disk holds one copy of the corpus.

A plain set is simpler and exact, but at 90 bytes per paragraph it is nine times the default filter; at a billion paragraphs that is the difference between 10 GB and 90 GB. Trusting the Bloom filter (dropping every hit) saves the confirm pass but silently deletes unique paragraphs at rate $p$, and at a small $B$ that is a third of what the filter has seen. The confirm pass makes correctness independent of $B$, so $B$ is purely a memory and time knob. A dict of counts for the confirm would also work while candidates fit in memory, but sort-merge gives the same answer and has a direct path to an external sort when they do not.

The barrier is the accepted cost. "First occurrence" cannot be decided until every earlier paragraph has been seen, and dropping a later one needs to know there was an earlier one, so some pass must see everything before emitting. Spooling to disk keeps that barrier out of memory.

## What a correct dedup must handle

The tests pin the definition of "first", the independence from filter size, and the cleanup:

| Case | What must hold | Why it matters |
|---|---|---|
| The hand example | A unchanged, B as P3, C dropped, the counts | "first occurrence" means exactly this |
| A corpus with planted copies | 200 documents: paragraph, repeat, and dropped counts and the sha256 of every kept text, all fixed by the generator | the counts in the manifest and the datasheet |
| A tiny filter | at $B = 0.25$ (over 50 false positives) the output equals a plain set-based dedup, and candidates = repeated digests + false positives | a Bloom positive never drops a unique paragraph |
| Random corpora at several sizes | 40 seeded corpora at $B$ = 0.1, 1, 10 equal the set-based dedup | correctness does not depend on the filter size |
| Paragraph splitting | blank lines with spaces or tabs, single line breaks, stripping, empty pieces | what counts as a repeat |
| A repeat inside one document | a repeated footer inside one document goes, the document stays | repeats are positional, not per document |
| Untouched documents | a document with no repeats comes back equal, line breaks and meta included | dedup changes only what it dedups |
| Filter sizing | `bloom_rate` values; $m = \lceil -N \ln p / (\ln 2)^2 \rceil$ within 1 of $8BN$ | the config's memory knob means what it says |
| Stats | every `STATS_KEYS` entry, also for an empty input | the manifest and datasheet fields |
| Spool cleanup | no spool left after a full read or an early `close()` | the spool is a full copy of the corpus |
| Use as a stage | composes with `functools.partial`; partial documents joined by `\n\n` | the pipeline wiring of `CorpusBuild` |

## What can go wrong

The worst pitfall is invisible.

> **Warning:** Dropping every Bloom positive without confirming makes unique paragraphs vanish with a small filter, and nothing reports it. In the solution a Bloom hit only adds to `candidates`; the sort-merge decides.

Two pitfalls break the merge itself. Merging occurrences without sorting them leaves equal digests non-adjacent, so repeats look unique and survive; the solution calls `occ.sort()` before the merge. Keeping the last occurrence instead of the first makes the output depend on what came later, and reruns with appended sources rewrite earlier documents; the solution drops `occ[k + 1 : end + 1]`, keeping the first element of each sorted run.

Two pitfalls are about the unit. Hashing whole documents lets copies with a new title or footer survive, and repeat counts come out far too low; the solution hashes each paragraph from `paragraphs(doc.text)`. Splitting paragraphs only on `\n\n` lets a blank line holding a space hide a paragraph boundary, so repeats stick to their neighbours; the solution splits on `\n[ \t]*\n`.

The rest are about side effects. Rebuilding every document from its paragraphs changes untouched documents (line breaks collapse), so later hashes no longer match the filter stage's output; the solution yields the original `doc` when `len(keep) == len(paras)`. Sizing the filter by documents rather than paragraphs gives a filter several times too small, with a false-positive rate far above the budget; the solution uses `_Bloom(max(n_paras, 1), rate)`. And a spool that outlives the stage leaves a full copy of the corpus in the temp directory on every run; the whole stage body runs inside `tempfile.TemporaryDirectory`.

## Where this leads

This stage relies on the `Doc`, `compose`, and the normalized text, whose paragraphs are separated by exactly one blank line, from [the filter stages](02-extract-normalize-and-quality-filters.md), and on FNV-1a and SplitMix64 for the shared Bloom hash rules. Going forward, [near dedup with MinHash and LSH](04-near-dedup-minhash-lsh.md) runs on what this stage keeps, so exact copies never reach the more expensive pass; without this stage, near dedup would have to remove exact copies itself, at MinHash cost. The manifest's `dedup.exact_dropped`, written by [the shard writer](06-parquet-shards-and-manifest.md), comes from these stats.

### Further reading

- Lee et al., *Deduplicating Training Data Makes Language Models Better* (2022), and its suffix-array dedup in `google-research/deduplicate-text-datasets`; datatrove's exact-substring stages (`datatrove/pipeline/dedup/exact_substrings.py`): repeated substrings of any length (50 tokens and up), across the whole corpus.
- Wenzek et al., *CCNet* (2019), section 3.1: paragraph-level dedup with sharded hash sets over Common Crawl snapshots.
- Dolma's `bff` (`allenai/bff`): a Rust Bloom filter deduplicator over billions of paragraphs.
- Broder and Mitzenmacher, "Network Applications of Bloom Filters: A Survey" (2004).
- Kirsch and Mitzenmacher, "Less Hashing, Same Performance" (2006): the double-hashing scheme used here.
- External merge sort and Spark `dropDuplicates` (see [Batch & Streaming Processing](../03-batch-streaming/README.md)): candidate lists larger than memory, spilled and merged from disk.
- Fan et al., "Cuckoo Filter" (2014), and counting Bloom filters: deletions, and better space at low false-positive rates.

## Key terms

- **Paragraph**: a stripped, non-empty piece of text between blank lines.
- **Repeat**: an occurrence of a paragraph whose digest occurred earlier in the input.
- **Bloom filter**: a bit array with $k$ hash functions that has no false negatives and a tunable false-positive rate.
- **Candidate**: a digest the Bloom filter reported as maybe seen; a superset of the true repeats.
- **Sort-merge confirm**: sorting candidate occurrences so equal digests form runs, keeping the first of each run.
- **Spool**: an on-disk copy of the input that lets a one-shot iterator be read several times.

## Summary

- Exact dedup keeps the first occurrence of every paragraph, in input order, and removes the later ones; a document that loses every paragraph is dropped.
- A Bloom filter answers "definitely new" or "maybe seen" in about 10 bytes per item, a tenth of a set of hashes; its "maybe" is only a candidate.
- A sort-merge over the candidates' exact hashes confirms which are real repeats, so no unique paragraph is ever dropped, whatever the filter size.
- The filter is sized from `bloom_bytes_per_item`: $p = e^{-8B(\ln 2)^2}$ and $m \approx 8BN$ bits for $N$ paragraphs.
- Dedup must read its whole input before emitting anything; it spools to disk, not memory, and cleans up after itself.

## Review questions

1. Why can a Bloom filter be used to find repeats without ever dropping a unique paragraph?
   <details><summary>Answer</summary>Because the filter is used only to nominate candidates. It has no false negatives, so every real repeat is nominated; its false positives are filtered out by the exact sort-merge, where a unique paragraph forms a run of length 1 and is kept.</details>
2. Why is the filter sized from the paragraph count of the first pass and not from the document count?
   <details><summary>Answer</summary>The filter holds paragraphs. Sizing it for documents gives an $m$ several times too small for the $N$ actually inserted, and the false-positive rate grows exponentially as $m/N$ falls ($p = e^{-(m/N)(\ln 2)^2}$). Correctness survives, but candidates and confirm cost balloon.</details>
3. Why does `occ.sort()` order occurrences correctly within a run, and why does that matter?
   <details><summary>Answer</summary>Tuples sort by digest, then by document index, then by paragraph index, so within one digest the occurrences are in input order. The first element of each run is therefore the first occurrence, which is the one to keep.</details>
4. Why does the emit pass yield the original `doc` when nothing was removed, rather than `"\n\n".join(paras)`?
   <details><summary>Answer</summary>Rebuilding from paragraphs would strip each paragraph and rewrite every blank line (a run of several, or one holding spaces or tabs) as exactly `\n\n`, changing documents dedup did not touch. Later stages (and the hashes in the manifest) would then disagree with the filter stage's output for no reason.</details>
5. What happens to the spool if a consumer stops reading after the first document?
   <details><summary>Answer</summary>Closing the generator raises `GeneratorExit` at the `yield`, which unwinds the `with tempfile.TemporaryDirectory(...)` block and deletes the directory. A corpus-sized copy is never left behind.</details>
