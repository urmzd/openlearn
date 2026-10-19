# Exact dedup: paragraph hashes, Bloom screen, sort-merge confirm

## Overview

| | |
|---|---|
| **Module** | `data.03` · build · Python · Pass 3 · 3 to 4 h |
| **Solution** | [`python/corpus/dedup.py`](../../../../course/solutions/python/corpus/dedup.py): `paragraphs`, `paragraph_hash`, `bloom_rate`, `exact_dedup` (a stage), `STATS_KEYS` |
| **Contract** | [`course/contracts/py/corpus/dedup.pyi`](../../../../course/contracts/py/corpus/dedup.pyi) · the Bloom sizing and hash rules: [`formats/bloom.md`](../../../../course/contracts/formats/bloom.md) |
| **Needs** | `data.02` `Doc` and `compose` · `M06.3` FNV-1a and SplitMix64 · reading: `S-M06b` (false-positive rate, optimal $k$) |
| **Used by** | near dedup (`data.04`) runs on what this stage keeps; it joins the registry with its batch |
| **Optional depth** | Lee et al., *Deduplicating Training Data Makes Language Models Better* (2022, free); Wenzek et al., *CCNet* (2019), section 3.1 (paragraph-level dedup, free); Broder and Mitzenmacher, "Network Applications of Bloom Filters: A Survey" (2004, free); Kirsch and Mitzenmacher, "Less Hashing, Same Performance" (2006, free) |

## Key Takeaways

- Exact dedup keeps the **first** occurrence of every paragraph, in input order, and removes the later ones; a document that loses every paragraph is dropped.
- A Bloom filter answers "definitely new" or "maybe seen" in about 10 bytes per item, a tenth of what a set of hashes costs. Its "maybe" is only a candidate.
- A sort-merge over the candidates' exact hashes confirms which are real repeats, so no unique paragraph is ever dropped, whatever the filter size.
- The filter is sized from the corpus config's `bloom_bytes_per_item`: $p = e^{-8B(\ln 2)^2}$ and $m \approx 8BN$ bits for $N$ paragraphs.
- Dedup must read its whole input before it can emit anything; it spools to disk, not to memory, and cleans up after itself.

## How to read this chapter

Sections 1 to 3 build exact dedup from first principles (paragraph hashes, the Bloom screen, the sort-merge confirm) and run all four passes by hand. Section 4 fixes the interface and lists the cases a correct solution must handle. Section 5 walks through `dedup.py` and explains why each choice is the right one. Try to predict each pass of the solution before reading it.

The Python Bloom screen uses the hash functions from `M06.3` and follows the shared bit-sizing rules in [`formats/bloom.md`](../../../../course/contracts/formats/bloom.md). The standalone Rust Bloom module is a separate unit and does not cross into this stage.

---

## 1. Why now

After `data.02` every document is clean, English, and plausibly prose, and many of them say the same thing. Web pages share navigation and license footers; dataset mirrors repeat whole files; story collections reuse an opening paragraph. A model trained on that sees the repeated text again and again: Lee et al. found that deduplicating C4 removed text that models otherwise reproduced verbatim, and trained faster to the same loss. Repeated rare strings are also the ones a model memorizes, which matters for privacy (`ethics.02`). The obvious fix, a Python set of every paragraph seen, costs about 90 bytes per paragraph (a 32-byte digest as a `bytes` object is 65 bytes, plus its slot in the set), so a billion paragraphs need 90 GB. A compact Python Bloom screen does the screening in about 10 bytes each. This module defines its local use and shows what false positives mean for correctness.

## 2. Principles

### 2.1 Paragraphs and their hashes

A **paragraph** is a piece of the text between blank lines; a blank line is empty or holds only spaces and tabs. Each piece is stripped of white space at both ends, and empty pieces are skipped. A single line break stays inside a paragraph. After `data.02`'s normalization, paragraphs are separated by exactly `\n\n`.

Comparing paragraphs by their text would mean storing the text. Instead compare their **SHA-256 digests**: `paragraph_hash(p) = sha256(p.encode("utf-8")).digest()`, 32 bytes. Equal paragraphs have equal digests; unequal ones collide with probability about $2^{-256}$, which for our purposes is never. A paragraph's **repeat** is any occurrence whose digest occurred earlier in the input, in an earlier document or earlier in the same one.

Why paragraphs and not whole documents? Duplicates are rarely whole documents: a copied story with a new title, or a page with a shared footer, differs as a document and repeats as paragraphs. Why not sentences or substrings? Shorter units remove ordinary phrases ("Once upon a time, there was a little girl.") from unrelated stories; finding repeated substrings of any length needs a suffix array (Lee et al.), a different tool. Paragraphs are the CCNet compromise.

### 2.2 The Bloom filter as a screen

A Bloom filter (`ds.08`, `formats/bloom.md`) is a bit array of $m$ bits and $k$ hash functions. Inserting an item sets its $k$ bits; asking about an item answers "present" when all $k$ bits are set.

| Symbol | Meaning | Type |
|---|---|---|
| $N$ | paragraphs in the input (the items the filter will hold) | integer |
| $m$ | bits in the filter | integer |
| $k$ | hash functions (bits per item) | integer |
| $B$ | `bloom_bytes_per_item`: filter bytes per paragraph, so $m = 8BN$ | float, $> 0$ |
| $p$ | the false-positive rate: the chance that a never-inserted item is reported present | float in $(0, 1)$ |

Two facts from `S-M06b` drive this module:

1. **No false negatives.** An inserted item's bits stay set, so the filter never says "new" about something it has seen.
2. **False positives.** With the optimal $k = (m/N)\ln 2$, the rate is $p = e^{-(m/N)(\ln 2)^2}$. Solving for the memory per item gives $m/N = -\ln p / (\ln 2)^2$. `formats/bloom.md` sizes a filter that way, so for a budget of $B$ bytes per item ($m/N = 8B$ bits) ask for

$$p = e^{-8B(\ln 2)^2}, \qquad m = \left\lceil \frac{-N \ln p}{(\ln 2)^2} \right\rceil \approx 8BN .$$

`bloom_rate(B)` computes that $p$: $B = 10$ gives $p \approx 2.0 \times 10^{-17}$; $B = 1$ gives $p \approx 0.021$; $B = 0.25$ gives $p \approx 0.38$.

The screen pass hashes every paragraph in input order and asks the filter: if it answers "present", the digest becomes a **candidate**; otherwise it is inserted. By fact 1, every digest that occurs twice or more becomes a candidate at its second occurrence. By fact 2, some digests that occur once become candidates too: the false positives. So the candidates are a **superset** of the repeated digests, and a small one: (repeated digests) + (about $pN$ false positives).

### 2.3 Confirming by sort-merge

The confirm pass reads the input again and records an **occurrence** $(h, i, j)$ for every paragraph whose digest $h$ is a candidate, where $i$ is the document's index and $j$ the paragraph's index within it. Sort the occurrences: equal digests become adjacent, and within one digest the order is input order (by $i$, then $j$). Then **merge**, walking runs of equal $h$:

- a run of length 1 is a false positive: that paragraph is unique, keep it;
- a run of length $r \ge 2$ is a confirmed repeat: keep its first occurrence and mark the other $r - 1$ for removal.

Sorting is what makes this scale: an external sort handles candidate lists larger than memory, which a hash map of counts does not. Here the candidates fit in a Python list.

### 2.4 Four passes over a spool

The stage consumes an iterator, which can be read once, but the algorithm needs the input three more times. So it **spools**: the first pass writes every document as one JSON line to a temporary file and counts $N$; the screen, confirm, and emit passes reread that file. Memory holds the filter, the candidate set, and the candidate occurrences, never the documents. The cost is that nothing is emitted until the input is exhausted: dedup is a barrier in the otherwise streaming pipeline. The spool lives in a `tempfile.TemporaryDirectory` (under `spool_dir` when given), which is removed when the generator finishes **or is closed early**, because the spool is a full copy of the corpus.

The emit pass yields each document with its marked paragraphs removed: unchanged (the same `Doc`, exact text and meta) when it lost nothing, with the remaining paragraphs joined by `\n\n` when it lost some, and not at all when it lost every paragraph. Output order is input order. With `stats={}` passed in, the stage fills `docs_in`, `paragraphs`, `bloom_bits` ($m$), `candidates`, `false_positives`, `duplicate_paragraphs`, `docs_dropped`, and `docs_out` once the output is exhausted; the corpus manifest's `dedup.exact_dropped` and the datasheet report them.

## 3. Worked example by hand

Three documents, $B = 10$:

| Doc | Text |
|---|---|
| A (`s:0`) | `The cat sat.` (P1), blank line, `The dog ran.` (P2) |
| B (`s:1`) | `The dog ran.` (P2), blank line, `A bird sang.` (P3) |
| C (`s:2`) | `The cat sat.` (P1) |

**Spool.** $N = 5$ paragraphs. The filter has $m = \lceil 5 \cdot 80.0 \rceil = 400$ bits and $k = \text{round}(400/5 \cdot \ln 2) = 55$ hashes. The digests begin `h(P1) = 84549cfa...`, `h(P2) = 0ea960d6...`, `h(P3) = c2c47b92...`.

**Screen**, in input order:

| Occurrence $(i, j)$ | Paragraph | Filter says | Action |
|---|---|---|---|
| (0, 0) | P1 | new | insert |
| (0, 1) | P2 | new | insert |
| (1, 0) | P2 | present | candidate `h(P2)` |
| (1, 1) | P3 | new | insert |
| (2, 0) | P1 | present | candidate `h(P1)` |

**Confirm.** The occurrences of the two candidates are $(h(P1), 0, 0)$, $(h(P2), 0, 1)$, $(h(P2), 1, 0)$, $(h(P1), 2, 0)$. Sorted (`0e...` before `84...`): $(h(P2), 0, 1)$, $(h(P2), 1, 0)$, $(h(P1), 0, 0)$, $(h(P1), 2, 0)$. Two runs of length 2: remove $(1, 0)$ and $(2, 0)$. No run of length 1, so no false positives.

**Emit.** A loses nothing and comes out unchanged. B loses P2 and comes out as `A bird sang.`. C loses its only paragraph and is dropped. Stats: 3 documents in, 5 paragraphs, 400 bits, 2 candidates, 0 false positives, 2 duplicate paragraphs, 1 document dropped, 2 out.

With $B = 0.25$ the filter would have $\lceil 5 \cdot 2 \rceil = 10$ bits and might report P3 as present on first sight. P3 would become a candidate, its run in the confirm pass would have length 1, and it would be kept: the output is identical, only `false_positives` grows.

## 4. The interface

```python
STATS_KEYS: tuple[str, ...]
def paragraphs(text: str) -> list[str]: ...
def paragraph_hash(p: str) -> bytes: ...
def bloom_rate(bloom_bytes_per_item: float) -> float: ...
def exact_dedup(docs: Iterable[Doc], bloom_bytes_per_item: float = 10, *,
                spool_dir: str | Path | None = None,
                stats: dict[str, Any] | None = None) -> Iterator[Doc]: ...
```

In a pipeline, bind the options: `compose(build_filters(cfg["filters"]), functools.partial(exact_dedup, bloom_bytes_per_item=cfg["dedup"]["bloom_bytes_per_item"]))`. The private Python filter is sized for `max(N, 1)` at `bloom_rate(B)`. Use `insert` and `contains`; its `m` is the bit count reported in stats.

### What a correct solution must handle

| Case | Kind of check | What must hold | Why it matters downstream |
|---|---|---|---|
| The hand example | unit | the section 3 example: A unchanged, B as P3, C dropped, the counts | "first occurrence" means exactly this |
| A corpus with planted copies | conformance | 200 documents: paragraph, repeat, and dropped counts and the sha256 of every kept text, all fixed by the generator | the counts in the manifest and the datasheet |
| A tiny filter | property | at $B = 0.25$ (over 50 false positives) the output equals a plain set-based dedup, and candidates = repeated digests + false positives | a Bloom positive never drops a unique paragraph |
| Random corpora at several sizes | property | 40 seeded corpora at $B$ = 0.1, 1, 10 equal the set-based dedup | correctness does not depend on the filter size |
| Paragraph splitting | unit | blank lines with spaces or tabs, single line breaks, stripping, empty pieces | what counts as a repeat |
| A repeat inside one document | boundary | a repeated footer inside one document goes, the document stays | repeats are positional, not per document |
| Untouched documents | unit | a document with no repeats comes back equal, line breaks and meta included | dedup changes only what it dedups |
| Filter sizing | unit | `bloom_rate` values; $m = \lceil -N \ln p / (\ln 2)^2 \rceil$ within 1 of $8BN$ | the memory knob of the config means what it says |
| Stats | unit | every `STATS_KEYS` entry, also for an empty input | the manifest and datasheet fields |
| Spool cleanup | unit | no spool left after a full read or an early `close()` | the spool is a full copy of the corpus |
| Use as a stage | unit | composes with `functools.partial`; partial documents joined by `\n\n` | the pipeline wiring of `CorpusBuild` |

## 5. The solution, step by step

### 5.1 From first principles

1. **"Keep the first occurrence" is a property of input order**, so the algorithm must see occurrences in order and identify each by position $(i, j)$: document index, paragraph index. Keeping the first makes the result stable when sources are appended later.
2. **Comparing text means storing text; comparing digests does not.** SHA-256 turns each paragraph into 32 bytes with collisions at $2^{-256}$, so "equal digest" can stand for "equal paragraph".
3. **A set of digests is exact but costs about 90 bytes each in Python.** A Bloom filter costs $B$ bytes each and never forgets an insert (no false negatives), so every true repeat is caught at its second occurrence. Its only error is the other direction: it may call a new paragraph "seen".
4. **So a Bloom hit can only nominate, never decide.** The candidates are a superset of the repeated digests. A second, exact pass over just the candidates removes the false positives; because candidates are few (repeats plus about $pN$), that pass is cheap.
5. **Exact confirmation needs equal digests together.** Sorting the occurrences $(h, i, j)$ makes runs of equal $h$ adjacent and, inside a run, puts them in input order. A run of length 1 is a false positive; in a longer run, everything after the first element is a repeat. Sorting also scales to an external sort when the list outgrows memory.
6. **The algorithm reads the input four times but the stage gets a one-shot iterator.** So the first pass spools it to disk, and memory holds only the filter, the candidates, and their occurrences. The spool is a corpus-sized copy, so it lives in a `TemporaryDirectory` that the generator's `with` block removes on exhaustion or `close()`.
7. **Size the filter by what it holds: paragraphs, not documents.** The first pass counts $N$, and `bloom_rate(B)` turns the config's byte budget into the target $p$.

### 5.2 The code, annotated

Paragraphs, digests, and the sizing rule ([`dedup.py`](../../../../course/solutions/python/corpus/dedup.py)):

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

The private filter follows `formats/bloom.md`: $m$ from $N$ and $p$, $k$ rounded from $(m/N)\ln 2$, and double hashing with FNV-1a and SplitMix64 from `M06.3`:

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

The stage itself: spool, screen, confirm, emit. The whole body sits inside the `TemporaryDirectory` block, so the spool goes away however the generator ends.

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

(In the solution file the `bloom = _Bloom(...)` line is indented one level deeper, so it runs just before the spool file is closed; the behaviour is the same.)

Full solution: [`dedup.py`](../../../../course/solutions/python/corpus/dedup.py).

### 5.3 Why this solution

- **Cost.** Four linear passes over the spool, each hashing every paragraph: $O(T)$ for $T$ bytes of text. Memory is $8BN$ bits for the filter plus $O(R + pN)$ for the candidates and their occurrences, where $R$ is the number of repeated digests; the sort is $O(C \log C)$ in the number of candidate occurrences $C$. Disk holds one copy of the corpus.
- **A plain set** is simpler and exact, but 90 bytes per paragraph, nine times the default filter. At a billion paragraphs that is the difference between 10 GB and 90 GB.
- **Trusting the Bloom filter** (drop every hit) saves the confirm pass but silently deletes unique paragraphs at rate $p$; at a small $B$ that is a third of what the filter has seen. The confirm pass makes correctness independent of $B$, so $B$ is purely a memory/time knob.
- **A dict of counts for the confirm** would also work while candidates fit in memory, but sort-merge has the same answer and a direct path to an external sort when they do not.
- **The barrier is the accepted cost.** "First occurrence" cannot be decided until every earlier paragraph has been seen, and dropping a later one needs to know there was an earlier one, so some pass must see everything before emitting. Spooling to disk keeps that barrier out of memory.

### 5.4 Explain it back

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

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. Keeping the last occurrence instead of the first | the output depends on what came later; reruns with appended sources rewrite earlier documents | drops `occ[k + 1 : end + 1]`, keeping the first element of each sorted run |
| 2. Hashing whole documents | copies with a new title or footer survive; repeat counts far too low | hashes each paragraph from `paragraphs(doc.text)` |
| 3. Dropping every Bloom positive without confirming | with a small filter, unique paragraphs vanish; nothing reports it | a Bloom hit only adds to `candidates`; the sort-merge decides |
| 4. Merging occurrences without sorting them | equal digests are not adjacent; repeats look unique and survive | `occ.sort()` before the merge |
| 5. Splitting paragraphs only on `\n\n` | a blank line holding a space hides a paragraph boundary; repeats stick to their neighbours | splits on `\n[ \t]*\n` |
| 6. Rebuilding every document from its paragraphs | untouched documents change (line breaks collapse); later hashes no longer match the filter stage's output | yields the original `doc` when `len(keep) == len(paras)` |
| 7. Sizing the filter by documents, not paragraphs | a filter several times too small; the false-positive rate is far above the budget | `_Bloom(max(n_paras, 1), rate)` |
| 8. A spool that outlives the stage | every run leaves a full copy of the corpus in the temp directory | the whole stage body runs inside `tempfile.TemporaryDirectory` |

## 7. Where it's used next


| Direction | Module | How it uses this |
|---|---|---|
| Back | `data.02` | `Doc`, `compose`, and the normalized text whose paragraphs are separated by exactly one blank line |
| Back | `M06.3` | FNV-1a and SplitMix64 supply the shared Bloom hash rules |
| Forward | `data.04` | near dedup (MinHash, LSH) runs on what this stage keeps, so exact copies never reach the more expensive pass |
| Forward | `data.06` | the manifest's `dedup.exact_dropped` comes from these stats |

Without this stage, the near-dedup stage would have to remove exact copies itself, at MinHash cost.

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `exact_dedup` | Lee et al.'s suffix-array dedup; datatrove's exact-substring stages (`datatrove/pipeline/dedup/exact_substrings.py`) | repeated substrings of any length (50 tokens and up), across the whole corpus | `google-research/deduplicate-text-datasets` |
| paragraph hashes | CCNet's paragraph dedup; Dolma's `bff` (a Bloom filter deduplicator) | sharded hash sets over Common Crawl snapshots; a Rust Bloom filter over billions of paragraphs | `allenai/bff` |
| sort-merge confirm | external merge sort; Spark `dropDuplicates` | candidate lists larger than memory, spilled and merged from disk | `sq.spark` |
| the screen | counting Bloom filters, cuckoo filters | deletions; better space at low false-positive rates | Fan et al., "Cuckoo Filter" (2014) |
