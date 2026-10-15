<!-- ol:module data.03 -->
# Exact dedup: paragraph hashes, Bloom screen, sort-merge confirm

## Overview

| | |
|---|---|
| **Module** | `data.03` · build · Python · Pass 3 · 3 to 4 h |
| **You build** | `python/corpus/dedup.py`: `paragraphs`, `paragraph_hash`, `bloom_rate`, `exact_dedup` (a stage), `STATS_KEYS` |
| **Contract** | [`course/contracts/py/corpus/dedup.pyi`](../../../../course/contracts/py/corpus/dedup.pyi) · the Bloom sizing and hash rules: [`formats/bloom.md`](../../../../course/contracts/formats/bloom.md) |
| **Tests** | `course/tests/data.03/` (what they check: section 4); fixtures `course/fixtures/data.03/` (200 documents with planted duplicates) |
| **Needs** | `data.02` `Doc` and `compose` · `M06.3` FNV-1a and SplitMix64 · reading: `S-M06b` (false-positive rate, optimal $k$) |
| **Used by** | near dedup (`data.04`) runs on what this stage keeps; it joins the registry with its batch |
| **Milestone** | `MS-corpus` |
| **Optional depth** | Lee et al., *Deduplicating Training Data Makes Language Models Better* (2022, free); Wenzek et al., *CCNet* (2019), section 3.1 (paragraph-level dedup, free); Broder and Mitzenmacher, "Network Applications of Bloom Filters: A Survey" (2004, free); Kirsch and Mitzenmacher, "Less Hashing, Same Performance" (2006, free) |

## Key Takeaways

- Exact dedup keeps the **first** occurrence of every paragraph, in input order, and removes the later ones; a document that loses every paragraph is dropped (`test_hand_example_three_documents`, `test_fixture_duplicate_counts`).
- A Bloom filter answers "definitely new" or "maybe seen" in about 10 bytes per item, a tenth of what a set of hashes costs. Its "maybe" is only a candidate (`test_zero_false_drops_with_a_tiny_filter`).
- A sort-merge over the candidates' exact hashes confirms which are real repeats, so no unique paragraph is ever dropped, whatever the filter size (`test_random_corpora_match_the_set_oracle`).
- The filter is sized from the corpus config's `bloom_bytes_per_item`: $p = e^{-8B(\ln 2)^2}$ and $m \approx 8BN$ bits for $N$ paragraphs (`test_bloom_is_sized_from_bytes_per_item`).
- Dedup must read its whole input before it can emit anything; it spools to disk, not to memory, and cleans up after itself (`test_the_spool_is_removed`).

## How to work this chapter

```bash
ol start data.03              # stubs python/corpus/dedup.py
ol tests data.03              # read the test catalog first
ol check data.03              # needs data.02 and M06.3 passing (or --ref-deps)
ol diff  data.03              # after passing: your code against the reference
```

The Python Bloom screen uses the hash functions from `M06.3` and follows the shared bit-sizing rules. The standalone Rust Bloom module has its own tests and does not cross into this stage.

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

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_three_documents` | unit | the section 3 example: A unchanged, B as P3, C dropped, the counts | you and the test agree on "first occurrence" |
| `test_fixture_duplicate_counts` | conformance | 200 documents with planted copies: paragraph, repeat, and dropped counts and the sha256 of every kept text, all fixed by the generator | the counts in the manifest and the datasheet |
| `test_zero_false_drops_with_a_tiny_filter` | property | at $B = 0.25$ (over 50 false positives) the output equals the set oracle, and candidates = repeated digests + false positives | a Bloom positive never drops a unique paragraph |
| `test_random_corpora_match_the_set_oracle` | property | 40 seeded corpora at $B$ = 0.1, 1, 10: equal to the set oracle | correctness does not depend on the filter size |
| `test_paragraph_split_rules` | unit | blank lines with spaces or tabs, single line breaks, stripping, empty pieces | what counts as a repeat |
| `test_a_repeat_inside_one_document_is_removed` | boundary | a repeated footer inside one document goes, the document stays | repeats are positional, not per document |
| `test_untouched_documents_keep_their_exact_text` | unit | a document with no repeats comes back equal, line breaks and meta included | dedup changes only what it dedups |
| `test_bloom_is_sized_from_bytes_per_item` | unit | `bloom_rate` values; $m = \lceil -N \ln p / (\ln 2)^2 \rceil$ within 1 of $8BN$ on the fixture | the memory knob of the config means what it says |
| `test_stats_name_every_key` | unit | every `STATS_KEYS` entry, also for an empty input | the manifest and datasheet fields |
| `test_the_spool_is_removed` | unit | no spool left after a full read or an early `close()` | the spool is a full copy of the corpus |
| `test_exact_dedup_is_a_stage` | unit | composes with `functools.partial`; partial documents joined by `\n\n` | the pipeline wiring of `CorpusBuild` |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. Keeping the last occurrence instead of the first | the output depends on what came later; reruns with appended sources rewrite earlier documents | `test_hand_example_three_documents` (mutant `s01`) |
| 2. Hashing whole documents | copies with a new title or footer survive; repeat counts far too low | `test_fixture_duplicate_counts` (mutant `s03`) |
| 3. Dropping every Bloom positive without confirming | with a small filter, unique paragraphs vanish; nothing reports it | `test_zero_false_drops_with_a_tiny_filter` (mutant `s04`) |
| 4. Merging occurrences without sorting them | equal digests are not adjacent; repeats look unique and survive | `test_fixture_duplicate_counts` (mutant `s05`) |
| 5. Splitting paragraphs only on `\n\n` | a blank line holding a space hides a paragraph boundary; repeats stick to their neighbours | `test_paragraph_split_rules` (mutant `s07`) |
| 6. Rebuilding every document from its paragraphs | untouched documents change (line breaks collapse); later hashes no longer match the filter stage's output | `test_untouched_documents_keep_their_exact_text` (mutant `s09`) |
| 7. Sizing the filter by documents, not paragraphs | a filter several times too small; the false-positive rate is far above the budget | `test_bloom_is_sized_from_bytes_per_item` (mutant `s10`) |
| 8. A spool that outlives the stage | every run leaves a full copy of the corpus in the temp directory | `test_the_spool_is_removed` (mutant `s13`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `data.02` | `Doc`, `compose`, and the normalized text whose paragraphs are separated by exactly one blank line |
| Back | `M06.3` | FNV-1a and SplitMix64 supply the shared Bloom hash rules |
| Forward | `data.04` | near dedup (MinHash, LSH) runs on what this stage keeps, so exact copies never reach the more expensive pass |
| Forward | `data.06` | the manifest's `dedup.exact_dropped` comes from these stats |

If you skip this module, the near-dedup stage has to remove exact copies itself, at MinHash cost.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `exact_dedup` | Lee et al.'s suffix-array dedup; datatrove's exact-substring stages (`datatrove/pipeline/dedup/exact_substrings.py`) | repeated substrings of any length (50 tokens and up), across the whole corpus | `google-research/deduplicate-text-datasets` |
| paragraph hashes | CCNet's paragraph dedup; Dolma's `bff` (a Bloom filter deduplicator) | sharded hash sets over Common Crawl snapshots; a Rust Bloom filter over billions of paragraphs | `allenai/bff` |
| sort-merge confirm | external merge sort; Spark `dropDuplicates` | candidate lists larger than memory, spilled and merged from disk | `sq.spark` |
| the screen | counting Bloom filters, cuckoo filters | deletions; better space at low false-positive rates | Fan et al., "Cuckoo Filter" (2014) |
