<!-- ol:module data.04 -->
# Near-duplicate dedup: MinHash, LSH, union-find, and decontamination

## Overview

| | |
|---|---|
| **Module** | `data.04` · build · Python · Pass 3 · 4 to 5 h |
| **You build** | `python/corpus/minhash.py`: `words`, `shingles`, `jaccard`, `shingle_hashes`, `perm_params`, `minhash`, `estimate`, `LSH`, `UnionFind`, `signatures` (process-parallel), `clusters`, `near_dedup` (a stage), `protected_ngrams`, `contaminated`, `decontaminate` (a stage) |
| **Contract** | [`course/contracts/py/corpus/minhash.pyi`](../../../../course/contracts/py/corpus/minhash.pyi) · the shard column it feeds: [`formats/corpus-shard.md`](../../../../course/contracts/formats/corpus-shard.md) |
| **Tests** | `course/tests/data.04/` (what they check: section 4); fixtures `course/fixtures/data.04/` (planted near-duplicate clusters, borderline pairs, protected eval sets) |
| **Needs** | `data.02` `Doc` · `data.03` exact dedup runs first · `M06.3` `PCG32` and FNV-1a (or `--ref-deps`) · reading: `S-M06b` (Jaccard and the S-curve by hand) |
| **Used by** | `data.05` scrubs what survives and keeps its cluster tag · `data.06` turns the tag into the `minhash_cluster` column · later: `data.09` runs this stage inside `CorpusBuild` |
| **Milestone** | `MS-corpus` |
| **Optional depth** | Broder, *On the Resemblance and Containment of Documents* (1997); Leskovec, Rajaraman, and Ullman, [*Mining of Massive Datasets*](http://www.mmds.org/), chapter 3 (free); Lee et al., [*Deduplicating Training Data Makes Language Models Better*](https://arxiv.org/abs/2107.06499) (free); Brown et al., [*Language Models are Few-Shot Learners*](https://arxiv.org/abs/2005.14165), appendix C (13-gram contamination, free) |

## Key Takeaways

- The share of equal entries in two MinHash signatures is an unbiased estimate of the Jaccard similarity of their shingle sets (`test_estimator_is_unbiased`).
- Banding the signature into $b$ bands of $r$ rows turns that estimate into an S-curve: pairs above about $(1/b)^{1/r}$ become candidates, pairs below rarely do, and no pair is compared to all the others (`test_lsh_s_curve`).
- Candidates are confirmed by their estimate and merged by union-find, whose root is the smallest id, so the clusters and the survivor never depend on input order or worker count (`test_invariant_to_input_order`, `test_invariant_to_worker_count`).
- Decontamination drops every document that shares a word 13-gram with a protected eval or validation text; that is what makes your benchmark numbers mean something (`test_decontaminate_fixture`).

## How to work this chapter

```bash
ol start data.04              # stubs minhash.py into python/corpus/
ol tests data.04              # the course test catalog
ol check data.04              # exit code is the verdict
ol check data.04 --ref-deps   # only if data.02, data.03, or M06.3 is not passing yet
ol tdd red data.04            # rung R4: write your property tests first (section 4)
ol mutate data.04             # how many planted bugs your tests catch
ol diff  data.04              # after passing: your code against the reference
```

---

## 1. Why now

After `data.03` your corpus has no exact copies left, but it is still full of almost-copies: the same story reposted with one word changed, a page scraped twice with a different footer, a template filled with other names. Exact hashes see two different documents. A model trained on them memorizes the repeated text and spends its capacity on it, and, worse, some of those almost-copies are your evaluation stories: train on them and your validation loss in `C1` measures recall, not learning. Today nothing in `python/corpus/` can tell that two texts are 95% the same without comparing every pair, which at a million documents is half a trillion comparisons. This module finds near duplicates in roughly linear time and removes anything that overlaps your protected eval and validation sets.

## 2. Principles

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

**Shingles.** Lower-case the text and split it into word runs (`re.findall(r"\w+", text.lower())`, so punctuation and case never matter). A $k$-shingle is $k$ consecutive words joined by one space; "the cat sat on the mat" has the 3-shingles "the cat sat", "cat sat on", "sat on the", "on the mat". Shingles turn "how similar are these texts?" into "how much do these sets overlap?". A text shorter than $k$ words is one shingle (all its words), so a three-word line can still match its copy; a text with no words has no shingles and is nobody's near duplicate.

**Jaccard similarity.** $J(A, B)$ is the size of the overlap divided by the size of the union: 1 for equal sets, 0 for disjoint ones. One changed word in a 150-word story changes at most $k$ shingles, so $J$ stays above 0.9; two unrelated stories share almost none.

**MinHash.** Pick a random hash function $h$ and look at the shingle of $A \cup B$ with the smallest hash. It is equally likely to be any element of the union, and the minimum of $A$ equals the minimum of $B$ exactly when that element is in both. So

$$\Pr[\min_{A} h = \min_{B} h] = \frac{\lvert A \cap B \rvert}{\lvert A \cup B \rvert} = J(A, B).$$

With $n$ independent hash functions, each equal entry is a coin that lands heads with probability $J$, and $\hat J$ is the fraction of heads: an unbiased estimate with standard deviation $\sqrt{J(1-J)/n}$, about 0.035 at $J = 0.8$ and $n = 128$. The signature has $n$ numbers whatever the document's length.

**The hash family, made exact.** A random permutation of all shingles is too expensive, so each $h_i$ is a Carter-Wegman universal hash $(a_i x + b_i) \bmod P$ over $x(s)$, the FNV-1a hash of the shingle's UTF-8 bytes reduced mod $P$. The parameters come from your PCG32 (`M06.3`): `rng = PCG32(seed)`, then for $i = 0, 1, \dots$ draw $a_i = 1 + $ `rng.below(P - 1)` and $b_i = $ `rng.below(P)`, interleaved. $a_i \ne 0$, because $a = 0$ maps every shingle to $b$ and that entry always agrees. Because $a_i, x < 2^{31}$, the product $a_i x + b_i < 2^{63}$ fits in `uint64`, so numpy computes every entry exactly and every implementation (Python, the Go port in `data.09`, yours) produces the same signature for the same seed. FNV-1a runs byte by byte, but every shingle runs the same steps, so you can loop over byte positions with all shingles at once: numpy's `uint64` multiply wraps modulo $2^{64}$, which is exactly FNV's arithmetic.

**LSH banding.** Comparing every pair of signatures is still quadratic. Cut each signature into $b$ bands of $r$ consecutive rows and hash each band, with its band index, into a bucket: two documents become **candidates** when they agree on all $r$ rows of at least one band. A band agrees with probability $s^r$, so it disagrees with probability $1 - s^r$, all $b$ bands disagree with probability $(1 - s^r)^b$, and

$$p(s) = 1 - (1 - s^r)^b.$$

That is an S-curve: near 0 for small $s$, near 1 for large $s$, steepest near $(1/b)^{1/r}$. With $b = 16, r = 8$ the steep part is around 0.7, below the 0.8 we want, so true near duplicates almost never slip through and the false candidates are filtered next.

**Confirm, then union-find.** A candidate pair is kept when $\hat J \ge t$. Near-duplication is not transitive ($x \sim y$ and $y \sim z$ do not imply $x \sim z$), but a cluster should hold the whole chain, so confirmed pairs are merged with a **union-find** (disjoint-set forest): `find(x)` follows parent pointers to the root, `union(x, y)` points one root at the other. Choose the root as the **smallest id** (string order) and the result is the same whatever order the pairs arrive in. `near_dedup` tags every document with its root (`meta["minhash_cluster"]`) and, with `drop=True`, keeps only the roots.

**Process-parallel, worker-invariant.** Signatures are independent per document, so `signatures(..., workers=N)` splits the texts into $N$ contiguous chunks and computes them on a `ProcessPoolExecutor`. Every chunk uses the same `(num_perm, seed)` and the chunks come back in order, so the array is identical for every $N$. Use the `spawn` start method (the same on every OS), and remember that `spawn` re-imports your `__main__`: the CLI's entry needs `if __name__ == "__main__":`.

**Decontamination.** A protected text is any eval or validation example you will report a number on. Its word $n$-grams ($n = 13$, the GPT-3 rule) form a set; a training document is contaminated when any of its word 13-grams is in that set. Thirteen words are long enough that a shared run is almost never a coincidence and short enough to catch a copied paragraph with edits around it. In `.jsonl` protected files every string value counts except the keys `case_id`, `id`, `tags`, and `scorer_args` (they are labels, not text the model is tested on). A protected text shorter than 13 words contributes no 13-gram: if your eval items are that short, lower $n$ for that suite.

## 3. Worked example by hand

**Shingles and Jaccard.** $A$ = "the cat sat on the mat today", $B$ = "the cat sat on the mat again", $k = 3$:

| | 3-shingles |
|---|---|
| $A$ | the cat sat · cat sat on · sat on the · on the mat · the mat today |
| $B$ | the cat sat · cat sat on · sat on the · on the mat · the mat again |

Four shared, six distinct in total: $J = 4/6 = 0.667$.

**MinHash with three toy hash functions.** Give the six distinct shingles ids $1$ to $6$ in the order above, with "the mat today" $= 5$ and "the mat again" $= 6$, so $A = \{1,2,3,4,5\}$ and $B = \{1,2,3,4,6\}$. Use $P = 7$:

| $x$ | 1 | 2 | 3 | 4 | 5 | 6 | min over $A$ | min over $B$ | equal? |
|---|---|---|---|---|---|---|---|---|---|
| $h_1 = (2x + 1) \bmod 7$ | 3 | 5 | 0 | 2 | 4 | 6 | 0 | 0 | yes |
| $h_2 = (3x + 2) \bmod 7$ | 5 | 1 | 4 | 0 | 3 | 6 | 0 | 0 | yes |
| $h_3 = (5x + 3) \bmod 7$ | 1 | 6 | 4 | 2 | 0 | 5 | 0 | 1 | no |

$\hat J = 2/3$. The third function's minimum over $A$ landed on shingle 5, which $B$ lacks, so the entries differ. (Three functions give a noisy estimate; 128 give a standard deviation near 0.04.)

**The S-curve.** $b = 16$, $r = 8$:

- threshold $(1/16)^{1/8} = 2^{-4/8} = 2^{-1/2} = 0.7071$;
- at $s = 0.8$: $0.8^8 = 0.16777$, $1 - 0.16777 = 0.83223$, $0.83223^{16} = 0.05295$, so $p = 0.9470$;
- at $s = 0.5$: $0.5^8 = 0.0039063$, $0.9960938^{16} = 0.93930$, so $p = 0.0607$.

A pair at 0.8 is a candidate 95 times in 100; a pair at 0.5 about 6 times in 100, and then its estimate (near 0.5) fails the 0.8 check.

**Union-find on a chain.** Confirmed pairs $(y, z)$ then $(x, y)$. `union(y, z)`: roots $y$, $z$, the smaller is $y$, so `parent[z] = y`. `union(x, y)`: roots $x$ and $y$, the smaller is $x$, so `parent[y] = x`. Now `find(z)` follows $z \to y \to x$: one cluster $\{x, y, z\}$ with root $x$, and `near_dedup` keeps only $x$.

**Decontamination.** A protected story contains "Tom found a shiny red shell by the river and carried it home to show his sister." A training document says "... and then TOM FOUND A SHINY, RED shell by the river and carried it home to show ..." Its words lower-cased are the same run, so the 13-gram "tom found a shiny red shell by the river and carried it home" is shared and the document is dropped. Had it copied only 12 of those words, it would stay.

These numbers are the first test, `test_hand_example`; the toy MinHash is the idea behind `test_signature_is_the_spec`, which checks the real formula with $P = 2^{31} - 1$, and the chain is `test_chain_is_one_cluster`.

## 4. The interface

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

`near_dedup` must see every document before it can cluster, so it reads its whole input (unlike the streaming stages of `data.02`); `decontaminate` streams. Both are stages: compose them after `exact_dedup`.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example` | unit | the section 3 shingles, $J = 4/6$, threshold 0.7071, $p(0.8) = 0.9470$, $p(0.5) = 0.0607$ | you and the tests agree on the definitions |
| `test_signature_is_the_spec` | unit | signatures recomputed from FNV-1a and interleaved PCG32 draws | every run, resumed or ported, clusters alike |
| `test_words_and_short_texts` | boundary | case and punctuation ignored; fewer than $k$ words is one shingle | short documents still match their copies |
| `test_empty_texts_are_nobodys_duplicate` | boundary | wordless texts: all-$P$ signature, $J = 0$, no cluster | empty pages do not collapse into one survivor |
| `test_estimator_is_unbiased` | statistical | mean of $\hat J$ over 40 fresh pairs at $J = 1/3$ within $\lvert z \rvert < 3.29$ | the estimate means what it says |
| `test_lsh_s_curve` | statistical | candidate rates at 13 similarities fit $p(s)$ (chi-square), 50% point within 0.05 of $(1/b)^{1/r}$ | the threshold you configure is the one you get |
| `test_bands_do_not_collide_with_each_other` | boundary | equal rows in different bands are not a candidate | false candidates cost time and, unconfirmed, data |
| `test_candidates_are_sorted_unique_pairs` | unit | one pair per key pair, smaller first, sorted; bad inserts raise | deterministic union order |
| `test_union_find_root_is_the_smallest_member` | property | every permutation of the unions gives the same groups and roots | the survivor is stable |
| `test_fixture_clusters_match_planted` | golden | planted clusters found exactly for 3 seeds; borderline pairs stay apart | the confirm step does its job |
| `test_chain_is_one_cluster` | unit | $x \sim y \sim z$, $x \not\sim z$: one cluster, keep $x$ | clusters are transitive |
| `test_invariant_to_worker_count` | property | 1 and 3 workers give identical signatures and clusters | MS-corpus compares hashes across worker counts |
| `test_invariant_to_input_order` | property | forward and reversed input keep the same documents | stages upstream may reorder |
| `test_near_dedup_tags_keeps_order_and_meta` | unit | `minhash_cluster` tags, `drop` flag, order and meta kept, repeated ids raise | `data.06` builds its column from the tag |
| `test_decontaminate_fixture` | golden | 13-word spans anywhere (upper-cased, punctuated) contaminate; 12 words, ids, and tags do not | benchmark numbers stay honest |
| `test_span_at_the_very_end` | boundary | the last window of a document is checked | an off-by-one hides the commonest copy |
| `test_protected_files` | boundary | JSON Lines string values minus id-like keys; plain files whole; bad lines raise | the right text is protected |
| `test_decontaminate_streams` | property | reads 5 of an endless stream without running ahead | the stage runs on any corpus size |
| `test_after_exact_dedup` | regression | after `data.03`, near dedup drops exactly the planted near duplicates | the two dedup counts add up in the manifest |

**Your tests (rung R4, properties).** Write them under `python/tests/data-04-minhash/` before the code (`ol tdd red data.04`): the hand example; the signature formula; for any 1 to $k-1$ words, one shingle; wordless texts never cluster; the estimate is unbiased over fresh pairs; equal values in different bands, or in only $r - 1$ rows of a band, are no candidate; union-find roots are the minimum of each group for any union sequence (Hypothesis); a chain is one cluster; candidates below the threshold are not merged; workers do not change signatures; tags, order, and meta survive `near_dedup`; a protected 13-gram anywhere (any case, any punctuation) contaminates and 12 words do not; id-like keys are not protected; `decontaminate` streams. `ol mutate data.04` grades them: 0.80 of the mutants, and every Pitfall below.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. returning no shingles for a text shorter than $k$ | short documents never match their copies | `test_words_and_short_texts` (mutant `s07`) |
| 2. indexing empty signatures | every wordless document lands in one bucket and all but one are dropped | `test_empty_texts_are_nobodys_duplicate` (mutant `s08`) |
| 3. a bucket key without the band index | equal rows in different bands make false candidates | `test_bands_do_not_collide_with_each_other` (mutant `s01`) |
| 4. union by size, or by argument order | the kept document depends on which pair came first | `test_union_find_root_is_the_smallest_member` (mutant `s09`) |
| 5. "drop it if it resembles a kept document" instead of union-find | a chain keeps both ends | `test_chain_is_one_cluster` (mutant `s12`) |
| 6. a seed per worker chunk, or chunks collected as they finish | the output hash changes with `--workers` | `test_invariant_to_worker_count` (mutants `s13`, `s14`) |
| 7. `range(len(w) - n)` instead of `range(len(w) - n + 1)` | a document ending with the protected span passes | `test_span_at_the_very_end` (mutant `s18`) |
| 8. comparing signature values as sets instead of position by position | biased estimates, missed clusters | `test_estimator_is_unbiased` (mutant `s11`) |
| 9. Python's `hash()` for shingles | salted per process: signatures change between runs and workers | `test_signature_is_the_spec` (mutant `s04`) |
| 10. merging every LSH candidate without the estimate check | borderline pairs (0.5 to 0.6) merge a quarter of the time | `test_fixture_clusters_match_planted` (mutant `s10`) |
| 11. substring matching on raw text for contamination | punctuation and case changes hide copies | `test_decontaminate_fixture` (mutant `s17`) |
| 12. treating `case_id`, `tags`, or `scorer_args` as protected text | labels, not eval text, start dropping documents | `test_protected_files` (mutant `s19`) |
| 13. `list(docs)` in `decontaminate` | the stage holds the whole corpus in memory | `test_decontaminate_streams` (mutant `s20`) |
| 14. no `if __name__ == "__main__":` in your CLI | with `--workers 4`, each spawned worker re-runs the CLI and the pool dies with "bootstrapping phase" | the MS-corpus step corpus-run-w4 (your entry point, not a unit) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `data.02` | `Doc` and the `Stage` shape these two stages share |
| Back | `data.03` | exact dedup runs first, so near dedup counts only near copies |
| Back | `M06.3` | `PCG32.below` draws the hash parameters; FNV-1a hashes the shingles |
| Back | `S-M06b` | Jaccard, the S-curve, and the threshold worked on paper |
| Forward | `data.05` | the PII scrub runs on the survivors and keeps `minhash_cluster` |
| Forward | `data.06` | `minhash_cluster` becomes the smallest row index of each cluster; `near_dropped` and `decontaminated` go into the manifest |
| Forward | `data.09` | `CorpusBuild` runs this stage as the `dedup_near` activity, with an output hash that must not depend on the worker count |

If you skip this module, `ol check data.05` stops with `data.05 needs data.04`: build it, or rerun with `--ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `signatures` + `LSH` | datatrove `MinhashDedupSignature`, `MinhashDedupBuckets` | signatures written to disk per shard, buckets sorted and merged across machines, 14 bands of 8 in FineWeb | datatrove `src/datatrove/pipeline/dedup/minhash.py` |
| `UnionFind` | datatrove `MinhashDedupCluster` | union-find over billions of pairs streamed from bucket files | same file |
| `clusters` confirm step | Dolma's dedup, text-dedup | suffix arrays for exact substring dedup, Bloom filters for paragraph dedup | `allenai/dolma`, `ChenghaoMou/text-dedup` |
| `decontaminate` | GPT-3 and Llama decontamination | 13-gram overlap with per-benchmark reports; Llama 2's token-level contamination scores | Brown et al. 2020 appendix C; Touvron et al. 2023 appendix A.6 |
