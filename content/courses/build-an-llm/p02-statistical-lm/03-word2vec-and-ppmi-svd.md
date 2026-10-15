<!-- ol:module L2.3 -->
# word2vec skip-gram with negative sampling, and the PPMI-SVD baseline

## Overview

| | |
|---|---|
| **Module** | `L2.3` · build · Python · Pass 3 · 4 to 5 h, plus your graded tests (rung R3) |
| **You build** | `python/tinyllm/lm/word2vec.py`: `sgns_loss` (loss and hand-derived gradients), `SkipGramNS` with `set_counts`, `noise_probs`, `keep_probs`, `pairs`, `negatives`, `step` (manual sparse SGD, no autograd), `fit`, `embeddings`; `cooccurrence` and `ppmi_svd_embeddings`; `word_vocab`, `analogy`, `spearman`, `word_similarity`; and your own tests in `python/tests/l2-3-word2vec/`, written first |
| **Contract** | [`course/contracts/py/tinyllm/lm/word2vec.pyi`](../../../../course/contracts/py/tinyllm/lm/word2vec.pyi) |
| **Tests** | `course/tests/L2.3/test_word2vec.py` (what they check: section 4); gold word-similarity pairs in `course/fixtures/L2.3/wordsim.json`, from the generator of the MS-L2 corpus; your tests are graded by mutation, threshold 0.70 plus one required fault, with a red-then-green journal |
| **Needs** | `M07.1` `AliasTable` (the negative sampler) · `M03.5` `svd` · `M03.6` `normalize`, `cosine_sim`, `topk_cosine` · `M11.4` `ppmi` · `M06.3` `PCG32` (or `--ref-deps`). Reading: `L2.2` (embeddings learned inside a language model), `S-M11b`, `S-M03b` |
| **Used by** | later `L6.7`: the zoo's word-similarity task and embedding utilities; `L3.6`: an optional pretrained embedding init (each joins the registry with its batch) |
| **Milestone** | `MS-L2` (the part's milestone; word2vec has no step of its own there, its call site is the zoo) |
| **Optional depth** | Mikolov et al., "Distributed Representations of Words and Phrases and their Compositionality" (NeurIPS 2013); Goldberg and Levy, "word2vec Explained" (2014, arXiv:1402.3722); Levy and Goldberg, "Neural Word Embedding as Implicit Matrix Factorization" (NeurIPS 2014); Levy, Goldberg, and Dagan, "Improving Distributional Similarity with Lessons Learned from Word Embeddings" (TACL 2015); Jurafsky and Martin, *Speech and Language Processing*, 3rd ed., ch. 6 |

## Key Takeaways

- Skip-gram with negative sampling turns "which words appear near which" into a logistic regression per (center, context) pair against a few random negatives; its gradient is three lines you derive and check by finite differences (`test_hand_example_sgns_loss`, `test_sgns_gradcheck`).
- The two distributions that make it work are exact formulas: negatives from counts to the power 0.75, and subsampling that keeps a token with probability $\sqrt{t / f}$, applied before the window is cut (`test_noise_and_keep_probabilities`, `test_subsampling_drops_before_windowing`).
- A sparse update must add every gradient a row receives: a word that appears twice in a batch is updated twice (`test_step_is_a_sparse_sum_of_gradients`).
- PPMI plus SVD factorizes the same co-occurrences explicitly and is a strong baseline; your Jacobi SVD agrees with LAPACK on the cosines (`test_ppmi_svd_matches_numpy`).
- Embeddings are judged by how well their cosine similarities rank gold word pairs (Spearman's rho), the zoo's word2vec metric (`test_sgns_learns_word_classes`).

## How to work this chapter

```bash
ol start L2.3              # stubs word2vec.py; prints your test path and rung (R3)
ol tests L2.3              # the course tests
# write ONE test in python/tests/l2-3-word2vec/ (start with section 4's), then:
ol tdd red L2.3            # must FAIL against your current code: records the red
# make it pass, then:
ol tdd green L2.3          # must PASS with the same test files: records the green
# repeat for each test; then:
ol check L2.3              # course tests, the red-then-green journal, the mutation grade
ol mutate L2.3             # the full grade, cached by your test files' hash
```

---

## 1. Why now

Your NPLM (`L2.2`) learned an embedding table as a side effect of predicting the next byte, and the table is where its generalization lives: windows that share similar vectors share predictions. word2vec isolates that table. It throws away the language model and keeps only the question "which words appear near which?", answered at a cost so low that it trained on billions of words in 2013. The result is the classic demonstration that meaning can be read off co-occurrence: on the MS-L2 stories, animals end up near animals and foods near foods, and the vectors rank 611 gold word pairs with a Spearman correlation around 0.73. The PPMI-SVD baseline gets about 0.78 on the same pairs with no training at all, which is the second lesson: the network was implicitly factorizing a matrix you can build and factor directly with your `M11.4` and `M03.5` code. Later, the zoo (`L6.7`) scores every embedding table this way, and the recurrent LM (`L3.6`) can start from these vectors.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $V$ | vocabulary size; words are ids $0 \dots V-1$ | `int` |
| $d$ | embedding size (`dim`) | `int` |
| $v_w$ | input (center) vector of word $w$, row $w$ of `W_in` | `float64[d]` |
| $u_w$ | output (context) vector of word $w$, row $w$ of `W_out` | `float64[d]` |
| $(c, o)$ | a center word and one context word near it | ids |
| $k_1 \dots k_K$ | $K$ negatives (`n_neg`) drawn from $P_n$ | ids |
| $\sigma(x) = 1 / (1 + e^{-x})$ | the logistic function; $1 - \sigma(x) = \sigma(-x)$ | |
| $s = u_o \cdot v_c$, $t_j = u_{k_j} \cdot v_c$ | the positive and negative scores | scalars |
| $\#(w)$, $f(w)$ | count of $w$ in the corpus, and its frequency $\#(w) / \sum \#$ | |
| $P_n(w) \propto \#(w)^{3/4}$ | the noise distribution | `float64[V]` |
| $t$ | the subsampling threshold (`subsample_t`) | `float` |
| $L$ | the window size: each center gets $r \in \{1 \dots L\}$ | `int` |
| $\eta$ | the learning rate | `float` |
| $M$ | the PPMI matrix (`M11.4`) | `float64[V, V]` |
| $\rho$ | Spearman's rank correlation | `float` |

### 2.1 Skip-gram and why the softmax is too expensive

Skip-gram predicts the words around each word: for every position, every word within the window is a (center, context) pair, and the model wants $P(o \mid c) = \mathrm{softmax}_o(u \cdot v_c)$ to be high. That softmax runs over all $V$ output vectors for every pair, $O(V d)$ work per pair; with $V$ in the hundreds of thousands it is the whole cost.

### 2.2 Negative sampling, the noise distribution, and subsampling

Negative sampling (Mikolov et al. 2013) replaces the softmax by a binary question: is $(c, o)$ a pair from the corpus, or a word drawn at random? The real context should score high, $K$ random **negatives** should score low:

$$\ell(c, o, k_{1..K}) = -\log \sigma(u_o \cdot v_c) - \sum_{j=1}^{K} \log \sigma(-u_{k_j} \cdot v_c).$$

Each pair now costs $O((K + 1) d)$. The negatives come from $P_n(w) \propto \#(w)^{3/4}$. With power 1 the frequent words ("the", ".") would be nearly all the negatives; the 0.75 power flattens the distribution and lifts rare words, which Mikolov et al. found clearly better. Drawing from it is `M07.1`'s alias table, built once in `set_counts`: one uniform per draw, $O(1)$ each.

**Subsampling.** Frequent words carry little information per occurrence ("the" near "cat" says almost nothing) and dominate the pairs. Each token of $w$ is kept with probability

$$p_{\text{keep}}(w) = \min\left(1, \sqrt{t / f(w)}\right),$$

where $f(w)$ is the **frequency**, not the count: with a count the threshold $t$ would mean something different on every corpus size. Subsampling happens before the windows are cut, so removing a "the" makes its neighbours adjacent: it widens the effective window around frequent words, which is part of why it helps.

**Dynamic window.** Each center draws $r = 1 + \text{below}(L)$ and pairs with the kept tokens within distance $r$. Near words are in more windows than far ones: a distance-1 neighbour always pairs, a distance-$L$ neighbour with probability $1/L$. This is word2vec.c's way of weighting closeness.

### 2.3 The gradient, by hand, and sparse SGD

With $\frac{d}{dx}[-\log \sigma(x)] = \sigma(x) - 1$ and $\frac{d}{dx}[-\log \sigma(-x)] = \sigma(x)$:

$$\frac{\partial \ell}{\partial v_c} = (\sigma(s) - 1)\, u_o + \sum_j \sigma(t_j)\, u_{k_j}, \qquad \frac{\partial \ell}{\partial u_o} = (\sigma(s) - 1)\, v_c, \qquad \frac{\partial \ell}{\partial u_{k_j}} = \sigma(t_j)\, v_c.$$

That is all of backpropagation for this model, so word2vec needs no autograd. A step on a batch of $B$ pairs computes these from the current rows and subtracts $\eta$ times the gradient from exactly the rows used: $B$ input rows and $B (K + 1)$ output rows, never the whole $V \times d$ tables. A row used several times (a frequent word appears in many pairs of one batch) receives the **sum** of its gradients, as if the pairs had been processed one after another from the same starting point. In numpy that is `np.add.at(W, idx, -lr * g)`; `W[idx] -= lr * g` keeps only one of the repeated writes.

`-log σ(x)` must be computed stably. Literally, $\sigma(-1000) = 1/(1 + e^{1000})$ overflows to $1 / \infty = 0$ and $-\log 0 = \infty$. As $\mathrm{softplus}(-x) = \log(1 + e^{-|x|}) + \max(-x, 0)$ it is exact for every finite $x$, and $\sigma$ itself is computed with $e^{-|x|}$ only.

**Initialization and the schedule.** word2vec.c starts the input vectors uniform in $[-0.5/d, 0.5/d)$ and the output vectors at zero (so the first step moves only the output vectors). `fit` counts the corpus, then for each epoch draws fresh pairs and steps through them in order, with the learning rate decayed linearly over the run to $10^{-4}$ of its start, as word2vec does. Every random choice (the initial vectors, subsampling, windows, negatives) comes from the one PCG32 given to the model, in call order, so a seed fixes the whole run. `embeddings()` returns the input table.

### 2.4 PPMI-SVD

Levy and Goldberg (2014) showed that SGNS at its optimum sets $u_o \cdot v_c = \mathrm{PMI}(c, o) - \log K$: it implicitly factorizes a shifted PMI matrix. So build the matrix and factor it. `cooccurrence(ids, V, L)` counts every ordered pair of positions at distance at most $L$, in both directions (unweighted, no subsampling). `M11.4`'s `ppmi` turns the counts into $\max(\mathrm{PMI}, 0)$ with context smoothing 0.75 (the same 0.75 as the noise distribution, for the same reason). `M03.5`'s `svd` gives $M = U S V^T$, and the embeddings are $U_d \sqrt{S_d}$: the symmetric split of the singular values between words and contexts, which Levy, Goldberg, and Dagan (2015) found better than $U_d S_d$ for similarity. Each column is defined only up to sign, but cosines between rows do not depend on column signs, so two correct SVDs give the same similarities.

### 2.5 Evaluating embeddings

**Word similarity.** A gold list of word pairs with scores (here 3: same class, 1: same part of speech, 0: otherwise). For each covered pair take the cosine similarity of the two vectors, then Spearman's $\rho$ between the cosines and the gold scores: the Pearson correlation of their **ranks**. Tied values share the average of the ranks they occupy, which matters because the gold scores have only three levels. Pairs with a word outside the vocabulary are skipped.

**Analogies.** "a is to b as c is to ?" by 3CosAdd: the word whose vector is most cosine-similar to $\hat b - \hat a + \hat c$ (hats: unit vectors). The query words themselves are usually the nearest vectors to that point and are excluded.

**Vocabulary.** `word_vocab` assigns ids by count, most frequent first, ties broken by the word, so a text gives the same ids on every machine.

## 3. Worked example by hand

**One SGNS pair** in $d = 2$: $v_c = (0.5, -1)$, $u_o = (2, 1)$, one negative $u_k = (0, 1)$.

- Scores: $s = 2 \cdot 0.5 + 1 \cdot (-1) = 0$ and $t = 0 \cdot 0.5 + 1 \cdot (-1) = -1$.
- Loss: $-\log \sigma(0) - \log \sigma(1) = \ln 2 + \ln(1 + e^{-1}) = 0.693147 + 0.313262 = 1.006409$.
- Weights: $\sigma(s) - 1 = -0.5$ and $\sigma(t) = \sigma(-1) = 0.268941$.
- $\partial \ell / \partial v_c = -0.5 \cdot (2, 1) + 0.268941 \cdot (0, 1) = (-1, -0.231059)$.
- $\partial \ell / \partial u_o = -0.5 \cdot (0.5, -1) = (-0.25, 0.5)$; $\partial \ell / \partial u_k = 0.268941 \cdot (0.5, -1) = (0.134471, -0.268941)$.

These are `test_hand_example_sgns_loss`. One step with $\eta = 0.1$ gives $v_c = (0.6, -0.976894)$, $u_o = (2.025, 0.95)$, $u_k = (-0.013447, 1.026894)$: the positive score rises to $0.286951$, the negative falls to $-1.011235$, and the loss drops to $0.870182$.

**Pairs.** Ids 0 1 2 3, no subsampling, window $L = 2$, and the generator's `below(2)` returning 1, 0, 0, 1, so $r = 2, 1, 1, 2$. Center 0 pairs with 1 and 2; center 1 with 0 and 2; center 2 with 1 and 3; center 3 with 1 and 2 (`test_pairs_hand_example`).

**Co-occurrence.** Ids 0 1 2 0 with $L = 1$: the adjacent pairs (0, 1), (1, 2), (2, 0) and their mirrors give

$$\begin{pmatrix} 0 & 1 & 1 \\ 1 & 0 & 1 \\ 1 & 1 & 0 \end{pmatrix};$$

with $L = 2$ the pairs (0, 2) at positions 0 and 2 and (1, 0) at positions 1 and 3 are added in both directions (`test_cooccurrence_hand_example`).

**Analogy.** With man $(1, 0, 0)$, woman $(0, 1, 0)$, king $(1, 0, 1)$, queen $(0.5, 1, 0.5)$, apple $(0, 0, -1)$: $\hat{\text{king}} - \hat{\text{man}} + \hat{\text{woman}} = (-0.293, 1, 0.707)$, whose cosines are woman 0.794, queen 0.783, king 0.233. Woman, a query word, is nearest; excluding the three query words, queen wins, then apple (`test_analogy_hand_example`).

## 4. The interface

```python
def word_vocab(words, min_count=1) -> tuple[list[str], NDArray]: ...
def sgns_loss(v, u_pos, u_neg) -> tuple[NDArray, NDArray, NDArray, NDArray]: ...  # loss[B], g_v, g_pos, g_neg

class SkipGramNS:
    W_in: NDArray; W_out: NDArray                                # float64 [V, d]
    def __init__(self, vocab, dim, n_neg=5, window=5, subsample_t=1e-5, rng=None): ...
    def set_counts(self, counts) -> None: ...
    def noise_probs(self) -> NDArray: ...                        # counts^0.75, normalized
    def keep_probs(self) -> NDArray: ...                         # min(1, sqrt(t / f))
    def pairs(self, ids) -> tuple[NDArray, NDArray]: ...         # subsample, then dynamic windows
    def negatives(self, n: int) -> NDArray: ...
    def step(self, centers, contexts, lr: float) -> float: ...   # mean loss before the update
    def fit(self, ids, epochs, batch_size, lr) -> list[float]: ...
    def embeddings(self) -> NDArray: ...

def cooccurrence(ids, vocab, window) -> NDArray: ...
def ppmi_svd_embeddings(cooc, dim) -> NDArray: ...               # U_d sqrt(S_d) of ppmi(cooc)
def analogy(emb, vocab, a, b, c, k=1) -> list[str]: ...
def spearman(x, y) -> float: ...
def word_similarity(emb, vocab, pairs) -> float: ...
```

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_sgns_loss` | unit, smoke | section 3's loss and three gradients | you and the tests agree on the loss before any code |
| `test_sgns_gradcheck` | gradcheck | every gradient entry against the frozen central differences, 5 pairs, 3 negatives | the update is only as good as the hand-derived gradient |
| `test_sgns_loss_is_stable_at_large_scores` | boundary | scores of $\pm 1000$ give finite, exact losses and gradients | trained vectors have large dot products |
| `test_noise_and_keep_probabilities` | unit | $P_n \propto \#^{0.75}$, $p_{\text{keep}} = \sqrt{t / f}$ with $f$ a frequency, unseen words | the two distributions word2vec's quality rests on |
| `test_negative_sampler_chi_square` | statistical | 20 000 negatives fit $P_n$ (p > 1e-3), the zero-count word never drawn, one uniform per draw | negatives are drawn from the right distribution |
| `test_pairs_hand_example` | unit | section 3's dynamic windows, pair order, draw counts | reproducible pairs from a seed |
| `test_subsampling_drops_before_windowing` | unit | a dropped frequent word makes its neighbours adjacent | the effective window of the real algorithm |
| `test_init_and_embeddings` | unit | `W_in` from one uniform per entry, `W_out` zero, `embeddings()` a copy of `W_in` | the zoo scores the input table |
| `test_step_is_a_sparse_sum_of_gradients` | unit | a batch with repeated rows against a one-pair-at-a-time update; untouched rows unchanged; mean loss before the update | the update every step relies on |
| `test_fit_is_pairs_then_decayed_steps` | differential | `fit` against the recipe rebuilt step by step with the linear decay | one seed, one run |
| `test_sgns_learns_word_classes` | learning | five epochs on the MS-L2 words reach the reference's Spearman $\rho$ (mean - 3 sd over 5 seeds) | the embeddings mean something |
| `test_cooccurrence_hand_example` | unit | section 3's symmetric counts for windows 1 and 2 | the PPMI input |
| `test_ppmi_svd_matches_numpy` | differential | cosines of every gold pair against numpy's PMI and LAPACK SVD; $\rho > 0.7$ | your `M11.4` and `M03.5` code on a real matrix |
| `test_analogy_hand_example` | unit | 3CosAdd excludes the query words; top 2 | the classic embedding check |
| `test_spearman_with_ties` | unit | average ranks for ties, $\rho = \pm 1$ for monotone data, constant input rejected | gold scores have three levels |
| `test_word_similarity_covers_known_words_only` | unit | unknown words skipped, fewer than two covered pairs rejected | the zoo scores any vocabulary |
| `test_word_vocab_order` | unit | ids by count, ties by word; `min_count` drops rare words | the same ids on every machine |
| `test_input_validation` | boundary | bad sizes, counts, ids, calls before `set_counts`, mismatched shapes | caller bugs fail at the call |

### Your graded tests (rung R3)

Rung R3 gives you the interface (above) and one test; you write the rest **before** the code that makes each pass. The given test:

```python
# python/tests/l2-3-word2vec/test_word2vec.py
import math
import numpy as np
from tinyllm.lm.word2vec import sgns_loss

def test_hand_example_loss_and_gradients():
    """v = (0.5, -1), u_o = (2, 1), u_k = (0, 1): s = 0, t = -1."""
    s1 = 1 / (1 + math.e)
    loss, gv, gp, gn = sgns_loss([[0.5, -1.0]], [[2.0, 1.0]], [[[0.0, 1.0]]])
    np.testing.assert_allclose(loss, [math.log(2) + math.log1p(math.exp(-1))], rtol=1e-12)
    np.testing.assert_allclose(gv, [[-1.0, -0.5 + s1]], rtol=1e-12)
```

Then, one at a time, red then green: finite differences on a few gradient entries; finite losses at scores of $\pm 1000$; the noise and keep probabilities on counts you choose; the pairs of a short sequence with a scripted generator (a small class whose `uniform()` and `below()` return values you list); a batch where one center appears twice; the initial tables; `fit` against the recipe rebuilt by hand; co-occurrence and PPMI-SVD against numpy on a 4-word matrix; the analogy example; Spearman with ties; the vocabulary order. Import only contract names. `ol check L2.3` requires, for every test file, a `ol tdd red` record before its last `ol tdd green`, a mutation score of at least 0.70, and the required fault killed: it is pitfall 1.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. the sparse update written `W[idx] -= lr * g` | a word repeated in a batch keeps one of its gradients: frequent words learn slower than they should | `test_step_is_a_sparse_sum_of_gradients` (mutant `s01`) |
| 2. negatives from the plain unigram distribution | "the" and "." are most of the negatives; rare words are never pushed apart | `test_noise_and_keep_probabilities`, `test_negative_sampler_chi_square` (mutant `s02`) |
| 3. subsampling from the raw count instead of the frequency | almost every token is dropped (or kept), whatever the corpus size | `test_subsampling_drops_before_windowing` (mutant `s03`) |
| 4. $-\log \sigma(x)$ computed literally | an infinite loss and NaN vectors once scores grow | `test_sgns_loss_is_stable_at_large_scores` (mutant `s04`) |
| 5. the center gradient without the negatives' term | centers are only pulled toward contexts, never pushed from noise: everything collapses together | `test_hand_example_sgns_loss`, `test_sgns_gradcheck` (mutant `s05`) |
| the positive weight $\sigma(s)$ instead of $\sigma(s) - 1$ | the context is pushed away instead of pulled closer | `test_sgns_gradcheck`, `test_step_is_a_sparse_sum_of_gradients` (mutant `s06`) |
| a fixed window ($r = L$ always) | far words count as much as near ones | `test_pairs_hand_example` (mutant `s07`) |
| subsampling that keeps every token | frequent words dominate the pairs | `test_subsampling_drops_before_windowing` (mutant `s08`) |
| PPMI-SVD embeddings $U_d S_d$ instead of $U_d \sqrt{S_d}$ | the top directions dominate the cosines | `test_ppmi_svd_matches_numpy` (mutant `s09`) |
| analogies that keep the query words | "king - man + woman" answers "woman" | `test_analogy_hand_example` (mutant `s10`) |
| ties ranked by position instead of averaged | Spearman depends on the order of the pair list | `test_spearman_with_ties` (mutant `s11`) |
| `fit` without the learning-rate decay | the last epochs keep jumping around instead of settling | `test_fit_is_pairs_then_decayed_steps` (mutant `s12`) |
| `embeddings()` returning the output table | the zoo scores vectors that start at zero | `test_init_and_embeddings` (mutant `s13`) |
| co-occurrence counted in one direction | an asymmetric matrix and wrong marginals for PMI | `test_cooccurrence_hand_example` (mutant `s14`) |
| vocabulary ties broken by first appearance | ids change with the order of the text | `test_word_vocab_order` (mutant `s15`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M07.1` | `AliasTable` draws the negatives in $O(1)$ each |
| Back | `M03.5` | `svd` factors the PPMI matrix |
| Back | `M03.6` | `normalize`, `cosine_sim`, `topk_cosine` for similarity and analogies |
| Back | `M11.4` | `ppmi` with context smoothing builds the matrix SGNS factorizes implicitly |
| Back | `M06.3` | `PCG32`: one generator for initialization, subsampling, windows, and negatives |
| Back | `L2.2` | the embedding table a language model learns as a side effect (reading) |
| Forward | `L6.7` | the zoo trains SGNS and PPMI-SVD and reports `word_similarity` next to every other family |
| Forward | `L3.6` | the recurrent LM can initialize its embedding table from `embeddings()` (optional) |

If you skip this module, `L6.7` stops with `BLOCKED ... needs L2.3` once it lands: build it, or pass `--ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| batched sparse SGD in numpy | word2vec.c and gensim | lock-free multi-threaded SGD (Hogwild) over the shared tables, a precomputed sigmoid table, a 100M-entry unigram table for negatives | `word2vec.c` (`TrainModelThread`), `gensim/models/word2vec_inner.pyx` |
| word ids | fastText (Bojanowski et al. 2017) | vectors for character n-grams, so unseen and misspelled words get embeddings | `facebookresearch/fastText`, `src/fasttext.cc` |
| PPMI-SVD | GloVe (Pennington et al. 2014) | a weighted least-squares fit to log co-occurrence counts | `stanfordnlp/GloVe`, `src/glove.c` |
| static vectors | contextual embeddings (ELMo, BERT; `L3.5`, `L6.2`) | one vector per occurrence, not per word type | `L6.2` in this course |
| the gold pairs of a synthetic corpus | WordSim-353, SimLex-999, MEN | human similarity judgments; SimLex separates similarity from relatedness | Hill, Reichart, Korhonen (2015) |
