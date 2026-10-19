# Unigram LM tokenizer (EM, Viterbi, subword sampling)

## Overview

| | |
|---|---|
| **Module** | `L1.4` · build · Python · Pass 3 · 3 to 4 h |
| **Solution** | [`unigram.py`](../../../../course/solutions/python/tinyllm/tok/unigram.py): `metaspace` and `UnigramTokenizer` (`viterbi`, `encode`, `sample_encode`, `log_likelihood`, `em_step`, `train`, `decode`, `save`, `load`, `from_hf_json`) · model tests: [`test_unigram.py`](../../../../course/solutions/learner-tests/L1.4/python/tests/l1-4-unigram/test_unigram.py) |
| **Contract** | [`course/contracts/py/tinyllm/tok/unigram.pyi`](../../../../course/contracts/py/tinyllm/tok/unigram.pyi) · format: [`tokenizer.md`](../../../../course/contracts/formats/tokenizer.md) (Unigram in the `tokenizer.json` subset) |
| **Needs** | `L1.1` the protocol and `check_ids` · `M06.2` `Trie.prefixes` builds the lattice · `M07.2` `mle` scores the seed pieces · reading: `M11.1` (log-likelihood and cross-entropy), `M06.3` (the PCG32 that `sample_encode` draws from), `S-M07b` q8 and q9 (one EM step and the MLE by hand) |
| **Used by** | `L1.6` measures a Unigram tokenizer loaded with `from_hf_json` · later: `C1` compares BPE and Unigram at 4096 tokens on the same sample · part of the Pass 3 tokenizer layer, with `L1.1` to `L1.3` and `L1.6` |
| **Optional depth** | Kudo, *Subword Regularization* (2018), sections 3 and 4; Kudo and Richardson, *SentencePiece* (2018); Dempster, Laird, and Rubin, *Maximum Likelihood from Incomplete Data via the EM Algorithm* (1977) |

## Key Takeaways

- A Unigram tokenizer is a probability for each piece; a segmentation's probability is the product of its pieces', and encoding picks the most probable segmentation by Viterbi over a lattice.
- The likelihood of a word sums over all its segmentations, so training is EM: expected piece counts by forward-backward, then count over total, and the log-likelihood never goes down.
- Unlike BPE, Unigram can sample segmentations with probability proportional to $P(s)^\alpha$, which `C1` uses as subword regularization.
- Every rule that decides ties and unknowns is part of the contract: the earliest-starting last piece wins a Viterbi tie, an unknown character scores the minimum minus 10, and adjacent unknowns fuse.
- With those rules exact, the tokenizer loads Hugging Face and sentencepiece models unchanged and reproduces their ids.

## How to read this chapter

Sections 1 to 3 build the Unigram model, Viterbi, EM, and sampling from first principles and work them by hand on one small vocabulary. Section 4 fixes the interface and lists what a correct solution must handle. Section 5 walks through `unigram.py` and explains why each choice is the right one. Try to predict each step of the solution before reading it.

`S-M07b` q8 and q9 are good preparation: they are one EM step and the count MLE on paper, the two halves of `em_step`.

---

## 1. Why now

`L1.2` built byte-level BPE, the tokenizer of GPT-2 and SmolLM2, and it has one answer for every word: the merges decide. The capstone (`C1`) has to choose a tokenizer for its own model, and the decision is an ablation: BPE against the other widely used family at the same vocabulary size, on the same sample, compared by the metrics of `L1.6` and by bits per byte after a short training run. That family is the Unigram language model of SentencePiece (T5, ALBERT, XLNet, and many multilingual models). It is a probabilistic model, so it can do what BPE cannot: score a segmentation, and sample a different segmentation of the same text on every pass, a regularizer for small models. Without this module the ablation has only one arm.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $\mathcal{V}$ | the pieces (strings); id 0 is `<unk>` | `list[str]` |
| $\theta_v = \ln P(v)$ | the log-probability (score) of piece $v$ | `float`, $\le 0$ |
| $w$ | one word after the Metaspace pre-tokenizer, $n$ code points | `str` |
| $s = (v_1, \dots, v_k)$ | a segmentation: pieces whose concatenation is $w$ | `list[str]` |
| $S(w)$ | every segmentation of $w$ into pieces of $\mathcal{V}$ | set |
| $P(s) = \prod_i P(v_i)$ | the probability of a segmentation | `float` |
| $Z(w) = \sum_{s \in S(w)} P(s)$ | the likelihood of the word | `float` |
| $\alpha(j)$, $\beta(i)$ | forward and backward log-sums over the lattice | `float[n + 1]` |
| $f(w)$ | how often word $w$ occurs in the training texts | `int` |
| $c_v$ | the expected count of piece $v$ (E-step) | `float` |
| $a \ge 0$ | the sampling smoothing exponent (the `alpha` argument) | `float` |

**Metaspace.** SentencePiece does not split on spaces; it turns every U+0020 into `▁` (U+2581), prepends one `▁` unless the text already starts with it, and splits before every `▁`. So `"a  b"` is `["▁a", "▁", "▁b"]`: a double space leaves a lone `▁`, and pieces never cross a word start. Decoding joins pieces, turns `▁` back into spaces, and drops the first piece's leading space.

**The lattice.** For a word $w$ with positions $0..n$, put an edge from $i$ to $j$ for every piece $w[i:j]$ in $\mathcal{V}$, weighted $\theta$. The `M06.2` trie gives every piece starting at $i$ in one walk (`Trie.prefixes`). A segmentation is a path from 0 to $n$, and its log-probability is the sum of its edge weights. Where no one-character piece starts at $i$, add an **unknown** edge $i \to i+1$ with id `<unk>` and score $\min_v \theta_v - 10$: low enough that a known path always wins when one exists, finite so the word still encodes. Adjacent unknowns **fuse** into one `<unk>` in the output.

**Viterbi.** $\mathrm{best}(0) = 0$ and $\mathrm{best}(j) = \max_{(i, v) \to j} \mathrm{best}(i) + \theta_v$, keeping a back pointer. Scanning the edges into $j$ by start position ascending and replacing only on a strictly better score means a tie goes to the edge that starts earliest, the longest last piece: that is sentencepiece's and Hugging Face's rule, and the ids match theirs only with it.

**Likelihood and EM.** The word's likelihood sums over paths, so replace max by log-sum-exp: $\alpha(0) = 0$, $\alpha(j) = \operatorname{logsumexp}_{(i, v) \to j}(\alpha(i) + \theta_v)$, and $\ln Z(w) = \alpha(n)$. Training wants $\theta$ that maximizes $\sum_w f(w) \ln Z(w)$, but which segmentation produced each word is hidden. EM alternates:

- **E-step**: the posterior probability that edge $(i, v, j)$ is used is $\exp(\alpha(i) + \theta_v + \beta(j) - \alpha(n))$, with the backward sums $\beta(n) = 0$ and $\beta(i) = \operatorname{logsumexp}_{(i, v) \to j}(\theta_v + \beta(j))$. Summing those posteriors over all words, each times $f(w)$, gives the expected count $c_v$.
- **M-step**: $P(v) = c_v / \sum_u c_u$, the maximum-likelihood estimate of `M07.2` applied to fractional counts (`S-M07b` q9 proves it is the maximum).

Each step can only raise $\sum_w f(w) \ln Z(w)$ (Dempster, Laird, Rubin), up to rounding. Hard EM, which counts only the Viterbi path, is a different objective and drifts.

**Training** (`train`, the contract fixes every constant):

1. **Seed**: every character, plus the `seed_factor * vocab_size` substrings of 2 to 16 code points inside one word with the largest frequency times length, scored with `mle` of their raw counts.
2. **EM**: `em_iters` steps; a piece expected fewer than 0.5 times is dropped, except characters, which keep a count of at least 0.5 so every training text still encodes.
3. **Prune**: if more than `vocab_size - 1` pieces remain, keep the characters and the non-character pieces with the largest **removal loss**, until $\max(\text{vocab\_size} - 1, \lfloor n \cdot \text{shrink} \rfloor)$ pieces remain. A piece's removal loss is its Viterbi frequency times its score minus the score of the best segmentation of its own text without it: how much likelihood the corpus loses if it has to spell that piece another way. Then back to 2.

The result is `<unk>` at id 0 with score 0, then the pieces by score, highest first (ties by string).

**Sampling.** `sample_encode(text, a, rng)` draws a segmentation with probability $P(s)^a / \sum_{s'} P(s')^a$. Scale every edge weight by $a$, run the forward sums, then walk **backward** from $n$: at position $j$, the edge $(i, v)$ into $j$ is chosen with probability $\exp(\alpha(i) + a\theta_v - \alpha(j))$, using one `rng.uniform()` per chosen piece and the inverse CDF over the candidates ordered by start position. Forgetting $\alpha(i)$ samples each piece by its own score, which is not the posterior. $a = 1$ samples the model's posterior, $a \to 0$ flattens toward uniform over segmentations ($a = 0$ is exactly uniform), and large $a$ approaches Viterbi.

## 3. Worked example by hand

Eight pieces whose probabilities sum to 1, plus `<unk>` at id 0:

| id | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| piece | `▁` | a | b | c | `▁a` | ab | bc | `▁ab` |
| $P$ | 0.1 | 0.1 | 0.1 | 0.1 | 0.2 | 0.15 | 0.15 | 0.1 |

The text `"abc"` is the one Metaspace word `"▁abc"`. Its six segmentations:

| Segmentation | $P(s)$ |
|---|---|
| `▁` a b c | $0.1^4 = 0.0001$ |
| `▁a` b c | $0.2 \cdot 0.1 \cdot 0.1 = 0.002$ |
| `▁` ab c | $0.1 \cdot 0.15 \cdot 0.1 = 0.0015$ |
| `▁` a bc | $0.1 \cdot 0.1 \cdot 0.15 = 0.0015$ |
| `▁a` bc | $0.2 \cdot 0.15 = 0.03$ |
| `▁ab` c | $0.1 \cdot 0.1 = 0.01$ |

**Viterbi** picks `▁a` + `bc` ($0.03$): `encode("abc")` is `[5, 7]`, and `decode([5, 7])` is `"abc"`.

**Likelihood.** $Z = 0.0001 + 0.002 + 0.0015 + 0.0015 + 0.03 + 0.01 = 0.0451$, so `log_likelihood(["abc"])` is $\ln 0.0451$, not the Viterbi path's $\ln 0.03$.

**One EM step.** Each piece's expected count is the sum of $P(s)/Z$ over the segmentations that use it. In units of $10^{-3}/Z$:

| piece | `▁` | a | b | c | `▁a` | ab | bc | `▁ab` | total |
|---|---|---|---|---|---|---|---|---|---|
| $Z \cdot c_v \cdot 10^3$ | 0.1 + 1.5 + 1.5 = 3.1 | 0.1 + 1.5 = 1.6 | 0.1 + 2 = 2.1 | 0.1 + 2 + 1.5 + 10 = 13.6 | 2 + 30 = 32.0 | 1.5 | 1.5 + 30 = 31.5 | 10.0 | 95.4 |

The M-step divides by the total, so the new $P(\texttt{▁a}) = 32.0/95.4 \approx 0.335$ and $P(\text{c}) = 13.6/95.4 \approx 0.143$: the pieces of the likely segmentation gain, `▁` and `a` alone lose.

**Metaspace.** `"a  b"` is `["▁a", "▁", "▁b"]`, `" x"` and `"▁x"` are both `["▁x"]`, and `""` is `[]`.

These are the first cases a correct solution must reproduce (section 4).

## 4. The interface

```python
# python/tinyllm/tok/unigram.py
def metaspace(text: str) -> list[str]

class UnigramTokenizer(Tokenizer):
    pieces: list[tuple[str, float]]       # id -> (piece, log-probability)
    min_score: float
    def __init__(self, pieces: Sequence[tuple[str, float]], unk_id: int = 0, specials: Sequence[str] = ()) -> None
    @classmethod
    def train(cls, texts, vocab_size: int, seed_factor: int = 10, em_iters: int = 2, shrink: float = 0.75) -> "UnigramTokenizer"
    def viterbi(self, word: str) -> list[int]
    def sample_encode(self, text: str, alpha: float, rng: PCG32) -> list[int]
    def log_likelihood(self, texts: Iterable[str]) -> float
    def em_step(self, texts: Iterable[str]) -> "UnigramTokenizer"
    @classmethod
    def from_hf_json(cls, tokenizer_json: str) -> "UnigramTokenizer"
```

`special_ids` are the `<unk>` piece plus `specials`; typed in the raw text, they match as written (leftmost longest) and are one id each. `sample_encode` takes any object with `uniform() -> float`: the `M06.3` PCG32, or a stub that replays fixed numbers. `save` writes a `tokenizer.json` with the `Unigram` model and the Metaspace pre-tokenizer and decoder (`prepend_scheme` `"always"`); `from_hf_json` refuses byte fallback, a normalizer, or another Metaspace scheme.

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the hand-worked Viterbi | unit | `"abc"` is `▁a bc`, ids `[5, 7]`, and decodes back | everyone agrees on the definition |
| the hand-worked likelihood | unit | $\ln 0.0451$ for one word, twice that for two | the EM objective sums over segmentations |
| the hand-worked EM step | unit | the section 3 table, each probability to 1e-12 | the M-step divides by the total |
| the hand-worked Metaspace | unit | the four Metaspace cases, a newline kept inside a word | ids match sentencepiece's word boundaries |
| E-step equals enumeration | differential | forward-backward equals brute force over every segmentation, words weighted by count | the E-step is exact |
| EM never decreases the likelihood | property | four EM steps from a uniform seed on a short story | EM's guarantee holds in the code |
| Viterbi tie rule | boundary | three tied paths of `"▁aaa"`: the earliest-starting last piece wins | ids equal Hugging Face's on ties |
| unknown characters fuse | boundary | `"a日本b"` is `▁a <unk> b` | one `<unk>` per unknown run |
| unknown character penalty | boundary | an unknown scores the minimum minus 10, so a known path wins | the same ids as sentencepiece |
| samples match the posterior | statistical | 6000 samples at $a = 1$ and $a = 0.5$ fit $P(s)^a$ (chi-square, $p > 10^{-3}$) | subword regularization samples the right distribution |
| $a = 0$ is uniform | statistical | $a = 0$ is uniform over the six segmentations | the flattening end of the knob |
| one uniform per piece | unit | the generator advanced exactly `len(ids)` uniforms; same seed, same samples | Python and a later port replay one stream |
| negative $a$ | boundary | $a < 0$ is `ValueError` | it would favor the least likely splits |
| Hugging Face Unigram ids match the oracle | golden | a Hugging Face UnigramTrainer file, 300 strings, ids equal | real Unigram files load unchanged |
| sentencepiece ids match | golden | a sentencepiece model, 236 strings, ids equal sentencepiece's | the second, independent oracle |
| decode matches the oracle | golden | decode and `skip_special` on the same strings | printed text matches Hugging Face's |
| training properties | property | 200 ids from a short story, `<unk>` first, every character kept, scores sorted, deterministic | `C1` trains its 4096-piece arm |
| save and load round trip | property | `load(save(t))` keeps pieces, scores, and ids | a trained tokenizer ships with its model |
| fields outside the subset | boundary | byte fallback, a normalizer, prepend `"first"`, a BPE model: each `ValueError` | never wrong ids without an error |
| bad vocabularies | boundary | duplicate pieces, an `unk_id` past the end, a special that is not a piece | ids stay unambiguous |

### The model tests

The model suite, [`test_unigram.py`](../../../../course/solutions/learner-tests/L1.4/python/tests/l1-4-unigram/test_unigram.py), imports only names from the contract; for `sample_encode` it uses a small class whose `uniform()` replays fixed numbers. Its names and docstrings:

```python
def test_metaspace_marks_word_starts():
    """"a  b" is ["▁a", "▁", "▁b"]; a leading ▁ is not doubled; "" is []."""
def test_viterbi_picks_the_most_probable_segmentation():
    """With the section 3 vocab, "abc" is ▁a + bc (0.03), not ▁ab + c (0.01)."""
def test_viterbi_tie_goes_to_earliest_last_piece():
    """▁ = a = -1, aa = -2: "aaa" is [▁, a, aa]."""
def test_unknown_run_is_one_unk_with_penalty():
    """"a日本b" is ▁a <unk> b; an unknown scores min_score - 10."""
def test_log_likelihood_sums_all_segmentations():
    """log_likelihood(["abc"]) is ln 0.0451, not the Viterbi path's ln 0.03."""
def test_em_step_expected_counts():
    """One EM step on "abc": p(▁a) becomes 32.0 / 95.4 and p(c) 13.6 / 95.4."""
def test_em_step_weights_words_by_count():
    """em_step(["abc abc ab"]) equals the enumeration with "▁abc" counted twice."""
def test_sample_with_fixed_uniforms():
    """Drawing backward, u = 0 picks the candidate that starts earliest (▁a bc); u near 1 the latest (▁ a b c)."""
def test_sample_uses_one_uniform_per_piece():
    """After a sample of k pieces the generator has been asked exactly k times."""
def test_negative_alpha_raises():
    """sample_encode with alpha < 0 is ValueError."""
def test_decode_drops_only_the_first_space():
    """decode(encode("ab c")) is "ab c": the first piece's prepended ▁ goes, the others become spaces."""
def test_train_keeps_characters_and_size():
    """train on a few lines: vocab_size ids, <unk> first, every character a piece."""
```

## 5. The solution, step by step

### From first principles

1. **Everything is a question about paths in one lattice.** Encoding asks for the best path, the likelihood for the sum over paths, EM for each edge's share of that sum, and sampling for a random path. So the solution builds the lattice once per word (`_edges`) and runs four small dynamic programs over it.
2. **Edges into $j$ are what every recursion needs.** Viterbi and the forward sums read `edges[j]` (all pieces ending at $j$); the backward sums and the E-step iterate over the same lists. Sorting each list by start position fixes the order that the tie rule and sampling rely on.
3. **Probabilities must stay in log space.** A word of 20 characters has segmentation probabilities around $10^{-40}$; multiplying them underflows. Max becomes max, sum becomes log-sum-exp, and every probability is $\exp$ of a difference of logs.
4. **The E-step is forward-backward.** An edge $(i, v, j)$ lies on a path with total weight $\alpha(i) + \theta_v + \beta(j)$; dividing by $Z = \alpha(n)$ gives its posterior. Multiplying by $f(w)$ and summing over words gives expected counts, and the M-step is count over total.
5. **Sampling from $P(s)^a$ factorizes backward.** Given that the path reaches $j$, the probability that its last edge is $(i, v)$ is proportional to the weight of all paths to $i$ times that edge: $\exp(\alpha(i) + a\theta_v - \alpha(j))$. Drawing one edge at a time from $n$ back to 0 gives an exact sample with one uniform per piece.
6. **Unknowns must not break any of this.** An unknown edge with a finite, very low score keeps every word connected, so $Z > 0$ and Viterbi always has a path, while a known spelling always outscores it.
7. **Training must shrink a large candidate set to the target.** Seed generously, let EM concentrate probability, then drop the pieces whose removal costs the least likelihood, and repeat.

### The code, annotated

**Metaspace.**

```python
def metaspace(text: str) -> list[str]:
    """The Metaspace pre-tokenizer (replacement "▁", prepend "always", split)."""
    if not text:
        return []
    s = text.replace(" ", SPACE)
    if not s.startswith(SPACE):        # never a second ▁: " x" and "▁x" agree
        s = SPACE + s
    words: list[str] = []
    start = 0
    for i in range(1, len(s)):
        if s[i] == SPACE:               # split *before* each ▁: it starts the next word
            words.append(s[start:i])
            start = i
    words.append(s[start:])
    return words
```

**The lattice.** One trie walk per position lists every piece starting there.

```python
    def _edges(self, word: str, scale: float = 1.0) -> list[list[tuple[int, int, float]]]:
        """edges[j] = every (start, id, score * scale) of a piece ending at j."""
        n = len(word)
        edges: list[list[tuple[int, int, float]]] = [[] for _ in range(n + 1)]
        unk = (self.min_score - UNK_PENALTY) * scale
        for i in range(n):
            single = False
            for length, pid in self._trie.prefixes(word, i):     # M06.2: all pieces at i
                edges[i + length].append((i, pid, self.pieces[pid][1] * scale))
                single = single or length == 1
            if not single:
                # no one-character piece here: a finite, heavily penalized escape
                edges[i + 1].append((i, self.unk_id, unk))
        for e in edges:
            e.sort(key=lambda t: t[0])  # start ascending: the tie rule and the sampling order
        return edges
```

The `scale` argument is how sampling gets $a\theta_v$ without a second lattice builder. The special pieces (including `<unk>`) are kept out of `_trie` in the constructor, so they can only come from the unknown rule or from special matching on the raw text.

**Viterbi.** Strict `>` keeps the first candidate that reaches the best score.

```python
    def viterbi(self, word: str) -> list[int]:
        n = len(word)
        if n == 0:
            return []
        edges = self._edges(word)
        best = [-math.inf] * (n + 1)
        back: list[Optional[tuple[int, int]]] = [None] * (n + 1)
        best[0] = 0.0
        for j in range(1, n + 1):
            for i, pid, s in edges[j]:  # start positions ascending
                if best[i] == -math.inf:
                    continue
                cand = best[i] + s
                # strict >: on a tie the earliest start (the longest last piece) stays
                if back[j] is None or cand > best[j]:
                    best[j], back[j] = cand, (i, pid)
        path = []
        j = n
        while j > 0:                    # follow back pointers from the end
            i, pid = back[j]
            path.append((i, j, pid))
            j = i
        path.reverse()
        return self._fuse(path)         # adjacent <unk> ids become one
```

On section 3's `"▁abc"`: `best[2]` is $\max(\ln 0.1 + \ln 0.1, \ln 0.2) = \ln 0.2$ via `▁a`; `best[4]` compares `▁a` + `bc` ($\ln 0.03$) with `▁ab` + `c` ($\ln 0.01$) and the rest, so the path is `[5, 7]`.

**The E-step.** Forward, backward, then each edge's posterior times the word count.

```python
    def _expected_counts(self, words: Counter) -> dict[str, float]:
        counts = {p: 0.0 for i, (p, _) in enumerate(self.pieces) if i not in self.special_ids.values()}
        for w, c in words.items():          # each distinct word once, weighted by c = f(w)
            n = len(w)
            edges = self._edges(w)
            fwd = [-math.inf] * (n + 1)
            bwd = [-math.inf] * (n + 1)
            fwd[0] = 0.0
            bwd[n] = 0.0
            for j in range(1, n + 1):
                fwd[j] = _logsumexp([fwd[i] + s for i, _, s in edges[j]])   # alpha
            for j in range(n, 0, -1):
                for i, _, s in edges[j]:
                    bwd[i] = _logsumexp([bwd[i], s + bwd[j]])               # beta, accumulated
            z = fwd[n]                                                      # ln Z(w)
            for j in range(1, n + 1):
                for i, pid, s in edges[j]:
                    if pid != self.unk_id:
                        # posterior of edge (i, pid, j), times how often the word occurs
                        counts[self.pieces[pid][0]] += c * math.exp(fwd[i] + s + bwd[j] - z)
        return counts

def _m_step(counts: dict[str, float]) -> dict[str, float]:
    """log(count / total): the maximum-likelihood estimate of M07.2 on
    fractional (expected) counts, which its integer `mle` does not take."""
    total = math.fsum(counts.values())      # exact summation of many small floats
    return {p: math.log(c / total) for p, c in counts.items()}
```

The backward loop goes over $j$ descending, so `bwd[j]` is complete before any edge into $j$ uses it. On section 3 the counts come out as the table there, total $95.4 \cdot 10^{-3}/Z$, and `_m_step` gives $P(\texttt{▁a}) = 32.0/95.4$. `em_step` is this pair of functions plus keeping `<unk>` and the specials.

**Sampling.** Forward sums with scaled scores, then a backward walk with the inverse CDF.

```python
    def sample_encode(self, text: str, alpha: float, rng) -> list[int]:
        if not alpha >= 0:                  # also rejects NaN
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        # ... specials split out as in encode
            for w in metaspace(seg):
                n = len(w)
                edges = self._edges(w, alpha)          # a * theta on every edge
                fwd = [-math.inf] * (n + 1)
                fwd[0] = 0.0
                for j in range(1, n + 1):
                    fwd[j] = _logsumexp([fwd[i] + s for i, _, s in edges[j]])
                path = []
                j = n
                while j > 0:
                    # P(last edge into j is (i, pid)) = exp(alpha(i) + a*theta - alpha(j))
                    cands = [(i, pid, fwd[i] + s - fwd[j]) for i, pid, s in edges[j]]
                    u = rng.uniform()                  # exactly one draw per chosen piece
                    acc = 0.0
                    pick = cands[-1]                   # rounding left u unreached: the last
                    for c in cands:                    # start ascending
                        acc += math.exp(c[2])
                        if u < acc:
                            pick = c
                            break
                    path.append((pick[0], j, pick[1]))
                    j = pick[0]
                path.reverse()
                ids += self._fuse(path)
        return ids
```

With $a = 0$ every edge weight is 0, so `fwd[j]` counts paths (in log space) and each backward choice is proportional to the number of paths reaching its start: exactly uniform over segmentations.

**Training.** Seed, EM, prune, repeat.

```python
        # 1. seed: frequent substrings (score = frequency x length) and every character
        subs: Counter[str] = Counter()
        for w, c in words.items():
            for i in range(len(w)):
                for j in range(i + 2, min(len(w), i + MAX_PIECE_LEN) + 1):   # 2 to 16 code points
                    subs[w[i:j]] += c
        ranked = sorted(subs, key=lambda s: (-subs[s] * len(s), s))[: seed_factor * vocab_size]
        freq = dict(chars)
        freq.update({s: subs[s] for s in ranked})
        probs = mle(freq)                              # M07.2: count over total
        model = {p: math.log(q) for p, q in probs.items()}
        while True:
            # 2. EM: expected counts by forward-backward, then their MLE
            for _ in range(em_iters):
                counts = cls._from_scores(model)._expected_counts(words)
                # SentencePiece drops pieces expected fewer than 0.5 times;
                # a character always stays (count >= 0.5) so every text encodes.
                counts = {p: c for p, c in counts.items() if c >= 0.5 or p in chars}
                for ch in chars:
                    counts[ch] = max(counts.get(ch, 0.0), 0.5)
                model = _m_step(counts)
            if len(model) <= target:                   # target = vocab_size - 1 (<unk>)
                break
            # 3. prune the pieces that cost the least likelihood
            tok = cls._from_scores(model)
            keep = max(target, int(len(model) * shrink))
            loss = tok._removal_loss(words)
            optional = sorted((p for p in model if p not in chars), key=lambda p: (-loss.get(p, 0.0), p))
            kept = set(chars) | set(optional[: keep - len(chars)])
            model = {p: s for p, s in model.items() if p in kept}
        ordered = sorted(model.items(), key=lambda kv: (-kv[1], kv[0]))
        return cls([(UNK, 0.0)] + ordered, unk_id=0)
```

`_removal_loss` computes, for each multi-character piece on some Viterbi path, its Viterbi frequency times its score minus the best score of spelling its own text without it. A piece that is never on a Viterbi path has loss 0 and is pruned first. Pruning by at most a factor `shrink` per round lets EM re-estimate between rounds, so the scores of survivors adapt to what was removed.

Full solution: [`unigram.py`](../../../../course/solutions/python/tinyllm/tok/unigram.py) (also `log_likelihood`, `em_step`, `_removal_loss`, `decode`, `save`, `from_hf_json`).

### Why this solution

- **Cost.** Building the lattice of a word of $n$ characters is $O(n \cdot \ell_{\max})$ trie steps for pieces up to $\ell_{\max} = 16$; each dynamic program is linear in the number of edges, $O(n \cdot \ell_{\max})$. The E-step runs once per distinct word, not per occurrence. Training cost is dominated by seeding ($O(\sum_w |w| \cdot 16)$ substrings) and the EM passes per pruning round.
- **Alternatives.** Enumerating all segmentations is exact and exponential in $n$; the section 3 word has 6, a 20-character word thousands. Hard EM (Viterbi counts) is cheaper per step but optimizes a different objective and drifts from sentencepiece. Sampling forward (choosing the first piece by its own score) does not sample $P(s)^a$; only forward-filtering, backward-sampling is exact. A suffix array (as in sentencepiece) finds seed substrings faster; the counter is simpler and fine at the sizes here.
- **Accepted tradeoffs.** No byte fallback: an unseen character becomes `<unk>` rather than byte pieces, which `from_hf_json` enforces by refusing `byte_fallback`. The M-step is the plain MLE, not sentencepiece's digamma-adjusted one, so trained scores differ slightly from sentencepiece's own trainer while encoding with a given model matches exactly.

### Explain it back

1. Why does Viterbi use `cand > best[j]` and scan edges by ascending start, and what changes with `>=`?
   <details><summary>Answer</summary>
   With strict `>`, the first edge to reach the best score keeps the back pointer, and edges are scanned by ascending start, so a tie goes to the edge that starts earliest, the longest last piece. That is sentencepiece's and Hugging Face's rule. With `>=`, the last tied edge wins, the shortest last piece, and ids differ on every tie (`"▁aaa"` in section 4).
   </details>
2. Why is the unknown edge scored `min_score - 10` and not $-\infty$?
   <details><summary>Answer</summary>
   With $-\infty$, a word containing an uncovered character would have no finite path: Viterbi could not encode it and $Z(w) = 0$ would make the E-step divide by zero. A finite score keeps the lattice connected, and 10 nats below the worst piece makes any path that avoids the unknown win whenever one exists.
   </details>
3. In the backward sampling step, why must the probability include `fwd[i]` and not just the piece's score?
   <details><summary>Answer</summary>
   The probability that the last piece is $(i, v)$ given that the path reaches $j$ is the total weight of all paths that end with that edge, divided by the total weight of all paths to $j$: $\exp(\alpha(i) + a\theta_v - \alpha(j))$. Dropping $\alpha(i)$ ignores how many ways there are to reach $i$, so samples no longer follow $P(s)^a$.
   </details>
4. Why does the E-step weight each word's posteriors by its count `c` instead of iterating over occurrences?
   <details><summary>Answer</summary>
   The posteriors depend only on the word and the model, so every occurrence of a word contributes the same amount. Computing them once per distinct word and multiplying by $f(w)$ gives the same expected counts at a fraction of the cost. Forgetting the multiplication would weigh rare and frequent words equally.
   </details>
5. Why does training keep every character with a count of at least 0.5, even when EM would drop it?
   <details><summary>Answer</summary>
   If a character were dropped, every place in the training text where no longer piece covers it would need an unknown edge, and the tokenizer would emit `<unk>` on its own training data. Keeping characters guarantees that every training text still has a known path.
   </details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. hard EM: max instead of log-sum-exp in the forward or backward pass | the likelihood is the Viterbi path's; trained scores drift from sentencepiece's | `fwd` and `bwd` use `_logsumexp`; `log_likelihood` returns `fwd[n]` |
| 2. `>=` in the Viterbi update | ties go to the latest-starting last piece, and ids differ from Hugging Face's | strict `>` over edges sorted by start |
| 3. no penalty on unknown edges | a path through `<unk>` beats a known spelling | unknown edges score `min_score - UNK_PENALTY` |
| 4. sampling each backward step by the piece's own score | samples do not follow $P(s)^a$; the regularizer trains on the wrong distribution | candidates weighted by `fwd[i] + s - fwd[j]` |
| 5. letting pruning remove characters | training text hits `<unk>` after the first prune | characters are excluded from `optional` and always in `kept` |
| 6. not fusing adjacent unknowns | an unknown word of five characters costs five ids | `_fuse` skips an `<unk>` that follows an `<unk>` |
| 7. ignoring word counts in the E-step | frequent words weigh as much as rare ones | posteriors multiplied by the word's count `c` |
| 8. one uniform per word, reused for every piece | samples are correlated and a port cannot replay the stream | `rng.uniform()` is called once per backward step |
| 9. Metaspace prepending a second `▁`, or splitting after it | word boundaries and ids differ on every text | prepends only if absent, splits before each `▁` |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L1.1` | the protocol, `check_ids`; `<unk>` at id 0 as in the char tokenizer |
| Back | `M06.2` | `Trie.prefixes` lists every piece starting at a position: the lattice's edges |
| Back | `M07.2` | `mle` turns the seed substrings' counts into their first probabilities |
| Back | `M11.1` | log-likelihood and cross-entropy: EM raises one and lowers the other (reading) |
| Back | `S-M07b` | q8 is one EM step by hand; q9 proves count over total is the MLE (reading) |
| Forward | `L1.6` | measures a Unigram tokenizer against Hugging Face's numbers, `<unk>` counted as a fallback |

`C1` trains a 4096-piece Unigram on the same 16 MB sample as its BPE and compares the two with `L1.6` and a short training run.

## Going further

| Piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `train` | SentencePiece `unigram_model_trainer` | a suffix array for seed substrings, digamma-adjusted M-step, multithreaded E-step | `sentencepiece/src/unigram_model_trainer.cc` |
| `viterbi` | Hugging Face `Unigram::encode_optimized` | the same lattice in Rust with a byte-indexed trie and a cache | `tokenizers/src/models/unigram/model.rs` |
| `sample_encode` | SentencePiece `SampleEncode` and `NBestEncode` | n-best by A* search, and sampling without replacement (Gumbel top-k) | `sentencepiece/src/unigram_model.cc` |
| unknown edges | byte fallback | unknown characters spelled as `<0xNN>` byte pieces instead of `<unk>` | Llama and Gemma tokenizers, `byte_fallback: true` |
