<!-- ol:module L1.4 -->
# Unigram LM tokenizer (EM, Viterbi, subword sampling)

## Overview

| | |
|---|---|
| **Module** | `L1.4` · build · Python · Pass 3 · 6 to 8 h |
| **You build** | `python/tinyllm/tok/unigram.py`: `metaspace` and `UnigramTokenizer` (`viterbi`, `encode`, `sample_encode`, `log_likelihood`, `em_step`, `train`, `decode`, `save`, `load`, `from_hf_json`) |
| **Contract** | [`course/contracts/py/tinyllm/tok/unigram.pyi`](../../../../course/contracts/py/tinyllm/tok/unigram.pyi) · format: [`tokenizer.md`](../../../../course/contracts/formats/tokenizer.md) (Unigram in the `tokenizer.json` subset) |
| **Tests** | `course/tests/L1.4/` (what they check: section 4) · your own tests in `python/tests/l1-4-unigram/`, rung R2, graded by mutation (threshold 0.60) |
| **Needs** | `L1.1` the protocol and `check_ids` · `M06.2` `Trie.prefixes` builds the lattice · `M07.2` `mle` scores the seed pieces · reading: `M11.1` (log-likelihood and cross-entropy), `M06.3` (the PCG32 that `sample_encode` draws from; the course tests pass the frozen one), `S-M07b` q8 and q9 (one EM step and the MLE by hand) (or `--ref-deps`) |
| **Used by** | `L1.6` measures a Unigram tokenizer loaded with `from_hf_json` · later: `C1` compares BPE and Unigram at 4096 tokens on the same sample |
| **Milestone** | `MS-L1` |
| **Optional depth** | Kudo, *Subword Regularization* (2018), sections 3 and 4; Kudo and Richardson, *SentencePiece* (2018); Dempster, Laird, and Rubin, *Maximum Likelihood from Incomplete Data via the EM Algorithm* (1977) |

## Key Takeaways

- A Unigram tokenizer is a probability for each piece; a segmentation's probability is the product of its pieces', and encoding picks the most probable segmentation by Viterbi over a lattice (`test_hand_example_viterbi`).
- The likelihood of a word sums over all its segmentations, so training is EM: expected piece counts by forward-backward, then count over total, and the log-likelihood never goes down (`test_hand_example_em_step`, `test_em_step_never_decreases_likelihood`).
- Unlike BPE, Unigram can sample segmentations with probability proportional to $P(s)^\alpha$, which `C1` uses as subword regularization (`test_sample_matches_posterior`).
- Every rule that decides ties and unknowns is part of the contract: the earliest-starting last piece wins a Viterbi tie, an unknown character scores the minimum minus 10, and adjacent unknowns fuse (`test_viterbi_tie_rule`, `test_unknown_character_penalty`, `test_unknown_characters_fuse`).
- With those rules exact, your tokenizer loads Hugging Face and sentencepiece models unchanged and reproduces their ids (`test_hf_unigram_ids_match_oracle`, `test_spm_unigram_ids_match_sentencepiece`).

## How to work this chapter

```bash
ol start L1.4              # stubs unigram.py into your repo
ol tests L1.4              # read the test catalog first
ol check L1.4              # exit code is the verdict; then grades your tests by mutation
ol check L1.4 --ref-deps   # only if your L1.1, M06.2, or M07.2 is not passing yet
ol diff  L1.4              # after passing: your code against the reference
```

Do `S-M07b` q8 and q9 first: they are one EM step and the count MLE on paper, the two halves of `em_step`.

---

## 1. Why now

`L1.2` gave you byte-level BPE, the tokenizer of GPT-2 and SmolLM2, and it has one answer for every word: the merges decide. The capstone (`C1`) has to choose a tokenizer for its own model, and the decision is an ablation: BPE against the other widely used family at the same vocabulary size, on the same sample, compared by the metrics of `L1.6` and by bits per byte after a short training run. That family is the Unigram language model of SentencePiece (T5, ALBERT, XLNet, and many multilingual models). It is a probabilistic model, so it can do what BPE cannot: score a segmentation, and sample a different segmentation of the same text on every pass, a regularizer for small models. Without this module the ablation has only one arm.

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

These are the first cases in section 4: `test_hand_example_viterbi`, `test_hand_example_likelihood`, `test_hand_example_em_step`, and `test_hand_example_metaspace`.

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

`special_ids` are the `<unk>` piece plus `specials`; typed in the raw text, they match as written (leftmost longest) and are one id each. `sample_encode` takes any object with `uniform() -> float`; the course tests pass the frozen PCG32 of `course/tests/_lib`, and your own `M06.3` PCG32 fits the same slot. `save` writes a `tokenizer.json` with the `Unigram` model and the Metaspace pre-tokenizer and decoder (`prepend_scheme` `"always"`); `from_hf_json` refuses byte fallback, a normalizer, or another Metaspace scheme.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_viterbi` | unit | `"abc"` is `▁a bc`, ids `[5, 7]`, and decodes back | you and the test agree on the definition |
| `test_hand_example_likelihood` | unit | $\ln 0.0451$ for one word, twice that for two | the EM objective sums over segmentations |
| `test_hand_example_em_step` | unit | the section 3 table, each probability to 1e-12 | the M-step divides by the total |
| `test_hand_example_metaspace` | unit | the four Metaspace cases, a newline kept inside a word | ids match sentencepiece's word boundaries |
| `test_em_step_matches_enumeration` | differential | forward-backward equals brute force over every segmentation, words weighted by count | your E-step is exact |
| `test_em_step_never_decreases_likelihood` | property | four EM steps from a uniform seed on the story fixture | EM's guarantee holds in your code |
| `test_viterbi_tie_rule` | boundary | three tied paths of `"▁aaa"`: the earliest-starting last piece wins | ids equal Hugging Face's on ties |
| `test_unknown_characters_fuse` | boundary | `"a日本b"` is `▁a <unk> b` | one `<unk>` per unknown run |
| `test_unknown_character_penalty` | boundary | an unknown scores the minimum minus 10, so a known path wins | the same ids as sentencepiece |
| `test_sample_matches_posterior` | statistical | 6000 samples at $a = 1$ and $a = 0.5$ fit $P(s)^a$ (chi-square, $p > 10^{-3}$) | subword regularization samples the right distribution |
| `test_sample_alpha_zero_is_uniform` | statistical | $a = 0$ is uniform over the six segmentations | the flattening end of the knob |
| `test_sample_draws_one_uniform_per_piece` | unit | the generator advanced exactly `len(ids)` uniforms; same seed, same samples | Python and a later port replay one stream |
| `test_sample_rejects_negative_alpha` | boundary | $a < 0$ is `ValueError` | it would favor the least likely splits |
| `test_hf_unigram_ids_match_oracle` | golden | a Hugging Face UnigramTrainer file, 300 strings, ids equal | real Unigram files load unchanged |
| `test_spm_unigram_ids_match_sentencepiece` | golden | a sentencepiece model, 236 strings, ids equal sentencepiece's | the second, independent oracle |
| `test_hf_unigram_decode_matches_oracle` | golden | decode and `skip_special` on the same strings | printed text matches Hugging Face's |
| `test_train_properties` | property | 200 ids from the story, `<unk>` first, every character kept, scores sorted, deterministic | `C1` trains its 4096-piece arm |
| `test_save_load_roundtrip` | property | `load(save(t))` keeps pieces, scores, and ids | a trained tokenizer ships with its model |
| `test_from_hf_json_rejects_outside_subset` | boundary | byte fallback, a normalizer, prepend `"first"`, a BPE model: each `ValueError` | never wrong ids without an error |
| `test_constructor_rejects_bad_vocab` | boundary | duplicate pieces, an `unk_id` past the end, a special that is not a piece | ids stay unambiguous |

### Your tests (rung R2)

Write these in `python/tests/l1-4-unigram/test_unigram.py`, importing only names from the contract (for `sample_encode`, a small class with a `uniform()` method that replays fixed numbers is enough):

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

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. hard EM: max instead of log-sum-exp in the forward or backward pass | the likelihood is the Viterbi path's; trained scores drift from sentencepiece's | `test_hand_example_em_step` (mutant `s08`), `test_hand_example_likelihood` (mutant `s19`) |
| 2. `>=` in the Viterbi update | ties go to the latest-starting last piece, and ids differ from Hugging Face's | `test_viterbi_tie_rule` (mutant `s01`) |
| 3. no penalty on unknown edges | a path through `<unk>` beats a known spelling | `test_unknown_character_penalty` (mutant `s03`) |
| 4. sampling each backward step by the piece's own score | samples do not follow $P(s)^a$; the regularizer trains on the wrong distribution | `test_sample_matches_posterior` (mutant `s05`) |
| 5. letting pruning remove characters | training text hits `<unk>` after the first prune | `test_train_properties` (mutant `s11`) |
| 6. not fusing adjacent unknowns | an unknown word of five characters costs five ids | `test_unknown_characters_fuse` (mutant `s02`) |
| 7. ignoring word counts in the E-step | frequent words weigh as much as rare ones | `test_em_step_matches_enumeration` (mutant `s07`) |
| 8. one uniform per word, reused for every piece | samples are correlated and a port cannot replay the stream | `test_sample_draws_one_uniform_per_piece` (mutant `s21`) |
| 9. Metaspace prepending a second `▁`, or splitting after it | word boundaries and ids differ on every text | `test_hand_example_metaspace` (mutants `s09`, `s10`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L1.1` | the protocol, `check_ids`; `<unk>` at id 0 as in the char tokenizer |
| Back | `M06.2` | `Trie.prefixes` lists every piece starting at a position: the lattice's edges |
| Back | `M07.2` | `mle` turns the seed substrings' counts into their first probabilities |
| Back | `M11.1` | log-likelihood and cross-entropy: EM raises one and lowers the other (reading) |
| Back | `S-M07b` | q8 is one EM step by hand; q9 proves count over total is the MLE (reading) |
| Forward | `L1.6` | measures a Unigram tokenizer against Hugging Face's numbers, `<unk>` counted as a fallback |

`C1` trains a 4096-piece Unigram on the same 16 MB sample as its BPE and compares the two with `L1.6` and a short training run. If you skip this module, `ol check L1.6` stops with `needs L1.4`: build it, or rerun with `--ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `train` | SentencePiece `unigram_model_trainer` | a suffix array for seed substrings, digamma-adjusted M-step, multithreaded E-step | `sentencepiece/src/unigram_model_trainer.cc` |
| `viterbi` | Hugging Face `Unigram::encode_optimized` | the same lattice in Rust with a byte-indexed trie and a cache | `tokenizers/src/models/unigram/model.rs` |
| `sample_encode` | SentencePiece `SampleEncode` and `NBestEncode` | n-best by A* search, and sampling without replacement (Gumbel top-k) | `sentencepiece/src/unigram_model.cc` |
| unknown edges | byte fallback | unknown characters spelled as `<0xNN>` byte pieces instead of `<unk>` | Llama and Gemma tokenizers, `byte_fallback: true` |
