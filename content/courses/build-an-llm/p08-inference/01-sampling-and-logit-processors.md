<!-- ol:module L8.1 -->
# Sampling and logit processors

## Overview

| | |
|---|---|
| **Module** | `L8.1` · build · Python · Pass 6 · 4 to 5 h |
| **You build** | `python/tinyllm/infer/sample.py`: `SamplingParams` (with `validate`), `request_rng`, `apply_penalties`, `token_logprobs`, `process_logits`, `sampling_distribution`, `sample`, `sampled_entropy` |
| **Contract** | [`course/contracts/py/tinyllm/infer/sample.pyi`](../../../../course/contracts/py/tinyllm/infer/sample.pyi) · the op order, to the bit: [`spec/sampling.md`](../../../../course/contracts/spec/sampling.md) · the generator: [`spec/pcg32.md`](../../../../course/contracts/spec/pcg32.md) |
| **Tests** | `course/tests/L8.1/` (what they check: section 4) · fixture `course/fixtures/L8.1/sampler_golden.json` (also the Rust port's) · your own tests in `python/tests/l8-1-sample/`, rung R5, graded by mutation (threshold 0.80, every pitfall mutant required) |
| **Needs** | `M07.1` (`sample_categorical` is step 11) · `M06.3` (`PCG32` and its `sample` sub-stream) · `M11.1` (`entropy` of what was sampled) · reading: `M09.2` the stable softmax, `M07.6` |
| **Used by** | `L8.2` samples every generated token · `ds.04` checks its C top-k against this top-k · `L8.7` samples from its masked logits · `L8.6` the target distribution of speculative decoding · later: `L10.1` the Rust sampler (same ids on the same logits and seed), `L12.3` GRPO rollouts |
| **Milestone** | `MS-L8` (inference: cache equivalence, quantization, speculative decoding) |
| **Optional depth** | Holtzman et al., "The Curious Case of Neural Text Degeneration" (2020, top-p); Keskar et al., "CTRL" (2019, repetition penalty); Nguyen et al., "Turning Up the Heat: Min-p Sampling" (2024); the OpenAI API reference for `presence_penalty` and `frequency_penalty` |

## Key Takeaways

- Sampling is a fixed pipeline over float64: penalties, then greedy or temperature, then top-k, top-p, min-p (each on what the previous one kept), then softmax, one uniform, and the inverse CDF; the spec fixes the order, the arithmetic, and the draw count, so Python and Rust emit the same token (`test_hand_example`, `test_golden_ids_and_logprobs`).
- Every sum is a left-to-right loop in ascending id order: Python's `sum()` (compensated since 3.12) and numpy's pairwise `np.sum` give different last bits, and a last bit is enough to flip a token (`test_sums_are_sequential_in_ascending_ids`).
- Top-p keeps the token that crosses $p$, the comparison is $\ge$, and ties anywhere go to the lowest id (`test_top_p_keeps_the_crossing_token`, `test_top_k_ties_go_to_the_lowest_id`).
- The logprob is taken after the penalties and before temperature and filtering, so it does not depend on how adventurous the sampler was (`test_hand_example_intermediates`).
- One uniform per sampled token, none per greedy token: the generator's position is a pure function of how many tokens were sampled (`test_one_draw_per_token_none_for_greedy`).

## How to work this chapter

```bash
ol start L8.1               # stubs sample.py into your repo
ol tests L8.1               # read the test catalog first
ol check L8.1               # course tests, then your tests graded by mutation
ol mutate L8.1              # the full mutation grade of your tests
ol check L8.1 --ref-deps    # only if you skipped M07.1, M06.3, or M11.1
ol diff  L8.1               # after passing: your code against the reference
```

---

## 1. Why now

Your engine has served greedy text since Pass 1: the tracer picks the largest logit. A model you trained yourself, decoded greedily, loops ("the cat sat on the mat. the cat sat on the mat."), and every serving API you will expose (`L10.5`) accepts `temperature`, `top_p`, `presence_penalty`, and a `seed`. Two engines will run them: this Python sampler, which the course tests and your own experiments use, and the Rust engine's (`L10.1`), which serves traffic. If they disagree on a single detail (the order of two filters, how a tie is broken, how a sum is accumulated), the same request with the same seed returns different text from the two, and every parity test, every reproduction of a bad output, and every A/B comparison of the engine against Python becomes noise. This module implements `spec/sampling.md` exactly and proves it on an independent transcription of the spec, the same fixture the Rust port must pass.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $V$ | vocabulary size | integer |
| $x \in \mathbb{R}^V$ | the model's next-token logits | `float32[V]` |
| $l \in \mathbb{R}^V$ | the logits in float64 after each step | `float64[V]` |
| $\mathrm{prompt}$, $\mathrm{out}$ | the request's prompt ids and the ids generated so far | lists of ids |
| $c_i$ | how many times id $i$ occurs in $\mathrm{out}$ | integer |
| $r$, $a_p$, $a_f$ | repetition, presence, and frequency penalties | floats |
| $T$ | temperature | float $\ge 0$ |
| $k$, $p$, $m$ | top-k, top-p, min-p | integer, floats |
| $K$ | the kept set of ids after the filters | set of ids |
| $q_i = e^{l_i - M} / Z$ | step 9: softmax over $K$, $M = \max_K l$, $Z = \sum_K e^{l_i - M}$ | `float64[V]`, 0 outside $K$ |
| $u$ | one uniform on $[0, 1)$ from the request's generator | float |

### 2.1 The pipeline

Steps 1 to 11 of `spec/sampling.md`, in this order and no other:

1. **Widen**: $l_i = \mathrm{f64}(x_i)$.
2. **Repetition** (Hugging Face): for each distinct $i$ in prompt and out, $l_i \leftarrow l_i / r$ if $l_i > 0$, else $l_i \cdot r$.
3. **Presence and frequency** (OpenAI): for each $i$ with $c_i > 0$, $l_i \leftarrow l_i - a_f c_i - a_p$. The **logprob** is $\mathrm{logsoftmax}(l)_i = l_i - M - \log Z$ over all $V$ ids, here.
4. **Greedy**: if $T = 0$, return $\arg\max_i l_i$, ties to the lowest id, with no draw.
5. **Temperature**: $l_i \leftarrow l_i / T$.
6. **Top-k**: order ids by $(l$ descending, id ascending$)$ and keep the first $k$.
7. **Top-p**: compute $q$ over the kept ids; walk them in that order adding $q$ to $s$; keep every id up to and including the first one with $s \ge p$.
8. **Min-p**: compute $q$ over the kept ids; keep those with $q_i \ge m \max_K q$.
9. **Softmax** over $K$, ascending ids.
10. **Draw** exactly one $u$.
11. **Inverse CDF**: walk $K$ in ascending id order adding $q_i$ to $c$; return the first id with $u < c$ (`M07.1`'s `sample_categorical`).

### 2.2 Penalties

The repetition penalty discourages any token already seen. Dividing a positive logit by $r > 1$ makes it smaller, but dividing a negative logit by $r$ moves it toward 0, making the token more likely; that is why the rule multiplies negative logits. It applies once per distinct id, so a token seen ten times is penalized once. Presence and frequency are additive and count only generated tokens: presence is a flat cost for having appeared, frequency grows with the count. Repetition first, then the additive ones: on $l = 3$, $r = 2$, $a_p = 1$ that gives $3/2 - 1 = 0.5$, while the other order gives $(3 - 1)/2 = 1$.

### 2.3 Filters

Each filter sees the distribution the previous one left. Top-k is a hard cap. Top-p (nucleus) keeps the smallest set of most likely tokens whose mass reaches $p$, recomputed on what top-k kept: with $q = [0.4, 0.3, 0.2, 0.1]$ and $k = 2$, id 0 is worth $0.4/0.7 = 0.57$ among the survivors, which alone reaches $p = 0.55$. Min-p keeps tokens at least $m$ times as likely as the best one, so it adapts: a confident distribution keeps few tokens, a flat one keeps many. Masked logits ($-\infty$, from `L8.7`) never enter $K$. Removed ids are $-\infty$ in `process_logits`, not a large negative number: $-10^9$ divided by a later temperature, or exponentiated in another language's float type, can become a small nonzero probability.

### 2.4 Greedy and temperature

$T \to 0$ concentrates all mass on the argmax: with a clear margin, $T = 10^{-3}$ and top-k $= 1$ give the greedy token for any $u$. $T = 0$ itself is defined as greedy, with no division and no draw. Ties go to the lowest id everywhere: in step 4, in the top-k order, and in the inverse CDF's walk.

### 2.5 Exactness and the draw

The inverse CDF maps $u \in [F_{i-1}, F_i)$ to id $i$, an interval of length $q_i$, so $P(i) = q_i$ exactly; `sampling_distribution` returns that $q$, and the tests check the boundaries $u = F_i^-$ and $u = F_i$ for every configuration. The normalizer $Z$ must be a plain running sum. Since Python 3.12, `sum()` of floats uses Neumaier's compensated summation, and `np.sum` adds pairwise: both are more accurate, and both differ from the Rust loop in the last bit of $q$. When $u$ falls within an ulp of a boundary, that flips the token. Accuracy is not the goal here; agreement is. If rounding leaves the final $c$ below $u$, the token is the largest kept id (which always has $q > 0$).

### 2.6 The request generator

A request with seed $s$ draws from $\mathrm{stream}(s, \texttt{sample})$ of `spec/pcg32.md`: `PCG32(s).substream("sample")`, a PCG32 seeded with $\mathrm{splitmix64}(s + 4 \cdot \mathrm{GOLDEN})$ on sequence 4. Exactly one $u$ per sampled token means a decode worker that takes over a request after $n$ sampled tokens skips exactly $n$ uniforms (disaggregated serving, `L10.6`), and speculative decoding's acceptance draws (`M07.6`, `L8.6`) interleave at known positions.

## 3. Worked example by hand

The spec's example: $V = 5$, $x = [1, 3, 2, 3, -1]$, no history, $T = 1$, top-k $= 3$, top-p $= 0.8$, seed 0.

| step | what happens | result |
|---|---|---|
| 1 to 3 | no penalties | $l = [1, 3, 2, 3, -1]$ |
| logprob | $M = 3$, $Z = e^{-2} + 1 + e^{-1} + 1 + e^{-4} = 2.52153$ | ids 1 and 3: $-\ln Z = -0.92487$ |
| 5 | $T = 1$ | unchanged |
| 6 | order $(3.0, 1), (3.0, 3), (2.0, 2), (1.0, 0), (-1.0, 4)$; keep 3 | $\{1, 3, 2\}$ |
| 7 | $q$ over $\{1, 2, 3\}$: $Z = 1 + e^{-1} + 1 = 2.3679$, $q = [0.42232, 0.15536, 0.42232]$; walk 1, 3: $s = 0.42232$, then $0.84464 \ge 0.8$ | $\{1, 3\}$ |
| 9 | softmax over $\{1, 3\}$ | $q_1 = q_3 = 0.5$ |
| 10 | $\mathrm{stream}(0, \texttt{sample})$: child seed `0xF88BB8A8724C81EC`, first uniform | $u = 0.80209$ |
| 11 | $c = 0.5$ at id 1 ($u$ not below), $c = 1.0$ at id 3 ($u < 1$) | **id 3**, logprob $-0.92487$ |

`process_logits` returns $[-\infty, 3, -\infty, 3, -\infty]$ and `sampling_distribution` returns $[0, 0.5, 0, 0.5, 0]$. These numbers are the first cases in section 4: `test_hand_example` and `test_hand_example_intermediates`.

## 4. The interface

```python
# python/tinyllm/infer/sample.py
@dataclass
class SamplingParams:
    temperature: float = 1.0; top_k: int = 0; top_p: float = 1.0; min_p: float = 0.0
    repetition_penalty: float = 1.0; presence_penalty: float = 0.0; frequency_penalty: float = 0.0
    seed: Optional[int] = None; max_tokens: int = 128; stop: list[str] = []; logprobs: int = 0
    def validate(self) -> None
def request_rng(seed: int) -> PCG32                                    # stream(seed, "sample")
def apply_penalties(logits, p, history=(), prompt=()) -> NDArray       # steps 1 to 3, float64
def token_logprobs(logits, p, history=(), prompt=()) -> NDArray        # log_softmax after step 3
def process_logits(logits, p, history=(), prompt=()) -> NDArray        # steps 1 to 8, -inf removed
def sampling_distribution(logits, p, history=(), prompt=()) -> NDArray # q, 0 outside K
def sample(logits, p, history, rng, prompt=()) -> tuple[int, float]    # (id, logprob)
def sampled_entropy(logits, p, history=(), prompt=()) -> float         # H(q) in nats (M11.1)
```

`history` is the ids generated so far (the spec's `out`) and `prompt` the prompt ids: the repetition penalty looks at both, presence and frequency at `history` only. The catalog's signature has one `history`; the contract adds `prompt` because the spec needs both (DEVIATIONS B81-05).

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example` | unit | section 3 end to end: id 3, logprob, processed logits, $q$ | you and the test agree on the spec |
| `test_hand_example_intermediates` | unit | top-k's $q$; logprobs ignore temperature and filters | what `L10.5` reports as `logprobs` |
| `test_request_rng_is_the_sample_stream` | golden | child seed and first uniform of seed 0 | the same seed gives the same text in Rust |
| `test_repetition_penalty_hf_semantics` | unit | divide positive, multiply negative, once per distinct id | repeated unlikely tokens stay unlikely |
| `test_presence_and_frequency_openai_semantics` | unit | counts of generated ids only | OpenAI compatibility |
| `test_penalty_order` | unit | repetition before presence | the spec's order |
| `test_top_p_keeps_the_crossing_token` | boundary | exact sums at $p$: $\ge$, and the crossing token stays | the nucleus is never empty |
| `test_top_k_ties_go_to_the_lowest_id` | boundary | ties at the cut | deterministic across languages |
| `test_min_p_is_relative_to_the_top` | unit | the cut is $m \max q$, inclusive | min-p adapts to confidence |
| `test_filters_compose_in_spec_order` | unit | top-p after top-k, min-p after top-p | the order changes the kept set |
| `test_masked_logits_are_never_sampled` | property | $-\infty$ ids have $q = 0$ and stay $-\infty$ | constrained decoding (`L8.7`) |
| `test_inverse_cdf_boundaries` | statistical | $u = F_i^-$ gives $i$, $u = F_i$ the next id, for six configurations | $P(i) = q_i$ exactly |
| `test_chi_square_against_the_distribution` | statistical | 20000 seeded draws follow $q$ | the sampler and its reported distribution agree |
| `test_one_draw_per_token_none_for_greedy` | unit | draw accounting | disaggregated serving, speculative decoding |
| `test_greedy_and_its_limits` | property | $T = 0$, top-k $= 1$, tiny $T$ all give the argmax, ties lowest | greedy parity under the near-tie rule |
| `test_sums_are_sequential_in_ascending_ids` | unit | $q$ and logprobs bitwise equal a plain loop | bit parity with Rust |
| `test_golden_ids_and_logprobs` | golden | 12 cases x 12 tokens from an independent transcription | the file `L10.1` is held to |
| `test_entropy_of_the_sampling_distribution` | unit | $H(q)$ after filtering; 0 when greedy | what `L8.2` logs per token |
| `test_params_are_validated` | boundary | out-of-range parameters and logits raise | caller bugs never return a token |

### Your tests (rung R5)

Write `python/tests/l8-1-sample/` against the contract only. The strongest oracle is the spec itself: transcribe steps 1 to 11 in plain Python (lists, `math.exp`, explicit loops), feed it and your sampler the same uniforms through a scripted source, and compare ids and logprobs on random logits with ties, masks, and every processor. Add the cases a random search rarely hits: sums exactly at $p$, ties at the top-k cut, greedy with no draw. `ol mutate L8.1` grades the suite by the planted bugs it kills.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. temperature before the penalties, or presence before repetition | penalized tokens shift by a temperature-dependent amount | `test_hand_example_intermediates` (mutant `s01`), `test_penalty_order` (mutant `s21`) |
| 2. top-p with `s > p`, or dropping the token that crosses $p$ | the nucleus loses a token; with $p$ small it can be empty | `test_top_p_keeps_the_crossing_token` (mutants `s03`, `s20`) |
| 3. dividing negative logits by $r$, penalizing per occurrence, or counting prompt ids in presence and frequency | repeated unlikely tokens become likely; prompts bias the output | `test_repetition_penalty_hf_semantics` (mutants `s05`, `s16`), `test_presence_and_frequency_openai_semantics` (mutants `s06`, `s07`) |
| 4. ties to the highest id | Python and Rust disagree on tied logits | `test_top_k_ties_go_to_the_lowest_id` (mutant `s02`), `test_greedy_and_its_limits` (mutant `s09`) |
| 5. a greedy step that draws | every later seeded token shifts | `test_one_draw_per_token_none_for_greedy` (mutant `s10`) |
| 6. logprobs of the filtered, tempered distribution | logprobs change with `top_p`; perplexity from logprobs is wrong | `test_hand_example` (mutant `s04`) |
| 7. `sum()` or `np.sum` for $Z$ | rare token flips against Rust, never reproducible | `test_sums_are_sequential_in_ascending_ids` (mutants `s11`, `s12`) |
| 8. top-p on the unfiltered distribution, or min-p before top-p | a different kept set than the spec's | `test_filters_compose_in_spec_order` (mutants `s13`, `s18`) |
| 9. `PCG32(seed)` instead of the `sample` sub-stream | a seeded request differs from the Rust engine's | `test_request_rng_is_the_sample_stream` (mutant `s15`) |
| 10. removed ids as $-10^9$ instead of $-\infty$ | masked tokens reappear after a later scaling | `test_masked_logits_are_never_sampled` (mutant `s22`) |

## 6. Where it's used next
| Forward | `L12.3` | Registered module relationship. |

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M07.1` | `sample_categorical` is step 11; `UniformSource` is the generator type |
| Back | `M06.3` | `PCG32(seed).substream("sample")` is the request's generator |
| Back | `M11.1` | `entropy` of the sampled distribution |
| Back | `M09.2` | the shift by the maximum in step 9 |
| Forward | `L8.2` | `generate` samples every token with `sample` and logs `sampled_entropy` |
| Forward | `ds.04` | the C heap top-k is checked against this sampler's top-k on fixture logits |
| Forward | `L10.1` | the Rust sampler, held to `sampler_golden.json` and to your ids on shared logits |
| Forward | `L8.6` | `sampling_distribution` is the target $p$ of `M07.6`'s acceptance test |
| Forward | `L8.7` | constrained decoding writes $-\infty$ into the logits before `sample` |

If you skip this module, `ol check L8.2` stops with `L8.2 needs L8.1`; `--ref-deps` substitutes the reference.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `sample` | vLLM `Sampler` | batched penalties and top-k/top-p over a whole batch on the GPU, per-request generators | `vllm/v1/sample/sampler.py` |
| `apply_penalties` | Hugging Face `LogitsProcessor`s | one composable processor per rule (`RepetitionPenaltyLogitsProcessor`, `MinPLogitsWarper`) | `transformers/generation/logits_process.py` |
| `process_logits` | llama.cpp samplers | a configurable chain (typical-p, mirostat, DRY) applied in a user-chosen order | `src/llama-sampling.cpp` |
| `request_rng` | SGLang deterministic sampling | per-request seeds that survive batching and preemption | `sglang/srt/sampling/` |
