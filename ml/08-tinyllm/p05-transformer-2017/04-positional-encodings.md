<!-- ol:module L5.4 -->
# Positional encodings: sinusoidal, learned

## Overview

| | |
|---|---|
| **Module** | `L5.4` · build · Python · Pass 5 · 2 h, plus your graded tests (rung R5) |
| **You build** | `python/tinyllm/xfmr/pos.py`: `sinusoidal_freqs`, `sinusoidal_pe`, `shift_pe`, `SinusoidalPE`, `LearnedPE`; and your own oracle tests in `python/tests/l5-4-pos/` |
| **Contract** | [`course/contracts/py/tinyllm/xfmr/pos.pyi`](../../../course/contracts/py/tinyllm/xfmr/pos.pyi) |
| **Tests** | `course/tests/L5.4/test_pos.py` (what they check: section 4); the oracle is mpmath at 50 digits; your tests are graded by mutation, threshold 0.80 with every pitfall fault required |
| **Needs** | `M00.2` `rotate_pairs` · `M00.3` `geometric` · `M07.3` `normal_init` · `M06.3` `PCG32` · `L0.1` `Tensor` · `L0.2` `embedding` · `L0.4` `Module` (or `--ref-deps`) |
| **Used by** | `L5.5` the transformer's sinusoidal positions · `L6.1` GPT-2's learned `wpe` · `L6.2` BERT's learned position embeddings  · `L14.3` Whisper decoding |
| **Milestone** | `MS-L5` (the addition model reads digit positions) |
| **Optional depth** | Vaswani et al. (2017), section 3.5; Kazemnejad, "Transformer Architecture: The Positional Encoding" (2019); Su et al., "RoFormer" (2021), for where the rotation goes next |

## Key Takeaways

- Attention sees its keys as a set, so position must be added to the input: $x_p = e(\text{token}_p) + \text{PE}(p)$ (`test_sinusoidal_module_has_no_parameters`).
- Pair $i$ of $\text{PE}(p)$ is the point $(\sin p\,\omega_i, \cos p\,\omega_i)$ with $\omega_i = 10000^{-2i/d}$, a geometric ladder from 1 radian per position down to about $10^{-4}$ (`test_hand_example_table`, `test_golden_mpmath`).
- Moving $k$ positions is a fixed rotation of every pair, the same for every $p$ (`test_shift_is_a_fixed_rotation`), so $\text{PE}(p) \cdot \text{PE}(p + k)$ depends only on $k$ (`test_dot_product_depends_only_on_the_offset`).
- Angles must be computed in float64: at $p = 10^4$ a float32 angle is already off in the fourth decimal (`test_golden_mpmath`).
- A learned table is just an embedding indexed by position: it adds rows `offset ..` and only those rows get gradient (`test_learned_pe_adds_rows_and_gets_their_gradient`).

## How to work this chapter

```bash
ol start L5.4              # stubs pos.py; prints your test path and rung (R5)
ol tests L5.4              # the course tests
ol check L5.4              # course tests and the mutation grade of your tests
ol diff  L5.4              # after passing: your code against the reference
```

---

## 1. Why now

Your multi-head attention (`L5.3`) gives the same output for "321+654" and "123+456" shuffled into any order of keys: permuting the keys changes nothing (`L5.3`'s `test_cross_attention_reads_keys_from_x_kv` proves it). For addition that is fatal, because which digit is the ones digit is all a matter of position. The RNNs of Parts 3 and 4 knew order for free, by reading one token at a time; the transformer reads all tokens at once and has to be told. This module builds the two ways the course uses: the fixed sinusoidal table of the 2017 transformer (`L5.5`), whose shift property is the seed of RoPE in Part 7, and the learned table of GPT-2 (`L6.1`) and BERT (`L6.2`).

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $p$ | position, counted from 0 | `int` |
| $d$ | model width, even | `int` |
| $i$ | pair index, $0 \le i < d/2$ | `int` |
| $b$ | base, 10000 in the paper | `float` |
| $\omega_i = b^{-2i/d}$ | angular frequency of pair $i$ (radians per position) | `float64[d/2]` |
| $\text{PE}(p)$ | the encoding of position $p$ | `float32[d]` |
| $R(\theta)$ | the 2D rotation by $\theta$ (`M00.2`) | `float64[2, 2]` |

### 2.1 The sinusoidal table

$$\text{PE}(p)_{2i} = \sin(p\,\omega_i), \qquad \text{PE}(p)_{2i+1} = \cos(p\,\omega_i), \qquad \omega_i = b^{-2i/d}.$$

The frequencies form a geometric sequence with first term 1 and ratio $b^{-2/d}$ (`M00.3`'s `geometric`). Pair 0 turns one radian per position and repeats every $2\pi$ positions; the last pair turns about $b^{-1}$ radians per position and repeats every $2\pi b$, about 63 000 positions for $b = 10^4$. Like the digits of a clock with hands of geometric speeds, the fast pairs tell neighbours apart and the slow ones tell far positions apart. Each pair lies on the unit circle, so $\lVert\text{PE}(p)\rVert = \sqrt{d/2}$ for every $p$: unlike adding $p$ itself, the signal never grows with the sequence. The columns are **interleaved**: sin at even columns, cos at odd ones, the layout of the paper and of `M00.2`'s pairs.

Precision: the angle $p\,\omega_i$ for $p = 10^4$ is about $10^4$ radians. float32 keeps about 7 significant digits, so the angle is off by roughly $10^4 \times 6 \times 10^{-8} \approx 6 \times 10^{-4}$ rad before the sine is even taken. The table is computed in float64 and rounded to float32 at the end.

### 2.2 A shift is a rotation

By the angle-addition formulas (`M00.2`),

$$\begin{pmatrix}\sin((p+k)\omega)\\ \cos((p+k)\omega)\end{pmatrix} = \begin{pmatrix}\cos k\omega & \sin k\omega\\ -\sin k\omega & \cos k\omega\end{pmatrix}\begin{pmatrix}\sin p\omega\\ \cos p\omega\end{pmatrix} = R(-k\omega)\begin{pmatrix}\sin p\omega\\ \cos p\omega\end{pmatrix}.$$

So $\text{PE}(p + k)$ is $\text{PE}(p)$ with pair $i$ turned by $-k\omega_i$: one block-diagonal rotation that depends on $k$ and not on $p$. `shift_pe(pe, k, d)` is `rotate_pairs(pe, -k * omega)`. Two consequences: a linear layer can learn to "look $k$ back" the same way at every position, and $\text{PE}(p) \cdot \text{PE}(p+k) = \sum_i \cos(k\,\omega_i)$ depends only on the distance. RoPE (`L7.3`) applies exactly this rotation to the queries and keys instead of adding it to the input.

### 2.3 Learned positions

GPT-2 and BERT learn a table $P \in \mathbb{R}^{L \times d}$ (`max_len` rows) and add row $p$ at position $p$. It is an embedding indexed by position (`L0.2`'s `embedding`), initialized $\mathcal{N}(0, 0.02^2)$ (`M07.3`), and only the rows actually used get gradient. It can learn any pattern, but it knows nothing about positions past $L$: a model trained on 1024 tokens has no row 1024.

### 2.4 Offsets

Both modules add positions `offset .. offset + T - 1` to an input of length $T$. During training `offset` is 0. When decoding one token at a time with a cache (`L8.2`), the new token sits at position `offset` = the number of tokens before it, and must get that row, not row 0.

## 3. Worked example by hand

$d = 4$, $b = 10000$: $\omega = (10000^0, 10000^{-1/2}) = (1, 0.01)$.

| $p$ | $\sin p$ | $\cos p$ | $\sin 0.01p$ | $\cos 0.01p$ |
|---|---|---|---|---|
| 0 | 0 | 1 | 0 | 1 |
| 1 | 0.841471 | 0.540302 | 0.010000 | 0.999950 |
| 2 | 0.909297 | -0.416147 | 0.019999 | 0.999800 |

Shift check, pair 0, $k = 1$: $R(-1)$ applied to $(\sin 1, \cos 1) = (0.841471, 0.540302)$ gives $(0.841471 \cos 1 + 0.540302 \sin 1,\; -0.841471 \sin 1 + 0.540302 \cos 1) = (0.454649 + 0.454649,\; -0.708073 + 0.291927) = (0.909297, -0.416147)$, the $p = 2$ row. These are `test_hand_example_table` and `test_hand_example_shift`.

## 4. The interface

```python
def sinusoidal_freqs(d: int, base: float = 10000.0) -> NDArray                   # float64 [d/2]
def sinusoidal_pe(T: int, d: int, base: float = 10000.0, offset: int = 0) -> NDArray   # float32 [T, d]
def shift_pe(pe, k: float, d: int, base: float = 10000.0) -> NDArray
class SinusoidalPE(Module):   # no parameters; forward(x, offset=0) = x + table[offset : offset + T]
class LearnedPE(Module):      # weight [max_len, d]; positions(T, offset); forward(x, offset=0)
```

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_table` | unit | section 3: frequencies (1, 0.01) and the three rows | you and the test agree on layout and ladder |
| `test_hand_example_shift` | unit | turning PE(1) by $-\omega$ gives PE(2) | the rotation's sign |
| `test_golden_mpmath` | golden | the table at positions up to 65535 for two settings against 50-digit mpmath | long contexts keep exact positions |
| `test_shift_is_a_fixed_rotation` | property | one $R(k)$ for every $p$, negative $k$ too | the relative-position property RoPE builds on |
| `test_dot_product_depends_only_on_the_offset` | property | $\text{PE}(p)\cdot\text{PE}(p+k) = \sum_i \cos k\omega_i$ | similarity by distance |
| `test_every_pair_is_on_the_unit_circle` | property | each pair has norm 1 | the signal does not grow with $p$ |
| `test_offset_continues_the_table` | property | `offset` gives the tail of a longer table; the module honours it | incremental decoding (`L8.2`) |
| `test_sinusoidal_module_has_no_parameters` | unit | empty state_dict; gradient passes through to $x$ | nothing to train or checkpoint |
| `test_learned_pe_adds_rows_and_gets_their_gradient` | unit | rows `offset..` added; only those get gradient | GPT-2's and BERT's tables train |
| `test_learned_pe_init` | statistical | seeded $\mathcal{N}(0, \text{std}^2)$, default 0.02 | the same init in every language |
| `test_validation` | boundary | odd $d$, base $\le 1$, positions past the table | no silent wrap-around |

### Your graded tests (rung R5)

Your oracle is the formula itself: compute $\sin(p / b^{2i/d})$ and $\cos(\cdot)$ with Python's `math` in float64 for a few positions, including large ones ($p \ge 10^4$), and compare with `sinusoidal_pe`. Test the shift law, the offset, and the learned table's rows and gradient. `ol check L5.4` requires 0.80 with the pitfall faults killed.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. sin and cos concatenated (or swapped) instead of interleaved | the shift law fails; checkpoints from other code disagree | `test_hand_example_table`, `test_golden_mpmath` (mutants `s01`, `s03`) |
| 2. exponent $i/d$ instead of $2i/d$ | the ladder ends at $b^{-1/2}$: far positions alias | `test_golden_mpmath` (mutant `s02`) |
| 3. angles in float32 | rows past a few thousand off in the 4th decimal | `test_golden_mpmath` (mutant `s07`) |
| 4. the shift turned the wrong way, or by the same angle for every pair | `shift_pe` goes back in position, or scrambles pairs | `test_hand_example_shift`, `test_shift_is_a_fixed_rotation` (mutants `s04`, `s06`) |
| 5. positions counted from 1 | every row one late; HF and torch disagree | `test_hand_example_table` (mutant `s05`) |
| 6. `offset` ignored | incremental decoding gives every new token position 0 | `test_offset_continues_the_table`, `test_learned_pe_adds_rows_and_gets_their_gradient` (mutants `s08`, `s10`) |
| 7. the fixed table registered as a parameter | it trains and lands in the checkpoint | `test_sinusoidal_module_has_no_parameters` (mutant `s09`) |
| 8. learned rows read as constants | `wpe` never learns | `test_learned_pe_adds_rows_and_gets_their_gradient` (mutant `s11`) |
| `std` ignored by the learned table | every table starts at 0.02 | `test_learned_pe_init` (mutant `s12`) |
| an odd width accepted | the last feature has no partner | `test_validation` (mutant `s13`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M00.3` | `geometric(1, b^{-2/d}, d/2)` is the frequency ladder |
| Back | `M00.2` | `rotate_pairs` is the shift |
| Back | `M07.3` | `normal_init` for the learned table |
| Back | `M06.3` | `PCG32`, the init stream |
| Back | `L0.2` | `embedding` reads learned rows with their gradient |
| Back | `L0.1` | the `Tensor` the table is added to |
| Back | `L0.4` | `Module` for both encodings |
| Forward | `L14.3` | Whisper decoding calls this module through its declared contract. |
| Forward | `L5.5` | `SinusoidalPE` on both the source and the target embeddings |
| Forward | `L6.1` | `LearnedPE` is GPT-2's `wpe` |
| Forward | `L6.2` | `LearnedPE` is BERT's position embedding |
| Forward | `L7.3` | RoPE rotates queries and keys by the same ladder instead |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `sinusoidal_pe` | fairseq `SinusoidalPositionalEmbedding` | the half layout (all sin, then all cos) and a padding offset | `fairseq/modules/sinusoidal_positional_embedding.py` |
| `shift_pe` | RoPE | the rotation applied inside attention, so scores depend on $p - q$ only | `L7.3`; Su et al. 2021 |
| `LearnedPE` | HF `GPT2Model.wpe`, `BertEmbeddings.position_embeddings` | absolute positions capped at `n_positions` | `transformers/models/gpt2/modeling_gpt2.py` |
| no position at all | ALiBi, NoPE | a distance penalty on scores, or causality alone, for length extrapolation | Press et al. 2022; Kazemnejad et al. 2023 |
