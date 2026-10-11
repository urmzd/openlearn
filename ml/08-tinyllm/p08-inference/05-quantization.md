<!-- ol:module L8.5 -->
# Quantization: int8, int4 group (packed), fp8, KV quant

## Overview

| | |
|---|---|
| **Module** | `L8.5` · build · Python · Pass 6 · 5 to 7 h |
| **You build** | `python/tinyllm/infer/quant.py`: `quantize_int8_per_channel`, `pack_int4`, `unpack_int4`, `quantize_int4_group`, `quantize_fp8_per_channel`, `quantize_mx`, `quantize_kv_fp8`, `dequantize`, `nbytes`, `QuantLinear`, `quantize_model`, `export_q4`, `output_error_bound`, `quant_ppl`, and the dataclasses `Q8Tensor`, `Q4Tensor`, `FP8Tensor`, `MXTensor`, `KVQuant` |
| **Contract** | [`course/contracts/py/tinyllm/infer/quant.pyi`](../../../course/contracts/py/tinyllm/infer/quant.pyi); the int4 bytes in [`formats/safetensors.md`](../../../course/contracts/formats/safetensors.md); KV format v2 in [`formats/kv-block.md`](../../../course/contracts/formats/kv-block.md) |
| **Tests** | `course/tests/L8.5/test_quant.py`, 17 tests (what they check: section 4) · golden `course/fixtures/parity/quant_int4.json` · your own tests in `python/tests/l8-5-quant/`, rung R4, graded by mutation (threshold 0.80) · parity suite `ol parity quant.int4` |
| **Needs** | `M09.4` [FP8 and microscaling formats](../../../math/09-numerical-methods-and-floating-point/04-fp8-and-microscaling-formats.md) (`tinyllm.num.lowp`) · `L0.4` [modules and `Linear`](../p00-foundations/04-module-system-and-layers.md) · `L0.1` [the `Tensor`](../p00-foundations/01-tensor-and-broadcasting-backward.md) · `M09.3` [tolerance budgets](../../../math/09-numerical-methods-and-floating-point/03-error-analysis-condition-numbers-and-tolerance-budgets.md) (`matmul_error_bound`) · `L6.7` [the evaluation harness](../p06-objectives/07-lm-evaluation-harness-and-the-model-zoo.md) (`eval_ppl`) · reading: `M09.1` [IEEE 754 and float16](../../../math/09-numerical-methods-and-floating-point/01-ieee-754.md), `M07.4` [confidence intervals](../../../math/07-probability-statistics/04-lln-clt-confidence-intervals-bootstrap.md) |
| **Used by** | later: `L9.5` (the C int4 and int8 kernels read these bytes), `L10.1` (the Rust runner loads `*.q4.safetensors`), `craft.13` (KV format v2 uses `quantize_kv_fp8` as its oracle) |
| **Milestone** | `MS-L8` (step 2: perplexity under `--quant int8, q4_g32, fp8_e4m3` within budget) |
| **Optional depth** | Dettmers et al., [*LLM.int8()*](https://arxiv.org/abs/2208.07339); Frantar et al., [*GPTQ*](https://arxiv.org/abs/2210.17323); Lin et al., [*AWQ*](https://arxiv.org/abs/2306.00978); OCP, [*Microscaling Formats (MX) v1.0*](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) |

## Key Takeaways

- Symmetric quantization stores small integers and one scale per row, group, or head; the scale maps the largest magnitude onto the largest code, so nothing clips and every element is within half a step (`test_int4_error_is_at_most_half_a_step`).
- int4 packs two weights per byte: the even column in the low nibble, each a 4-bit two's complement value, exactly the bytes the C kernel reads (`test_hand_example_int4_group`, `test_packed_bytes_match_the_golden`).
- The scale is stored as float16, so the codes must be chosen against the float16 value actually stored, not the float32 one computed (`test_int4_dequantizes_with_the_stored_scale`).
- Decoding is memory-bound, so 4.25 bits per weight instead of 32 is up to 7.5x less to read per token; each scheme has an error budget the model's output must stay within (`test_quantize_model_within_budget`).
- The KV cache quantizes the same way: one fp8 scale per head, the layout of KV format v2 (`test_kv_fp8_follows_format_v2`).

## How to work this chapter

```bash
ol start L8.5               # stubs python/tinyllm/infer/quant.py
ol tests L8.5               # read the test catalog first
ol check L8.5               # exit code is the verdict; then grades your tests by mutation
ol parity quant.int4        # your packed bytes against the golden file
ol diff  L8.5               # after passing: your code against the reference
```

Do int8 first (one scale per row, no packing), then `pack_int4`/`unpack_int4` on the 256 pairs of section 4, then `quantize_int4_group`, then fp8, MX, and KV through `M09.4`, then `QuantLinear` and `quantize_model`.

---

## 1. Why now

Your inference stack decodes one token at a time, and each token reads every weight of the model once (`L8.2`): decoding is limited by bytes moved, not by arithmetic. SmolLM2-135M is 540 MB in float32, and reading those bytes is most of what a token costs. Storing weights in 8, 4, or fewer bits cuts those bytes by 4 to 7.5 times, and the same applies to the KV cache, which grows with every sequence. This module turns float weights into compact codes and back, in the exact byte layout your C kernel (`L9.5`) will multiply without ever expanding the weights, and measures how much accuracy each scheme costs.

## 2. Principles

| Symbol | Meaning | Type |
|---|---|---|
| $W$ | a Linear weight `[out, in]` | float32 |
| $s$ | a scale (per row, per group, or per head) | float32 or float16 |
| $q$ | a code: an integer (int8, int4) or a low-precision bit pattern (fp8, fp4) | int8, uint8 |
| $g$ | int4 group size: consecutive columns of a row sharing one scale | `int` |
| $\text{amax}$ | the largest magnitude in a row, group, or head | float32 |
| $\text{rint}$ | rounding to the nearest integer, ties to even | |

**Symmetric scaling.** Choose $s = \text{amax} / q_{\max}$, store $q = \text{rint}(W / s)$, and read back $\hat W = q \cdot s$. Then $|W| \le \text{amax}$ means $|W / s| \le q_{\max}$, nothing clips, and $|W - \hat W| \le s / 2$: rounding moves each value at most half a step. There is no zero point, so 0 maps to 0 exactly. Rounding is ties-to-even everywhere (`np.rint`), and every quotient $W / s$ is formed in float32 before rounding, so two implementations of the rule produce the same codes.

**int8 per output channel.** One float32 scale per row, $q_{\max} = 127$: codes in $[-127, 127]$ (symmetric, so $-128$ is never used). A zero row gets $s = 0$ and codes 0.

**int4 per group.** One scale per $g$ consecutive columns (32 by default), $q_{\max} = 7$, codes in $[-8, 7]$. A per-row scale would let one outlier column flatten the other columns of its row to a few codes; a group of 32 confines the damage. The scale is stored as **float16** (`formats/safetensors.md`), so compute $s_{16} = \text{float16}(\text{amax} / 7)$ first and quantize against $s_{16}$: rounding the scale can make $\text{amax} / s_{16}$ slightly above 7, at most $7 (1 + 2^{-11}) < 7.5$, which still rounds to 7 and keeps the half-step bound. Quantizing against the float32 scale and storing the float16 one breaks the bound and disagrees with the kernel. A group whose scale is 0 (all zeros, or values so small $\text{amax}/7$ underflows float16) gets codes 0; a scale that overflows float16 is an error.

**Packing.** Two int4 codes per byte: byte $b$ of a row holds column $2b$ in its low nibble and $2b + 1$ in its high nibble, each as 4-bit two's complement (`q & 0xF`, so $-1$ is `0xF`, $-8$ is `0x8`). Unpacking sign-extends: a nibble $v \ge 8$ means $v - 16$.

**FP8 per channel.** E4M3 codes through `M09.4`'s `quantize_fp8` with one float32 scale per row, $s = \text{amax} / 448$ (448 is E4M3's largest finite value). E4M3 keeps 3 mantissa bits, so a value at least $2^{-6}$ of the scale lands within a relative $2^{-4}$ of itself.

**MXFP4.** `M09.4`'s `mx_quantize`: blocks of 32 values share one power-of-two E8M0 scale $X$, elements are FP4 E2M1 codes ($0, 0.5, 1, 1.5, 2, 3, 4, 6$ times $X$). The largest gap is from $4X$ to $6X$, and values above $6X$ saturate, so every element is within $2X$; storage is $4 + 8/32 = 4.25$ bits per weight.

**KV fp8 (format v2).** A K or V slab `[n_kv_heads, T, d_head]` gets one float32 scale per head, $s = \text{float32}(\text{amax}) / 448$ (1.0 for a head of zeros), and E4M3 codes of $x / s$: the payload of KV format v2, which `craft.13` migrates the C pool to.

**Quantized layers.** `QuantLinear(q, bias)` holds a quantized weight, dequantizes it once, and computes `x @ W^T + b` on `L0.1` Tensors; it has no parameters (it is for inference). `quantize_model(model, scheme)` replaces every `L0.4` `Linear` except those named in `skip` (default `lm_head`, which a model often ties to its embeddings) in place. `export_q4` writes the tensors of a `*.q4.safetensors` file for the Rust runner (`L10.1`).

**The error budget.** A quantized layer computes $x \hat W^T$ instead of $x W^T$. Each output is a dot product of $k$ = `in` terms, so `M09.3`'s bound applies with the weight perturbed by $\Delta = |W - \hat W|$ (at most $s/2$ per element for int8 and int4): $|y' - y| \le \Delta |x|^T + \gamma_k (|W| + \Delta) |x|^T$ elementwise, in the format the product ran in. `output_error_bound(w, q, x)` returns exactly `M09.3`'s `matmul_error_bound(w, x^T, dtype, dA=Δ)^T`. The first term is the quantization, the second the float32 rounding; for every scheme here the first dominates by three orders of magnitude, which is why a test with rounding-only tolerances fails a correct quantizer.

**The perplexity cost.** A layer's error budget does not say what a model loses. `quant_ppl(model, scheme, ids, ctx_len, stride)` measures it: `L6.7`'s `eval_ppl` on the float model, then on a quantized **copy** (`quantize_model` works in place, and the caller still needs the float model), and returns `ppl`, `ppl_quant`, their difference `delta`, and `ratio`. MS-L8's budgets read these numbers.

## 3. Worked example by hand

int4 with one group of 4 per row:

| | Row 0 | Row 1 |
|---|---|---|
| $W$ | `1.75 -0.6 0.1 0.3` | `0.375 -0.125 0.625 1.75` |
| amax, $s_{16} = \text{float16}(\text{amax}/7)$ | 1.75, 0.25 (exact) | 1.75, 0.25 |
| $W / s$ | `7 -2.4 0.4 1.2` | `1.5 -0.5 2.5 7` |
| $q = \text{rint}$ | `7 -2 0 1` | `2 0 2 7` (ties go to even: 1.5 to 2, $-0.5$ to 0, 2.5 to 2) |
| nibbles (`q & 0xF`) | `7 E 0 1` | `2 0 2 7` |
| bytes (even column low) | `E7 10` | `02 72` |
| $\hat W = q \cdot s$ | `1.75 -0.5 0 0.25` | `0.5 0 0.5 1.75` |

Every error is at most $s / 2 = 0.125$ (the largest, 0.125, is the tie $0.375 \to 0.5$). This is `test_hand_example_int4_group`.

int8, scale 1 (row amax 127): `127 -3.5 2.5 0.4` becomes `127 -4 2 0` ($-3.5$ and $2.5$ are ties, to the even $-4$ and $2$). A row of zeros has scale 0 and codes 0. This is `test_hand_example_int8_per_channel`.

## 4. The interface

```python
@dataclass class Q8Tensor:  q: NDArray; scales: NDArray                  # int8 [out, in], float32 [out]
@dataclass class Q4Tensor:  packed: NDArray; scales: NDArray; group: int; shape: tuple[int, int]
@dataclass class FP8Tensor: codes: NDArray; scales: NDArray; fmt: str    # uint8 [out, in], float32 [out]
@dataclass class MXTensor:  codes: NDArray; scales: NDArray; block: int; elem: str
@dataclass class KVQuant:   codes: NDArray; scales: NDArray              # uint8 [H, T, D], float32 [H]
def quantize_int8_per_channel(w) -> tuple[NDArray, NDArray]
def pack_int4(q) -> NDArray;  def unpack_int4(packed) -> NDArray
def quantize_int4_group(w, group: int = 32) -> Q4Tensor
def quantize_fp8_per_channel(w, fmt="e4m3") -> FP8Tensor
def quantize_mx(w, block=32, elem="fp4_e2m1") -> MXTensor
def quantize_kv_fp8(x) -> KVQuant
def dequantize(q) -> NDArray            # float32, any form above (an int8 (q, scales) tuple too)
def nbytes(q) -> int
class QuantLinear(Module): def __init__(self, q, bias=None); def forward(self, x: Tensor) -> Tensor
def quantize_model(model: Module, scheme: str, skip=("lm_head",)) -> Module   # "int8" "q4_g32" "fp8_e4m3" "mxfp4"
def export_q4(model: Module) -> tuple[dict[str, NDArray], dict[str, str]]
def output_error_bound(w, q, x, dtype: str = "f32") -> NDArray      # [n, out], M09.3's matmul_error_bound
def quant_ppl(model, scheme, ids, ctx_len, stride, skip=("lm_head",)) -> dict[str, float]  # L6.7's eval_ppl
```

The design sketch's optional `awq_search_scales` and `gptq_quantize` are not part of this module; section "Going further" points at them.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_int4_group` | unit | section 3: codes, scales, bytes `E7 10 02 72`, dequantized values | you, the tests, and the C kernel agree on the layout |
| `test_hand_example_int8_per_channel` | unit | section 3's int8 row, a zero row, ties at scale 2, codes within $[-127, 127]$ | L9.5's `tl_matmul_q8_f32` input |
| `test_packed_bytes_match_the_golden` | golden | every case of the `quant.int4` parity golden (1 x 2 at group 2 to 5 x 48 at group 16 and 2 x 64 at group 64) byte for byte | the bytes `ol parity quant.int4` and L9.5 compare |
| `test_nibble_roundtrip_every_pair` | property | all 256 pairs pack to distinct bytes and unpack to themselves; out-of-range and odd widths refused | two's complement, even column low |
| `test_int4_error_is_at_most_half_a_step` | property | every element within $s_{16}/2$ at scales 1e-3 to 100, groups 16 to 64 | the scheme's promise |
| `test_int4_dequantizes_with_the_stored_scale` | unit | dequantize is code times float16 scale exactly; codes are $\text{rint}(W / s_{16})$, including a hand-made tie at 3.5 stored scales | quantize against what is stored |
| `test_zero_and_tiny_groups` | boundary | zero and underflowing groups give zeros, no NaN; a float16 overflow raises | pruned weights and outliers |
| `test_shape_rules` | boundary | groups that do not divide the width, odd groups, 1-D input, NaN, a non-quantized argument | bytes the kernel would misread are never written |
| `test_int8_error_bound_and_range` | property | half-step bound per row; each row's amax maps to $\pm127$ | per-channel, not per-tensor |
| `test_fp8_per_channel_relative_error` | property | scale = amax / 448 per row; relative error $\le 2^{-4}$ above $2^{-6}$ of the scale; amax reproduced | fp8 weights without saturation |
| `test_mxfp4_blocks` | property | one E8M0 scale per 32 values; error within $2X$; 4.25 bits per weight | MX weights through M09.4 |
| `test_kv_fp8_follows_format_v2` | unit | per-head scale float32(amax)/448, 1.0 for a zero head; relative error bound | craft.13's oracle for the C writer |
| `test_quant_linear_is_the_dequantized_matmul` | differential | `QuantLinear(x)` equals `x @ dequantize(q)^T + b` for int4, int8, fp8; no parameters | L10.1's q4 runner is checked against it |
| `test_quantize_model_within_budget` | property | every Linear but `lm_head` swapped; logits within the scheme's relative error budget (int8 0.02, q4_g32 0.2, fp8 0.08, mxfp4 0.35) | the stand-in for MS-L8's perplexity budgets |
| `test_export_q4_names_and_metadata` | unit | `<name>.qweight` uint8, `<name>.scales` float16, biases and `lm_head` kept, metadata `int4-g32-sym` | the file the Rust runner loads |
| `test_output_error_stays_within_the_budget` | property | for int8, int4, fp8, and mxfp4, every output of `QuantLinear` is within `output_error_bound`, which uses more than 1% of itself | `M09.3`'s budget turned into a test tolerance that a wrong quantizer cannot pass |
| `test_quant_ppl_reports_the_cost_and_keeps_the_model` | property | `eval_ppl` before and after; int8 costs less than q4; the caller's model stays float | MS-L8's perplexity budgets |

**Your tests (rung R4).** Write `python/tests/l8-5-quant/test_*.py`, importing only names from `contracts/py`, with properties in prose turned into code (Hypothesis is allowed): "every element is within half its group's stored scale", "pack then unpack is the identity on $[-8, 7]$", "the even column is the low nibble", "QuantLinear equals the dequantized matmul". The planted bugs "the nibbles are swapped" (`s01`) and "codes are chosen against the float32 scale" (`s07`) are required; overall 80%.

**Parity.** `ol parity quant.int4` runs your `quantize_int4_group` through `course/conformance/parity/drivers/quant_int4.py` on the golden cases; `L9.5`'s C side joins the same suite.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| Even column in the high nibble | the C kernel multiplies every pair swapped | `test_hand_example_int4_group`, `test_nibble_roundtrip_every_pair` (mutant `s01`) |
| Offset binary (code + 8) instead of two's complement | every code off by 8 for the kernel | `test_hand_example_int4_group`, `test_nibble_roundtrip_every_pair` (mutant `s02`) |
| Unpacking without sign extension | negative weights read as 9 to 15 | `test_hand_example_int4_group`, `test_nibble_roundtrip_every_pair` (mutant `s03`) |
| Truncating instead of rounding | errors up to a whole step, biased toward zero | `test_hand_example_int4_group`, `test_int4_error_is_at_most_half_a_step` (mutant `s04`) |
| Casting int8 codes without rounding | the same truncation for int8 | `test_hand_example_int8_per_channel` (mutant `s05`) |
| int8 with scale amax / 128 | the row maximum clips | `test_hand_example_int8_per_channel` (mutant `s06`) |
| Quantizing against the float32 scale, storing float16 | a weight of exactly 3.5 stored scales rounds to 3 instead of 4; the kernel's products disagree | `test_int4_dequantizes_with_the_stored_scale` (mutant `s07`) |
| Scale amax / 8 | the largest weight clips at 7 | `test_int4_error_is_at_most_half_a_step` (mutant `s08`) |
| Writing an overflowing float16 scale | `inf` in the file, NaN in every product | `test_zero_and_tiny_groups` (mutant `s09`) |
| Accepting an odd group | a group splits a byte between two scales | `test_shape_rules` (mutant `s10`) |
| One int8 scale per tensor | small rows lose all precision | `test_int8_error_bound_and_range` (mutant `s11`) |
| fp8 scale ignoring the format's maximum | wasted range or saturation | `test_fp8_per_channel_relative_error` (mutant `s12`) |
| MX block size ignored | scales do not line up with the format's blocks | `test_mxfp4_blocks` (mutant `s13`) |
| One KV scale for all heads | quiet heads lose their precision | `test_kv_fp8_follows_format_v2` (mutant `s14`) |
| A zero KV head with scale 0 | division by zero in `quantize_fp8` | `test_kv_fp8_follows_format_v2` (mutant `s15`) |
| `QuantLinear` dropping the bias | every output shifted | `test_quant_linear_is_the_dequantized_matmul` (mutant `s16`) |
| Multiplying by $W$ instead of $W^T$ | shapes fail, or square layers compute garbage | `test_quant_linear_is_the_dequantized_matmul` (mutant `s17`) |
| Quantizing `lm_head` | tied embeddings change; logits lose the most precision | `test_quantize_model_within_budget` (mutant `s18`) |
| `quantize_model` losing biases | the model's output drifts past the budget | `test_quantize_model_within_budget` (mutant `s19`) |
| Wrong tensor names in the export | the Rust runner cannot find the weights | `test_export_q4_names_and_metadata` (mutant `s20`) |
| Repeating group scales along the wrong axis | shapes fail on dequantize | `test_hand_example_int4_group` (mutant `m01`) |
| Counting a byte per fp4 code | memory budgets double-count MX weights | `test_mxfp4_blocks` (mutant `m02`) |
| A budget with rounding only ($\Delta = 0$) | every quantized output fails its tolerance by a factor of a thousand | `test_output_error_stays_within_the_budget` (mutant `s21`) |
| Quantizing the caller's model in `quant_ppl` | the float model the caller compares against is gone | `test_quant_ppl_reports_the_cost_and_keeps_the_model` (mutant `s22`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M09.4` | `quantize_fp8`, `dequantize_fp8`, `fp8_max`, `mx_quantize`, `mx_dequantize` |
| Back | `L0.4` | `QuantLinear` is a `Module`; `quantize_model` walks `named_modules` and finds `Linear`s |
| Back | `L0.1` | `QuantLinear.forward` runs on `Tensor` |
| Back | `M09.3` | `output_error_bound` is `matmul_error_bound` with the quantization step as `dA` |
| Back | `L6.7` | `quant_ppl` runs `eval_ppl` on the float model and on its quantized copy |
| Forward | `L9.5` | `tl_matmul_q4_f32` and `tl_matmul_q8_f32` multiply these bytes without expanding them; `parity/quant.int4` |
| Forward | `L10.1` | the Rust runner loads `export_q4`'s tensors and checks its logits against `QuantLinear` |
| Forward | `craft.13` | KV format v2's fp8 payload follows `quantize_kv_fp8` |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| int4 group, symmetric | llama.cpp `Q4_0` and K-quants | 32-weight blocks with float16 scales, mixed precision per tensor (`Q4_K_M`) | [`ggml/src/ggml-quants.c`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c) |
| round to nearest | GPTQ, AWQ | second-order error compensation (GPTQ) or activation-aware scaling of salient channels (AWQ) at the same bit width | [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ), [llm-awq](https://github.com/mit-han-lab/llm-awq) |
| fp8 per channel | vLLM FP8 W8A8 | fp8 activations too, with per-token dynamic scales and fused kernels | [`vllm/model_executor/layers/quantization/fp8.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/fp8.py) |
| KV fp8 | vLLM `kv_cache_dtype="fp8"` | per-tensor or per-head KV scales calibrated offline | vLLM docs, "Quantized KV Cache" |
