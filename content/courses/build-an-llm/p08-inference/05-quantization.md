# Quantization: int8, int4 group (packed), fp8, KV quant

## Overview

| | |
|---|---|
| **Module** | `L8.5` · build · Python · Pass 6 |
| **Solution** | [`quant.py`](../../../../course/solutions/python/tinyllm/infer/quant.py): `quantize_int8_per_channel`, `pack_int4`, `unpack_int4`, `quantize_int4_group`, `quantize_fp8_per_channel`, `quantize_mx`, `quantize_kv_fp8`, `dequantize`, `nbytes`, `QuantLinear`, `quantize_model`, `export_q4`, `output_error_bound`, `quant_ppl`, and the dataclasses `Q8Tensor`, `Q4Tensor`, `FP8Tensor`, `MXTensor`, `KVQuant` |
| **Contract** | [`course/contracts/py/tinyllm/infer/quant.pyi`](../../../../course/contracts/py/tinyllm/infer/quant.pyi); the int4 bytes in [`formats/safetensors.md`](../../../../course/contracts/formats/safetensors.md); KV format v2 in [`formats/kv-block.md`](../../../../course/contracts/formats/kv-block.md) |
| **Needs** | `M09.4` [FP8 and microscaling formats](../../../tracks/math/09-numerical-methods-and-floating-point/04-fp8-and-microscaling-formats.md) (`tinyllm.num.lowp`) · `L0.4` [modules and `Linear`](../p00-foundations/04-module-system-and-layers.md) · `L0.1` [the `Tensor`](../p00-foundations/01-tensor-and-broadcasting-backward.md) · `M09.3` [tolerance budgets](../../../tracks/math/09-numerical-methods-and-floating-point/03-error-analysis-condition-numbers-and-tolerance-budgets.md) (`matmul_error_bound`) · `L6.7` [the evaluation harness](../p06-objectives/07-lm-evaluation-harness-and-the-model-zoo.md) (`eval_ppl`) · reading: `M09.1` [IEEE 754 and float16](../../../tracks/math/09-numerical-methods-and-floating-point/01-ieee-754.md), `M07.4` [confidence intervals](../../../tracks/math/07-probability-statistics/04-lln-clt-confidence-intervals-bootstrap.md) |
| **Used by** | later: `L9.5` (the C int4 and int8 kernels read these bytes), `L10.1` (the Rust runner loads `*.q4.safetensors`), `craft.13` (KV format v2 uses `quantize_kv_fp8` as its oracle); the pass's end-to-end run reports perplexity under `int8`, `q4_g32`, and `fp8_e4m3` |
| **Optional depth** | Dettmers et al., [*LLM.int8()*](https://arxiv.org/abs/2208.07339); Frantar et al., [*GPTQ*](https://arxiv.org/abs/2210.17323); Lin et al., [*AWQ*](https://arxiv.org/abs/2306.00978); OCP, [*Microscaling Formats (MX) v1.0*](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) |

## Key Takeaways

- Symmetric quantization stores small integers and one scale per row, group, or head; the scale maps the largest magnitude onto the largest code, so nothing clips and every element is within half a step.
- int4 packs two weights per byte: the even column in the low nibble, each a 4-bit two's complement value, exactly the bytes the C kernel reads.
- The scale is stored as float16, so the codes must be chosen against the float16 value actually stored, not the float32 one computed.
- Decoding is memory-bound, so 4.25 bits per weight instead of 32 is up to 7.5x less to read per token; each scheme has an error budget the model's output must stay within.
- The KV cache quantizes the same way: one fp8 scale per head, the layout of KV format v2.

## How to read this chapter

Sections 1 to 3 build symmetric quantization from first principles and work int4 and int8 by hand. Section 4 fixes the interface. Section 5 walks through the solution in the order it is best understood: int8 (one scale per row, no packing), the int4 nibble layout, int4 groups against a float16 scale, fp8, MX and KV through `M09.4`, then the layer swap and the error budget. Try to predict each step before reading it.

---

## 1. Why now

The inference stack decodes one token at a time, and each token reads every weight of the model once (`L8.2`): decoding is limited by bytes moved, not by arithmetic. SmolLM2-135M is 540 MB in float32, and reading those bytes is most of what a token costs. Storing weights in 8, 4, or fewer bits cuts those bytes by 4 to 7.5 times, and the same applies to the KV cache, which grows with every sequence. This module turns float weights into compact codes and back, in the exact byte layout the C kernel (`L9.5`) will multiply without ever expanding the weights, and measures how much accuracy each scheme costs.

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

**The perplexity cost.** A layer's error budget does not say what a model loses. `quant_ppl(model, scheme, ids, ctx_len, stride)` measures it: `L6.7`'s `eval_ppl` on the float model, then on a quantized **copy** (`quantize_model` works in place, and the caller still needs the float model), and returns `ppl`, `ppl_quant`, their difference `delta`, and `ratio`. The pass's end-to-end perplexity budgets read these numbers.

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

Every error is at most $s / 2 = 0.125$ (the largest, 0.125, is the tie $0.375 \to 0.5$).

int8, scale 1 (row amax 127): `127 -3.5 2.5 0.4` becomes `127 -4 2 0` ($-3.5$ and $2.5$ are ties, to the even $-4$ and $2$). A row of zeros has scale 0 and codes 0.

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

The optional `awq_search_scales` and `gptq_quantize` of an earlier design sketch are not part of this solution; section "Going further" points at them.

### What a correct solution must handle

| Case | Checks | Why it matters downstream |
|---|---|---|
| The int4 hand example | section 3: codes, scales, bytes `E7 10 02 72`, dequantized values | the solution and the C kernel agree on the layout |
| The int8 hand example | section 3's int8 row, a zero row, ties at scale 2, codes within $[-127, 127]$ | L9.5's `tl_matmul_q8_f32` input |
| Packed bytes against the `quant.int4` golden | every recorded case (1 x 2 at group 2 to 5 x 48 at group 16 and 2 x 64 at group 64) byte for byte | the bytes L9.5's C side is compared against |
| Every nibble pair round-trips | all 256 pairs pack to distinct bytes and unpack to themselves; out-of-range and odd widths refused | two's complement, even column low |
| int4 error at most half a step | every element within $s_{16}/2$ at scales 1e-3 to 100, groups 16 to 64 | the scheme's promise |
| int4 dequantizes with the stored scale | dequantize is code times float16 scale exactly; codes are $\text{rint}(W / s_{16})$, including a hand-made tie at 3.5 stored scales | quantize against what is stored |
| Zero and tiny groups | zero and underflowing groups give zeros, no NaN; a float16 overflow raises | pruned weights and outliers |
| Shape rules | groups that do not divide the width, odd groups, 1-D input, NaN, a non-quantized argument | bytes the kernel would misread are never written |
| int8 error bound and range | half-step bound per row; each row's amax maps to $\pm127$ | per-channel, not per-tensor |
| fp8 per-channel relative error | scale = amax / 448 per row; relative error $\le 2^{-4}$ above $2^{-6}$ of the scale; amax reproduced | fp8 weights without saturation |
| MXFP4 blocks | one E8M0 scale per 32 values; error within $2X$; 4.25 bits per weight | MX weights through M09.4 |
| KV fp8 follows format v2 | per-head scale float32(amax)/448, 1.0 for a zero head; relative error bound | craft.13's oracle for the C writer |
| `QuantLinear` is the dequantized matmul | `QuantLinear(x)` equals `x @ dequantize(q)^T + b` for int4, int8, fp8; no parameters | L10.1's q4 runner is checked against it |
| `quantize_model` within budget | every Linear but `lm_head` swapped; logits within the scheme's relative error budget (int8 0.02, q4_g32 0.2, fp8 0.08, mxfp4 0.35) | a fast stand-in for the perplexity budgets |
| `export_q4` names and metadata | `<linear>.weight.qweight` uint8, `<linear>.weight.scales` float16, biases and `lm_head` kept, metadata `int4-g32-sym` | the file the Rust runner loads |
| Output error within the budget | for int8, int4, fp8, and mxfp4, every output of `QuantLinear` is within `output_error_bound`, and uses more than 1% of it | `M09.3`'s budget turned into a tolerance that a wrong quantizer cannot pass |
| `quant_ppl` reports the cost and keeps the model | `eval_ppl` before and after; int8 costs less than q4; the caller's model stays float | the pass's perplexity budgets |

Properties worth stating in your own words: every element is within half its group's stored scale; pack then unpack is the identity on $[-8, 7]$; the even column is the low nibble; `QuantLinear` equals the dequantized matmul.

## 5. The solution, step by step

### From first principles

1. **Decode is bound by bytes, so the goal is fewer bytes per weight with a known error.** Each token reads every weight once; arithmetic is cheap by comparison. Any scheme is judged by bits per weight and by the error it adds to each output.
2. **One scale per set of values, chosen from the set's largest magnitude.** If $s = \text{amax}/q_{\max}$, the largest value lands exactly on the largest code and every other value lands inside the code range: no clipping, so the only error is rounding, at most $s/2$. A smaller $q_{\max}$ (for example $\text{amax}/128$ for int8, or $\text{amax}/8$ for int4) would push amax past the last code and clip it.
3. **Symmetric, with no zero point.** Weights are roughly centred on 0. Without a zero point, 0 is exact, dequantize is a single multiply, and the kernel needs no subtraction. int8 uses $[-127, 127]$, not $-128$, so the range is symmetric.
4. **The granularity of the scale trades bytes against outliers.** One scale per tensor lets a large row crush small rows; per row is enough at 8 bits (255 levels). At 4 bits (16 levels) one outlier column flattens its row, so int4 uses one scale per group of 32 columns, at a cost of 16 bits per 32 weights, $0.5$ bit per weight.
5. **Quantize against the scale that is stored.** The kernel multiplies codes by the float16 scale in the file. If codes were chosen against the float32 scale, a value at exactly 3.5 stored scales could round to 3 against the slightly different float32 scale, and the bound $|W - q s_{16}| \le s_{16}/2$ would fail. So: round the scale to float16 first, then divide by it.
6. **The byte layout is a contract with the kernel.** Two codes per byte, column $2b$ in the low nibble, each 4-bit two's complement. Unpack must sign-extend. The worked example's bytes `E7 10 02 72` are the check.
7. **fp8 and MX reuse `M09.4`; only the scale rule differs.** E4M3 has a floating grid, so the scale maps amax onto 448, the format's largest finite value. MX blocks get a power-of-two shared scale from `mx_quantize`. KV slabs get one fp8 scale per head because heads differ in magnitude.
8. **The error budget is a consequence, not a guess.** A quantized weight is the float weight perturbed by $\Delta = |W - \hat W|$, so `M09.3`'s matmul bound with `dA` $= \Delta$ is a rigorous per-output tolerance.

### The code, annotated

A shared guard turns any input into a finite float32 matrix, so every quantizer starts from the same values:

```python
def _matrix(w: ArrayLike, fn: str) -> NDArray:
    a = np.asarray(w, dtype=np.float32)   # quotients are formed in float32
    if a.ndim != 2:
        raise ValueError(f"{fn}: want a 2-D weight [out, in], got shape {a.shape}")
    if not np.isfinite(a).all():          # a NaN would poison amax and every code
        raise ValueError(f"{fn}: the weight holds a NaN or an infinity")
    return a
```

int8 per output channel, step 2 with $q_{\max} = 127$:

```python
def quantize_int8_per_channel(w: ArrayLike) -> tuple[NDArray, NDArray]:
    a = _matrix(w, "quantize_int8_per_channel")
    # One scale per row: amax maps onto 127, so nothing clips.
    scales = (np.abs(a).max(axis=1) / np.float32(127.0)).astype(np.float32)
    safe = np.where(scales > 0, scales, np.float32(1.0))   # no division by 0
    q = np.rint(a / safe[:, None])                         # ties to even
    q = np.where(scales[:, None] > 0, q, 0.0)              # a zero row: codes 0
    return np.clip(q, -127, 127).astype(np.int8), scales   # clip is a no-op guard
```

Packing and unpacking, step 6:

```python
def pack_int4(q: ArrayLike) -> NDArray:
    v = np.asarray(q)
    if v.ndim != 2 or v.shape[1] % 2:          # an odd width would split a byte
        raise ValueError(f"pack_int4: want [out, in] with in even, got shape {v.shape}")
    if v.size and (v.min() < -8 or v.max() > 7):
        raise ValueError("pack_int4: values must be in [-8, 7]")
    n = (v.astype(np.int16) & 0xF).astype(np.uint8)  # 4-bit two's complement: -1 -> 0xF
    # the even column goes in the low nibble, the odd one in the high nibble
    return (n[:, 0::2] | (n[:, 1::2] << 4)).astype(np.uint8)

def unpack_int4(packed: ArrayLike) -> NDArray:
    p = np.asarray(packed, dtype=np.uint8)
    out = np.empty((p.shape[0], 2 * p.shape[1]), dtype=np.int8)
    for col, nib in ((0, p & 0xF), (1, p >> 4)):
        s = nib.astype(np.int8)
        out[:, col::2] = np.where(s >= 8, s - 16, s)  # sign-extend bit 3
    return out
```

Row 0 of the worked example: codes `7 -2 0 1` become nibbles `7 E 0 1`; bytes are `(E << 4) | 7 = E7` and `(1 << 4) | 0 = 10`.

int4 groups, steps 4 and 5:

```python
def quantize_int4_group(w: ArrayLike, group: int = 32) -> Q4Tensor:
    a = _matrix(w, "quantize_int4_group")
    rows, cols = a.shape
    # Even groups so no byte straddles two scales; groups must tile the row.
    if group < 2 or group % 2 or cols % group:
        raise ValueError(...)
    g = a.reshape(rows, cols // group, group)        # [rows, n_groups, group]
    amax = np.abs(g).max(axis=2)
    with np.errstate(over="ignore"):
        s16 = (amax / np.float32(7.0)).astype(np.float16)  # the scale that is stored
    if not np.isfinite(s16).all():                   # inf in the file = NaN products
        raise ValueError(...)
    s = s16.astype(np.float32)[:, :, None]           # quantize against the stored scale
    # A scale that is 0 (all zeros, or underflowed in float16) gives codes 0.
    q = np.where(s > 0, np.rint(g / np.where(s > 0, s, 1.0)), 0.0)
    # Rounding the scale can make amax / s16 slightly above 7 (below 7.5),
    # which still rounds to 7; the clip only guards the range.
    q = np.clip(q, -8, 7).astype(np.int8).reshape(rows, cols)
    return Q4Tensor(pack_int4(q), s16, int(group), (rows, cols))
```

fp8 per channel and KV fp8, step 7. Both delegate encoding to `M09.4` and only choose the scale; a zero row or head gets scale 1.0, not 0, because `quantize_fp8` divides by it:

```python
def quantize_fp8_per_channel(w, fmt="e4m3") -> FP8Tensor:
    a = _matrix(w, "quantize_fp8_per_channel")
    top = np.float32(fp8_max(fmt))                   # 448 for E4M3
    amax = np.abs(a).max(axis=1)
    scales = np.where(amax > 0, amax / top, np.float32(1.0)).astype(np.float32)
    codes = np.empty(a.shape, dtype=np.uint8)
    for r in range(a.shape[0]):
        codes[r] = quantize_fp8(a[r], fmt, float(scales[r]))
    return FP8Tensor(codes, scales, fmt)

def quantize_kv_fp8(x) -> KVQuant:
    a = np.asarray(x, dtype=np.float32)              # [n_kv_heads, T, d_head]
    # ... shape and finiteness checks ...
    flat = np.abs(a).reshape(a.shape[0], -1)
    amax = flat.max(axis=1) if a.size else np.zeros(a.shape[0], np.float32)  # per head
    e4m3_max = np.float32(448.0)
    scales = np.where(amax > 0, amax / e4m3_max, np.float32(1.0)).astype(np.float32)
    codes = np.empty(a.shape, dtype=np.uint8)
    for h in range(a.shape[0]):
        codes[h] = quantize_fp8(a[h], "e4m3", float(scales[h]))
    return KVQuant(codes, scales)
```

`quantize_mx` is a thin wrapper over `mx_quantize(a, block, elem)` that stores the codes and E8M0 scales as `uint8`.

Dequantize mirrors each form; for int4 the group scales are repeated along the column axis, the one the groups run along:

```python
if isinstance(q, Q4Tensor):
    vals = unpack_int4(q.packed).astype(np.float32)
    s = np.repeat(np.asarray(q.scales).astype(np.float32), q.group, axis=1)  # [rows, cols]
    return vals * s
```

`nbytes` counts what would be stored; for MX fp4 it counts two codes per byte (`data.size // 2`) plus the scales, so a $32$-value block is $16 + 1 = 17$ bytes, the 4.25 bits per weight of section 2.

The layer and the model swap:

```python
class QuantLinear(Module):
    def __init__(self, q, bias=None) -> None:
        super().__init__()
        w = dequantize(q)
        self.q = q                                  # kept for export_q4
        self.out_f, self.in_f = int(w.shape[0]), int(w.shape[1])
        self._w_t = np.ascontiguousarray(w.T)       # [in, out], dequantized once
        # Plain arrays, not Parameters: the layer is for inference only.
        self.bias = None if bias is None else np.asarray(bias, dtype=np.float32).reshape(self.out_f)

    def forward(self, x: Tensor) -> Tensor:
        y = x @ self._w_t                           # x W^T
        return y + self.bias if self.bias is not None else y

def quantize_model(model, scheme, skip=("lm_head",)) -> Module:
    # ... scheme check ...
    found = dict(model.named_modules())
    for name, mod in list(found.items()):
        if not isinstance(mod, Linear) or not name:
            continue
        # lm_head is often tied to the embeddings; quantizing it changes both.
        if name in skip or any(part in skip for part in name.split(".")):
            continue
        parent_name, _, attr = name.rpartition(".")
        bias = None if mod.bias is None else mod.bias.data   # keep the bias
        setattr(found[parent_name], attr,
                QuantLinear(_quantize_weight(mod.weight.data, scheme), bias))
    return model
```

`export_q4` walks the swapped model, writes `<linear>.weight.qweight` (the packed bytes, `uint8`) and `<linear>.weight.scales` (`float16`) for every int4 layer plus its bias, refuses mixed group sizes, then copies every remaining parameter (the embeddings and `lm_head` among them) and returns the metadata `{"format": "tinyllm", "quant": "int4-g32-sym"}`.

The error budget and the perplexity cost, step 8:

```python
def output_error_bound(w, q, x, dtype="f32") -> NDArray:
    a = _matrix(w, "output_error_bound").astype(np.float64)
    xs = np.asarray(x, dtype=np.float64)
    # ... shape checks ...
    d = dequantize(q).astype(np.float64)
    # The quantization moved W by |W - W'| elementwise; M09.3 adds that
    # perturbation to the rounding of the k-term dot products in dtype.
    return matmul_error_bound(a, xs.T, dtype, np.abs(d - a)).T

def quant_ppl(model, scheme, ids, ctx_len, stride, skip=("lm_head",)) -> dict[str, float]:
    base = eval_ppl(model, ids, ctx_len, stride)
    # quantize_model works in place: quantize a copy, so the caller keeps
    # the float model it passed in.
    quant = quantize_model(copy.deepcopy(model), scheme, skip)
    got = eval_ppl(quant, ids, ctx_len, stride)
    return {"ppl": float(base["ppl"]), "ppl_quant": float(got["ppl"]),
            "delta": float(got["ppl"] - base["ppl"]), "ratio": float(got["ppl"] / base["ppl"])}
```

Full solution: [`quant.py`](../../../../course/solutions/python/tinyllm/infer/quant.py).

### Why this solution

- **Cost.** Every quantizer is one pass over the weight, $O(\text{out} \cdot \text{in})$, vectorised except the per-row or per-head loops into `M09.4`'s encoders. Storage per weight: int8 8 bits plus 32 per row; int4 4 bits plus 16 per 32 (4.5 bits); fp8 8 bits; MXFP4 4.25 bits.
- **Round to nearest, not GPTQ or AWQ.** Round to nearest with a well-chosen scale is the baseline every smarter method is measured against, needs no calibration data, and has a provable per-element bound. GPTQ and AWQ reduce output error at the same bit width but need activations and a search; they are in Going further.
- **Asymmetric (zero-point) quantization** would use the code range better for skewed groups, but costs a stored offset per group and a subtraction in the kernel's inner loop; weights are close enough to symmetric that the gain is small.
- **`QuantLinear` dequantizes once.** For a Python reference the point is a layer that computes exactly `x @ dequantize(q)^T + b`, the value the C and Rust kernels are compared against. It does not save memory at run time; the C kernel (`L9.5`) is what multiplies packed bytes directly.
- **Tradeoff accepted.** Python loops over rows for fp8 and KV keep the encoding identical to `M09.4` (and so to the C conversions) at the cost of speed in an offline step.

### Explain it back

1. Why is the int4 scale rounded to float16 before the codes are computed?
   <details><summary>Answer</summary>The kernel dequantizes with the float16 scale in the file. Codes chosen against the float32 scale can round the other way at a tie (3.5 stored scales rounds to 3 instead of 4), breaking the half-step bound and making the kernel's products disagree with the reference.</details>
2. Why can `amax / s16` exceed 7, and why does that not clip?
   <details><summary>Answer</summary>Rounding $\text{amax}/7$ to float16 can make $s_{16}$ slightly smaller than $\text{amax}/7$, by a relative $2^{-11}$ at most, so the quotient is at most $7(1 + 2^{-11}) < 7.5$. That still rounds to 7, so no value leaves $[-8, 7]$ and the bound holds.</details>
3. Why must the group size be even?
   <details><summary>Answer</summary>Two columns share a byte. An odd group would put the last column of one group and the first of the next in the same byte, so one byte would need two scales and the kernel's group-by-bytes loop would misread it.</details>
4. Why does a zero KV head get scale 1.0 while a zero int8 row gets scale 0?
   <details><summary>Answer</summary>`quantize_fp8` divides by the scale, so 0 would produce NaN codes; 1.0 encodes zeros exactly. The int8 path never divides by the stored scale (it uses a safe substitute) and the kernel multiplies by it, so 0 gives exactly 0 outputs.</details>
5. Why does `quant_ppl` deep-copy the model?
   <details><summary>Answer</summary>`quantize_model` replaces layers in place. Without the copy the caller's float model would be quantized and every later comparison against it would compare quantized to quantized.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| Even column in the high nibble | the C kernel multiplies every pair swapped | `pack_int4` puts `n[:, 0::2]` in the low nibble |
| Offset binary (code + 8) instead of two's complement | every code off by 8 for the kernel | masks with `& 0xF`, so $-1$ is `0xF` |
| Unpacking without sign extension | negative weights read as 9 to 15 | `unpack_int4` maps nibbles $\ge 8$ to $v - 16$ |
| Truncating instead of rounding | errors up to a whole step, biased toward zero | `np.rint` everywhere, ties to even |
| Casting int8 codes without rounding | the same truncation for int8 | `np.rint` before the `int8` cast |
| int8 with scale amax / 128 | the row maximum clips | divides by 127 |
| Quantizing against the float32 scale, storing float16 | a weight of exactly 3.5 stored scales rounds to 3 instead of 4; the kernel's products disagree | casts the scale to float16, then divides by that value (step 5) |
| Scale amax / 8 | the largest weight clips at 7 | divides by 7 |
| Writing an overflowing float16 scale | `inf` in the file, NaN in every product | raises when `s16` is not finite |
| Accepting an odd group | a group splits a byte between two scales | rejects `group % 2` and groups that do not divide the width |
| One int8 scale per tensor | small rows lose all precision | `max(axis=1)`: one scale per row |
| fp8 scale ignoring the format's maximum | wasted range or saturation | divides amax by `fp8_max(fmt)` |
| MX block size ignored | scales do not line up with the format's blocks | passes `block` through to `mx_quantize` |
| One KV scale for all heads | quiet heads lose their precision | amax over each head's flattened slab |
| A zero KV head with scale 0 | division by zero in `quantize_fp8` | scale 1.0 for a zero head |
| `QuantLinear` dropping the bias | every output shifted | adds `self.bias` when present |
| Multiplying by $W$ instead of $W^T$ | shapes fail, or square layers compute garbage | stores `w.T` once and computes `x @ _w_t` |
| Quantizing `lm_head` | tied embeddings change; logits lose the most precision | `skip=("lm_head",)` by default |
| `quantize_model` losing biases | the model's output drifts past the budget | passes `mod.bias.data` into `QuantLinear` |
| Wrong tensor names in the export | the Rust runner cannot find the weights | writes `<linear>.weight.qweight` and `<linear>.weight.scales` per the safetensors contract |
| Repeating group scales along the wrong axis | shapes fail on dequantize | `np.repeat(..., q.group, axis=1)` |
| Counting a byte per fp4 code | memory budgets double-count MX weights | `nbytes` counts `size // 2` for fp4 |
| A budget with rounding only ($\Delta = 0$) | every quantized output fails its tolerance by a factor of a thousand | passes $|W - \hat W|$ as `dA` to `matmul_error_bound` |
| Quantizing the caller's model in `quant_ppl` | the float model the caller compares against is gone | quantizes a `copy.deepcopy` |

## 7. Where it's used next

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

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| int4 group, symmetric | llama.cpp `Q4_0` and K-quants | 32-weight blocks with float16 scales, mixed precision per tensor (`Q4_K_M`) | [`ggml/src/ggml-quants.c`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c) |
| round to nearest | GPTQ, AWQ | second-order error compensation (GPTQ) or activation-aware scaling of salient channels (AWQ) at the same bit width | [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ), [llm-awq](https://github.com/mit-han-lab/llm-awq) |
| fp8 per channel | vLLM FP8 W8A8 | fp8 activations too, with per-token dynamic scales and fused kernels | [`vllm/model_executor/layers/quantization/fp8.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/fp8.py) |
| KV fp8 | vLLM `kv_cache_dtype="fp8"` | per-tensor or per-head KV scales calibrated offline | vLLM docs, "Quantized KV Cache" |
