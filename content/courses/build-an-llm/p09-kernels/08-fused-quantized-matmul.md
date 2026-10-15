<!-- ol:module L9.5 -->
# Fused int4 and int8 dequantize-matmul in C

## Overview

| | |
|---|---|
| **Module** | `L9.5` · side · C · Pass 6 · 4 to 5 h |
| **You build** | `c/src/kernels/qmatmul.c`: `tl_matmul_q4_f32` (W4A32: signed 4-bit weights, one f16 scale per group) and `tl_matmul_q8_f32` (W8A32: int8 weights, one f32 scale per output row), each computing $y = x W^\top$ without ever writing $W$ out in float32 |
| **Contract** | [`course/contracts/c/include/tinyllm/qmatmul.h`](../../../../course/contracts/c/include/tinyllm/qmatmul.h) · byte layout: [`formats/safetensors.md`](../../../../course/contracts/formats/safetensors.md) (Int4 weights) · rules: [`c/ABI.md`](../../../../course/contracts/c/ABI.md) |
| **Tests** | `course/tests/L9.5/test_qmatmul.c` (standalone C tests under ASan and UBSan, against dequantize-then-`tl_matmul_f32`) and `course/tests/L9.5/bench/q4_gemv_2048.c` |
| **Needs** | `rt.02` the loader · `rt.03` the pool · `M09.7` `tl_f16_to_f32` · `L9.1` your float32 matmul, the baseline · `L8.5` your Python quantizer (or `--ref-deps`). Reading: `M09.3` (error bounds) |
| **Used by** | None: this optional C exercise is tested as a standalone binary; the Rust engine implements quantized inference independently |
| **Milestone** | `MS-L9` |
| **Optional depth** | Frantar et al., "GPTQ" (2022); Lin et al., "AWQ" (2023); Dettmers et al., "LLM.int8()" (2022); the llama.cpp `Q4_0` block format (`ggml-quants.c`) |

## Key Takeaways

- **Decode is memory-bound,** so reading 4.5 bits per weight instead of 32 makes it faster even though every weight must be unpacked first: the reference int4 GEMV is about 3.9 times faster than your float32 kernel at $N = K = 2048$ (`ol bench L9.5`).
- **The layout is a contract:** byte $b$ of row $n$ holds column $2b$ in the low nibble and $2b + 1$ in the high nibble, each a signed 4-bit two's complement value in $[-8, 7]$ (`every_nibble_code`, `hand_example`).
- **One scale per group of columns**, an IEEE half-precision bit pattern, multiplies the group's partial sum once: $y = \sum_g s_g \sum_{k \in g} x_k q_k$ (`matches_dequantize_then_matmul`, `f16_scales_decode_exactly`).
- **Fused means the float32 weight never exists:** each chunk of 64 weights is unpacked into a small buffer and consumed immediately.
- **Speed comes from independent accumulators the compiler can keep in registers.** Eight interleaved lanes per group, combined in a fixed tree, keep the kernel fast and every row's bits independent of the batch (`batch_invariant_rows`).

## How to work this chapter

```bash
ol start L9.5              # stubs c/src/kernels/qmatmul.c into your repo
ol tests L9.5              # read the test catalog first
ol check L9.5              # exit code is the verdict
ol check L9.5 --ref-deps   # only if rt.02, rt.03, M09.7, L9.1, or L8.5 is not passing yet
ol bench L9.5 --assert     # at least 2x your float32 kernel on one decode step (local only)
ol parity quant.int4       # your L8.5 encoder and this decoder against one golden
ol diff  L9.5              # after passing: your code against the reference
```

---

## 1. Why now

`L8.5` taught your Python stack to quantize a Llama: `quantize_int4_group` packs every projection into 4-bit codes with float16 scales, and `export_q4` writes them as `*.qweight` and `*.scales` tensors. Python's `QuantLinear` dequantizes to float32 and calls numpy, so it saves disk, not time. This optional standalone C exercise reads packed bytes directly and compares its output with dequantize-then-matmul. The Rust serving path is a separate implementation that consumes the same safetensors format without calling this C kernel.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $x \in \mathbb{R}^{M \times K}$ | activations, $M$ rows (1 for decode) | `float[M][K]` |
| $W \in \mathbb{R}^{N \times K}$ | the Linear weight, `[out, in]` as stored | never materialized |
| $y = x W^\top$ | the output | `float[M][N]` |
| $G$ | `group`: columns that share one scale | `int64_t` |
| $q_{nk} \in \{-8, \dots, 7\}$ | the 4-bit code of $W_{nk}$ | 4 bits |
| $s_{n,g}$ | the scale of row $n$, group $g = \lfloor k / G \rfloor$ | IEEE f16 |
| $W_{nk} = q_{nk}\, s_{n, \lfloor k/G \rfloor}$ | the value the code stands for | |
| $\ell$ | a lane index, $0 \le \ell < 8$ | |

### 2.1 Why quantized decode is faster: bytes per weight

A decode step multiplies one activation row by every weight matrix: $2NK$ FLOPs for $NK$ weights, one multiply-add per weight loaded. The arithmetic intensity is about 0.5 FLOP per byte in float32, far below what a core can compute per byte of bandwidth (`L9.1` 2.1), so the step takes as long as it takes to *read* the weights. int4 with groups of 32 stores $4 + 16/32 = 4.5$ bits per weight, a factor of 7.1 fewer bytes; int8 per channel stores about 8. The kernel can spend several instructions per weight on unpacking and still win, as long as it keeps up with memory.

### 2.2 The int4 layout

`formats/safetensors.md` (shared with `L8.5`, which writes it, and the Rust engine, which loads the same file format independently) fixes:

- `qweight`: `uint8 [N, K/2]`. Byte $b$ of row $n$ holds column $2b$ in its **low** nibble (bits 0 to 3) and column $2b + 1$ in its **high** nibble (bits 4 to 7).
- Each nibble is **signed 4-bit two's complement**: the unsigned value $u \in [0, 15]$ means $u$ when $u < 8$ and $u - 16$ otherwise, so `0x8` is $-8$ and `0xF` is $-1$. In C: `u - ((u & 8) << 1)`.
- `scales`: `f16 [N, K/G]`, the raw bit patterns of IEEE half precision (1 sign, 5 exponent, 10 mantissa bits). Decoded by your `tl_f16_to_f32` from `M09.7`, not by shifting into the top of a float32, which is what bfloat16 would be.

The kernel validates the shape before touching anything: $K$ even (two codes per byte), $G$ even and positive (a group is whole bytes), $K$ a multiple of $G$ (whole groups); otherwise `TL_ESHAPE`.

### 2.3 Fusing the dequantization

Dequantize-then-multiply would write $W$ in float32 (as many bytes as the model you were trying not to read) and read it back. The fused kernel instead factors each group's scale out of its sum:

$$y_{mn} = \sum_{k} x_{mk}\, q_{nk}\, s_{n, \lfloor k/G \rfloor} = \sum_{g} s_{n,g} \underbrace{\sum_{k \in g} x_{mk}\, q_{nk}}_{\text{partial}_g} .$$

One multiply by the scale per group instead of one per weight, and the codes are turned into floats only in a small buffer: 64 at a time (32 bytes), consumed by the next loop while they are still in L1. int8 is the same with one group per row: $y_{mn} = s_n \sum_k x_{mk} q_{nk}$.

### 2.4 Lanes: fast and still batch-invariant

A single running sum is a chain: each addition waits for the previous one, about 4 cycles on a modern core, so one chain caps the kernel at a quarter of an addition per cycle whatever the vector width. The kernel therefore keeps **8 independent lanes** per group: lane $\ell$ accumulates the products of positions $\ell, \ell + 8, \ell + 16, \dots$ of the group, in increasing order, and the compiler maps the 8 lanes onto vector registers (two 4-wide NEON or one 8-wide AVX register). At the end of the group the lanes combine in a fixed tree,

$$\text{partial}_g = \big((\ell_0 + \ell_1) + (\ell_2 + \ell_3)\big) + \big((\ell_4 + \ell_5) + (\ell_6 + \ell_7)\big),$$

and the output accumulates $s_{n,g}\,\text{partial}_g$ in increasing $g$. Every one of those steps depends only on $(m, n)$: not on $M$, not on the thread, not on the row's position. That keeps the batch invariance of `c/ABI.md` rule 10 (the order is fixed, though it is not a single left-to-right sum).

Two C details decide whether the lanes stay in registers. If the lane array is a parameter, the compiler must assume it could alias `x` or the code buffer and reloads it on every step; copying it into a local array and marking the pointers `restrict` lets the lanes live in registers. In the reference that one change took the kernel from 1.75 ms to 0.99 ms per step. And `#pragma STDC FP_CONTRACT OFF` keeps the compiler from fusing a multiply and an add on some paths but not others (`L9.1` 2.4).

### 2.5 Threads

Output rows $n$ split across `rt.03` workers in ranges of 8; each $y_{mn}$ is computed entirely by one worker with the arithmetic above, so threads never change the bits.

## 3. Worked example by hand

$W$ is $2 \times 4$, $G = 2$, $x = [1, 2, 3, 4]$, $M = 1$:

| Row | codes $q$ | scales $s$ (f16 bits) | bytes |
|---|---|---|---|
| 0 | $[1, -2 \mid 3, -8]$ | $[0.5 \mid 2]$ = `0x3800`, `0x4000` | `0xE1`, `0x83` |
| 1 | $[7, 0 \mid -1, 4]$ | $[1 \mid 0.25]$ = `0x3C00`, `0x3400` | `0x07`, `0x4F` |

Packing row 0: $1 = $ `0x1` (low), $-2 = 16 - 2 = 14 = $ `0xE` (high), so byte 0 is `0xE1`; $3 = $ `0x3`, $-8 = $ `0x8`, byte 1 is `0x83`. Row 1: `0x07` and ($-1 = $ `0xF` low, $4$ high) `0x4F`.

Decoding byte `0xE1`: low nibble $1 < 8$ gives $1$; high nibble $14 \ge 8$ gives $14 - 16 = -2$.

$$y_0 = 0.5\,(1 \cdot 1 + 2 \cdot (-2)) + 2\,(3 \cdot 3 + 4 \cdot (-8)) = 0.5 \cdot (-3) + 2 \cdot (-23) = -47.5$$

$$y_1 = 1\,(1 \cdot 7 + 2 \cdot 0) + 0.25\,(3 \cdot (-1) + 4 \cdot 4) = 7 + 0.25 \cdot 13 = 10.25$$

Every value is exact in float32. With the same codes as int8 and one scale per row ($0.5$ and $0.25$): $y = [0.5 \cdot (1 - 4 + 9 - 32), 0.25 \cdot (7 + 0 - 3 + 16)] = [-13, 5]$. These are the values checked by `hand_example` and `q8_hand_example`.

## 4. The interface

```c
/* tinyllm/qmatmul.h: y [M, N] = x [M, K] @ W^T, W [N, K] quantized */
tl_status tl_matmul_q4_f32(const float *x, const uint8_t *wq, const uint16_t *scales_f16,
                           float *y, int64_t M, int64_t N, int64_t K, int64_t group, tl_pool *tp);
tl_status tl_matmul_q8_f32(const float *x, const int8_t *wq, const float *scales,
                           float *y, int64_t M, int64_t N, int64_t K, tl_pool *tp);
/* TL_ESHAPE: K odd, group odd or <= 0, K % group != 0. TL_EINVAL: negative dims,
   NULL pointers. M or N == 0: no-op. K == 0: y = 0. */
```

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `hand_example` | unit, smoke | section 3 exactly | the nibble order, the sign, the scale per group |
| `q8_hand_example` | unit | the int8 version of section 3 | one scale per row |
| `every_nibble_code` | boundary | all 16 codes in both nibbles decode to $0..7, -8..-1$ | an unsigned or swapped decode is caught exactly |
| `f16_scales_decode_exactly` | boundary | the f16 bits `0x0001` ($2^{-24}$) and `0x3555` | half precision, not bfloat16 |
| `matches_dequantize_then_matmul` | differential | groups 2, 32, 64, and all of $K = 384$ against dequantize + your `tl_matmul_f32` | the fused kernel equals the obvious recipe |
| `q8_matches_dequantize_then_matmul` | differential | $K = 200$ (not a multiple of the chunk) against the same recipe | int8 per channel |
| `batch_invariant_rows` | property | 12 rows alone vs in batches, bitwise | decode and batched decode agree |
| `shape_and_argument_errors` | boundary | `TL_ESHAPE` and `TL_EINVAL` cases leave $y$ untouched; empty dims; $K = 0$ | errors before any read |
| `pool_result_equals_serial_bitwise` | property | 4 threads vs serial, $N = 37$ | threads never change bits |
| `hand_example` | unit, smoke | numpy packs section 3's bytes; the kernel returns $[-47.5, 10.25]$ | the layout from Python |

The bench (`ol bench L9.5`, local only) runs one decode step, $M = 1$, $N = K = 2048$, group 32, and reports `q4_speedup_vs_f32` (budget $\ge 2$; the reference reaches about 3.9) and the int8 ratio.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| reading nibbles as unsigned $0..15$ | negative weights become large positive ones | `every_nibble_code` (mutant `s01`) |
| the high nibble as the even column | columns swapped in pairs: plausible, wrong | `every_nibble_code` (mutant `s02`) |
| indexing scales by group only, not by row | every row uses row 0's scales | `hand_example` (mutant `s03`) |
| not resetting the partial sum per group | each scale multiplies the sum of all earlier groups too | `hand_example` (mutant `s04`) |
| decoding the f16 scale as bfloat16 | scales off by orders of magnitude | `f16_scales_decode_exactly` (mutant `s05`) |
| unpacking at the chunk offset instead of the group's | every group reads the first group's codes | `matches_dequantize_then_matmul` (mutant `s06`) |
| the int8 scale indexed by the activation row | wrong for $M > 1$ only | `q8_matches_dequantize_then_matmul` (mutant `s07`) |
| forgetting the int8 scale | outputs 50 to 1000 times too large | `q8_hand_example` (mutant `s08`) |
| a special summation order for $M = 1$ | decode and batched decode differ in the last bit | `batch_invariant_rows` (mutant `s09`) |
| accepting a group that does not divide $K$ | reads past `x` and the weights | `shape_and_argument_errors` (mutant `s10`) |
| skipping the NULL check on the weights | a crash instead of `TL_EINVAL` | `shape_and_argument_errors` (mutant `s11`) |
| a pooled partition that drops a row | wrong only with threads | `pool_result_equals_serial_bitwise` (mutant `s12`) |
| accumulating through a pointer parameter | correct but about 1.8 times slower: the lanes are reloaded every step | the bench (ol bench L9.5, local only; no course test checks speed) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `rt.02` | the error slot and allocator support |
| Back | `rt.03` | `tl_parallel_for` over output rows |
| Back | `M09.7` | `tl_f16_to_f32` decodes every scale (and `tl_f32_to_f16` encodes them in the tests) |
| Back | `L9.1` | the float32 baseline: dequantize, then `tl_matmul_f32`; the bench's comparison |
| Back | `L8.5` | the quantizer and the byte layout this kernel decodes |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| int4 with f16 group scales | llama.cpp `Q4_0`, `Q4_K` | blocks of 32 with the scale stored inline next to the codes (one cache line), super-blocks with 6-bit sub-scales | `ggml/src/ggml-quants.c` |
| W4A32 | W4A8 / W4A16 kernels | activations quantized too, integer dot products (`sdot`, VNNI) | llama.cpp `ggml_vec_dot_q4_0_q8_0` |
| absmax quantization | GPTQ, AWQ | choose codes to minimize the layer's output error, not each weight's | the GPTQ and AWQ papers |
| CPU GEMV | Marlin, ExLlamaV2 | GPU kernels that reach memory bandwidth at batch sizes up to 16 | IST-DASLab `marlin` |
