# Fused int4 and int8 dequantize-matmul in C

## Overview

| | |
|---|---|
| **Module** | `L9.5` · side · C · Pass 6 · 4 to 5 h |
| **Solution** | [`c/src/kernels/qmatmul.c`](../../../../course/solutions/c/src/kernels/qmatmul.c): `tl_matmul_q4_f32` (W4A32: signed 4-bit weights, one f16 scale per group) and `tl_matmul_q8_f32` (W8A32: int8 weights, one f32 scale per output row), each computing $y = x W^\top$ without ever writing $W$ out in float32; helpers `unpack_q4`, `lanes_dot`, `lanes_sum`, `q4_rows`, `q8_rows` |
| **Contract** | [`course/contracts/c/include/tinyllm/qmatmul.h`](../../../../course/contracts/c/include/tinyllm/qmatmul.h) · byte layout: [`formats/safetensors.md`](../../../../course/contracts/formats/safetensors.md) (Int4 weights) · rules: [`c/ABI.md`](../../../../course/contracts/c/ABI.md) |
| **Needs** | `rt.02` the loader · `rt.03` the pool · `M09.7` `tl_f16_to_f32` · `L9.1` the float32 matmul, the baseline · `L8.5` the Python quantizer. Reading: `M09.3` (error bounds) |
| **Used by** | None: this optional C kernel stands alone; the Rust engine implements quantized inference independently. With `L9.3`, `L9.4`, and `L9.6` it completes the C kernel set of Pass 6 |
| **Optional depth** | Frantar et al., "GPTQ" (2022); Lin et al., "AWQ" (2023); Dettmers et al., "LLM.int8()" (2022); the llama.cpp `Q4_0` block format (`ggml-quants.c`) |

## Key Takeaways

- **Decode is memory-bound,** so reading 4.5 bits per weight instead of 32 makes it faster even though every weight must be unpacked first: the reference int4 GEMV is about 3.9 times faster than the float32 kernel at $N = K = 2048$.
- **The layout is a contract:** byte $b$ of row $n$ holds column $2b$ in the low nibble and $2b + 1$ in the high nibble, each a signed 4-bit two's complement value in $[-8, 7]$.
- **One scale per group of columns**, an IEEE half-precision bit pattern, multiplies the group's partial sum once: $y = \sum_g s_g \sum_{k \in g} x_k q_k$.
- **Fused means the float32 weight never exists:** each chunk of 64 weights is unpacked into a small buffer and consumed immediately.
- **Speed comes from independent accumulators the compiler can keep in registers.** Eight interleaved lanes per group, combined in a fixed tree, keep the kernel fast and every row's bits independent of the batch.

## How to read this chapter

Sections 1 to 3 explain why quantized decode is faster, fix the int4 byte layout, and work a small product by hand. Section 4 fixes the C interface and lists what a correct kernel must handle. Section 5 walks through the solution, [`qmatmul.c`](../../../../course/solutions/c/src/kernels/qmatmul.c), and explains why each choice (fused unpacking, eight lanes, a fixed combining tree) is the right one. Try to predict each step of the solution before reading it.

---

## 1. Why now

`L8.5` taught the Python stack to quantize a Llama: `quantize_int4_group` packs every projection into 4-bit codes with float16 scales, and `export_q4` writes them as `*.qweight` and `*.scales` tensors. Python's `QuantLinear` dequantizes to float32 and calls numpy, so it saves disk, not time. This optional standalone C kernel reads packed bytes directly and must agree with dequantize-then-matmul. The Rust serving path is a separate implementation that consumes the same safetensors format without calling this C kernel.

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
- `scales`: `f16 [N, K/G]`, the raw bit patterns of IEEE half precision (1 sign, 5 exponent, 10 mantissa bits). Decoded by `tl_f16_to_f32` from `M09.7`, not by shifting into the top of a float32, which is what bfloat16 would be.

The kernel validates the shape before touching anything: $K$ even (two codes per byte), $G$ even and positive (a group is whole bytes), $K$ a multiple of $G$ (whole groups); otherwise `TL_ESHAPE`.

### 2.3 Fusing the dequantization

Dequantize-then-multiply would write $W$ in float32 (as many bytes as the float32 model the kernel is trying not to read) and read it back. The fused kernel instead factors each group's scale out of its sum:

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

Every value is exact in float32. With the same codes as int8 and one scale per row ($0.5$ and $0.25$): $y = [0.5 \cdot (1 - 4 + 9 - 32), 0.25 \cdot (7 + 0 - 3 + 16)] = [-13, 5]$. A correct kernel returns these values exactly.

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

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the hand example | unit | section 3 exactly | the nibble order, the sign, the scale per group |
| the int8 hand example | unit | the int8 version of section 3 | one scale per row |
| every nibble code | boundary | all 16 codes in both nibbles decode to $0..7, -8..-1$ | an unsigned or swapped decode is caught exactly |
| f16 scales decode exactly | boundary | the f16 bits `0x0001` ($2^{-24}$) and `0x3555` | half precision, not bfloat16 |
| int4 against dequantize-then-matmul | differential | groups 2, 32, 64, and all of $K = 384$ against dequantize + `tl_matmul_f32` | the fused kernel equals the obvious recipe |
| int8 against dequantize-then-matmul | differential | $K = 200$ (not a multiple of the chunk) against the same recipe | int8 per channel |
| batch-invariant rows | property | 12 rows alone vs in batches, bitwise | decode and batched decode agree |
| shape and argument errors | boundary | `TL_ESHAPE` and `TL_EINVAL` cases leave $y$ untouched; empty dims; $K = 0$ | errors before any read |
| pooled equals serial | property | 4 threads vs serial, $N = 37$ | threads never change bits |
| bytes packed by numpy | unit | numpy packs section 3's bytes; the kernel returns $[-47.5, 10.25]$ | the layout from Python |

On one decode step ($M = 1$, $N = K = 2048$, group 32) the reference int4 kernel runs about 3.9 times faster than the float32 matmul of `L9.1`; a slower kernel can still be correct, but about 2 times is the least speedup that makes it worth having.

## 5. The solution, step by step

### 5.1 From first principles

1. **The step is bound by bytes read, not FLOPs** (section 2.1). So the design target is: read each packed byte once, and spend few enough instructions per weight that unpacking keeps up with memory.
2. **Scales factor out of each group's sum** (section 2.3). So the inner loop works on integer codes turned into floats, and the scale is applied once per group: in the worked example, $0.5 \cdot (-3)$ and $2 \cdot (-23)$.
3. **The float32 weight must never exist as a whole.** Unpacking into a fixed buffer of 64 floats (256 bytes) keeps the unpacked codes in L1 until the next loop consumes them.
4. **One accumulator is a dependency chain; eight are not.** Lanes $\ell$ over positions $\ell, \ell + 8, \dots$ let the compiler use vector registers. Because a chunk of 64 is a multiple of 8, a lane's positions are the same whether counted from the chunk or the group.
5. **Batch invariance requires the order of operations to depend only on $(m, n)$.** So: lanes reset per group, a fixed combining tree, groups added in increasing order, and no special path for $M = 1$.
6. **Threads split output rows $n$,** each computed wholly by one worker. No reduction across threads, so no change in bits.

### 5.2 The code, annotated

Unpacking a byte follows section 2.2 literally. `(u & 8) << 1` is 16 exactly when the sign bit of the nibble is set, so `u - 16` maps 8..15 to -8..-1.

```c
/* Signed 4-bit two's complement: 0..7 stay, 8..15 become -8..-1. */
static void unpack_q4(const uint8_t *bytes, int64_t n_bytes, float *q) {
    for (int64_t b = 0; b < n_bytes; b++) {
        int lo = bytes[b] & 0x0F, hi = bytes[b] >> 4;
        q[2 * b] = (float)(lo - ((lo & 8) << 1));     /* low nibble: column 2b (even) */
        q[2 * b + 1] = (float)(hi - ((hi & 8) << 1)); /* high nibble: column 2b + 1 (odd) */
    }
}
```

On `0xE1` from section 3: `lo = 1` stays 1, `hi = 14` becomes $14 - 16 = -2$.

The lane dot product is the hot loop. The lanes are copied into a local array so the compiler can prove they do not alias `x` or `q` and keep them in registers; `restrict` says the same about the pointers.

```c
static void lanes_dot(const float *restrict x, const float *restrict q, int64_t n, float *restrict lane) {
    float acc[LANES];                        /* local copy: registers, not memory */
    for (int l = 0; l < LANES; l++) acc[l] = lane[l];
    int64_t k = 0;
    for (; k + LANES <= n; k += LANES)       /* 8 independent chains */
        for (int l = 0; l < LANES; l++) acc[l] += x[k + l] * q[k + l];
    for (int l = 0; k + l < n; l++) acc[l] += x[k + l] * q[k + l]; /* tail: still lane = k mod 8 */
    for (int l = 0; l < LANES; l++) lane[l] = acc[l];
}

/* The fixed combination of the lanes: ((0+1)+(2+3)) + ((4+5)+(6+7)). */
static float lanes_sum(const float lane[LANES]) {
    return ((lane[0] + lane[1]) + (lane[2] + lane[3])) + ((lane[4] + lane[5]) + (lane[6] + lane[7]));
}
```

`q4_rows` puts it together for a range of output rows. Note where each thing is reset and indexed: lanes per group, the scale by row *and* group, the codes at the group's offset `k0`.

```c
    for (int64_t n = lo; n < hi; n++) {
        const uint8_t *wrow = wq + n * (a->K / 2);          /* K/2 bytes per row */
        for (int64_t m = 0; m < a->M; m++) {
            const float *xrow = a->x + m * a->K;
            float acc = 0.0f;
            for (int64_t g = 0; g < ngroups; g++) {
                float lane[LANES] = {0};                    /* partial sum resets per group */
                for (int64_t c = 0; c < a->group; c += CHUNK) {
                    int64_t len = a->group - c < CHUNK ? a->group - c : CHUNK;
                    int64_t k0 = g * a->group + c;          /* absolute column of the chunk */
                    unpack_q4(wrow + k0 / 2, len / 2, q);   /* 64 codes at most, in L1 */
                    lanes_dot(xrow + k0, q, len, lane);
                }
                acc += tl_f16_to_f32(sc[n * ngroups + g]) * lanes_sum(lane); /* scale once per group */
            }
            a->y[m * a->N + n] = acc;
        }
    }
```

`q8_rows` is the same with one group spanning the row: the codes are already signed bytes, and the single scale `sc[n]` (indexed by the output row, never by `m`) multiplies the combined lanes once.

```c
            float lane[LANES] = {0};
            for (int64_t c = 0; c < a->K; c += CHUNK) {
                int64_t len = a->K - c < CHUNK ? a->K - c : CHUNK;
                for (int64_t i = 0; i < len; i++) q[i] = (float)wrow[c + i];
                lanes_dot(xrow + c, q, len, lane);
            }
            a->y[m * a->N + n] = sc[n] * lanes_sum(lane);    /* per-channel scale */
```

The entry point validates the shape before any read (`K` even, `group` even and positive and dividing `K`), handles $K = 0$ as the empty sum without touching `x` or `wq`, and hands rows to the pool in ranges of 8:

```c
    if (group <= 0 || group % 2 != 0 || K % 2 != 0 || K % group != 0) {
        tl_set_last_error("tl_matmul_q4_f32: need K even, group even and > 0, K % group == 0");
        return TL_ESHAPE;
    }
    // ... M or N == 0: no-op; NULL checks; K == 0: y = 0 ...
    return tl_parallel_for(tp, a->N, 8, fn, a);   /* via run(); serial when tp == NULL */
```

Full solution: [`qmatmul.c`](../../../../course/solutions/c/src/kernels/qmatmul.c).

### 5.3 Why this solution

- **Cost.** $O(MNK)$ multiply-adds, like any matmul, but $NK/2 + 2NK/G$ bytes of weights instead of $4NK$. For $M = 1$, $G = 32$ that is 7.1 times fewer bytes, which is where the measured 3.9 times speedup comes from (unpacking eats the rest).
- **Why not dequantize the whole matrix and call `tl_matmul_f32`?** It writes and re-reads $4NK$ bytes, more than reading the float32 model directly: slower than not quantizing at all.
- **Why not integer dot products (quantize $x$ to int8 too)?** That is faster still (W4A8), but it changes the numerics: the contract is W4A32, matching `L8.5`'s `QuantLinear` up to float rounding.
- **Why 8 lanes and a fixed tree rather than a single sum, or whatever order the vectorizer picks?** A single sum is a 4-cycle chain per addition; letting the compiler reassociate freely (`-ffast-math`) would make the bits depend on the build and on $M$. Explicit lanes give the speed with a defined order.
- **Tradeoff accepted:** the order is not a left-to-right sum, so the result differs from a naive float32 loop in the last bits; it is compared to dequantize-then-matmul with a tolerance, and to itself bitwise.

### 5.4 Explain it back

1. Why is the int4 kernel faster than the float32 one even though it does strictly more work per weight?
   <details><summary>Answer</summary>Decode reads each weight once and does one multiply-add with it, about 0.5 FLOP per byte in float32. The core is waiting on memory, so cutting the bytes by about 7 times matters far more than the extra unpack instructions.</details>
2. Why is `lane[]` copied into a local `acc[]` inside `lanes_dot`?
   <details><summary>Answer</summary>Through a pointer parameter the compiler must assume `lane` might alias `x` or `q`, so it reloads and stores the lanes on every step. A local array cannot alias, so it can live in vector registers. The reference measured 1.75 ms versus 0.99 ms per step.</details>
3. What goes wrong if the lanes are not reset at the start of each group?
   <details><summary>Answer</summary>Each group's scale would multiply the running sum of all earlier groups as well, so earlier groups are counted many times with the wrong scales.</details>
4. Why would a special fast path for $M = 1$ be a bug even if it is accurate?
   <details><summary>Answer</summary>Any different summation order changes the last bits, so a sequence decoded alone would produce different logits from the same sequence in a batch: the batch invariance of `c/ABI.md` rule 10 would be broken.</details>
5. How does `u - ((u & 8) << 1)` implement two's complement?
   <details><summary>Answer</summary>For $u < 8$, bit 3 is clear and the expression is $u$. For $u \ge 8$, `(u & 8) << 1` is 16, giving $u - 16$, which is the signed value of a 4-bit two's complement pattern.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| reading nibbles as unsigned $0..15$ | negative weights become large positive ones | `u - ((u & 8) << 1)` |
| the high nibble as the even column | columns swapped in pairs: plausible, wrong | low nibble to `q[2b]`, high to `q[2b + 1]` |
| indexing scales by group only, not by row | every row uses row 0's scales | `sc[n * ngroups + g]` |
| not resetting the partial sum per group | each scale multiplies the sum of all earlier groups too | `float lane[LANES] = {0}` inside the group loop |
| decoding the f16 scale as bfloat16 | scales off by orders of magnitude | `tl_f16_to_f32` |
| unpacking at the chunk offset instead of the group's | every group reads the first group's codes | `k0 = g * group + c` |
| the int8 scale indexed by the activation row | wrong for $M > 1$ only | `sc[n]`, never `sc[m]` |
| forgetting the int8 scale | outputs 50 to 1000 times too large | `y = sc[n] * lanes_sum(lane)` |
| a special summation order for $M = 1$ | decode and batched decode differ in the last bit | one code path for every $M$ |
| accepting a group that does not divide $K$ | reads past `x` and the weights | `TL_ESHAPE` before any read |
| skipping the NULL check on the weights | a crash instead of `TL_EINVAL` | NULL checks before the $K = 0$ shortcut |
| a pooled partition that drops a row | wrong only with threads | `tl_parallel_for(tp, N, 8, ...)` covers every row |
| accumulating through a pointer parameter | correct but about 1.8 times slower: the lanes are reloaded every step | local `acc[LANES]` with `restrict` pointers |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `rt.02` | the error slot and allocator support |
| Back | `rt.03` | `tl_parallel_for` over output rows |
| Back | `M09.7` | `tl_f16_to_f32` decodes every scale (and `tl_f32_to_f16` encodes them when building test inputs) |
| Back | `L9.1` | the float32 baseline: dequantize, then `tl_matmul_f32`; the speed comparison |
| Back | `L8.5` | the quantizer and the byte layout this kernel decodes |

## Going further

| This solution | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| int4 with f16 group scales | llama.cpp `Q4_0`, `Q4_K` | blocks of 32 with the scale stored inline next to the codes (one cache line), super-blocks with 6-bit sub-scales | `ggml/src/ggml-quants.c` |
| W4A32 | W4A8 / W4A16 kernels | activations quantized too, integer dot products (`sdot`, VNNI) | llama.cpp `ggml_vec_dot_q4_0_q8_0` |
| absmax quantization | GPTQ, AWQ | choose codes to minimize the layer's output error, not each weight's | the GPTQ and AWQ papers |
| CPU GEMV | Marlin, ExLlamaV2 | GPU kernels that reach memory bandwidth at batch sizes up to 16 | IST-DASLab `marlin` |
